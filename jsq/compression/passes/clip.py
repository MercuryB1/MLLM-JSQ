"""Clipping pass: search for optimal weight clipping thresholds."""
import gc
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger

from .base import CompressionPass
from jsq.quant.ops import quantize_weight_per_tensor_absmax



@torch.no_grad()
def _clip_layer(
    w: torch.Tensor,
    feat: torch.Tensor,
    w_bits: int,
    n_grid: int = 20,
    max_shrink: float = 0.5,
    n_sample_token: int = 512,
    feat_hessian: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Search for the best per-group clipping threshold via grid search.

    When *feat_hessian* is provided (shape [n_tokens], computed as mean(feat²)
    over the hidden dim), token-level errors are weighted by it — layers whose
    inputs carry more variance are penalised more heavily.

    Returns best_max_val of shape [out_features, n_groups].
    """
    group_size = w.shape[1]
    feat = feat.view(-1, feat.shape[-1])

    # Sub-sample tokens to keep memory bounded
    step = max(1, feat.shape[0] // n_sample_token)
    feat = feat[::step]
    h_weights: Optional[torch.Tensor] = None
    if feat_hessian is not None:
        feat_hessian = feat_hessian.view(-1)
        feat_hessian = feat_hessian[::step]
        h_sum = feat_hessian.sum().clamp(min=1e-8)
        h_weights = (feat_hessian / h_sum).reshape(1, -1, 1)  # [1, n_tok, 1]

    feat = feat.reshape(1, feat.shape[0], -1, group_size)
    w_4d = w.reshape(w.shape[0], 1, -1, group_size)

    oc_batch = 256 if w_4d.shape[0] % 256 == 0 else 64
    if w_4d.shape[0] % oc_batch != 0:
        oc_batch = 1  # fallback

    best_max_val_all: List[torch.Tensor] = []

    for i_b in range(w_4d.shape[0] // oc_batch):
        w_b = w_4d[i_b * oc_batch:(i_b + 1) * oc_batch]
        org_max = w_b.abs().amax(dim=-1, keepdim=True)  # [oc, 1, n_g, 1]
        best_max = org_max.clone()
        min_errs = torch.full_like(org_max, 1e9)

        feat_dev = feat.to(w_b.device)
        org_out = (feat_dev * w_b).sum(dim=-1)  # [oc, n_tok, n_g]

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max * (1 - i_s / n_grid)
            cur_w = torch.clamp(w_b, -max_val, max_val)
            q_w = quantize_weight_per_tensor_absmax(cur_w.clone(), w_bits=w_bits)
            cur_out = (feat_dev * q_w).sum(dim=-1)
            diff_sq = (cur_out - org_out).pow(2)  # [oc, n_tok, n_g]
            if h_weights is not None:
                hw = h_weights.to(diff_sq.device)  # [1, n_tok, 1]
                err = (diff_sq * hw).sum(dim=1).view(min_errs.shape)
            else:
                err = diff_sq.mean(dim=1).view(min_errs.shape)
            better = err < min_errs
            min_errs[better] = err[better]
            best_max[better] = max_val[better]

        best_max_val_all.append(best_max)

    gc.collect()
    torch.cuda.empty_cache()
    return torch.cat(best_max_val_all, dim=0).squeeze(1)  # [out, n_groups, 1]


@torch.no_grad()
def _apply_clip(module: nn.Module, clip_list: List[Tuple[str, torch.Tensor]]) -> None:
    """Apply pre-computed clip thresholds to named layers inside a module."""
    for name, max_val in clip_list:
        layer = dict(module.named_modules()).get(name)
        if layer is None:
            logger.warning(f"ClippingPass: layer '{name}' not found in block")
            continue
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        w = layer.weight.data.reshape(*max_val.shape[:2], -1)
        w = torch.clamp(w, -max_val, max_val)
        layer.weight.data = w.reshape(org_shape)



class ClippingPass(CompressionPass):
    """Search for optimal weight clip thresholds and apply them.

    Uses Hessian-weighted error (second moment of input activations) to bias
    the grid search towards preserving outputs on high-variance tokens.
    """

    def apply(self, block, input_feat: Dict[str, torch.Tensor], adapter, config) -> None:
        named_linears = adapter.get_named_linears(block)
        clip_list: List[Tuple[str, torch.Tensor]] = []

        for name, linear in named_linears.items():
            # Skip Q/K projections — quantizing their output (BMM inputs) is handled
            # separately and precise clipping is difficult.
            if any(tok in name for tok in ["q_proj", "k_proj", "query", "key", "Wqkv"]):
                continue
            if name not in input_feat:
                continue

            feat = input_feat[name].float()

            # Hessian proxy: per-token importance = mean(feat²) over hidden dim
            feat_flat = feat.view(-1, feat.shape[-1])
            feat_hessian = feat_flat.pow(2).mean(dim=-1)  # [n_tokens]

            try:
                max_val = _clip_layer(
                    linear.weight.data.float(),
                    feat,
                    w_bits=config.w_bits,
                    n_sample_token=config.nsamples,
                    feat_hessian=feat_hessian,
                )
                clip_list.append((name, max_val))
            except Exception as e:
                logger.warning(f"ClippingPass: failed on '{name}': {e}")

        _apply_clip(block, clip_list)
