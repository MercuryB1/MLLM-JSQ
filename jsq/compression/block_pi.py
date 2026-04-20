"""Block-wise adaptive pi_t estimation for JSQ v5.

This module estimates a per-block text mixture weight ``pi_t^(b)`` from
text-only vs vision-only v5 importance statistics.  The goal is to stay at
the block granularity already used by Innovation 2, while making the signal
explicitly multimodal instead of relying on prune-only damage or BI proxies.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .passes.prune import _compute_mask, _jsq_v5_metric


def _has_modal_split(vision_mask: Optional[torch.Tensor]) -> bool:
    if vision_mask is None:
        return False
    vm = vision_mask.reshape(-1).bool()
    return bool(vm.any() and (~vm).any())


def _prepare_feat(feat: torch.Tensor, input_feat: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Match PruningPass semantics: text is 3D, multimodal is flattened 2D."""
    if feat.dim() == 2:
        _ = int(input_feat.get("__nsamples__", 1))
        return feat.unsqueeze(0)
    return feat


@torch.no_grad()
def estimate_block_pi(
    block,
    input_feat: Dict[str, torch.Tensor],
    adapter,
    config,
    vision_mask: Optional[torch.Tensor] = None,
) -> Tuple[Optional[float], Dict[str, float]]:
    """Estimate a block-wise pi_t from v5 text-only / vision-only signals.

    Returns:
        (pi_t_block, stats). ``pi_t_block`` is None when the estimate is not
        meaningful (e.g. text-only mode or no mixed text/vision tokens).
    """
    method = getattr(config, "block_pi_method", "none")
    if config.pruning_method != "jsq_v5" or method == "none":
        return None, {}
    if not _has_modal_split(vision_mask):
        return None, {}

    named_linears = adapter.get_named_linears(block)

    total_params = 0.0
    mass_t_total = 0.0
    mass_v_total = 0.0
    conflict_total = 0.0
    iou_total = 0.0
    n_layers = 0

    for name, linear in named_linears.items():
        if name not in input_feat:
            continue

        feat = _prepare_feat(input_feat[name], input_feat).to(linear.weight.device)
        layer_weight = float(linear.weight.numel())

        metric_t = _jsq_v5_metric(
            linear.weight.data,
            feat,
            vision_mask=vision_mask,
            pi_t=1.0,
            lambda_floor=config.lambda_floor,
            max_tokens=getattr(config, "block_pi_max_tokens", 1024),
            w_bits_act=config.a_bits,
        )
        metric_v = _jsq_v5_metric(
            linear.weight.data,
            feat,
            vision_mask=vision_mask,
            pi_t=0.0,
            lambda_floor=config.lambda_floor,
            max_tokens=getattr(config, "block_pi_max_tokens", 1024),
            w_bits_act=config.a_bits,
        )

        mass_t_total += layer_weight * float(metric_t.mean().item())
        mass_v_total += layer_weight * float(metric_v.mean().item())
        total_params += layer_weight
        n_layers += 1

        if method == "conflict_weighted":
            mask_t = _compute_mask(metric_t, config.sparsity_ratio, config.prune_n, config.prune_m)
            mask_v = _compute_mask(metric_v, config.sparsity_ratio, config.prune_n, config.prune_m)
            inter = float((mask_t & mask_v).sum().item())
            union = float((mask_t | mask_v).sum().item())
            iou = inter / union if union > 0 else 1.0
            iou_total += layer_weight * iou
            conflict_total += layer_weight * (1.0 - iou)
            del mask_t, mask_v

        del metric_t, metric_v

    if total_params <= 0:
        return None, {}

    denom = mass_t_total + mass_v_total
    dominance = mass_t_total / denom if denom > 0 else float(config.pi_t)
    mean_iou = iou_total / total_params if total_params > 0 else 1.0
    conflict = conflict_total / total_params if total_params > 0 else 0.0

    if method == "dominance":
        pi_t_raw = dominance
    elif method == "conflict_weighted":
        # If text/vision masks nearly agree, stay close to neutral; let
        # dominance matter only when cross-modal pruning preferences diverge.
        pi_t_raw = 0.5 + conflict * (dominance - 0.5)
    else:
        return None, {}

    blend = float(getattr(config, "block_pi_blend", 1.0))
    pi_t_block = (1.0 - blend) * float(config.pi_t) + blend * float(pi_t_raw)
    pi_t_block = max(float(getattr(config, "block_pi_min", 0.1)),
                     min(float(getattr(config, "block_pi_max", 0.9)), pi_t_block))

    return pi_t_block, {
        "pi_t_raw": float(pi_t_raw),
        "pi_t_block": float(pi_t_block),
        "dominance": float(dominance),
        "conflict": float(conflict),
        "mask_iou": float(mean_iou),
        "mass_t": float(mass_t_total / total_params),
        "mass_v": float(mass_v_total / total_params),
        "n_layers": float(n_layers),
    }
