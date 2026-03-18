"""Smoothing pass: redistribute activation ranges between LayerNorm and Linear."""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from loguru import logger

from .base import CompressionPass



@torch.no_grad()
def smooth_ln_fcs_rms(
    ln: nn.Module,
    fcs: List[nn.Linear],
    act_scales: torch.Tensor,
    alpha: float = 0.5,
) -> None:
    """Smooth a RMSNorm (no bias) followed by one or more Linear layers."""
    if not isinstance(fcs, list):
        fcs = [fcs]

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)

    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    ).max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device=device, dtype=dtype)
    )

    ln.weight.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_ln_fcs(
    ln: nn.LayerNorm,
    fcs: List[nn.Linear],
    act_scales: torch.Tensor,
    alpha: float = 0.5,
) -> None:
    """Smooth a standard LayerNorm (weight + bias) followed by Linear layers."""
    if not isinstance(fcs, list):
        fcs = [fcs]

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)

    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    ).max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device=device, dtype=dtype)
    )

    ln.weight.div_(scales)
    if ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


def _get_act_scale(feat: torch.Tensor) -> torch.Tensor:
    """Compute per-channel max absolute activation from a collected feature tensor."""
    if feat.dim() == 3:
        feat = feat.reshape(-1, feat.shape[-1])
    return feat.abs().max(dim=0)[0].float()



class SmoothingPass(CompressionPass):
    """Smooth activations to improve quantization-friendliness.

    Obtains (norm, [linears]) pairs from adapter.get_smooth_pairs() and
    applies smooth_ln_fcs / smooth_ln_fcs_rms accordingly.
    """

    def apply(self, block, input_feat: Dict[str, torch.Tensor], adapter, config) -> None:
        pairs: List[Tuple[nn.Module, List[nn.Linear]]] = adapter.get_smooth_pairs(block)

        for ln, fcs in pairs:
            if not fcs:
                continue

            # Use activation scale from the first linear's input features.
            # Derive the name of that linear relative to the block.
            first_fc = fcs[0]
            feat_key = None
            for name, m in block.named_modules():
                if m is first_fc:
                    feat_key = name
                    break

            if feat_key is None or feat_key not in input_feat:
                logger.warning(
                    f"SmoothingPass: cannot find input_feat for {first_fc}, skipping pair"
                )
                continue

            act_scales = _get_act_scale(input_feat[feat_key]).to(fcs[0].weight.device)

            # Detect norm type: RMSNorm has no bias attribute (or bias is None)
            has_bias = getattr(ln, "bias", None) is not None
            if has_bias:
                smooth_ln_fcs(ln, fcs, act_scales, alpha=config.smooth_alpha)
            else:
                smooth_ln_fcs_rms(ln, fcs, act_scales, alpha=config.smooth_alpha)
