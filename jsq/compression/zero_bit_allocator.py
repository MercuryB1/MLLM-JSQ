"""Zero-bit rate-distortion allocation for JSQ v5.

This module keeps the existing JSQ v5 mixture-Hessian curvature, but derives
per-layer sparsity inside each block from a joint prune-vs-W8 objective:

    prune  -> 0-bit action
    keep   -> 8-bit quantized action

The final mask remains unstructured.  What changes is how the block budget is
distributed across layers.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .passes.prune import compute_jsq_v5_zero_bit_metric


def _prepare_feat(feat: torch.Tensor) -> torch.Tensor:
    """Match PruningPass semantics: multimodal feat is flattened 2D."""
    if feat.dim() == 2:
        return feat.unsqueeze(0)
    return feat


def _build_layer_rd_profile(
    utility: torch.Tensor,
    d0: torch.Tensor,
    d_out: int,
    d_in: int,
) -> Dict[str, object]:
    """Construct the per-layer keep curve from per-weight utility."""
    sorted_u, _ = torch.sort(utility.float(), dim=-1, descending=True)
    col_gain = sorted_u.sum(dim=0).double().cpu()  # marginal gain per extra keep

    prefix_gain = torch.empty(col_gain.numel() + 1, dtype=torch.float64)
    prefix_gain[0] = 0.0
    prefix_gain[1:] = torch.cumsum(col_gain, dim=0)

    base0 = float(d0.double().sum().item())
    return {
        "d_out": int(d_out),
        "d_in": int(d_in),
        "base0": base0,
        "col_gain": col_gain,
        "prefix_gain": prefix_gain,
    }


def _optimal_k_for_lambda(prefix_gain: torch.Tensor, d_out: int, lam: float) -> int:
    """Best keep-per-row count for a given Lagrange multiplier."""
    k_idx = torch.arange(prefix_gain.numel(), dtype=torch.float64)
    scores = prefix_gain - (lam * float(d_out)) * k_idx
    return int(torch.argmax(scores).item())


def _solve_block_keep_allocation(
    layer_profiles: Dict[str, Dict[str, object]],
    target_keep: int,
    max_iter: int = 64,
) -> Dict[str, int]:
    """Solve min sum_l E_l(k_l) s.t. sum_l d_out_l * k_l ~= target_keep."""
    names = list(layer_profiles.keys())
    if not names:
        return {}

    max_ratio = 0.0
    max_total_keep = 0
    for prof in layer_profiles.values():
        d_out = int(prof["d_out"])
        d_in = int(prof["d_in"])
        col_gain = prof["col_gain"]
        if col_gain.numel() > 0:
            max_ratio = max(max_ratio, float(col_gain.abs().max().item()) / max(d_out, 1))
        max_total_keep += d_out * d_in

    target_keep = max(0, min(int(target_keep), int(max_total_keep)))
    lam_lo = -max(max_ratio * 2.0, 1e-6)
    lam_hi = max(max_ratio * 2.0, 1e-6)

    def total_keep_for_lambda(lam: float) -> Tuple[int, Dict[str, int]]:
        alloc: Dict[str, int] = {}
        total_keep = 0
        for name in names:
            prof = layer_profiles[name]
            k = _optimal_k_for_lambda(prof["prefix_gain"], int(prof["d_out"]), lam)
            alloc[name] = k
            total_keep += int(prof["d_out"]) * k
        return total_keep, alloc

    for _ in range(8):
        keep_lo, _ = total_keep_for_lambda(lam_lo)
        if keep_lo >= target_keep:
            break
        lam_lo *= 2.0
    for _ in range(8):
        keep_hi, _ = total_keep_for_lambda(lam_hi)
        if keep_hi <= target_keep:
            break
        lam_hi *= 2.0

    best_keep, best_alloc = total_keep_for_lambda(lam_lo)
    best_err = abs(best_keep - target_keep)

    for _ in range(max_iter):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        keep_mid, alloc_mid = total_keep_for_lambda(lam_mid)
        err_mid = abs(keep_mid - target_keep)
        if err_mid < best_err:
            best_keep, best_alloc, best_err = keep_mid, alloc_mid, err_mid
            if best_err == 0:
                break
        if keep_mid > target_keep:
            lam_lo = lam_mid
        elif keep_mid < target_keep:
            lam_hi = lam_mid
        else:
            best_keep, best_alloc, best_err = keep_mid, alloc_mid, 0
            break

    return best_alloc


@torch.no_grad()
def allocate_zero_bit_layer_sparsity(
    block,
    input_feat: Dict[str, torch.Tensor],
    adapter,
    config,
    *,
    target_sparsity: float,
    vision_mask: Optional[torch.Tensor] = None,
    block_pi_t: Optional[float] = None,
    method: str = "zero_bit_joint",
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Allocate per-layer sparsity inside a block from zero-bit RD curves."""
    if method not in {"zero_bit_d0", "zero_bit_joint"}:
        raise ValueError(f"unknown zero-bit allocation method: {method}")
    if config.pruning_method != "jsq_v5":
        raise ValueError("zero-bit allocation currently requires pruning_method=jsq_v5")
    if config.prune_n != 0 or config.prune_m != 0:
        raise ValueError("zero-bit allocation currently requires unstructured pruning")

    named_linears = adapter.get_named_linears(block)
    layer_profiles: Dict[str, Dict[str, object]] = {}

    pi_t_eff = float(config.pi_t if block_pi_t is None else block_pi_t)
    joint = method == "zero_bit_joint"

    for name, linear in named_linears.items():
        if name not in input_feat:
            continue

        feat = _prepare_feat(input_feat[name]).to(linear.weight.device)
        scores = compute_jsq_v5_zero_bit_metric(
            linear.weight.data,
            feat,
            vision_mask=vision_mask,
            pi_t=pi_t_eff,
            lambda_floor=config.lambda_floor,
            w_bits_act=config.a_bits,
            w_bits=config.w_bits,
            weight_quant=config.weight_quant,
            joint=joint,
        )

        d0 = scores["d0"]
        utility = scores["utility"]
        profile = _build_layer_rd_profile(
            utility=utility,
            d0=d0,
            d_out=int(linear.weight.shape[0]),
            d_in=int(linear.weight.shape[1]),
        )
        profile["utility_mean"] = float(utility.mean().item())
        profile["utility_max"] = float(utility.max().item())
        if joint and "d8" in scores:
            profile["d8_mean"] = float(scores["d8"].mean().item())
        layer_profiles[name] = profile

        del d0, utility, scores

    target_keep = int(round((1.0 - target_sparsity) * sum(
        int(p["d_out"]) * int(p["d_in"]) for p in layer_profiles.values()
    )))
    keep_alloc = _solve_block_keep_allocation(layer_profiles, target_keep)

    layer_sparsity: Dict[str, float] = {}
    layer_stats: Dict[str, Dict[str, float]] = {}
    realized_keep = 0
    predicted_error = 0.0

    for name, prof in layer_profiles.items():
        d_in = int(prof["d_in"])
        d_out = int(prof["d_out"])
        base0 = float(prof["base0"])
        prefix_gain = prof["prefix_gain"]

        k = int(keep_alloc.get(name, 0))
        k = max(0, min(k, d_in))
        s_l = 1.0 - (k / max(d_in, 1))
        layer_sparsity[name] = float(s_l)

        realized_keep += d_out * k
        predicted_error += base0 - float(prefix_gain[k].item())

        layer_stats[name] = {
            "keep_per_row": float(k),
            "sparsity": float(s_l),
            "base0": base0,
            "pred_err": base0 - float(prefix_gain[k].item()),
            "utility_mean": float(prof["utility_mean"]),
            "utility_max": float(prof["utility_max"]),
        }
        if "d8_mean" in prof:
            layer_stats[name]["d8_mean"] = float(prof["d8_mean"])

    stats: Dict[str, object] = {
        "target_keep": float(target_keep),
        "realized_keep": float(realized_keep),
        "predicted_error": float(predicted_error),
        "layers": layer_stats,
    }
    return layer_sparsity, stats
