"""Analytic two-layer Hessian search for JSQ v5.

The v5 per-element importance ``I(i,j) = W_ij² / [H⁻¹]_jj`` is the second-order
Taylor coefficient of the expected input-MSE under pruning (OBS theorem).  For
a given per-layer sparsity ``s_l`` with per-row top-k pruning, the expected
layer MSE equals the sum of pruned I-values:

    E_l(s_l) = Σ_i  Σ_{j ∈ bottom-k_l(row i)} I_l[i, j],    k_l = ⌊d_in · s_l⌋

Under residual + LayerNorm decoupling, block-level error ≈ Σ_l E_l(s_l).
Allocating ``{s_l}`` then becomes a convex budget-constrained problem solved
by Lagrangian (water-filling) in O(L log L) — no block-forward pass required,
and exactly aligned with the inner v5 metric.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from loguru import logger


def compute_layer_error_curve(importance: torch.Tensor) -> torch.Tensor:
    """Precompute per-layer cumulative error curve for all possible budgets.

    Args:
        importance: [d_out, d_in] FP32 tensor of per-element I scores.

    Returns:
        curve: [d_in + 1] FP64 tensor where ``curve[k]`` = sum of the k smallest
            I values across each row, summed over rows.  ``curve[0] = 0``,
            ``curve[d_in] = importance.sum()``.
    """
    if importance.dim() != 2:
        raise ValueError(f"importance must be 2D, got {importance.shape}")
    # Sort each row ascending so smallest-k pruned values are the prefix.
    sorted_imp, _ = torch.sort(importance.float(), dim=-1, stable=False)
    # Sum across rows per column index → marginal cost at budget index k.
    col_sum = sorted_imp.sum(dim=0).double()        # [d_in]
    curve = torch.empty(col_sum.numel() + 1, dtype=torch.float64, device="cpu")
    curve[0] = 0.0
    curve[1:] = torch.cumsum(col_sum.cpu(), dim=0)
    return curve


def _budget_index(d_in: int, s: float) -> int:
    """k = round(d_in * s), clamped to [0, d_in]."""
    k = int(round(d_in * s))
    return max(0, min(d_in, k))


def _total_weighted_budget(
    s_per_layer: Dict[str, float],
    layer_params: Dict[str, int],
) -> float:
    """Return Σ_l s_l · p_l  (unweighted by total; used for ratio checks)."""
    return sum(s_per_layer[n] * layer_params[n] for n in s_per_layer)


def _optimal_s_for_lambda(
    curve: torch.Tensor,
    d_in: int,
    p_l: int,
    lam: float,
    s_min: float = 0.0,
    s_max: float = 0.9,
) -> float:
    """Given a Lagrange multiplier, pick s_l minimising E_l(s_l) + λ·s_l·p_l.

    The discrete marginal is::

        Δ_k = curve[k+1] − curve[k]    (= Σ_i I_sorted[i, k])

    The optimal k is the largest index where Δ_k ≤ λ · p_l / d_in (scaled so
    the per-sparsity unit cost is commensurate with marginal-per-index).
    """
    diffs = (curve[1:] - curve[:-1]).to(torch.float64)          # [d_in], ascending
    thresh = lam * p_l / max(d_in, 1)
    # diffs is monotonically non-decreasing (sorted → col_sum cumulative), so
    # the crossing point can be found by searchsorted.
    k_star = int(torch.searchsorted(diffs, torch.tensor(thresh, dtype=torch.float64)).item())
    s = k_star / d_in
    return max(s_min, min(s_max, s))


def water_fill_allocation(
    curves: Dict[str, torch.Tensor],
    layer_dims: Dict[str, int],
    layer_params: Dict[str, int],
    s_target: float,
    s_min: float = 0.0,
    s_max: float = 0.9,
    tol: float = 1e-4,
    max_iter: int = 64,
) -> Dict[str, float]:
    """Solve min_{s_l} Σ_l E_l(s_l) s.t. weighted-avg(s_l) == s_target.

    Uses bisection on the Lagrange multiplier λ.  The per-layer optimum
    ``s_l(λ)`` is monotonically non-decreasing in λ, so the weighted-average
    budget is monotone in λ as well.

    Args:
        curves: layer-name → [d_in+1] cumulative-error curve.
        layer_dims: layer-name → d_in.
        layer_params: layer-name → total parameter count (d_out · d_in).
        s_target: desired weighted-average sparsity.
        s_min, s_max: per-layer clamps.
        tol: bisection tolerance on weighted-avg.
        max_iter: bisection cap.

    Returns:
        Dict layer-name → s_l.  The weighted average matches s_target within
        ``tol`` (or best effort if boundaries are hit).
    """
    names = list(curves.keys())
    total_params = sum(layer_params[n] for n in names)
    target_weighted = s_target * total_params

    # λ bounds: λ=0 → every layer picks s_min (no cost to pruning beyond what
    # E already says).  Large λ → every layer hits s_max.  Initialize from the
    # range of observed marginal slopes.
    max_slope = 0.0
    for n in names:
        c = curves[n]
        if c.numel() > 1:
            d_in = layer_dims[n]
            slope = (c[-1] - c[0]) * d_in / max(layer_params[n], 1)
            max_slope = max(max_slope, float(slope))
    lam_lo, lam_hi = 0.0, max(max_slope * 2.0, 1e-6)

    def weighted_avg(lam: float) -> Tuple[float, Dict[str, float]]:
        alloc = {
            n: _optimal_s_for_lambda(
                curves[n], layer_dims[n], layer_params[n], lam, s_min, s_max
            )
            for n in names
        }
        weighted = sum(alloc[n] * layer_params[n] for n in names)
        return weighted, alloc

    # Verify bounds bracket the target; expand lam_hi if not.
    for _ in range(8):
        w_hi, _ = weighted_avg(lam_hi)
        if w_hi >= target_weighted:
            break
        lam_hi *= 4.0

    alloc = {n: s_target for n in names}
    for _ in range(max_iter):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        weighted, alloc = weighted_avg(lam_mid)
        err = weighted - target_weighted
        if abs(err) <= tol * total_params:
            break
        if err > 0:
            lam_hi = lam_mid
        else:
            lam_lo = lam_mid

    # Final rescale to exactly match budget (absorbs discretization error).
    weighted = sum(alloc[n] * layer_params[n] for n in names)
    if weighted > 1e-8 and abs(weighted - target_weighted) > tol * total_params:
        ratio = target_weighted / weighted
        alloc = {n: max(s_min, min(s_max, s * ratio)) for n, s in alloc.items()}

    return alloc


def allocate_v5(
    importance_per_layer: Dict[str, torch.Tensor],
    layer_params: Dict[str, int],
    s_target: float,
    s_min: float = 0.0,
    s_max: float = 0.9,
) -> Tuple[Dict[str, float], float]:
    """End-to-end: from per-layer I matrices to optimal {s_l}.

    Returns:
        (allocation, predicted_block_error) — the latter is Σ_l E_l(s_l*) and
        is analytic; useful for logging vs Fisher-proxy baseline.
    """
    curves: Dict[str, torch.Tensor] = {}
    layer_dims: Dict[str, int] = {}
    for name, I in importance_per_layer.items():
        curves[name] = compute_layer_error_curve(I)
        layer_dims[name] = I.shape[1]

    alloc = water_fill_allocation(
        curves, layer_dims, layer_params, s_target, s_min=s_min, s_max=s_max
    )

    total_err = 0.0
    for name, s in alloc.items():
        k = _budget_index(layer_dims[name], s)
        total_err += float(curves[name][k].item())
    return alloc, total_err
