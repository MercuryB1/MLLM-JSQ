"""Per-block multimodal sensitivity allocator for JSQ v5 Option E.

Computes modality-split Block Influence (BI) for every decoder block, then
allocates non-uniform sparsity based on mixed sensitivity.

Design (symmetric to the element-level mixture-H metric):

    S_l  = pi_t * BI_t[l] + pi_v * BI_v[l]             (block score)

    invert=False (protect sensitive):
        raw  = 1 / (S_l ** alpha + eps)    → high BI → low sparsity
    invert=True  (prune sensitive):
        raw  = S_l ** alpha + eps          → high BI → high sparsity

    s_l  = s_target * raw_l / mean(raw) , then clip + rescale to preserve budget.

Where BI_t / BI_v are computed from per-token cosine distance:

    BI_m[l] = mean_{i in modality m} (1 - cos(x_i, y_i))
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger


@torch.no_grad()
def block_influence_split(
    x: torch.Tensor,
    y: torch.Tensor,
    vision_mask: Optional[torch.Tensor],
) -> Tuple[float, float, int, int]:
    """Compute per-token (1 - cos(x, y)) split by modality.

    Args:
        x: [tokens, hidden] input to the block.
        y: [tokens, hidden] output of the block.
        vision_mask: optional [tokens] bool tensor (True = vision token).

    Returns:
        (sum_text, sum_vision, n_text, n_vision) — use sums and counts so
        callers can aggregate across samples before dividing.
    """
    x = x.reshape(-1, x.shape[-1]).float()
    y = y.reshape(-1, y.shape[-1]).float()
    if x.shape != y.shape:
        raise ValueError(f"x/y shape mismatch: {x.shape} vs {y.shape}")

    xn = x.norm(dim=-1).clamp_min(1e-8)
    yn = y.norm(dim=-1).clamp_min(1e-8)
    cos = (x * y).sum(dim=-1) / (xn * yn)
    dist = (1.0 - cos).clamp_min(0.0)  # [tokens]

    if vision_mask is None or not vision_mask.any():
        return float(dist.sum()), 0.0, int(dist.numel()), 0

    vmask = vision_mask.to(dist.device).bool()
    if vmask.numel() != dist.numel():
        logger.warning(
            f"block_influence_split: mask size {vmask.numel()} != tokens {dist.numel()}, "
            f"treating all as text"
        )
        return float(dist.sum()), 0.0, int(dist.numel()), 0

    d_v = dist[vmask]
    d_t = dist[~vmask]
    return float(d_t.sum()), float(d_v.sum()), int(d_t.numel()), int(d_v.numel())


def aggregate_sample_bi(
    per_sample: List[Tuple[float, float, int, int]],
) -> Tuple[float, float]:
    """Aggregate per-sample (sum_t, sum_v, n_t, n_v) into (BI_t, BI_v)."""
    st = sum(p[0] for p in per_sample)
    sv = sum(p[1] for p in per_sample)
    nt = sum(p[2] for p in per_sample)
    nv = sum(p[3] for p in per_sample)
    bi_t = st / max(nt, 1)
    bi_v = sv / max(nv, 1)
    return bi_t, bi_v


def allocate_per_block_sparsity(
    bi_t: List[float],
    bi_v: List[float],
    *,
    pi_t: float,
    target: float,
    alpha: float = 1.0,
    s_min: float = 0.1,
    s_max: float = 0.7,
    method: str = "bi_mixture",
    invert: bool = False,
    eps: float = 1e-6,
    max_iters: int = 20,
) -> List[float]:
    """Allocate per-block sparsity from modality-split BI scores.

    Args:
        bi_t, bi_v: per-block (length L) Block Influence for text / vision.
        pi_t: text weight in the mixed score (pi_v = 1 - pi_t).
        target: global target sparsity (mean over blocks, unweighted).
        alpha: inverse-sensitivity exponent (higher = more aggressive spread).
        s_min, s_max: per-block clip range.
        method: "bi_mixture" | "bi_text" | "bi_vision".
        invert: if True, high BI → high sparsity (prune sensitive blocks).

    Returns:
        list of length L; mean(result) ≈ target.
    """
    if len(bi_t) != len(bi_v):
        raise ValueError("bi_t and bi_v must have equal length")
    if not (0.0 < target < 1.0):
        raise ValueError(f"target sparsity out of range: {target}")
    if s_min >= s_max:
        raise ValueError(f"s_min ({s_min}) must be < s_max ({s_max})")
    if not (s_min <= target <= s_max):
        logger.warning(
            f"target {target} outside clip range [{s_min}, {s_max}] — rescale will saturate"
        )

    pi_v = 1.0 - pi_t
    if method == "bi_text":
        scores = [float(t) for t in bi_t]
    elif method == "bi_vision":
        scores = [float(v) for v in bi_v]
    elif method == "bi_mixture":
        scores = [pi_t * float(t) + pi_v * float(v) for t, v in zip(bi_t, bi_v)]
    else:
        raise ValueError(f"unknown block alloc method: {method}")

    import numpy as np
    s_arr = np.asarray(scores, dtype=np.float64)
    if invert:
        raw = np.power(s_arr, alpha) + eps       # high sensitivity → high sparsity
    else:
        raw = 1.0 / (np.power(s_arr, alpha) + eps)  # high sensitivity → low sparsity

    # Initial allocation proportional to raw, mean = target
    alloc = raw / raw.mean() * target

    # Iterated clip + rescale on free entries so mean(alloc) == target
    L = len(alloc)
    for _ in range(max_iters):
        clipped_low = alloc < s_min
        clipped_high = alloc > s_max
        alloc = np.clip(alloc, s_min, s_max)
        fixed_mass = alloc[clipped_low].sum() + alloc[clipped_high].sum()
        free_mask = ~(clipped_low | clipped_high)
        n_free = int(free_mask.sum())
        if n_free == 0:
            break
        free_target_mass = target * L - fixed_mass
        current_free_mass = alloc[free_mask].sum()
        if current_free_mass < 1e-9 or abs(current_free_mass - free_target_mass) < 1e-6:
            break
        alloc[free_mask] *= free_target_mass / current_free_mass
        if (alloc[free_mask] >= s_min - 1e-9).all() and (alloc[free_mask] <= s_max + 1e-9).all():
            alloc = np.clip(alloc, s_min, s_max)
            break

    realized = float(alloc.mean())
    logger.info(
        f"Block sparsity alloc: target={target:.3f} realized={realized:.3f} "
        f"min={float(alloc.min()):.3f} max={float(alloc.max()):.3f} "
        f"method={method} alpha={alpha} invert={invert}"
    )
    return [float(x) for x in alloc]


def summarize_allocation(alloc: List[float], scores: Optional[List[float]] = None) -> str:
    """Human-readable summary for logging."""
    import numpy as np
    a = np.asarray(alloc)
    lines = [
        f"  n_blocks={len(alloc)} mean={a.mean():.3f} std={a.std():.3f} "
        f"min={a.min():.3f} max={a.max():.3f}",
    ]
    if scores is not None:
        s = np.asarray(scores)
        lines.append(
            f"  scores: mean={s.mean():.4f} min={s.min():.4f} max={s.max():.4f}"
        )
    return "\n".join(lines)
