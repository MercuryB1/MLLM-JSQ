"""Pruning pass: JSQ v1/v2, WANDA, and Magnitude."""
from typing import Dict, Optional

import torch
import torch.nn as nn
from loguru import logger

from .base import CompressionPass



def _wanda_metric(w: torch.Tensor, inp: torch.Tensor, nsamples: int) -> torch.Tensor:
    """WANDA metric: |W| * sqrt(||X||_2^2 / nsamples)."""
    if inp.dim() == 2:
        inp = inp.unsqueeze(0)
    inp = inp.reshape(-1, inp.shape[-1]).t().float().to(w.device)
    scaler_row = torch.norm(inp, p=2, dim=1) ** 2 / nsamples
    return w.abs() * torch.sqrt(scaler_row.reshape(1, -1))


def _jsq_v1_metric(
    w: torch.Tensor, inp: torch.Tensor, nsamples: int, rho: float,
    max_tokens: int = 4096,
) -> torch.Tensor:
    """JSQ v1: WANDA metric + rho * sensitivity (std of leave-one-out output).

    Replaces the double Python loop with a single cross-covariance matmul:

        Var(out[:,j] - act[:,c]*w[j,c])
            = Var(out[:,j]) - 2*Cov(out[:,j], act[:,c])*w[j,c] + Var(act[:,c])*w[j,c]^2

    Complexity is still O(T·cout·cin) but executed as a single BLAS call,
    giving ~10-100× speedup.  Token count is capped at max_tokens for memory safety.
    """
    base = _wanda_metric(w, inp, nsamples)

    act = inp[0].to(w.device) if inp.dim() == 3 else inp.to(w.device)
    if act.dim() == 3:
        act = act.reshape(-1, act.shape[-1])
    act = act.float()
    w_f = w.float()
    N = act.shape[0]

    # Uniform stride subsampling — deterministic, avoids OOM on large layers
    if N > max_tokens:
        step = (N + max_tokens - 1) // max_tokens
        act = act[::step].contiguous()
        N = act.shape[0]

    out = act @ w_f.T          # [N, cout]

    E_out = out.mean(0, keepdim=True)   # [1, cout]
    E_act = act.mean(0, keepdim=True)   # [1, cin]
    out_c = out - E_out                 # [N, cout]
    act_c = act - E_act                 # [N, cin]

    var_out = (out_c ** 2).mean(0)      # [cout]
    var_act = (act_c ** 2).mean(0)      # [cin]
    cov = (out_c.T @ act_c) / N        # [cout, cin]  — single matmul

    ss = (
        var_out.unsqueeze(1)
        - 2.0 * cov * w_f
        + var_act.unsqueeze(0) * w_f.pow(2)
    ).clamp(min=0.0).sqrt().clamp(max=100.0)

    return base + rho * ss.to(w.dtype)


def _jsq_v2_metric(
    w: torch.Tensor, inp: torch.Tensor, nsamples: int, rho: float
) -> torch.Tensor:
    """JSQ v2: WANDA metric + rho * per-weight sensitivity (slow, exhaustive)."""
    base = _wanda_metric(w, inp, nsamples)

    activation = inp[0].to(w.device) if inp.dim() == 3 else inp.to(w.device)
    if activation.dim() == 3:
        activation = activation.reshape(-1, activation.shape[-1])

    cout, cin = w.shape
    ss = torch.zeros_like(w)

    for i in range(cout):
        for j in range(cin):
            modified_w = w.clone()
            modified_w[i, j] = 0
            modified_out = activation @ modified_w.T
            row_diff = modified_out.max(dim=1)[0] - modified_out.min(dim=1)[0]
            ss[i, j] = row_diff.sum()
        ss[torch.isinf(ss)] = 100.0

    return base + rho * ss


def _apply_mask(w: torch.Tensor, metric: torch.Tensor, sparsity_ratio: float,
                prune_n: int, prune_m: int) -> None:
    """Apply pruning mask to w in-place."""
    if prune_n != 0:
        mask = torch.zeros_like(metric, dtype=torch.bool)
        for i in range(0, metric.shape[1], prune_m):
            block = metric[:, i:i + prune_m].float()
            idx = torch.topk(block, prune_n, dim=1, largest=False)[1]
            mask.scatter_(1, i + idx, True)
    else:
        sorted_idx = torch.sort(metric, dim=-1, stable=True)[1]
        k = int(metric.shape[1] * sparsity_ratio)
        mask = torch.zeros_like(metric, dtype=torch.bool)
        mask.scatter_(1, sorted_idx[:, :k], True)

    w[mask] = 0.0



class PruningPass(CompressionPass):
    """Prune weights in every Linear layer of the block.

    Supports an optional *per_layer_sparsity* dict that maps layer name to a
    specific sparsity ratio, enabling block-level search (MA-JSQ).  When the
    dict is supplied, each layer uses its own ratio instead of the global
    ``config.sparsity_ratio``.  Layers absent from the dict fall back to the
    global value.
    """

    _supports_per_layer = True

    def apply(
        self,
        block,
        input_feat: Dict[str, torch.Tensor],
        adapter,
        config,
        per_layer_sparsity: Optional[Dict[str, float]] = None,
    ) -> None:
        if config.sparsity_ratio == 0.0 and config.prune_n == 0 and not per_layer_sparsity:
            return

        named_linears = adapter.get_named_linears(block)

        for name, linear in named_linears.items():
            if name not in input_feat:
                logger.warning(f"PruningPass: no input_feat for '{name}', skipping")
                continue

            # Resolve per-layer or global sparsity ratio
            if per_layer_sparsity is not None:
                layer_sparsity = per_layer_sparsity.get(name, config.sparsity_ratio)
            else:
                layer_sparsity = config.sparsity_ratio

            if layer_sparsity == 0.0 and config.prune_n == 0:
                continue

            w = linear.weight
            feat = input_feat[name]

            # Multimodal: feat is 2D [total_tokens, hidden_in] + __nsamples__ key.
            # Text: feat is 3D [n_samples, seq_len, hidden_in].
            if feat.dim() == 2:
                nsamples = int(input_feat.get("__nsamples__", 1))
                feat = feat.unsqueeze(0)  # → [1, total_tokens, hidden_in]
            else:
                nsamples = feat.shape[0]

            feat = feat.to(w.device)

            if config.pruning_method == "magnitude":
                metric = w.abs()
            elif config.pruning_method == "wanda":
                metric = _wanda_metric(w.data, feat, nsamples)
            elif config.pruning_method == "jsq_v1":
                metric = _jsq_v1_metric(w.data, feat, nsamples, config.rho)
            elif config.pruning_method == "jsq_v2":
                metric = _jsq_v2_metric(w.data, feat, nsamples, config.rho)
            else:
                raise ValueError(f"Unknown pruning_method: {config.pruning_method}")

            _apply_mask(
                w.data, metric,
                layer_sparsity, config.prune_n, config.prune_m,
            )
