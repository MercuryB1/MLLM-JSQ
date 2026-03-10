"""Pruning pass: JSQ v1/v2, WANDA, and Magnitude."""
from typing import Dict

import torch
import torch.nn as nn
from loguru import logger

from .base import CompressionPass


# ---------------------------------------------------------------------------
# Pruning metric functions (pure, no model-type dependency)
# ---------------------------------------------------------------------------

def _wanda_metric(w: torch.Tensor, inp: torch.Tensor, nsamples: int) -> torch.Tensor:
    """WANDA metric: |W| * sqrt(||X||_2^2 / nsamples)."""
    if inp.dim() == 2:
        inp = inp.unsqueeze(0)
    inp = inp.reshape(-1, inp.shape[-1]).t().float().to(w.device)
    scaler_row = torch.norm(inp, p=2, dim=1) ** 2 / nsamples
    return w.abs() * torch.sqrt(scaler_row.reshape(1, -1))


def _jsq_v1_metric(
    w: torch.Tensor, inp: torch.Tensor, nsamples: int, rho: float
) -> torch.Tensor:
    """JSQ v1: WANDA metric + rho * sensitivity scores (row-parallel)."""
    base = _wanda_metric(w, inp, nsamples)

    activation = inp[0].to(w.device) if inp.dim() == 3 else inp.to(w.device)
    if activation.dim() == 3:
        activation = activation.reshape(-1, activation.shape[-1])

    cout, cin = w.shape
    original_out = activation @ w.T  # (N, cout)
    ss = torch.zeros_like(w)

    for i in range(cout):
        col_out = original_out[:, i]                        # (N,)
        contributions = activation * w[i].unsqueeze(0)     # (N, cin)
        modified = col_out.unsqueeze(1) - contributions    # (N, cin)
        ss[i] = modified.max(dim=0)[0] - modified.min(dim=0)[0]
        del contributions, modified, col_out

    ss[torch.isinf(ss)] = 100.0
    return base + rho * ss


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


# ---------------------------------------------------------------------------
# CompressionPass implementation
# ---------------------------------------------------------------------------

class PruningPass(CompressionPass):
    """Prune weights in every Linear layer of the block."""

    def apply(self, block, input_feat: Dict[str, torch.Tensor], adapter, config) -> None:
        if config.sparsity_ratio == 0.0 and config.prune_n == 0:
            return

        named_linears = adapter.get_named_linears(block)

        for name, linear in named_linears.items():
            if name not in input_feat:
                logger.warning(f"PruningPass: no input_feat for '{name}', skipping")
                continue

            w = linear.weight
            feat = input_feat[name]

            # "__nsamples__" is set by collect_block_input_feat in multimodal mode
            # (where feat is 2D [total_tokens, hidden_in]).
            # In text mode feat is 3D [n_samples, seq_len, hidden_in] and
            # nsamples = feat.shape[0].
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
                config.sparsity_ratio, config.prune_n, config.prune_m,
            )
