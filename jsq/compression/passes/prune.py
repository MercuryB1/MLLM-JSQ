"""Pruning pass: JSQ v1/v2/v3/v4, WANDA, and Magnitude."""
from typing import Dict, List, Optional

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


def _density_aware_scale(
    inp_flat: torch.Tensor, nsamples: int, alpha: float = 0.5,
) -> torch.Tensor:
    """Density-weighted activation scale per input channel.

    Upweights outlier tokens (high activation norm) regardless of modality.
    When alpha=0, degenerates to standard WANDA scale.

    Args:
        inp_flat: [total_tokens, cin] all calibration tokens concatenated.
        nsamples: number of calibration samples.
        alpha: density weighting strength (0 = standard WANDA).

    Returns:
        scale: [cin] density-weighted activation scale.
    """
    if alpha == 0.0:
        return (inp_flat.pow(2).sum(0) / nsamples).sqrt()

    token_norm = inp_flat.norm(dim=-1)
    mu = token_norm.mean()
    sigma = token_norm.std() + 1e-8
    density = ((token_norm - mu) / sigma).clamp(min=0)
    token_weight = 1.0 + alpha * density  # [tokens]

    scale_sq = (inp_flat.pow(2) * token_weight.unsqueeze(-1)).sum(0) / nsamples
    return scale_sq.sqrt()


def _quant_damage(
    w: torch.Tensor, w_bits: int = 8, top_k: int = 3,
) -> torch.Tensor:
    """Per-weight quantization damage score.

    Penalises top-k outlier weights per row: pruning them shrinks the
    per-channel absmax quantization step, improving quantization precision
    for all remaining weights in the same row.

    Args:
        w: [cout, cin] weight matrix (float).
        w_bits: target quantization bit width.
        top_k: number of top outlier positions to penalize per row.

    Returns:
        damage: [cout, cin] quantization damage score (>= 0).
    """
    w_abs = w.abs()
    q_max = 2 ** (w_bits - 1) - 1

    # Current per-row quantization error
    row_max = w_abs.max(dim=-1, keepdim=True)[0]  # [cout, 1]
    step = row_max / q_max
    quant_err = ((w / step).round() * step - w).pow(2)  # [cout, cin]
    row_mean_err = quant_err.mean(dim=-1, keepdim=True)  # [cout, 1]

    # Top-k outlier detection
    if top_k >= w.shape[1]:
        return torch.zeros_like(w)
    sorted_abs, _ = w_abs.sort(dim=-1, descending=True)
    threshold = sorted_abs[:, top_k : top_k + 1]  # [cout, 1]
    is_outlier = w_abs > threshold  # [cout, cin]

    # Severity: how much this weight sticks out above threshold
    severity = ((w_abs - threshold) / (threshold + 1e-8)).clamp(min=0)

    return is_outlier.float() * severity * row_mean_err


def _jsq_v3_metric(
    w: torch.Tensor,
    inp: torch.Tensor,
    nsamples: int,
    rho: float = 1.0,
    alpha: float = 0.5,
    beta: float = 0.5,
    w_bits: int = 8,
    top_k: int = 3,
    max_tokens: int = 4096,
) -> torch.Tensor:
    """JSQ v3: unified quantization-aware + density-aware pruning metric.

    metric(i,j) = |W_ij| * S_j(density)
                  + rho * norm_sensitivity(i,j)
                  - beta * quant_damage(i,j) * S_j

    Three orthogonal, independently ablatable components:
      - alpha: density-aware activation scale (token info density weighting)
      - rho:   leave-one-out sensitivity (inherited from JSQ v1), normalized
      - beta:  quantization damage penalty (prune outliers that hurt quant)

    Special cases:
      - alpha=0, rho>0, beta=0 → ~JSQ v1 (with standard WANDA scale)
      - alpha=0, rho=0, beta=0 → WANDA
    """
    # Use ALL tokens for scale computation (matching _wanda_metric behavior)
    act_full = inp.reshape(-1, inp.shape[-1]).float().to(w.device)

    # Component 1: density-aware activation scale (on full tokens)
    scale = _density_aware_scale(act_full, nsamples, alpha)  # [cin]
    base = w.abs() * scale.unsqueeze(0)  # [cout, cin]

    # Subsample tokens only for the expensive sensitivity matmul
    act = act_full
    if act.shape[0] > max_tokens:
        step = (act.shape[0] + max_tokens - 1) // max_tokens
        act = act[::step].contiguous()
    N = act.shape[0]

    # Component 2: sensitivity (reuse JSQ v1 cross-covariance trick)
    # Normalized to same scale as base to ensure balanced contribution
    if rho > 0:
        w_f = w.float()
        out = act @ w_f.T  # [N, cout]
        out_c = out - out.mean(0, keepdim=True)
        act_c = act - act.mean(0, keepdim=True)
        var_out = (out_c ** 2).mean(0)  # [cout]
        var_act = (act_c ** 2).mean(0)  # [cin]
        cov = (out_c.T @ act_c) / N  # [cout, cin]
        ss = (
            var_out.unsqueeze(1)
            - 2.0 * cov * w_f
            + var_act.unsqueeze(0) * w_f.pow(2)
        ).clamp(min=0.0).sqrt().clamp(max=100.0)
        # Normalize sensitivity to same magnitude as base so rho is meaningful
        base_mean = base.mean() + 1e-8
        ss_mean = ss.mean() + 1e-8
        ss_norm = ss * (base_mean / ss_mean)
        base = base + rho * ss_norm.to(w.dtype)

    # Component 3: quantization damage penalty
    if beta > 0:
        qd = _quant_damage(w.data.float(), w_bits=w_bits, top_k=top_k)
        qd = qd * scale.unsqueeze(0).to(qd.device)
        # Normalize qd to same scale as base
        base_mean = base.mean() + 1e-8
        qd_mean = qd.mean() + 1e-8
        qd_norm = qd * (base_mean / qd_mean)
        base = base - beta * qd_norm.to(w.dtype)

    return base


def _leave_one_out_sensitivity(
    act: torch.Tensor, w_f: torch.Tensor, max_clamp: float = 100.0,
) -> torch.Tensor:
    """Leave-one-out output sensitivity via cross-covariance trick.

    Args:
        act: [N, cin] activation matrix (float32, on device).
        w_f: [cout, cin] weight matrix (float32).
        max_clamp: upper clamp for numerical stability.

    Returns:
        ss: [cout, cin] sensitivity scores.
    """
    N = act.shape[0]
    if N == 0:
        return torch.zeros_like(w_f)
    out = act @ w_f.T                       # [N, cout]
    out_c = out - out.mean(0, keepdim=True)
    act_c = act - act.mean(0, keepdim=True)
    var_out = (out_c ** 2).mean(0)          # [cout]
    var_act = (act_c ** 2).mean(0)          # [cin]
    cov = (out_c.T @ act_c) / N            # [cout, cin]
    ss = (
        var_out.unsqueeze(1)
        - 2.0 * cov * w_f
        + var_act.unsqueeze(0) * w_f.pow(2)
    ).clamp(min=0.0).sqrt().clamp(max=max_clamp)
    return ss


def _jsq_v4_metric(
    w: torch.Tensor,
    inp: torch.Tensor,
    nsamples: int,
    vision_mask: Optional[torch.Tensor] = None,
    rho: float = 2.1,
    gamma: float = 1.0,
    w_bits: int = 8,
    max_tokens: int = 4096,
) -> torch.Tensor:
    """JSQ v4: joint quantization-aware + modality-split pruning metric.

    Three innovations over WANDA/JSQ v1:

    1. Quantization-aware importance: replace |W| with (|W| - |W - W_q|),
       reflecting post-quantization contribution rather than raw magnitude.
       W_q is per-channel absmax quantization: step = max(|W_row|) / (2^(b-1)-1).

    2. Modality-split activation scale: S_vis + gamma * S_txt instead of
       a single mixed S, so text channels get explicit gamma weighting
       regardless of the (typically large) number of vision tokens.

    3. Modality-split leave-one-out sensitivity: ss_vis + gamma * ss_txt,
       same cross-covariance trick as JSQ v1 but computed per-modality.

    Falls back to JSQ v1 behavior when vision_mask is None (text-only).

    Args:
        w: [cout, cin] weight matrix.
        inp: activations (2D or 3D).
        nsamples: number of calibration samples.
        vision_mask: [total_tokens] bool (True = vision), or None.
        rho: sensitivity weight.
        gamma: text-modality balance factor (>1 upweights text).
        w_bits: target quantization bit width.
        max_tokens: max tokens for sensitivity matmul (memory cap).
    """
    act = inp.reshape(-1, inp.shape[-1]).float().to(w.device)
    w_f = w.float()

    # --- Component 1: Quantization-aware importance ---
    q_max = 2 ** (w_bits - 1) - 1  # 127 for INT8
    row_max = w_f.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
    step = row_max / q_max
    w_q = (w_f / step).round() * step
    # |W| - |W - W_q|: rewards weights whose quantized value is close to original
    quant_imp = (w_f.abs() - (w_f - w_q).abs()).clamp(min=0.0)

    # --- Component 2: Modality-split activation scale ---
    has_modal_split = (
        vision_mask is not None
        and vision_mask.any()
        and (~vision_mask).any()
    )

    if has_modal_split:
        vm = vision_mask.to(act.device)
        act_vis = act[vm]       # [n_vis, cin]
        act_txt = act[~vm]      # [n_txt, cin]
        s_vis = (act_vis.pow(2).sum(0) / nsamples).sqrt()   # [cin]
        s_txt = (act_txt.pow(2).sum(0) / nsamples).sqrt()   # [cin]
        scale = s_vis + gamma * s_txt                        # [cin]
    else:
        scale = (act.pow(2).sum(0) / nsamples).sqrt()        # [cin]

    base = quant_imp * scale.unsqueeze(0)  # [cout, cin]

    # --- Component 3: Modality-split sensitivity ---
    if rho > 0:
        if has_modal_split:
            vm = vision_mask.to(act.device)
            a_vis = act[vm]
            a_txt = act[~vm]
            # Subsample each modality independently
            if a_vis.shape[0] > max_tokens:
                st = (a_vis.shape[0] + max_tokens - 1) // max_tokens
                a_vis = a_vis[::st].contiguous()
            if a_txt.shape[0] > max_tokens:
                st = (a_txt.shape[0] + max_tokens - 1) // max_tokens
                a_txt = a_txt[::st].contiguous()
            ss_vis = _leave_one_out_sensitivity(a_vis, w_f)
            ss_txt = _leave_one_out_sensitivity(a_txt, w_f)
            ss = ss_vis + gamma * ss_txt
        else:
            act_sub = act
            if act_sub.shape[0] > max_tokens:
                st = (act_sub.shape[0] + max_tokens - 1) // max_tokens
                act_sub = act_sub[::st].contiguous()
            ss = _leave_one_out_sensitivity(act_sub, w_f)

        base = base + rho * ss.to(w.dtype)

    return base


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
        vision_mask: Optional[torch.Tensor] = None,
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
            elif config.pruning_method == "jsq_v3":
                metric = _jsq_v3_metric(
                    w.data, feat, nsamples,
                    rho=config.rho,
                    alpha=config.alpha,
                    beta=config.beta,
                    w_bits=config.w_bits,
                    top_k=config.top_k,
                )
            elif config.pruning_method == "jsq_v4":
                metric = _jsq_v4_metric(
                    w.data, feat, nsamples,
                    vision_mask=vision_mask,
                    rho=config.rho,
                    gamma=config.gamma,
                    w_bits=config.w_bits,
                )
            else:
                raise ValueError(f"Unknown pruning_method: {config.pruning_method}")

            _apply_mask(
                w.data, metric,
                layer_sparsity, config.prune_n, config.prune_m,
            )
