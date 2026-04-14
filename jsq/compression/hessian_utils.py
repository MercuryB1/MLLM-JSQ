"""Hessian utilities for JSQ v5 mixture-Hessian metric.

Computes diag(H^-1) for H = U^T U + lam * I where U stacks per-modality
activations. Uses Woodbury identity to avoid forming d x d matrices:

    H^-1 = (1/lam) I - (1/lam^2) U^T (I_n + U U^T / lam)^-1 U

    diag(H^-1)_j = 1/lam - (1/lam^2) * || L^{-T} U[:, j] ||^2

where L is Cholesky of M = I_n + U U^T / lam.

Cost per layer:
  - U U^T : O(n^2 d)
  - Cholesky of n x n : O(n^3)
  - Solve + squared norm per column : O(n^2 d)

For Qwen2-VL-7B (d_mlp ~ 19k, n ~ 4k after subsampling), this stays in
single-digit seconds per layer.
"""
from __future__ import annotations

import torch


def _subsample_rows(x: torch.Tensor, max_rows: int) -> torch.Tensor:
    """Uniform-stride subsample to at most *max_rows* rows."""
    n = x.shape[0]
    if n <= max_rows:
        return x.contiguous()
    step = max(1, n // max_rows)
    return x[::step][:max_rows].contiguous()


def mixture_hinv_diag(
    x_t: torch.Tensor | None,
    x_v: torch.Tensor | None,
    pi_t: float,
    pi_v: float,
    lam: float,
    max_rows: int = 4096,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute diag((pi_t C_t + pi_v C_v + lam I)^-1) via Woodbury.

    C_mod = X_mod^T X_mod / n_mod is the per-token second-moment matrix.
    Normalizing by n_mod before weighting by pi ensures pi actually controls
    the modality prior: without it, the modality with more tokens dominates
    regardless of pi, since X^T X grows linearly with row count.

    Implementation: build U by scaling each modality's rows by
    sqrt(pi_mod / n_mod). Then U^T U = sum_mod (pi_mod / n_mod) X_mod^T X_mod
    = sum_mod pi_mod C_mod, and lam matches the per-token variance scale.

    Args:
        x_t: text activations [n_t, d] or None.
        x_v: vision activations [n_v, d] or None.
        pi_t, pi_v: mixture weights (sum should be 1.0 but not enforced).
        lam: regularizer in per-token-variance units (see estimate_lam).
        max_rows: per-modality row cap for memory.
        device: target device (defaults to the first available input's device).
        dtype: compute dtype (fp32 recommended for numerical stability).

    Returns:
        diag: [d] tensor with diag(H^-1) entries (> 0).
    """
    tensors = []
    if x_t is not None and pi_t > 0 and x_t.shape[0] > 0:
        xt = _subsample_rows(x_t, max_rows).to(device=device, dtype=dtype)
        n_t = xt.shape[0]
        tensors.append(xt * ((pi_t / n_t) ** 0.5))
    if x_v is not None and pi_v > 0 and x_v.shape[0] > 0:
        xv = _subsample_rows(x_v, max_rows).to(device=device, dtype=dtype)
        n_v = xv.shape[0]
        tensors.append(xv * ((pi_v / n_v) ** 0.5))

    if not tensors:
        d = (x_t if x_t is not None else x_v).shape[-1]
        return torch.full((d,), 1.0 / max(lam, 1e-8),
                          device=device, dtype=dtype)

    U = torch.cat(tensors, dim=0)  # [n, d]
    n, d = U.shape
    inv_lam = 1.0 / max(lam, 1e-8)

    # M = I_n + U U^T / lam  (n x n)
    M = U @ U.t()
    M.mul_(inv_lam)
    M.diagonal().add_(1.0)

    # Cholesky: M = L L^T
    # Add small jitter if numerically indefinite
    jitter = 0.0
    for _ in range(3):
        try:
            L = torch.linalg.cholesky(M)
            break
        except Exception:
            jitter = max(jitter * 10, 1e-6)
            M.diagonal().add_(jitter)
    else:
        raise RuntimeError("mixture_hinv_diag: Cholesky failed after jitter retries")

    # S = L^{-1} U  ->  then || S[:, j] ||^2  is  U[:, j]^T M^{-1} U[:, j]
    S = torch.linalg.solve_triangular(L, U, upper=False)  # [n, d]
    col_quad = (S * S).sum(dim=0)  # [d]

    diag = inv_lam - (inv_lam ** 2) * col_quad
    # Clamp for numerical safety; by construction diag > 0.
    diag.clamp_(min=inv_lam * 1e-6)
    return diag


def estimate_lam(
    x: torch.Tensor,
    w_bits_act: int = 8,
    floor: float = 1e-3,
) -> float:
    """Estimate the regularizer lam = sigma_A^2 + lw_shrinkage.

    sigma_A^2: expected per-token activation quantization variance under
    per-token absmax W{w_bits_act} quant, averaged over tokens and channels.
    For a per-token step s_t = max_j|X_tj| / q_max, uniform noise variance
    is s_t^2 / 12. We take the mean over tokens.

    Ledoit-Wolf shrinkage is approximated as trace(X^T X)/(n*d) to provide
    a diagonal floor when the empirical covariance is low-rank.
    """
    q_max = 2 ** (w_bits_act - 1) - 1
    x = x.reshape(-1, x.shape[-1])
    # per-token absmax
    s_t = x.abs().amax(dim=-1) / q_max        # [n]
    sigma_a_sq = (s_t.pow(2).mean() / 12.0).item()

    # Diagonal magnitude of X^T X / n (per-channel variance average)
    lw = (x.pow(2).mean()).item()
    lw_shrink = 0.01 * lw   # conservative shrinkage weight

    return max(sigma_a_sq + lw_shrink, floor)
