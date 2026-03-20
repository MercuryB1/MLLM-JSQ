"""BlockSearcher: block-level search for optimal per-layer sparsity allocation.

Implements the MA-JSQ (Modal-Aware Block-level JSQ) algorithm described in
AGENT.md.  For each Transformer block it:

  1. Runs the original (uncompressed) block forward on a small eval subset to
     get Y_orig_lite and the Fisher proxy H = mean(Y_orig²).
  2. Computes per-layer Hessian trace tr(H_l) from the collected input_feat
     (chunked, stays in fp16 to avoid OOM).
  3. Creates lite_feat: input_feat subsampled to max_feat_tokens per layer so
     that candidate evaluation fits in GPU memory regardless of nsamples.
  4. Generates ~5–10 candidate per-layer sparsity configurations.
  5. Evaluates each candidate (deepcopy block → apply passes → forward on
     lite subset → Hessian-weighted block error).
  6. Applies the best-found configuration in-place to the original block,
     using lite_feat (consistent with candidate evaluation).

Memory strategy
---------------
With nsamples=128, input_feat can exceed 18 GB on 80 GB GPUs.  After
computing statistics (trace_H, act_scales via the smoothing pass), large
tensors are not needed at their full resolution.  We:
  * Keep only a lite_feat (≤ max_feat_tokens tokens per layer) on GPU.
  * Off-load the remaining full input_feat to CPU immediately.
  * Use only n_eval_samples for the candidate forward passes.

For text-only models the loss degenerates to a standard Hessian-weighted MSE.
For MLLMs the loss is modal-decoupled:

    L = mean(H_v * ||Y_v - Ŷ_v||²) + γ * mean(H_t * ||Y_t - Ŷ_t||²)
"""

import copy
import gc
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger

from .collector import _to_device, _slice_kw_for_sample
from .passes.base import CompressionPass


# ---------------------------------------------------------------------------
# Memory-efficient feature helpers
# ---------------------------------------------------------------------------

def _subsample_feat(
    input_feat: Dict[str, torch.Tensor],
    max_tokens: int = 4096,
) -> Dict[str, torch.Tensor]:
    """Return a copy of input_feat with each tensor subsampled to ≤ max_tokens.

    Subsampling uses uniform stride to preserve the full token range.  The
    result lives on the same device as the original tensors.
    """
    result: Dict = {}
    for name, feat in input_feat.items():
        if not isinstance(feat, torch.Tensor):
            result[name] = feat
            continue
        f = feat.reshape(-1, feat.shape[-1])
        n = f.shape[0]
        if n > max_tokens:
            step = max(1, n // max_tokens)
            result[name] = f[::step][:max_tokens].contiguous()
        else:
            result[name] = f.contiguous()
    return result


def _offload_feat(input_feat: Dict[str, torch.Tensor]) -> None:
    """Move all GPU tensors in input_feat to CPU in-place to free GPU memory."""
    for name in input_feat:
        if isinstance(input_feat[name], torch.Tensor) and input_feat[name].is_cuda:
            input_feat[name] = input_feat[name].cpu()


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def _hessian_block_error(
    Y_orig_flat: torch.Tensor,
    Y_hat_flat: torch.Tensor,
    H: torch.Tensor,
    vision_mask_flat: Optional[torch.Tensor],
    gamma: float,
) -> float:
    """Hessian-weighted block reconstruction error.

    Args:
        Y_orig_flat: [tokens, hidden] original block outputs (fp16 or fp32).
        Y_hat_flat:  [tokens, hidden] compressed block outputs (fp16 or fp32).
        H:           [tokens, hidden] Fisher proxy (= Y_orig²), FP32.
        vision_mask_flat: optional [tokens] bool (True = vision token), any device.
        gamma:       text-token weight factor.

    Returns:
        Scalar error (float).
    """
    # H is FP32; cast Y tensors to FP32 to avoid overflow in the difference
    diff_sq = H * (Y_orig_flat.float() - Y_hat_flat.float()).pow(2)  # FP32

    if vision_mask_flat is None:
        return diff_sq.mean().item()

    # vision_mask_flat may be on CPU while diff_sq is on GPU (multimodal mode)
    mask = vision_mask_flat.to(diff_sq.device)
    if mask.all() or (~mask).all():
        return diff_sq.mean().item()

    err_v = diff_sq[mask].mean().item()
    err_t = diff_sq[~mask].mean().item()
    return err_v + gamma * err_t


def _compute_layer_trace_H(
    input_feat: Dict[str, torch.Tensor],
    chunk_size: int = 512,
) -> Dict[str, float]:
    """Compute tr(H_l) ≈ mean(feat²) per layer.

    Processed in chunks of *chunk_size* tokens to avoid materialising a large
    fp32 copy: each chunk is cast to fp32 for numerical stability, then
    reduced to a scalar.
    """
    trace: Dict[str, float] = {}
    for name, feat in input_feat.items():
        if name == "__nsamples__" or not isinstance(feat, torch.Tensor):
            continue
        f = feat.reshape(-1, feat.shape[-1])
        n = f.shape[0]
        acc, cnt = 0.0, 0
        for i in range(0, n, chunk_size):
            c = f[i:i + chunk_size]
            acc += c.float().pow(2).sum().item()
            cnt += c.numel()
        trace[name] = acc / cnt if cnt > 0 else 0.0
    return trace


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _generate_candidates(
    layer_names: List[str],
    s_target: float,
    layer_params: Dict[str, int],
    trace_H: Dict[str, float],
    delta: float = 0.05,
    n_candidates: int = 8,
    max_sens_delta: float = 0.1,
) -> List[Dict[str, float]]:
    """Generate candidate per-layer sparsity allocations.

    Budget constraint: the parameter-weighted average of all s_l must equal
    s_target (within 0.1%).  All sparsities are clamped to [0, 0.95].

    Strategies:
      1. Uniform — all layers at s_target (baseline).
      2. Attn-light — attention layers at s_target − delta, MLP compensates.
      3. MLP-light  — reverse of Attn-light.
      4. Sensitivity-driven — high tr(H_l) ⇒ less pruning, clamped to
         [s_target − max_sens_delta, s_target + max_sens_delta] per layer to
         prevent extreme allocations when attention inputs are LayerNorm-
         normalised (trace_H ≈ 1) while MLP inputs are not.
      5–6. Variants of Attn/MLP-light with delta * 0.5.

    Returns a deduplicated list of at most *n_candidates* dicts.
    """
    if s_target == 0.0 or not layer_names:
        return [{}]

    ATTN_KEYS = {"q_proj", "k_proj", "v_proj", "o_proj",
                 "query", "key", "value", "out_proj"}

    def is_attn(name: str) -> bool:
        return any(k in name for k in ATTN_KEYS)

    def clamp_s(s: float) -> float:
        return max(0.0, min(0.95, s))

    total_params = sum(layer_params.values())

    def scale_to_budget(cfg: Dict[str, float]) -> Dict[str, float]:
        weighted = sum(cfg[n] * layer_params[n] for n in cfg)
        actual = weighted / total_params
        if abs(actual - s_target) < 1e-4 or actual < 1e-8:
            return cfg
        ratio = s_target / actual
        return {n: clamp_s(s * ratio) for n, s in cfg.items()}

    attn_names = [n for n in layer_names if is_attn(n)]
    mlp_names  = [n for n in layer_names if not is_attn(n)]
    n_attn = sum(layer_params[n] for n in attn_names)
    n_mlp  = sum(layer_params[n] for n in mlp_names)

    candidates: List[Dict[str, float]] = []

    # 1. Uniform
    candidates.append({n: s_target for n in layer_names})

    # 2 & 3. Attn-light / MLP-light at two delta scales.
    # Use symmetric ±d instead of budget-compensation formula.  For GQA
    # models, attn params ≪ MLP params, so compensation would assign extreme
    # sparsity (e.g. 78%) to small attn layers.  Using ±d and letting
    # scale_to_budget do a mild uniform rescale keeps allocations sane.
    for d in [delta, delta * 0.5]:
        if attn_names and mlp_names:
            for attn_less in (True, False):
                if attn_less:
                    a_s = clamp_s(s_target - d)
                    m_s = clamp_s(s_target + d)
                else:
                    m_s = clamp_s(s_target - d)
                    a_s = clamp_s(s_target + d)
                candidates.append(scale_to_budget({
                    n: (a_s if is_attn(n) else m_s) for n in layer_names
                }))

    # 4. Sensitivity-driven (inversely proportional to tr(H_l))
    # Clamp each layer's raw sparsity to [s_target - max_sens_delta,
    # s_target + max_sens_delta] before budget rescaling.  Without this cap,
    # attention layers (whose inputs are LayerNorm-normalised, trace_H ≈ 1)
    # receive extreme sparsity (~78%) while MLP layers (larger activations)
    # receive too little — crushing attention quality.
    if trace_H:
        eps = 1e-8
        inv_sens = {n: 1.0 / (trace_H.get(n, eps) + eps) for n in layer_names}
        total_inv_w = sum(inv_sens[n] * layer_params[n] for n in layer_names)
        if total_inv_w > 0:
            s_lo = max(0.0, s_target - max_sens_delta)
            s_hi = min(0.95, s_target + max_sens_delta)
            raw = {
                n: max(s_lo, min(s_hi,
                    inv_sens[n] * s_target * total_params / total_inv_w))
                for n in layer_names
            }
            candidates.append(scale_to_budget(raw))

    # Deduplicate and limit
    seen: set = set()
    unique: List[Dict[str, float]] = []
    for cfg in candidates:
        key = tuple(sorted((n, round(s, 4)) for n, s in cfg.items()))
        if key not in seen:
            seen.add(key)
            unique.append(cfg)

    return unique[:n_candidates]


# ---------------------------------------------------------------------------
# BlockSearcher
# ---------------------------------------------------------------------------

class BlockSearcher:
    """Block-level searcher for per-layer sparsity under a fixed budget.

    Parameters
    ----------
    passes:
        Ordered list of CompressionPass objects (same as CompressionPipeline).
    adapter:
        ModelAdapter instance.
    gamma:
        Modal balance factor γ (text vs vision error weight).
    n_search_candidates:
        Maximum number of per-layer sparsity configs to evaluate per block.
    max_feat_tokens:
        Maximum number of tokens kept per layer when subsampling input_feat
        for candidate evaluation.  Lower values save GPU memory; 4096 is a
        good default since all passes internally subsample to ≤ 4096 anyway.
    n_eval_samples:
        Number of calibration samples used for the candidate forward pass
        (error computation).  8 is enough for a reliable relative ranking.
    """

    def __init__(
        self,
        passes: List[CompressionPass],
        adapter,
        gamma: float = 1.0,
        n_search_candidates: int = 8,
        max_feat_tokens: int = 4096,
        n_eval_samples: int = 8,
    ) -> None:
        self.passes = passes
        self.adapter = adapter
        self.gamma = gamma
        self.n_search_candidates = n_search_candidates
        self.max_feat_tokens = max_feat_tokens
        self.n_eval_samples = n_eval_samples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flatten(self, outputs) -> torch.Tensor:
        """Flatten block outputs to [total_tokens, hidden] in native dtype."""
        if isinstance(outputs, torch.Tensor):
            return outputs.reshape(-1, outputs.shape[-1])
        return torch.cat([o.reshape(-1, o.shape[-1]) for o in outputs], dim=0)

    @torch.no_grad()
    def _run_forward_lite(
        self,
        block: nn.Module,
        inps,
        layer_kwargs,
        n_samples: Optional[int] = None,
    ):
        """Run block forward on at most *n_samples* samples, one at a time.

        Processing one sample per call (batch=1) avoids any batch-dimension
        mismatch in MRoPE position_ids or attention_mask, regardless of how
        the upstream model prepared those tensors.

        Returns:
          - Tensor [n_use, seq, h] for text mode.
          - List[Tensor] for multimodal mode.
        """
        device = next(block.parameters()).device
        n_use = n_samples or self.n_eval_samples

        if isinstance(inps, list):
            n_use = min(n_use, len(inps))
            kw_iter = (
                layer_kwargs[:n_use]
                if isinstance(layer_kwargs, list)
                else [layer_kwargs] * n_use
            )
            return [
                block(inp.to(device), **_to_device(kw, device))[0]
                for inp, kw in zip(inps[:n_use], kw_iter)
            ]

        # Text mode: one sample at a time
        n_use = min(n_use, inps.shape[0])
        batch_size = inps.shape[0]
        outputs = []
        for i in range(n_use):
            inp_i = inps[i: i + 1].to(device)
            kw_i = _slice_kw_for_sample(layer_kwargs, i, batch_size)
            # Disable KV cache to avoid shape mismatch from StaticCache/HybridCache
            kw_i["past_key_values"] = None
            kw_i["use_cache"] = False
            out = block(inp_i, **_to_device(kw_i, device))[0]
            outputs.append(out.cpu())  # keep on CPU to save GPU memory
        return torch.cat(outputs, dim=0)

    def _build_flat_vision_mask(
        self,
        vision_masks: Optional[List[Optional[torch.Tensor]]],
        n_samples_used: int,
        inps,
    ) -> Optional[torch.Tensor]:
        """Concatenate per-sample vision masks for the first n_samples_used."""
        if vision_masks is None:
            return None

        parts: List[torch.Tensor] = []
        samples = inps if isinstance(inps, list) else list(inps.unbind(0))

        for i, inp in enumerate(samples[:n_samples_used]):
            n_tok = inp.reshape(-1, inp.shape[-1]).shape[0]
            m = vision_masks[i] if i < len(vision_masks) else None
            if m is None:
                parts.append(torch.zeros(n_tok, dtype=torch.bool))
            else:
                m = m[:n_tok] if m.shape[0] > n_tok else torch.cat(
                    [m, m.new_zeros(n_tok - m.shape[0])]
                )
                parts.append(m)

        if not parts:
            return None
        flat = torch.cat(parts, dim=0)
        return flat if flat.any() else None

    @torch.no_grad()
    def _evaluate_candidate(
        self,
        block: nn.Module,
        lite_feat: Dict[str, torch.Tensor],
        inps,
        layer_kwargs,
        config,
        per_layer_sparsity: Dict[str, float],
        Y_orig_flat: torch.Tensor,
        H: torch.Tensor,
        vision_mask_flat: Optional[torch.Tensor],
    ) -> float:
        device = next(block.parameters()).device
        block_copy = copy.deepcopy(block).to(device)
        try:
            for pass_ in self.passes:
                if getattr(pass_, "_supports_per_layer", False):
                    pass_.apply(block_copy, lite_feat, self.adapter, config,
                                per_layer_sparsity=per_layer_sparsity)
                else:
                    pass_.apply(block_copy, lite_feat, self.adapter, config)

            Y_hat = self._run_forward_lite(block_copy, inps, layer_kwargs)
            Y_hat_flat = self._flatten(Y_hat).to(Y_orig_flat.device)
            return _hessian_block_error(
                Y_orig_flat, Y_hat_flat, H, vision_mask_flat, self.gamma
            )
        except Exception as e:
            logger.warning(f"  Candidate evaluation failed: {e}")
            return float("inf")
        finally:
            del block_copy
            gc.collect()
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_and_apply(
        self,
        block: nn.Module,
        input_feat: Dict[str, torch.Tensor],
        inps,
        layer_kwargs,
        config,
        Y_orig,                                          # unused (see note)
        vision_masks: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> None:
        """Search for the best per-layer sparsity config and apply it in-place.

        Note: *Y_orig* is accepted for API compatibility but not used —
        Y_orig_lite is computed internally from a small eval subset so that the
        reference output matches the *unmodified* original block exactly.

        Args:
            block:         Transformer block (already on device).
            input_feat:    Per-layer activations from collect_block_input_feat_and_output.
                           May be large; will be partially off-loaded to CPU.
            inps:          Current block inputs (text: Tensor, MM: List[Tensor]).
            layer_kwargs:  Attention kwargs.
            config:        CompressConfig.
            Y_orig:        Ignored (kept for interface compatibility with pipeline).
            vision_masks:  Optional per-sample vision token masks.
        """
        # Fast path: no pruning search needed
        if config.sparsity_ratio == 0.0 and config.prune_n == 0:
            for pass_ in self.passes:
                pass_.apply(block, input_feat, self.adapter, config)
            return

        named_linears = self.adapter.get_named_linears(block)
        layer_names   = list(named_linears.keys())
        layer_params  = {n: l.weight.numel() for n, l in named_linears.items()}

        # ---- Step 1: cheap statistics from the full input_feat ----
        # trace_H computed in fp16 chunks — no large fp32 copy needed
        trace_H = _compute_layer_trace_H(input_feat)

        # ---- Step 2: lite_feat — subsample for candidate evaluation ----
        lite_feat = _subsample_feat(input_feat, max_tokens=self.max_feat_tokens)

        # Off-load the large original input_feat tensors to CPU to free GPU RAM.
        # The passes will receive lite_feat; act_scales / JSQ metrics computed
        # on lite_feat are nearly identical to full-feat for relative ranking.
        _offload_feat(input_feat)
        gc.collect()
        torch.cuda.empty_cache()

        # ---- Step 3: Y_orig_lite for error computation ----
        # Run the ORIGINAL (unmodified) block one-sample-at-a-time on the
        # first n_eval_samples to get a reference output without any batch-
        # dimension issues in MRoPE position_ids or attention_mask.
        n_use = min(
            self.n_eval_samples,
            inps.shape[0] if isinstance(inps, torch.Tensor) else len(inps),
        )
        with torch.no_grad():
            Y_orig_lite = self._run_forward_lite(block, inps, layer_kwargs)

        # Text mode: outputs are on CPU (moved in _run_forward_lite).
        # Multimodal mode: outputs remain on GPU.
        Y_orig_flat = self._flatten(Y_orig_lite)
        # Fisher proxy computed in FP32: hidden states can reach ~100-1000,
        # and FP16 max is 65504, so Y²  overflows for deeper blocks.
        H = Y_orig_flat.float().pow(2)                # [tokens, hidden], FP32

        # Vision mask aligned to the n_use samples used
        vision_mask_flat = self._build_flat_vision_mask(vision_masks, n_use, inps)

        # ---- Step 4: generate candidates ----
        candidates = _generate_candidates(
            layer_names=layer_names,
            s_target=config.sparsity_ratio,
            layer_params=layer_params,
            trace_H=trace_H,
            n_candidates=self.n_search_candidates,
        )
        logger.info(f"  Block search: {len(candidates)} candidates "
                    f"(eval on {n_use} samples / {Y_orig_flat.shape[0]} tokens)")

        # ---- Step 5: evaluate candidates ----
        best_err = float("inf")
        best_candidate = candidates[0]

        for idx, cand in enumerate(candidates):
            err = self._evaluate_candidate(
                block, lite_feat, inps, layer_kwargs, config,
                cand, Y_orig_flat, H, vision_mask_flat,
            )
            logger.debug(f"  Candidate {idx}: err={err:.6e}")
            if err < best_err:
                best_err = err
                best_candidate = cand

        logger.info(f"  Best err={best_err:.6e}: {best_candidate}")

        # ---- Step 6: apply best config in-place ----
        for pass_ in self.passes:
            if getattr(pass_, "_supports_per_layer", False):
                pass_.apply(block, lite_feat, self.adapter, config,
                            per_layer_sparsity=best_candidate)
            else:
                pass_.apply(block, lite_feat, self.adapter, config)
