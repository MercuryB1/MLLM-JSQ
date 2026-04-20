"""CompressionPipeline: orchestrates per-block compression.

Two modes:
  - search_method="none": direct uniform compression (no block search overhead).
  - search_method="candidate"/etc: BlockSearcher finds per-layer sparsity allocation.
"""
import gc
from typing import List, Optional

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from .block_search import BlockSearcher
from .block_pi import estimate_block_pi
from .block_vl_allocator import (
    allocate_from_scores,
    allocate_per_block_sparsity,
    summarize_allocation,
)
from .collector import (
    collect_block_input_feat,
    collect_block_input_feat_and_output,
    collect_block_pruning_damage,
    collect_block_sensitivity,
    collect_first_layer_inputs,
    run_block,
)
from .passes.base import CompressionPass


class CompressionPipeline:
    """Chains a list of CompressionPass objects and runs them block-by-block.

    When *search_method* != "none" and sparsity > 0, a BlockSearcher is used to
    find the optimal per-layer sparsity allocation (MA-JSQ).  Otherwise the
    passes are applied directly with uniform sparsity.

    Usage:
        pipeline = CompressionPipeline(
            passes=[PruningPass(), SmoothingPass(), ClippingPass(), QuantizationPass()],
            adapter=get_adapter(model),
        )
        pipeline.run(model, calib_samples, config, device, vision_masks=masks)
    """

    def __init__(self, passes: List[CompressionPass], adapter):
        self.passes = passes
        self.adapter = adapter

    def _use_search(self, config) -> bool:
        """Whether to use block-level search."""
        return (
            config.search_method != "none"
            and config.sparsity_ratio > 0.0
        )

    @staticmethod
    def _build_flat_vision_mask(
        vision_masks: Optional[List],
        inps,
    ) -> Optional[torch.Tensor]:
        """Concatenate per-sample vision masks into a flat [total_tokens] bool tensor.

        Returns None if vision_masks is not available or all-None.
        """
        if vision_masks is None:
            return None
        if not isinstance(inps, list):
            return None

        parts: List[torch.Tensor] = []
        for idx, inp in enumerate(inps):
            n_tok = inp.reshape(-1, inp.shape[-1]).shape[0]
            m = vision_masks[idx] if idx < len(vision_masks) else None
            if m is None:
                parts.append(torch.zeros(n_tok, dtype=torch.bool))
            else:
                if m.shape[0] != n_tok:
                    parts.append(torch.zeros(n_tok, dtype=torch.bool))
                else:
                    parts.append(m.bool())
        flat = torch.cat(parts, dim=0)
        return flat if flat.any() else None

    @torch.no_grad()
    def run(
        self,
        model: nn.Module,
        calib_samples,
        config,
        device: torch.device,
        vision_masks: Optional[List] = None,
    ) -> None:
        blocks = self.adapter.get_llm_blocks(model)
        use_search = self._use_search(config)
        logger.info(
            f"Starting compression: {len(blocks)} blocks, "
            f"{len(self.passes)} passes per block, "
            f"search={'on (' + config.search_method + ')' if use_search else 'off'}."
        )

        searcher = None
        if use_search:
            searcher = BlockSearcher(
                passes=self.passes,
                adapter=self.adapter,
                gamma=config.gamma,
                n_search_candidates=config.n_search_candidates,
            )

        logger.info("Collecting calibration inputs for block[0]...")
        inps, layer_kwargs = collect_first_layer_inputs(
            model, calib_samples, blocks, self.adapter, device
        )
        if isinstance(inps, torch.Tensor):
            logger.info(f"Captured inputs: shape={inps.shape}")
        else:
            logger.info(f"Captured inputs: {len(inps)} multimodal samples")

        # Optional: per-block sparsity allocation (Option E / damage).
        per_block_sparsity: Optional[List[float]] = None
        alloc_method = getattr(config, "block_alloc_method", "uniform")
        if (
            alloc_method != "uniform"
            and config.sparsity_ratio > 0.0
            and not use_search
        ):
            if alloc_method == "damage":
                logger.info("Running trial-pruning damage pre-pass...")
                # Find the PruningPass from self.passes
                pruning_pass = None
                for p in self.passes:
                    if hasattr(p, "_supports_per_layer"):
                        pruning_pass = p
                        break
                if pruning_pass is None:
                    raise RuntimeError("No PruningPass found for damage estimation")
                use_seq = getattr(config, "block_alloc_seq", False)
                logger.info(f"  sequential={use_seq}")
                damage_scores = collect_block_pruning_damage(
                    blocks, inps, layer_kwargs,
                    pruning_pass=pruning_pass,
                    adapter=self.adapter,
                    config=config,
                    vision_masks=vision_masks,
                    device=device,
                    sequential=use_seq,
                )
                per_block_sparsity = allocate_from_scores(
                    damage_scores,
                    target=config.sparsity_ratio,
                    alpha=config.block_alloc_alpha,
                    s_min=config.block_alloc_s_min,
                    s_max=config.block_alloc_s_max,
                    invert=getattr(config, "block_alloc_invert", False),
                    log_transform=getattr(config, "block_alloc_log", False),
                )
                logger.info("Per-block sparsity:\n" + summarize_allocation(per_block_sparsity, damage_scores))
                for li, s_l in enumerate(per_block_sparsity):
                    logger.info(
                        f"  block {li}: s={s_l:.3f} damage={damage_scores[li]:.6f}"
                    )
            else:
                logger.info(
                    f"Running block sensitivity pre-pass (method={alloc_method})..."
                )
                bi_t, bi_v = collect_block_sensitivity(
                    blocks, inps, layer_kwargs, vision_masks=vision_masks, device=device,
                )
                per_block_sparsity = allocate_per_block_sparsity(
                    bi_t, bi_v,
                    pi_t=config.pi_t,
                    target=config.sparsity_ratio,
                    alpha=config.block_alloc_alpha,
                    s_min=config.block_alloc_s_min,
                    s_max=config.block_alloc_s_max,
                    method=alloc_method,
                    invert=getattr(config, "block_alloc_invert", False),
                )
                logger.info("Per-block sparsity:\n" + summarize_allocation(per_block_sparsity))
                for li, s_l in enumerate(per_block_sparsity):
                    logger.info(
                        f"  block {li}: s={s_l:.3f} BI_t={bi_t[li]:.4f} BI_v={bi_v[li]:.4f}"
                    )

        for i, block in enumerate(tqdm(blocks, desc="Compressing blocks")):
            block.to(device)

            if use_search:
                # Search mode: collect feat + Y_orig together, then search
                input_feat, next_inps, layer_kwargs = collect_block_input_feat_and_output(
                    block, inps, layer_kwargs
                )
                flat_vmask = self._build_flat_vision_mask(vision_masks, inps)
                block_pi_t, block_pi_stats = estimate_block_pi(
                    block, input_feat, self.adapter, config, vision_mask=flat_vmask
                )
                if block_pi_t is not None:
                    logger.info(
                        f"Block {i}: block_pi_t={block_pi_t:.3f} "
                        f"dominance={block_pi_stats.get('dominance', 0.0):.3f} "
                        f"conflict={block_pi_stats.get('conflict', 0.0):.3f} "
                        f"iou={block_pi_stats.get('mask_iou', 1.0):.3f}"
                    )
                logger.info(f"Block {i}: running block search ({config.search_method})")
                searcher.search_and_apply(
                    block=block,
                    input_feat=input_feat,
                    inps=inps,
                    layer_kwargs=layer_kwargs,
                    config=config,
                    Y_orig=next_inps,
                    vision_masks=vision_masks,
                    block_pi_t=block_pi_t,
                )
            else:
                # Direct mode: collect feat, apply passes, then forward
                input_feat = collect_block_input_feat(block, inps, layer_kwargs)
                # Build flat vision mask for modality-aware pruning (v4)
                flat_vmask = self._build_flat_vision_mask(vision_masks, inps)
                block_pi_t, block_pi_stats = estimate_block_pi(
                    block, input_feat, self.adapter, config, vision_mask=flat_vmask
                )
                if block_pi_t is not None:
                    logger.info(
                        f"Block {i}: block_pi_t={block_pi_t:.3f} "
                        f"dominance={block_pi_stats.get('dominance', 0.0):.3f} "
                        f"conflict={block_pi_stats.get('conflict', 0.0):.3f} "
                        f"iou={block_pi_stats.get('mask_iou', 1.0):.3f}"
                    )
                # Optional per-block sparsity override (Option E)
                block_sparsity_dict: Optional[dict] = None
                if per_block_sparsity is not None:
                    s_l = per_block_sparsity[i]
                    linears = self.adapter.get_named_linears(block)
                    block_sparsity_dict = {name: s_l for name in linears}
                for pass_ in self.passes:
                    kw = {}
                    if hasattr(pass_, "_supports_per_layer"):
                        kw["vision_mask"] = flat_vmask
                        if block_pi_t is not None:
                            kw["block_pi_t"] = block_pi_t
                        if block_sparsity_dict is not None:
                            kw["per_layer_sparsity"] = block_sparsity_dict
                    pass_.apply(block, input_feat, self.adapter, config, **kw)
                next_inps, layer_kwargs = run_block(block, inps, layer_kwargs)

            inps = next_inps

            del input_feat
            gc.collect()
            torch.cuda.empty_cache()

            block.cpu()
            gc.collect()
            torch.cuda.empty_cache()

        logger.info("Compression complete.")
