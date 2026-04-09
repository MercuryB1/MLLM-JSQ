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
from .collector import (
    collect_block_input_feat,
    collect_block_input_feat_and_output,
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

        for i, block in enumerate(tqdm(blocks, desc="Compressing blocks")):
            block.to(device)

            if use_search:
                # Search mode: collect feat + Y_orig together, then search
                input_feat, next_inps, layer_kwargs = collect_block_input_feat_and_output(
                    block, inps, layer_kwargs
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
                )
            else:
                # Direct mode: collect feat, apply passes, then forward
                input_feat = collect_block_input_feat(block, inps, layer_kwargs)
                for pass_ in self.passes:
                    pass_.apply(block, input_feat, self.adapter, config)
                next_inps, layer_kwargs = run_block(block, inps, layer_kwargs)

            inps = next_inps

            del input_feat
            gc.collect()
            torch.cuda.empty_cache()

            block.cpu()
            gc.collect()
            torch.cuda.empty_cache()

        logger.info("Compression complete.")
