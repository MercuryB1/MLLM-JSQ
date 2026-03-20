"""CompressionPipeline: orchestrates per-block compression with MA-JSQ search."""
import gc
from typing import List, Optional

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from .block_search import BlockSearcher
from .collector import (
    collect_block_input_feat_and_output,
    collect_first_layer_inputs,
)
from .passes.base import CompressionPass


class CompressionPipeline:
    """Chains a list of CompressionPass objects and runs them block-by-block.

    When *sparsity_ratio* > 0, a BlockSearcher is used to find the optimal
    per-layer sparsity allocation within each block (MA-JSQ).  Otherwise the
    passes are applied directly without search overhead.

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
        logger.info(f"Starting compression: {len(blocks)} blocks, "
                    f"{len(self.passes)} passes per block.")

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

            # One forward pass on the ORIGINAL block collects input_feat and
            # produces Y_orig (= next block's input).  Using original activations
            # for calibration is standard (GPTQ, AWQ, SmoothQuant).
            input_feat, next_inps, layer_kwargs = collect_block_input_feat_and_output(
                block, inps, layer_kwargs
            )

            logger.info(f"Block {i}: running block search")
            searcher.search_and_apply(
                block=block,
                input_feat=input_feat,
                inps=inps,
                layer_kwargs=layer_kwargs,
                config=config,
                Y_orig=next_inps,
                vision_masks=vision_masks,
            )

            inps = next_inps  # next block's input = original block's output

            del input_feat
            gc.collect()
            torch.cuda.empty_cache()

            block.cpu()
            gc.collect()
            torch.cuda.empty_cache()

        logger.info("Compression complete.")
