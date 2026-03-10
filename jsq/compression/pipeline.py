"""CompressionPipeline: orchestrates per-layer compression loop."""
import gc
from typing import List

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from .collector import (
    collect_block_input_feat,
    collect_first_layer_inputs,
    run_block,
)
from .passes.base import CompressionPass


class CompressionPipeline:
    """Chains a list of CompressionPass objects and runs them layer-by-layer.

    Usage:
        pipeline = CompressionPipeline(
            passes=[PruningPass(), SmoothingPass(), ClippingPass(), QuantizationPass()],
            adapter=get_adapter(model),
        )
        pipeline.run(model, calib_samples, config, device)
    """

    def __init__(self, passes: List[CompressionPass], adapter):
        self.passes = passes
        self.adapter = adapter

    @torch.no_grad()
    def run(self, model: nn.Module, calib_samples, config, device: torch.device) -> None:
        blocks = self.adapter.get_llm_blocks(model)
        logger.info(f"Starting compression: {len(blocks)} blocks, "
                    f"{len(self.passes)} passes per block.")

        # ------------------------------------------------------------------ #
        # Step 1: collect first-layer inputs via Catcher hook
        # ------------------------------------------------------------------ #
        logger.info("Collecting calibration inputs for block[0]...")
        inps, layer_kwargs = collect_first_layer_inputs(
            model, calib_samples, blocks, self.adapter, device
        )
        logger.info(f"Captured inputs: shape={inps.shape}")

        # ------------------------------------------------------------------ #
        # Step 2: per-block compression loop
        # ------------------------------------------------------------------ #
        for i, block in enumerate(tqdm(blocks, desc="Compressing blocks")):
            block.to(device)

            # Collect input activations for every Linear inside this block
            input_feat = collect_block_input_feat(block, inps, layer_kwargs)

            # Run all passes in order
            for pass_ in self.passes:
                pass_name = type(pass_).__name__
                logger.info(f"Block {i}: {pass_name}")
                pass_.apply(block, input_feat, self.adapter, config)

            # Update inps to this block's output (= next block's input)
            inps, layer_kwargs = run_block(block, inps, layer_kwargs)

            del input_feat
            block.cpu()
            gc.collect()
            torch.cuda.empty_cache()

        logger.info("Compression complete.")
