"""CompressionPipeline: orchestrates per-layer compression loop."""
import gc
from typing import List

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from .collector import (
    collect_block_input_feat_and_output,
    collect_first_layer_inputs,
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

            # Collect features and next-block inputs from the ORIGINAL (uncompressed)
            # block in one pass.  Using original activations for calibration is standard
            # practice (GPTQ, AWQ, SmoothQuant) and avoids running the compressed block
            # for the forward pass (which triggers attention-mask shape mismatches).
            input_feat, inps, layer_kwargs = collect_block_input_feat_and_output(
                block, inps, layer_kwargs
            )

            for pass_ in self.passes:
                logger.info(f"Block {i}: {type(pass_).__name__}")
                pass_.apply(block, input_feat, self.adapter, config)

            del input_feat
            gc.collect()
            torch.cuda.empty_cache()

            block.cpu()
            gc.collect()
            torch.cuda.empty_cache()

        logger.info("Compression complete.")
