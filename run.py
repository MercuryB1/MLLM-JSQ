"""Core run() function: loads model → compresses → evaluates."""
import random

import numpy as np
import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from jsq.config import CompressConfig
from jsq.models.registry import get_adapter

# Register all adapters by importing them
import jsq.models.llama      # noqa: F401
import jsq.models.qwen2      # noqa: F401
import jsq.models.qwen2_vl   # noqa: F401

from jsq.compression.pipeline import CompressionPipeline
from jsq.compression.passes.prune import PruningPass
from jsq.compression.passes.smooth import SmoothingPass
from jsq.compression.passes.clip import ClippingPass
from jsq.compression.passes.quantize import QuantizationPass
from jsq.quant.save_load import save_compressed, load_compressed


_MULTIMODAL_TYPES = {"qwen2_vl", "qwen2_5_vl"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(config: CompressConfig):
    """Load model and processor/tokenizer.

    Returns (model, processor_or_tokenizer).
    For MLLMs, the second element is an AutoProcessor.
    For text-only LLMs, it is an AutoTokenizer.
    """
    model_cfg = AutoConfig.from_pretrained(config.model, trust_remote_code=True)
    model_cfg.use_cache = False

    model_type = getattr(model_cfg, "model_type", "")
    is_mllm = model_type in _MULTIMODAL_TYPES

    logger.info(f"Loading model: {config.model} (type={model_type}, mllm={is_mllm})")

    model_cls = AutoModelForImageTextToText if is_mllm else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        config.model,
        config=model_cfg,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    if is_mllm:
        processor = AutoProcessor.from_pretrained(config.model, trust_remote_code=True)
    else:
        processor = AutoTokenizer.from_pretrained(
            config.model, use_fast=False, trust_remote_code=True
        )

    return model, processor


def load_calib_data(config: CompressConfig, processor_or_tokenizer, model_type: str):
    """Load calibration data appropriate for the model type."""
    mm_datasets = {"coco_captions", "sharegpt4v", "gqa"}

    if config.calib_dataset in mm_datasets:
        if model_type not in _MULTIMODAL_TYPES:
            raise ValueError(
                f"Multimodal calibration dataset '{config.calib_dataset}' "
                f"requires an MLLM, but model_type='{model_type}'."
            )
        from jsq.calibration.multimodal import get_multimodal_calib_data
        return get_multimodal_calib_data(
            dataset=config.calib_dataset,
            processor=processor_or_tokenizer,
            model_type=model_type,
            n_samples=config.nsamples,
            seed=config.seed,
            data_dir=config.data_dir,
            batch_size=config.calib_batch_size,
        )
    else:
        from jsq.calibration.text import get_text_calib_data
        return get_text_calib_data(
            dataset=config.calib_dataset,
            tokenizer=processor_or_tokenizer,
            n_samples=config.nsamples,
            seq_len=config.seqlen,
            seed=config.seed,
        )


def build_passes(config: CompressConfig):
    """Build the ordered list of compression passes."""
    if config.no_compress:
        return []

    passes = []

    if config.pruning_method != "none" and config.sparsity_ratio > 0.0:
        passes.append(PruningPass())

    passes.append(SmoothingPass())
    passes.append(ClippingPass())
    passes.append(QuantizationPass())

    return passes


def check_sparsity(model, adapter) -> float:
    """Compute overall weight sparsity across all LLM blocks."""
    blocks = adapter.get_llm_blocks(model)
    total = zero = 0
    for block in blocks:
        for name, m in block.named_modules():
            if hasattr(m, "weight") and m.weight is not None:
                w = m.weight.data
                zero += (w == 0).sum().item()
                total += w.numel()
    sparsity = zero / total if total > 0 else 0.0
    logger.info(f"Overall LLM sparsity: {sparsity:.4f}")
    return sparsity


def run_eval(config: CompressConfig) -> None:
    """Load a saved compressed model and evaluate it with lmms-eval."""
    seed_everything(config.seed)

    if not config.save_dir:
        raise ValueError("--save_dir must point to a compressed model directory for --eval_only")

    logger.info(f"Loading compressed model from {config.save_dir} for evaluation")
    model, processor = load_compressed(config.save_dir)
    model_type = getattr(model.config, "model_type", "")

    model.cuda()

    if config.eval_ppl:
        tokenizer = processor if model_type not in _MULTIMODAL_TYPES else \
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        from jsq.eval.ppl import eval_ppl
        eval_ppl(model, tokenizer, seq_len=config.seqlen)

    if config.tasks:
        if model_type in _MULTIMODAL_TYPES:
            from jsq.eval.lmms_eval import run_lmms_eval
            run_lmms_eval(
                model=model,
                processor=processor,
                tasks=config.tasks,
                num_fewshot=config.num_fewshot,
                batch_size=config.batch_size,
                limit=None if config.limit == -1 else config.limit,
            )
        else:
            logger.warning("lmms-eval tasks specified but model is text-only.")


def run(config: CompressConfig) -> None:
    seed_everything(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, processor = load_model(config)
    model_type = getattr(model.config, "model_type", "")
    adapter = get_adapter(model)

    passes = build_passes(config)
    if passes:
        calib_data = load_calib_data(config, processor, model_type)
    else:
        calib_data = None
        logger.info("No compression passes enabled — skipping calibration data loading.")

    if passes:
        pipeline = CompressionPipeline(passes=passes, adapter=adapter)
        pipeline.run(model, calib_data, config, device)

    check_sparsity(model, adapter)

    if config.eval_ppl:
        tokenizer = processor if model_type not in _MULTIMODAL_TYPES else \
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        model.cuda()
        from jsq.eval.ppl import eval_ppl
        eval_ppl(model, tokenizer, seq_len=config.seqlen)

    if config.tasks:
        if model_type in _MULTIMODAL_TYPES:
            from jsq.eval.lmms_eval import run_lmms_eval
            run_lmms_eval(
                model=model,
                processor=processor,
                tasks=config.tasks,
                num_fewshot=config.num_fewshot,
                batch_size=config.batch_size,
                limit=None if config.limit == -1 else config.limit,
            )
        else:
            logger.warning("lmms-eval tasks specified but model is text-only.")

    if config.save_dir:
        save_compressed(model, processor, config.save_dir)
