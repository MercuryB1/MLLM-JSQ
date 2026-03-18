"""lmms-eval integration for evaluating compressed MLLMs.

This module wraps a compressed MLLM so it can be evaluated by the
lmms-eval framework (https://github.com/EvolvingLMMs-Lab/lmms-eval).
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from loguru import logger


def run_lmms_eval(
    model: nn.Module,
    processor,
    tasks: str,
    num_fewshot: int = 0,
    batch_size: int = 1,
    limit: Optional[int] = None,
) -> Dict:
    """Run lmms-eval benchmarks on a compressed MLLM."""
    try:
        from lmms_eval import evaluator
    except ImportError:
        raise ImportError(
            "lmms-eval is not installed. Install it with:\n"
            "  pip install lmms-eval"
        )

    task_names = [t.strip() for t in tasks.split(",")]
    logger.info(f"Running lmms-eval on tasks: {task_names}")

    wrapper = _build_wrapper(model, processor, batch_size)

    results = evaluator.simple_evaluate(
        model=wrapper,
        tasks=task_names,
        num_fewshot=num_fewshot,
        limit=limit,
        log_samples=False,
    )

    _log_results(results)
    return results


_WRAPPER_CLASSES = {
    "qwen2_vl":   ("lmms_eval.models.simple.qwen2_vl",   "Qwen2_VL"),
    "qwen2_5_vl": ("lmms_eval.models.simple.qwen2_5_vl", "Qwen2_5_VL"),
    "qwen3_vl":   ("lmms_eval.models.simple.qwen3_vl",   "Qwen3_VL"),
}


def _build_wrapper(model: nn.Module, processor, batch_size: int):
    model_type = getattr(model.config, "model_type", "")
    if model_type not in _WRAPPER_CLASSES:
        raise NotImplementedError(
            f"lmms-eval wrapper not yet implemented for model_type='{model_type}'. "
            f"Currently supported: {', '.join(_WRAPPER_CLASSES)}"
        )
    module_path, class_name = _WRAPPER_CLASSES[model_type]
    return _make_wrapper(model, processor, batch_size, module_path, class_name)


def _make_wrapper(model: nn.Module, processor, batch_size: int,
                  module_path: str, class_name: str):
    """Build an lmms-eval wrapper that injects the already-loaded compressed model.

    Inherits all inference logic (generate_until, etc.) from the upstream class,
    but bypasses __init__'s weight loading to avoid OOM on 7B+ models.
    """
    import importlib
    from lmms_eval.api.model import lmms

    base_cls = getattr(importlib.import_module(module_path), class_name)

    class _Wrapper(base_cls):
        def __init__(self, model, processor, batch_size):
            lmms.__init__(self)  # skip base class weight loading
            device = next(model.parameters()).device
            self._model = model
            self.processor = processor
            self._tokenizer = (
                processor.tokenizer if hasattr(processor, "tokenizer") else processor
            )
            self._config = model.config
            self._max_length = 2048
            self.batch_size_per_gpu = int(batch_size)
            self.use_cache = True
            self.max_pixels = 1605632
            self.min_pixels = 256 * 28 * 28
            self.max_num_frames = 32
            self.system_prompt = "You are a helpful assistant."
            self.interleave_visuals = False
            self.reasoning_prompt = None
            self._device = device
            self.device_map = str(device)
            self._rank = 0
            self._world_size = 1

    return _Wrapper(model, processor, batch_size)


def _log_results(results: Dict) -> None:
    try:
        from lmms_eval.utils import make_table
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))
    except Exception:
        pass
    if "results" not in results:
        logger.info(str(results))
        return
    for task, metrics in results["results"].items():
        metric_str = ", ".join(
            f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))
        )
        logger.info(f"  {task}: {metric_str}")
