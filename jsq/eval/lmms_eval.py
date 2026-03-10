"""lmms-eval integration for evaluating compressed MLLMs.

This module wraps a compressed MLLM so it can be evaluated by the
lmms-eval framework (https://github.com/EvolvingLMMs-Lab/lmms-eval).

Usage:
    results = run_lmms_eval(model, processor, config)
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
    """Run lmms-eval benchmarks on a compressed MLLM.

    Args:
        model: The compressed multimodal model.
        processor: The model's processor (handles image+text preprocessing).
        tasks: Comma-separated lmms-eval task names.
              E.g. "mmbench_en_dev,seedbench,mme"
        num_fewshot: Number of few-shot examples (default 0).
        batch_size: Batch size for evaluation.
        limit: If set, limit evaluation to this many samples per task.

    Returns:
        Dict with evaluation results from lmms-eval.

    Supported tasks (non-exhaustive):
        mmbench_en_dev   - MMBench English Dev
        seedbench        - SEED-Bench
        mme              - MME benchmark
        gqa              - GQA
        textvqa_val      - TextVQA
        vqav2_val        - VQAv2
        chartqa          - ChartQA
        docvqa_val       - DocVQA
    """
    try:
        from lmms_eval import evaluator
        from lmms_eval.models import get_model
    except ImportError:
        raise ImportError(
            "lmms-eval is not installed. Install it with:\n"
            "  pip install lmms-eval\n"
            "or from source:\n"
            "  git clone https://github.com/EvolvingLMMs-Lab/lmms-eval && pip install -e lmms-eval"
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


def _build_wrapper(model: nn.Module, processor, batch_size: int):
    """Build the appropriate lmms-eval model wrapper based on model type."""
    model_type = getattr(model.config, "model_type", "")

    if model_type == "qwen2_vl":
        return _Qwen2VLWrapper(model, processor, batch_size)
    else:
        raise NotImplementedError(
            f"lmms-eval wrapper not yet implemented for model_type='{model_type}'. "
            f"Currently supported: qwen2_vl"
        )


def _log_results(results: Dict) -> None:
    """Pretty-print lmms-eval results."""
    if "results" not in results:
        logger.info(str(results))
        return
    for task, metrics in results["results"].items():
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                               if isinstance(v, float))
        logger.info(f"  {task}: {metric_str}")


# ---------------------------------------------------------------------------
# Model-specific wrappers
# ---------------------------------------------------------------------------

class _Qwen2VLWrapper:
    """Wraps a compressed Qwen2-VL model for lmms-eval evaluation.

    lmms-eval already ships a Qwen2-VL implementation. This wrapper delegates
    to the same inference logic but uses the compressed model weights.
    """

    def __init__(self, model: nn.Module, processor, batch_size: int):
        try:
            from lmms_eval.models.qwen2_vl import Qwen2_VL
        except ImportError:
            raise ImportError(
                "lmms-eval Qwen2-VL model class not found. "
                "Make sure you have lmms-eval >= 0.3.0 installed."
            )

        self._lmms_model = Qwen2_VL(
            pretrained=model.config._name_or_path,
            batch_size=batch_size,
        )
        # Swap in the compressed model weights
        self._lmms_model._model = model
        self._lmms_model._processor = processor

    def __getattr__(self, name):
        return getattr(self._lmms_model, name)

    def generate_until(self, requests):
        return self._lmms_model.generate_until(requests)

    def loglikelihood(self, requests):
        return self._lmms_model.loglikelihood(requests)
