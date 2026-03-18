"""Save and load helpers for compressed (pruned + fake-quantized) models.

Save flow:
  QuantLinear → nn.Linear (weights already in quantized fp16)
  + save_pretrained (standard HF format)
  + quant_config.json (records which layers were quantized and how)

Load flow:
  from_pretrained (loads nn.Linear with quantized fp16 weights)
  + quant_config.json → rebuild QuantLinear wrappers
"""
import json
import os
from typing import Dict

import torch
import torch.nn as nn
from loguru import logger

from .linear import QuantLinear



def _collect_quant_config(model: nn.Module) -> Dict:
    """Walk the model and record every QuantLinear's configuration."""
    config = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            config[name] = {
                "w_bits": module.w_bits,
                "weight_quant": module.weight_quant_name,
                "act_quant": module.act_quant_name,
                "a_bits": module.a_bits,
                "quantize_output": module.output_quant_name is not None,
            }
    return config


def _quant_to_linear(module: QuantLinear) -> nn.Linear:
    """Convert QuantLinear → nn.Linear, preserving the quantized fp16 weights."""
    has_bias = module.bias is not None
    linear = nn.Linear(module.in_features, module.out_features, bias=has_bias,
                       dtype=module.weight.dtype, device=module.weight.device)
    linear.weight = nn.Parameter(module.weight.data.clone())
    if has_bias:
        # QuantLinear stores bias as [1, out]; nn.Linear wants [out]
        linear.bias = nn.Parameter(module.bias.data.clone().view(-1))
    return linear


def _set_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def save_compressed(model: nn.Module, processor, save_dir: str) -> None:
    """Save a compressed model to save_dir in HuggingFace format.

    QuantLinear layers are converted back to nn.Linear before calling
    save_pretrained, so the checkpoint is loadable by standard HF code.
    A quant_config.json is saved alongside to allow rebuilding the
    QuantLinear wrappers on reload.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Collect quant metadata before modifying the model
    quant_config = _collect_quant_config(model)
    logger.info(f"Saving {len(quant_config)} quantized layers' configs")

    # 2. Convert QuantLinear → nn.Linear in-place
    for name in quant_config:
        parts = name.split(".")
        module = model
        for part in parts:
            module = getattr(module, part)
        _set_module(model, name, _quant_to_linear(module))

    # 3. Standard HF save (weights + config + tokenizer)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    # 4. Save quant metadata
    quant_config_path = os.path.join(save_dir, "quant_config.json")
    with open(quant_config_path, "w") as f:
        json.dump(quant_config, f, indent=2)

    logger.info(f"Compressed model saved to {save_dir}")



def _restore_quant_linear(model: nn.Module, name: str, qc: Dict) -> None:
    """Replace the nn.Linear at `name` with a QuantLinear using saved config."""
    parts = name.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)

    if not isinstance(module, nn.Linear):
        logger.warning(f"Expected nn.Linear at '{name}', got {type(module)}; skipping")
        return

    ql = QuantLinear(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None,
        act_quant=qc["act_quant"],
        quantize_output=qc["quantize_output"],
        a_bits=qc["a_bits"],
    )
    ql.weight = module.weight.data.clone().half()
    ql.weight_quant_name = qc["weight_quant"]
    ql.w_bits = qc["w_bits"]
    if module.bias is not None:
        ql.bias = module.bias.data.clone().half().view(1, -1)

    _set_module(model, name, ql)


def load_compressed(load_dir: str, torch_dtype=torch.float16):
    """Load a compressed model saved by save_compressed().

    Returns (model, processor) with QuantLinear layers restored.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    _MULTIMODAL_TYPES = {"qwen2_vl"}

    quant_config_path = os.path.join(load_dir, "quant_config.json")
    if not os.path.exists(quant_config_path):
        raise FileNotFoundError(
            f"quant_config.json not found in {load_dir}. "
            "Was this model saved with save_compressed()?"
        )

    with open(quant_config_path) as f:
        quant_config = json.load(f)

    model_cfg = AutoConfig.from_pretrained(load_dir, trust_remote_code=True)
    model_cfg.use_cache = False
    model_type = getattr(model_cfg, "model_type", "")
    is_mllm = model_type in _MULTIMODAL_TYPES

    logger.info(f"Loading compressed model from {load_dir} (type={model_type})")

    model_cls = AutoModelForImageTextToText if is_mllm else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        load_dir,
        config=model_cfg,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    # Restore QuantLinear wrappers
    logger.info(f"Restoring {len(quant_config)} QuantLinear layers")
    for name, qc in quant_config.items():
        _restore_quant_linear(model, name, qc)

    if is_mllm:
        processor = AutoProcessor.from_pretrained(load_dir, trust_remote_code=True)
    else:
        processor = AutoTokenizer.from_pretrained(load_dir, use_fast=False, trust_remote_code=True)

    return model, processor
