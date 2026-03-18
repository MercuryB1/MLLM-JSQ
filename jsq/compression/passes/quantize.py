"""Quantization pass: replace nn.Linear with QuantLinear inside a block."""
from typing import Dict

import torch
import torch.nn as nn
from loguru import logger

from .base import CompressionPass
from jsq.quant.linear import QuantLinear


def _set_module_by_name(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    """Set a nested attribute by dotted name, e.g. 'self_attn.q_proj'."""
    parts = name.split(".")
    obj = parent
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], new_module)


class QuantizationPass(CompressionPass):
    """Replace every Linear in the block with QuantLinear.

    Retrieves the list of target linears from adapter.get_named_linears(),
    so this pass never needs to know which specific Attention/MLP class is used.
    """

    def apply(self, block, input_feat: Dict[str, torch.Tensor], adapter, config) -> None:
        named_linears = adapter.get_named_linears(block)
        bmm_names = {"q_proj", "k_proj", "query", "key"}

        for name, linear in named_linears.items():
            if not isinstance(linear, nn.Linear):
                # Already quantized in a previous pass
                continue

            is_bmm = any(tok in name for tok in bmm_names)
            quantize_output = is_bmm and config.quantize_bmm_input

            try:
                ql = QuantLinear.from_float(
                    linear,
                    weight_quant=config.weight_quant,
                    w_bits=config.w_bits,
                    act_quant=config.act_quant,
                    a_bits=config.a_bits,
                    quantize_output=quantize_output,
                )
                _set_module_by_name(block, name, ql)
            except Exception as e:
                logger.warning(f"QuantizationPass: failed on '{name}': {e}")
