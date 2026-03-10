"""QuantLinear: a fake-quantized linear layer."""
from functools import partial

import torch
import torch.nn as nn

from .ops import (
    quantize_activation_per_tensor_absmax,
    quantize_activation_per_token_absmax,
    quantize_weight_per_channel_absmax,
    quantize_weight_per_tensor_absmax,
)


class QuantLinear(nn.Module):
    """Drop-in replacement for nn.Linear that simulates W+A quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        act_quant: str = "per_token",
        quantize_output: bool = False,
        a_bits: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.zeros(out_features, in_features, dtype=torch.float16),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(1, out_features, dtype=torch.float16),
            )
        else:
            self.register_buffer("bias", None)

        # Activation quantizer
        self.a_bits = a_bits
        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, a_bits=a_bits)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, a_bits=a_bits)
        else:
            raise ValueError(f"Unknown act_quant: {act_quant}")

        # Optional output quantizer (for simulating BMM quantization)
        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = None
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_x = self.act_quant(x).to(self.weight.dtype)
        y = torch.nn.functional.linear(q_x, self.weight, self.bias)
        return self.output_quant(y)

    @staticmethod
    def from_float(
        module: nn.Linear,
        weight_quant: str = "per_channel",
        w_bits: int = 8,
        act_quant: str = "per_token",
        a_bits: int = 8,
        quantize_output: bool = False,
    ) -> "QuantLinear":
        assert isinstance(module, nn.Linear), f"Expected nn.Linear, got {type(module)}"

        ql = QuantLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            a_bits=a_bits,
        )

        w = module.weight.data.clone().float()
        if weight_quant == "per_channel":
            ql.weight = quantize_weight_per_channel_absmax(w, w_bits=w_bits).half()
        elif weight_quant == "per_tensor":
            ql.weight = quantize_weight_per_tensor_absmax(w, w_bits=w_bits).half()
        else:
            raise ValueError(f"Unknown weight_quant: {weight_quant}")

        ql.weight_quant_name = weight_quant
        ql.w_bits = w_bits

        if module.bias is not None:
            ql.bias = module.bias.data.clone().half().view(1, -1)

        return ql

    def __repr__(self) -> str:
        return (
            f"QuantLinear({self.in_features}, {self.out_features}, "
            f"bias={self.bias is not None}, "
            f"w_quant={self.weight_quant_name}, w_bits={self.w_bits}, "
            f"a_quant={self.act_quant_name}, a_bits={self.a_bits}, "
            f"out_quant={self.output_quant_name})"
        )
