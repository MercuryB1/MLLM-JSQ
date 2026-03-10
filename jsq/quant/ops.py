"""Pure quantization functions with no model-type dependencies."""
import torch


@torch.no_grad()
def quantize_weight_per_channel_absmax(w: torch.Tensor, w_bits: int = 8) -> torch.Tensor:
    """Per output-channel absmax weight quantization (in-place simulation)."""
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (w_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w: torch.Tensor, w_bits: int = 8) -> torch.Tensor:
    """Per-tensor absmax weight quantization (in-place simulation)."""
    scales = w.abs().max()
    q_max = 2 ** (w_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(x: torch.Tensor, a_bits: int = 8) -> torch.Tensor:
    """Per-token absmax activation quantization (in-place simulation)."""
    scales = x.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (a_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    x.div_(scales).round_().mul_(scales)
    return x


@torch.no_grad()
def quantize_activation_per_tensor_absmax(x: torch.Tensor, a_bits: int = 8) -> torch.Tensor:
    """Per-tensor absmax activation quantization (in-place simulation)."""
    scales = x.abs().max()
    q_max = 2 ** (a_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    x.div_(scales).round_().mul_(scales)
    return x
