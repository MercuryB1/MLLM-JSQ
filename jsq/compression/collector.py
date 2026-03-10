"""Utilities for collecting layer inputs during calibration."""
import functools
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn


class Catcher(nn.Module):
    """Wraps the first block to intercept its inputs.

    Each forward call appends (inp, kwargs) to self.captured, then raises
    ValueError("catcher_exit") to abort the model forward early.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.captured: List[Tuple[torch.Tensor, Dict]] = []

    def forward(self, inp, **kwargs):
        kw = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        kw["use_cache"] = False
        self.captured.append((inp.detach().cpu(), kw))
        raise ValueError("catcher_exit")


def _to_device(kwargs: Dict, device) -> Dict:
    """Move all Tensor values in a kwargs dict to device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in kwargs.items()
    }


@torch.no_grad()
def collect_first_layer_inputs(
    model: nn.Module,
    calib_samples,
    blocks: List[nn.Module],
    adapter,
    device: torch.device,
) -> Tuple:
    """Run calibration forward(s) to capture the inputs arriving at blocks[0].

    Handles two modes:

    Text mode (calib_samples is torch.Tensor):
        One batched forward pass captures inps of shape [n_samples, seq_len, hidden].
        Returns (Tensor[n, s, h], Dict).

    Multimodal mode (calib_samples is List[Dict]):
        One forward pass per sample; each fires the Catcher once.
        Returns (List[Tensor[1, s_i, h]], List[Dict]).

    The vision encoder (ViT) is temporarily moved to device for multimodal
    calibration and moved back to CPU afterwards.
    """
    is_multimodal = not isinstance(calib_samples, torch.Tensor)

    # Wrap blocks[0] in Catcher
    catcher = Catcher(blocks[0])
    blocks[0] = catcher
    blocks[0].to(device)
    adapter.move_llm_embed(model, device)

    if is_multimodal:
        # Move vision encoder to device for the calibration forward passes,
        # then return it to CPU to free memory for the compression loop.
        adapter.move_vision_encoder(model, device)

    if not is_multimodal:
        # Text: single batched forward
        try:
            adapter.run_forward_for_calibration(model, calib_samples.to(device))
        except ValueError as e:
            if "catcher_exit" not in str(e):
                raise
    else:
        # Multimodal: one forward per sample
        for sample in calib_samples:
            try:
                adapter.run_forward_for_calibration(model, sample)
            except ValueError as e:
                if "catcher_exit" not in str(e):
                    raise

    # Unwrap Catcher, restore device state
    blocks[0] = catcher.module
    adapter.move_llm_embed(model, "cpu")
    if is_multimodal:
        adapter.move_vision_encoder(model, "cpu")
    blocks[0] = blocks[0].cpu()
    torch.cuda.empty_cache()

    if not catcher.captured:
        raise RuntimeError(
            "Catcher did not capture any inputs. "
            "Check model architecture and adapter configuration."
        )

    if not is_multimodal:
        # Single capture → batched Tensor + shared kwargs
        inps, layer_kwargs = catcher.captured[0]
        return inps, layer_kwargs
    else:
        # Multiple captures → list of per-sample tensors + list of per-sample kwargs
        inps_list = [c[0] for c in catcher.captured]
        kwargs_list = [c[1] for c in catcher.captured]
        return inps_list, kwargs_list


@torch.no_grad()
def collect_block_input_feat(
    block: nn.Module,
    inps,
    layer_kwargs,
) -> Dict[str, torch.Tensor]:
    """Collect input activations for every Linear layer inside a block.

    Text mode   (inps: Tensor[n, s, h], layer_kwargs: Dict):
        One batched forward; features have shape [n, s, hidden_in].

    Multimodal mode (inps: List[Tensor[1, s_i, h]], layer_kwargs: List[Dict]):
        One forward per sample; features are flattened to [total_tokens, hidden_in]
        and the entry "__nsamples__" is set to the sample count.

    Returns:
        Dict mapping linear name -> feature Tensor.
        For multimodal, also contains "__nsamples__" (int key).
    """
    named_linears = {
        name: m for name, m in block.named_modules() if isinstance(m, nn.Linear)
    }

    feat: Dict[str, List[torch.Tensor]] = defaultdict(list)

    def _hook_batched(m, x, y, name):
        feat[name].append(x[0].detach().cpu())

    def _hook_flatten(m, x, y, name):
        # Flatten to 2D so variable-length sequences can be concatenated
        act = x[0].detach().cpu()
        feat[name].append(act.reshape(-1, act.shape[-1]))

    device = next(block.parameters()).device

    if isinstance(inps, torch.Tensor):
        # ---- Text mode: one batched forward ----
        handles = [
            mod.register_forward_hook(functools.partial(_hook_batched, name=name))
            for name, mod in named_linears.items()
        ]
        block(inps.to(device), **_to_device(layer_kwargs, device))
        for h in handles:
            h.remove()
        return {k: torch.cat(v, dim=0) for k, v in feat.items()}

    else:
        # ---- Multimodal mode: iterate per sample ----
        handles = [
            mod.register_forward_hook(functools.partial(_hook_flatten, name=name))
            for name, mod in named_linears.items()
        ]
        for inp, kw in zip(inps, layer_kwargs):
            block(inp.to(device), **_to_device(kw, device))
        for h in handles:
            h.remove()
        result: Dict = {k: torch.cat(v, dim=0) for k, v in feat.items()}
        result["__nsamples__"] = len(inps)
        return result


@torch.no_grad()
def run_block(
    block: nn.Module,
    inps,
    layer_kwargs,
) -> Tuple:
    """Run block forward and return (output_inps, layer_kwargs).

    Text mode:   returns (Tensor[n, s, h], Dict)
    Multimodal:  returns (List[Tensor[1, s_i, h]], List[Dict])
                 (layer_kwargs is unchanged across blocks)
    """
    device = next(block.parameters()).device

    if isinstance(inps, torch.Tensor):
        out = block(inps.to(device), **_to_device(layer_kwargs, device))[0]
        return out.cpu(), layer_kwargs
    else:
        outputs = [
            block(inp.to(device), **_to_device(kw, device))[0].cpu()
            for inp, kw in zip(inps, layer_kwargs)
        ]
        return outputs, layer_kwargs
