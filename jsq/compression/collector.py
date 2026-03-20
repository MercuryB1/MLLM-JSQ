"""Utilities for collecting layer inputs during calibration."""
import functools
from collections import defaultdict
from typing import Dict, List, Tuple

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

    def __getattr__(self, name: str):
        # Proxy attribute access so model code (e.g. Qwen2VL accessing
        # decoder_layer.attention_type) still works.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def forward(self, inp, **kwargs):
        kw = {
            k: v.detach() if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        kw["use_cache"] = False
        self.captured.append((inp.detach(), kw))
        raise ValueError("catcher_exit")


def _to_device(kwargs: Dict, device) -> Dict:
    """Move all Tensor values in a kwargs dict to device."""
    result = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, tuple) and all(isinstance(t, torch.Tensor) for t in v):
            result[k] = tuple(t.to(device) for t in v)
        else:
            result[k] = v
    return result


def _slice_kw_for_sample(kwargs: Dict, idx: int, batch_size: int) -> Dict:
    """Extract kwargs for a single sample index from a batched kwargs dict.

    Handles:
      - torch.Tensor [batch, ...]        → [1, ...]
      - tuple of tensors (e.g. position_embeddings = (cos, sin))
        where each tensor has batch as dim 0 → tuple of [1, ...]
      - scalars / non-tensor values      → passed through unchanged
    """
    result: Dict = {}
    for k, v in kwargs.items():
        if isinstance(v, tuple) and v and all(isinstance(t, torch.Tensor) for t in v):
            sliced = []
            for t in v:
                if t.shape[0] == batch_size:
                    sliced.append(t[idx: idx + 1])
                elif t.dim() >= 2 and t.shape[1] == batch_size:
                    sliced.append(t[:, idx: idx + 1])
                else:
                    sliced.append(t)
            result[k] = tuple(sliced)
        elif not isinstance(v, torch.Tensor) or v.dim() == 0:
            result[k] = v
        elif v.shape[0] == batch_size:
            result[k] = v[idx: idx + 1]
        elif v.dim() >= 2 and v.shape[1] == batch_size:
            result[k] = v[:, idx: idx + 1]
        else:
            result[k] = v
    return result


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

    catcher = Catcher(blocks[0])
    blocks[0] = catcher
    blocks[0].to(device)
    adapter.move_llm_embed(model, device)

    if is_multimodal:
        adapter.move_vision_encoder(model, device)

    if not is_multimodal:
        try:
            adapter.run_forward_for_calibration(model, calib_samples.to(device))
        except ValueError as e:
            if "catcher_exit" not in str(e):
                raise
    else:
        for sample in calib_samples:
            try:
                adapter.run_forward_for_calibration(model, sample)
            except ValueError as e:
                if "catcher_exit" not in str(e):
                    raise

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
        inps, layer_kwargs = catcher.captured[0]
        return inps, layer_kwargs
    else:
        inps_list = [c[0] for c in catcher.captured]
        kwargs_list = [c[1] for c in catcher.captured]
        return inps_list, kwargs_list


@torch.no_grad()
def collect_block_input_feat_and_output(
    block: nn.Module,
    inps,
    layer_kwargs,
):
    """Collect input features and run the block in a single forward pass.

    Returns (input_feat, next_inps, next_layer_kwargs).
    """
    named_linears = {
        name: m for name, m in block.named_modules() if isinstance(m, nn.Linear)
    }
    feat: Dict[str, List[torch.Tensor]] = defaultdict(list)

    def _hook_batched(m, x, y, name):
        feat[name].append(x[0].detach())

    def _hook_flatten(m, x, y, name):
        act = x[0].detach()
        feat[name].append(act.reshape(-1, act.shape[-1]))

    device = next(block.parameters()).device

    if isinstance(inps, torch.Tensor):
        handles = [
            mod.register_forward_hook(functools.partial(_hook_batched, name=name))
            for name, mod in named_linears.items()
        ]
        batch_size = inps.shape[0]
        outputs = []
        for i in range(batch_size):
            inp_i = inps[i: i + 1].to(device)
            kw_i = _slice_kw_for_sample(layer_kwargs, i, batch_size)
            kw_i["past_key_values"] = None
            kw_i["use_cache"] = False
            out_i = block(inp_i, **_to_device(kw_i, device))[0]
            outputs.append(out_i.detach().cpu())
        for h in handles:
            h.remove()
        input_feat = {k: torch.cat(v, dim=0) for k, v in feat.items()}
        next_inps = torch.cat(outputs, dim=0)
        return input_feat, next_inps, layer_kwargs

    else:
        handles = [
            mod.register_forward_hook(functools.partial(_hook_flatten, name=name))
            for name, mod in named_linears.items()
        ]
        outputs = []
        for inp, kw in zip(inps, layer_kwargs):
            out = block(inp.to(device), **_to_device(kw, device))[0]
            outputs.append(out)
        for h in handles:
            h.remove()
        input_feat: Dict = {k: torch.cat(v, dim=0) for k, v in feat.items()}
        input_feat["__nsamples__"] = len(inps)
        return input_feat, outputs, layer_kwargs


@torch.no_grad()
def collect_block_input_feat(
    block: nn.Module,
    inps,
    layer_kwargs,
) -> Dict[str, torch.Tensor]:
    """Collect input activations for every Linear layer inside a block.

    Text mode (inps: Tensor[n, s, h]): one batched forward, features shape [n, s, hidden_in].
    Multimodal mode (inps: List[Tensor]): one forward per sample, features flattened to
    [total_tokens, hidden_in]; also sets "__nsamples__" in the returned dict.
    """
    named_linears = {
        name: m for name, m in block.named_modules() if isinstance(m, nn.Linear)
    }

    feat: Dict[str, List[torch.Tensor]] = defaultdict(list)

    def _hook_batched(m, x, y, name):
        feat[name].append(x[0].detach())

    def _hook_flatten(m, x, y, name):
        act = x[0].detach()
        feat[name].append(act.reshape(-1, act.shape[-1]))

    device = next(block.parameters()).device

    if isinstance(inps, torch.Tensor):
        handles = [
            mod.register_forward_hook(functools.partial(_hook_batched, name=name))
            for name, mod in named_linears.items()
        ]
        batch_size = inps.shape[0]
        for i in range(batch_size):
            inp_i = inps[i: i + 1].to(device)
            kw_i = _slice_kw_for_sample(layer_kwargs, i, batch_size)
            kw_i["past_key_values"] = None
            kw_i["use_cache"] = False
            block(inp_i, **_to_device(kw_i, device))
        for h in handles:
            h.remove()
        return {k: torch.cat(v, dim=0) for k, v in feat.items()}

    else:
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
        batch_size = inps.shape[0]
        outputs = []
        for i in range(batch_size):
            inp_i = inps[i: i + 1].to(device)
            kw_i = _slice_kw_for_sample(layer_kwargs, i, batch_size)
            kw_i["past_key_values"] = None
            kw_i["use_cache"] = False
            out_i = block(inp_i, **_to_device(kw_i, device))[0]
            outputs.append(out_i.detach().cpu())
        return torch.cat(outputs, dim=0), layer_kwargs
    else:
        outputs = [
            block(inp.to(device), **_to_device(kw, device))[0]
            for inp, kw in zip(inps, layer_kwargs)
        ]
        return outputs, layer_kwargs
