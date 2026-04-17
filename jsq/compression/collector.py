"""Utilities for collecting layer inputs during calibration."""
import copy
import functools
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger


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
            x = inp.to(device)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            out = block(x, **_to_device(kw, device))[0]
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
            x = inp.to(device)
            if x.dim() == 2:
                x = x.unsqueeze(0)  # [seq, h] → [1, seq, h]
            block(x, **_to_device(kw, device))
        for h in handles:
            h.remove()
        result: Dict = {k: torch.cat(v, dim=0) for k, v in feat.items()}
        result["__nsamples__"] = len(inps)
        return result


@torch.no_grad()
def collect_block_sensitivity(
    blocks: List[nn.Module],
    inps,
    layer_kwargs,
    vision_masks: Optional[List] = None,
    device: Optional[torch.device] = None,
) -> Tuple[List[float], List[float]]:
    """Uncompressed forward pass through all blocks; compute per-block BI_t / BI_v.

    Block Influence (BI) per modality m:
        BI_m[l] = mean_{i in m} (1 - cos(x_i, y_i))
    computed from the inputs x and outputs y of block l.

    In text mode (inps is Tensor), vision_masks is ignored → BI_t only.
    In multimodal mode (inps is List[Tensor]), modality split follows
    vision_masks[s] (per-sample [seq] bool). Samples whose vision_mask is
    missing/mismatched are counted as text.

    Blocks are moved to device one at a time and returned to CPU afterwards.
    inps is consumed forward-by-forward; a fresh copy is produced per block
    and original *inps* is not mutated.

    Returns:
        (bi_t, bi_v) — each a list of length len(blocks), one float per block.
    """
    from .block_vl_allocator import block_influence_split, aggregate_sample_bi

    n_blocks = len(blocks)
    bi_t: List[float] = [0.0] * n_blocks
    bi_v: List[float] = [0.0] * n_blocks

    is_multimodal = not isinstance(inps, torch.Tensor)
    current = inps

    for li, block in enumerate(blocks):
        tgt_dev = device if device is not None else next(block.parameters()).device
        block.to(tgt_dev)

        per_sample_stats: List[Tuple[float, float, int, int]] = []

        if is_multimodal:
            outputs: List[torch.Tensor] = []
            for s_idx, (inp, kw) in enumerate(zip(current, layer_kwargs)):
                x = inp.to(tgt_dev)
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                y = block(x, **_to_device(kw, tgt_dev))[0]
                x_flat = x.reshape(-1, x.shape[-1])
                y_flat = y.reshape(-1, y.shape[-1])
                vm = None
                if vision_masks is not None and s_idx < len(vision_masks):
                    m = vision_masks[s_idx]
                    if m is not None:
                        vm = m
                stats = block_influence_split(x_flat, y_flat, vm)
                per_sample_stats.append(stats)
                outputs.append(y.detach())
            next_inps = outputs
        else:
            batch_size = current.shape[0]
            outputs = []
            for i in range(batch_size):
                inp_i = current[i: i + 1].to(tgt_dev)
                kw_i = _slice_kw_for_sample(layer_kwargs, i, batch_size)
                kw_i["past_key_values"] = None
                kw_i["use_cache"] = False
                y = block(inp_i, **_to_device(kw_i, tgt_dev))[0]
                x_flat = inp_i.reshape(-1, inp_i.shape[-1])
                y_flat = y.reshape(-1, y.shape[-1])
                stats = block_influence_split(x_flat, y_flat, None)
                per_sample_stats.append(stats)
                outputs.append(y.detach().cpu())
            next_inps = torch.cat(outputs, dim=0)

        bt, bv = aggregate_sample_bi(per_sample_stats)
        bi_t[li], bi_v[li] = bt, bv
        logger.info(f"block {li}: BI_t={bt:.4f} BI_v={bv:.4f}")

        current = next_inps
        block.cpu()
        torch.cuda.empty_cache()

    return bi_t, bi_v


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
        def _fwd(inp, kw):
            x = inp.to(device)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            return block(x, **_to_device(kw, device))[0]
        outputs = [_fwd(inp, kw) for inp, kw in zip(inps, layer_kwargs)]
        return outputs, layer_kwargs


@torch.no_grad()
def collect_block_pruning_damage(
    blocks: List[nn.Module],
    inps,
    layer_kwargs,
    pruning_pass,
    adapter,
    config,
    vision_masks: Optional[List] = None,
    device: Optional[torch.device] = None,
    sequential: bool = False,
) -> List[float]:
    """Measure per-block reconstruction error from trial pruning.

    For each block:
      1. Forward current inputs through clean block → y_clean
      2. Collect input features (for pruning metric computation)
      3. Save weight copies, apply trial pruning at target sparsity
      4. Forward current inputs through pruned block → y_pruned
      5. damage[l] = MSE(y_clean, y_pruned)
      6. Restore original weights

    Args:
        sequential: if False (default), propagate y_clean to next block
            (independent per-block damage). If True, propagate y_pruned
            to next block (sequential damage — captures cascading errors
            from earlier blocks being pruned). Weights are always restored
            after measurement; only the signal propagation differs.

    Returns:
        damage: list of length len(blocks), one float per block.
    """
    n_blocks = len(blocks)
    damage: List[float] = [0.0] * n_blocks
    is_multimodal = not isinstance(inps, torch.Tensor)
    current = inps

    def _flat_vmask(cur_inps):
        if vision_masks is None or not isinstance(cur_inps, list):
            return None
        parts = []
        for idx, inp in enumerate(cur_inps):
            n_tok = inp.reshape(-1, inp.shape[-1]).shape[0]
            m = vision_masks[idx] if idx < len(vision_masks) else None
            if m is None or m.shape[0] != n_tok:
                parts.append(torch.zeros(n_tok, dtype=torch.bool))
            else:
                parts.append(m.bool())
        flat = torch.cat(parts, dim=0)
        return flat if flat.any() else None

    for li, block in enumerate(blocks):
        tgt_dev = device if device is not None else next(block.parameters()).device
        block.to(tgt_dev)

        # Step 1: forward clean → y_clean, and collect input features
        input_feat, y_clean_inps, _ = collect_block_input_feat_and_output(
            block, current, layer_kwargs
        )

        # Step 2: save original weights
        saved_weights = {}
        named_linears = adapter.get_named_linears(block)
        for name, linear in named_linears.items():
            saved_weights[name] = linear.weight.data.clone()

        # Step 3: apply trial pruning at target sparsity
        flat_vm = _flat_vmask(current)
        kw = {}
        if hasattr(pruning_pass, "_supports_per_layer"):
            kw["vision_mask"] = flat_vm
        pruning_pass.apply(block, input_feat, adapter, config, **kw)

        # Step 4: forward pruned → y_pruned and compute damage
        if is_multimodal:
            mse_total = 0.0
            y_pruned_list: List[torch.Tensor] = []
            for s_idx, (inp, kw_s) in enumerate(zip(current, layer_kwargs)):
                x = inp.to(tgt_dev)
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                y_pruned = block(x, **_to_device(kw_s, tgt_dev))[0]
                y_c = y_clean_inps[s_idx]
                if isinstance(y_c, torch.Tensor):
                    y_c = y_c.to(tgt_dev)
                mse_total += float((y_pruned.float() - y_c.float()).pow(2).mean())
                y_pruned_list.append(y_pruned.detach())
            damage[li] = mse_total / max(len(current), 1)
        else:
            batch_size = current.shape[0]
            mse_total = 0.0
            y_pruned_parts: List[torch.Tensor] = []
            for i in range(batch_size):
                inp_i = current[i: i + 1].to(tgt_dev)
                kw_i = _slice_kw_for_sample(layer_kwargs, i, batch_size)
                kw_i["past_key_values"] = None
                kw_i["use_cache"] = False
                y_pruned = block(inp_i, **_to_device(kw_i, tgt_dev))[0]
                y_c = y_clean_inps[i: i + 1].to(tgt_dev)
                mse_total += float((y_pruned.float() - y_c.float()).pow(2).mean())
                y_pruned_parts.append(y_pruned.detach().cpu())
            damage[li] = mse_total / max(batch_size, 1)

        # Step 5: restore original weights
        for name, linear in named_linears.items():
            linear.weight.data.copy_(saved_weights[name])

        logger.info(f"block {li}: pruning_damage={damage[li]:.6f}")

        # Propagate: sequential uses pruned outputs, independent uses clean
        if sequential:
            if is_multimodal:
                current = y_pruned_list
            else:
                current = torch.cat(y_pruned_parts, dim=0)
        else:
            current = y_clean_inps

        del input_feat, saved_weights
        block.cpu()
        torch.cuda.empty_cache()

    return damage
