"""Qwen2-VL model adapter.

Architecture:
    model.visual          ← ViT encoder (keep FP16, never touch)
    model.visual.merger   ← Projector (keep FP16, never touch)
    model.model.layers    ← LLM decoder (compression target)
    model.model.embed_tokens
    model.model.rotary_emb  (MRoPE)
    model.model.norm
    model.lm_head
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .base import ModelAdapter
from .registry import register_adapter


@register_adapter("qwen2_vl")
class Qwen2VLAdapter(ModelAdapter):
    """Only compresses the LLM decoder; ViT and Projector are never modified."""

    def get_llm_blocks(self, model: nn.Module) -> List[nn.Module]:
        # model.model is the Qwen2VLModel; its .layers is the LLM decoder stack
        return model.model.layers

    def move_llm_embed(self, model: nn.Module, device) -> None:
        # Only move LLM-side embeddings.
        # model.visual is handled separately via move_vision_encoder().
        model.model.embed_tokens.to(device)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb.to(device)

    def move_vision_encoder(self, model: nn.Module, device) -> None:
        """Move ViT + Projector to device.

        Called before multimodal calibration forward passes so the ViT output
        (visual tokens) is on the same device as embed_tokens, avoiding device
        mismatch errors. After calibration, called with device="cpu" to free
        GPU memory for the per-layer compression loop.
        """
        model.visual.to(device)

    def get_named_linears(self, block: nn.Module) -> Dict[str, nn.Linear]:
        return {n: m for n, m in block.named_modules() if isinstance(m, nn.Linear)}

    def get_smooth_pairs(self, block: nn.Module) -> List[Tuple[nn.Module, List[nn.Linear]]]:
        # Qwen2-VL LLM decoder blocks have the same structure as Qwen2
        attn = block.self_attn
        mlp = block.mlp
        return [
            (block.input_layernorm,
             [attn.q_proj, attn.k_proj, attn.v_proj]),
            (block.post_attention_layernorm,
             [mlp.gate_proj, mlp.up_proj]),
        ]

    def run_forward_for_calibration(self, model: nn.Module, samples, **kwargs):
        """Handle both pure-text and image-text calibration inputs.

        If samples is a dict (image+text batch from multimodal calibration),
        we pass it through the full model (ViT FP16 → merger → LLM).
        The Catcher hook on model.model.layers[0] will intercept the LLM input,
        which already contains the fused visual tokens.
        """
        if isinstance(samples, dict):
            return model(**{k: v.to(next(model.parameters()).device)
                            if isinstance(v, torch.Tensor) else v
                            for k, v in samples.items()})
        return model(samples.to(next(model.parameters()).device))
