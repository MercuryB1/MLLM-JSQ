"""Qwen3-VL model adapter.

Architecture (transformers >= 5.x):
    model.model.visual                   ← ViT encoder (keep FP16, never touch)
    model.model.language_model.layers    ← LLM decoder (compression target)
    model.model.language_model.embed_tokens
    model.model.language_model.rotary_emb  (MRoPE)
    model.model.language_model.norm
    model.lm_head

Note: Qwen3-VL attention uses QK-norm (q_norm, k_norm) but these are
      applied after projection, so smooth_pairs targets q_proj/k_proj/v_proj
      as usual.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .base import ModelAdapter
from .registry import register_adapter


@register_adapter("qwen3_vl")
class Qwen3VLAdapter(ModelAdapter):
    """Only compresses the LLM decoder; ViT and Projector are never modified."""

    def get_llm_blocks(self, model: nn.Module) -> List[nn.Module]:
        return model.model.language_model.layers

    def move_llm_embed(self, model: nn.Module, device) -> None:
        lm = model.model.language_model
        lm.embed_tokens.to(device)
        if hasattr(lm, "rotary_emb"):
            lm.rotary_emb.to(device)

    def move_vision_encoder(self, model: nn.Module, device) -> None:
        model.model.visual.to(device)

    def get_named_linears(self, block: nn.Module) -> Dict[str, nn.Linear]:
        return {n: m for n, m in block.named_modules() if isinstance(m, nn.Linear)}

    def get_smooth_pairs(self, block: nn.Module) -> List[Tuple[nn.Module, List[nn.Linear]]]:
        attn = block.self_attn
        mlp = block.mlp
        return [
            (block.input_layernorm,
             [attn.q_proj, attn.k_proj, attn.v_proj]),
            (block.post_attention_layernorm,
             [mlp.gate_proj, mlp.up_proj]),
        ]

    def run_forward_for_calibration(self, model: nn.Module, samples, **kwargs):
        if isinstance(samples, dict):
            return model(**{k: v.to(next(model.parameters()).device)
                            if isinstance(v, torch.Tensor) else v
                            for k, v in samples.items()})
        return model(samples.to(next(model.parameters()).device))
