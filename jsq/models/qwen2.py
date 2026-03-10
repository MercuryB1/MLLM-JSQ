"""Qwen2 model adapter."""
import torch.nn as nn
from typing import Dict, List, Tuple

from .base import ModelAdapter
from .registry import register_adapter


@register_adapter("qwen2")
class Qwen2Adapter(ModelAdapter):
    """Qwen2 has the same decoder structure as LLaMA (RMSNorm + GQA + SwiGLU MLP)."""

    def get_llm_blocks(self, model: nn.Module) -> List[nn.Module]:
        return model.model.layers

    def move_llm_embed(self, model: nn.Module, device) -> None:
        model.model.embed_tokens.to(device)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb.to(device)

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
