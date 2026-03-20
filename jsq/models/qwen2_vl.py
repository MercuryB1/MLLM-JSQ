"""Qwen2-VL model adapter.

Architecture (transformers >= 5.x):
    model.model.visual                   ← ViT encoder (keep FP16, never touch)
    model.model.language_model.layers    ← LLM decoder (compression target)
    model.model.language_model.embed_tokens
    model.model.language_model.rotary_emb  (MRoPE)
    model.model.language_model.norm
    model.lm_head
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .base import ModelAdapter
from .registry import register_adapter


@register_adapter("qwen2_vl")
class Qwen2VLAdapter(ModelAdapter):
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

    def get_vision_token_mask(
        self, calib_samples, processor=None
    ) -> Optional[List[torch.Tensor]]:
        """Extract per-sample vision token masks from <|image_pad|> positions."""
        if not isinstance(calib_samples, list):
            return None

        # Resolve image_pad token ID via processor or fall back to Qwen2-VL default
        image_pad_id = 151655  # <|image_pad|> in Qwen2-VL tokenizer
        if processor is not None:
            tok = getattr(processor, "tokenizer", processor)
            _id = tok.convert_tokens_to_ids("<|image_pad|>")
            if _id not in (None, tok.unk_token_id):
                image_pad_id = _id

        masks: List[torch.Tensor] = []
        for sample in calib_samples:
            if not isinstance(sample, dict) or "input_ids" not in sample:
                masks.append(None)
                continue
            ids = sample["input_ids"]
            if isinstance(ids, torch.Tensor):
                mask = ids.squeeze(0) == image_pad_id
            else:
                mask = torch.tensor([x == image_pad_id for x in ids], dtype=torch.bool)
            masks.append(mask)

        return masks if any(m is not None for m in masks) else None

    def run_forward_for_calibration(self, model: nn.Module, samples, **kwargs):
        if isinstance(samples, dict):
            # Multimodal: ViT has been moved to device by move_vision_encoder,
            # so next(model.parameters()) is on the correct device.
            dev = next(model.parameters()).device
            return model(**{k: v.to(dev) if isinstance(v, torch.Tensor) else v
                            for k, v in samples.items()})
        # Text-only calibration: use the LLM embed_tokens device, not the ViT
        # device (ViT may still be on CPU while embed_tokens is on GPU).
        dev = model.model.language_model.embed_tokens.weight.device
        return model(samples.to(dev))
