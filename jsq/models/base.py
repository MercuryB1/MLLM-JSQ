"""Abstract base class for model adapters."""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class ModelAdapter(ABC):
    """
    Each model family implements this interface.
    All core compression code depends only on this ABC —
    never on concrete model classes.
    """

    @abstractmethod
    def get_llm_blocks(self, model: nn.Module) -> List[nn.Module]:
        """Return the list of Transformer blocks to compress (LLM decoder only).
        For MLLMs, exclude the ViT and projector."""

    @abstractmethod
    def move_llm_embed(self, model: nn.Module, device) -> None:
        """Move LLM-side embedding layers to device.
        For MLLMs, do NOT move ViT-related modules."""

    @abstractmethod
    def get_named_linears(self, block: nn.Module) -> Dict[str, nn.Linear]:
        """Return all Linear layers in the block that should participate in compression.
        Keys are relative names within the block (e.g. 'self_attn.q_proj')."""

    @abstractmethod
    def get_smooth_pairs(self, block: nn.Module) -> List[Tuple[nn.Module, List[nn.Linear]]]:
        """Return (norm_layer, [linear, ...]) pairs for activation smoothing.
        Each tuple represents one LayerNorm → Linear(s) smoothing relationship.

        Example (LLaMA):
            [(input_layernorm, [q_proj, k_proj, v_proj]),
             (post_attention_layernorm, [gate_proj, up_proj])]
        """

    def move_vision_encoder(self, model: nn.Module, device) -> None:
        """Move the vision encoder (ViT + projector) to device.

        Called before and after the calibration forward pass so the ViT is
        on the same device as the LLM embed during MLLM calibration, then
        moved back to CPU to free GPU memory for the per-layer compression loop.

        Default implementation is a no-op (text-only LLMs have no ViT).
        MLLM adapters should override this."""

    def run_forward_for_calibration(
        self, model: nn.Module, samples, **kwargs
    ):
        """Drive model forward to capture LLM first-layer inputs.
        Default: call model(samples). MLLMs override to handle image+text inputs."""
        if isinstance(samples, dict):
            return model(**samples)
        return model(samples, **kwargs)
