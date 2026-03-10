"""Abstract base class for compression passes."""
from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class CompressionPass(ABC):
    """A single in-place transformation applied to each Transformer block.

    Each pass receives the block, pre-collected input features for all
    Linear layers inside the block, the model adapter, and the config.
    Passes must not return anything — all modifications are in-place.
    """

    @abstractmethod
    def apply(
        self,
        block: nn.Module,
        input_feat: Dict[str, torch.Tensor],
        adapter,
        config,
    ) -> None:
        """Apply the compression transformation in-place.

        Args:
            block: The Transformer block being compressed.
            input_feat: Dict mapping Linear layer name (relative to block)
                        to its collected input activations tensor.
            adapter: ModelAdapter instance for this model family.
            config: CompressConfig instance.
        """
