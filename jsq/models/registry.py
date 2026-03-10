"""Model adapter registry."""
from typing import Dict, Type

import torch.nn as nn

from .base import ModelAdapter

_REGISTRY: Dict[str, Type[ModelAdapter]] = {}


def register_adapter(*model_types: str):
    """Decorator to register an Adapter class for one or more model_type strings."""
    def decorator(cls: Type[ModelAdapter]) -> Type[ModelAdapter]:
        for t in model_types:
            _REGISTRY[t] = cls
        return cls
    return decorator


def get_adapter(model: nn.Module) -> ModelAdapter:
    """Look up and instantiate the Adapter for the given model.

    Uses model.config.model_type as the key.
    Raises NotImplementedError if no adapter is registered.
    """
    model_type: str = getattr(model.config, "model_type", "")
    if model_type not in _REGISTRY:
        registered = list(_REGISTRY.keys())
        raise NotImplementedError(
            f"No adapter registered for model_type='{model_type}'. "
            f"Registered types: {registered}"
        )
    return _REGISTRY[model_type]()
