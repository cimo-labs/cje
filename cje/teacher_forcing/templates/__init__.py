"""Chat template configurations."""

from .base import ChatTemplateConfig
from .llama import Llama3TemplateConfig, Llama4TemplateConfig
from .fireworks import FireworksTemplateConfig, FireworksTemplateError

__all__ = [
    "ChatTemplateConfig",
    "Llama3TemplateConfig",
    "Llama4TemplateConfig",
    "FireworksTemplateConfig",
    "FireworksTemplateError",
]

# Optional: HuggingFace template (requires transformers)
try:
    from .huggingface import HuggingFaceTemplateConfig

    __all__.append("HuggingFaceTemplateConfig")
except ImportError:
    pass  # transformers not installed
