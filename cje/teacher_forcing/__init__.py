"""Teacher forcing utilities for computing log probabilities.

This module provides:
- Fireworks API integration for teacher forcing
- Chat to completions format conversion
- Support for various model templates
"""

# API implementations
from .api import (
    compute_teacher_forced_logprob,
)

# Template configurations
from .templates import (
    ChatTemplateConfig,
    Llama3TemplateConfig,
    HuggingFaceTemplateConfig,
    FireworksTemplateConfig,
    FireworksTemplateError,
)

# Chat utilities
from .chat import (
    compute_chat_logprob,
    convert_chat_to_completions,
)

__all__ = [
    # Fireworks teacher forcing
    "compute_teacher_forced_logprob",
    # Template configurations
    "ChatTemplateConfig",
    "Llama3TemplateConfig",
    "HuggingFaceTemplateConfig",
    "FireworksTemplateConfig",
    "FireworksTemplateError",
    # Chat support
    "compute_chat_logprob",
    "convert_chat_to_completions",
]
