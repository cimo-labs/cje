"""Teacher forcing utilities for computing log probabilities.

This module provides:
- Fireworks API integration for teacher forcing
- Chat to completions format conversion
- Support for various model templates

Note: Requires optional 'teacher-forcing' dependencies (fireworks-ai, transformers).
Install with: pip install cje-eval[teacher-forcing]
"""

# Template configurations (no dependencies)
from .templates import (
    ChatTemplateConfig,
    Llama3TemplateConfig,
    FireworksTemplateConfig,
    FireworksTemplateError,
)

# Chat utilities (no dependencies)
from .chat import (
    compute_chat_logprob,
    convert_chat_to_completions,
)

__all__ = [
    # Template configurations (always available)
    "ChatTemplateConfig",
    "Llama3TemplateConfig",
    "FireworksTemplateConfig",
    "FireworksTemplateError",
    # Chat support (always available)
    "compute_chat_logprob",
    "convert_chat_to_completions",
]

# Optional: Fireworks API (requires fireworks-ai)
try:
    from .api import compute_teacher_forced_logprob

    __all__.append("compute_teacher_forced_logprob")
except ImportError:
    pass  # Silently skip - api/__init__.py already warned

# Optional: HuggingFace templates (requires transformers)
try:
    from .templates import HuggingFaceTemplateConfig

    __all__.append("HuggingFaceTemplateConfig")
except ImportError:
    pass  # Silently skip - will fail when user tries to use it
