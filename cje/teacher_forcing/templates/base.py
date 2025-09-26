"""Base class for chat template configurations."""

from abc import ABC, abstractmethod


class ChatTemplateConfig(ABC):
    """Abstract base class for chat template configuration."""

    @abstractmethod
    def format_message(self, role: str, content: str) -> str:
        """Format a single message with role and content."""
        pass

    @abstractmethod
    def format_message_header(self, role: str) -> str:
        """Format just the message header (for empty assistant stub)."""
        pass

    @abstractmethod
    def should_add_bos(self) -> bool:
        """Whether to add beginning-of-sequence token."""
        pass
