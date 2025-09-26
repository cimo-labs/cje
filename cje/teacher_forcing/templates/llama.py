"""Llama chat template configurations."""

from dataclasses import dataclass
from .base import ChatTemplateConfig


@dataclass
class Llama4TemplateConfig(ChatTemplateConfig):
    """Llama 4 chat template configuration.

    Uses the format:
    <|header_start|>role<|header_end|>\n\ncontent<|eot|>
    """

    begin_of_text: str = "<|begin_of_text|>"
    header_start: str = "<|header_start|>"
    header_end: str = "<|header_end|>"
    eot: str = "<|eot|>"
    newlines_after_header: int = 2
    add_bos_token: bool = False

    def format_message(self, role: str, content: str) -> str:
        """Format a complete message."""
        header = self.format_message_header(role)
        return f"{header}{content}{self.eot}"

    def format_message_header(self, role: str) -> str:
        """Format just the header."""
        newlines = "\n" * self.newlines_after_header
        return f"{self.header_start}{role}{self.header_end}{newlines}"

    def should_add_bos(self) -> bool:
        return self.add_bos_token


@dataclass
class Llama3TemplateConfig(ChatTemplateConfig):
    """Llama 3 chat template configuration.

    Uses the format:
    <|start_header_id|>role<|end_header_id|>\ncontent<|eot_id|>
    """

    begin_of_text: str = "<|begin_of_text|>"
    header_start: str = "<|start_header_id|>"
    header_end: str = "<|end_header_id|>"
    eot: str = "<|eot_id|>"
    newlines_after_header: int = 1
    add_bos_token: bool = False

    def format_message(self, role: str, content: str) -> str:
        """Format a complete message."""
        header = self.format_message_header(role)
        return f"{header}{content}{self.eot}"

    def format_message_header(self, role: str) -> str:
        """Format just the header."""
        newlines = "\n" * self.newlines_after_header
        return f"{self.header_start}{role}{self.header_end}{newlines}"

    def should_add_bos(self) -> bool:
        return self.add_bos_token
