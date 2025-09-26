"""Chat conversation utilities for teacher forcing.

This module provides a streamlined interface for computing log probabilities
of chat conversations using the teacher forcing method.
"""

from typing import List, Dict, Optional, Tuple
import logging

from ..data.models import LogProbResult, LogProbStatus
from .api import compute_teacher_forced_logprob
from .templates import (
    ChatTemplateConfig,
    HuggingFaceTemplateConfig,
    FireworksTemplateConfig,
    FireworksTemplateError,
)

logger = logging.getLogger(__name__)


def convert_chat_to_completions(
    chat: List[Dict[str, str]], template_config: ChatTemplateConfig
) -> Tuple[str, str]:
    """Convert chat messages to completion format for teacher forcing.

    Args:
        chat: List of messages with 'role' and 'content' keys
        template_config: Chat template configuration to use

    Returns:
        Tuple of (prompt_only, prompt_plus_reply) for teacher forcing

    Raises:
        ValueError: If chat format is invalid
    """
    if not chat:
        raise ValueError("Chat cannot be empty")

    if chat[-1]["role"] != "assistant":
        raise ValueError(
            "Last message must be assistant reply for teacher forcing. "
            f"Got role='{chat[-1]['role']}'"
        )

    # Use apply_chat_template for HuggingFace and Fireworks configs
    if isinstance(
        template_config, (HuggingFaceTemplateConfig, FireworksTemplateConfig)
    ):
        # Full prompt = context + assistant reply
        prompt_plus_reply = template_config.apply_chat_template(
            chat, add_generation_prompt=False
        )

        # Prefix prompt = context only, with empty assistant header
        prompt_only = template_config.apply_chat_template(
            chat[:-1], add_generation_prompt=True
        )

        return prompt_only, prompt_plus_reply

    # Manual template construction
    context_parts = []

    # Add begin_of_text if needed
    if template_config.should_add_bos() and hasattr(template_config, "begin_of_text"):
        context_parts.append(template_config.begin_of_text)

    # Add all messages except the last assistant reply
    for msg in chat[:-1]:
        context_parts.append(
            template_config.format_message(msg["role"], msg["content"])
        )

    # Add empty assistant header for prefix
    assistant_header = template_config.format_message_header("assistant")
    prompt_only = "".join(context_parts) + assistant_header

    # Full prompt includes the assistant reply
    last_msg = chat[-1]
    context_parts.append(
        template_config.format_message(last_msg["role"], last_msg["content"])
    )
    prompt_plus_reply = "".join(context_parts)

    return prompt_only, prompt_plus_reply


def compute_chat_logprob(
    chat: List[Dict[str, str]],
    model: str,
    template_config: Optional[ChatTemplateConfig] = None,
    api_key: Optional[str] = None,
    temperature: float = 1.0,
    system_prompt: Optional[str] = None,
) -> LogProbResult:
    """Compute log probability of assistant's reply in a chat conversation.

    This function uses the teacher forcing method: computing log P(assistant_reply | context)
    by making two API calls and subtracting log probabilities.

    Args:
        chat: Chat messages (last must be assistant)
        model: Model name (e.g., "accounts/fireworks/models/llama-v3-8b-instruct")
        template_config: Chat template configuration (auto-detected if None)
        api_key: API key (uses env var if not provided)
        temperature: Sampling temperature (default: 1.0)
        system_prompt: Optional system prompt to prepend

    Returns:
        LogProbResult with log P(assistant_reply | context)

    Example:
        >>> from cje.teacher_forcing import HuggingFaceTemplateConfig
        >>>
        >>> chat = [
        ...     {"role": "user", "content": "What is 2+2?"},
        ...     {"role": "assistant", "content": "4"}
        ... ]
        >>> config = HuggingFaceTemplateConfig("meta-llama/Llama-3.2-3B-Instruct")
        >>> result = compute_chat_logprob(
        ...     chat,
        ...     "accounts/fireworks/models/llama-v3-8b-instruct",
        ...     config
        ... )
        >>> if result.is_valid:
        ...     print(f"Log probability: {result.value}")
    """
    # Add system prompt if provided
    if system_prompt and (not chat or chat[0]["role"] != "system"):
        chat = [{"role": "system", "content": system_prompt}] + chat

    # Auto-detect template if not provided
    if template_config is None:
        # Check if it's a Fireworks model
        if model.startswith("accounts/fireworks/models/"):
            try:
                template_config = FireworksTemplateConfig(model)
                logger.debug(f"Auto-detected Fireworks template for {model}")
            except FireworksTemplateError as e:
                # Return error with helpful message
                return LogProbResult(
                    value=None,
                    status=LogProbStatus.API_ERROR,
                    error=str(e),
                )
        else:
            # For non-Fireworks models, require explicit template
            return LogProbResult(
                value=None,
                status=LogProbStatus.API_ERROR,
                error=(
                    f"No template config provided for model '{model}'. "
                    "For Fireworks models, templates are auto-detected. "
                    "For other models, please provide an explicit template_config."
                ),
            )

    # Validate chat format
    if not chat or chat[-1]["role"] != "assistant":
        return LogProbResult(
            value=None,
            status=LogProbStatus.API_ERROR,
            error="Chat must end with assistant message for teacher forcing",
        )

    # Extract the assistant reply we're scoring
    assistant_reply = chat[-1]["content"]

    # Convert to completions format
    try:
        prompt_only, prompt_plus_reply = convert_chat_to_completions(
            chat, template_config
        )
    except Exception as e:
        return LogProbResult(
            value=None,
            status=LogProbStatus.API_ERROR,
            error=f"Chat conversion failed: {str(e)}",
        )

    # Make two API calls for teacher forcing
    try:
        # Get log P(prompt + reply)
        full_result = compute_teacher_forced_logprob(
            prompt="",
            response=prompt_plus_reply,
            model=model,
            api_key=api_key,
            temperature=temperature,
        )
        if not full_result.is_valid:
            return full_result

        # Get log P(prompt only)
        prefix_result = compute_teacher_forced_logprob(
            prompt="",
            response=prompt_only,
            model=model,
            api_key=api_key,
            temperature=temperature,
        )
        if not prefix_result.is_valid:
            return prefix_result

        # Compute log P(reply | prompt) = log P(full) - log P(prefix)
        if full_result.value is None or prefix_result.value is None:
            return LogProbResult(
                value=None,
                status=LogProbStatus.API_ERROR,
                error="Missing value from full or prefix computation",
            )
        lp_reply = full_result.value - prefix_result.value

        # Sanity checks
        if lp_reply > 0:
            return LogProbResult(
                value=None,
                status=LogProbStatus.API_ERROR,
                error=f"Positive log probability: {lp_reply}",
                metadata={
                    "lp_full": full_result.value,
                    "lp_prefix": prefix_result.value,
                },
            )

        # Check for extreme values
        tokens_estimate = len(assistant_reply) / 4  # ~4 chars per token
        if tokens_estimate > 0:
            avg_per_token = lp_reply / tokens_estimate
            if avg_per_token < -10:
                logger.warning(
                    f"Extreme negative log prob: {lp_reply:.2f} "
                    f"for ~{tokens_estimate:.0f} tokens "
                    f"(avg: {avg_per_token:.2f}/token)"
                )

        return LogProbResult(
            value=lp_reply,
            status=LogProbStatus.SUCCESS,
            error=None,
            metadata={
                "method": "chat_teacher_forcing",
                "lp_full": full_result.value,
                "lp_prefix": prefix_result.value,
                "reply_length": len(assistant_reply),
                "template_type": type(template_config).__name__,
            },
        )

    except Exception as e:
        return LogProbResult(
            value=None,
            status=LogProbStatus.API_ERROR,
            error=f"API calls failed: {str(e)}",
        )
