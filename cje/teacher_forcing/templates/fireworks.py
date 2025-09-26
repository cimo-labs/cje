"""Fireworks API template configuration with auto-detection."""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .base import ChatTemplateConfig

logger = logging.getLogger(__name__)

# Module-level cache for templates
_TEMPLATE_CACHE: Dict[str, Dict[str, Any]] = {}


class FireworksTemplateError(Exception):
    """Raised when Fireworks template cannot be loaded."""

    def __init__(self, model: str, details: str = ""):
        self.model = model
        self.details = details

        message = f"""No template found for model '{model}'

This model does not provide a chat template through Fireworks API."""

        if details:
            message += f"\n\nTechnical details: {details}"

        super().__init__(message)


def fetch_template_from_fireworks(model: str) -> Dict[str, Any]:
    """Fetch template configuration from Fireworks API.

    Args:
        model: Full model path (e.g., "accounts/fireworks/models/llama4-maverick-instruct-basic")

    Returns:
        Dictionary with conversation config

    Raises:
        FireworksTemplateError: If template cannot be fetched or is empty
    """
    import requests  # type: ignore

    # Get API key
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise FireworksTemplateError(
            model, "FIREWORKS_API_KEY environment variable not set"
        )

    # Parse model path
    parts = model.split("/")
    if len(parts) < 4 or parts[0] != "accounts" or parts[2] != "models":
        raise FireworksTemplateError(
            model,
            f"Invalid model path format. Expected 'accounts/X/models/Y', got '{model}'",
        )

    account_id = parts[1]
    model_id = parts[3]

    # Construct API endpoint
    endpoint = f"https://api.fireworks.ai/v1/accounts/{account_id}/models/{model_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.get(endpoint, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        conversation_config = data.get("conversationConfig", {})

        # Check if template exists and is non-empty
        template = conversation_config.get("template", "")
        if not template:
            raise FireworksTemplateError(model, "Model has empty template")

        return dict(conversation_config)

    except requests.exceptions.RequestException as e:
        if hasattr(e, "response") and e.response is not None:
            if e.response.status_code == 404:
                details = "Model not found on Fireworks"
            elif e.response.status_code == 401:
                details = "Invalid FIREWORKS_API_KEY"
            else:
                details = f"API error: {e.response.status_code}"
        else:
            details = f"Network error: {str(e)}"

        raise FireworksTemplateError(model, details)


class FireworksTemplateConfig(ChatTemplateConfig):
    """Auto-detected template configuration from Fireworks API.

    This uses Fireworks' conversation config to render chat templates.
    Templates are cached at the module level to avoid repeated API calls.
    """

    def __init__(self, model: str):
        """Initialize with Fireworks model path.

        Args:
            model: Full model path (e.g., "accounts/fireworks/models/llama4-maverick-instruct-basic")
        """
        self.model = model
        self._template: Optional[str] = None
        self._style: Optional[str] = None
        self._jinja_template: Optional[Any] = (
            None  # Will be jinja2.Template when loaded
        )

        # Load template (from cache or API)
        self._load_template()

    def _load_template(self) -> None:
        """Load template from cache or fetch from API."""
        if self.model not in _TEMPLATE_CACHE:
            logger.debug(f"Fetching template for {self.model}")
            _TEMPLATE_CACHE[self.model] = fetch_template_from_fireworks(self.model)

        config = _TEMPLATE_CACHE[self.model]
        self._template = config["template"]
        self._style = config.get("style", "")

        # Only support Jinja2 templates for now
        if self._style != "jinja":
            raise FireworksTemplateError(
                self.model,
                f"Unsupported template style: {self._style}. Only 'jinja' is supported.",
            )

        # Create Jinja2 template
        try:
            from jinja2 import Template

            if self._template is not None:
                self._jinja_template = Template(self._template)
            else:
                raise ValueError("Template string is None")
        except ImportError:
            raise ImportError(
                "jinja2 library required for Fireworks templates. "
                "Install with: pip install jinja2"
            )

    def format_message(self, role: str, content: str) -> str:
        """Format a single message - not supported for Jinja templates."""
        raise NotImplementedError(
            "FireworksTemplateConfig uses full chat rendering. "
            "Use apply_chat_template() instead."
        )

    def format_message_header(self, role: str) -> str:
        """Format message header - not supported for Jinja templates."""
        raise NotImplementedError(
            "FireworksTemplateConfig uses full chat rendering. "
            "Use apply_chat_template() instead."
        )

    def should_add_bos(self) -> bool:
        """BOS token handled by template."""
        return False

    def apply_chat_template(
        self, chat: List[Dict[str, str]], add_generation_prompt: bool = False
    ) -> str:
        """Apply the Fireworks template to format a chat conversation.

        Args:
            chat: List of message dictionaries with 'role' and 'content'
            add_generation_prompt: Whether to add prompt for assistant response

        Returns:
            Formatted string ready for the model
        """
        if self._jinja_template is None:
            raise RuntimeError("Template not loaded")

        # Prepare template variables
        # These are common variables expected by Fireworks templates
        template_vars = {
            "messages": chat,
            "add_generation_prompt": add_generation_prompt,
            "bos_token": "<|begin_of_text|>",  # Common default
            "eos_token": "<|end_of_text|>",  # Common default
            # Tool-related variables (set to None/empty)
            "tools": None,
            "custom_tools": None,
            "tool_definition": "",
        }

        try:
            result = self._jinja_template.render(**template_vars)
            return str(result)
        except Exception as e:
            raise FireworksTemplateError(
                self.model, f"Template rendering error: {str(e)}"
            )
