"""High-level interface for CJE.

This module provides the main user-facing tools:
- analyze_dataset(): One-line analysis function
- CLI: Command-line interface
"""

from .analysis import analyze_dataset

__all__ = ["analyze_dataset"]
