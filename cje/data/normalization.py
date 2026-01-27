"""Auto-normalization utilities for handling arbitrary label scales.

CJE internally works with values in [0, 1]. This module provides utilities
to automatically detect input ranges and normalize/inverse-transform values,
allowing users to work with any bounded scale (0-100, Likert 1-5, etc.).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ScaleInfo:
    """Information about a value scale for normalization.

    Stores min/max values and provides normalize/inverse transform methods.

    Example:
        >>> scale = ScaleInfo(min_val=0, max_val=100)
        >>> scale.normalize(75)
        0.75
        >>> scale.inverse(0.75)
        75.0
    """

    min_val: float
    max_val: float

    def normalize(self, x: float) -> float:
        """Normalize a value from [min_val, max_val] to [0, 1].

        Args:
            x: Value in original scale

        Returns:
            Value normalized to [0, 1]
        """
        if self.max_val == self.min_val:
            # Degenerate case: all values are the same
            return 0.5
        return (x - self.min_val) / (self.max_val - self.min_val)

    def inverse(self, x: float) -> float:
        """Inverse-transform a value from [0, 1] back to original scale.

        Args:
            x: Value in [0, 1]

        Returns:
            Value in original scale [min_val, max_val]
        """
        return x * (self.max_val - self.min_val) + self.min_val

    def normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Normalize an array of values.

        Args:
            arr: Array of values in original scale

        Returns:
            Array of values normalized to [0, 1]
        """
        if self.max_val == self.min_val:
            return np.full_like(arr, 0.5, dtype=float)
        return (arr - self.min_val) / (self.max_val - self.min_val)

    def inverse_array(self, arr: np.ndarray) -> np.ndarray:
        """Inverse-transform an array of values.

        Args:
            arr: Array of values in [0, 1]

        Returns:
            Array of values in original scale
        """
        return arr * (self.max_val - self.min_val) + self.min_val

    def is_identity(self, tolerance: float = 1e-6) -> bool:
        """Check if this scale is approximately [0, 1] (no transform needed).

        Args:
            tolerance: Tolerance for floating point comparison

        Returns:
            True if scale is approximately [0, 1]
        """
        return (
            abs(self.min_val - 0.0) < tolerance and abs(self.max_val - 1.0) < tolerance
        )

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "min_val": self.min_val,
            "max_val": self.max_val,
        }


def detect_range(
    values: np.ndarray,
    field_name: str = "values",
) -> ScaleInfo:
    """Detect the range of values and return ScaleInfo for normalization.

    Auto-scales to whatever min/max is detected in the data. Isotonic regression
    will learn the mapping regardless of scale.

    Args:
        values: Array of values to detect range from
        field_name: Name of field for error messages

    Returns:
        ScaleInfo with detected min/max

    Raises:
        ValueError: If no valid values

    Example:
        >>> values = np.array([20, 50, 80, 100])
        >>> scale = detect_range(values, "judge_score")
        >>> scale.min_val, scale.max_val
        (20.0, 100.0)
    """
    values = np.asarray(values, dtype=float)

    # Filter out NaN values
    valid_values = values[~np.isnan(values)]

    if len(valid_values) == 0:
        raise ValueError(f"No valid values to detect range for {field_name}")

    min_v = float(valid_values.min())
    max_v = float(valid_values.max())

    return ScaleInfo(min_val=min_v, max_val=max_v)


def detect_and_normalize(
    values: np.ndarray,
    field_name: str = "values",
) -> tuple[np.ndarray, ScaleInfo]:
    """Detect range and normalize values in one step.

    Args:
        values: Array of values (can contain NaN)
        field_name: Name of field for error messages

    Returns:
        Tuple of (normalized_values, scale_info)

    Example:
        >>> values = np.array([0, 50, 100])
        >>> normalized, scale = detect_and_normalize(values, "judge_score")
        >>> normalized
        array([0. , 0.5, 1. ])
    """
    scale = detect_range(values, field_name)

    # Normalize values (preserve NaN positions)
    normalized = np.where(np.isnan(values), np.nan, scale.normalize_array(values))

    return normalized, scale
