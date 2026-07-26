"""Auto-normalization utilities for handling arbitrary label scales.

CJE internally works with values in [0, 1]. This module provides utilities
to automatically detect input ranges and normalize/inverse-transform values,
allowing users to work with any bounded scale (0-100, Likert 1-5, etc.).
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union
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

    @property
    def span(self) -> float:
        """Width of the public scale."""
        return self.max_val - self.min_val

    def to_dict(self) -> dict:
        """JSON-friendly scale declaration."""
        return {
            "min": float(self.min_val),
            "max": float(self.max_val),
            "is_identity": self.is_identity(),
        }


ScaleDeclaration = Optional[Union[ScaleInfo, Sequence[float]]]


def coerce_scale(
    value: ScaleDeclaration,
    *,
    field_name: str,
) -> Optional[ScaleInfo]:
    """Validate a public ``(minimum, maximum)`` scale declaration.

    ``None`` means that the caller did not declare a scale. A declared scale
    must be finite and non-degenerate; observed-range inference remains a
    separate compatibility behavior in the fresh-draw loader.
    """
    if value is None:
        return None
    if isinstance(value, ScaleInfo):
        scale = value
    else:
        try:
            valid_pair = not isinstance(value, (str, bytes)) and len(value) == 2
        except TypeError:
            valid_pair = False
        if not valid_pair:
            raise ValueError(
                f"{field_name} must be a (minimum, maximum) pair, got {value!r}"
            )
        try:
            scale = ScaleInfo(min_val=float(value[0]), max_val=float(value[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} bounds must be numeric, got {value!r}"
            ) from exc
    if not np.isfinite(scale.min_val) or not np.isfinite(scale.max_val):
        raise ValueError(f"{field_name} bounds must be finite, got {value!r}")
    if scale.max_val <= scale.min_val:
        raise ValueError(
            f"{field_name} maximum must be greater than its minimum, got {value!r}"
        )
    return scale


def unit_scale() -> ScaleInfo:
    """Return the canonical unit-interval scale."""
    return ScaleInfo(0.0, 1.0)


def validate_values_on_scale(
    values: np.ndarray,
    scale: ScaleInfo,
    *,
    field_name: str,
    tolerance: float = 1e-9,
) -> None:
    """Fail when finite values fall outside their declared public scale."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return
    bad = (finite < scale.min_val - tolerance) | (finite > scale.max_val + tolerance)
    if np.any(bad):
        observed = (float(np.min(finite)), float(np.max(finite)))
        raise ValueError(
            f"{field_name} values fall outside declared scale "
            f"[{scale.min_val}, {scale.max_val}] (observed range "
            f"{observed[0]}-{observed[1]}, {int(np.sum(bad))} rows)."
        )


class ScaledCalibrator:
    """Public-unit facade over an internally unit-scaled calibrator.

    Estimation continues to use the raw calibrator on ``[0, 1]``. Results
    expose this facade so documented reuse and transport checks accept the
    same judge units as evaluation input and return the result's oracle units.
    """

    def __init__(
        self,
        calibrator: Any,
        *,
        judge_scale: ScaleInfo,
        output_scale: ScaleInfo,
    ) -> None:
        self.raw_calibrator = calibrator
        self.judge_scale = judge_scale
        self.output_scale = output_scale

    def predict(
        self, judge_scores: Any, covariates: Optional[Any] = None
    ) -> np.ndarray:
        scores = np.asarray(judge_scores, dtype=float)
        validate_values_on_scale(scores, self.judge_scale, field_name="judge_scores")
        internal_scores = self.judge_scale.normalize_array(scores)
        predictions = np.asarray(
            self.raw_calibrator.predict(internal_scores, covariates=covariates),
            dtype=float,
        )
        return self.output_scale.inverse_array(predictions)

    @property
    def oracle_s_range(self) -> Optional[tuple]:
        internal = getattr(self.raw_calibrator, "oracle_s_range", None)
        if internal is None:
            return None
        converted = self.judge_scale.inverse_array(np.asarray(internal, dtype=float))
        return (float(converted[0]), float(converted[1]))

    @property
    def oracle_reward_range(self) -> Optional[tuple]:
        internal = getattr(self.raw_calibrator, "oracle_reward_range", None)
        if internal is None:
            return None
        converted = self.output_scale.inverse_array(np.asarray(internal, dtype=float))
        return (float(converted[0]), float(converted[1]))

    @property
    def covariate_names(self) -> list:
        return list(getattr(self.raw_calibrator, "covariate_names", None) or [])

    def get_calibration_info(self) -> dict:
        info = dict(self.raw_calibrator.get_calibration_info())
        for key in ("rmse", "oof_rmse"):
            value = info.get(key)
            if value is not None:
                info[key] = float(value) * self.output_scale.span
        info["coverage_tolerance"] = 0.1 * self.output_scale.span
        info["judge_input_scale"] = self.judge_scale.to_dict()
        info["oracle_output_scale"] = self.output_scale.to_dict()
        return info

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw_calibrator, name)


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
