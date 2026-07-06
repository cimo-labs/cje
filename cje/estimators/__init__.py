"""Core CJE estimators and types.

This module contains:
- Estimators: CalibratedDirectEstimator
- Data models: Pydantic models for type safety
"""

from .direct_method import CalibratedDirectEstimator
from ..data.models import (
    Sample,
    Dataset,
    EstimationResult,
)

__all__ = [
    # Estimators
    "CalibratedDirectEstimator",
    # Data models
    "Sample",
    "Dataset",
    "EstimationResult",
]


def __getattr__(name: str) -> object:
    if name == "BaseCJEEstimator":
        raise ImportError(
            "cje.estimators.BaseCJEEstimator was removed in 0.5.0 — "
            "CalibratedDirectEstimator is the only estimator (the base "
            "class had exactly one subclass and was merged into it)."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
