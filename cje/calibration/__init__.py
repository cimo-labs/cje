"""Calibration utilities for CJE.

This module contains all calibration functionality:
- Judge score calibration to match oracle labels
- Dataset calibration workflows
"""

from .judge import (
    JudgeCalibrator,
    CalibrationResult,
    resolve_n_folds,
)
from .dataset import (
    calibrate_dataset,
)

__all__ = [
    # Judge calibration
    "JudgeCalibrator",
    "CalibrationResult",
    "resolve_n_folds",
    # Dataset calibration
    "calibrate_dataset",
]
