"""Calibration utilities for CJE.

This module contains all calibration functionality:
- Judge score calibration to match oracle labels
- Dataset calibration workflows
"""

from .judge import (
    JudgeCalibrator,
    calibrate_judge_scores,
    CalibrationResult,
)
from .dataset import (
    calibrate_dataset,
    calibrate_from_raw_data,
)

__all__ = [
    # Judge calibration
    "JudgeCalibrator",
    "calibrate_judge_scores",
    "CalibrationResult",
    # Dataset calibration
    "calibrate_dataset",
    "calibrate_from_raw_data",
]
