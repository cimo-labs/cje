"""Calibration utilities for CJE.

This module contains all calibration functionality:
- Optimized isotonic regression for weight calibration
- Judge score calibration to match oracle labels
- Dataset calibration workflows
- Oracle slice uncertainty augmentation
"""

from .isotonic import (
    calibrate_to_target_mean,
)
from .simcal import (
    SIMCalibrator,
    SimcalConfig,
)
from .judge import (
    JudgeCalibrator,
    calibrate_judge_scores,
    CalibrationResult,
)
from .dataset import (
    calibrate_dataset,
    calibrate_from_raw_data,
)
from .oracle_slice import (
    OracleSliceAugmentation,
    OracleSliceConfig,
)

__all__ = [
    # Isotonic regression utilities
    "calibrate_to_target_mean",
    # SIMCal
    "SIMCalibrator",
    "SimcalConfig",
    # Judge calibration
    "JudgeCalibrator",
    "calibrate_judge_scores",
    "CalibrationResult",
    # Dataset calibration
    "calibrate_dataset",
    "calibrate_from_raw_data",
    # Oracle slice augmentation
    "OracleSliceAugmentation",
    "OracleSliceConfig",
]
