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


# Raw-array calibration helpers removed in 0.5.0. A module __getattr__
# turns `from cje.calibration import calibrate_from_raw_data` (and plain
# attribute access) into an informative ImportError instead of a bare
# AttributeError.
_REMOVED_IN_0_5_0 = (
    "calibrate_from_raw_data",
    "calibrate_judge_scores",
)


def __getattr__(name: str) -> object:
    if name in _REMOVED_IN_0_5_0:
        raise ImportError(
            f"cje.calibration.{name} was removed in 0.5.0 — use "
            "calibrate_dataset or JudgeCalibrator.fit_cv (or "
            "cje.calibrated_mean_ci for a one-call calibrated mean + CI "
            "from raw arrays)."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
