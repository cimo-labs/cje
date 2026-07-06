"""Advanced CJE API for power users.

This module exposes additional functionality for users who need more control.
Import from here when you need to:
- Use specific estimators directly
- Customize calibration behavior
- Access diagnostic tools
- Build custom pipelines

Example:
    from cje.advanced import (
        CalibratedDirectEstimator,
        calibrate_dataset,
        DirectDiagnostics,
    )

    # Custom pipeline with manual control
    dataset = load_dataset_from_jsonl("labels.jsonl")
    calibrated, cal_result = calibrate_dataset(dataset)
    estimator = CalibratedDirectEstimator(
        target_policies=["policy_a"],
        reward_calibrator=cal_result.calibrator,
    )
    estimator.add_fresh_draws("policy_a", fresh_draws)
    results = estimator.fit_and_estimate()
"""

# Estimators
from .estimators import (
    BaseCJEEstimator,
    CalibratedDirectEstimator,
)

# Data components
from .data import (
    Dataset,
    Sample,
    EstimationResult,
)
from .data.fresh_draws import (
    FreshDrawDataset,
    load_fresh_draws_from_jsonl,
    load_fresh_draws_auto,
)

# Calibration
from .calibration import (
    calibrate_dataset,
    calibrate_judge_scores,
    JudgeCalibrator,
    CalibrationResult,
)

# Diagnostics (IPSDiagnostics is a deprecated alias, removed in 0.5.0)
from .diagnostics import (
    DirectDiagnostics,
    IPSDiagnostics,
    Status,
)

# Utilities
from .utils.export import (
    export_results_json,
    export_results_csv,
)

# Visualization (if available)
try:
    from .visualization import (
        plot_calibration_comparison,
        plot_policy_estimates,
    )

    _viz_available = True
except ImportError:
    _viz_available = False

__all__ = [
    # Estimators
    "BaseCJEEstimator",
    "CalibratedDirectEstimator",
    # Data
    "Dataset",
    "Sample",
    "EstimationResult",
    "FreshDrawDataset",
    "load_fresh_draws_from_jsonl",
    "load_fresh_draws_auto",
    # Calibration
    "calibrate_dataset",
    "calibrate_judge_scores",
    "JudgeCalibrator",
    "CalibrationResult",
    # Diagnostics
    "DirectDiagnostics",
    "IPSDiagnostics",
    "Status",
    # Utilities
    "export_results_json",
    "export_results_csv",
]

if _viz_available:
    __all__.extend(
        [
            "plot_calibration_comparison",
            "plot_policy_estimates",
        ]
    )


# OPE classes removed in 0.4.0. A module __getattr__ turns
# `from cje.advanced import CalibratedIPS` (and plain attribute access)
# into an informative ImportError instead of a bare AttributeError.
_REMOVED_IN_0_4_0 = (
    "CalibratedIPS",
    "PrecomputedSampler",
    "DRCPOEstimator",
    "MRDREstimator",
    "TMLEEstimator",
    "StackedDREstimator",
)


def __getattr__(name: str) -> object:
    if name in _REMOVED_IN_0_4_0:
        raise ImportError(
            f"cje.advanced.{name} was removed in 0.4.0 — "
            f'pip install "cje-eval==0.3.*" for OPE'
        )
    if name in ("plot_calibration_comparison", "plot_policy_estimates"):
        # Only reached when the viz imports above failed (matplotlib missing)
        raise ImportError(
            f"{name} requires the viz extra. "
            f'Install with: pip install "cje-eval[viz]"'
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
