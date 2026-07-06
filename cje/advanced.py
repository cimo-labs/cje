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
    JudgeCalibrator,
    CalibrationResult,
)

# Diagnostics (IPSDiagnostics is a deprecated alias slated for removal)
from .diagnostics import (
    DirectDiagnostics,
    IPSDiagnostics,
    Status,
)

# Utilities
from .utils.export import (
    export_results_json,
)

__all__ = [
    # Estimators
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
    "JudgeCalibrator",
    "CalibrationResult",
    # Diagnostics
    "DirectDiagnostics",
    "IPSDiagnostics",
    "Status",
    # Utilities
    "export_results_json",
]


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

# Visualization (viz extra) — resolved lazily so `import cje.advanced` never
# loads matplotlib; access without the extra raises the standard install
# hint (from cje.visualization).
_VIZ_EXPORTS = ("plot_policy_estimates",)


def __getattr__(name: str) -> object:
    if name in _REMOVED_IN_0_4_0:
        raise ImportError(
            f"cje.advanced.{name} was removed in 0.4.0 — "
            f'pip install "cje-eval==0.3.*" for OPE'
        )
    if name == "BaseCJEEstimator":
        raise ImportError(
            "cje.advanced.BaseCJEEstimator was removed in 0.5.0 — "
            "CalibratedDirectEstimator is the only estimator (the base "
            "class had exactly one subclass and was merged into it)."
        )
    if name in _VIZ_EXPORTS:
        from . import visualization

        return getattr(visualization, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Advertise the plot_* names in __all__ only when the viz extra is
# installed (checked without importing matplotlib, which is slow).
from importlib.util import find_spec as _find_spec

if _find_spec("matplotlib") is not None:
    __all__.extend(_VIZ_EXPORTS)
