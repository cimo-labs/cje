"""Advanced CJE API for power users.

This module exposes additional functionality for users who need more control.
Import from here when you need to:
- Use specific estimators directly
- Customize calibration behavior
- Access diagnostic tools
- Build custom pipelines

Example:
    from cje.advanced import (
        PrecomputedSampler,
        CalibratedIPS,
        calibrate_dataset,
        IPSDiagnostics
    )

    # Custom pipeline with manual control
    dataset = load_dataset_from_jsonl("data.jsonl")
    calibrated, cal_result = calibrate_dataset(dataset, oracle_coverage=0.1)
    sampler = PrecomputedSampler(calibrated)
    estimator = CalibratedIPS(sampler, var_cap=10.0)
    results = estimator.fit_and_estimate()
"""

# Estimators
from .estimators import (
    BaseCJEEstimator,
    CalibratedIPS,
)

# Data components
from .data import (
    PrecomputedSampler,
    Dataset,
    Sample,
    EstimationResult,
    DatasetFactory,
    default_factory,
)

# Calibration
from .calibration import (
    calibrate_dataset,
    calibrate_judge_scores,
    JudgeCalibrator,
    CalibrationResult,
)

# Diagnostics
from .diagnostics import (
    IPSDiagnostics,
    DRDiagnostics,
    Status,
)

# Utilities
from .utils import (
    create_weight_summary_table,
    analyze_extreme_weights,
)
from .utils.export import (
    export_results_json,
    export_results_csv,
)

# DR estimators (if available)
try:
    from .estimators.dr_base import DRCPOEstimator
    from .estimators.mrdr import MRDREstimator
    from .estimators.tmle import TMLEEstimator
    from .data.fresh_draws import (
        FreshDrawDataset,
        load_fresh_draws_from_jsonl,
        load_fresh_draws_auto,
    )

    _dr_available = True
except ImportError:
    _dr_available = False

# Visualization (if available)
try:
    from .visualization import (
        plot_weight_dashboard_summary,
        plot_calibration_comparison,
        plot_policy_estimates,
    )

    _viz_available = True
except ImportError:
    _viz_available = False

__all__ = [
    # Estimators
    "BaseCJEEstimator",
    "CalibratedIPS",
    # Data
    "PrecomputedSampler",
    "Dataset",
    "Sample",
    "EstimationResult",
    "DatasetFactory",
    "default_factory",
    # Calibration
    "calibrate_dataset",
    "calibrate_judge_scores",
    "JudgeCalibrator",
    "CalibrationResult",
    # Diagnostics
    "IPSDiagnostics",
    "DRDiagnostics",
    "Status",
    # Utilities
    "create_weight_summary_table",
    "analyze_extreme_weights",
    "export_results_json",
    "export_results_csv",
]

if _dr_available:
    __all__.extend(
        [
            "DRCPOEstimator",
            "MRDREstimator",
            "TMLEEstimator",
            "FreshDrawDataset",
            "load_fresh_draws_from_jsonl",
            "load_fresh_draws_auto",
        ]
    )

if _viz_available:
    __all__.extend(
        [
            "plot_weight_dashboard_summary",
            "plot_calibration_comparison",
            "plot_policy_estimates",
        ]
    )
