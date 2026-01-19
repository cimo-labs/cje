"""CJE: Causal Judge Evaluation - Unbiased LLM Policy Evaluation.

Simple API for off-policy evaluation with judge scores.

Example:
    from cje import analyze_dataset

    results = analyze_dataset(
        "data.jsonl",
        estimator="calibrated-ips",
    )
    print(results.summary())
"""

__version__ = "0.2.5"

# Simple API - what 90% of users need
from .interface import analyze_dataset

# Core data structures
from .data import Dataset, Sample, EstimationResult

# Simple data loading
from .data import load_dataset_from_jsonl

# Budget planning
from .diagnostics.planning import (
    CostModel,
    FittedVarianceModel,
    EvaluationPlan,
    fit_variance_model,
    plan_evaluation,
    plan_for_mde,
)

# Visualization functions (optional - requires matplotlib)
try:
    from .visualization import (
        plot_policy_estimates,
        plot_calibration_comparison,
        plot_weight_dashboard_summary,
        plot_weight_dashboard_detailed,
        plot_dr_dashboard,
    )

    _has_visualization = True
except ImportError:
    _has_visualization = False

__all__ = [
    # Simple API
    "analyze_dataset",
    # Core data structures
    "Dataset",
    "Sample",
    "EstimationResult",
    # Data loading
    "load_dataset_from_jsonl",
    # Budget planning
    "CostModel",
    "FittedVarianceModel",
    "EvaluationPlan",
    "fit_variance_model",
    "plan_evaluation",
    "plan_for_mde",
]

# Add visualization functions to __all__ if available
if _has_visualization:
    __all__.extend(
        [
            "plot_policy_estimates",
            "plot_calibration_comparison",
            "plot_weight_dashboard_summary",
            "plot_weight_dashboard_detailed",
            "plot_dr_dashboard",
        ]
    )
