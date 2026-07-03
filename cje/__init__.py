"""CJE: Causal Judge Evaluation - Calibrated LLM Policy Evaluation.

Simple API for evaluating policies from fresh draws with judge scores.

Example:
    from cje import analyze_dataset

    results = analyze_dataset(fresh_draws_dir="responses/")
    print(results.summary())
"""

__version__ = "0.3.0"

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

# Simulation-based planning
from .diagnostics.simulation_planning import (
    SimulationPlanningResult,
    simulate_variance_model,
    simulate_planning,
    simulate_planning_sweep,
    correlation_to_r2,
)

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
    # Simulation-based planning
    "SimulationPlanningResult",
    "simulate_variance_model",
    "simulate_planning",
    "simulate_planning_sweep",
    "correlation_to_r2",
]

# Visualization functions (optional - requires the viz extra). Resolved
# lazily so a no-viz install imports cje without warnings; accessing a
# plot_* name without matplotlib raises an ImportError with the install
# hint (from cje.visualization) instead of an AttributeError.
_VIZ_EXPORTS = (
    "plot_policy_estimates",
    "plot_calibration_comparison",
    "plot_planning_dashboard",
)


def __getattr__(name: str) -> object:
    if name in _VIZ_EXPORTS:
        from . import visualization

        return getattr(visualization, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Advertise the plot_* names in __all__ only when the viz extra is
# installed (checked without importing matplotlib, which is slow).
from importlib.util import find_spec as _find_spec

if _find_spec("matplotlib") is not None:
    __all__.extend(_VIZ_EXPORTS)
