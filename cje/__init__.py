"""CJE: Causal Judge Evaluation - Calibrated LLM Policy Evaluation.

Simple API for evaluating policies from fresh draws with judge scores.

Example:
    from cje import analyze_dataset

    results = analyze_dataset(fresh_draws_dir="responses/")
    print(results.summary())
"""


# Single-source the version. Installed packages read the cje-eval
# distribution metadata (which poetry generates from pyproject.toml); source
# checkouts read the adjacent pyproject.toml directly, so a stale cje-eval
# dist elsewhere in the environment cannot shadow the checkout's version.
def _resolve_version() -> str:
    try:
        import re as _re
        from pathlib import Path as _Path

        _pyproject = _Path(__file__).resolve().parent.parent / "pyproject.toml"
        if _pyproject.exists():
            _match = _re.search(
                r'^version\s*=\s*"([^"]+)"', _pyproject.read_text(), _re.MULTILINE
            )
            if _match:
                return _match.group(1)
        from importlib.metadata import version as _dist_version

        return _dist_version("cje-eval")
    except Exception:  # PackageNotFoundError, or metadata unavailable
        return "0.5.0"


__version__ = _resolve_version()

# Simple API - what 90% of users need
from .interface import analyze_dataset

# Core data structures
from .data import Dataset, Sample, EstimationResult

# Simple data loading
from .data import load_dataset_from_jsonl

# Array-first primitive (single-policy calibrated mean + transport audit)
from .array_api import CalibratedMeanResult, calibrated_mean_ci, transport_audit

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
    # Array-first primitive
    "CalibratedMeanResult",
    "calibrated_mean_ci",
    "transport_audit",
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
    "correlation_to_r2",
]

# Visualization functions (optional - requires the viz extra). Resolved
# lazily so a no-viz install imports cje without warnings; accessing a
# plot_* name without matplotlib raises an ImportError with the install
# hint (from cje.visualization) instead of an AttributeError.
_VIZ_EXPORTS = (
    "plot_policy_estimates",
    "plot_planning_dashboard",
)


def __getattr__(name: str) -> object:
    if name in _VIZ_EXPORTS:
        from . import visualization

        return getattr(visualization, name)
    if name == "plot_calibration_comparison":
        raise ImportError(
            "cje.plot_calibration_comparison was removed in 0.5.0. "
            "Calibration quality lives in results.diagnostics "
            "(calibration_rmse, calibration_coverage); use "
            "plot_policy_estimates for estimate plots."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Advertise the plot_* names in __all__ only when the viz extra is
# installed (checked without importing matplotlib, which is slow).
from importlib.util import find_spec as _find_spec

if _find_spec("matplotlib") is not None:
    __all__.extend(_VIZ_EXPORTS)
