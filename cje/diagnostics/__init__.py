"""CJE Diagnostics System.

Consolidated module for all diagnostic functionality:
- Data models (DirectDiagnostics)
- Transportability auditing
- Reward boundary / coverage badge
- Robust inference tools
- Budget planning
"""

from typing import Any

# Data models. IPSDiagnostics is a DEPRECATED alias of DirectDiagnostics
# (0.3.x name); it will be removed in 0.5.0.
from .models import (
    DirectDiagnostics,
    IPSDiagnostics,
    Status,
)

# Canonical gate thresholds and status helpers (single source of truth)
from .gates import (
    BOUNDARY_CARD_STATUS_TO_STATUS,
    OUT_OF_RANGE_REFUSE_THRESHOLD,
    SATURATION_CAUTION_THRESHOLD,
    TRANSPORT_FAIL_DELTA_THRESHOLD,
    TRANSPORT_STATUS_TO_STATUS,
    worst_status,
)

# Transport diagnostics
from .transport import (
    TransportDiagnostics,
    audit_transportability,
    compute_residuals,
)

# Robust inference
from .robust_inference import (
    DirectEvalTable,
    build_direct_eval_table,
    make_calibrator_factory,
    cluster_bootstrap_direct_with_refit,
    cluster_robust_se,
)

# Reward boundary / coverage badge (paper's REFUSE-LEVEL gate)
from .reward_boundary import (
    BoundaryCard,
    boundary_card,
    boundary_card_dict,
)

# Budget optimization / Planning
from .planning import (
    CostModel,
    FittedVarianceModel,
    EvaluationPlan,
    fit_variance_model,
    plan_evaluation,
    plan_for_mde,
)

# Simulation-based planning
from .simulation_planning import (
    SimulationPlanningResult,
    simulate_variance_model,
    simulate_planning,
    correlation_to_r2,
)

__all__ = [
    # Data models (IPSDiagnostics is a deprecated alias, removed in 0.5.0)
    "DirectDiagnostics",
    "IPSDiagnostics",
    "Status",
    # Canonical gates
    "BOUNDARY_CARD_STATUS_TO_STATUS",
    "OUT_OF_RANGE_REFUSE_THRESHOLD",
    "SATURATION_CAUTION_THRESHOLD",
    "TRANSPORT_FAIL_DELTA_THRESHOLD",
    "TRANSPORT_STATUS_TO_STATUS",
    "worst_status",
    # Transport
    "TransportDiagnostics",
    "audit_transportability",
    "compute_residuals",
    "plot_transport_comparison",
    # Robust inference
    "DirectEvalTable",
    "build_direct_eval_table",
    "make_calibrator_factory",
    "cluster_bootstrap_direct_with_refit",
    "cluster_robust_se",
    # Reward boundary / coverage badge
    "BoundaryCard",
    "boundary_card",
    "boundary_card_dict",
    # Budget optimization / Planning
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


def __getattr__(name: str) -> Any:
    """Lazy re-export of the transport plot (moved to cje.visualization).

    `from cje.diagnostics import plot_transport_comparison` keeps working
    (the demo notebook uses it) without this package importing matplotlib;
    calling the function without the viz extra raises the install hint.
    """
    if name == "plot_transport_comparison":
        from ..visualization.transport import plot_transport_comparison

        return plot_transport_comparison
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
