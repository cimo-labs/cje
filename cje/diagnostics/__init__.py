"""CJE Diagnostics System.

Consolidated module for all diagnostic functionality:
- Data models (DirectDiagnostics)
- Transportability auditing
- Reward boundary / coverage badge
- Robust inference tools
- Budget planning
"""

from typing import Any

from .models import (
    DirectDiagnostics,
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
    TransportAuditConfig,
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
    # Data models
    "DirectDiagnostics",
    "Status",
    # Canonical gates
    "BOUNDARY_CARD_STATUS_TO_STATUS",
    "OUT_OF_RANGE_REFUSE_THRESHOLD",
    "SATURATION_CAUTION_THRESHOLD",
    "TRANSPORT_FAIL_DELTA_THRESHOLD",
    "TRANSPORT_STATUS_TO_STATUS",
    "worst_status",
    # Transport
    "TransportAuditConfig",
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
    if name == "compare_policies_bootstrap":
        raise ImportError(
            "cje.diagnostics.compare_policies_bootstrap was removed in "
            "0.5.0 — use EstimationResult.compare_policies(i, j) for "
            "pairwise claims, or compute paired contrasts from the "
            "bootstrap_matrix returned by cluster_bootstrap_direct_with_refit."
        )
    if name == "IPSDiagnostics":
        raise ImportError(
            "cje.diagnostics.IPSDiagnostics was removed in 0.6.0; "
            "use DirectDiagnostics. OPE estimators remain available only "
            "on the frozen cje-eval 0.3.x line."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
