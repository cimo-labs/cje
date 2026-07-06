"""CJE Diagnostics System.

Consolidated module for all diagnostic functionality:
- Data models (DirectDiagnostics)
- Transportability auditing
- Reward boundary / coverage badge
- Display utilities
- Robust inference tools
- Budget planning
"""

# Data models. IPSDiagnostics is a DEPRECATED alias of DirectDiagnostics
# (0.3.x name); it will be removed in 0.5.0.
from .models import (
    DirectDiagnostics,
    IPSDiagnostics,
    Status,
)

# Canonical gate thresholds and status helpers (single source of truth)
from .gates import (
    OUT_OF_RANGE_REFUSE_THRESHOLD,
    worst_status,
)

# Transport diagnostics
from .transport import (
    TransportDiagnostics,
    audit_transportability,
    compute_residuals,
    plot_transport_comparison,
)

# Display utilities
from .display import (
    format_diagnostic_comparison,
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
    simulate_planning_sweep,
    correlation_to_r2,
)

__all__ = [
    # Data models (IPSDiagnostics is a deprecated alias, removed in 0.5.0)
    "DirectDiagnostics",
    "IPSDiagnostics",
    "Status",
    # Canonical gates
    "OUT_OF_RANGE_REFUSE_THRESHOLD",
    "worst_status",
    # Transport
    "TransportDiagnostics",
    "audit_transportability",
    "compute_residuals",
    "plot_transport_comparison",
    # Display
    "format_diagnostic_comparison",
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
    "simulate_planning_sweep",
    "correlation_to_r2",
]
