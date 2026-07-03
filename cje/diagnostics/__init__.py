"""CJE Diagnostics System.

Consolidated module for all diagnostic functionality:
- Data models (IPSDiagnostics)
- Transportability auditing
- Reward boundary / coverage badge
- Display utilities
- Robust inference tools
- Budget planning
"""

# Data models
from .models import (
    IPSDiagnostics,
    DRDiagnostics,
    CJEDiagnostics,
    Status,
    GateState,
)

# Canonical gate thresholds and status helpers (single source of truth)
from .gates import (
    OUT_OF_RANGE_REFUSE_THRESHOLD,
    ESS_GOOD_THRESHOLD,
    ESS_WARNING_THRESHOLD,
    ess_status,
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
    create_weight_summary_table,
    format_dr_diagnostic_summary,
    format_diagnostic_comparison,
)

# Robust inference
from .robust_inference import (
    DirectEvalTable,
    build_direct_eval_table,
    make_calibrator_factory,
    cluster_bootstrap_direct_with_refit,
    compare_policies_bootstrap,
    cluster_robust_se,
)

# Reward boundary / coverage badge (paper's REFUSE-LEVEL gate)
from .reward_boundary import (
    BoundaryCard,
    boundary_card,
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
    # Data models
    "IPSDiagnostics",
    "DRDiagnostics",
    "CJEDiagnostics",
    "Status",
    "GateState",
    # Canonical gates
    "OUT_OF_RANGE_REFUSE_THRESHOLD",
    "ESS_GOOD_THRESHOLD",
    "ESS_WARNING_THRESHOLD",
    "ess_status",
    "worst_status",
    # Transport
    "TransportDiagnostics",
    "audit_transportability",
    "compute_residuals",
    "plot_transport_comparison",
    # Display
    "create_weight_summary_table",
    "format_dr_diagnostic_summary",
    "format_diagnostic_comparison",
    # Robust inference
    "DirectEvalTable",
    "build_direct_eval_table",
    "make_calibrator_factory",
    "cluster_bootstrap_direct_with_refit",
    "compare_policies_bootstrap",
    "cluster_robust_se",
    # Reward boundary / coverage badge
    "BoundaryCard",
    "boundary_card",
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
