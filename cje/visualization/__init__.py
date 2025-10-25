"""Visualization utilities for CJE framework.

This module provides plotting functions organized by domain:
- Weight diagnostics and dashboards
- DR diagnostics and dashboards
- Calibration comparison plots
- Policy estimate visualizations

Note: Requires optional 'viz' dependencies (matplotlib, seaborn).
Install with: pip install cje-eval[viz]
"""

# Handle optional matplotlib/seaborn dependencies
try:
    # Import core visualization functions
    from .calibration import plot_calibration_comparison
    from .estimates import plot_policy_estimates

    # Import weight dashboards
    from .weight_dashboards import (
        plot_weight_dashboard_summary,
        plot_weight_dashboard_detailed,
    )

    # Import DR dashboards
    from .dr_dashboards import plot_dr_dashboard

    # Import transport diagnostics
    from .transport import (
        plot_transport_audit,
        plot_transport_comparison,
    )

    __all__ = [
        # Calibration
        "plot_calibration_comparison",
        # Policy estimates
        "plot_policy_estimates",
        # Weight dashboards
        "plot_weight_dashboard_summary",
        "plot_weight_dashboard_detailed",
        # DR dashboards
        "plot_dr_dashboard",
        # Transport
        "plot_transport_audit",
        "plot_transport_comparison",
    ]

except ImportError as e:
    import warnings

    warnings.warn(
        "Visualization functions require optional dependencies. "
        "Install with: pip install cje-eval[viz]",
        ImportWarning,
    )

    __all__ = []
