"""Visualization utilities for CJE framework.

This module provides plotting functions organized by domain:
- Calibration comparison plots
- Policy estimate visualizations
- Planning dashboards

Note: Requires optional 'viz' dependencies (matplotlib, seaborn).
Install with: pip install cje-eval[viz]
"""

# Handle optional matplotlib/seaborn dependencies
try:
    # Import core visualization functions
    from .calibration import plot_calibration_comparison
    from .estimates import plot_policy_estimates

    # Import planning visualizations
    from .planning import plot_planning_dashboard

    __all__ = [
        # Calibration
        "plot_calibration_comparison",
        # Policy estimates
        "plot_policy_estimates",
        # Planning
        "plot_planning_dashboard",
    ]

except ImportError:
    import warnings

    warnings.warn(
        "Visualization functions require optional dependencies. "
        "Install with: pip install cje-eval[viz]",
        ImportWarning,
    )

    __all__ = []
