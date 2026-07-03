"""Visualization utilities for CJE framework.

This module provides plotting functions organized by domain:
- Calibration comparison plots
- Policy estimate visualizations
- Planning dashboards

Note: Requires optional 'viz' dependencies (matplotlib, seaborn).
Install with: pip install "cje-eval[viz]"

Without them, importing this module succeeds silently (no import-time
warning) and accessing any plot_* name raises an ImportError with the
install hint.
"""

from typing import Optional

_VIZ_EXPORTS = (
    "plot_calibration_comparison",
    "plot_policy_estimates",
    "plot_planning_dashboard",
)

__all__ = list(_VIZ_EXPORTS)

# Handle optional matplotlib/seaborn dependencies
try:
    # Import core visualization functions
    from .calibration import plot_calibration_comparison
    from .estimates import plot_policy_estimates

    # Import planning visualizations
    from .planning import plot_planning_dashboard

    _VIZ_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as e:
    _VIZ_IMPORT_ERROR = e


def __getattr__(name: str) -> object:
    """Raise an actionable hint for plot_* access on a no-viz install.

    Only reached when the imports above failed (otherwise the names are
    real module attributes); previously a no-viz install warned at import
    time and then raised a bare AttributeError on access.
    """
    if name in _VIZ_EXPORTS:
        raise ImportError(
            f"{name} requires the viz extra. "
            f'Install with: pip install "cje-eval[viz]"'
        ) from _VIZ_IMPORT_ERROR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
