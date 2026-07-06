"""Visualization utilities for CJE framework.

This module provides plotting functions organized by domain:
- Policy estimate visualizations
- Planning dashboards
- Transport-audit plots

Note: Requires the optional 'viz' dependency (matplotlib).
Install with: pip install "cje-eval[viz]"

Without it, importing this module succeeds silently (no import-time
warning) and accessing any plot_* name raises an ImportError with the
install hint (produced by _require_viz, the single source for that hint).
"""

from typing import Optional

_VIZ_EXPORTS = (
    "plot_policy_estimates",
    "plot_planning_dashboard",
    "plot_transport_comparison",
)

__all__ = list(_VIZ_EXPORTS)


def _require_viz(name: str) -> None:
    """Raise the standard viz-extra ImportError for `name`.

    Single source for the install hint: every surface that gates on the viz
    extra (module __getattr__s, lazily-imported plot functions) calls this
    instead of hand-rolling its own ImportError copy. No-op when matplotlib
    is importable.
    """
    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"{name} requires the viz extra. "
            f'Install with: pip install "cje-eval[viz]"'
        ) from e


# Handle optional matplotlib/seaborn dependencies
try:
    # Import core visualization functions
    from .estimates import plot_policy_estimates

    # Import planning visualizations
    from .planning import plot_planning_dashboard

    # Transport plots gate lazily inside their functions, but bind them
    # eagerly here alongside the other names when the extra is installed
    from .transport import plot_transport_comparison

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
        _require_viz(name)
    if name == "plot_calibration_comparison":
        raise ImportError(
            "cje.visualization.plot_calibration_comparison was removed in "
            "0.5.0. Calibration quality lives in results.diagnostics "
            "(calibration_rmse, calibration_coverage); use "
            "plot_policy_estimates for estimate plots."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
