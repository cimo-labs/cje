"""Utility functions for CJE.

This module contains:
- Export helpers for results (JSON/CSV)
- Visualization re-exports (see cje.visualization)
"""

# Import visualization functions if matplotlib is available
# Note: visualization functions have moved to cje.visualization module
try:
    from ..visualization import (
        plot_calibration_comparison,
        plot_policy_estimates,
    )

    _visualization_available = True
except ImportError:
    _visualization_available = False

__all__: list = []

if _visualization_available:
    __all__.extend(
        [
            # Visualization (re-exported for backward compatibility)
            "plot_calibration_comparison",
            "plot_policy_estimates",
        ]
    )
