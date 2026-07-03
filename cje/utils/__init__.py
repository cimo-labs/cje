"""Utility functions for CJE.

This module contains:
- Export helpers for results (JSON/CSV)
- Diagnostics display re-exports (see cje.diagnostics)
"""

# Display utilities moved to cje.diagnostics
# Keeping this import for backward compatibility
from ..diagnostics.display import (
    create_weight_summary_table,
)

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

__all__ = [
    # Diagnostics display
    "create_weight_summary_table",
]

if _visualization_available:
    __all__.extend(
        [
            # Visualization (re-exported for backward compatibility)
            "plot_calibration_comparison",
            "plot_policy_estimates",
        ]
    )
