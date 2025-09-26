"""Utility functions for diagnostics.

This module contains:
- Weight Diagnostics: Debug importance sampling issues
- Visualization: Plotting utilities for weight diagnostics
"""

# Display utilities moved to cje.diagnostics
# Keeping this import for backward compatibility
from ..diagnostics.display import (
    create_weight_summary_table,
)

from .extreme_weights_analysis import (
    analyze_extreme_weights,
)

# Import visualization functions if matplotlib is available
# Note: visualization functions have moved to cje.visualization module
try:
    from ..visualization import (
        plot_weight_dashboard_summary,
        plot_weight_dashboard_detailed,
        plot_calibration_comparison,
        plot_policy_estimates,
    )

    _visualization_available = True
except ImportError:
    _visualization_available = False

__all__ = [
    # Weight diagnostics
    "create_weight_summary_table",
    # Extreme weights analysis
    "analyze_extreme_weights",
]

if _visualization_available:
    __all__.extend(
        [
            # Visualization (re-exported for backward compatibility)
            "plot_weight_dashboard_summary",
            "plot_weight_dashboard_detailed",
            "plot_calibration_comparison",
            "plot_policy_estimates",
        ]
    )
