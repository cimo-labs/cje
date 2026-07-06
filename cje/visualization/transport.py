"""Transport-audit visualizations (requires the viz extra).

Moved here from cje.diagnostics.transport in 0.5.0 so diagnostics stay
matplotlib-free. This module imports matplotlib lazily: importing it (and
`from cje.diagnostics import plot_transport_comparison`, which re-exports
lazily) works without the viz extra; calling the plot functions raises the
standard install hint via _require_viz.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from . import _require_viz

if TYPE_CHECKING:
    from ..diagnostics.transport import TransportDiagnostics


def plot_transport_diagnostics(
    diagnostics: "TransportDiagnostics",
    ax: Optional[Any] = None,
    figsize: tuple = (10, 5),
) -> Any:
    """Plot one transport audit: decile residuals with overall mean and CI.

    Backs TransportDiagnostics.plot().

    Args:
        diagnostics: TransportDiagnostics to render.
        ax: Optional matplotlib axes. If None, creates new figure.
        figsize: Figure size if creating new figure.

    Returns:
        matplotlib figure object
    """
    _require_viz("TransportDiagnostics.plot")
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    n_bins = len(diagnostics.decile_residuals)
    x = np.arange(n_bins)

    # Filter out NaN values for plotting
    residuals = np.array(diagnostics.decile_residuals)
    counts = np.array(diagnostics.decile_counts)
    valid_mask = ~np.isnan(residuals)

    # Modern color palette (positive=green, negative=red)
    colors = ["#ef4444" if r < 0 else "#10b981" for r in residuals[valid_mask]]

    # Plot decile bars
    ax.bar(
        x[valid_mask],
        residuals[valid_mask],
        color=colors,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )

    # Add overall mean line with CI band
    ax.axhline(
        y=diagnostics.delta_hat,
        color="#374151",
        linewidth=2,
        linestyle="-",
    )
    ax.axhspan(
        diagnostics.delta_ci[0],
        diagnostics.delta_ci[1],
        alpha=0.15,
        color="#6b7280",
    )

    # Zero line
    ax.axhline(y=0, color="#9ca3af", linewidth=1.5, linestyle="--")

    # Labels
    title = f"Transportability: {diagnostics.status}"
    if diagnostics.group_label:
        title += f" ({diagnostics.group_label})"
    ax.set_title(title, fontsize=12, fontweight="bold", color="#111827")
    ax.set_xlabel("Score Decile", fontsize=10, color="#374151")
    ax.set_ylabel("Mean Residual (Y − Ŷ)", fontsize=10, color="#374151")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i+1}" for i in range(len(counts))], fontsize=9)

    # Status indicator with modern colors
    status_colors = {"PASS": "#10b981", "WARN": "#f59e0b", "FAIL": "#ef4444"}
    status_color = status_colors.get(diagnostics.status, "#6b7280")
    ax.text(
        0.02,
        0.98,
        diagnostics.status,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=status_color,
        verticalalignment="top",
    )

    # Stats text
    ax.text(
        0.98,
        0.98,
        f"δ̂={diagnostics.delta_hat:+.3f}  "
        f"CI=[{diagnostics.delta_ci[0]:+.2f}, {diagnostics.delta_ci[1]:+.2f}]",
        transform=ax.transAxes,
        fontsize=9,
        color="#6b7280",
        verticalalignment="top",
        horizontalalignment="right",
    )

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    return fig


def plot_transport_comparison(
    diagnostics: Dict[str, "TransportDiagnostics"],
    figsize: tuple = (10, 6),
    title: str = "Transportability Audit",
) -> Any:
    """Plot multiple transport diagnostics as a forest plot.

    Args:
        diagnostics: Dict mapping labels to TransportDiagnostics objects
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib figure object

    Example:
        >>> diag_clone = audit_transportability(calibrator, clone_probe, group_label="clone")
        >>> diag_unhelpful = audit_transportability(calibrator, unhelpful_probe, group_label="unhelpful")
        >>> fig = plot_transport_comparison({"clone": diag_clone, "unhelpful": diag_unhelpful})
        >>> plt.show()
    """
    _require_viz("plot_transport_comparison")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by mean residual (most negative at bottom)
    sorted_items = sorted(
        diagnostics.items(), key=lambda x: x[1].delta_hat, reverse=True
    )
    labels = [k for k, _ in sorted_items]
    diags = [v for _, v in sorted_items]

    y_pos = list(range(len(labels)))

    # Extract data
    means = [d.delta_hat for d in diags]
    ci_lowers = [d.delta_ci[0] for d in diags]
    ci_uppers = [d.delta_ci[1] for d in diags]
    statuses = [d.status for d in diags]

    # Modern color palette
    status_colors = {"PASS": "#10b981", "WARN": "#f59e0b", "FAIL": "#ef4444"}

    # Plot CI lines and point estimates
    for i, (y, diag) in enumerate(zip(y_pos, diags)):
        color = status_colors.get(diag.status, "#6b7280")

        # CI line
        ax.plot(
            [ci_lowers[i], ci_uppers[i]],
            [y, y],
            color=color,
            linewidth=2.5,
            solid_capstyle="round",
        )

        # Point estimate
        ax.scatter(
            [means[i]],
            [y],
            color=color,
            s=80,
            zorder=5,
            edgecolors="white",
            linewidth=1.5,
        )

    # Zero reference line
    ax.axvline(x=0, color="#9ca3af", linestyle="--", linewidth=1.5, zorder=0)

    # Y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)

    # X-axis
    ax.set_xlabel("Calibration Error (Y − Ŷ)", fontsize=11, color="#374151")

    # Status labels on right (outside axes)
    if ci_lowers and ci_uppers:
        x_max = max(abs(min(ci_lowers)), abs(max(ci_uppers))) * 1.15
    else:
        x_max = 0.1
    for i, (y, status) in enumerate(zip(y_pos, statuses)):
        color = status_colors.get(status, "#6b7280")
        ax.text(
            x_max,
            y,
            status,
            ha="left",
            va="center",
            fontsize=10,
            color=color,
            fontweight="bold",
            clip_on=False,
        )

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    # Set x limits with padding for status labels
    x_min = min(ci_lowers) - abs(min(ci_lowers)) * 0.1
    ax.set_xlim(x_min, x_max + x_max * 0.3)

    plt.tight_layout()
    return fig
