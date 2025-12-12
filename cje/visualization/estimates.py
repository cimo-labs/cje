"""Policy estimate visualization utilities."""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_policy_estimates(
    estimates: Dict[str, float],
    standard_errors: Dict[str, float],
    oracle_values: Optional[Dict[str, float]] = None,
    base_policy: Optional[str] = "base",
    policy_labels: Optional[Dict[str, str]] = None,
    figsize: tuple = (10, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create forest plot of policy performance estimates with confidence intervals.

    Shows policy estimates as a forest plot with optional oracle comparison.

    Args:
        estimates: Dict mapping policy names to estimates
        standard_errors: Dict mapping policy names to standard errors
        oracle_values: Optional dict of oracle ground truth values
        base_policy: Name of base policy (for reference line), or None for no base
        policy_labels: Optional dict mapping policy names to display labels.
            Example: {"prompt_v1": "Conversational tone"}
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort policies: base first, then others alphabetically
    policies = []
    if base_policy in estimates:
        policies.append(base_policy)
    for p in sorted(estimates.keys()):
        if p != base_policy:
            policies.append(p)

    y_positions = np.arange(len(policies))[::-1]  # Reverse so first policy is at top

    # Identify best policy (excluding base)
    non_base_policies = [p for p in policies if p != base_policy]
    if non_base_policies:
        best_policy = max(non_base_policies, key=lambda p: estimates[p])
    else:
        best_policy = None

    # Modern color palette
    color_base = "#6b7280"  # Gray
    color_best = "#10b981"  # Green
    color_default = "#3b82f6"  # Blue
    color_oracle = "#ef4444"  # Red

    # Plot each policy
    for i, policy in enumerate(policies):
        y = y_positions[i]
        est = estimates[policy]
        se = standard_errors[policy]

        # Confidence interval
        ci_lower = est - 1.96 * se
        ci_upper = est + 1.96 * se

        # Determine color
        if policy == base_policy:
            color = color_base
        elif policy == best_policy:
            color = color_best
        else:
            color = color_default

        # Plot CI line
        ax.plot(
            [ci_lower, ci_upper],
            [y, y],
            color=color,
            linewidth=2.5,
            solid_capstyle="round",
        )

        # Plot estimate point
        ax.scatter(
            est, y, color=color, s=80, zorder=5, edgecolors="white", linewidth=1.5
        )

        # Add oracle value if available
        if oracle_values and policy in oracle_values:
            oracle_val = oracle_values[policy]
            ax.scatter(
                oracle_val,
                y,
                color=color_oracle,
                s=60,
                marker="d",
                zorder=4,
                edgecolors="white",
                linewidth=1,
            )

    # Add vertical line at base estimate
    if base_policy in estimates:
        ax.axvline(
            estimates[base_policy],
            color="#9ca3af",
            linestyle="--",
            linewidth=1.5,
        )

    # Labels and formatting
    ax.set_yticks(y_positions)
    # Use display labels if provided, otherwise use policy names
    display_labels = [policy_labels.get(p, p) if policy_labels else p for p in policies]
    ax.set_yticklabels(display_labels, fontsize=11)
    ax.set_xlabel("Estimated Performance", fontsize=11, color="#374151")

    # Add RMSE if oracle values available
    if oracle_values:
        squared_errors = []
        for policy in policies:
            if policy in oracle_values:
                error = estimates[policy] - oracle_values[policy]
                squared_errors.append(error**2)
        if squared_errors:
            rmse = np.sqrt(np.mean(squared_errors))
            ax.text(
                0.98,
                0.02,
                f"RMSE vs Oracle: {rmse:.3f}",
                transform=ax.transAxes,
                verticalalignment="bottom",
                horizontalalignment="right",
                fontsize=9,
                color="#6b7280",
            )

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    # Status labels on right (after spines so xlim is stable)
    x_min, x_max = ax.get_xlim()
    x_padding = (x_max - x_min) * 0.08
    ax.set_xlim(x_min, x_max + x_padding)  # Extend first

    label_x = x_max + x_padding * 0.3
    for i, policy in enumerate(policies):
        y = y_positions[i]
        if policy == best_policy:
            ax.text(
                label_x,
                y,
                "BEST",
                ha="left",
                va="center",
                fontsize=9,
                color=color_best,
                fontweight="bold",
                clip_on=False,
            )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")

    return fig
