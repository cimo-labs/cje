"""Policy estimate visualization utilities."""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_policy_estimates(
    estimates: Dict[str, float],
    standard_errors: Dict[str, float],
    oracle_values: Optional[Dict[str, float]] = None,
    base_policy: str = "base",
    figsize: tuple = (10, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create forest plot of policy performance estimates with confidence intervals.

    Shows policy estimates as a forest plot with optional oracle comparison.

    Args:
        estimates: Dict mapping policy names to estimates
        standard_errors: Dict mapping policy names to standard errors
        oracle_values: Optional dict of oracle ground truth values
        base_policy: Name of base policy (for reference line)
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
            color = "gray"
            marker = "s"  # Square for base
        elif policy == best_policy:
            color = "green"
            marker = "o"
        else:
            color = "steelblue"
            marker = "o"

        # Plot CI line
        ax.plot([ci_lower, ci_upper], [y, y], color=color, linewidth=2, alpha=0.7)

        # Plot estimate point
        ax.scatter(est, y, color=color, s=100, marker=marker, zorder=5, label=None)

        # Add oracle value if available
        if oracle_values and policy in oracle_values:
            oracle_val = oracle_values[policy]
            ax.scatter(
                oracle_val,
                y,
                color="red",
                s=100,
                marker="d",
                alpha=0.7,
                zorder=4,
                label="Oracle Truth" if i == 0 else None,
            )

    # Add vertical line at base estimate
    if base_policy in estimates:
        ax.axvline(
            estimates[base_policy],
            color="gray",
            linestyle=":",
            alpha=0.5,
            label=f"{base_policy} (reference)",
        )

    # Labels and formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(policies)
    ax.set_xlabel("Estimated Performance")
    ax.set_title("Policy Performance Estimates (95% CI)")
    ax.grid(True, alpha=0.3, axis="x")

    # Add RMSE if oracle values available
    if oracle_values:
        # Calculate RMSE vs oracle
        squared_errors = []
        for policy in policies:
            if policy in oracle_values:
                error = estimates[policy] - oracle_values[policy]
                squared_errors.append(error**2)
        if squared_errors:
            rmse = np.sqrt(np.mean(squared_errors))
            ax.text(
                0.02,
                0.98,
                f"RMSE vs Oracle: {rmse:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    # Legend
    handles = [
        plt.Line2D(
            [0], [0], color="steelblue", marker="o", linestyle="", label="CJE Estimate"
        ),
    ]
    if oracle_values:
        handles.append(
            plt.Line2D(
                [0], [0], color="red", marker="d", linestyle="", label="Oracle Truth"
            )
        )
    if best_policy:
        handles.append(
            plt.Line2D(
                [0], [0], color="green", marker="o", linestyle="", label="Best Policy"
            )
        )

    ax.legend(handles=handles, loc="lower right", fontsize=9)

    # Adjust layout
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")

    return fig
