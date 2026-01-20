"""Policy estimate visualization utilities."""

from pathlib import Path
from typing import Dict, List, Optional, Literal
import numpy as np
import matplotlib.pyplot as plt


def plot_policy_estimates(
    estimates: Dict[str, float],
    standard_errors: Dict[str, float],
    oracle_values: Optional[Dict[str, float]] = None,
    policy_labels: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    sort_by: Literal["estimate", "name", "none"] = "estimate",
    figsize: tuple = (12, None),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create forest plot of policy performance estimates with confidence intervals.

    Shows policy estimates as a forest plot with optional oracle comparison.
    By default, policies are sorted by estimate (best at top).

    Args:
        estimates: Dict mapping policy names to estimates
        standard_errors: Dict mapping policy names to standard errors
        oracle_values: Optional dict of oracle ground truth values
        policy_labels: Optional dict mapping policy names to display labels.
            Example: {"prompt_v1": "Conversational tone"}
        title: Optional plot title.
        sort_by: How to order policies. "estimate" (best at top, default),
            "name" (alphabetical), or "none" (preserve input order)
        figsize: Figure size (width, height). Height auto-calculated if None.
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    # Determine policy order
    if sort_by == "estimate":
        policies = sorted(estimates.keys(), key=lambda p: estimates[p], reverse=True)
    elif sort_by == "name":
        policies = sorted(estimates.keys())
    else:
        policies = list(estimates.keys())

    # Auto-calculate height based on number of policies
    n_policies = len(policies)
    height = figsize[1] if figsize[1] is not None else max(3, n_policies * 1.1 + 1)
    fig, ax = plt.subplots(figsize=(figsize[0], height))

    y_positions = np.arange(n_policies)[::-1]  # Reverse so first policy is at top

    # Alternating row shading
    for i, y in enumerate(y_positions):
        if i % 2 == 0:
            ax.axhspan(y - 0.4, y + 0.4, color="#f3f4f6", zorder=0)

    # Light grid
    ax.grid(True, axis="x", alpha=0.3, linestyle="-", linewidth=0.5, zorder=1)

    # Colors
    color_estimate = "#1f2937"  # Dark gray/black
    color_oracle = "#dc2626"  # Red

    # Plot each policy
    for i, policy in enumerate(policies):
        y = y_positions[i]
        est = estimates[policy]
        se = standard_errors[policy]

        # Confidence interval
        ci_lower = est - 1.96 * se
        ci_upper = est + 1.96 * se

        # Black CI line
        ax.plot(
            [ci_lower, ci_upper],
            [y, y],
            color=color_estimate,
            linewidth=2.5,
            solid_capstyle="round",
            zorder=3,
        )

        # Black estimate point
        ax.scatter(
            est,
            y,
            color=color_estimate,
            s=90,
            zorder=5,
            edgecolors="white",
            linewidth=1.5,
        )

        # Red oracle diamond if available
        if oracle_values and policy in oracle_values:
            oracle_val = oracle_values[policy]
            ax.scatter(
                oracle_val,
                y,
                color=color_oracle,
                s=70,
                marker="D",
                zorder=4,
                edgecolors="white",
                linewidth=1,
            )

    # Labels
    ax.set_yticks(y_positions)
    display_labels = [policy_labels.get(p, p) if policy_labels else p for p in policies]
    ax.set_yticklabels(display_labels, fontsize=11)
    ax.set_xlabel("Estimated Performance", fontsize=11, color="#374151")

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    # Add value annotations on right
    x_min, x_max = ax.get_xlim()
    x_padding = (x_max - x_min) * 0.28
    ax.set_xlim(x_min, x_max + x_padding)

    for i, policy in enumerate(policies):
        y = y_positions[i]
        est = estimates[policy]
        se = standard_errors[policy]
        ci_lower = est - 1.96 * se
        ci_upper = est + 1.96 * se

        # Annotation: estimate [CI]
        annotation = f"{est:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]"
        ax.text(
            x_max + x_padding * 0.08,
            y,
            annotation,
            va="center",
            ha="left",
            fontsize=9,
            fontfamily="monospace",
            color="#374151",
        )

    # Legend
    legend_elements = [
        plt.scatter([], [], color=color_estimate, s=60, label="Calibrated estimate"),
    ]
    if oracle_values:
        legend_elements.append(
            plt.scatter(
                [],
                [],
                color=color_oracle,
                s=50,
                marker="D",
                label="Oracle ground truth",
            )
        )
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)

    # Add RMSE if oracle values available (position in upper right to avoid overlap)
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
                0.98,
                f"RMSE vs Oracle: {rmse:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=9,
                color="#6b7280",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    # Title
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color="#1f2937", pad=12)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        fig.savefig(
            save_path.with_suffix(".png"),
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )

    return fig
