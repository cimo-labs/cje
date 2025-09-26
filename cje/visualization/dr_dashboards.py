"""Doubly Robust (DR) diagnostics dashboards for CJE framework.

Provides comprehensive visualization of DR estimation diagnostics including:
- Direct method vs IPS contributions
- Orthogonality checks
- Influence function tail behavior
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Any, Tuple, List


def plot_dr_dashboard(
    estimation_result: Any, figsize: Tuple[float, float] = (15, 5)
) -> Tuple[Figure, Dict[str, Any]]:
    """Create a compact 3-panel DR diagnostic dashboard.

    Panel A: DM vs IPS contributions per policy
    Panel B: Orthogonality check (score mean ± 2SE)
    Panel C: EIF tail behavior (CCDF)

    Args:
        estimation_result: Result from DR estimator with diagnostics
        figsize: Figure size (width, height)

    Returns:
        (fig, summary_metrics) tuple
    """
    # Check for DRDiagnostics object first (new way)
    from ..diagnostics import DRDiagnostics

    if isinstance(estimation_result.diagnostics, DRDiagnostics):
        # Use the new diagnostic object
        dr_diags = estimation_result.diagnostics.dr_diagnostics_per_policy
    elif "dr_diagnostics" in estimation_result.metadata:
        # Fallback to old way
        dr_diags = estimation_result.metadata["dr_diagnostics"]
    else:
        raise ValueError("No DR diagnostics found in estimation result")

    policies = list(dr_diags.keys())
    n_policies = len(policies)

    # Set up figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("DR Diagnostics Dashboard", fontsize=14, fontweight="bold")

    # Color palette
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, n_policies))

    # Panel A: DM vs IPS contributions
    _plot_dr_contributions(axes[0], dr_diags, policies, colors)

    # Panel B: Orthogonality check
    _plot_orthogonality_check(axes[1], dr_diags, policies, colors, estimation_result)

    # Panel C: EIF tail behavior
    _plot_eif_tail_behavior(axes[2], estimation_result, policies, colors)

    plt.tight_layout()

    # Compute summary metrics
    summary_metrics = _compute_dr_summary_metrics(dr_diags, estimation_result)

    return fig, summary_metrics


def _plot_dr_contributions(
    ax: Any, dr_diags: Dict, policies: List[str], colors: Any
) -> None:
    """Plot direct method vs IPS correction contributions."""
    n_policies = len(policies)
    x = np.arange(n_policies)
    width = 0.35

    dm_means = [dr_diags[p]["dm_mean"] for p in policies]
    ips_corrs = [dr_diags[p]["ips_corr_mean"] for p in policies]
    dr_estimates = [dr_diags[p]["dr_estimate"] for p in policies]

    # Bar plots for DM and IPS
    ax.bar(x - width / 2, dm_means, width, label="DM", color="steelblue", alpha=0.7)
    ax.bar(
        x + width / 2,
        ips_corrs,
        width,
        label="IPS Correction",
        color="coral",
        alpha=0.7,
    )

    # Add DR estimate markers
    ax.scatter(
        x, dr_estimates, color="black", s=50, zorder=5, label="DR Estimate", marker="D"
    )

    ax.set_xlabel("Policy")
    ax.set_ylabel("Value")
    ax.set_title("A: Contributions (DM vs IPS)")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)


def _plot_orthogonality_check(
    ax: Any, dr_diags: Dict, policies: List[str], colors: Any, estimation_result: Any
) -> None:
    """Plot orthogonality check with score means and confidence intervals."""
    n_policies = len(policies)

    # Check if we have score information (requires influence functions)
    has_scores = any("score_mean" in dr_diags[p] for p in policies)

    if not has_scores:
        # No scores available - show informative message
        ax.text(
            0.5,
            0.5,
            "Score test unavailable\n(influence functions not stored)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
            style="italic",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("B: Orthogonality Check (unavailable)")
    else:
        for i, policy in enumerate(policies):
            diag = dr_diags[policy]
            score_mean = diag.get("score_mean", 0.0)
            score_se = diag.get("score_se", 0.0)

            # Plot point with error bars (2 SE)
            ax.errorbar(
                i,
                score_mean,
                yerr=2 * score_se,
                fmt="o",
                color=colors[i],
                markersize=8,
                capsize=5,
                capthick=2,
                label=policy,
            )

            # Add p-value annotation
            p_val = diag.get("score_p", 1.0)
            if p_val < 0.05:
                ax.text(
                    i,
                    score_mean + 2.5 * score_se,
                    f"p={p_val:.3f}",
                    ha="center",
                    fontsize=8,
                    color="red",
                )

        ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax.set_xlabel("Policy")
        ax.set_ylabel("Score Mean")
        ax.set_title("B: Orthogonality Check (mean ± 2SE)")
        ax.set_xticks(range(n_policies))
        ax.set_xticklabels(policies, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)

    # Add note for TMLE
    if estimation_result.method == "tmle":
        ax.text(
            0.5,
            0.95,
            "TMLE: bars should straddle 0",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
            color="gray",
        )


def _plot_eif_tail_behavior(
    ax: Any, estimation_result: Any, policies: List[str], colors: Any
) -> None:
    """Plot empirical influence function tail behavior (CCDF)."""

    # Check if actual influence functions are available
    has_empirical_ifs = False
    ifs_data = None

    # First check the first-class location (new API)
    if estimation_result.influence_functions is not None:
        ifs_data = estimation_result.influence_functions
        if ifs_data and all(policy in ifs_data for policy in policies):
            has_empirical_ifs = True
    # Fallback to legacy location in metadata
    elif "dr_influence" in estimation_result.metadata:
        ifs_data = estimation_result.metadata["dr_influence"]
        if ifs_data and all(policy in ifs_data for policy in policies):
            has_empirical_ifs = True

    if has_empirical_ifs and ifs_data is not None:
        # Use empirical influence functions for exact CCDF
        for i, policy in enumerate(policies):
            ifs = ifs_data[policy]
            if isinstance(ifs, np.ndarray) and len(ifs) > 0:
                # Compute empirical CCDF
                abs_ifs = np.abs(ifs)
                sorted_ifs = np.sort(abs_ifs)[::-1]  # Descending
                ccdf = np.arange(1, len(sorted_ifs) + 1) / len(sorted_ifs)

                # Plot with appropriate sampling for large n
                if len(sorted_ifs) > 10000:
                    # Downsample for plotting efficiency
                    indices = np.logspace(
                        0, np.log10(len(sorted_ifs) - 1), 1000, dtype=int
                    )
                    ax.loglog(
                        sorted_ifs[indices],
                        ccdf[indices],
                        label=policy,
                        color=colors[i],
                        linewidth=2,
                    )
                else:
                    ax.loglog(
                        sorted_ifs, ccdf, label=policy, color=colors[i], linewidth=2
                    )

        # Add reference lines at p95 and p99
        ax.axhline(y=0.05, color="gray", linestyle=":", alpha=0.5, label="p95")
        ax.axhline(y=0.01, color="gray", linestyle="--", alpha=0.5, label="p99")
    else:
        # No influence functions available - show message
        ax.text(
            0.5,
            0.5,
            "Influence functions not available\n(set store_influence=True)",
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.set_xlabel("|IF| (log scale)")
    ax.set_ylabel("CCDF (log scale)")
    ax.set_title("C: EIF Tail Behavior")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)


def _compute_dr_summary_metrics(
    dr_diags: Dict, estimation_result: Any
) -> Dict[str, Any]:
    """Compute summary metrics from DR diagnostics."""
    summary_metrics = {}

    # IF tail ratios
    if_tail_ratios = [d.get("if_tail_ratio_99_5", 0.0) for d in dr_diags.values()]
    if if_tail_ratios:
        summary_metrics["worst_if_tail_ratio"] = max(if_tail_ratios)

    # R² values
    r2_values = [d.get("r2_oof", np.nan) for d in dr_diags.values() if "r2_oof" in d]
    if r2_values and not all(np.isnan(r2_values)):
        valid_r2 = [r for r in r2_values if not np.isnan(r)]
        if valid_r2:
            summary_metrics["best_r2_oof"] = max(valid_r2)
            summary_metrics["worst_r2_oof"] = min(valid_r2)

    # RMSE values
    rmse_values = [
        d.get("residual_rmse", np.nan)
        for d in dr_diags.values()
        if "residual_rmse" in d
    ]
    if rmse_values and not all(np.isnan(rmse_values)):
        valid_rmse = [r for r in rmse_values if not np.isnan(r)]
        if valid_rmse:
            summary_metrics["avg_residual_rmse"] = np.mean(valid_rmse)

    if estimation_result.method == "tmle":
        score_means = [abs(d.get("score_mean", 0.0)) for d in dr_diags.values()]
        if score_means:
            summary_metrics["tmle_max_abs_score"] = max(score_means)

    return summary_metrics
