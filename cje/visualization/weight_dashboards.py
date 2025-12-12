"""Weight diagnostics dashboards for CJE framework.

This module provides two complementary weight diagnostic visualizations:
1. plot_weight_dashboard_summary: 6-panel overview across all policies
2. plot_weight_dashboard_detailed: Individual panels per policy with judge score analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path

# Import shared utilities
from ..diagnostics import compute_ess


# ============================================================================
# Summary Dashboard (6-panel multi-policy overview)
# ============================================================================


def plot_weight_dashboard_summary(
    raw_weights_dict: Dict[str, np.ndarray],
    calibrated_weights_dict: Optional[Dict[str, np.ndarray]] = None,
    n_samples: Optional[int] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 12),
    random_seed: int = 42,
    diagnostics: Optional[Any] = None,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """Create comprehensive 6-panel weight diagnostics dashboard.

    Displays essential weight diagnostics across all policies:
    - Panel A: ESS comparison (raw vs calibrated)
    - Panel B: Maximum weight concentration
    - Panel C: Weight distributions
    - Panel D: Tail behavior (CCDF)
    - Panel E: Sample efficiency
    - Panel F: Summary table with recommendations

    Args:
        raw_weights_dict: Dict mapping policy names to raw weight arrays
        calibrated_weights_dict: Optional dict of calibrated weights
        n_samples: Total number of samples
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        random_seed: Random seed for reproducibility
        diagnostics: Optional IPSDiagnostics or DRDiagnostics object

    Returns:
        Tuple of (matplotlib Figure, metrics dict)
    """
    # Check if we have a diagnostics object to use
    from ..diagnostics import IPSDiagnostics, DRDiagnostics

    if diagnostics is not None and isinstance(
        diagnostics, (IPSDiagnostics, DRDiagnostics)
    ):
        # Extract weight info from diagnostics
        if n_samples is None:
            n_samples = diagnostics.n_samples_valid

    policies = list(raw_weights_dict.keys())
    n_policies = len(policies)
    metrics = {}

    # Use calibrated weights if provided, otherwise use raw
    use_calibrated = calibrated_weights_dict is not None
    weights_to_plot = calibrated_weights_dict if use_calibrated else raw_weights_dict

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Infer n_samples if not provided - but track per-policy
    if n_samples is None:
        n_samples = len(next(iter(raw_weights_dict.values())))

    # Compute metrics for all policies
    for policy in policies:
        raw_w = raw_weights_dict[policy]
        cal_w = (
            calibrated_weights_dict.get(policy, raw_w)
            if calibrated_weights_dict
            else raw_w
        )

        # Track actual sample size per policy
        policy_n_samples = len(raw_w)

        # ESS metrics
        ess_raw = compute_ess(raw_w)
        ess_cal = compute_ess(cal_w)

        # Sample efficiency: how many samples contribute X% of weight
        sorted_w = np.sort(cal_w)[::-1]
        cumsum_w = np.cumsum(sorted_w)
        total_w = cumsum_w[-1]

        n_for_50 = np.searchsorted(cumsum_w, 0.5 * total_w) + 1
        n_for_90 = np.searchsorted(cumsum_w, 0.9 * total_w) + 1

        # Count extreme weights
        n_above_10 = np.sum(cal_w > 10)
        n_above_100 = np.sum(cal_w > 100)

        metrics[policy] = {
            "ess_raw": ess_raw,
            "ess_cal": ess_cal,
            "ess_raw_frac": ess_raw / policy_n_samples,
            "ess_cal_frac": ess_cal / policy_n_samples,
            "ess_improvement": ess_cal / max(ess_raw, 1e-10),
            "max_weight_raw": np.max(raw_w),
            "max_weight_cal": np.max(cal_w),
            "n_for_50pct": n_for_50,
            "n_for_90pct": n_for_90,
            "n_samples": policy_n_samples,
            "n_above_10": n_above_10,
            "n_above_100": n_above_100,
        }

    # Create figure with 3x2 grid
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        3, 2, hspace=0.35, wspace=0.35, left=0.08, right=0.95, top=0.94, bottom=0.06
    )

    # Row 1: Core metrics
    ax_ess = fig.add_subplot(gs[0, 0])
    _plot_ess_comparison(ax_ess, metrics, policies)

    ax_max = fig.add_subplot(gs[0, 1])
    _plot_max_weight_comparison(ax_max, metrics, policies)

    # Row 2: Distribution analysis
    ax_transform = fig.add_subplot(gs[1, 0])
    if calibrated_weights_dict is not None:
        _plot_weight_histograms(
            ax_transform, raw_weights_dict, calibrated_weights_dict, policies
        )
    else:
        _plot_weight_histograms(ax_transform, raw_weights_dict, {}, policies)

    ax_tail = fig.add_subplot(gs[1, 1])
    if weights_to_plot is not None:
        _plot_tail_ccdf_combined(ax_tail, weights_to_plot, policies)

    # Row 3: Efficiency and summary
    ax_eff = fig.add_subplot(gs[2, 0])
    _plot_sample_efficiency(ax_eff, metrics, policies)

    ax_table = fig.add_subplot(gs[2, 1])
    _plot_summary_table(ax_table, metrics, policies, use_calibrated)

    # Main title
    title = "Weight Diagnostics Dashboard"
    if use_calibrated:
        title += " (Calibrated Weights)"
    plt.suptitle(title, fontsize=14, fontweight="bold")

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")

    return fig, metrics


# ============================================================================
# Detailed Dashboard (per-policy with judge scores)
# ============================================================================


def plot_weight_dashboard_detailed(
    raw_weights_dict: Dict[str, np.ndarray],
    calibrated_weights_dict: Optional[Dict[str, np.ndarray]] = None,
    n_samples: Optional[int] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (16, 10),
    random_seed: int = 42,
    diagnostics: Optional[Any] = None,
    **kwargs: Any,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """Create per-policy weight dashboards with ordering index visualization.

    Creates a grid of subplots, one dashboard per policy, each showing:
    - Weight smoothing by ordering index (g_oof when available, else judge score)
    - ESS and tail diagnostics
    - Clear per-policy view

    Args:
        raw_weights_dict: Dict mapping policy names to raw weight arrays
        calibrated_weights_dict: Optional dict of calibrated weights
        n_samples: Total number of samples
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        random_seed: Random seed for reproducibility
        diagnostics: Optional IPSDiagnostics or DRDiagnostics object
        **kwargs: Must include either 'judge_scores' dict or 'sampler'
                 Can also include 'ordering_indices' dict and 'calibrator'

    Returns:
        Tuple of (matplotlib Figure, metrics dict)
    """
    np.random.seed(random_seed)

    policies = list(raw_weights_dict.keys())
    n_policies = len(policies)

    # Get judge scores and ordering indices
    judge_scores_dict = kwargs.get("judge_scores", {})
    ordering_indices_dict = kwargs.get("ordering_indices", {})
    sampler = kwargs.get("sampler")
    calibrator = kwargs.get("calibrator")

    # Extract judge scores from sampler if not provided directly
    if not judge_scores_dict and sampler is not None:
        judge_scores_dict = {}
        for policy in policies:
            data = sampler.get_data_for_policy(policy)
            if data:
                scores = np.array([d.get("judge_score", np.nan) for d in data])
                valid = ~np.isnan(scores)
                if valid.sum() > 0:
                    judge_scores_dict[policy] = scores[valid]

    # Try to compute ordering indices (g_oof) if not provided but calibrator available
    if not ordering_indices_dict and calibrator is not None and sampler is not None:
        if hasattr(calibrator, "predict_oof"):
            ordering_indices_dict = {}
            for policy in policies:
                data = sampler.get_data_for_policy(policy)
                if data:
                    # Get judge scores and compute fold IDs from prompt_ids
                    judge_scores = np.array(
                        [d.get("judge_score", np.nan) for d in data]
                    )
                    # Compute folds on-demand from prompt_ids
                    from ..data.folds import get_fold

                    fold_list = [
                        (
                            get_fold(d.get("prompt_id"), 5, 42)
                            if d.get("prompt_id")
                            else None
                        )
                        for d in data
                    ]

                    # Check if we have valid fold IDs
                    if all(v is not None for v in fold_list) and len(fold_list) == len(
                        judge_scores
                    ):
                        fold_ids = np.asarray(fold_list, dtype=int)
                        try:
                            # Compute cross-fitted predictions
                            g_oof = calibrator.predict_oof(judge_scores, fold_ids)
                            if g_oof is not None and not np.all(g_oof == 0):
                                ordering_indices_dict[policy] = g_oof
                        except:
                            pass  # Fall back to judge scores

    # Determine grid layout
    if n_policies <= 2:
        rows, cols = 1, n_policies
    elif n_policies <= 4:
        rows, cols = 2, 2
    elif n_policies <= 6:
        rows, cols = 2, 3
    else:
        rows = int(np.ceil(n_policies / 3))
        cols = 3

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Hide extra subplots
    for i in range(n_policies, len(axes_flat)):
        axes_flat[i].axis("off")

    # Metrics storage
    all_metrics = {}

    # Plot each policy
    for idx, policy in enumerate(policies):
        ax = axes_flat[idx]

        # Get data for this policy
        raw_w = raw_weights_dict[policy]
        cal_w = (
            calibrated_weights_dict.get(policy, raw_w)
            if calibrated_weights_dict
            else raw_w
        )
        judge_scores = judge_scores_dict.get(policy, None)
        ordering_index = ordering_indices_dict.get(policy, None)

        # Use ordering index if available, otherwise fall back to judge scores
        plot_index = ordering_index if ordering_index is not None else judge_scores
        index_label = (
            "Calibrated Reward g(s)" if ordering_index is not None else "Judge Score"
        )

        # Compute metrics for this policy
        ess_raw = compute_ess(raw_w)
        ess_cal = compute_ess(cal_w)
        uplift = ess_cal / max(ess_raw, 1e-12)

        # Top 1% mass
        w_sorted = np.sort(raw_w)[::-1]
        k = max(1, int(len(w_sorted) * 0.01))
        top1_raw = w_sorted[:k].sum() / max(w_sorted.sum(), 1e-12)

        w_sorted = np.sort(cal_w)[::-1]
        k = max(1, int(len(w_sorted) * 0.01))
        top1_cal = w_sorted[:k].sum() / max(w_sorted.sum(), 1e-12)

        # Store metrics
        all_metrics[policy] = {
            "ess_raw": ess_raw,
            "ess_cal": ess_cal,
            "ess_improvement": uplift,
            "top1_raw": top1_raw,
            "top1_cal": top1_cal,
            "n_samples": len(raw_w),
            "ordering_index_used": (
                "g_oof" if ordering_index is not None else "judge_score"
            ),
        }

        if plot_index is not None and len(plot_index) == len(raw_w):
            # Plot weight smoothing by ordering index
            _plot_single_policy_weight_smoothing(
                ax,
                plot_index,
                raw_w,
                cal_w,
                policy,
                ess_raw,
                ess_cal,
                uplift,
                top1_raw,
                top1_cal,
                index_label=index_label,
            )
        else:
            # Fallback: simple histogram comparison
            _plot_single_policy_weight_histogram(
                ax, raw_w, cal_w, policy, ess_raw, ess_cal, uplift, top1_raw, top1_cal
            )

    # Main title
    fig.suptitle(
        f"Weight Diagnostics by Policy (n={n_samples or len(next(iter(raw_weights_dict.values())))})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")

    return fig, all_metrics


# ============================================================================
# Helper functions for summary dashboard
# ============================================================================


def _plot_ess_comparison(ax: Any, metrics: Dict, policies: List[str]) -> None:
    """Plot ESS as effective samples (not percentage)."""
    n_policies = len(policies)
    x = np.arange(n_policies)
    width = 0.35

    # Get values
    raw_ess = [metrics[p]["ess_raw"] for p in policies]
    cal_ess = [metrics[p]["ess_cal"] for p in policies]
    improvements = [metrics[p]["ess_improvement"] for p in policies]

    # Use consistent tab10 colormap
    colors = plt.cm.get_cmap("tab10")

    # Plot bars
    bars1 = ax.bar(
        x - width / 2, raw_ess, width, label="Raw", color=colors(0), alpha=0.7
    )
    bars2 = ax.bar(
        x + width / 2, cal_ess, width, label="Calibrated", color=colors(1), alpha=0.7
    )

    # Labels on bars with improvement factor
    for i, (r, c, imp) in enumerate(zip(raw_ess, cal_ess, improvements)):
        ax.text(i - width / 2, r + 5, f"{r:.0f}", ha="center", fontsize=8)
        ax.text(i + width / 2, c + 5, f"{c:.0f}", ha="center", fontsize=8)
        if imp > 1.5:  # Only show significant improvements
            ax.text(
                i + width / 2,
                c / 2,
                f"+{imp:.1f}×",
                ha="center",
                fontsize=7,
                style="italic",
                color="darkgreen",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.set_ylabel("Effective Samples")
    ax.set_title("A. Effective Sample Size")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Add reference lines
    if policies:
        min_n_samples = min(metrics[p]["n_samples"] for p in policies)
        max_n_samples = max(metrics[p]["n_samples"] for p in policies)

        if max_n_samples > min_n_samples * 1.1:
            ax.text(
                0.02,
                0.98,
                f"n varies: {min_n_samples}-{max_n_samples}",
                transform=ax.transAxes,
                fontsize=7,
                va="top",
            )

        # 50% line - good threshold
        ax.axhline(
            min_n_samples * 0.5, color="green", linestyle="--", alpha=0.3, linewidth=1
        )
        ax.text(
            n_policies - 0.5,
            min_n_samples * 0.5 + 10,
            "50% (good)",
            fontsize=7,
            color="green",
            alpha=0.7,
            ha="right",
        )

        # 10% line - warning threshold
        ax.axhline(
            min_n_samples * 0.1, color="orange", linestyle="--", alpha=0.3, linewidth=1
        )
        ax.text(
            n_policies - 0.5,
            min_n_samples * 0.1 + 10,
            "10% (marginal)",
            fontsize=7,
            color="orange",
            alpha=0.7,
            ha="right",
        )


def _plot_max_weight_comparison(ax: Any, metrics: Dict, policies: List[str]) -> None:
    """Plot maximum weights showing weight concentration risk."""
    n_policies = len(policies)
    x = np.arange(n_policies)
    width = 0.35

    raw_max = [metrics[p]["max_weight_raw"] for p in policies]
    cal_max = [metrics[p]["max_weight_cal"] for p in policies]

    # Use log scale if any weight > 10
    if max(raw_max + cal_max) > 10:
        ax.set_yscale("log")

    colors = plt.cm.get_cmap("tab10")

    bars1 = ax.bar(
        x - width / 2, raw_max, width, label="Raw", color=colors(0), alpha=0.7
    )
    bars2 = ax.bar(
        x + width / 2, cal_max, width, label="Calibrated", color=colors(1), alpha=0.7
    )

    # Add value labels
    for i, (r, c) in enumerate(zip(raw_max, cal_max)):
        ax.text(
            i - width / 2,
            r * 1.1 if r > 0 else 0.1,
            f"{r:.0f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )
        ax.text(
            i + width / 2,
            c * 1.1 if c > 0 else 0.1,
            f"{c:.0f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    # Reference lines
    ax.axhline(1, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(10, color="orange", linestyle="--", alpha=0.5)
    ax.axhline(100, color="red", linestyle="--", alpha=0.5)

    ax.text(n_policies - 0.5, 1.2, "Target", fontsize=7, color="gray", alpha=0.7)
    ax.text(n_policies - 0.5, 12, "High", fontsize=7, color="orange", alpha=0.7)
    ax.text(n_policies - 0.5, 120, "Extreme", fontsize=7, color="red", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(
        policies,
        rotation=45 if n_policies > 3 else 0,
        ha="right" if n_policies > 3 else "center",
    )
    ax.set_ylabel("Weight of Most Important Sample")
    ax.set_title("B. Weight Concentration: Single Sample Dominance")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")


def _plot_sample_efficiency(ax: Any, metrics: Dict, policies: List[str]) -> None:
    """Plot how many samples contribute 50% and 90% of weight."""
    n_policies = len(policies)

    # Prepare data
    data = []
    for p in policies:
        n_50 = metrics[p]["n_for_50pct"]
        n_90 = metrics[p]["n_for_90pct"]
        n_total = metrics[p]["n_samples"]
        n_rest = n_total - n_90

        # Percentages for stacking
        pct_50 = 100 * n_50 / n_total
        pct_90_50 = 100 * (n_90 - n_50) / n_total
        pct_rest = 100 * n_rest / n_total

        data.append((pct_50, pct_90_50, pct_rest, n_50, n_90))

    # Create stacked bars
    x = np.arange(n_policies)
    colors = plt.cm.get_cmap("tab10")

    bars1 = ax.bar(
        x,
        [d[0] for d in data],
        label=f"Samples carrying 50% of weight",
        color=colors(3),
        alpha=0.8,
    )

    bars2 = ax.bar(
        x,
        [d[1] for d in data],
        bottom=[d[0] for d in data],
        label="Additional samples for 90% weight",
        color=colors(1),
        alpha=0.6,
    )

    bottom_sum = [d[0] + d[1] for d in data]
    bars3 = ax.bar(
        x,
        [d[2] for d in data],
        bottom=bottom_sum,
        label="Samples with minimal weight (<10%)",
        color="lightgray",
        alpha=0.4,
    )

    # Add text annotations
    for i, (p50, p90_50, prest, n50, n90) in enumerate(data):
        policy = policies[i]
        n_total = metrics[policy]["n_samples"]

        if p50 > 3:  # Only show if segment is large enough
            label_50 = f"{n50}\n({n50/n_total*100:.0f}%)"
            ax.text(
                i,
                p50 / 2,
                label_50,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        policies,
        rotation=45 if n_policies > 3 else 0,
        ha="right" if n_policies > 3 else "center",
    )
    ax.set_ylabel("% of Total Samples")
    ax.set_ylim(0, 100)
    ax.set_title("E. Sample Efficiency: How Many Samples Actually Matter?")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def _plot_weight_histograms(
    ax: Any, raw_weights_dict: Dict, calibrated_weights_dict: Dict, policies: List[str]
) -> None:
    """Plot weight distribution histograms comparing raw vs calibrated."""
    for i, policy in enumerate(policies[:3]):  # Limit to 3 policies for clarity
        raw_w = raw_weights_dict[policy]
        cal_w = (
            calibrated_weights_dict.get(policy, raw_w)
            if calibrated_weights_dict
            else raw_w
        )

        # Create log-spaced bins
        raw_positive = raw_w[raw_w > 0]
        cal_positive = cal_w[cal_w > 0]

        if len(raw_positive) > 0 and len(cal_positive) > 0:
            min_val = min(raw_positive.min(), cal_positive.min())
            max_val = max(raw_positive.max(), cal_positive.max())
            bins = np.logspace(np.log10(max(min_val, 1e-6)), np.log10(max_val), 40)

            ax.hist(
                raw_positive, bins=bins, alpha=0.3, label=f"{policy} raw", density=True
            )
            ax.hist(
                cal_positive,
                bins=bins,
                alpha=0.5,
                label=f"{policy} cal",
                density=True,
                histtype="step",
                linewidth=2,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Density")
    ax.set_title("C. Weight Distributions")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_tail_ccdf_combined(ax: Any, weights_dict: Dict, policies: List[str]) -> None:
    """CCDF on log-log scale, all policies overlaid."""
    colors = plt.cm.get_cmap("Set2")(np.linspace(0, 1, len(policies)))

    for policy, color in zip(policies, colors):
        weights = weights_dict[policy]

        # Sort weights and compute CCDF
        w_sorted = np.sort(weights[weights > 0])
        if len(w_sorted) == 0:
            continue

        # CCDF: fraction of weights >= x
        ccdf = 1.0 - np.arange(len(w_sorted)) / len(w_sorted)

        ax.loglog(w_sorted, ccdf, label=policy, linewidth=2, alpha=0.7, color=color)

    ax.set_xlabel("Weight")
    ax.set_ylabel("P(W ≥ x)")
    ax.set_title("D. Tail Behavior (CCDF)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # Add reference lines
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(10.0, color="orange", linestyle="--", alpha=0.5)
    ax.axvline(100.0, color="red", linestyle="--", alpha=0.5)


def _plot_summary_table(
    ax: Any, metrics: Dict, policies: List[str], use_calibrated: bool
) -> None:
    """Summary table with status and recommendations."""
    ax.axis("off")

    # Prepare table data
    headers = ["Policy", "ESS", "Status", "Recommendation"]
    rows = []

    for policy in policies:
        m = metrics[policy]
        ess_frac = m["ess_cal_frac"] if use_calibrated else m["ess_raw_frac"]
        ess_val = m["ess_cal"] if use_calibrated else m["ess_raw"]

        # Status based on ESS
        if ess_frac > 0.5:
            status = "Excellent"
            rec = "Ready for production"
        elif ess_frac > 0.2:
            status = "Good"
            rec = "Usable with caution"
        elif ess_frac > 0.1:
            status = "Marginal"
            rec = "Consider more data"
        else:
            status = "Poor"
            rec = "Insufficient overlap"

        # Add calibration recommendation if relevant
        if not use_calibrated and m["ess_improvement"] > 2.0:
            rec = f"Use calibration ({m['ess_improvement']:.1f}× gain)"

        rows.append(
            [
                policy[:12],  # Truncate long names
                f"{ess_val:.0f} ({100*ess_frac:.0f}%)",
                status,
                rec[:20],  # Truncate long recommendations
            ]
        )

    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="left",
        loc="center",
        colWidths=[0.2, 0.25, 0.2, 0.35],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#E8E8E8")
        table[(0, i)].set_text_props(weight="bold")

    # Color code by status
    for i, policy in enumerate(policies):
        ess_frac = (
            metrics[policy]["ess_cal_frac"]
            if use_calibrated
            else metrics[policy]["ess_raw_frac"]
        )
        if ess_frac > 0.5:
            color = "#90EE90"  # Light green
        elif ess_frac > 0.2:
            color = "#FFFACD"  # Light yellow
        elif ess_frac > 0.1:
            color = "#FFE4B5"  # Light orange
        else:
            color = "#FFB6C1"  # Light red
        table[(i + 1, 2)].set_facecolor(color)


# ============================================================================
# Helper functions for detailed dashboard
# ============================================================================


def _plot_single_policy_weight_smoothing(
    ax: Any,
    ordering_index: np.ndarray,
    raw_w: np.ndarray,
    cal_w: np.ndarray,
    policy: str,
    ess_raw: float,
    ess_cal: float,
    uplift: float,
    top1_raw: float,
    top1_cal: float,
    index_label: str = "Judge Score",
) -> None:
    """Plot weights vs ordering index with calibration effect.

    The ordering index can be either judge scores or calibrated rewards g(s),
    depending on what was used for SIMCal calibration.
    """

    # Filter to valid values
    mask = (
        np.isfinite(ordering_index)
        & np.isfinite(raw_w)
        & np.isfinite(cal_w)
        & (raw_w > 0)
        & (cal_w > 0)
    )
    S = ordering_index[mask]
    W_raw = raw_w[mask]
    W_cal_actual = cal_w[mask]

    n = len(S)

    # Sort by ordering index
    sort_idx = np.argsort(S)
    S_sorted = S[sort_idx]
    W_raw_sorted = W_raw[sort_idx]
    W_cal_actual_sorted = W_cal_actual[sort_idx]

    # Plot raw weights vs judge scores
    if n > 2000:
        # Subsample for visibility
        step = max(1, n // 1000)
        indices = np.arange(0, n, step)
        ax.scatter(
            S_sorted[indices],
            W_raw_sorted[indices],
            s=2,
            alpha=0.2,
            color="C0",
            label="raw weights",
            rasterized=True,
        )
    else:
        ax.scatter(
            S,
            W_raw,
            s=3,
            alpha=0.3,
            color="C0",
            label="raw weights",
        )

    # Plot actual calibrated weights as thick line
    ax.plot(
        S_sorted,
        W_cal_actual_sorted,
        color="C2",
        linewidth=2.5,
        label="calibrated",
        zorder=10,
    )

    # Add horizontal line at y=1 (target mean)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, linewidth=1)

    # Set log scale for y-axis only
    ax.set_yscale("log")

    # Title with ESS diagnostics
    ax.set_title(
        f"{policy}\n" f"ESS: {ess_raw:.0f}→{ess_cal:.0f} ({uplift:.1f}×)",
        fontsize=10,
    )
    ax.set_xlabel(index_label, fontsize=9)
    ax.set_ylabel("Weight (log scale)", fontsize=9)
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3, which="both", linestyle=":")
    ax.tick_params(labelsize=8)


def _plot_single_policy_weight_histogram(
    ax: Any,
    raw_w: np.ndarray,
    cal_w: np.ndarray,
    policy: str,
    ess_raw: float,
    ess_cal: float,
    uplift: float,
    top1_raw: float,
    top1_cal: float,
) -> None:
    """Fallback: histogram comparison when judge scores unavailable."""

    # Create log-spaced bins
    raw_positive = raw_w[raw_w > 0]
    cal_positive = cal_w[cal_w > 0]

    if len(raw_positive) > 0 and len(cal_positive) > 0:
        min_val = min(raw_positive.min(), cal_positive.min())
        max_val = max(raw_positive.max(), cal_positive.max())
        bins = np.logspace(np.log10(max(min_val, 1e-6)), np.log10(max_val), 40)

        # Plot histograms
        ax.hist(
            raw_positive, bins=bins, alpha=0.4, color="C0", label="raw", density=True
        )
        ax.hist(
            cal_positive,
            bins=bins,
            alpha=0.6,
            color="C1",
            label="calibrated",
            density=True,
            histtype="step",
            linewidth=2,
        )

        ax.set_xscale("log")
        ax.set_xlabel("Weight", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)

    # Title with diagnostics
    ax.set_title(
        f"{policy} (no judge scores)\nESS: {ess_raw:.0f}→{ess_cal:.0f} ({uplift:.1f}×), "
        f"Top1%: {100*top1_raw:.1f}%→{100*top1_cal:.1f}%",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)


# ============================================================================
# Backward compatibility aliases
# ============================================================================
