"""Transport diagnostics visualization.

Modern, clean visualizations for the simplified unbiasedness test approach.
"""

from pathlib import Path
from typing import Any, Optional, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ..diagnostics.transport import TransportDiagnostics


def plot_transport_audit(
    diag: TransportDiagnostics,
    probe_samples: list,
    calibrator: Any,
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Plot transport audit results with scatter of residuals vs judge score.

    Shows individual residuals, mean residual, 95% CI band, and zero reference line.

    Args:
        diag: TransportDiagnostics from audit_transportability()
        probe_samples: List of probe samples for scatter plot
        calibrator: Calibrator for computing residuals
        save_path: Optional path to save figure
        figsize: Figure size (default: 8x6)

    Returns:
        matplotlib Figure

    Example:
        >>> from cje.diagnostics import audit_transportability
        >>> from cje.visualization.transport import plot_transport_audit
        >>>
        >>> diag = audit_transportability(calibrator, probe_samples)
        >>> plot_transport_audit(diag, probe_samples, calibrator, save_path="transport.png")
    """
    from ..data.models import Sample

    # Extract data
    judge_scores = []
    oracle_labels = []
    for sample in probe_samples:
        if not isinstance(sample, Sample):
            raise TypeError(f"Expected Sample, got {type(sample)}")
        if sample.oracle_label is None:
            continue
        judge_scores.append(sample.judge_score)
        oracle_labels.append(sample.oracle_label)

    S = np.array(judge_scores)
    Y = np.array(oracle_labels)
    Y_pred = calibrator.predict(S)
    residuals = Y - Y_pred

    # Status colors
    status_colors = {
        "PASS": "#2ecc71",
        "WARN": "#f39c12",
        "FAIL": "#e74c3c",
    }
    status_color = status_colors.get(diag.status, "#95a5a6")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # CI band
    ci_lower, ci_upper = diag.delta_ci
    ax.axhspan(
        ci_lower,
        ci_upper,
        alpha=0.3,
        color=status_color,
        label=f"95% CI",
        zorder=1,
        linewidth=0,
    )

    # Scatter plot - translucent black points
    ax.scatter(
        S,
        residuals,
        alpha=0.2,
        s=30,
        color="black",
        edgecolors="none",
        zorder=3,
        label="Individual residuals",
    )

    # Zero line (perfect calibration)
    ax.axhline(
        0,
        color="black",
        linestyle="-",
        linewidth=2,
        label="Perfect calibration",
        zorder=5,
    )

    # Mean residual
    ax.axhline(
        diag.delta_hat,
        color="darkgray",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {diag.delta_hat:+.3f}",
        zorder=4,
        alpha=0.7,
    )

    # Formatting
    ax.set_xlabel("Judge Score (S)", fontsize=11)
    ax.set_ylabel("Residual (Y - f̂(S))", fontsize=11)

    # Title with status
    policy_name = diag.group_label if diag.group_label else "Target Policy"
    ax.set_title(
        f"{policy_name}\n{diag.status} | n={diag.n_probe}",
        fontsize=12,
        fontweight="bold",
        color=status_color,
    )

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.set_xlim(-0.05, 1.05)

    # Add status annotation
    zero_in_ci = ci_lower <= 0 <= ci_upper
    status_text = "Unbiased ✓" if zero_in_ci else "Biased ✗"
    ax.text(
        0.02,
        0.98,
        status_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"Saved transport audit to {save_path.with_suffix('.png')}")

    return fig


def plot_transport_comparison(
    results: Dict[str, TransportDiagnostics],
    probe_samples_by_policy: Dict[str, list],
    calibrator: Any,
    save_path: Optional[Path] = None,
    bonferroni_correct: bool = True,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Plot transportability comparison across multiple policies.

    Creates one subplot per policy showing residuals, CI bands, and unbiasedness test.

    Args:
        results: Dict mapping policy names to TransportDiagnostics
        probe_samples_by_policy: Dict mapping policy names to probe sample lists
        calibrator: Fitted calibrator
        save_path: Optional path to save figure
        bonferroni_correct: Apply Bonferroni correction for multiple testing
        figsize: Figure size (auto if None)

    Returns:
        matplotlib Figure

    Example:
        >>> results = {}
        >>> for policy in ["clone", "premium", "unhelpful"]:
        >>>     diag = audit_transportability(calibrator, probes[policy], group_label=policy)
        >>>     results[policy] = diag
        >>>
        >>> plot_transport_comparison(results, probes, calibrator, save_path="comparison.png")
    """
    from ..data.models import Sample

    n_policies = len(results)
    if figsize is None:
        figsize = (4 * n_policies, 4.5)

    # Status colors
    status_colors = {
        "PASS": "#2ecc71",
        "WARN": "#f39c12",
        "FAIL": "#e74c3c",
    }

    # Bonferroni correction
    if bonferroni_correct:
        alpha_bonferroni = 0.05 / n_policies
        z_bonf = stats.norm.ppf(1 - alpha_bonferroni / 2)
    else:
        z_bonf = 1.96

    # Create subplots
    fig, axes = plt.subplots(1, n_policies, figsize=figsize, sharey=True)
    if n_policies == 1:
        axes = [axes]

    for idx, (policy, diag) in enumerate(results.items()):
        ax = axes[idx]

        # Get probe samples for this policy
        probe_samples = probe_samples_by_policy.get(policy, [])
        if not probe_samples:
            continue

        # Extract data
        judge_scores = []
        oracle_labels = []
        for sample in probe_samples:
            if not isinstance(sample, Sample):
                raise TypeError(f"Expected Sample, got {type(sample)}")
            if sample.oracle_label is None:
                continue
            judge_scores.append(sample.judge_score)
            oracle_labels.append(sample.oracle_label)

        S = np.array(judge_scores)
        Y = np.array(oracle_labels)
        Y_pred = calibrator.predict(S)
        residuals = Y - Y_pred

        # Get status color
        status_color = status_colors.get(diag.status, "#95a5a6")

        # Bonferroni-corrected CI band
        ci_bonf_lower = diag.delta_hat - z_bonf * diag.delta_se
        ci_bonf_upper = diag.delta_hat + z_bonf * diag.delta_se
        ax.axhspan(
            ci_bonf_lower,
            ci_bonf_upper,
            alpha=0.4,
            color=status_color,
            label=f'95% CI{" (Bonf.)" if bonferroni_correct else ""}',
            zorder=1,
            linewidth=0,
        )

        # Scatter plot - translucent blue points
        ax.scatter(
            S,
            residuals,
            alpha=0.2,
            s=20,
            color="#3498db",
            edgecolors="none",
            zorder=3,
            label="Individual residuals",
        )

        # Zero line (perfect calibration)
        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Perfect calibration",
            zorder=2,
            alpha=0.5,
        )

        # Mean residual - solid line colored by status
        ax.axhline(
            diag.delta_hat,
            color=status_color,
            linestyle="-",
            linewidth=1,
            label=f"Mean: {diag.delta_hat:+.3f}",
            zorder=4,
        )

        # Formatting
        ax.set_xlabel("Judge Score (S)", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Residual (Y - f̂(S))", fontsize=11)

        # Title with status
        policy_name = policy.replace("_", " ").title()
        ax.set_title(
            f"{policy_name}\n{diag.status} | n={diag.n_probe}",
            fontsize=12,
            fontweight="bold",
            color=status_color,
        )

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.set_xlim(-0.05, 1.05)

        # Add status annotation
        zero_in_bonf = ci_bonf_lower <= 0 <= ci_bonf_upper
        status_text = "Unbiased ✓" if zero_in_bonf else "Biased ✗"
        ax.text(
            0.02,
            0.98,
            status_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Overall title
    mode = getattr(calibrator, "selected_mode", "unknown")
    correction_text = (
        f"Bonferroni-corrected α={alpha_bonferroni:.4f}"
        if bonferroni_correct
        else "α=0.05"
    )
    fig.suptitle(
        f"Transportability Analysis: Calibrator Unbiasedness Test\n"
        f"Testing H₀: E[Y - f̂(S)] = 0 for each policy ({correction_text} for {n_policies} tests)\n"
        f"Mode: {mode}",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"Saved transport comparison to {save_path.with_suffix('.png')}")

    return fig
