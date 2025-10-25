"""Transport diagnostics visualization."""

from pathlib import Path
from typing import Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from ..diagnostics.transport import TransportDiagnostics


def plot_transport_audit(
    diag: TransportDiagnostics,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Plot transport audit results: decile residuals and coverage.

    Creates a two-panel figure showing:
    - Left: Decile residuals with ±0.05 threshold bands (gray out sparse deciles)
    - Right: Sample count distribution across deciles

    Args:
        diag: TransportDiagnostics from audit_transportability()
        save_path: Optional path to save figure
        figsize: Figure size (default: 12x5)

    Returns:
        matplotlib Figure

    Example:
        >>> from cje.diagnostics.transport import audit_transportability
        >>> from cje.visualization.transport import plot_transport_audit
        >>>
        >>> diag = audit_transportability(calibrator, probe_samples)
        >>> plot_transport_audit(diag, save_path="transport_audit.png")
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Define colors by status
    status_colors = {
        "PASS": "#2ecc71",  # Green
        "WARN": "#f39c12",  # Orange
        "FAIL": "#e74c3c",  # Red
    }
    status_color = status_colors.get(diag.status, "#95a5a6")

    # ========== Left panel: Decile residuals ==========
    deciles = np.arange(len(diag.decile_residuals))
    residuals = np.array(diag.decile_residuals)
    counts = np.array(diag.decile_counts)

    # Identify sparse deciles (< 3 samples or < 5% of n_probe / bins)
    min_count = max(3, diag.n_probe / len(deciles) * 0.5)
    sparse_mask = counts < min_count

    # Plot bars for decile residuals
    for i, (r, c, sparse) in enumerate(zip(residuals, counts, sparse_mask)):
        if np.isnan(r):
            # Empty decile
            ax1.bar(i, 0, color="lightgray", alpha=0.3, edgecolor="gray", linewidth=0.5)
        elif sparse:
            # Sparse decile - gray out
            ax1.bar(
                i, r, color="lightgray", alpha=0.5, edgecolor="darkgray", linewidth=0.8
            )
        else:
            # Normal decile - use status color
            alpha = 0.7 if abs(r) > 0.05 else 0.5
            ax1.bar(
                i, r, color=status_color, alpha=alpha, edgecolor="black", linewidth=0.5
            )

    # Add threshold bands at ±0.05
    ax1.axhline(
        0.05,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="±0.05 threshold",
    )
    ax1.axhline(-0.05, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax1.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)

    # Add global mean shift line if significant
    if abs(diag.delta_hat) > 0.01:
        ax1.axhline(
            diag.delta_hat,
            color="blue",
            linestyle=":",
            linewidth=2,
            alpha=0.7,
            label=f"Mean shift: {diag.delta_hat:+.3f}",
        )

    # Formatting
    ax1.set_xlabel("Risk Index Decile", fontsize=10)
    ax1.set_ylabel("Mean Residual (Y - f(S))", fontsize=10)
    ax1.set_title("Regional Residuals by Decile", fontsize=11, fontweight="bold")
    ax1.set_xticks(deciles)
    ax1.set_xticklabels([f"{i}" for i in deciles], fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend(loc="upper right", fontsize=8)

    # Set y-limits to show ±0.1 range minimum
    y_max = max(0.1, np.nanmax(np.abs(residuals)) * 1.2)
    ax1.set_ylim(-y_max, y_max)

    # ========== Right panel: Sample counts per decile ==========
    # Plot bar chart of counts
    for i, (c, sparse) in enumerate(zip(counts, sparse_mask)):
        if c == 0:
            # Empty decile
            ax2.bar(i, 0, color="lightgray", alpha=0.3, edgecolor="gray", linewidth=0.5)
        elif sparse:
            # Sparse decile - orange
            ax2.bar(
                i, c, color="#f39c12", alpha=0.6, edgecolor="darkorange", linewidth=0.8
            )
        else:
            # Normal decile - green
            ax2.bar(
                i, c, color="#2ecc71", alpha=0.6, edgecolor="darkgreen", linewidth=0.5
            )

    # Add threshold line for minimum count
    ax2.axhline(
        min_count,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label=f"Min count: {min_count:.0f}",
    )

    # Add mean count line
    mean_count = diag.n_probe / len(deciles)
    ax2.axhline(
        mean_count,
        color="blue",
        linestyle=":",
        linewidth=1.5,
        alpha=0.6,
        label=f"Uniform: {mean_count:.1f}",
    )

    # Formatting
    ax2.set_xlabel("Risk Index Decile", fontsize=10)
    ax2.set_ylabel("Sample Count", fontsize=10)
    ax2.set_title("Coverage by Decile", fontsize=11, fontweight="bold")
    ax2.set_xticks(deciles)
    ax2.set_xticklabels([f"{i}" for i in deciles], fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(loc="upper right", fontsize=8)

    # ========== Overall title and stats ==========
    status_symbols = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}
    status_symbol = status_symbols.get(diag.status, "?")

    title = f"Transport Diagnostic: {status_symbol} {diag.status}"
    if diag.group_label:
        title += f" | {diag.group_label}"

    fig.suptitle(title, fontsize=13, fontweight="bold", color=status_color)

    # Add summary stats box
    stats_text = (
        f"N = {diag.n_probe}\n"
        f"Mean shift: {diag.delta_hat:+.3f} (95% CI: [{diag.delta_ci[0]:+.3f}, {diag.delta_ci[1]:+.3f}])\n"
        f"Coverage: {diag.coverage:.1%}\n"
        f"Action: {diag.recommended_action}"
    )

    fig.text(
        0.5,
        0.02,
        stats_text,
        ha="center",
        va="bottom",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            alpha=0.9,
            edgecolor=status_color,
            linewidth=1.5,
        ),
    )

    plt.tight_layout(rect=(0, 0.12, 1, 0.96))

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        print(f"Saved transport audit plot to {save_path.with_suffix('.png')}")

    return fig


def plot_transport_residuals_scatter(
    probe_samples: list,
    calibrator: Any,
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Plot scatter of residuals vs. risk index (detailed diagnostic).

    Args:
        probe_samples: List of Sample objects with oracle_label
        calibrator: Fitted JudgeCalibrator
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    from ..data.models import Sample

    # Extract data
    probe_scores = []
    probe_labels = []
    probe_covariates = []

    for sample in probe_samples:
        if not isinstance(sample, Sample):
            raise TypeError(f"Expected Sample object, got {type(sample)}")

        judge_score = sample.metadata.get("judge_score")
        if judge_score is None or sample.oracle_label is None:
            continue

        probe_scores.append(judge_score)
        probe_labels.append(sample.oracle_label)

        if calibrator.covariate_names:
            cov_values = [
                sample.metadata.get(cov) for cov in calibrator.covariate_names
            ]
            probe_covariates.append(cov_values)

    S = np.array(probe_scores)
    Y = np.array(probe_labels)
    X_cov = np.array(probe_covariates) if probe_covariates else None

    # Get predictions and residuals
    R_hat = calibrator.predict(S, covariates=X_cov)
    residuals = Y - R_hat

    # Get risk index
    if (
        hasattr(calibrator, "_flexible_calibrator")
        and calibrator._flexible_calibrator is not None
    ):
        U = calibrator._flexible_calibrator.index(S, folds=None, covariates=X_cov)
    else:
        U = S

    # Normalize U to [0, 1]
    U_min, U_max = U.min(), U.max()
    if U_max > U_min:
        U_norm = (U - U_min) / (U_max - U_min)
    else:
        U_norm = np.zeros_like(U)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    scatter = ax.scatter(
        U_norm,
        residuals,
        c=np.abs(residuals),
        cmap="RdYlGn_r",
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("|Residual|", fontsize=9)

    # Add threshold bands
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(-0.05, color="red", linestyle="--", linewidth=1, alpha=0.5)

    # Add LOWESS smoothing
    try:
        from scipy.signal import savgol_filter

        # Sort by U for smooth curve
        sort_idx = np.argsort(U_norm)
        U_sorted = U_norm[sort_idx]
        resid_sorted = residuals[sort_idx]

        # Apply Savitzky-Golay filter for smooth trend
        if len(U_sorted) > 11:
            window = min(51, len(U_sorted) // 2 * 2 + 1)  # Odd window
            smooth_resid = savgol_filter(
                resid_sorted, window_length=window, polyorder=3
            )
            ax.plot(
                U_sorted,
                smooth_resid,
                color="blue",
                linewidth=2,
                alpha=0.7,
                label="Trend (LOWESS)",
            )
            ax.legend(loc="upper right", fontsize=8)
    except Exception:
        pass  # Skip smoothing if it fails

    # Formatting
    ax.set_xlabel("Risk Index U (normalized)", fontsize=10)
    ax.set_ylabel("Residual (Y - f(S))", fontsize=10)
    ax.set_title("Residuals vs. Risk Index", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Add stats box
    stats_text = (
        f"N = {len(residuals)}\n"
        f"Mean: {residuals.mean():+.3f}\n"
        f"Std: {residuals.std():.3f}\n"
        f"RMSE: {np.sqrt((residuals**2).mean()):.3f}"
    )

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")

    return fig
