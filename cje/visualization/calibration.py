"""Calibration visualization utilities."""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


def plot_calibration_comparison(
    judge_scores: np.ndarray,
    oracle_labels: np.ndarray,
    calibrated_scores: Optional[np.ndarray] = None,
    calibrator: Optional[object] = None,
    n_bins: int = 10,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Plot calibration comparison showing transformation and improvement.

    Shows the calibration transformation and its effect on oracle alignment.

    Args:
        judge_scores: Raw judge scores
        oracle_labels: True oracle labels
        calibrated_scores: Calibrated judge scores (optional)
        n_bins: Number of bins for grouping
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Compute calibration metrics
    def compute_calibration_error(
        predictions: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, float]:
        """Compute Expected Calibration Error (ECE) and RMSE."""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins) - 1

        ece = 0.0
        total_samples = 0
        squared_errors = []

        for i in range(n_bins):
            mask = bin_indices == i
            n_in_bin = mask.sum()
            if n_in_bin > 0:
                pred_in_bin = predictions[mask].mean()
                true_in_bin = labels[mask].mean()

                # ECE: weighted average of bin-wise calibration errors
                ece += n_in_bin * abs(pred_in_bin - true_in_bin)
                total_samples += n_in_bin

                # For RMSE
                squared_errors.extend((predictions[mask] - labels[mask]) ** 2)

        ece = ece / total_samples if total_samples > 0 else 0.0
        rmse = np.sqrt(np.mean(squared_errors)) if squared_errors else 0.0

        return ece, rmse

    # Create a 2D histogram for density visualization
    H, xedges, yedges = np.histogram2d(judge_scores, oracle_labels, bins=20)
    H = H.T  # Transpose for correct orientation

    # Apply logarithmic transformation to make low counts more visible
    # Add 1 to avoid log(0), then apply log scaling
    H_log = np.log1p(H)  # log(1 + H) to handle zeros gracefully

    # Apply minimal smoothing to the log-transformed data
    from scipy.ndimage import gaussian_filter

    H_log_smooth = gaussian_filter(H_log, sigma=0.5)

    # Create mesh grid for contours
    X, Y = np.meshgrid(
        xedges[:-1] + (xedges[1] - xedges[0]) / 2,
        yedges[:-1] + (yedges[1] - yedges[0]) / 2,
    )

    # Create custom colormap for filled contours
    import matplotlib.colors as mcolors

    # White to much darker blue/navy for high contrast
    colors = [
        "#FFFFFF",
        "#F0F8FF",
        "#E0F2FF",
        "#D6EDFF",
        "#CCE7FF",
        "#99CEFF",
        "#66B2FF",
        "#3395FF",
        "#0078FF",
        "#0055CC",
        "#003D99",
        "#002866",
        "#001A33",
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("light_blues", colors, N=256)

    # Create filled contours using log-scale data
    # Levels are now in log space
    max_log_count = H_log_smooth.max()
    min_log_count = (
        H_log_smooth[H_log_smooth > 0].min() if np.any(H_log_smooth > 0) else 0
    )

    # Create levels in log space for better visibility of low-density regions
    filled_levels = np.linspace(min_log_count, max_log_count, 25)

    # Create filled contours
    contourf = ax.contourf(
        X, Y, H_log_smooth, levels=filled_levels, cmap=cmap, alpha=0.65, extend="both"
    )

    # Add colorbar for density scale with custom labels showing actual counts
    cbar = plt.colorbar(contourf, ax=ax, pad=0.02)
    cbar.set_label("Sample Count", fontsize=9)

    # Create custom tick positions and labels
    # Choose nice round numbers for actual counts
    max_count = np.expm1(max_log_count)  # Convert back from log space

    # Select appropriate tick values based on data range
    if max_count > 500:
        tick_counts = [0, 5, 10, 25, 50, 100, 250, 500, 1000]
    elif max_count > 100:
        tick_counts = [0, 5, 10, 25, 50, 100, 200]
    elif max_count > 50:
        tick_counts = [0, 5, 10, 25, 50, 100]
    else:
        tick_counts = [0, 2, 5, 10, 25, 50]

    # Filter to only include ticks within data range
    tick_counts = [t for t in tick_counts if t <= max_count]

    # Convert counts to log space for positioning
    tick_positions = [np.log1p(count) for count in tick_counts]

    # Set the ticks and labels
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([str(int(count)) for count in tick_counts])

    # Now create the regular 2D histogram for contour lines (not smoothed)
    H, xedges, yedges = np.histogram2d(judge_scores, oracle_labels, bins=20)
    H = H.T  # Transpose for correct orientation

    # Create mesh grid for contours
    X, Y = np.meshgrid(
        xedges[:-1] + (xedges[1] - xedges[0]) / 2,
        yedges[:-1] + (yedges[1] - yedges[0]) / 2,
    )

    # Adjust contour levels based on dataset size
    n_samples = len(judge_scores)
    if H.max() > 5:
        # Scale contour levels based on data size
        if n_samples < 500:
            levels = [5, 10, 25, 50]
        elif n_samples < 1000:
            levels = [10, 25, 50, 100]
        elif n_samples < 5000:
            levels = [10, 25, 50, 100, 200, 500]
        else:
            levels = [10, 50, 100, 250, 500, 1000]

        # Only use levels that exist in the data
        levels = [l for l in levels if l < H.max()]

        if levels:
            # Draw contour lines
            contours = ax.contour(
                X, Y, H, levels=levels, colors="darkgray", linewidths=0.7, alpha=0.7
            )
            # Label the contours with sample counts
            ax.clabel(contours, inline=True, fontsize=8, fmt="%d")

    # Compute binned statistics for empirical relationship
    bin_width = 0.05
    bin_edges = np.arange(0, 1.0 + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    binned_means = []
    binned_counts = []

    for i in range(len(bin_edges) - 1):
        if i == len(bin_edges) - 2:
            # Last bin: include right edge
            mask = (judge_scores >= bin_edges[i]) & (judge_scores <= bin_edges[i + 1])
        else:
            # All other bins: exclude right edge
            mask = (judge_scores >= bin_edges[i]) & (judge_scores < bin_edges[i + 1])
        if np.any(mask):
            binned_means.append(np.mean(oracle_labels[mask]))
            binned_counts.append(np.sum(mask))
        else:
            binned_means.append(np.nan)
            binned_counts.append(0)

    # Filter out empty bins
    valid_bins = ~np.isnan(binned_means)
    bin_centers_valid = bin_centers[valid_bins]
    binned_means_valid = np.array(binned_means)[valid_bins]
    binned_counts_valid = np.array(binned_counts)[valid_bins]

    # Plot smoothed empirical relationship using local polynomial regression (LOWESS)
    from scipy.signal import savgol_filter
    from scipy.interpolate import interp1d

    if len(bin_centers_valid) > 3:
        # Create smooth interpolation through binned means
        # Use cubic spline for smooth curve
        interp_func = interp1d(
            bin_centers_valid,
            binned_means_valid,
            kind="cubic" if len(bin_centers_valid) > 3 else "linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        # Create fine grid for smooth curve
        # Extend to the actual data range, not just valid bin centers
        x_min = max(0, judge_scores.min())
        x_max = min(1, judge_scores.max())
        x_smooth = np.linspace(x_min, x_max, 200)
        y_smooth = interp_func(x_smooth)

        # Clip to valid range
        y_smooth = np.clip(y_smooth, 0, 1)

        # Plot smoothed empirical curve
        ax.plot(
            x_smooth,
            y_smooth,
            "-",
            color="darkblue",
            alpha=0.9,
            linewidth=3,
            label="Empirical mean E[Oracle|Judge]",
            zorder=10,  # Make sure it's on top
        )

        # Also plot the binned points for reference
        ax.scatter(
            bin_centers_valid,
            binned_means_valid,
            s=40,
            alpha=0.7,
            color="darkblue",
            edgecolor="white",
            linewidth=0.5,
            zorder=11,
        )

    # Plot the calibration function if calibrator is available
    if calibrator is not None and hasattr(calibrator, "predict"):
        # Create a fine grid of judge scores
        judge_grid = np.linspace(0, 1, 200)

        # Get predictions from the calibrator
        try:
            calibrated_grid = calibrator.predict(judge_grid)
            calibrated_grid = np.clip(calibrated_grid, 0, 1)  # Ensure in [0,1]

            # Plot the calibration function
            ax.plot(
                judge_grid,
                calibrated_grid,
                "-",
                color="red",
                alpha=0.9,
                linewidth=3,
                label="Calibration function f(Judge)",
                zorder=12,  # On top
            )
        except Exception as e:
            print(f"Warning: Could not plot calibration function: {e}")

    # Also plot individual calibrated points if available
    elif calibrated_scores is not None:
        # Sort for smooth curve
        sorted_idx = np.argsort(judge_scores)
        judge_sorted = judge_scores[sorted_idx]
        calibrated_sorted = calibrated_scores[sorted_idx]

        # Plot as scatter to show actual calibrated values
        ax.scatter(
            judge_sorted[::10],  # Subsample for visibility
            calibrated_sorted[::10],
            s=20,
            alpha=0.5,
            color="red",
            label="Calibrated rewards",
            zorder=11,
        )

    # Add diagonal reference
    ax.plot(
        [0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Perfect calibration (y=x)"
    )

    # Labels and formatting
    ax.set_xlabel("Judge Score")
    ax.set_ylabel("Oracle Label / Calibrated Reward")
    ax.set_title("Judge→Oracle Calibration")
    ax.grid(True, alpha=0.3)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.legend(loc="upper left", fontsize=9)

    # Compute comprehensive statistics
    total_samples = len(judge_scores)

    # Compute calibration metrics
    ece_before, rmse_before = compute_calibration_error(judge_scores, oracle_labels)

    # Build comprehensive stats text
    if calibrated_scores is not None:
        ece_after, rmse_after = compute_calibration_error(
            calibrated_scores, oracle_labels
        )

        stats_text = (
            f"Samples: {total_samples:,}\n"
            f"ECE: {ece_before:.3f} → {ece_after:.3f}\n"
            f"RMSE: {rmse_before:.3f} → {rmse_after:.3f}\n"
            f"Judge: [{judge_scores.min():.3f}, {judge_scores.max():.3f}]\n"
            f"Oracle: [{oracle_labels.min():.3f}, {oracle_labels.max():.3f}]\n"
            f"Calibrated: [{calibrated_scores.min():.3f}, {calibrated_scores.max():.3f}]"
        )
    else:
        stats_text = (
            f"Samples: {total_samples:,}\n"
            f"ECE: {ece_before:.3f}\n"
            f"RMSE: {rmse_before:.3f}\n"
            f"Judge: [{judge_scores.min():.3f}, {judge_scores.max():.3f}]\n"
            f"Oracle: [{oracle_labels.min():.3f}, {oracle_labels.max():.3f}]"
        )

    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        horizontalalignment="right",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    # Save and return figure
    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")

    return fig
