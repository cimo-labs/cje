#!/usr/bin/env python
"""Generate comparison plot showing monotone vs two-stage calibration.

This creates a side-by-side visualization demonstrating when two-stage
calibration helps (when judge-oracle relationship is non-monotone).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from cje.calibration import JudgeCalibrator

# Load arena data
data_path = (
    Path(__file__).parent.parent.parent
    / "cje/experiments/arena_10k_simplified/data/cje_dataset.jsonl"
)

print(f"Loading data from {data_path}...")
samples = []
with open(data_path) as f:
    for line in f:
        samples.append(json.loads(line))

print(f"Loaded {len(samples)} samples")

# Extract judge scores and oracle labels
judge_scores = []
oracle_labels = []

for sample in samples:
    metadata = sample.get("metadata", {})
    judge_score = metadata.get("judge_score")
    oracle_label = metadata.get("oracle_label")

    if judge_score is not None and oracle_label is not None:
        judge_scores.append(judge_score)
        oracle_labels.append(oracle_label)

judge_scores = np.array(judge_scores)
oracle_labels = np.array(oracle_labels)

print(f"Found {len(judge_scores)} samples with both judge scores and oracle labels")
print(f"Judge score range: [{judge_scores.min():.3f}, {judge_scores.max():.3f}]")
print(f"Oracle label range: [{oracle_labels.min():.3f}, {oracle_labels.max():.3f}]")

# Fit both calibrators
print("\nFitting monotone calibrator...")
cal_monotone = JudgeCalibrator(calibration_mode="monotone")
result_monotone = cal_monotone.fit_transform(judge_scores, oracle_labels)

print("Fitting two-stage calibrator...")
cal_two_stage = JudgeCalibrator(calibration_mode="two_stage")
result_two_stage = cal_two_stage.fit_transform(judge_scores, oracle_labels)

print(
    f"\nMonotone - Calibrated range: [{result_monotone.calibrated_scores.min():.3f}, {result_monotone.calibrated_scores.max():.3f}]"
)
print(
    f"Two-stage - Calibrated range: [{result_two_stage.calibrated_scores.min():.3f}, {result_two_stage.calibrated_scores.max():.3f}]"
)

# Create comparison visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Common parameters
bin_width = 0.05
bin_edges = np.arange(0, 1.0 + bin_width, bin_width)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Compute binned statistics
binned_means = []
binned_counts = []

for i in range(len(bin_edges) - 1):
    if i == len(bin_edges) - 2:
        mask = (judge_scores >= bin_edges[i]) & (judge_scores <= bin_edges[i + 1])
    else:
        mask = (judge_scores >= bin_edges[i]) & (judge_scores < bin_edges[i + 1])

    if np.any(mask):
        binned_means.append(np.mean(oracle_labels[mask]))
        binned_counts.append(np.sum(mask))
    else:
        binned_means.append(np.nan)
        binned_counts.append(0)

valid_bins = ~np.isnan(binned_means)
bin_centers_valid = bin_centers[valid_bins]
binned_means_valid = np.array(binned_means)[valid_bins]


# Function to plot calibration
def plot_calibration(ax, calibrator, result, title, mode):
    # Scatter plot with transparency
    ax.scatter(
        judge_scores, oracle_labels, alpha=0.1, s=10, color="gray", label="Samples"
    )

    # Plot binned empirical means
    ax.scatter(
        bin_centers_valid,
        binned_means_valid,
        s=80,
        alpha=0.7,
        color="darkblue",
        edgecolor="white",
        linewidth=1,
        label="Empirical mean E[Oracle|Judge]",
        zorder=10,
    )

    # Plot calibration function
    judge_grid = np.linspace(0, 1, 200)
    calibrated_grid = calibrator.predict(judge_grid)
    ax.plot(
        judge_grid,
        calibrated_grid,
        "-",
        color="red",
        alpha=0.9,
        linewidth=3,
        label=f"{mode} calibration function",
        zorder=12,
    )

    # Plot diagonal reference
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Perfect (y=x)")

    # Compute RMSE
    rmse = np.sqrt(np.mean((oracle_labels - result.calibrated_scores) ** 2))

    # Labels and formatting
    ax.set_xlabel("Judge Score", fontsize=16, fontweight="bold")
    ax.set_ylabel("Oracle Label / Calibrated Reward", fontsize=16, fontweight="bold")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=14)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    # Legend with better readability
    legend = ax.legend(
        loc="upper left",
        fontsize=13,
        framealpha=0.98,
        edgecolor="gray",
        fancybox=False,
        shadow=False,
    )
    legend.get_frame().set_linewidth(1.5)

    # Add stats box
    stats_text = (
        f"RMSE: {rmse:.3f}\n"
        f"Range: [{result.calibrated_scores.min():.3f}, {result.calibrated_scores.max():.3f}]\n"
        f"Samples: {len(judge_scores):,}"
    )
    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=13,
        horizontalalignment="right",
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            alpha=0.98,
            edgecolor="gray",
            linewidth=2,
        ),
        fontweight="bold",
    )


# Plot both calibrators
plot_calibration(
    axes[0],
    cal_monotone,
    result_monotone,
    "Monotone Calibration (Standard)",
    "Monotone",
)
plot_calibration(
    axes[1],
    cal_two_stage,
    result_two_stage,
    "Two-Stage Calibration (Flexible)",
    "Two-stage",
)

plt.tight_layout()

# Save plot
output_path = Path(__file__).parent / "two_stage_comparison.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"\n✓ Comparison plot saved to {output_path}")
print("\nPlot shows:")
print("  Left: Standard monotone isotonic calibration")
print("  Right: Two-stage calibration (spline index → isotonic)")
print("\nTwo-stage helps when:")
print("  - Judge-oracle relationship is non-monotone")
print("  - Regional miscalibration at low/mid/high scores")
print("  - Length bias or other confounders at fixed judge score")
