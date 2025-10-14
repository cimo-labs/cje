#!/usr/bin/env python
"""Generate detailed visualization of two-stage calibration showing the transformation pipeline.

Shows:
1. Raw judge scores → oracle labels (empirical relationship)
2. Stage 1: Judge scores → smooth g(S) transformation
3. Stage 2: g(S) → ranked space → isotonic calibration
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

print(f"Found {len(judge_scores)} samples")

# Fit two-stage calibrator
print("\nFitting two-stage calibrator...")
calibrator = JudgeCalibrator(calibration_mode="two_stage")
result = calibrator.fit_transform(judge_scores, oracle_labels)

# Access the underlying flexible calibrator
flex_cal = calibrator._flexible_calibrator

# Get intermediate transformations
print("Computing intermediate stages...")

# Stage 1: g(S) - smooth transformation
judge_grid = np.linspace(0, 1, 200)
if flex_cal._full_g_model is not None:
    g_values = flex_cal._full_g_model.predict(judge_grid.reshape(-1, 1))
else:
    g_values = judge_grid  # Fallback

# Stage 2: Rank transform
if flex_cal._full_ecdf is not None:
    ranked_values = flex_cal._full_ecdf(g_values)
else:
    ranked_values = judge_grid  # Fallback

# Final: Isotonic on ranked space
final_calibrated = calibrator.predict(judge_grid)

# Create visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

# Panel 1: Raw data with empirical means
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(judge_scores, oracle_labels, alpha=0.05, s=8, color="gray", label="Samples")

# Compute binned empirical means
bin_width = 0.05
bin_edges = np.arange(0, 1.0 + bin_width, bin_width)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
binned_means = []

for i in range(len(bin_edges) - 1):
    if i == len(bin_edges) - 2:
        mask = (judge_scores >= bin_edges[i]) & (judge_scores <= bin_edges[i + 1])
    else:
        mask = (judge_scores >= bin_edges[i]) & (judge_scores < bin_edges[i + 1])
    if np.any(mask):
        binned_means.append(np.mean(oracle_labels[mask]))
    else:
        binned_means.append(np.nan)

valid_bins = ~np.isnan(binned_means)
ax1.scatter(
    bin_centers[valid_bins],
    np.array(binned_means)[valid_bins],
    s=60,
    alpha=0.8,
    color="darkblue",
    edgecolor="white",
    linewidth=1,
    label="Empirical E[Oracle|Judge]",
    zorder=10,
)

ax1.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Perfect (y=x)")
ax1.set_xlabel("Judge Score", fontsize=11)
ax1.set_ylabel("Oracle Label", fontsize=11)
ax1.set_title(
    "① Raw Data: Judge Scores vs Oracle Labels", fontsize=12, fontweight="bold"
)
ax1.grid(True, alpha=0.3)
ax1.set_xlim((0, 1))
ax1.set_ylim((0, 1))
ax1.legend(loc="upper left", fontsize=9)

# Panel 2: Stage 1 - Smooth g(S) transformation
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(
    judge_grid,
    g_values,
    "-",
    color="purple",
    linewidth=3,
    label="g(S) = spline(Judge)",
    alpha=0.8,
)
ax2.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Identity (y=x)")
ax2.set_xlabel("Judge Score", fontsize=11)
ax2.set_ylabel("g(Judge)", fontsize=11)
ax2.set_title("② Stage 1: Smooth Transformation g(S)", fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.set_xlim((0, 1))
ax2.legend(loc="upper left", fontsize=9)

# Add explanation text
explanation = (
    "Flexible spline learns non-monotone\n"
    "patterns (e.g., length bias at fixed\n"
    "judge score)"
)
ax2.text(
    0.05,
    0.95,
    explanation,
    transform=ax2.transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lavender", alpha=0.7),
)

# Panel 3: Stage 2 - Rank transformation
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(
    g_values,
    ranked_values,
    "-",
    color="orange",
    linewidth=3,
    label="rank_transform(g(S))",
    alpha=0.8,
)
ax3.plot(
    [g_values.min(), g_values.max()],
    [0, 1],
    "--",
    color="gray",
    alpha=0.5,
    label="Monotone reference",
)
ax3.set_xlabel("g(Judge)", fontsize=11)
ax3.set_ylabel("Ranked Index ∈ [0,1]", fontsize=11)
ax3.set_title(
    "③ Stage 2: Rank Transform (Enforces Monotonicity)", fontsize=12, fontweight="bold"
)
ax3.grid(True, alpha=0.3)
ax3.legend(loc="upper left", fontsize=9)

# Add explanation text
explanation = "ECDF rank transform ensures\n" "monotonicity before isotonic fit"
ax3.text(
    0.05,
    0.95,
    explanation,
    transform=ax3.transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7),
)

# Panel 4: Final calibration function
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(judge_scores, oracle_labels, alpha=0.05, s=8, color="gray", label="Samples")
ax4.scatter(
    bin_centers[valid_bins],
    np.array(binned_means)[valid_bins],
    s=60,
    alpha=0.7,
    color="darkblue",
    edgecolor="white",
    linewidth=1,
    label="Empirical mean",
    zorder=10,
)
ax4.plot(
    judge_grid,
    final_calibrated,
    "-",
    color="red",
    linewidth=3,
    label="Two-stage f(Judge)",
    alpha=0.9,
    zorder=12,
)
ax4.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Perfect (y=x)")
ax4.set_xlabel("Judge Score", fontsize=11)
ax4.set_ylabel("Calibrated Reward", fontsize=11)
ax4.set_title("④ Final: Two-Stage Calibration Function", fontsize=12, fontweight="bold")
ax4.grid(True, alpha=0.3)
ax4.set_xlim((0, 1))
ax4.set_ylim((0, 1))
ax4.legend(loc="upper left", fontsize=9)

# Compute final RMSE
rmse = np.sqrt(np.mean((oracle_labels - result.calibrated_scores) ** 2))

stats_text = (
    f"RMSE: {rmse:.3f}\n"
    f"Range: [{result.calibrated_scores.min():.3f}, {result.calibrated_scores.max():.3f}]\n"
    f"Mode: two_stage"
)
ax4.text(
    0.98,
    0.02,
    stats_text,
    transform=ax4.transAxes,
    fontsize=9,
    horizontalalignment="right",
    verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
)

# Overall title
fig.suptitle(
    "Two-Stage Calibration Pipeline: g(S) → rank → isotonic",
    fontsize=14,
    fontweight="bold",
    y=0.995,
)

# Save plot
output_path = Path(__file__).parent / "two_stage_detailed.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"\n✓ Detailed pipeline plot saved to {output_path}")
print("\nPipeline stages:")
print("  ① Raw data: Judge scores vs oracle labels")
print("  ② Stage 1: Smooth spline transformation g(S)")
print("  ③ Stage 2: Rank transform for monotonicity")
print("  ④ Final: Isotonic regression on ranked space")
print("\nTwo-stage allows flexible shape while maintaining monotonicity guarantee.")
