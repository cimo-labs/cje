#!/usr/bin/env python
"""Generate simple two-stage calibration visualization showing covariate effects.

Shows 3 key panels:
1. Judge score effect (holding length at mean)
2. Response length effect (holding judge at mean)
3. Final monotone mapping (risk index → calibrated reward)
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

# Extract features and labels
judge_scores_list = []
oracle_labels_list = []
response_lengths_list = []

for sample in samples:
    metadata = sample.get("metadata", {})
    judge_score = metadata.get("judge_score")
    oracle_label = metadata.get("oracle_label")

    if judge_score is not None and oracle_label is not None:
        judge_scores_list.append(judge_score)
        oracle_labels_list.append(oracle_label)
        response_lengths_list.append(len(sample.get("response", "")))

judge_scores = np.array(judge_scores_list)
oracle_labels = np.array(oracle_labels_list)
response_lengths = np.array(response_lengths_list)

# Normalize response length to [0, 1]
response_lengths_norm = (response_lengths - response_lengths.min()) / (
    response_lengths.max() - response_lengths.min()
)

print(f"Found {len(judge_scores)} samples")
print(f"Judge score range: [{judge_scores.min():.3f}, {judge_scores.max():.3f}]")
print(f"Response length range: [{response_lengths.min()}, {response_lengths.max()}]")

# Fit two-stage calibrator with covariates
print("\nFitting two-stage calibrator with covariates...")
additional_covariates = response_lengths_norm.reshape(-1, 1)

cal_two_stage = JudgeCalibrator(
    calibration_mode="two_stage", covariate_names=["response_length"]
)
result = cal_two_stage.fit_cv(
    judge_scores, oracle_labels, n_folds=5, covariates=additional_covariates
)

print(f"Two-stage - OOF RMSE: {result.oof_rmse:.6f}")

# Compute OOF R²
ss_res = np.sum((oracle_labels - result.calibrated_scores) ** 2)
ss_tot = np.sum((oracle_labels - np.mean(oracle_labels)) ** 2)
oof_r2 = 1 - (ss_res / ss_tot)
print(f"Two-stage - OOF R²: {oof_r2:.6f}")

# Get the fitted spline model
if hasattr(cal_two_stage, "_flexible_calibrator"):
    flex_cal = cal_two_stage._flexible_calibrator
    if hasattr(flex_cal, "_full_g_model"):
        spline = flex_cal._full_g_model
    else:
        spline = None
else:
    spline = None

# Create simple 3-panel visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# ====================
# Panel A: Judge Score Effect
# ====================
ax1 = axes[0]

mean_length = response_lengths_norm.mean()
judge_grid = np.linspace(0, 1, 100)

if spline is not None:
    cov_grid = np.column_stack([judge_grid, np.full_like(judge_grid, mean_length)])
    predicted = spline.predict(cov_grid)
else:
    predicted = cal_two_stage.predict(
        judge_grid, covariates=np.full_like(judge_grid, mean_length).reshape(-1, 1)
    )

# Plot x=y diagonal for reference
ax1.plot(
    [0, 1],
    [0, 1],
    "--",
    color="black",
    linewidth=2,
    alpha=0.4,
    zorder=8,
)

# Scatter plot (sample for visibility)
sample_idx = np.random.choice(
    len(judge_scores), size=min(2000, len(judge_scores)), replace=False
)
ax1.scatter(
    judge_scores[sample_idx],
    oracle_labels[sample_idx],
    alpha=0.15,
    s=8,
    color="gray",
    label="Data",
)

# Binned empirical means
bin_edges = np.linspace(0, 1, 21)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
binned_means = []
for i in range(len(bin_edges) - 1):
    mask = (judge_scores >= bin_edges[i]) & (judge_scores < bin_edges[i + 1])
    if np.any(mask):
        binned_means.append(np.mean(oracle_labels[mask]))
    else:
        binned_means.append(np.nan)

valid = ~np.isnan(binned_means)
ax1.scatter(
    bin_centers[valid],
    np.array(binned_means)[valid],
    s=120,
    alpha=1.0,
    color="steelblue",
    edgecolor="white",
    linewidth=2,
    label="Empirical mean",
    zorder=10,
)

# Prediction line
ax1.plot(
    judge_grid,
    predicted,
    "-",
    color="#D95319",
    linewidth=3.5,
    alpha=1.0,
    label=f"First-stage prediction\n(length fixed at mean)",
    zorder=11,
)

ax1.set_xlabel("Judge Score", fontsize=14, fontweight="bold")
ax1.set_ylabel("Oracle Score", fontsize=14, fontweight="bold")
ax1.set_title(
    "Stage 1: Judge Score Covariate",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=12, loc="upper left")
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])

# ====================
# Panel B: Response Length Effect
# ====================
ax2 = axes[1]

mean_judge = judge_scores.mean()
length_grid = np.linspace(0, 1, 100)

if spline is not None:
    cov_grid = np.column_stack([np.full_like(length_grid, mean_judge), length_grid])
    predicted_length = spline.predict(cov_grid)
else:
    predicted_length = cal_two_stage.predict(
        np.full_like(length_grid, mean_judge), covariates=length_grid.reshape(-1, 1)
    )

# Plot x=y diagonal for reference
ax2.plot(
    [0, 1],
    [0, 1],
    "--",
    color="black",
    linewidth=2,
    alpha=0.4,
    zorder=8,
)

# Scatter plot (sample)
ax2.scatter(
    response_lengths_norm[sample_idx],
    oracle_labels[sample_idx],
    alpha=0.15,
    s=8,
    color="gray",
    label="Data",
)

# Binned empirical means
bin_edges_len = np.linspace(0, 1, 21)
bin_centers_len = (bin_edges_len[:-1] + bin_edges_len[1:]) / 2
binned_means_len = []
for i in range(len(bin_edges_len) - 1):
    mask = (response_lengths_norm >= bin_edges_len[i]) & (
        response_lengths_norm < bin_edges_len[i + 1]
    )
    if np.any(mask):
        binned_means_len.append(np.mean(oracle_labels[mask]))
    else:
        binned_means_len.append(np.nan)

valid_len = ~np.isnan(binned_means_len)
ax2.scatter(
    bin_centers_len[valid_len],
    np.array(binned_means_len)[valid_len],
    s=120,
    alpha=1.0,
    color="steelblue",
    edgecolor="white",
    linewidth=2,
    label="Empirical mean",
    zorder=10,
)

# Prediction line
ax2.plot(
    length_grid,
    predicted_length,
    "-",
    color="#D95319",
    linewidth=3.5,
    alpha=1.0,
    label=f"First-stage prediction\n(judge fixed at mean)",
    zorder=11,
)

ax2.set_xlabel("Response Length (normalized)", fontsize=14, fontweight="bold")
ax2.set_ylabel("Oracle Score", fontsize=14, fontweight="bold")
ax2.set_title(
    "Stage 1: Response Length Covariate",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=12, loc="lower left")
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])

# ====================
# Panel C: Final Monotone Mapping
# ====================
ax3 = axes[2]

# Compute risk index for all samples
if spline is not None:
    covariates_full = np.column_stack([judge_scores, response_lengths_norm])
    risk_index_all = spline.predict(covariates_full)
else:
    risk_index_all = result.calibrated_scores

# Sort by risk index
sort_idx = np.argsort(risk_index_all)
risk_sorted = risk_index_all[sort_idx]
calib_sorted = result.calibrated_scores[sort_idx]

# Scatter (sample)
ax3.scatter(
    risk_index_all[sample_idx],
    oracle_labels[sample_idx],
    alpha=0.15,
    s=8,
    color="gray",
    label="Oracle labels",
)

# Binned means
risk_min, risk_max = risk_index_all.min(), risk_index_all.max()
bin_edges_risk = np.linspace(risk_min, risk_max, 21)
bin_centers_risk = (bin_edges_risk[:-1] + bin_edges_risk[1:]) / 2
binned_oracle_risk = []

for i in range(len(bin_edges_risk) - 1):
    mask = (risk_index_all >= bin_edges_risk[i]) & (
        risk_index_all < bin_edges_risk[i + 1]
    )
    if np.any(mask):
        binned_oracle_risk.append(np.mean(oracle_labels[mask]))
    else:
        binned_oracle_risk.append(np.nan)

valid_risk = ~np.isnan(binned_oracle_risk)
ax3.scatter(
    bin_centers_risk[valid_risk],
    np.array(binned_oracle_risk)[valid_risk],
    s=120,
    alpha=1.0,
    color="steelblue",
    edgecolor="white",
    linewidth=2,
    label="Empirical mean",
    zorder=10,
)

# Plot x=y diagonal for reference
ax3.plot(
    [0, 1],
    [0, 1],
    "--",
    color="black",
    linewidth=2,
    alpha=0.4,
    zorder=8,
)

# Plot monotone calibration function
unique_risk, unique_idx = np.unique(risk_sorted, return_index=True)
unique_calib = calib_sorted[unique_idx]

ax3.plot(
    unique_risk,
    unique_calib,
    "-",
    color="#D95319",
    linewidth=3.5,
    alpha=1.0,
    label="Isotonic calibration\n(Stage 2: enforces monotonicity)",
    zorder=12,
)

ax3.set_xlabel("Risk Index (Stage 1 output)", fontsize=14, fontweight="bold")
ax3.set_ylabel("Oracle Score", fontsize=14, fontweight="bold")
ax3.set_title(
    "Stage 2: Monotone Mapping",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=12, loc="upper left")
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])

# Overall title
fig.suptitle(
    "Two-Stage Calibration with Covariates",
    fontsize=17,
    fontweight="bold",
    y=1.02,
)

plt.tight_layout()

# Save plot
output_path = Path(__file__).parent / "two_stage_simple.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"\n✓ Simple two-stage plot saved to {output_path}")
print("\nKey insights:")
print("  • Left: Judge score has non-monotone relationship with oracle reward")
print("  • Middle: Response length has weak/flat effect")
print("  • Right: Stage 2 isotonic regression enforces monotonicity")
print("\nTwo-stage = flexible Stage 1 (spline) + monotone Stage 2 (isotonic)")
