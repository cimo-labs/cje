#!/usr/bin/env python
"""Generate simplified weight comparison plot showing raw vs calibrated weights.

Shows just two policies (clone and parallel_universe_prompt) with ESS improvement.
Removes variance ratio for cleaner visualization.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cje.data import Dataset
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.estimators import CalibratedIPS
from cje.calibration import calibrate_dataset
from cje.diagnostics import compute_ess

# Load arena data
data_path = (
    Path(__file__).parent.parent.parent
    / "cje/experiments/arena_10k_simplified/data/cje_dataset.jsonl"
)

print(f"Loading data from {data_path}...")
with open(data_path) as f:
    samples = [json.loads(line) for line in f]

print(f"Loaded {len(samples)} samples")

# Create dataset
dataset = Dataset(samples=samples)

# Calibrate dataset
print("Calibrating dataset...")
cal_result = calibrate_dataset(
    dataset,
    judge_field="judge_score",
    oracle_field="oracle_label",
    calibration_mode="monotone",
)

calibrated_dataset = cal_result.calibrated_dataset

# Create sampler
sampler = PrecomputedSampler(
    calibrated_dataset,
    target_policies=["clone", "parallel_universe_prompt"],
)

# Fit calibrated IPS estimator
print("Fitting calibrated IPS estimator...")
estimator = CalibratedIPS()
result = estimator.fit_and_estimate(sampler)

print("\nResults:")
for policy in ["clone", "parallel_universe_prompt"]:
    est = result.estimates.get(policy)
    ci = result.confidence_intervals.get(policy)
    if est is not None and ci is not None:
        print(f"  {policy}: {est:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

# Get weights
raw_weights_dict = {}
calibrated_weights_dict = {}

for policy in ["clone", "parallel_universe_prompt"]:
    raw_w = estimator.get_raw_weights(policy)
    cal_w = estimator.get_weights(policy)

    if raw_w is not None:
        raw_weights_dict[policy] = raw_w
    if cal_w is not None:
        calibrated_weights_dict[policy] = cal_w

# Get calibrated rewards (ordering indices)
ordering_indices_dict = {}
for policy in ["clone", "parallel_universe_prompt"]:
    data = sampler.get_data_for_policy(policy)
    if data:
        # Get calibrated rewards
        g_values = np.array([d.get("reward", np.nan) for d in data])
        valid = ~np.isfinite(g_values)
        if valid.sum() < len(g_values):  # Has some valid values
            ordering_indices_dict[policy] = g_values

# Create 1x2 plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

policy_names = {
    "clone": "clone",
    "parallel_universe_prompt": "parallel_universe_prompt"
}

for idx, policy in enumerate(["clone", "parallel_universe_prompt"]):
    ax = axes[idx]

    raw_w = raw_weights_dict[policy]
    cal_w = calibrated_weights_dict[policy]

    # Get ordering index (calibrated rewards)
    ordering_index = ordering_indices_dict.get(policy, None)

    # Compute ESS
    ess_raw = compute_ess(raw_w)
    ess_cal = compute_ess(cal_w)
    uplift = ess_cal / max(ess_raw, 1e-12)

    if ordering_index is not None and len(ordering_index) == len(raw_w):
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
        W_cal = cal_w[mask]

        n = len(S)

        # Sort by ordering index
        sort_idx = np.argsort(S)
        S_sorted = S[sort_idx]
        W_raw_sorted = W_raw[sort_idx]
        W_cal_sorted = W_cal[sort_idx]

        # Subsample for visibility if needed
        if n > 2000:
            step = max(1, n // 1000)
            indices = np.arange(0, n, step)
            ax.scatter(
                S_sorted[indices],
                W_raw_sorted[indices],
                s=2,
                alpha=0.2,
                color="gray",
                label="raw weights",
                zorder=1,
            )
        else:
            ax.scatter(
                S_sorted,
                W_raw_sorted,
                s=2,
                alpha=0.3,
                color="gray",
                label="raw weights",
                zorder=1,
            )

        # Plot calibrated weights as smooth line
        ax.plot(
            S_sorted,
            W_cal_sorted,
            "-",
            color="green",
            linewidth=2,
            alpha=0.9,
            label="calibrated",
            zorder=2,
        )

        # Add horizontal line for raw weights mean
        raw_mean = np.mean(W_raw_sorted)
        ax.axhline(
            raw_mean,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"raw mean ({raw_mean:.2f})",
            zorder=3,
        )
    else:
        # Fallback: just plot distributions
        ax.hist(raw_w, bins=50, alpha=0.5, label="raw weights", color="gray")
        ax.hist(cal_w, bins=50, alpha=0.5, label="calibrated", color="green")

    # Title with ESS improvement only
    ax.set_title(
        f"{policy_names[policy]}\nESS: {ess_raw:.0f}→{ess_cal:.0f} ({uplift:.1f}×)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Calibrated Reward g(s)", fontsize=11)
    ax.set_ylabel("Weight (log scale)", fontsize=11)
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=10, frameon=False)
    ax.grid(True, alpha=0.3)

# Overall title
fig.suptitle(
    "SIMCal Weight Calibration: Raw vs. Calibrated Weights",
    fontsize=14,
    fontweight="bold",
    y=1.00,
)

plt.tight_layout()

# Save plot
output_path = Path(__file__).parent / "weight_comparison.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"\n✓ Weight comparison plot saved to {output_path}")
print("\nShows:")
print("  • Gray scatter: Raw IPS weights (highly variable)")
print("  • Red dashed line: Empirical mean of raw weights")
print("  • Green line: SIMCal calibrated weights (smooth monotone function)")
print("  • ESS improvement demonstrates variance reduction from weight calibration")
