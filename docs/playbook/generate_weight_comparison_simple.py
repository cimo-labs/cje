#!/usr/bin/env python
"""Generate simplified weight comparison plot from existing arena results.

Loads the existing weight dashboard data and creates a 2-policy version.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cje.interface import analyze_dataset
from cje.diagnostics import compute_ess

# Use the arena dataset
data_dir = Path(__file__).parent.parent.parent / "cje/experiments/arena_10k_simplified/data"
dataset_path = data_dir / "cje_dataset.jsonl"

print(f"Running analysis on {dataset_path}...")
print("This will take a minute...")

# Run analysis to get weights
results = analyze_dataset(
    logged_data_path=str(dataset_path),
    estimator="calibrated-ips",
    verbose=False,
)

print("\nExtracting weight data...")

# Get raw and calibrated weights from the estimator
estimator = results.metadata.get("estimator_obj")

if estimator is None:
    print("Error: Could not get estimator object from results")
    sys.exit(1)

# Filter to just the two policies we want
target_policies = ["clone", "parallel_universe_prompt"]

raw_weights_dict = {}
calibrated_weights_dict = {}
ordering_indices_dict = {}

# Get sampler to extract calibrated rewards
sampler = results.metadata.get("sampler")

for policy in target_policies:
    # Get weights
    raw_w = estimator.get_raw_weights(policy)
    cal_w = estimator.get_weights(policy)

    if raw_w is not None and cal_w is not None:
        raw_weights_dict[policy] = raw_w
        calibrated_weights_dict[policy] = cal_w

        # Get calibrated rewards (g(s)) for x-axis
        if sampler is not None:
            data = sampler.get_data_for_policy(policy)
            if data:
                g_values = np.array([d.get("reward", np.nan) for d in data])
                if np.isfinite(g_values).any():
                    ordering_indices_dict[policy] = g_values

print(f"Got weights for {len(raw_weights_dict)} policies")

# Create 1x2 plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

policy_names = {
    "clone": "clone",
    "parallel_universe_prompt": "parallel_universe_prompt"
}

for idx, policy in enumerate(target_policies):
    ax = axes[idx]

    raw_w = raw_weights_dict[policy]
    cal_w = calibrated_weights_dict[policy]
    ordering_index = ordering_indices_dict.get(policy, None)

    # Compute ESS
    ess_raw = compute_ess(raw_w)
    ess_cal = compute_ess(cal_w)
    uplift = ess_cal / max(ess_raw, 1e-12)

    print(f"{policy}: ESS {ess_raw:.0f}→{ess_cal:.0f} ({uplift:.1f}×)")

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
                s=3,
                alpha=0.25,
                color="gray",
                label="raw weights",
                zorder=1,
            )
        else:
            ax.scatter(
                S_sorted,
                W_raw_sorted,
                s=3,
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
            linewidth=2.5,
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
        print(f"Warning: No ordering index for {policy}, using fallback plot")
        # Fallback: just plot distributions
        ax.hist(raw_w, bins=50, alpha=0.5, label="raw weights", color="gray")
        ax.hist(cal_w, bins=50, alpha=0.5, label="calibrated", color="green")

    # Title with ESS improvement only (no variance ratio)
    ax.set_title(
        f"{policy_names[policy]}\nESS: {ess_raw:.0f}→{ess_cal:.0f} ({uplift:.1f}×)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Calibrated Reward g(s)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Weight (log scale)", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=11, frameon=False)
    ax.grid(True, alpha=0.3)

# Overall title
fig.suptitle(
    "SIMCal Weight Calibration: Raw vs. Calibrated Weights",
    fontsize=15,
    fontweight="bold",
    y=0.98,
)

plt.tight_layout()

# Save plot
output_path = Path(__file__).parent / "weight_comparison.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"\n✓ Weight comparison plot saved to {output_path}")
print("\nVisualization shows:")
print("  • Gray scatter: Raw IPS weights (highly variable)")
print("  • Red dashed line: Empirical mean of raw weights")
print("  • Green line: SIMCal calibrated weights (smooth monotone function)")
print("  • ESS improvement quantifies variance reduction from weight calibration")
