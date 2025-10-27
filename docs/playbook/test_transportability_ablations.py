#!/usr/bin/env python
"""Test transportability of judge→oracle calibrator across policies.

Learns calibrator on base policy, then audits transport to each target policy
using their fresh draws as probes. Shows whether the mapping generalizes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from collections import defaultdict

from cje import load_dataset_from_jsonl
from cje.calibration import calibrate_dataset
from cje.diagnostics.transport import audit_transportability, TransportDiagnostics
from cje.data.fresh_draws import load_fresh_draws_auto
from cje.data.models import Sample

# Configuration
DATA_DIR = Path(__file__).parent.parent.parent / "cje/experiments/arena_10k_simplified/data"
DATASET_PATH = DATA_DIR / "cje_dataset.jsonl"
FRESH_DRAWS_DIR = DATA_DIR / "fresh_draws"

# Subsample size for faster testing
SUBSAMPLE_SIZE = None  # Use all base policy data (don't subsample)
ORACLE_COVERAGE = 0.25  # 25% oracle coverage
PROBE_SIZE = 200  # Number of fresh draws to use as probe per policy (increased)
SEED = 42

print("=" * 70)
print("TRANSPORTABILITY TEST: Base Policy → Target Policies")
print("=" * 70)

# Set seed
np.random.seed(SEED)

# ========== Step 1: Load and prepare base policy data ==========
print("\n1. Loading base policy data...")
dataset = load_dataset_from_jsonl(str(DATASET_PATH))
print(f"   Loaded {dataset.n_samples} base policy samples")

# Subsample to specified size
if SUBSAMPLE_SIZE and SUBSAMPLE_SIZE < len(dataset.samples):
    indices = np.random.choice(len(dataset.samples), SUBSAMPLE_SIZE, replace=False)
    dataset.samples = [dataset.samples[i] for i in sorted(indices)]
    print(f"   Subsampled to {len(dataset.samples)} samples")

# Mask oracle labels to simulate partial oracle coverage
n_with_oracle = sum(1 for s in dataset.samples if s.oracle_label is not None)
n_keep = int(n_with_oracle * ORACLE_COVERAGE)
oracle_indices = [i for i, s in enumerate(dataset.samples) if s.oracle_label is not None]
keep_indices = set(np.random.choice(oracle_indices, n_keep, replace=False))

for i, sample in enumerate(dataset.samples):
    if i not in keep_indices and sample.oracle_label is not None:
        sample.oracle_label = None

print(f"   Kept oracle labels for {n_keep}/{n_with_oracle} samples ({ORACLE_COVERAGE:.0%})")

# ========== Step 2: Fit calibrator on base policy ==========
print("\n2. Fitting calibrator on base policy...")
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    judge_field="judge_score",
    oracle_field="oracle_label",
    calibration_mode="auto",  # Let it choose monotone vs two-stage
    enable_cross_fit=True,
    n_folds=5,
    random_seed=SEED,
)

calibrator = cal_result.calibrator
print(f"   ✓ Fitted calibrator (mode: {calibrator.selected_mode if hasattr(calibrator, 'selected_mode') else 'unknown'})")
print(f"   ✓ Calibration RMSE: {cal_result.calibration_rmse:.4f}")

# ========== Step 3: Load fresh draws for each target policy ==========
print("\n3. Loading fresh draws for target policies...")
target_policies = ["clone", "parallel_universe_prompt", "premium", "unhelpful"]

fresh_draws_by_policy = {}
for policy in target_policies:
    try:
        fresh_draws = load_fresh_draws_auto(DATA_DIR, policy, verbose=False)
        # Sample PROBE_SIZE draws
        if len(fresh_draws.samples) > PROBE_SIZE:
            indices = np.random.choice(len(fresh_draws.samples), PROBE_SIZE, replace=False)
            fresh_draws.samples = [fresh_draws.samples[i] for i in sorted(indices)]

        # Convert to Sample objects with required fields
        probe_samples = []
        for fd_sample in fresh_draws.samples:
            # Create Sample with judge_score and oracle_label
            sample = Sample(
                prompt_id=fd_sample.prompt_id,
                prompt=f"prompt_{fd_sample.prompt_id}",  # Dummy
                response=fd_sample.response if fd_sample.response else "",
                base_policy="base",
                base_policy_logprob=0.0,  # Dummy
                target_policy_logprobs={policy: 0.0},  # Dummy
                judge_score=fd_sample.judge_score,  # Top-level field
                oracle_label=fd_sample.oracle_label,
                metadata={
                    "judge_score": fd_sample.judge_score,
                    "oracle_label": fd_sample.oracle_label,
                }
            )
            probe_samples.append(sample)

        fresh_draws_by_policy[policy] = probe_samples
        print(f"   ✓ {policy}: {len(probe_samples)} probe samples")
    except FileNotFoundError:
        print(f"   ✗ {policy}: No fresh draws found")
        continue

# ========== Step 4: Run transportability audit for each policy ==========
print("\n4. Running transportability audits...")
print("=" * 70)

results: Dict[str, TransportDiagnostics] = {}

for policy in target_policies:
    if policy not in fresh_draws_by_policy:
        continue

    probe_samples = fresh_draws_by_policy[policy]

    # Run audit
    diag = audit_transportability(
        calibrator=calibrator,
        probe_samples=probe_samples,
        bins=10,
        group_label=f"policy:{policy}"
    )

    results[policy] = diag

    # Print summary
    print(f"\n{policy.upper()}")
    print("-" * 70)
    print(diag.summary())

# ========== Step 5: Create summary table ==========
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

print(f"{'Policy':<25} {'Status':<8} {'δ̂':<10} {'CI(δ̂)':<20} {'Worst Bin':<12} {'Cov':<8} {'Action':<20}")
print("-" * 110)

for policy in target_policies:
    if policy not in results:
        continue

    diag = results[policy]

    # Compute worst bin residual
    worst_bin = max(abs(r) for r in diag.decile_residuals if not np.isnan(r))

    print(f"{policy:<25} {diag.status:<8} {diag.delta_hat:>+9.3f} "
          f"[{diag.delta_ci[0]:>+6.3f}, {diag.delta_ci[1]:>+6.3f}] "
          f"{worst_bin:>11.3f} {diag.coverage:>7.1%} {diag.recommended_action:<20}")

print("-" * 110)

# ========== Step 6: Create visualization ==========
print("\n5. Creating visualization...")

n_policies = len(results)
fig, axes = plt.subplots(2, n_policies, figsize=(5 * n_policies, 8))
if n_policies == 1:
    axes = axes.reshape(2, 1)

status_colors = {
    "PASS": "#2ecc71",
    "WARN": "#f39c12",
    "FAIL": "#e74c3c",
}

for idx, (policy, diag) in enumerate(results.items()):
    ax_resid = axes[0, idx]
    ax_counts = axes[1, idx]

    status_color = status_colors.get(diag.status, "#95a5a6")

    # Top: Decile residuals
    deciles = np.arange(len(diag.decile_residuals))
    residuals = np.array(diag.decile_residuals)
    counts = np.array(diag.decile_counts)

    # Plot bars
    for i, (r, c) in enumerate(zip(residuals, counts)):
        if np.isnan(r):
            ax_resid.bar(i, 0, color="lightgray", alpha=0.3)
        else:
            alpha = 0.7 if abs(r) > 0.05 else 0.5
            ax_resid.bar(i, r, color=status_color, alpha=alpha, edgecolor="black", linewidth=0.5)

    # Add threshold lines
    ax_resid.axhline(0.05, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax_resid.axhline(-0.05, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax_resid.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)

    # Add mean shift line
    if abs(diag.delta_hat) > 0.01:
        ax_resid.axhline(diag.delta_hat, color="blue", linestyle=":", linewidth=2, alpha=0.7)

    ax_resid.set_title(f"{policy}\n{diag.status}", fontsize=11, fontweight="bold")
    ax_resid.set_xlabel("Decile", fontsize=9)
    ax_resid.set_ylabel("Mean Residual", fontsize=9)
    ax_resid.grid(True, alpha=0.3, axis="y")
    ax_resid.set_ylim(-0.15, 0.15)

    # Bottom: Sample counts
    for i, c in enumerate(counts):
        color = "#2ecc71" if c >= 3 else "#f39c12"
        ax_counts.bar(i, c, color=color, alpha=0.6, edgecolor="darkgray", linewidth=0.5)

    ax_counts.axhline(diag.n_probe / len(deciles), color="blue", linestyle=":", linewidth=1.5, alpha=0.6)
    ax_counts.set_xlabel("Decile", fontsize=9)
    ax_counts.set_ylabel("Count", fontsize=9)
    ax_counts.set_title("Coverage", fontsize=10)
    ax_counts.grid(True, alpha=0.3, axis="y")

# Overall title
fig.suptitle(
    f"Transportability: Base Policy Calibrator → Target Policies\n"
    f"(n={SUBSAMPLE_SIZE}, oracle={ORACLE_COVERAGE:.0%}, probe={PROBE_SIZE}/policy)",
    fontsize=13,
    fontweight="bold",
    y=0.98
)

plt.tight_layout()

# Save
output_path = Path(__file__).parent / "transportability_test.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"   ✓ Saved visualization to {output_path}")

# ========== Step 7: Export results ==========
output_json = Path(__file__).parent / "transportability_test.json"
export_data = {
    "config": {
        "subsample_size": SUBSAMPLE_SIZE,
        "oracle_coverage": ORACLE_COVERAGE,
        "probe_size": PROBE_SIZE,
        "seed": SEED,
    },
    "calibrator": {
        "mode": calibrator.selected_mode if hasattr(calibrator, "selected_mode") else "unknown",
        "rmse": float(cal_result.calibration_rmse),
    },
    "results": {
        policy: diag.to_dict() for policy, diag in results.items()
    }
}

with open(output_json, "w") as f:
    json.dump(export_data, f, indent=2)

print(f"   ✓ Saved results to {output_json}")

# ========== Summary interpretation ==========
print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

n_pass = sum(1 for d in results.values() if d.status == "PASS")
n_warn = sum(1 for d in results.values() if d.status == "WARN")
n_fail = sum(1 for d in results.values() if d.status == "FAIL")

print(f"\nResults: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL (out of {len(results)})")

if n_pass == len(results):
    print("\n✓ Excellent! The base policy calibrator transports well to all target policies.")
    print("  You can safely reuse it for evaluation.")
elif n_fail == 0:
    print("\n⚠ Marginal transport. Some policies show small shifts or regional issues.")
    print("  Consider mean anchoring for WARN policies or collecting more oracle data.")
else:
    print("\n✗ Poor transport. Some policies fail the audit.")
    print("  Recommended: Refit calibrator with pooled data from multiple policies,")
    print("  or use two-stage calibration with a 'policy' covariate.")

print("\n" + "=" * 70)
