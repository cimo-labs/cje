#!/usr/bin/env python3
"""
Example 4: Comparing Policies

Find the best policy and compare against a baseline using proper statistical inference.
"""

from pathlib import Path
from cje import analyze_dataset
import numpy as np

DATA_PATH = Path(__file__).parent / "arena_sample" / "logged_data.jsonl"
FRESH_DRAWS_DIR = Path(__file__).parent / "arena_sample" / "fresh_draws"

# Auto mode selects stacked-dr when fresh draws are provided
results = analyze_dataset(
    logged_data_path=str(DATA_PATH),
    fresh_draws_dir=str(FRESH_DRAWS_DIR),
)

policies = results.metadata["target_policies"]
estimates = results.estimates
cis = results.ci()

# Find best policy
best_idx = np.argmax(estimates)
lower, upper = cis[best_idx]
print(
    f"🏆 Best policy: {policies[best_idx]} ({estimates[best_idx]:.3f} [{lower:.3f}, {upper:.3f}])"
)
print()

# Compare all policies to baseline using proper inference
baseline_idx = 0
print(f"Comparison to baseline ({policies[baseline_idx]}):")
print(f"  {'Policy':<30} {'Diff':>8} {'SE':>7} {'p-value':>8} {'Sig':>4}")
print(f"  {'-'*30} {'-'*8} {'-'*7} {'-'*8} {'-'*4}")

for i, policy in enumerate(policies):
    if i == baseline_idx:
        continue

    # Use built-in compare_policies() with influence functions
    # compare_policies(i, baseline) computes: policy_i - baseline
    # Positive difference means policy i is better than baseline
    comparison = results.compare_policies(i, baseline_idx)

    sig_marker = "*" if comparison["significant"] else ""
    print(
        f"  {policy:<30} {comparison['difference']:+8.3f} "
        f"{comparison['se_difference']:7.3f} "
        f"{comparison['p_value']:8.4f} "
        f"{sig_marker:>4}"
    )
