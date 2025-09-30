#!/usr/bin/env python3
"""
Example 3: Comparing Policies

Find the best policy and compare against a baseline.
"""

from pathlib import Path
from cje import analyze_dataset
import numpy as np

DATA_PATH = Path(__file__).parent.parent / "cje/tests/data/arena_sample/dataset.jsonl"

results = analyze_dataset(str(DATA_PATH), estimator="calibrated-ips")

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

# Compare all policies to baseline (first policy)
baseline_idx = 0
print(f"Comparison to baseline ({policies[baseline_idx]}):")
for i, policy in enumerate(policies):
    if i == baseline_idx:
        continue
    diff = estimates[i] - estimates[baseline_idx]
    # Note: This is a simplified comparison - proper inference would account for correlation
    print(f"  {policy:<30} {diff:+.3f}")
