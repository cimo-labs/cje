#!/usr/bin/env python3
"""
Example 1: Minimal CJE Usage

The simplest possible CJE workflow - load data and get estimates.
"""

from pathlib import Path
from cje import analyze_dataset

# Use the arena sample data included with CJE examples
DATA_PATH = Path(__file__).parent / "arena_sample" / "dataset.jsonl"

# Run analysis
results = analyze_dataset(str(DATA_PATH), estimator="calibrated-ips")

# Print results with 95% confidence intervals
print("Policy Estimates:")
cis = results.ci()
for i, policy in enumerate(results.metadata["target_policies"]):
    lower, upper = cis[i]
    print(f"  {policy}: {results.estimates[i]:.3f} [{lower:.3f}, {upper:.3f}]")
