#!/usr/bin/env python3
"""
Example 2: Minimal CJE Usage

The simplest possible CJE workflow - load data and get estimates.
Auto mode will select the best estimator based on your data.
"""

from pathlib import Path
from cje import analyze_dataset

# Use the arena sample data included with CJE examples
DATA_PATH = Path(__file__).parent / "arena_sample" / "logged_data.jsonl"

# Run analysis (auto-selects calibrated-ips when no fresh draws provided)
results = analyze_dataset(logged_data_path=str(DATA_PATH))

# Print results with 95% confidence intervals
print("Policy Estimates:")
cis = results.ci()
for i, policy in enumerate(results.metadata["target_policies"]):
    lower, upper = cis[i]
    print(f"  {policy}: {results.estimates[i]:.3f} [{lower:.3f}, {upper:.3f}]")
