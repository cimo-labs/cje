#!/usr/bin/env python3
"""
Example 2: Using Fresh Draws for Doubly-Robust Estimation

Fresh draws (new responses from target policies) enable more robust estimation
via doubly-robust methods. These are less sensitive to model misspecification.
"""

from pathlib import Path
from cje import analyze_dataset

# Paths to arena sample data (included with examples)
DATA_DIR = Path(__file__).parent / "arena_sample"
DATASET = DATA_DIR / "dataset.jsonl"
FRESH_DRAWS = DATA_DIR / "responses"  # Directory with {policy}_responses.jsonl files

# Auto mode selects stacked-dr when fresh draws are available
# Stacked-DR combines multiple DR estimators for robustness
results = analyze_dataset(
    str(DATASET),
    fresh_draws_dir=str(FRESH_DRAWS),
)

# Compare with calibrated-ips (no fresh draws needed)
results_ips = analyze_dataset(str(DATASET), estimator="calibrated-ips")

print("Stacked-DR vs Calibrated-IPS:")
print(f"{'Policy':<30} {'Stacked-DR':>12} {'Cal-IPS':>12}")
for i, policy in enumerate(results.metadata["target_policies"]):
    print(
        f"{policy:<30} {results.estimates[i]:>12.3f} {results_ips.estimates[i]:>12.3f}"
    )
