#!/usr/bin/env python3
"""
Example 1: Understanding CJE's Three Analysis Modes

CJE automatically selects the best analysis mode based on your data.
This example explains the three modes and when each is used.

Three Modes:
------------
1. Direct Mode: On-policy evaluation (fresh draws, no logprobs needed)
   - Estimand: "Which policy performs best on this evaluation set?"
   - Use when: You have responses from multiple policies, no logprobs

2. IPS Mode: Off-policy evaluation via importance sampling
   - Estimand: "What would deployment value be if we switched policies?"
   - Use when: You have logged data with logprobs, no fresh draws

3. DR Mode: Doubly robust (combines IPS with outcome models)
   - Estimand: "What would deployment value be?" (most accurate)
   - Use when: You have both logprobs and fresh draws

Key Difference:
---------------
Direct mode: Compares policies on your specific eval set (non-counterfactual)
IPS/DR modes: Estimates counterfactual deployment value (causal inference)
"""

from pathlib import Path
from cje import analyze_dataset

DATA_PATH = Path(__file__).parent / "arena_sample" / "logged_data.jsonl"
FRESH_DRAWS_DIR = Path(__file__).parent / "arena_sample" / "fresh_draws"

print("=" * 70)
print("CJE's Three Analysis Modes")
print("=" * 70)
print()

# Mode 1: IPS (logged data only)
print("Mode 1: IPS (Importance Sampling)")
print("-" * 70)
print("Data: Logged responses with logprobs")
print("Estimand: Counterfactual deployment value")
print()

results_ips = analyze_dataset(
    logged_data_path=str(DATA_PATH),
    estimator="calibrated-ips",
    verbose=False,
)
print(f"Mode selected: {results_ips.metadata['estimator']}")
print(f"Estimates: {results_ips.estimates}")
print(f"Method: Reweight logged data using importance weights")
print()

# Mode 2: DR (logged data + fresh draws)
print("Mode 2: DR (Doubly Robust)")
print("-" * 70)
print("Data: Logged responses + fresh draws from target policies")
print("Estimand: Counterfactual deployment value (more accurate)")
print()

results_dr = analyze_dataset(
    logged_data_path=str(DATA_PATH),
    fresh_draws_dir=str(FRESH_DRAWS_DIR),
    estimator="stacked-dr",
    estimator_config={"parallel": False},
    verbose=False,
)
print(f"Mode selected: {results_dr.metadata['estimator']}")
print(f"Estimates: {results_dr.estimates}")
print(f"Method: Combine importance weights with outcome models")
print()

# Mode 3: Direct (requires fresh draws)
print("Mode 3: Direct (On-Policy Evaluation)")
print("-" * 70)
print("Data: Fresh responses from target policies (requires fresh_draws_dir)")
print("Estimand: Performance on this specific evaluation set")
print()

results_direct = analyze_dataset(
    logged_data_path=str(DATA_PATH),
    fresh_draws_dir=str(FRESH_DRAWS_DIR),
    estimator="direct",
    verbose=False,
)
print(f"Mode selected: {results_direct.metadata['mode']}")
print(f"Estimates: {results_direct.estimates}")
print(f"Method: Average calibrated rewards on fresh draws")
print(f"Note: {results_direct.metadata.get('caveat', '')}")
print()

# Summary comparison
print("=" * 70)
print("Comparison")
print("=" * 70)
print(f"{'Mode':<20} {'Counterfactual?':<20} {'Needs Logged Data?':<20}")
print("-" * 70)
print(f"{'IPS':<20} {'Yes':<20} {'Yes':<20}")
print(f"{'DR':<20} {'Yes':<20} {'Yes (+ fresh draws)':<20}")
print(f"{'Direct':<20} {'No':<20} {'Optional (for calibration)':<20}")
print()

print("When to use each mode:")
print("  • IPS: Logged data with logprobs, want counterfactual estimates")
print("  • DR: Have both logged data and fresh draws, want most accurate estimates")
print("  • Direct: Have fresh draws, just want on-policy comparison")
print()
print("Note: Direct mode can work with just fresh_draws_dir (no logged data)")
print("      Adding logged data enables calibration for better accuracy")
print("Pro tip: Use estimator='auto' and CJE will choose for you!")
