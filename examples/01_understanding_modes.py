#!/usr/bin/env python3
"""
Example 1: Understanding CJE's Three Analysis Modes

CJE automatically selects the best analysis mode based on your data.
This example demonstrates the three modes and how they differ.

Three Modes:
------------
1. IPS Mode: Off-policy evaluation via importance sampling
   - Estimand: "What would deployment value be if we switched policies?"
   - Data needed: Logged data with logprobs, no fresh draws
   - Default estimator: calibrated-ips

2. DR Mode: Doubly robust (combines IPS with outcome models)
   - Estimand: "What would deployment value be?" (most accurate)
   - Data needed: Logged data with logprobs AND fresh draws
   - Default estimator: stacked-dr

3. Direct Mode: On-policy evaluation
   - Estimand: "Which policy performs best on this evaluation set?"
   - Data needed: Fresh draws (with optional logged data for calibration)
   - Default estimator: direct

Key Difference:
---------------
IPS/DR modes: Estimate counterfactual deployment value (causal inference)
Direct mode: Compare policies on your specific eval set (non-counterfactual)

Note: Mode is determined by your DATA. Within each mode, you can choose
different ESTIMATORS (e.g., calibrated-ips vs raw-ips for IPS mode).
"""

from pathlib import Path
from cje import analyze_dataset

DATA_PATH = Path(__file__).parent / "arena_sample" / "logged_data.jsonl"
FRESH_DRAWS_DIR = Path(__file__).parent / "arena_sample" / "fresh_draws"

print("=" * 70)
print("CJE's Three Analysis Modes")
print("=" * 70)
print()

# Mode 1: IPS (logged data only, auto-selects calibrated-ips estimator)
print("Mode 1: IPS (Importance Sampling)")
print("-" * 70)
print("Data: Logged responses with logprobs")
print("Estimand: Counterfactual deployment value")
print()

results_ips = analyze_dataset(
    logged_data_path=str(DATA_PATH),
    estimator="auto",  # Auto-selects mode based on data
    verbose=False,
)
print(f"Detected mode: {results_ips.metadata['mode']}")
print(f"Selected estimator: {results_ips.metadata['estimator']}")
print(f"Estimates: {results_ips.estimates}")
print(f"Method: Reweight logged data using importance weights")
print()

# Mode 2: DR (logged data + fresh draws, auto-selects stacked-dr estimator)
print("Mode 2: DR (Doubly Robust)")
print("-" * 70)
print("Data: Logged responses + fresh draws from target policies")
print("Estimand: Counterfactual deployment value (more accurate)")
print()

results_dr = analyze_dataset(
    logged_data_path=str(DATA_PATH),
    fresh_draws_dir=str(FRESH_DRAWS_DIR),
    estimator="auto",  # Auto-selects mode based on data
    estimator_config={"parallel": False},
    verbose=False,
)
print(f"Detected mode: {results_dr.metadata['mode']}")
print(f"Selected estimator: {results_dr.metadata['estimator']}")
print(f"Estimates: {results_dr.estimates}")
print(f"Method: Combine importance weights with outcome models")
print()

# Mode 3: Direct (fresh draws only - learns calibration from oracle labels in fresh draws)
print("Mode 3: Direct (On-Policy Evaluation)")
print("-" * 70)
print("Data: Fresh responses from target policies (with 50% oracle coverage)")
print("Estimand: Performance on this specific evaluation set")
print()

results_direct = analyze_dataset(
    fresh_draws_dir=str(FRESH_DRAWS_DIR),  # Fresh draws only, no logged data!
    estimator="auto",  # Auto-selects Direct mode
    verbose=False,
)
print(f"Detected mode: {results_direct.metadata['mode']}")
print(f"Selected estimator: {results_direct.metadata['estimator']}")
print(f"Estimates: {results_direct.estimates}")
print(f"Method: Average calibrated rewards (calibration learned from fresh draws)")
print(f"Calibration: {results_direct.metadata.get('calibration', 'none')}")
print(f"Oracle coverage: {results_direct.metadata.get('oracle_coverage', 0):.0%}")
print()

# Summary comparison
print("=" * 70)
print("Mode Comparison")
print("=" * 70)
print(f"{'Mode':<20} {'Counterfactual?':<20} {'Data Requirements':<30}")
print("-" * 70)
print(f"{'IPS':<20} {'Yes':<20} {'Logged data + logprobs':<30}")
print(f"{'DR':<20} {'Yes':<20} {'Logged data + fresh draws':<30}")
print(f"{'Direct':<20} {'No':<20} {'Fresh draws + calibration':<30}")
print()

print("When to use each mode:")
print("  • IPS: Logged data with logprobs, want counterfactual estimates")
print("  • DR: Have both logged data and fresh draws, want most accurate estimates")
print("  • Direct: Have fresh draws, just want on-policy comparison")
print()
print("Within each mode, you can choose different estimators:")
print("  • IPS mode: calibrated-ips (default), raw-ips")
print("  • DR mode: stacked-dr (default), dr-cpo, tmle, mrdr")
print("  • Direct mode: direct (calibrated if logged data available)")
print()
print("Pro tip: Use estimator='auto' and CJE will choose the mode AND")
print("         the best default estimator for you!")
