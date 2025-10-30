#!/usr/bin/env python3
"""
CJE Quickstart - Evaluate LLM policies in 15 lines
==================================================

This example shows the most common workflow:
1. Load your evaluation data (logged responses with judge scores)
2. Run CJE analysis (automatic mode selection)
3. Get policy estimates with confidence intervals

Dataset: 1000 Arena prompts with 3 LLM policies
Oracle: GPT-5 labels (simulated ground truth)
Judge: GPT-4.1-nano (cheap proxy)
"""

from pathlib import Path
from cje import analyze_dataset

# Load Arena sample data (logged responses from base policy)
DATA_DIR = Path(__file__).parent / "arena_sample"
DATA_PATH = DATA_DIR / "logged_data.jsonl"
FRESH_DRAWS_DIR = DATA_DIR / "fresh_draws"

# Run analysis with doubly robust estimation (most accurate)
# Auto-mode detects logged data + fresh draws → selects DR mode
results = analyze_dataset(
    logged_data_path=str(DATA_PATH),
    fresh_draws_dir=str(FRESH_DRAWS_DIR),
    verbose=True,  # Show diagnostic info
)

# Display results with 95% confidence intervals
print("\n" + "=" * 70)
print("Policy Performance Estimates")
print("=" * 70)
cis = results.ci()  # 95% CIs by default
for i, policy in enumerate(results.metadata["target_policies"]):
    est = results.estimates[i]
    se = results.standard_errors[i]
    lower, upper = cis[i]
    print(f"{policy:30s}  {est:.3f} ± {se:.3f}  [{lower:.3f}, {upper:.3f}]")

# Check reliability
print(f"\nMode: {results.metadata['mode']}")
print(f"Estimator: {results.metadata['estimator']}")
print(f"Oracle coverage: {results.metadata.get('oracle_coverage', 0):.0%}")

# Next steps:
# - Add include_response_length=True for two-stage calibration
# - Try estimator="stacked-dr" for ensemble DR methods
# - Check diagnostics: results.diagnostics.weight_ess for overlap quality
# - See cje_tutorial.ipynb for comprehensive walkthrough
