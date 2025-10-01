#!/usr/bin/env python3
"""
Example 4: Checking Estimate Reliability

CJE provides diagnostics to help you assess whether estimates are trustworthy.
Key metric: Effective Sample Size (ESS) - higher is better.
"""

from pathlib import Path
from cje import analyze_dataset

DATA_PATH = Path(__file__).parent / "arena_sample" / "dataset.jsonl"

# Auto mode selects calibrated-ips when no fresh draws provided
results = analyze_dataset(str(DATA_PATH))

# Check overall diagnostics
diag = results.diagnostics
print("Reliability Diagnostics:")
print(f"  Overall ESS: {diag.weight_ess:.1%}")
print(f"  Status: {diag.weight_status.value}")
print(f"  Overlap quality: {diag.overlap_quality}")
print()

# Check per-policy ESS
print("Per-Policy Effective Sample Size:")
for policy, ess in diag.ess_per_policy.items():
    status = "✓" if ess > 0.1 else "⚠️"
    print(f"  {status} {policy:<30} {ess:>6.1%}")
print()

# Interpretation
if diag.weight_ess < 0.1:
    print("⚠️  Low ESS - estimates may be unreliable. Consider:")
    print("   1. Using DR methods with fresh draws")
    print("   2. Evaluating policies closer to the logging policy")
else:
    print("✅ Estimates are reliable")
