# Diagnostics: When to Trust Your Results

**The Safety Net.** CJE provides extensive diagnostics to help you know when results are reliable—and when they're not.

---

## The Pre-Flight Checklist

Before trusting any CJE result, verify:

- [ ] **ESS > 10%** (for IPS/DR modes)
- [ ] **No NaN estimates** (CJE refuses garbage)
- [ ] **Calibration R² > 0.3** (judge predicts oracle)
- [ ] **CI width reasonable** (not ±0.5 for a [0,1] outcome)
- [ ] **Oracle coverage > 5%** (enough labels to calibrate)

---

## Understanding Standard Errors

### Why CJE Standard Errors Are Larger

CJE's confidence intervals account for **all** sources of uncertainty:

| Source | What It Captures |
|:-------|:-----------------|
| Sampling variance | Finite number of prompts |
| Calibration uncertainty | Oracle labels are sparse |
| Weight uncertainty | Importance weights are estimated |
| Clustering | Correlation across policies on same prompts |

**This is honest.** Other tools often ignore calibration uncertainty, giving you artificially tight CIs.

### OUA: Oracle Uncertainty Aware

CJE uses delete-one-fold jackknife to measure how much uncertainty comes from the calibrator itself:

```python
# Check OUA contribution
if "se_components" in results.metadata:
    components = results.metadata["se_components"]
    if components.get("includes_oracle_uncertainty"):
        print("Oracle uncertainty included in standard errors")
```

### Reading the Components

```python
# Variance decomposition (if available)
if hasattr(results, 'var_cluster_robust') and hasattr(results, 'var_oua'):
    total = results.var_cluster_robust + results.var_oua
    oua_share = results.var_oua / total
    print(f"OUA share: {oua_share:.1%}")

    if oua_share > 0.5:
        print("→ Oracle labels are the bottleneck. Add more labels.")
    else:
        print("→ Sampling is the bottleneck. Add more prompts.")
```

---

## Effective Sample Size (ESS)

**The #1 diagnostic for IPS/DR modes.**

```
ESS = (Σwᵢ)² / Σwᵢ²
ESS_fraction = ESS / n
```

### Interpretation

| ESS Fraction | Status | Action |
|:-------------|:-------|:-------|
| > 30% | ✅ Good | Results reliable |
| 10-30% | ⚠️ Marginal | Check other diagnostics |
| 1-10% | 🔴 Poor | Consider DR mode |
| < 1% | ❌ Catastrophic | Don't trust results |

### Checking ESS

```python
# Overall ESS
print(f"ESS: {results.diagnostics.weight_ess:.1%}")

# Per-policy ESS
for policy, ess in results.diagnostics.ess_per_policy.items():
    status = "✅" if ess > 0.1 else "🔴"
    print(f"{status} {policy}: {ess:.1%}")
```

### Fixing Low ESS

1. **Use DR mode** — Outcome model provides stability
2. **Restrict to similar policies** — Don't compare GPT-4 → Llama-7B
3. **Collect more diverse logs** — Improve overlap
4. **Check for data issues** — Missing logprobs, wrong policies

---

## Weight Diagnostics

### Maximum Weight

A single sample shouldn't dominate the estimate:

```python
for policy, max_w in results.diagnostics.max_weight_per_policy.items():
    if max_w > 100:
        print(f"⚠️ {policy}: max weight = {max_w:.0f} (too high)")
```

| Max Weight | Status | Meaning |
|:-----------|:-------|:--------|
| < 10 | ✅ Good | Balanced contributions |
| 10-100 | ⚠️ Marginal | Some concentration |
| > 100 | 🔴 Poor | Few samples dominate |

### Hill Tail Index

Measures how "heavy-tailed" the weight distribution is:

```python
if hasattr(results.diagnostics, 'tail_indices'):
    for policy, alpha in results.diagnostics.tail_indices.items():
        if alpha < 2:
            print(f"⚠️ {policy}: tail index = {alpha:.2f} (infinite variance)")
```

| Tail Index | Meaning |
|:-----------|:--------|
| α > 2 | ✅ Finite variance |
| 1 < α < 2 | ⚠️ Infinite variance |
| α < 1 | ❌ Infinite mean |

---

## Refusal Gates

CJE sometimes returns `NaN` instead of an estimate. **This is intentional.**

### Why CJE Refuses

When data quality is too poor, any estimate would be misleading. CJE refuses rather than give garbage:

```python
import numpy as np

for i, policy in enumerate(results.metadata['target_policies']):
    if np.isnan(results.estimates[i]):
        print(f"❌ {policy}: CJE refused to estimate")
```

### Gate Conditions

CJE refuses when:
- ESS < 30%
- More than 85% of raw weights near zero
- Top 5% of weights hold >30% of total AND CV > 2.0

### Overriding Refusal (Not Recommended)

```python
# Get estimates even when quality is poor (use with caution!)
results = analyze_dataset(
    ...,
    estimator_config={"refuse_unreliable": False}
)
```

> ⚠️ **Warning:** Overriding refusal gates means you accept responsibility for potentially meaningless estimates.

---

## Calibration Diagnostics

### R² (Coefficient of Determination)

How well judge scores predict oracle labels:

```python
if hasattr(results, 'calibration_r2'):
    r2 = results.calibration_r2
    if r2 < 0.3:
        print(f"⚠️ Calibration R² = {r2:.3f} (judge poorly predicts oracle)")
```

| R² | Quality |
|:---|:--------|
| > 0.7 | ✅ Excellent |
| 0.5-0.7 | Good |
| 0.3-0.5 | ⚠️ Moderate |
| < 0.3 | 🔴 Poor |

### RMSE

Root mean squared error of calibration:

```python
if hasattr(results, 'calibration_rmse'):
    rmse = results.calibration_rmse
    print(f"Calibration RMSE: {rmse:.3f}")
```

### Coverage

Does calibrator's training range cover the test data?

```python
# Check if any scores are outside calibration range
if hasattr(results.diagnostics, 'coverage_warnings'):
    for warning in results.diagnostics.coverage_warnings:
        print(f"⚠️ {warning}")
```

---

## DR-Specific Diagnostics

### Outcome Model R²

How well the outcome model predicts rewards:

```python
if hasattr(results.diagnostics, 'outcome_r2_range'):
    min_r2, max_r2 = results.diagnostics.outcome_r2_range
    print(f"Outcome R² range: [{min_r2:.3f}, {max_r2:.3f}]")

    if min_r2 < 0:
        print("⚠️ Negative R² indicates model misspecification")
```

### DM vs IPS Decomposition

How much of the estimate comes from each component:

```python
if hasattr(results.diagnostics, 'dm_ips_decompositions'):
    for policy, decomp in results.diagnostics.dm_ips_decompositions.items():
        dm = decomp.get('dm_contribution', 0)
        ips = decomp.get('ips_contribution', 0)
        total = dm + ips
        print(f"{policy}: {dm/total:.0%} DM + {ips/total:.0%} IPS")
```

---

## Common Problems & Solutions

### Problem: NaN Estimates

**Diagnosis:**
```python
if np.isnan(results.estimates[i]):
    print(f"Policy {i} refused")
    print(f"ESS: {results.diagnostics.ess_per_policy.get(policy)}")
```

**Solutions:**
1. Use DR mode with fresh draws
2. Restrict to more similar policies
3. Check for missing/corrupt logprobs

### Problem: Very Wide CIs

**Diagnosis:**
```python
for i, se in enumerate(results.standard_errors):
    if se > 0.1:  # Relative to [0,1] scale
        print(f"Wide CI for policy {i}: ±{1.96*se:.3f}")
```

**Solutions:**
1. Add more prompts (sampling variance)
2. Add more oracle labels (calibration uncertainty)
3. Improve judge quality

### Problem: ESS Very Low

**Diagnosis:**
```python
if results.diagnostics.weight_ess < 0.05:
    print("Catastrophic overlap")
```

**Solutions:**
1. Switch to DR mode
2. Check policy similarity
3. Verify logprobs are correct

### Problem: Estimates Far from Oracle

**Diagnosis:**
```python
# If you have oracle values to compare
for policy, est in results.estimates.items():
    oracle = oracle_values.get(policy)
    if oracle and abs(est - oracle) > 0.1:
        print(f"⚠️ {policy}: estimate {est:.3f} vs oracle {oracle:.3f}")
```

**Solutions:**
1. Check calibration quality
2. Increase oracle coverage
3. Verify data pipeline

---

## Exporting Diagnostics

### To Dictionary

```python
diag_dict = results.diagnostics.to_dict()
```

### To JSON

```python
import json
with open("diagnostics.json", "w") as f:
    f.write(results.diagnostics.to_json())
```

### To CSV Row

```python
import pandas as pd
df = pd.DataFrame([results.diagnostics.to_csv_row()])
df.to_csv("diagnostics.csv", index=False)
```

---

## Reference

For implementation details, see [Diagnostics Module](../cje/diagnostics/README.md).
