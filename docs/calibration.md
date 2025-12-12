# Calibration: How AutoCal-R Fixes Your Judge

**The Core Problem:** Your LLM judge is biased. A raw score of 0.8 might actually mean 0.6 in reality.

**The Solution:** Learn the mapping from judge scores to ground truth using a small sample of oracle labels.

---

## Why Calibration Matters

### The Verbosity Problem

Most LLM judges reward length:

```
Short, correct answer:  Judge = 0.7
Long, verbose answer:   Judge = 0.9
Actual quality:         Both = 0.75
```

Without calibration, you'd wrongly prefer the verbose answer.

### The Sycophancy Problem

Judges often agree with confident-sounding responses:

```
Confident wrong answer: Judge = 0.8
Hedged correct answer:  Judge = 0.7
Actual quality:         0.3 vs 0.9
```

Calibration corrects for this by anchoring to ground truth.

---

## How AutoCal-R Works

### Step 1: Collect Oracle Labels

On 5-25% of your samples, collect "ground truth" labels:
- Human evaluation
- Stronger LLM (GPT-4, Claude)
- Objective metrics (correctness, factuality)

### Step 2: Learn the Mapping

CJE fits a function `f̂(S) → Y` that maps judge scores to oracle labels.

```
Judge Score (S)    →    f̂(S)    →    Calibrated Score
    0.8            →             →         0.65
    0.9            →             →         0.72
    0.7            →             →         0.71
```

### Step 3: Apply to All Samples

The learned mapping is applied to ALL judge scores, not just the ones with oracle labels.

---

## Calibration Methods

### 1. Monotone (Isotonic Regression)

**The Default.** Enforces: higher judge score → no worse expected outcome.

```
         Oracle
           │
      0.9  │           ●──────────
           │         ●/
      0.7  │    ●────●
           │   /
      0.5  │──●
           └────────────────────── Judge
              0.5   0.7   0.9
```

**Why it works:**
- Minimal assumptions (just monotonicity)
- Preserves oracle mean by construction
- Very efficient with small samples
- Won't create "perverse" regions where higher S → lower Y

**When to use:** Default choice. Works well when judge-oracle relationship is roughly monotone.

### 2. Two-Stage (+ Covariates)

**For complex biases.** Learns: `g(S, X) → rank → isotonic`

Stage 1: Fit a spline model incorporating covariates (e.g., response length)
Stage 2: Apply isotonic regression to the learned risk index

**Why it works:**
- Fixes length bias and other covariate effects
- Handles non-monotone regional patterns
- Still guarantees monotonicity in the final output

**When to use:**
- Judge shows length bias
- Different domains have different calibration
- Monotone has poor regional fit

### Auto Mode (Default)

CJE automatically chooses between Monotone and Two-Stage:

1. Fit both methods
2. Compare cross-validated performance
3. Use 1-SE rule: prefer simpler (Monotone) unless Two-Stage is significantly better

```python
# Auto mode (default)
results = analyze_dataset(fresh_draws_dir="data/responses/")

# Force a specific mode
results = analyze_dataset(
    fresh_draws_dir="data/responses/",
    calibration_mode="monotone"  # or "two_stage"
)
```

---

## Oracle Budget Guide

How many oracle labels do you need?

| Oracle Labels | Calibration Quality | Use Case |
|:--------------|:--------------------|:---------|
| 0 | None (just averaging) | Quick sanity check |
| 20-50 | Rough | Exploratory analysis |
| 50-100 | Moderate | Development decisions |
| 100-500 | Good | Production decisions |
| 500+ | Excellent | High-stakes deployment |

### Cost-Effective Labeling

You don't need to label every sample. Strategy:

1. **Random sample** 5-25% across the score range
2. **Stratify** by judge score (ensure coverage at low/mid/high)
3. **Prioritize boundaries** (scores near decision thresholds)

```python
# CJE automatically uses available oracle labels
# Just include oracle_label field where you have it
{"prompt_id": "1", "judge_score": 0.85, "oracle_label": 0.82}
{"prompt_id": "2", "judge_score": 0.72}  # No oracle - still used for estimation
{"prompt_id": "3", "judge_score": 0.91, "oracle_label": 0.88}
```

---

## Using Covariates

### Response Length

The most common covariate. Fixes verbosity bias:

```python
results = analyze_dataset(
    fresh_draws_dir="data/responses/",
    include_response_length=True  # Auto-computes word count
)
```

### Custom Covariates

Add metadata fields as covariates:

```python
# Data format - include metadata
{"prompt_id": "1", "judge_score": 0.85, "domain": "math", "difficulty": "hard"}

# Analysis
results = analyze_dataset(
    fresh_draws_dir="data/responses/",
    calibration_covariates=["domain", "difficulty"]
)
```

### When Covariates Help

Use covariates when:
- Calibration differs across domains
- Judge has known biases (length, format)
- You see regional miscalibration in reliability plots

---

## Transportability

**Question:** Can a calibrator learned on one policy/domain work on another?

### Testing Transportability

```python
from cje.diagnostics import audit_transportability

# Fit calibrator on source data
source_result = analyze_dataset(fresh_draws_dir="source_data/")
calibrator = source_result.metadata.get('calibrator')

# Test on target with small probe (50 oracle labels)
diag = audit_transportability(
    calibrator,
    probe_samples,  # 50 samples with oracle labels
    group_label="target_policy"
)

print(diag.summary())
# "Transport: PASS | δ̂: +0.012 (CI: [-0.008, +0.032])"
```

### Interpretation

| Status | Meaning | Action |
|:-------|:--------|:-------|
| PASS | Calibrator works on target | Safe to reuse |
| WARN | Small bias detected | Monitor or mean-anchor |
| FAIL | Large systematic bias | Refit calibrator |

### When to Test

- Applying calibrator to new policy
- Reusing across time periods (Q1 → Q2)
- After judge model updates
- Different domains (biology → coding)

---

## Calibration Diagnostics

### Reliability Plot

Shows calibration quality by score region:

```python
from cje.visualization import plot_calibration_comparison

plot_calibration_comparison(
    judge_scores=judge_scores,
    oracle_labels=oracle_labels,
    calibrated_scores=calibrated_scores
)
```

### Key Metrics

| Metric | Good Value | Meaning |
|:-------|:-----------|:--------|
| RMSE | < 0.1 | Low prediction error |
| R² | > 0.5 | Judge predicts oracle well |
| Coverage | > 0.9 | Calibration range covers data |

---

## Common Issues

### "Calibration R² is very low"

**Cause:** Judge scores poorly predict oracle labels.

**Solutions:**
- Increase oracle coverage
- Improve judge prompt/model
- Check if oracle labels are noisy

### "Two-stage worse than monotone"

**Cause:** Not enough data for flexible model.

**Solution:** Use monotone mode, or get more oracle labels.

### "Calibrated scores all similar"

**Cause:** Judge scores don't differentiate quality.

**Solution:** Your judge isn't working. Consider a different judge model.

---

## Reference

For implementation details, see [Calibration Module](../cje/calibration/README.md).
