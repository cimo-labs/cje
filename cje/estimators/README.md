# CJE Estimators

## Overview

Estimators that turn judge-scored fresh draws into reliable policy value estimates with proper uncertainty quantification.

> **Note (0.4.0):** The off-policy estimators (CalibratedIPS, DR-CPO, MRDR, TMLE, StackedDR) were removed. For IPS/DR workflows pin `pip install "cje-eval==0.3.*"`.

## Estimator Hierarchy

```
BaseCJEEstimator (abstract)
└── CalibratedDirectEstimator  # Direct (on-policy) evaluation with fresh draws
```

## Core Concepts

### Direct Method (On-Policy Evaluation)
Evaluates target policies using fresh draws sampled directly from those policies. No importance weighting needed since samples come from the target distribution. Supports optional reward calibration (judge → oracle) when labeled data is available.

## File Structure

```
estimators/
├── base_estimator.py       # Abstract base
└── direct_method.py        # Direct (on-policy) estimator
```

## Common Interface

```python
from cje.estimators import CalibratedDirectEstimator
from cje.calibration import calibrate_dataset

# 1. Calibrate judge scores against oracle labels
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    enable_cross_fit=True,
    calibration_mode='auto'  # Auto-selects monotone or two-stage
)

# 2. Initialize estimator
estimator = CalibratedDirectEstimator(
    target_policies=["policy_a", "policy_b"],
    reward_calibrator=cal_result.calibrator,
)

# 3. Add fresh draws for each policy
# - easiest: use analyze_dataset(..., fresh_draws_dir=...)
# - or: call estimator.add_fresh_draws(policy, fresh_draws) per policy

# 4. Fit and estimate
result = estimator.fit_and_estimate()

# 5. Access results
estimates = result.estimates           # Point estimates
std_errors = result.standard_errors    # Complete standard errors (IF + oracle)
cis = result.ci()                      # Confidence intervals as (lower, upper) tuples
diagnostics = result.diagnostics       # Health metrics
influence = result.influence_functions # For inference
```

## Fresh Draws

Fresh draws can be auto-loaded from standard locations relative to a provided `fresh_draws_dir` (see `load_fresh_draws_auto(...)`):
- `{policy}_responses.jsonl`
- `responses/{policy}_responses.jsonl`
- `{policy}_fresh.jsonl`
- `fresh_draws/{policy}.jsonl`

Or add manually:
```python
estimator.add_fresh_draws('policy', FreshDrawDataset(samples=[...]))
```

## Standard Errors and Uncertainty Quantification

### Complete Standard Errors

`standard_errors` always includes all sources of uncertainty:
- **Influence function (IF) variance**: Base sampling uncertainty
- **Oracle variance**: When calibration uses partial oracle labels (oracle_coverage < 100%)

### Confidence Intervals with t-Distribution

**All estimators use t-critical values by default** (not z-critical) to account for finite degrees of freedom from:
- Cluster-robust standard errors (clustering by prompt_id in Direct Mode)
- Oracle uncertainty adjustment with K oracle folds (df = K - 1)

The degrees of freedom is determined by the limiting factor (minimum across sources). This ensures proper 95% coverage even with small numbers of clusters or oracle folds.

**How it works:**
- Estimators store DF information in `result.metadata["degrees_of_freedom"]`
- `EstimationResult.confidence_interval()` automatically uses t-critical values when DF info is available
- Falls back to z-critical for large-sample approximation when DF info is missing
- This is completely automatic - no user configuration needed

### Direct Mode Standard Errors
Direct Mode automatically adapts its standard error calculation based on the data structure and inference method:

**Inference Methods** (controlled via `inference_method` parameter):
- **`bootstrap`** (default): Cluster bootstrap with calibrator refit + θ̂_aug (achieves **~95% coverage**)
- **`cluster_robust`**: Standard cluster-robust SEs by prompt (fastest, ~22-55% coverage)
- **`auto`**: Uses cluster_robust; switches to bootstrap when coupling detected

**Separate flag:** `oua_jackknife` is not an `inference_method` value.
Set `oua_jackknife=True` to add oracle jackknife augmentation on top of the chosen inference method.

**Bootstrap with θ̂_aug** uses an AIPW-style bias correction:
```
θ̂_aug = mean(f̂_full(S)) + mean(Y - f̂_oof(S))
```
This applies a per-policy residual correction that is the default first-moment transport fix in Direct mode. Refitting the calibrator on each bootstrap replicate captures the calibration/evaluation covariance that analytic SEs miss.

```python
# Explicit bootstrap for small samples or coupled calibration/evaluation
estimator = CalibratedDirectEstimator(
    target_policies=policies,
    reward_calibrator=calibrator,
    inference_method="bootstrap",  # or "auto", "cluster_robust"
    n_bootstrap=2000,              # Number of bootstrap replicates
)
```

**When bootstrap is preferred (recommended for all cases):**
- **Always** - bootstrap achieves ~95% coverage vs ~22-55% for cluster-robust
- Few evaluation clusters (< 20 prompts) - asymptotic approximation unreliable
- Calibration and evaluation data overlap (coupled) - analytic SEs miss covariance
- Need valid confidence intervals (the default cluster_robust severely undercovers)

**Standard (non-bootstrap) cluster-robust SEs:**
```python
# Single policy or unpaired: Standard SE
standard_errors = np.sqrt(variance/n)

# Paired comparisons (same prompts across policies): Cluster-robust SE
# Clusters by prompt_id to account for within-prompt correlation
standard_errors = cluster_robust_se(influence_functions, cluster_ids=prompt_ids)

# Check which method was used
result.metadata["se_methods"]  # e.g., {"policy_a": "cluster_robust", ...}
result.metadata["n_clusters"]  # e.g., {"policy_a": 1000, ...}
```

**When cluster-robust SEs are used:**
- Evaluating multiple policies on the **same prompts** (paired comparison)
- Example: 3 policies × 1000 prompts = 3000 samples, but only 1000 independent clusters
- Clusters by `prompt_id` to account for correlation across policies
- Provides honest uncertainty for policy comparisons

**Important:** Bootstrap inference affects only SEs and CIs, not point estimates. Point estimates always use the original calibrator for consistency.

**Transport-Aware Bootstrap** (`calibration_policy` parameter):
When evaluating multiple policies where calibration was learned on a base policy, use `calibration_policy` to enable transport-aware bias correction:

```python
estimator = CalibratedDirectEstimator(
    target_policies=["base", "verbose", "contrarian"],
    reward_calibrator=calibrator,
    inference_method="bootstrap",
    calibration_policy="base",  # Fit calibrator only on base policy
)
```

This separates:
- **Calibration oracle**: Only base policy samples (for fitting the calibrator)
- **Residual oracle**: All policies (for computing transport bias corrections in θ̂_aug)

When the calibrator doesn't transport to target policies, the residual correction `mean(Y - f̂(S))` captures this bias per policy. Treat a single global offset as a baseline only. See `diagnostics/README.md` for details.

**Philosophy:** Cluster by the source of dependence. Direct Mode clusters by prompts when paired.

### Convenience Method
```python
# Get confidence intervals as list of (lower, upper) tuples
# Uses t-critical values automatically (accounts for finite degrees of freedom)
cis = result.ci(alpha=0.05)  # 95% CIs by default
for i, (lower, upper) in enumerate(cis):
    print(f"Policy {i}: [{lower:.3f}, {upper:.3f}]")

# Check degrees of freedom used (optional)
if "degrees_of_freedom" in result.metadata:
    df_info = result.metadata["degrees_of_freedom"]
    for policy, info in df_info.items():
        print(f"{policy}: df={info['df']}, t_crit={info['t_critical']:.3f}")
```

## Advanced Features

### Oracle Uncertainty Augmentation (calibration-aware)
Delete-one-fold jackknife accounts for calibrator uncertainty from finite oracle samples. **Note: calibration-aware is automatically skipped at 100% oracle coverage** since there's no oracle uncertainty when all samples have ground truth labels.

```python
# Enabled by default
estimator = CalibratedDirectEstimator(
    target_policies=policies,
    reward_calibrator=calibrator,
    oua_jackknife=True,
)

# Oracle uncertainty is automatically included in standard_errors
result = estimator.fit_and_estimate()

# Check if oracle uncertainty was added
if "se_components" in result.metadata:
    if result.metadata["se_components"].get("oracle_uncertainty_skipped"):
        print("calibration-aware skipped - 100% oracle coverage")
    elif result.metadata["se_components"].get("includes_oracle_uncertainty"):
        print("Oracle uncertainty included in standard_errors")
```

### Custom Estimators
Inherit from `BaseCJEEstimator` and implement `fit()` and `estimate()`. Always compute and store influence functions in `_influence_functions`.

## Implementation Notes

### Cross-Fitting
Calibration uses k-fold cross-fitting via the unified fold system in `cje.data.folds` (deterministic: `hash(prompt_id) % k`).

### Influence Functions
Always computed and stored for proper inference, policy comparison, and diagnostics.

## Summary

A focused toolkit for on-policy evaluation of LLM outputs from judge-scored fresh draws, with calibrated rewards, honest uncertainty quantification, and transparent diagnostics.
