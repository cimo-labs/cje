# CJE Estimators

## Overview

Causal inference methods for unbiased off-policy evaluation of LLMs, transforming logged data and importance weights into reliable policy value estimates with proper uncertainty quantification.

## Estimator Hierarchy

```
BaseCJEEstimator (abstract)
├── CalibratedDirectEstimator  # Direct (on-policy) evaluation with fresh draws
├── CalibratedIPS              # IPS with optional SIMCal weight calibration
│   └── OrthogonalizedCalibratedIPS  # OC-IPS with robustness to calibration errors
├── StackedDREstimator         # Optimal stacking of DR estimators
└── DREstimator                # Doubly robust base (abstract)
    ├── DRCPOEstimator         # Basic DR with CPO
    ├── OrthogonalizedCalibratedDRCPO  # OC-DR-CPO with first-order insensitivity
    ├── MRDREstimator          # Multiple robust DR
    ├── TMLEEstimator          # Targeted maximum likelihood
    └── TRCPOEstimator         # Triply robust CPO
```

## Core Concepts

### 1. Direct Method (On-Policy Evaluation)
Evaluates target policies using fresh draws sampled directly from those policies. No importance weighting needed since samples come from the target distribution. Supports optional reward calibration (judge → oracle) when logged data with oracle labels is available.

### 2. Importance Sampling (IPS)
Foundation of off-policy evaluation. Reweights logged data to estimate performance under new policies using importance weights W = π_target/π_base.

### 3. SIMCal Weight Calibration
Stabilizes importance weights through monotone projection with variance control. Independent of reward calibration. CalibratedIPS now uses outer CV by default (`use_outer_cv=True`) for honest inference accounting for weight learning uncertainty.

### 4. Doubly Robust (DR) Estimation
Combines direct method (outcome model) with IPS correction. Provides two chances to get the estimate right - if either the outcome model OR the weights are correct, DR is consistent.

### 5. Multiple Robustness (MRDR)
Achieves robustness to outcome model misspecification, propensity score misspecification, and both simultaneously through cross-fitting.

### 6. Targeted Learning (TMLE)
Optimally combines outcome models and importance weights through targeted fluctuation to achieve optimal asymptotic efficiency.

### 7. Estimator Stacking
Forms optimal convex combination of DR estimators by minimizing combined influence function variance. Uses oracle IC approach (w₀ᵀφ(Z)) with ridge regularization for numerical stability.

### 8. Orthogonalized Estimators
Achieve first-order insensitivity to nuisance estimation errors:
- **OC-IPS**: Robust to errors in f̂(S) and m̂(S)
- **OC-DR-CPO**: Additionally robust to q̂(X,A) errors

### 9. Triply Robust (TR-CPO)
Robust to weight calibration, reward calibration, and outcome model errors simultaneously. TR-CPO-E variant (recommended) uses m̂(S)=E[W|S] for variance reduction.

## File Structure

```
estimators/
├── base_estimator.py               # Abstract base
├── direct_method.py               # Direct (on-policy) estimator
├── calibrated_ips.py              # IPS with optional SIMCal
├── orthogonalized_ips.py          # OC-IPS
├── stacking.py                    # Optimal stacking
├── dr_base.py                     # DR base + DRCPOEstimator
├── orthogonalized_calibrated_dr.py # OC-DR-CPO
├── mrdr.py                        # Multiple robust DR
├── tmle.py                        # TMLE
├── tr_cpo.py                      # Triply robust CPO
└── outcome_models.py              # Outcome models
```

## Common Interface

All estimators follow the same pattern:

```python
from cje import CalibratedIPS, PrecomputedSampler
from cje.calibration import calibrate_dataset

# 1. Calibrate dataset (if using reward calibration)
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    enable_cross_fit=True,  # Required for DR methods
    calibration_mode='auto'  # Auto-selects monotone or two-stage
)

# 2. Create sampler with data
sampler = PrecomputedSampler(calibrated_dataset)

# 3. Initialize estimator
# For IPS:
estimator = CalibratedIPS(sampler)
# For DR (requires fresh draws):
estimator = StackedDREstimator(sampler)

# 4. Fit and estimate
result = estimator.fit_and_estimate()

# 5. Access results
estimates = result.estimates           # Point estimates
std_errors = result.standard_errors    # Complete standard errors (IF + MC + oracle)
cis = result.ci()                      # Confidence intervals as (lower, upper) tuples
diagnostics = result.diagnostics       # Health metrics
influence = result.influence_functions # For inference
```

## Default Recommendation

**Use StackedDREstimator** - Combines multiple DR methods via optimal weighting to minimize variance. Requires fresh draws. Provides modest improvements (1-5% SE reduction) over best single method.

**Default estimators**: DR-CPO, TMLE, MRDR (3 core methods)

**Optional additions**: OC-DR-CPO, TR-CPO-E (can be added via `estimators` parameter)

## Diagnostic Warnings & Quality Detection

CalibratedIPS provides estimates while detecting quality issues through comprehensive diagnostics. When overlap problems are detected, actionable warnings guide you toward fixes.

**Quality indicators checked:**

1. **ESS < 30%**: Low effective sample size - over 70% of data has negligible weight
2. **Raw near-zero > 85%**: Severe distribution mismatch - policies very different
3. **Top 5% concentration > 30% with CV > 2.0**: Few outliers dominate - unstable estimate

**Default workflow: Detect → Fix → Re-run**

```python
# Run analysis - always get estimates + diagnostics
estimator = CalibratedIPS(sampler)
result = estimator.fit_and_estimate()

# Check diagnostics
if result.diagnostics.weight_ess < 0.30:
    print("⚠️ Low ESS detected")
    print("Fixes: (1) Use DR mode, (2) Restrict cohort, (3) Collect more diverse data")
    # Apply fix and re-run

# Optional: Strict validation mode (returns NaN for quality issues)
estimator = CalibratedIPS(sampler, refuse_unreliable=True)
```

**Philosophy:** Provide estimates with health metrics, let users decide when to iterate. Critical issues trigger warnings with specific remediation steps.


## Key Design Decisions

1. **Always estimate**: Provide results with diagnostics rather than hard failures
2. **Actionable warnings**: Each issue includes specific fixes to try
3. **Influence Functions**: Always computed for proper inference
4. **Diagnostics**: Automatically attached to all results
5. **Modularity**: DR estimators compose CalibratedIPS for weights

## Outcome Models

- **IsotonicOutcomeModel**: Monotonic regression with judge scores, no parametric assumptions
- **LinearOutcomeModel**: Simple linear regression baseline, fast and stable
- **CalibratorBackedOutcomeModel**: Uses same calibrator as rewards for consistency
- **WeightedIsotonicOutcomeModel**: Policy-specific models for MRDR with omega weights ("w", "w2", or "snips")

**Covariate Support**: All outcome models accept optional `covariates` array during `fit()` and `predict()`. Covariates are automatically extracted from sample metadata (e.g., response_length, domain) when specified in calibration configuration. When using two-stage calibration, covariates are incorporated via the calibrator's index transformation g(S, X_cov), maintaining the same cross-fitted structure.

## Fresh Draws

DR estimators auto-load fresh draws from:
- `data/{policy}_responses.jsonl`
- `data/responses/{policy}_responses.jsonl`
- `data/{policy}_fresh.jsonl`
- `data/fresh_draws/{policy}.jsonl`

Or add manually:
```python
estimator.add_fresh_draws('policy', FreshDrawDataset(samples=[...]))
```



## Standard Errors and Uncertainty Quantification

### Complete Standard Errors

`standard_errors` always includes all sources of uncertainty:
- **Influence function (IF) variance**: Base sampling uncertainty
- **Monte Carlo (MC) variance**: For DR estimators with finite fresh draws
- **Oracle variance**: When calibration uses partial oracle labels (oracle_coverage < 100%)

### Confidence Intervals with t-Distribution

**All estimators use t-critical values by default** (not z-critical) to account for finite degrees of freedom from:
- Cluster-robust standard errors (e.g., clustering by prompt_id in Direct Mode, by CV folds in IPS/DR)
- Oracle uncertainty adjustment with K oracle folds (df = K - 1)

The degrees of freedom is determined by the limiting factor (minimum across sources). This ensures proper 95% coverage even with small numbers of clusters or oracle folds.

**How it works:**
- Estimators store DF information in `result.metadata["degrees_of_freedom"]`
- `EstimationResult.confidence_interval()` automatically uses t-critical values when DF info is available
- Falls back to z-critical for large-sample approximation when DF info is missing
- This is completely automatic - no user configuration needed

### Direct Mode Standard Errors
Direct Mode automatically adapts its standard error calculation based on the data structure:

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

**Philosophy:** Cluster by the source of dependence. Direct Mode clusters by prompts when paired, IPS/DR cluster by cross-validation folds.

### IPS Standard Errors
```python
# Complete SE includes IF variance + oracle uncertainty
standard_errors = np.sqrt(if_variance/n + oracle_variance)

# Oracle variance is automatically skipped at 100% oracle coverage
```

### DR Standard Errors (with Monte Carlo Variance)
```python
# Complete SE includes all three components
standard_errors = np.sqrt(if_variance/n + mc_variance + oracle_variance)

# Check metadata for what's included
result.metadata["se_components"]["includes_mc_variance"]  # True for DR
result.metadata["se_components"]["includes_oracle_uncertainty"]  # True if OUA applied
```

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

### Automatic MC Variance Handling
When only one fresh draw per prompt (M=1), DR estimators automatically use a conservative upper bound:
- Total variance across single draws bounds within-prompt variance
- Capped at 0.25 for binary [0,1] outcomes
- Mixed cases (some M≥2, some M=1) combine exact computation with upper bound


## Advanced Features

### Stacked DR Configuration
```python
StackedDREstimator(
    sampler,
    estimators=['dr-cpo', 'tmle', 'mrdr', 'oc-dr-cpo', 'tr-cpo-e'],
    covariance_regularization=1e-4,  # Ridge regularization strength
    n_folds=20                       # Cross-fitting folds
)
```
Uses regularized covariance estimation to handle highly correlated component estimators.

### Oracle Uncertainty Augmentation (OUA)
All estimators support OUA via delete-one-fold jackknife to account for calibrator uncertainty from finite oracle samples. **Note: OUA is automatically skipped at 100% oracle coverage** since there's no oracle uncertainty when all samples have ground truth labels.

```python
# Enabled by default
estimator = CalibratedIPS(sampler, oua_jackknife=True)

# Oracle uncertainty is automatically included in standard_errors
result = estimator.fit_and_estimate()

# Check if oracle uncertainty was added
if "se_components" in result.metadata:
    if result.metadata["se_components"].get("oracle_uncertainty_skipped"):
        print("OUA skipped - 100% oracle coverage")
    elif result.metadata["se_components"].get("includes_oracle_uncertainty"):
        print("Oracle uncertainty included in standard_errors")
```

### Honest Inference with Outer CV
CalibratedIPS uses outer cross-validation by default (`use_outer_cv=True`) to account for weight learning uncertainty:
```python
# Default: Outer CV enabled
estimator = CalibratedIPS(sampler)  # use_outer_cv=True by default

# Customize settings
estimator = CalibratedIPS(
    sampler,
    n_outer_folds=10,       # More folds for stability
)
```

### Custom Estimators
Inherit from `BaseCJEEstimator` or `DREstimator` and implement `fit()` and `estimate()`. Always compute and store influence functions in `_influence_functions`.



## Common Issues

- **NaN estimates**: Check ESS in diagnostics. Likely poor overlap - try DR methods with fresh draws
- **Low ESS**: Policies too different. Consider collecting more diverse base data
- **DR fails**: All DR methods require fresh draws. Generate them first
- **Underestimated SEs**: Ensure `use_outer_cv=True` for honest inference (enabled by default)
- **Different results between runs**: Set random seeds for reproducibility in cross-fitting

## Implementation Notes

### Cross-Fitting
DR estimators use k-fold cross-fitting for orthogonality:
- Unified fold system via `cje.data.folds` (deterministic: `hash(prompt_id) % k`)
- Each fold gets predictions from model trained on other folds
- Prevents overfitting in outcome models

### Weight Caching
Estimators cache computed weights to avoid recomputation across policies.

### Influence Functions
Always computed and stored for proper inference, policy comparison, and diagnostics.

## References

- **IPS**: Horvitz & Thompson (1952)
- **Doubly Robust**: Robins et al. (1994)
- **TMLE**: van der Laan & Rubin (2006)
- **SIMCal**: Surrogate-indexed monotone calibration (2024)

## Summary

Comprehensive toolkit for causal inference on LLM outputs, from simple importance sampling to sophisticated multiply-robust methods. All estimators follow the same interface, compute influence functions, and provide transparent diagnostics for reliability assessment. **StackedDREstimator is recommended for production use** when fresh draws are available.
