# CJE Estimators

## Overview

Causal inference methods for unbiased off-policy evaluation of LLMs, transforming logged data and importance weights into reliable policy value estimates with proper uncertainty quantification.

## Estimator Hierarchy

```
BaseCJEEstimator (abstract)
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

### 1. Importance Sampling (IPS)
Foundation of off-policy evaluation. Reweights logged data to estimate performance under new policies using importance weights W = π_target/π_base.

### 2. SIMCal Weight Calibration
Stabilizes importance weights through monotone projection with variance control. Independent of reward calibration. CalibratedIPS now uses outer CV by default (`use_outer_cv=True`) for honest inference accounting for weight learning uncertainty.

### 3. Doubly Robust (DR) Estimation
Combines direct method (outcome model) with IPS correction. Provides two chances to get the estimate right - if either the outcome model OR the weights are correct, DR is consistent.

### 4. Multiple Robustness (MRDR)
Achieves robustness to outcome model misspecification, propensity score misspecification, and both simultaneously through cross-fitting.

### 5. Targeted Learning (TMLE)
Optimally combines outcome models and importance weights through targeted fluctuation to achieve optimal asymptotic efficiency.

### 6. Estimator Stacking
Forms optimal convex combination of DR estimators by minimizing combined influence function variance. Uses oracle IC approach (w₀ᵀφ(Z)) with ridge regularization for numerical stability.

### 7. Orthogonalized Estimators
Achieve first-order insensitivity to nuisance estimation errors:
- **OC-IPS**: Robust to errors in f̂(S) and m̂(S)
- **OC-DR-CPO**: Additionally robust to q̂(X,A) errors

### 8. Triply Robust (TR-CPO)
Robust to weight calibration, reward calibration, and outcome model errors simultaneously. TR-CPO-E variant (recommended) uses m̂(S)=E[W|S] for variance reduction.

## File Structure

```
estimators/
├── base_estimator.py               # Abstract base
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
std_errors = result.standard_errors    # Standard errors
diagnostics = result.diagnostics       # Health metrics
influence = result.influence_functions # For inference
```

## Default Recommendation

**Use StackedDREstimator** - Combines multiple DR methods (DR-CPO, TMLE, MRDR, OC-DR-CPO, TR-CPO-E) via optimal weighting to minimize variance. Requires fresh draws. Provides modest improvements (1-5% SE reduction) over best single method.

## Refusal Gates

CalibratedIPS includes safety mechanisms that detect when estimates would be unreliable due to poor overlap:

1. **ESS < 30%**: Over 70% of data effectively ignored
2. **Raw near-zero > 85%**: Severe distribution mismatch that calibration may mask
3. **Top 5% concentration > 30% with CV > 2.0**: Few outliers dominate estimate

Default behavior: Provides estimates with warnings. Set `refuse_unreliable=True` to return NaN for unreliable policies.

```python
# Default: Warn but estimate
estimator = CalibratedIPS(sampler)

# Strict mode: Return NaN for unreliable
estimator = CalibratedIPS(sampler, refuse_unreliable=True)
```


## Key Design Decisions

1. **Transparency**: Default to warnings over silent failures
2. **Influence Functions**: Always computed for proper inference
3. **Diagnostics**: Automatically attached to all results
4. **Modularity**: DR estimators compose CalibratedIPS for weights

## Outcome Models

- **IsotonicOutcomeModel**: Monotonic regression with judge scores, no parametric assumptions
- **LinearOutcomeModel**: Simple linear regression baseline, fast and stable
- **CalibratorBackedOutcomeModel**: Uses same calibrator as rewards for consistency
- **WeightedIsotonicOutcomeModel**: Policy-specific models for MRDR with omega weights ("w", "w2", or "snips")

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

### Three Types of Standard Errors

1. **`standard_errors`**: Base uncertainty from sampling (includes MC variance for DR estimators)
2. **`robust_standard_errors`**: Adds oracle uncertainty from finite calibration sample
3. **Method-specific robust SEs**: Some estimators add additional robustness adjustments

### IPS Standard Errors
```python
# Base SE from influence functions
standard_errors = np.std(influence_functions, ddof=1) / np.sqrt(n)

# Robust SE adds oracle uncertainty (only when oracle_coverage < 100%)
robust_standard_errors = np.sqrt(standard_errors² + oracle_variance)
```

### DR Standard Errors (with Monte Carlo Variance)
```python
# Base IF variance + MC variance from finite fresh draws
standard_errors = np.sqrt(if_variance/n + mc_variance)

# Robust SE adds oracle uncertainty on top
robust_standard_errors = np.sqrt(standard_errors² + oracle_variance)
```

**Important**: For DR estimators, `standard_errors` already includes MC variance. Check `mc_variance_included: True` in metadata.

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

# Access OUA-adjusted standard errors
result = estimator.fit_and_estimate()
robust_ses = result.robust_standard_errors  # Includes oracle uncertainty

# At 100% oracle coverage: robust_ses == standard_errors (no OUA applied)
# At <100% coverage: robust_ses >= standard_errors (OUA adds uncertainty)
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
    honest_iic=True         # Apply honest IIC for variance reduction
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
- **SIMCal**: Score-indexed monotone calibration (2024)

## Summary

Comprehensive toolkit for causal inference on LLM outputs, from simple importance sampling to sophisticated multiply-robust methods. All estimators follow the same interface, compute influence functions, and provide transparent diagnostics for reliability assessment. **StackedDREstimator is recommended for production use** when fresh draws are available.
