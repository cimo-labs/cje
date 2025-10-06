# CJE Calibration Module

## Overview

The calibration module implements **AutoCal-R** (Automatic Calibration for Rewards), the core mathematical machinery that enables unbiased causal inference from judge-based evaluations. AutoCal-R provides three distinct calibration approaches that work together to transform raw logged data into reliable policy value estimates with controlled variance:

1. **Judge→Oracle calibration**: Maps judge scores to oracle labels with automatic mode selection
2. **Weight stabilization (SIMCal)**: Stabilizes importance weights for off-policy estimation
3. **Cross-fitted models**: Enables orthogonality guarantees for doubly robust methods

## When to Use Each Calibration

### Use **Reward Calibration** when:
- You have judge scores and some oracle labels
- You want to map judge scores → oracle scale
- You're using any estimation method

### Use **Weight Calibration** (SIMCal) when:
- Importance weights have high variance
- You want to stabilize IPS estimates
- You're using CalibratedIPS estimator

### Use **Cross-Fitted Models** when:
- You're using DR estimators
- You need orthogonality guarantees
- You have enough data for stable folds

## File Structure

```
calibration/
├── __init__.py          # Public API exports
├── dataset.py           # High-level dataset calibration workflows
├── flexible_calibrator.py # Flexible calibration for non-monotone relationships
├── isotonic.py          # Core isotonic regression and variance control
├── judge.py             # Judge score calibration to oracle labels
├── oracle_slice.py      # Oracle slice configuration (deprecated)
└── simcal.py            # Stacked SIMCal implementation
```

## Core Concepts

### 1. Judge Score Calibration (AutoCal-R Core)
AutoCal-R maps cheap LLM judge scores to expensive oracle labels with automatic mode selection. Default is 'auto' mode which automatically chooses between:
- **Monotone calibration**: Standard isotonic regression (when relationship is monotone)
- **Flexible calibration**: Two-stage g(S)→isotonic for non-monotone relationships

Auto mode detects non-monotonicity by comparing regional performance and selects the appropriate method. The selected mode is stored in metadata for transparency. This automatic selection is a key feature of AutoCal-R.

### 2. Weight Calibration (SIMCal)
Stabilizes importance weights through score-indexed monotone projection:
- Projects weights to be monotone with an ordering index
- Enforces variance constraints via blending
- Maintains mean-1 property for unbiasedness

### 3. Cross-Fitted Models
For doubly robust methods, provides out-of-fold predictions to maintain orthogonality between nuisance functions.
Stacking relies on the component estimators' influence functions and does not re-fit nuisances at the stack level.

### 4. Oracle Uncertainty Quantification (Two Approaches)
When we calibrate judge scores using only a subset of oracle labels (e.g., 10% coverage), the calibration function f̂ itself has uncertainty. We handle this through two complementary mechanisms:

**Oracle Uncertainty Augmentation (OUA)**: The default approach that uses fold-jackknife to add a **variance** component to CIs, accounting for calibration-induced uncertainty. Used by all Cal-IPS/DR estimators.

**Oracle Slice Augmentation**: An optional point-estimate **bias correction** term `(L/π_L)m̂(S)(Y-f̂(S))` used **only** in TR-CPO under MAR with fitted π_L(S), or optionally as an MCAR engineering fallback (off by default).

## Module Descriptions

### `dataset.py` - Dataset Calibration Workflows (AutoCal-R API)
High-level functions that orchestrate the AutoCal-R calibration process for entire datasets:
- `calibrate_dataset()`: Main AutoCal-R entry point - transforms Dataset objects with judge scores into calibrated rewards
- `calibrate_from_raw_data()`: Works with raw dictionaries for pipeline integration
- Handles both standard and cross-fitted calibration
- Preserves metadata and adds calibration diagnostics

### `judge.py` - Judge Calibration (AutoCal-R Implementation)
Implements the core AutoCal-R algorithm for calibration from judge scores to oracle labels:
- `JudgeCalibrator`: Core AutoCal-R class with flexible mode support and automatic selection
- `fit_transform()`: Standard calibration on oracle subset
- `fit_cv()`: Cross-fitted calibration for DR methods
- `index()`: Returns transformation for outcome models (S for monotone, g(S) for two-stage)
- `CalibrationResult`: Container for calibrated scores and diagnostics
- Auto mode (default): Automatically selects monotone or flexible calibration
- Supports partial labeling (oracle coverage)

### `flexible_calibrator.py` - Non-Monotone Calibration
Handles non-monotone judge→oracle relationships via two-stage approach:
- `FlexibleCalibrator`: Implements g(S)→isotonic calibration
- First stage: Learn smooth transformation g(S) using splines
- Second stage: Apply isotonic regression on g(S)
- `index()`: Exposes the transformation T=g(S) for outcome models
- Per-fold ECDF for consistent rank transformation
- Auto selection based on regional performance comparison

**Mode Selection Logic:**
- Compares monotone vs two-stage using 1-SE rule
- Checks performance across score regions (low/mid/high)
- Selects two-stage if better in ≥2/3 regions or significantly better overall
- Optimized to skip two-stage training when monotone is clearly sufficient

**Technical Details:**
- ECDF-based ranking prevents distribution leakage between folds
- Minimum 5 spline knots to avoid underfitting
- Fallback to monotone for small samples (<20)
- Clipping to [0,1] ensures valid reward range

### `isotonic.py` - Isotonic Weight Calibration
Core mathematical operations for weight calibration:
- `calibrate_to_target_mean()`: Main entry point for weight calibration
- `_pav_mean1_projection_sorted()`: Pool Adjacent Violators with mean preservation
- `_variance_safe_blend_closed_form()`: Optimal blending for variance control
- Uses "exact" mode (bisection) for consistency
- Handles ordering by arbitrary index (e.g., judge scores)

### `simcal.py` - Stacked SIMCal
Advanced weight calibration through stacking:
- `SIMCalibrator`: Combines {baseline, increasing, decreasing} candidates
- Out-of-fold (OOF) influence function minimization
- Quadratic program on simplex for optimal mixture
- Uniform blending for ESS/variance constraints
- Configurable via `SimcalConfig` dataclass
- **New**: Supports fit/predict separation for honest inference
  - `fit()`: Learn isotonic models and mixture weights on training data
  - `predict()`: Apply learned calibration to new data with score clipping
  - `fit_transform()`: Backward-compatible single-pass method

### `oracle_slice.py` - Oracle Slice Augmentation
Implements the optional point-estimate bias correction (used primarily in TR-CPO):
- **Problem**: We use f̂(S) everywhere but only have true Y on oracle subset  
- **Solution**: Add augmentation term `(L/π_L) * m̂(S) * (Y - f̂(S))` where:
  - L indicates oracle label presence, π_L = labeling propensity
  - m̂(S) = E[W|S] estimated via isotonic regression
  - Unbiased correction for proxy-truth gap under MAR/MCAR
- **Usage**: Enabled in TR-CPO for MAR setting; optional MCAR fallback (off by default)
- **Note**: This is separate from OUA jackknife variance (the default uncertainty method)

## Key Design Decisions

### 1. **Separation of Concerns**
Each calibration type is isolated with clear interfaces:
- Reward calibration doesn't know about weights
- Weight calibration doesn't know about rewards
- Outcome models are separate from both

### 2. **Mean Preservation**
Calibrations preserve means for unbiased estimation:
- Isotonic preserves the **slice sample mean** exactly, and the **population mean asymptotically** under J₁ (representative slice)
- Weight projections preserve the **sample** mean-one exactly (Hájek normalization)
- Critical for unbiased estimation

### 3. **Variance Control**
Multiple mechanisms for variance reduction:
- **Isotonic projection**: Can reduce variance when weights correlate with ordering index
- **Variance cap**: Explicit upper bound on weight variance via blending
- **ESS floor**: Minimum effective sample size constraint
- **Baseline shrinkage**: Small bias for large variance reduction

### 4. **Cross-Fitting Support**
Built-in support for cross-fitted calibration:
- Prevents overfitting in DR methods
- Maintains orthogonality between nuisance functions
- Uses unified fold system from `cje.data.folds` for consistency
- Fold assignments computed deterministically from prompt_id

### 5. **Numerical Robustness**
Careful handling of edge cases:
- Zero weights: Fallback to uniform
- Constant weights: Return target mean
- Sparse weights: Relaxed tolerance
- Numerical precision: Multiple safety checks


## Mathematical Foundations

### Isotonic Regression (PAV Algorithm)
Finds the best-fitting monotone function: `min ||f(x) - y||²` subject to monotonicity.
- **Time**: O(n log n) 
- **Property**: When ordered by uncorrelated index, produces nearly constant weights

### Mean-Preserving Projection  
Ensures calibrated weights have exactly mean=1 via bisection on Lagrange multipliers.
- **Why**: Critical for unbiased estimation (E[W] = 1)
- **Implementation**: ~30-40 PAV calls for exact solution

### Variance-Safe Blending
Optimally blends raw and calibrated weights to satisfy variance constraints:
```
w_final = (1-α)·raw + α·calibrated
where Var(w_final) ≤ ρ·Var(raw)
```
- **Solution**: Closed-form via quadratic formula

### Stacked SIMCal
Combines K=3 candidates by minimizing OOF influence variance:
```
min_π π'Σπ s.t. π ≥ 0, Σπ = 1
```
- **Candidates**: {baseline, increasing, decreasing}
- **Solution**: Quadratic program on simplex

## Usage Patterns

### Basic Reward Calibration
```python
from cje.calibration import calibrate_dataset

# Calibrate judge scores to oracle labels (auto mode by default)
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    judge_field="judge_score",
    oracle_field="oracle_label",
    calibration_mode="auto",  # Or "monotone", "two_stage"
    random_seed=42  # For reproducibility
)

# Access calibration quality metrics and metadata
print(f"RMSE: {cal_result.calibration_rmse:.3f}")
print(f"Coverage: {cal_result.coverage_at_01:.1%}")
print(f"Selected mode: {calibrated_dataset.metadata.get('calibration_info', {}).get('selected_mode')}")
```

### Weight Calibration (Direct)
```python
from cje.calibration import calibrate_to_target_mean

# Calibrate weights with variance control
calibrated_weights, info = calibrate_to_target_mean(
    raw_weights,
    target_mean=1.0,
    enforce_variance_nonincrease=True,
    ordering_index=judge_scores,  # Order by judge scores
    return_diagnostics=True
)

print(f"Variance reduction: {info['var_before']/info['var_after']:.2f}x")
```

### Stacked SIMCal
```python
from cje.calibration import SIMCalibrator, SimcalConfig

# Configure stacked calibration
config = SimcalConfig(
    ess_floor=0.2,      # Minimum 20% ESS
    var_cap=1.0,        # No variance increase
    include_baseline=False,
)

# Run calibration
calibrator = SIMCalibrator(config)
calibrated, info = calibrator.transform(
    weights, 
    judge_scores,
    rewards=rewards  # For IPS influence functions
)

print(f"Mixture: {info['mixture_weights']}")
print(f"ESS improvement: {info['ess_after']/info['ess_before']:.2f}x")
```

### Cross-Fitted Calibration (for DR)
```python
from cje.calibration import JudgeCalibrator

# Fit with cross-validation for DR methods
calibrator = JudgeCalibrator()
result = calibrator.fit_cv(
    judge_scores,
    oracle_labels,
    oracle_mask,
    n_folds=5
)

# Get out-of-fold predictions
oof_predictions = calibrator.predict_oof(judge_scores, fold_ids)
```

### Oracle Uncertainty (Default: OUA Jackknife)
```python
from cje import CalibratedIPS

# Default: OUA jackknife for oracle uncertainty (recommended)
estimator = CalibratedIPS(sampler, oua_jackknife=True)  # Default
result = estimator.fit_and_estimate()
# Result has both standard_errors and robust_standard_errors

# Optional: Enable bias correction augmentation (engineering fallback)
from cje.calibration import OracleSliceConfig
oracle_config = OracleSliceConfig(
    enable_augmentation=True,
    enable_cross_fit=True,
    min_pi=0.01,
    use_mar=False  # MCAR assumption
)

estimator = CalibratedIPS(
    sampler,
    oracle_slice_config=oracle_config
)

# The augmentation automatically adjusts standard errors
# to account for calibration uncertainty
result = estimator.fit_and_estimate()

# Check oracle uncertainty via OUA jackknife (if enabled)
if result.robust_standard_errors is not None:
    print(f"Standard SE: {result.standard_errors[0]:.4f}")
    print(f"OUA-adjusted SE: {result.robust_standard_errors[0]:.4f}")
    oracle_var = result.robust_standard_errors[0]**2 - result.standard_errors[0]**2
    print(f"Oracle uncertainty contribution: {oracle_var:.6f}")
```

## Configuration Options

### SimcalConfig Parameters
- `ess_floor`: Minimum ESS as fraction (e.g., 0.2 = 20%)
- `var_cap`: Maximum variance (e.g., 1.0 = no increase)
- `include_baseline`: Include raw weights in stack
- `baseline_shrink`: Shrinkage toward baseline (0-1)
- `ridge_lambda`: Ridge regularization for covariance
- `n_folds`: Number of OOF folds if not provided

### Calibration Modes
- **Auto** (default): Automatically selects between monotone and two-stage based on performance
- **Monotone**: Standard isotonic regression (forces monotone relationship)
- **Two-stage**: Flexible g(S)→isotonic for non-monotone relationships
- **Cross-fitted**: K-fold models for DR orthogonality (enable_cross_fit=True)
- **Projection mode**: Always uses "exact" (bisection) for consistency

## Implementation Details

### Ordering Index Flexibility
The `ordering_index` parameter in isotonic calibration allows weights to be monotone in any score:
- **None**: Sort by raw weights (backward compatibility)
- **Judge scores**: Align with human evaluation
- **Calibrated rewards**: Align with outcome models (for DR)

When the ordering index is uncorrelated with weights, isotonic projection produces nearly constant weights - this is expected and provides stabilization.

### Tie Handling
When the ordering index has ties (common with discrete judge scores):
1. Pool weights within tied groups (average)
2. Apply isotonic regression to pooled values
3. Assign same calibrated weight to all tied samples

### Numerical Tolerances
- `EPS = 1e-12`: Machine epsilon for comparisons
- `MEAN_TOL = 1e-10`: Tolerance for mean preservation
- `VAR_TOL = 1.001`: Allow 0.1% slack on variance cap

### Memory Efficiency
- Isotonic regression is O(n log n) time, O(n) space
- Stacked calibration builds K=3 candidates
- Cross-fitting stores K models but applies one at a time

## Common Issues and Solutions

### Issue: "Judge field 'reward' not allowed"
**Cause**: Trying to use 'reward' as judge field to avoid confusion  
**Solution**: Use a different field name in metadata (e.g., 'judge_score')

### Issue: Low calibration R² (< 0.3)
**Cause**: Judge scores poorly predict oracle labels  
**Solution**: 
- Increase oracle coverage (aim for >10%)
- Improve judge prompt/model
- Consider using a different judge
- Check if oracle labels are noisy

### Issue: Nearly constant calibrated weights
**Cause**: Ordering index uncorrelated with importance ratios  
**Solution**: This is expected and actually good - provides maximum variance stabilization

### Issue: Variance cap not satisfied exactly
**Cause**: Numerical precision or infeasible constraint  
**Solution**: Check info dict for 'feasible' flag and 'note' field

### Issue: ESS floor conflicts with variance cap
**Cause**: ESS implies tighter variance constraint than specified  
**Solution**: ESS constraint will dominate (warning issued)

### Issue: Very low oracle coverage (<5%)
**Cause**: Too few labeled samples for reliable calibration
**Solution**: 
- Collect more oracle labels
- Consider using judge scores directly (uncalibrated)
- Use bootstrapping to assess calibration uncertainty

## Testing

The calibration module has comprehensive test coverage:
- `test_stacked_simcal.py`: Stacked SIMCal functionality
- Integration tests verify calibration in full pipeline
- Edge case tests for degenerate inputs

Run tests:
```bash
poetry run pytest cje/tests/ -k calibration
```

## Performance Considerations

### Computational Complexity
- **Isotonic regression**: O(n log n) via PAV
- **Exact projection**: ~30-40 PAV calls (still O(n log n))
- **Stacked SIMCal**: O(nK²) time, O(K²) memory (K=3 candidates)
- **Cross-fitting**: K × isotonic regression cost


### When to Use Each Method

**Use standard calibration when:**
- You have sufficient oracle labels (>100)
- Not using DR methods
- Speed is critical

**Use cross-fitted calibration when:**
- Using DR estimators
- Need orthogonality guarantees
- Have enough data for stable fold models

**Use stacked SIMCal when:**
- Weights have high variance
- Multiple candidate projections make sense
- OOF validation is feasible


## Advanced Topics

### Bootstrapping Calibration Uncertainty
```python
# For low oracle coverage scenarios
n_bootstrap = 100
calibrations = []
for _ in range(n_bootstrap):
    idx = np.random.choice(n_oracle, n_oracle, replace=True)
    cal = JudgeCalibrator()
    result = cal.fit_transform(judge_scores[idx], oracle_labels[idx])
    calibrations.append(result.calibrated_scores)
```

### Debugging SIMCal
```python
# Check intermediate steps
calibrated, info = calibrator.transform(weights, scores, rewards=rewards)
print(f"Mixture weights: {info['mixture_weights']}")
print(f"Variance reduction: {info['var_before']/info['var_after']:.2f}x")
```


## References

- **Isotonic Regression**: Robertson et al. (1988), "Order Restricted Statistical Inference"
- **PAV Algorithm**: Ayer et al. (1955), "An Empirical Distribution Function for Sampling with Incomplete Information"  
- **Majorization**: Marshall & Olkin (1979), "Inequalities: Theory of Majorization"
- **SIMCal**: CJE paper (2025), "Surrogate-Indexed Monotone Calibration"
- **Cross-fitting**: Chernozhukov et al. (2018), "Double/Debiased Machine Learning"

## Summary

The calibration module provides three essential transformations for causal inference: mapping judge scores to oracle labels, stabilizing importance weights through SIMCal, and enabling cross-fitted models for DR methods. Each calibration type maintains mean preservation for unbiased estimation while controlling variance through different mechanisms.
