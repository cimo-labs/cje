# CJE Calibration Module

## Overview

The calibration module implements **AutoCal-R** (Automatic Calibration for Rewards), the core mathematical machinery that maps judge scores to oracle labels with automatic mode selection. The module also provides **SIMCal-W** (Surrogate-Indexed Monotone Calibration for Weights), a separate method for stabilizing importance weights in off-policy estimation. These are independent calibration techniques:

1. **AutoCal-R (Reward Calibration)**: Maps judge scores to oracle labels with automatic mode selection between monotone and two-stage calibration
2. **SIMCal-W (Weight Stabilization)**: Stabilizes importance weights for off-policy estimation via surrogate-indexed monotone projection (separate from AutoCal-R)
3. **Cross-fitted models**: Enables orthogonality guarantees for doubly robust methods (used by both AutoCal-R and SIMCal-W when needed)

## When to Use Each Calibration

### Use **Reward Calibration** when:
- You have judge scores and some oracle labels
- You want to map judge scores → oracle scale
- You're using any estimation method

### Use **Weight Calibration** (SIMCal) when:
- Importance weights have high variance
- You want to stabilize IPS estimates
- You're using CalibratedIPS or Calibrated DR estimators

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

**Why isotonic?** Isotonic regression is the default because it imposes exactly the right inductive bias (monotonicity) while making minimal assumptions, preserves oracle KPI levels by construction, and is highly efficient with small label budgets (5-10% coverage often sufficient). See the detailed rationale below.

### 2. Weight Calibration (SIMCal)
Stabilizes importance weights through surrogate-indexed monotone projection:
- Projects weights to be monotone with an ordering index
- Enforces variance constraints via blending
- Maintains mean-1 property for unbiasedness

### 3. Cross-Fitted Models
For doubly robust methods, provides out-of-fold predictions to maintain orthogonality between nuisance functions.
Stacking relies on the component estimators' influence functions and does not re-fit nuisances at the stack level.

### 4. Oracle-Uncertainty Aware (OUA) Inference
When we calibrate judge scores using only a subset of oracle labels (e.g., 10% coverage), the calibration function f̂ itself has uncertainty. **OUA** uses delete-one-fold jackknife to add a **variance** component to standard errors, accounting for calibration learning uncertainty. Used by all Cal-IPS/DR estimators and enabled by default.

## Why Isotonic Regression for Reward Calibration?

Isotonic regression is the default choice for learning f̂(S) = E[Y|S] because it imposes exactly the right inductive bias while making minimal assumptions:

### The Right Structural Prior
**Monotonicity is all you want to assume**: If a judge says S₂ > S₁, the oracle label shouldn't go down in expectation. Isotonic regression enforces exactly this constraint—and nothing else. Unlike parametric approaches (sigmoid/beta), it doesn't impose a rigid functional form that could misspecify the true relationship.

### Mean Preservation by Construction
Least-squares isotonic regression is the orthogonal projection onto the monotone cone, which contains constants. By KKT conditions, this guarantees:
```
(1/n)Σf̂(Sᵢ) = (1/n)ΣYᵢ
```
Your oracle KPI level stays on the right scale automatically—critical for unbiased estimation without post-hoc adjustments.

### Small-Label Efficiency
With few oracle labels (5-10% coverage is often sufficient), shape constraints buy stability:
- **No overfitting**: Can't create spurious non-monotone regions
- **Adaptive complexity**: Naturally produces piecewise-constant regions when data supports them
- **Edge robustness**: Flattens at boundaries (explains why edge-slope diagnostics are so informative)
- **Fast iteration**: O(n log n) via PAVA algorithm

### Ranking-Sane and Interpretable
- **Never inverts judge order**: No "perverse" regions where higher S predicts lower Y
- **Human-readable**: Step blocks give actionable thresholds ("above 7.8, pass rate ≈ 0.81")
- **Operator trust**: Matches how humans think about judge reliability

### Diagnostic-Friendly
The 1-D monotone structure makes diagnostics tractable:
- **Reliability by region**: Easy to compute and visualize
- **S-coverage & edge slopes**: Fragility is visible; fix with targeted labels
- **OUA jackknife**: Delete-one-fold refits propagate calibrator noise cleanly

### Consistency Guarantees
When the true E[Y|S] is monotone, f̂ is L²-consistent. When it's not, the two-stage variant (g(S)→isotonic) provides a safety net by learning a smooth transformation first.

### When to Consider Alternatives
- **Parametric calibration (Platt/Beta)**: Lower variance if you're confident about the link shape
- **Two-stage flexible**: When S has systematic bias (length effects, prompt families)
- **Unconstrained methods**: Only if monotonicity fails and you have abundant oracle labels

**Bottom line**: Isotonic hits the sweet spot of correct inductive bias, minimal assumptions, strong stability with few labels, and clean uncertainty accounting—which is why it's the default in AutoCal-R.

## Why Two-Stage (Index → Rank → Isotonic) When Needed?

**Two-stage makes sense because it keeps the only belief we really trust—monotonicity—while fixing the two places plain isotonic on raw S can stumble: (i) regional miscalibration and slice heterogeneity, and (ii) density/scale weirdness along S.** It does this with almost no extra assumption and tiny label budgets.

### What the Two Stages Do

1. **Learn a low-capacity "risk index"** T = g(S, X_cov)
   - Uses a spline of S and optional covariates X_cov (e.g., response_length, domain)
   - Goal: cheaply improve the *ordering* of examples by expected outcome, not to nail absolute levels
   - **Covariate support**: Handles judge bias where judge scores at fixed S have different oracle outcomes based on observable features

2. **Uniformize & enforce shape**: U = ECDF(T) ∈ [0,1] then fit **isotonic** h(U)
   - Rank/ECDF makes the axis scale-free and density-balanced
   - Isotonic imposes exactly "higher risk index ⇒ no worse KPI"

3. **Mean-preserve & cross-fit**
   - We recentre so the oracle mean matches
   - We cross-fit so selection noise and kinks are handled
   - OUA jackknife then measures calibrator variance honestly

### Why This Is Better Than "Plain Isotonic on S"

**Captures slice heterogeneity with one degree of freedom.** If Y depends on both S and a coarse slice Z (e.g., long answers at the same S do worse), a single index g(S,Z) lets you *re-order* cases correctly, then isotonic maps that order to the KPI scale. You avoid brittle 2D isotonic or exploding bins.

**Stable at small n.** Shape constraint + 1D axis keeps variance down; ECDF removes density distortions (big flat regions where labels are dense vs. sparse).

**Consistent under a mild "single-index monotone" truth.** If E[Y|S,Z] = μ*(g*(S,Z)) with μ* nondecreasing, and g approximates g*, then isotonic on U=rank(g(S,Z)) recovers μ*∘g* in L². That's exactly the failure mode you see in reliability plots with "S-shaped" regional bias.

**Scale & transport friendly.** Because we learn on U (quantiles), the mapping is invariant to monotone rescalings of the index and more robust to density shifts; your **S-coverage/edge-slope** diagnostic still applies (now on U near 0/1).

**Interpretability preserved.** Output is still a monotone f(S,Z) ∈ [0,1] with mean preserved; panels (reliability by region, coverage, OUA share) remain readable.

### When to Auto-Switch to Two-Stage

- Reliability plot shows **persistent regional bias** after plain isotonic (low/mid/high S)
- Clear **slice effects at fixed S** (length, family) in residuals
- Edge fragility (flat boundary slopes) that a smarter ordering could stabilize

### Why Not Fancier Stuff?

- Full 2D monotone fits or unconstrained models need lots more labels and can invert judge order locally
- Parametric links (sigmoid/beta) impose shape you don't actually believe
- Two-stage keeps it nonparametric and monotone

### One-Line Mental Model

> **First get the order right (cheap index), then calibrate that order to the KPI scale (isotonic).**
> Minimal bias, low variance, and diagnostics stay meaningful.

That's why the two-stage AutoCal-R fallback is a great default: it fixes exactly the observed failure modes, assumes very little extra, and slots cleanly into OUA + your report panels.

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

## Key Design Decisions

### 1. **Separation of Concerns**
Each calibration type is isolated with clear interfaces:
- Reward calibration doesn't know about weights
- Weight calibration doesn't know about rewards
- Outcome models are separate from both

### 2. **Mean Preservation**
Calibrations preserve means for unbiased estimation:
- Isotonic preserves the **slice sample mean** exactly on the labeled calibration data; population mean preservation requires representative slice (MAR/MCAR), monotone relationship, and successful transport to target policies/contexts
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

# Default: Judge score only (no covariates, auto-selects monotone/two-stage via CV)
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    judge_field="judge_score",
    oracle_field="oracle_label"
)

# Include response_length covariate with two-stage calibration
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    use_response_length=True
)

# Add domain as additional covariate (combine with response_length)
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    use_response_length=True,
    covariate_names=["domain"]
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
