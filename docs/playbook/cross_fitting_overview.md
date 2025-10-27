# Cross-Fitting in CJE: A Comprehensive Overview

## Executive Summary

**Cross-fitting** (also called cross-validation or K-fold fitting) is a critical technique in CJE that enables **valid statistical inference** for doubly robust (DR) estimators by maintaining **orthogonality** between nuisance parameters. Without cross-fitting, DR estimators can have biased standard errors and invalid confidence intervals, even when point estimates converge to the truth.

CJE implements a unified fold management system (cje/data/folds.py) that ensures consistent fold assignment across all components using stable prompt_id hashing. Cross-fitting is **enabled by default** (`enable_cross_fit=True`) in CJE and used automatically by all DR estimators.

---

## What is Cross-Fitting?

### The Core Idea

Cross-fitting splits the data into K folds (typically K=5), then:

1. **Trains nuisance models** (calibrators, outcome models) on K-1 folds
2. **Makes predictions** on the held-out Kth fold
3. **Repeats** for all K folds to get **out-of-fold (OOF) predictions** for the entire dataset
4. **Combines** OOF predictions to compute the final estimate

This ensures predictions used in estimation are made by models that **never saw** those data points during training, breaking the dependence that would otherwise invalidate inference.

### Why Cross-Fitting Matters

**Problem without cross-fitting:** When you use the same data to both:
- Fit nuisance parameters (e.g., f̂(S) = E[Y|S], ĝ(X) = E[Y|X,A])
- Compute your final estimate

The estimation error becomes **correlated** with the nuisance fitting error, causing:
- **Biased standard errors** (usually underestimated)
- **Invalid confidence intervals** (don't achieve nominal coverage)
- **Overfitting artifacts** that look like real effects

**Solution with cross-fitting:** By using out-of-fold predictions:
- Nuisance errors and estimation errors are **independent** (orthogonal)
- Standard errors are **asymptotically valid**
- Confidence intervals achieve **correct coverage** (95% CIs cover truth 95% of the time)
- DR estimators achieve their **efficiency bound** (√n convergence)

### Mathematical Foundation: Neyman Orthogonality

DR estimators have the form:
```
θ̂_DR = (1/n) Σᵢ ψ(Zᵢ; η̂)
```

where ψ is the **score function** (influence function) and η̂ represents nuisance parameters (f̂, ĝ, ŵ).

**Key insight:** If ψ is **Neyman orthogonal** (satisfies dψ/dη|_{η=η*} = 0), then estimation error in η̂ only affects θ̂ at **second order**. This means:
- First-order bias vanishes: Error = O_p(||η̂ - η*||²)
- √n asymptotic normality is preserved even with √n-consistent nuisance estimates
- Valid inference is possible

Cross-fitting achieves this orthogonality by ensuring **independence** between η̂ and the data used for estimation.

---

## Where CJE Uses Cross-Fitting

### 1. Reward Calibration (Judge → Oracle Mapping)

**Purpose:** Learn f̂(S) = E[Y|S] to map judge scores to oracle labels

**Implementation:** `JudgeCalibrator` in cje/calibration/judge.py

**How it works:**
```python
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    judge_field="judge_score",
    oracle_field="oracle_label",
    enable_cross_fit=True,  # Default: True
    n_folds=5,              # Default: 5
    calibration_mode='auto' # Auto-select monotone or two-stage
)
```

**What happens:**
- Splits data into 5 folds based on `prompt_id` hashing (stable assignment)
- For each fold k:
  - Trains isotonic regression on folds {1,2,3,4} \ {k}
  - Predicts on fold k using f̂^(-k)(S)
- Stores both:
  - `fold_ids`: Fold assignment for each sample
  - `_fold_models`: K trained models (one per fold)

**Why it matters:**
- OOF predictions f̂^(-k)(S_i) are independent of oracle labels Y_i in fold k
- Enables honest uncertainty quantification via Oracle Uncertainty Augmentation (OUA)
- DR estimators use these OOF predictions for g(X,A) when outcome model uses calibrator

### 2. Outcome Models for DR Estimation

**Purpose:** Learn ĝ(X,A) = E[Y|X,A] for doubly robust correction term

**Implementation:** Outcome models in cje/estimators/outcome_models.py

**Supported Models:**
- `IsotonicOutcomeModel`: Monotone regression on rewards (default for DR)
- `LinearOutcomeModel`: Simple linear baseline
- `CalibratorBackedOutcomeModel`: Reuses reward calibrator structure
- `WeightedIsotonicOutcomeModel`: Policy-specific for MRDR with omega weights

**How it works:**
```python
# During DR estimation
for policy in target_policies:
    # Get cross-fitted outcome predictions
    g_logged = outcome_model.predict(
        logged_rewards,
        fold_ids=fold_ids,      # Uses OOF predictions
        covariates=covariates   # Optional features
    )
    g_fresh = outcome_model.predict(
        fresh_rewards,
        fold_ids=fresh_fold_ids,
        covariates=fresh_covariates
    )
```

**Key behaviors:**
- When `fold_ids` provided: Uses OOF model f̂^(-k) for fold k samples
- Without `fold_ids`: Falls back to single model (not cross-fitted)
- Supports **covariates** (e.g., response_length, domain) for two-stage calibration

### 3. SIMCal Weight Calibration (Optional)

**Purpose:** Stabilize importance weights via monotone projection

**Implementation:** `SIMCalCalibrator` in cje/calibration/simcal.py

**How it works:**
```python
calibrator = SIMCalCalibrator(
    n_folds=5,
    use_outer_cv=True,  # Default: True for honest inference
    var_cap=1.0        # Variance budget
)
calibrator.fit(weights_raw, ordering_index)
weights_calibrated = calibrator.transform(weights_raw, ordering_index)
```

**What happens:**
- Outer CV loop: Splits data, fits on K-1 folds, predicts on Kth
- Inner stacking: For each outer fold, stacks {baseline, ↑mono, ↓mono} candidates
- Variance capping: Projects to satisfy Var(W_cal) ≤ ρ·Var(W_baseline)

**Why outer CV matters:**
- Accounts for **uncertainty in learning** the calibration function
- Increases standard errors appropriately (honest inference)
- Prevents overly optimistic CIs that would arise from in-sample calibration

### 4. Stacked DR Estimator

**Purpose:** Optimal convex combination of DR methods via influence function weighting

**Implementation:** `StackedDREstimator` in cje/estimators/stacking.py

**Cross-fitting approach:**
- **Component estimators** (DR-CPO, TMLE, MRDR, OC-DR-CPO, TR-CPO) each use cross-fitted nuisances
- **Stacking weights** use outer V-fold CV (default V=5) via `v_folds_stacking` parameter
- Minimizes combined IF variance: min ||Σ_j α_j φ_j||² subject to Σα_j=1, α≥0

**Key insight:** Stacking itself doesn't refit nuisances—it relies on component IFs which already embed cross-fitted predictions. The outer CV is purely for weight selection to avoid overfitting the stacking weights.

---

## Fold Management System

### Unified Design Principle

**Single source of truth:** All fold assignment in CJE goes through `cje/data/folds.py`

```python
from cje.data.folds import get_fold, get_folds_for_dataset

# Get fold for a single prompt
fold = get_fold("prompt_123", n_folds=5, seed=42)  # Always returns same fold

# Get folds for entire dataset
fold_ids = get_folds_for_dataset(dataset, n_folds=5, seed=42)
```

**Hash-based assignment:** Uses BLAKE2b hash of `f"{prompt_id}-{seed}-{n_folds}"`
- **Stable:** Same prompt_id → same fold across runs
- **Survives filtering:** Removing samples doesn't change other fold assignments
- **Works with fresh draws:** Same prompt in logged + fresh data → same fold
- **Deterministic:** Seed makes it reproducible

### Why Prompt-Level Folding?

**Problem:** If we fold at the sample level (random assignment per sample), we'd leak information:
- Same prompt appears in both train and test folds
- Model learns patterns specific to that prompt
- Predictions aren't truly "out-of-fold"

**Solution:** Fold by `prompt_id`
- All samples with same prompt go to same fold
- Train/test split is at prompt level (natural cluster)
- Models can't memorize prompt-specific patterns

**Bonus:** Works seamlessly with fresh draws since fresh responses share prompt_ids with logged data.

### Oracle Fold Balancing (Legacy)

For backward compatibility with `JudgeCalibrator`, there's `get_folds_with_oracle_balance()`:
- Oracle samples: Round-robin assignment for perfect balance (important when oracle n is small)
- Unlabeled samples: Standard hash-based assignment
- Ensures each fold gets similar number of oracle labels for stable model fitting

**Note:** New code should use standard `get_folds_for_dataset()` which provides good balance in expectation without special handling.

---

## Cross-Fitting in the Ablations Study

### Configuration

**File:** `cje/experiments/arena_10k_simplified/ablations/config.py`

**Key parameters:**
```python
DR_CONFIG = {
    "n_folds": 5,              # K-fold cross-fitting for all DR methods
    "v_folds_stacking": 5,     # Outer folds for StackedDR weight selection
}
```

**Fixed across all experiments:**
- Cross-fitting is **always enabled** for DR methods (not an ablation axis)
- All DR estimators use K=5 folds (standard choice)
- Stacked-DR uses V=5 outer folds for meta-learning

### What Gets Ablated

The ablation study varies:
1. **Estimator type** (raw-ips, calibrated-ips, dr-cpo, tr-cpo-e, stacked-dr, etc.)
2. **Sample size** (250, 500, 1000, 2500, 5000)
3. **Oracle coverage** (5%, 10%, 25%, 50%, 100%)
4. **Weight calibration** (SIMCal on/off)
5. **Covariates** (response_length on/off)

**Cross-fitting is NOT ablated** because:
- It's a **requirement** for valid DR inference (not optional)
- Turning it off would give invalid standard errors (not comparable)
- The goal is to compare **correctly implemented** methods

### How Experiments Work

For each configuration:

```python
# 1. Load and calibrate dataset
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    enable_cross_fit=True,    # Always True for DR
    n_folds=5,
    calibration_mode='auto',
    covariate_names=['response_length'] if use_covariates else None
)

# 2. Create sampler
sampler = PrecomputedSampler(calibrated_dataset)

# 3. Initialize estimator
if estimator_name == "stacked-dr":
    estimator = StackedDREstimator(
        sampler=sampler,
        v_folds_stacking=5,  # Outer CV for stacking
        components=['dr-cpo', 'tmle', 'mrdr', 'oc-dr-cpo', 'tr-cpo-e']
    )
elif estimator_name in DR_ESTIMATORS:
    estimator = get_estimator(estimator_name, sampler)

# 4. Fit and estimate (uses cross-fitted models internally)
result = estimator.fit_and_estimate()
```

### What Gets Measured

**Outputs for each experiment:**
```jsonl
{
  "spec": {
    "estimator": "stacked-dr",
    "sample_size": 1000,
    "oracle_coverage": 0.25,
    "use_weight_calibration": true,
    "use_covariates": true,
    "seed": 42
  },
  "estimates": {...},              # Policy values
  "standard_errors": {...},        # Complete SEs (IF + MC + oracle)
  "confidence_intervals": {...},   # 95% CIs
  "rmse_vs_oracle": 0.0234,       # Error vs ground truth
  "orthogonality_scores": {        # DR orthogonality check
    "policy_1": {
      "score": 0.0012,             # Should be ≈ 0
      "ci": [-0.003, 0.005],       # CI should contain 0
      "passed": true
    }
  },
  "ess_relative": 0.82,           # Effective sample size (% of n)
  "diagnostics": {...}
}
```

**Key cross-fitting metrics:**
- **OOF RMSE** (`oof_rmse`): Out-of-fold calibration error (honest performance)
- **Orthogonality scores**: Tests whether DR achieves first-order insensitivity
  - Score ≈ 0 means orthogonality holds
  - CI containing 0 means test passes
  - Computed via `compute_orthogonality_score()` in diagnostics module
- **Fold stability**: Variance across folds in outcome model quality

---

## Technical Details

### Out-of-Fold Prediction Mechanics

**During calibration:**
```python
# JudgeCalibrator.fit_cross_fitted()
for k in range(n_folds):
    # Train folds
    train_mask = fold_ids != k
    test_mask = fold_ids == k

    # Fit model on train
    model = IsotonicRegression()
    model.fit(judge_scores[train_mask], oracle_labels[train_mask])

    # Predict on test (OOF)
    calibrated_scores[test_mask] = model.predict(judge_scores[test_mask])

    # Store model for later use
    self._fold_models[k] = model
```

**During DR estimation:**
```python
# Outcome model prediction
def predict(self, rewards, fold_ids=None, covariates=None):
    if fold_ids is None:
        # No cross-fitting: use single model
        return self.model.predict(rewards)

    # Cross-fitted: use OOF models
    predictions = np.zeros_like(rewards)
    for k in np.unique(fold_ids):
        mask = fold_ids == k
        predictions[mask] = self._fold_models[k].predict(rewards[mask])

    return predictions
```

### Orthogonality Score Computation

**What it tests:** Whether dψ/dη ≈ 0 empirically

**Implementation:** `compute_orthogonality_score()` in cje/diagnostics/dr.py

```python
def compute_orthogonality_score(
    residuals,      # Y - ĝ(X,A)
    weights,        # W(A)
    outcomes,       # Y
    n_folds=5
):
    """
    Regresses influence function on nuisance residuals.
    Under orthogonality, slope should be ≈ 0.

    Returns:
        score: Regression coefficient (should be ≈ 0)
        ci: Confidence interval (should contain 0)
        passed: Boolean test result
    """
    # Compute IF contributions
    phi = weights * (outcomes - g_hat)  # Simplified form

    # Regress IF on residuals (cross-fitted to avoid circularity)
    coef, se = robust_regression(phi ~ residuals, clusters=fold_ids)

    ci = [coef - 1.96*se, coef + 1.96*se]
    passed = (ci[0] <= 0 <= ci[1])

    return {"score": coef, "ci": ci, "passed": passed}
```

**Interpretation:**
- **Score ≈ 0:** Orthogonality holds, DR is valid
- **CI contains 0:** Test passes, no evidence of violation
- **Score >> 0:** Possible model misspecification or convergence issue

### Number of Folds (K) Selection

**Standard choice:** K=5
- **Why not K=2?** Too few training samples per fold, high variance
- **Why not K=10?** Diminishing returns, slower computation, smaller folds
- **K=5 sweet spot:** Good bias-variance tradeoff, standard in ML

**CJE default:** K=5 across all components
- Reward calibration: 5 folds
- Outcome models: 5 folds (reuses calibration folds)
- SIMCal outer CV: 5 folds
- Stacked-DR: V=5 outer folds

**Can be changed:**
```python
calibrate_dataset(dataset, enable_cross_fit=True, n_folds=10)  # Use K=10
```

### Memory and Computation

**Storage overhead:**
- Stores K models instead of 1 (minimal for isotonic regression)
- Keeps `fold_ids` array (int8, negligible)
- No duplication of data (just pointers to train/test splits)

**Computation:**
- K model fits instead of 1 (K× cost)
- Typically: 5× slower than no cross-fitting
- **Worth it:** Required for valid inference

**Optimization:**
- Parallel fold fitting possible (not currently implemented)
- Isotonic regression is fast: O(n log n) per fold
- Caching fold assignments speeds up repeated runs

---

## Benefits of Cross-Fitting in CJE

### 1. Valid Statistical Inference

**Without cross-fitting:**
- Standard errors underestimated by 20-50%
- 95% CIs cover truth only 85-90% of the time
- Hypothesis tests have inflated Type I error

**With cross-fitting:**
- Standard errors asymptotically correct
- 95% CIs achieve nominal coverage
- Hypothesis tests have correct size

### 2. Honest Performance Evaluation

**OOF metrics are honest:**
- OOF RMSE reflects true generalization error
- In-sample RMSE would be overly optimistic
- Diagnostics based on OOF predictions are trustworthy

### 3. Efficient Estimation

**DR achieves efficiency bound:**
- With cross-fitting: √n convergence to semiparametric efficiency bound
- Without: Slower convergence, higher asymptotic variance
- Enables tighter confidence intervals for same sample size

### 4. Robustness to Model Misspecification

**Orthogonality provides robustness:**
- First-order insensitivity to nuisance estimation errors
- Can use flexible models without paying bias penalty
- Allows adaptive/data-driven model selection

### 5. Oracle Uncertainty Quantification

**OUA (Oracle Uncertainty Augmentation):**
- Uses fold-jackknife on cross-fitted calibrator
- Quantifies uncertainty from partial oracle labels
- Automatically inflates SEs when oracle coverage is low

---

## Common Issues and Solutions

### Issue 1: "Missing fold_ids" Error

**Symptom:**
```
Error: DR estimator requires cross-fitted calibration.
Ensure calibrate_dataset() uses enable_cross_fit=True.
```

**Cause:** Calibration was done without cross-fitting

**Solution:**
```python
# Wrong
calibrated_dataset, _ = calibrate_dataset(dataset, enable_cross_fit=False)

# Correct
calibrated_dataset, _ = calibrate_dataset(dataset, enable_cross_fit=True)  # Default
```

### Issue 2: Different Folds for Logged vs Fresh Data

**Symptom:** Fresh draw predictions use different fold assignments

**Cause:** Fresh draws have same prompt_ids, so should automatically align

**Solution:**
```python
# Fold assignment is deterministic by prompt_id
logged_folds = get_folds_for_dataset(logged_dataset, n_folds=5, seed=42)
fresh_folds = get_folds_for_dataset(fresh_dataset, n_folds=5, seed=42)

# For matching prompt_ids, folds will be identical
assert np.all(logged_folds[logged.prompt_id == p] ==
              fresh_folds[fresh.prompt_id == p])
```

### Issue 3: Orthogonality Test Failing

**Symptom:** `orthogonality_scores[policy]["passed"] = False`

**Possible causes:**
1. Sample size too small (n < 500)
2. Model misspecification (linear outcome model when nonlinear)
3. Oracle coverage too low (< 5%)
4. Extreme weight variance (poor overlap)

**Solutions:**
- Increase sample size
- Use more flexible outcome model (isotonic > linear)
- Increase oracle coverage
- Check weight diagnostics (ESS, concentration)

### Issue 4: High OOF RMSE

**Symptom:** `oof_rmse >> in_sample_rmse`

**Interpretation:** Calibrator is overfitting

**Solutions:**
- Increase oracle coverage (more labels → less overfitting)
- Use simpler calibration (monotone instead of two-stage)
- Check for data quality issues (outliers, label noise)

---

## Best Practices

### 1. Always Enable Cross-Fitting for DR

```python
# Default is correct
calibrated_dataset, _ = calibrate_dataset(dataset, enable_cross_fit=True)

# Only disable if you have a very good reason (and don't care about inference)
```

### 2. Use Consistent Fold Seeds

```python
# Use same seed across calibration and estimation
SEED = 42
fold_ids = get_folds_for_dataset(dataset, n_folds=5, seed=SEED)
```

### 3. Check Orthogonality Diagnostics

```python
result = estimator.fit_and_estimate()

# Always inspect for DR methods
for policy, ortho in result.diagnostics.orthogonality_scores.items():
    if not ortho["passed"]:
        print(f"⚠️ Orthogonality test failed for {policy}")
        print(f"   Score: {ortho['score']:.4f}, CI: {ortho['ci']}")
```

### 4. Monitor OOF Performance

```python
# Compare in-sample vs OOF
print(f"In-sample RMSE: {cal_result.calibration_rmse:.4f}")
print(f"OOF RMSE: {cal_result.oof_rmse:.4f}")
print(f"Generalization gap: {cal_result.oof_rmse - cal_result.calibration_rmse:.4f}")

# Large gap → overfitting
if cal_result.oof_rmse > 1.5 * cal_result.calibration_rmse:
    print("⚠️ Possible overfitting detected")
```

### 5. Scale Fold Count with Data Size

```python
# Small data (n < 1000): Use K=3-5
# Medium data (n = 1000-5000): Use K=5
# Large data (n > 5000): Can use K=10

n_folds = min(10, max(3, n_samples // 200))  # Adaptive rule of thumb
```

---

## Summary

**Cross-fitting is essential infrastructure** in CJE that enables:
- ✅ Valid confidence intervals with correct coverage
- ✅ Honest out-of-fold performance metrics
- ✅ Efficient doubly robust estimation
- ✅ First-order insensitivity to nuisance errors
- ✅ Proper oracle uncertainty quantification

**Key implementation details:**
- Unified fold management via prompt_id hashing
- K=5 folds by default across all components
- Enabled by default (`enable_cross_fit=True`)
- Used automatically by all DR estimators
- Not ablated in experiments (always on)

**Bottom line:** Cross-fitting is not optional for DR methods—it's a requirement for valid statistical inference. CJE makes it automatic and seamless through careful API design and unified fold management.
