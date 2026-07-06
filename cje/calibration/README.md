# CJE Calibration Module

## Overview

This module implements **reward calibration**: learning the mapping from cheap LLM-judge scores to expensive oracle labels, with automatic mode selection between standard isotonic regression and a two-stage flexible variant. The fitted calibrator is what turns judge scores into rewards everywhere else in CJE, and its cross-fitted fold models are what make calibration-aware inference (the oracle jackknife) possible.

## When to Use

- **`calibrate_dataset()`** — you have a `Dataset` with judge scores and a partially-labeled oracle slice and want calibrated rewards plus a reusable calibrator. This is what `analyze_dataset` calls internally.
- **`JudgeCalibrator`** — you are working with raw numpy arrays or need direct control over fitting (`fit_cv`) and prediction (`predict`, `predict_oof`).
- **`FlexibleCalibrator`** — used internally for non-monotone relationships and covariates; you rarely construct it yourself.

## File Structure

```
calibration/
├── __init__.py             # Public API exports
├── dataset.py              # calibrate_dataset workflow
├── judge.py                # JudgeCalibrator + CalibrationResult
└── flexible_calibrator.py  # Two-stage (index → rank → isotonic) calibration
```

## Core Concepts

### 1. Judge → oracle calibration with automatic mode selection

`calibration_mode='auto'` compares two candidates by cross-validation and picks per the 1-SE rule:

- **Monotone**: standard isotonic regression on S (chosen when the relationship is monotone — the common case).
- **Two-stage**: learn a smooth index g(S) (splines, optionally with covariates), rank-transform it, then fit isotonic on the ranks (chosen when there is regional miscalibration or covariate structure).

The selected mode is recorded (`calibrator.selected_mode`, and in dataset metadata under `calibration_info`) so runs are auditable.

### 2. Cross-fitting

`fit_cv()` fits a global model f_all (used for stable reward predictions) plus per-fold models f^(−k) for out-of-fold predictions. Fold assignment is deterministic via the unified fold system (`cje.data.folds`, `hash(prompt_id) % k`). Cross-fitting is what keeps downstream inference honest: OOF predictions never score a sample with a model that saw its label.

### 3. Calibration-aware inference (oracle uncertainty)

With oracle labels on only a slice, the calibrator f̂ is itself a noisy estimate. The per-fold models from `fit_cv` (exposed via `get_fold_models_for_oua()`) let estimators run a delete-one-fold jackknife that adds the calibration-learning variance to reported standard errors. Enabled by default in `CalibratedDirectEstimator` (`oua_jackknife=True`).

### 4. Fit-time support and quality recording

At fit time the calibrator records:

- `oracle_s_range` / `oracle_reward_range` — the judge-score and reward support of the oracle slice, consumed by the coverage badge (`boundary_card`) to refuse level claims for policies whose scores fall outside the calibrated range.
- Quality metrics (RMSE, coverage@0.1, oracle count, OOF variants) — exposed via `get_calibration_info()` and surfaced in Direct diagnostics.

## Why Isotonic Regression?

Isotonic regression is the default for learning f̂(S) = E[Y|S] because it imposes exactly the right inductive bias while assuming almost nothing else:

### The right structural prior
If the judge says S₂ > S₁, the oracle label shouldn't go *down* in expectation. Isotonic regression enforces exactly this constraint — and nothing else. Unlike parametric links (sigmoid/beta), it can't be misspecified in shape.

### Mean preservation by construction
Least-squares isotonic regression is the orthogonal projection onto the monotone cone, which contains constants. By the KKT conditions:

```
(1/n)Σf̂(Sᵢ) = (1/n)ΣYᵢ
```

The calibrated rewards preserve the oracle slice's sample mean exactly — no recentering step, no post-hoc adjustment. (Population-level preservation additionally needs a representative slice and successful transport; that is what the transport audit checks.)

### Small-label efficiency
With 5–25% oracle coverage, shape constraints buy stability: no spurious non-monotone regions, adaptive piecewise-constant complexity, and O(n log n) fitting via PAVA.

### Ranking-sane and interpretable
Never inverts judge order; step blocks read as actionable thresholds ("above 0.78, pass rate ≈ 0.81").

### When to consider alternatives
- **Parametric (Platt/beta)**: lower variance *if* you truly know the link shape.
- **Two-stage**: when S has systematic regional bias or slice effects (length, domain) — this is built in and auto-selected.
- **Unconstrained models**: only with abundant labels and evidence monotonicity fails.

## Why Two-Stage (Index → Rank → Isotonic) When Needed?

Two-stage keeps the only belief we trust — monotonicity — while fixing the two places plain isotonic on raw S stumbles: regional miscalibration / slice heterogeneity, and density or scale weirdness along S.

1. **Learn a low-capacity index** T = g(S, X_cov) with splines (minimum 5 knots) and optional covariates such as `response_length` or `domain`. The goal is a better *ordering*, not levels.
2. **Uniformize and enforce shape**: U = ECDF(T), then isotonic h(U). The rank transform makes the axis scale-free and density-balanced; per-fold ECDFs prevent leakage between folds.
3. **Mean-preserve and cross-fit**: the terminal isotonic stage preserves the oracle-slice mean by construction, and cross-fitting handles selection noise. Samples smaller than 20 fall back to monotone.

**Auto-selection logic** (in `FlexibleCalibrator`): compare monotone vs two-stage OOF performance overall (1-SE rule) and across low/mid/high score regions; select two-stage only when it is significantly better overall or better in ≥ 2/3 regions. Simpler wins ties.

> **One-line mental model:** first get the order right (cheap index), then calibrate that order to the KPI scale (isotonic).

## Common Interface

### Dataset-level calibration

```python
from cje.calibration import calibrate_dataset

# Default: cross-fitted, auto mode selection
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    judge_field="judge_score",
    oracle_field="oracle_label",
)

# Two-stage with covariates (response_length auto-computed from response text)
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    use_response_length=True,
    covariate_names=["domain"],
)

print(f"RMSE: {cal_result.calibration_rmse:.3f}")
print(f"Coverage@0.1: {cal_result.coverage_at_01:.1%}")
print(cal_result.summary())
```

`calibrate_dataset(dataset, judge_field="judge_score", oracle_field="oracle_label", n_folds=5, calibration_mode=None, use_response_length=False, covariate_names=None, random_seed=42)` returns `(calibrated_dataset, CalibrationResult)`. When `calibration_mode` is None it defaults to `'two_stage'` with covariates present, `'auto'` otherwise. Calibration is always cross-fitted; the fold count auto-reduces when labels are scarce (`resolve_n_folds`).

`CalibrationResult` fields: `calibrated_scores`, `calibration_rmse`, `coverage_at_01`, `n_oracle`, `calibrator`, `fold_ids`, `oof_rmse`, `oof_coverage_at_01`.

### Array-level calibration

```python
from cje.calibration import JudgeCalibrator

calibrator = JudgeCalibrator()  # calibration_mode="auto" by default
result = calibrator.fit_cv(judge_scores, oracle_labels, oracle_mask, n_folds=5)

rewards = calibrator.predict(judge_scores)                # global model
oof = calibrator.predict_oof(judge_scores, result.fold_ids)  # out-of-fold
info = calibrator.get_calibration_info()                  # fit-time metrics
```

(For a one-call version of this — calibrated mean with a CI from plain arrays — use `cje.calibrated_mean_ci`.)

## Key Design Decisions

1. **Mean preservation is structural, not corrective.** The isotonic projection preserves the slice mean by construction; there is no recentering code to drift or fail.
2. **Auto mode prefers the simpler model.** Two-stage must earn its complexity via the 1-SE rule or regional wins; ties go to monotone.
3. **Deterministic folds from `prompt_id`.** Calibration folds are reproducible across runs and shared with the estimators' jackknife, so uncertainty accounting composes correctly.
4. **Fit-time recording over recomputation.** Oracle S-range and quality metrics are captured when the calibrator is fitted — downstream consumers (boundary cards, diagnostics) never have to re-derive them from data they may not have.
5. **Numerical robustness.** Degenerate fold assignments fall back to fitting on all oracle samples; constant fits with varying labels warn loudly (inverted judge scale); predictions are clipped to [0, 1].

## Common Issues

- **Low calibration R² / high RMSE** — the judge poorly predicts the oracle. Increase oracle coverage (>10%), improve the judge prompt, or check label noise.
- **"Only N oracle-labeled samples"** — cross-fitted calibration needs at least 2 labels per fold (10 for the default 5 folds); with 4–9 labels CJE reduces the fold count with a warning, below 4 it raises.
- **Non-monotone reliability plots after monotone calibration** — auto mode should catch this; force `calibration_mode="two_stage"` (optionally with covariates) if you have external evidence.
- **Constant calibrated rewards while labels vary** — usually an anti-correlated (inverted) judge scale; the fit warns. Check the judge's score direction.

## References

- **Isotonic regression**: Robertson et al. (1988), *Order Restricted Statistical Inference*
- **PAV algorithm**: Ayer et al. (1955)
- **Cross-fitting**: Chernozhukov et al. (2018), *Double/Debiased Machine Learning*

## Summary

One calibration, done carefully: judge → oracle reward calibration via cross-fitted isotonic regression (two-stage when the data demands it), mean-preserving by construction, with fit-time support ranges and fold models that power CJE's coverage badge and calibration-aware standard errors.
