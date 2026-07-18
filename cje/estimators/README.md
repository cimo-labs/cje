# CJE Estimators

## Overview

Direct-mode estimation: turn judge-scored fresh draws into per-policy value estimates with honest uncertainty quantification. The estimand is the **mean calibrated reward of each policy on a shared prompt set** — "which of these policies produces the best outputs on my eval set, and by how much?"

> **Note (0.4.0):** The off-policy estimators were removed; for IPS/DR workflows pin `pip install "cje-eval==0.3.*"`.

## File Structure

```
estimators/
└── direct_method.py    # CalibratedDirectEstimator
```

The OUA jackknife recipes it shares with the array API (`oracle_jackknife_variance`, `oracle_jackknife_estimates`, `combine_cluster_and_oracle`) live in `cje.diagnostics.robust_inference`.

## Common Interface

`analyze_dataset(...)` does all of this for you; use the estimator directly when you need control over calibration or inference settings.

```python
from cje.calibration import calibrate_dataset
from cje.estimators import CalibratedDirectEstimator

# 1. Learn the judge → oracle calibration (always cross-fitted)
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    judge_field="judge_score",
    oracle_field="oracle_label",
)

# 2. Build the estimator
estimator = CalibratedDirectEstimator(
    target_policies=["policy_a", "policy_b"],
    reward_calibrator=cal_result.calibrator,
)

# 3. Attach fresh draws per policy, then estimate
# estimator.add_fresh_draws("policy_a", fresh_draws_a)
# estimator.add_fresh_draws("policy_b", fresh_draws_b)
result = estimator.fit_and_estimate()

# 4. Access results
estimates = result.estimates           # Point estimates per policy
std_errors = result.standard_errors    # Complete SEs (sampling + calibration)
cis = result.ci()                      # (lower, upper) tuples: percentile bootstrap
                                       # by default; t-based under cluster_robust
diagnostics = result.diagnostics       # DirectDiagnostics incl. boundary cards
```

Fresh draws are auto-discovered from a `fresh_draws_dir` under the canonical `POLICY_FILE_PATTERNS` names: `{policy}_responses.jsonl`, `{policy}.jsonl`, `responses/{policy}.jsonl`, `fresh_draws/{policy}.jsonl`.

## Standard Errors

`standard_errors` always includes every uncertainty source: sampling noise on the eval set **and** the uncertainty from learning the calibrator on a finite oracle slice. Confidence intervals are percentile bootstrap intervals under the default `bootstrap` inference; under `cluster_robust` they use t-critical values with the limiting degrees of freedom (stored per policy in `result.metadata["degrees_of_freedom"]` with `df`, `t_critical`, `se_method`, `n_clusters` — that metadata only exists there).

### Inference methods (`inference_method` parameter)

- **`"bootstrap"` (default):** cluster bootstrap by prompt with a **calibrator refit per replicate**, applied to the augmented estimate `θ̂_aug = mean(f̂_full(S)) + mean(Y − f̂_oof(S))` (an AIPW-style per-policy residual correction; `use_augmented_estimator=True` by default). Refitting captures the calibration/evaluation covariance that analytic SEs miss — this is what achieves ~95% CI coverage.
- **`"cluster_robust"`:** CRV1 cluster-robust SE of the plug-in mean (clustered by `prompt_id` for paired comparisons), augmented with the oracle-jackknife variance. Fastest; undercovers when calibration and evaluation are coupled.
- **`"auto"`:** uses cluster_robust, switching to bootstrap when there are fewer than 20 prompt clusters or when the calibration data overlaps the evaluation draws (coupling).

```python
estimator = CalibratedDirectEstimator(
    target_policies=["policy_a", "policy_b"],
    reward_calibrator=cal_result.calibrator,
    inference_method="bootstrap",  # or "cluster_robust", "auto"
    n_bootstrap=2000,
)
```

### Automatic fallback when the eval draws carry no oracle labels

With a separate calibration source (`calibration_data_path`) and **label-free** fresh draws, the bootstrap's per-replicate refit has nothing to refit on. The estimator detects this before dispatching and falls back to cluster-robust + oracle jackknife — which is exact there, not an approximation: calibration and evaluation are independent, so the covariance the bootstrap exists to capture is zero. The downgrade is loud (warning) and recorded:

```python
result.metadata["inference"]
# {"method": "cluster_robust", "requested_method": "bootstrap",
#  "fallback_reason": "no_oracle_labels_in_evaluation_data"}
```

### Oracle uncertainty (calibration-aware inference)

`oua_jackknife=True` (default) adds the delete-one-oracle-fold jackknife variance so SEs reflect that the calibrator was *learned*, not given. Analytic inference reports `oracle_variance_per_policy`; the joint refit bootstrap captures calibration uncertainty by construction but does not claim a separate variance decomposition. The jackknife is skipped per policy when that policy routes directly to complete evaluation oracle labels.

### Paired comparisons

When multiple policies are evaluated on the same prompts (`paired_comparison=True`, default), difference inference preserves shared prompt weights and covariance. With `paired_comparison=False`, policy/prompt clusters receive independent weights and analytic differences combine per-policy SEs without prompt covariance. Per-policy method bookkeeping lives in `result.metadata["se_methods"]` and `["n_clusters"]`.

## The Coverage Gate (boundary cards)

`estimate()` computes the paper's coverage badge per policy: the fraction of that policy's judge scores falling **outside the calibrator's oracle S-range** (`calibrator.oracle_s_range`, recorded at fit time). Isotonic calibration extrapolates flatly outside its support, so out-of-range mass makes *level* claims untrustworthy even when rankings survive.

- Cards are attached to `result.diagnostics.boundary_cards` and `result.metadata["boundary_cards"]`.
- At ≥ 5% out-of-range mass (`OUT_OF_RANGE_REFUSE_THRESHOLD` in `cje.diagnostics.gates`), the card's status is **REFUSE-LEVEL**: the estimator warns loudly, sets that policy's status to CRITICAL, and flags it in `result.metadata["reliability_gates"]` (`flagged`, `refuse_level_claims`, `reasons`). The `cje analyze` CLI keeps the point winner visible and attaches the limitation.
- Fix: collect oracle labels covering the missing score range.

```python
for policy, card in (result.metadata.get("boundary_cards") or {}).items():
    print(policy, card["status"], f"{card['out_of_range']:.1%} out of range")
```

## Key Design Decisions

1. **Calibrate rewards, never fabricate them.** Without a `reward_calibrator`, estimation runs on raw judge scores and is loudly labeled `method="naive_direct"` — uncalibrated means are never passed off as calibrated results.
2. **Cluster by the source of dependence.** Prompts are the sampling unit; every inference path clusters by `prompt_id`.
3. **Influence functions are first-class.** Always computed and stored (`result.influence_functions`) for policy comparisons and downstream inference.
4. **Gates change the output.** Coverage violations alter statuses and metadata that the CLI and diagnostics consume — they are not log-only footnotes.

### Cross-fitting

Calibration uses k-fold cross-fitting with deterministic fold assignment from the unified fold system in `cje.data.folds` (`hash(prompt_id) % k`), so folds are stable across runs and datasets.

## Common Issues

- **"No fresh draws added"** — call `add_fresh_draws()` for every policy in `target_policies` before `fit_and_estimate()`.
- **"Only N oracle-labeled samples"** — cross-fitted calibration needs at least 2 labels per fold (10 for the default 5 folds); with 4–9 labels CJE reduces the fold count with a warning, below 4 it raises.
- **REFUSE-LEVEL badge** — not an error: do not ship absolute numbers from that calibration fit until labels cover the policy's score range. The scalar-support check alone does not certify rankings or residual transport.

## Summary

One estimator, honestly reported: `CalibratedDirectEstimator` turns calibrated judge scores on fresh draws into per-policy estimates with complete standard errors (bootstrap-with-refit by default), calibration-aware uncertainty, and a coverage gate that refuses level claims the data cannot support.
