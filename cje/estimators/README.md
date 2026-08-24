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
cis = result.ci()                      # (lower, upper) tuples: t-based jackknife
                                       # by default; percentile under bootstrap
diagnostics = result.diagnostics       # DirectDiagnostics incl. boundary cards
```

Fresh draws are auto-discovered from a `fresh_draws_dir` under the canonical `POLICY_FILE_PATTERNS` names: `{policy}_responses.jsonl`, `{policy}.jsonl`, `responses/{policy}.jsonl`, `fresh_draws/{policy}.jsonl`.

## Standard Errors

On supported calibrated routes, the default `standard_errors` includes evaluation sampling noise **and** uncertainty from learning the calibrator on a finite oracle slice. The `cluster_robust` path combines CRV1 sampling variance with the calibration-aware oracle jackknife and uses t-critical values with an approximate Welch–Satterthwaite effective df (stored per policy in `result.metadata["degrees_of_freedom"]`). Bootstrap inference reports percentile intervals. Inspect `result.metadata["se_components"]["oracle_jackknife_status_per_policy"]`: disabling the jackknife or using a calibrator without enough fold models omits calibration variance and is reported there.

One exception: a policy whose fresh draws all share a single prompt cluster has fewer than two independent clusters, so no valid cluster-level inference exists. Such a policy returns its point estimate with **SE = NaN** (`se_method: "unavailable_one_cluster"`, loud warning) rather than falling back to invalid row-level IID inference. It is listed in `result.metadata["inference_unavailable_policies"]`, and `compare_policies` refuses pairs involving it with `InferenceUnavailableError` (from `cje.data`) instead of returning an anti-conservative difference SE.

### Inference methods (`inference_method` parameter)

- **`"cluster_robust"` (default):** CRV1 cluster-robust SE of the augmented pseudo-outcome mean (clustered by `prompt_id`), combined with the delete-one-oracle-fold jackknife variance and a t-based CI. An approximate Welch–Satterthwaite effective df weights the two components by their realized variance shares. The corrected additive variance procedure is the paper's recommended negligible-compute path; the effective-df rule is separately regression-tested for nominal coverage.
- **`"bootstrap"`:** cluster bootstrap by prompt — positive exponential mean-one weights per prompt cluster, no replicate discarded or retried — with a **calibrator refit per replicate**, applied to the same augmented estimate. By refitting and evaluating inside each replicate it jointly represents calibration/evaluation dependence. It returns percentile CIs and a joint replicate matrix for paired contrasts.
- **`"auto"`:** uses cluster_robust, switching to bootstrap when there are fewer than 20 prompt clusters or when the calibration data overlaps the evaluation draws (coupling). A run in which all evaluated policies have complete oracle coverage is not marked coupled because none of its point estimates uses the calibrator; mixed complete/partial runs still receive the coupling check.

```python
estimator = CalibratedDirectEstimator(
    target_policies=["policy_a", "policy_b"],
    reward_calibrator=cal_result.calibrator,
    inference_method="cluster_robust",  # default; or "bootstrap", "auto"
)
```

For backward compatibility, supplying `n_bootstrap` or `bootstrap_seed` while
omitting `inference_method` selects bootstrap with a warning. An explicit
`inference_method="cluster_robust"` wins and ignores those bootstrap-only
settings with a warning.

### Automatic fallback when bootstrap is selected

The refit bootstrap needs the exact rows the calibrator was fit on. When bootstrap is selected explicitly or through `auto` and those rows are unavailable, the estimator detects it before dispatching and falls back to cluster-robust + oracle jackknife, in exactly two cases:

- a calibrator exists but its fit rows (calibration provenance) are unavailable — `fallback_reason: "calibration_provenance_unavailable"`;
- no calibrator exists and at least one policy lacks complete evaluation oracle coverage — `fallback_reason: "calibrator_unavailable_for_non_oracle_routes"`.

A `calibration_data_path` run with **label-free** fresh draws is *not* a fallback case: the refit bootstrap runs normally on the calibration rows (`result.metadata["inference"]["method"] == "cluster_bootstrap_refit"`). When calibration and evaluation data are truly independent the additive variance decomposition is exact. When a fallback does occur, the downgrade is loud (warning) and recorded:

```python
result.metadata["inference"]
# {"method": "cluster_robust", "requested_method": "bootstrap",
#  "fallback_reason": "calibration_provenance_unavailable"}
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

Calibration uses k-fold cross-fitting. `fit_cv` assigns whole oracle **prompt clusters** to folds by a seeded-blake2b sort with round-robin assignment — deterministic given (prompt ids, seed, k) and balanced (fold sizes differ by at most one cluster, so small oracle slices cannot produce empty folds) — and resolves the fold count from unique labeled clusters. Fold membership depends on the whole oracle cluster set, so a single prompt's fold can change when the labeled set changes; `get_fold`/`get_folds_for_prompts` in `cje.data.folds` are stable hash utilities that do not predict calibration fold assignment — read the recorded assignments (`CalibrationResult.fold_ids`, `calibration_info["n_folds"]`) instead.

## Common Issues

- **"No fresh draws added"** — call `add_fresh_draws()` for every policy in `target_policies` before `fit_and_estimate()`.
- **"Only N oracle-labeled samples"** — cross-fitted calibration needs at least 2 labels per fold (10 for the default 5 folds); with 4–9 labels CJE reduces the fold count with a warning, below 4 it raises.
- **REFUSE-LEVEL badge** — not an error: do not ship absolute numbers from that calibration fit until labels cover the policy's score range. The scalar-support check alone does not certify rankings or residual transport.

## Summary

One estimator, honestly reported: `CalibratedDirectEstimator` turns calibrated judge scores on fresh draws into per-policy estimates with an analytic calibration-aware jackknife by default on supported calibrated routes, explicit uncertainty-component metadata, and a coverage gate that refuses level claims the data cannot support.
