# CJE Estimators

## Overview

Direct-mode estimation: turn judge-scored fresh draws into per-policy value estimates with honest uncertainty quantification. The estimand is each policy's **mean oracle label over the prompt population the evaluation prompts were sampled from**, estimated from calibrated judge scores on those prompts — "which of these policies produces the best outputs, and by how much?" The assumptions behind it are under [Methods](#methods).

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
estimator.add_fresh_draws("policy_a", fresh_draws_a)
estimator.add_fresh_draws("policy_b", fresh_draws_b)
result = estimator.fit_and_estimate()

# 4. Access results
estimates = result.estimates           # Point estimates per policy
std_errors = result.standard_errors    # Complete SEs (sampling + calibration)
cis = result.ci()                      # (lower, upper) tuples: t-based jackknife
                                       # by default; percentile under bootstrap
diagnostics = result.diagnostics       # DirectDiagnostics incl. boundary cards
```

A result built this way does not record which policies have oracle labels of their own, so
`best_policy().decision_ready` is always False and comparisons carry
`conditional_on_transport=None`; `analyze_dataset` records that provenance.

Fresh draws are auto-discovered from a `fresh_draws_dir` under the canonical `POLICY_FILE_PATTERNS` names: `{policy}_responses.jsonl`, `{policy}.jsonl`, `responses/{policy}.jsonl`, `fresh_draws/{policy}.jsonl`.

## Standard Errors

On supported calibrated routes, the default `standard_errors` includes evaluation sampling noise **and** uncertainty from learning the calibrator on a finite oracle slice. The `cluster_robust` path combines CRV1 sampling variance with the calibration-aware oracle jackknife and uses t-critical values with an approximate Welch–Satterthwaite effective df (stored per policy in `result.metadata["degrees_of_freedom"]`). For a policy whose estimate is corrected by representative labels (route `augmented`), the correction is a mean over its `n_L` labeled prompt clusters that fits `q` parameters on them (1 for weight one, 2 for the tuned weight), so the interval takes `n_L − q` degrees of freedom (capped by the Welch df with the jackknife's `K − 1`, and by the unadjusted interval's Welch df so it never narrows) and scales the labeled clusters' share of the CRV1 variance by `n_L / (n_L − q)` (`df_method: "labelled_clusters"`, with `n_labelled_clusters`, `fitted_parameters`, `labelled_variance_inflation`, `labelled_variance_share`, `labels_coupled` and `oracle_df_cap_applied`; issue #60). Paired comparisons inherit the rule. Bootstrap inference reports percentile intervals; they do not get this adjustment and also under-cover at about 10 to 20 labeled prompts, and `"auto"` resolves to the bootstrap for coupled or small designs. Inspect `result.metadata["se_components"]["oracle_jackknife_status_per_policy"]`: disabling the jackknife or using a calibrator without enough fold models omits calibration variance and is reported there.

One exception: a policy whose fresh draws all share a single prompt cluster has fewer than two independent clusters, so no valid cluster-level inference exists. Such a policy returns its point estimate with **SE = NaN** (`se_method: "unavailable_one_cluster"`, loud warning) rather than falling back to invalid row-level IID inference. It is listed in `result.metadata["inference_unavailable_policies"]`, and `compare_policies` refuses pairs involving it with `InferenceUnavailableError` (from `cje.data`) instead of returning an anti-conservative difference SE. The same holds for a policy corrected by representative labels with a single labeled prompt under weight one (`se_method: "unavailable_too_few_labelled_clusters"`). `result.metadata["inference_unavailable_reasons"]` says which (`"one_cluster"` or `"too_few_labelled_clusters"`). `compare_all_policies` raises for the whole table when any pair is refused; `to_dict(detail="portable")` skips the refused pairs.

### Inference methods (`inference_method` parameter)

- **`"cluster_robust"` (default):** CRV1 cluster-robust SE of the augmented pseudo-outcome mean (clustered by `prompt_id`), combined with the delete-one-oracle-fold jackknife variance and a t-based CI. An approximate Welch–Satterthwaite effective df weights the two components by their realized variance shares; augmented policies under representative labels take their df from their labeled prompt clusters instead (see Standard Errors above). The corrected additive variance procedure is the paper's recommended negligible-compute path; the effective-df rule is separately regression-tested for nominal coverage.
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

## Methods

### Estimand and assumptions

Each estimate targets the policy's mean oracle label over the prompt population your prompts were sampled from, reported on the label scale. It assumes:

1. the labels are a probability sample of the responses they describe (simple random under the default `label_design="representative"`, or known inclusion probabilities under `"known_propensity"`);
2. the judge and rubric stay fixed;
3. for a policy without labels of its own, the reused calibration has zero mean error on its responses (transport).

CJE cannot check (1) or (2); it reports (3) as `NOT_CHECKED` until a held-out audit grades it ([diagnostics](../diagnostics/README.md#claims-cje-refuses-to-make)). Confidence intervals include finite-label calibration uncertainty on supported inference paths ([Standard Errors](#standard-errors)).

### How it works

1. **Calibrate**: learn the judge → oracle mapping on the labeled slice (isotonic regression, unless the default auto mode's cross-validation prefers a two-stage variant that is isotonic in a learned index of the score, so it need not be monotone in the raw score; two-stage whenever covariates are used; cross-fitted by prompt). Details in the [calibration README](../calibration/README.md).
2. **Evaluate**: average each policy's calibrated scores; when a policy has random labels of its own, correct that average by their mean residual (label minus calibrated score; the `augmented` route). Compare policies on the same prompts.
3. **Diagnose**: automatically report scalar score-range support, and optionally run a held-out residual equivalence audit with a predeclared practical margin ([diagnostics](../diagnostics/README.md)).

A level claim concerns a policy's mean oracle outcome; a ranking concerns the oracle difference between two policies, which equals the calibrated-prediction difference plus the difference in their mean residuals. A common residual offset cancels, so a failed level audit does not by itself prove the ordering wrong, and a scalar support badge does not certify the ordering under a shift.

### Why Direct mode only

CJE is **Direct-mode only**: fresh draws, calibrated judge, audits. There is no off-policy machinery: no importance-sampling or doubly-robust estimators (`calibrated-ips`, `dr-cpo`, `mrdr`, `tmle`, `stacked-dr`), teacher forcing, SIMCal weight stabilization, or overlap diagnostics. Our own paper's results drove that design: for realistic LLM policy pairs, importance weighting failed even when ESS looked healthy (target-typicality coverage 0.19–0.49, far below the 0.70 gate), and the best DR stack merely matched Direct mode's accuracy at ~12× the compute.

- **Need IPS/DR from logged propensities?** Pin the frozen OPE line: `pip install "cje-eval==0.3.*"` (maintained on the `0.3.x` branch; docs at the `v0.3.0` tag; requires Python <=3.12).
- **Have old logged data with `judge_score` + `oracle_label`?** It works as the calibration source: `analyze_dataset(fresh_draws_dir=..., calibration_data_path="logged.jsonl")`. Its values are read as [0, 1] unless you declare `calibration_judge_scale=(lo, hi)` / `calibration_oracle_scale=(lo, hi)`.

### Validation against reference labels

- **HealthBench Consensus (29,511 response–criterion records)**: a custom confidence-augmented regrade found judge overconfidence of 24.5 (gpt-4o-mini) and 13.0 (Claude Haiku 4.5) percentage points against strict positive physician majority, with ties coded not met (14.4 and 3.0 points, respectively, against mean physician agreement). One seeded retrospective replay exposed 5% of aggregate labels (1,454 endpoints, each based on 2–5 physician grades); calibrated estimates were within 1.4–2.1 points of the full aggregate endpoint. This was not prospective annotation or repeated-split validation. [Read the full audit →](https://cimolabs.com/research/healthbench-judge-audit)
- **Chatbot Arena (4,961 prompts, 5 policies; GPT-5 ratings stand in for human labels, so this is a model-reference study, not human validation)**: 99% pairwise ranking accuracy in the headline 5%-oracle configuration, and 94% averaged across configurations with a response-length covariate (92% without it, the `analyze_dataset` default). The arXiv v3 cost model gives 14× lower total cost (oracle labels plus judge calls) than full labeling under that reference: 8.8× for a single policy, and reusing one calibration across the five policies raises it to 14× ([how many labels a judge saves](../../guides/evaluation-planning.md)). Nominal 95% intervals on raw judge means covered the reference mean 0% of the time; the calibration-aware intervals the library uses by default covered about 95% (93.9% Direct, 95.6% with the covariate) in a [corrected rerun](https://github.com/cimo-labs/cje-arena-experiments/blob/main/erratum_rerun/DELTAS.md) of the paper's experiments. A deliberately unhelpful policy, which the judge already ranks last but whose level the base-policy calibration overstates, fails the residual transport audit and is flagged. [Paper →](https://arxiv.org/abs/2512.11150)

The root README's forest plot is drawn from the shipped Arena sample (`examples/arena_sample`) by `scripts/make_readme_forest_plot.py`.

## Summary

One estimator, honestly reported: `CalibratedDirectEstimator` turns calibrated judge scores on fresh draws into per-policy estimates with an analytic calibration-aware jackknife by default on supported calibrated routes, explicit uncertainty-component metadata, and a coverage gate that refuses level claims the data cannot support.
