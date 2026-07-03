# CJE Diagnostics System

For the operational runbook that combines diagnostics + action policy + budgeting, see [`PLAYBOOK.md`](../../PLAYBOOK.md).

## Overview

Diagnostics answer the question the estimates alone cannot: **should you trust this result?** CJE follows a **push-based architecture** — the estimator computes diagnostics during estimation and attaches them to results, so every `analyze_dataset(...)` run arrives with its own audit trail. Failing gates change the output (loud warnings, CRITICAL statuses, a demoted "best policy" in the CLI); they are not footnotes.

Direct mode has no importance weights, so there are no weight/overlap metrics here (the 0.3.x ESS/tail-index/overlap diagnostics were removed with the OPE estimators — `pip install "cje-eval==0.3.*"` if you need them). The two identification risks that remain, and the two audits that cover them:

1. **Coverage** — did the calibrator see oracle labels where this policy's judge scores live? → the **boundary card** (coverage badge).
2. **Transport** — does a calibrator learned on one policy/era still hold on another? → the **transport audit**.

## File Structure

```
cje/diagnostics/
├── __init__.py             # Public API exports
├── gates.py                # Canonical gate thresholds + status helpers
├── models.py               # DirectDiagnostics, Status
├── reward_boundary.py      # Coverage badge (boundary_card, REFUSE-LEVEL gate)
├── transport.py            # Transportability auditing
├── display.py              # format_diagnostic_comparison
├── robust_inference.py     # Bootstrap + cluster-robust inference
├── planning.py             # Budget optimization (Square Root Allocation Law)
├── simulation_planning.py  # Simulation-based planning (no pilot data required)
└── README.md               # This documentation
```

## DirectDiagnostics

The single diagnostics class in 0.4.0, attached to results as `results.diagnostics`. (`IPSDiagnostics` survives as a deprecated alias of `DirectDiagnostics` for 0.3.x consumers that read the shared attributes; it will be removed in 0.5.0. `DRDiagnostics` and `CJEDiagnostics` were removed with the OPE estimators.)

Field groups:

- **Identification**: `estimator_type` ("Direct"), `method` (`calibrated_direct` | `naive_direct`, `_bootstrap` suffix when bootstrap inference ran), `policies`
- **Sample counts**: `n_samples_total`, `n_samples_valid`, per-policy `n_samples_used`
- **Results**: `estimates`, `standard_errors` (per-policy dicts)
- **Status**: `status_per_policy` (`Status.CRITICAL` when the policy's coverage badge refuses level claims)
- **Coverage badges**: `boundary_cards` — serialized per-policy `BoundaryCard` dicts, including the `oracle_s_range` they were graded against
- **Calibration quality**: `calibration_rmse`, `calibration_coverage` (P(|pred − oracle| < 0.1)), `n_oracle_labels` (all None in naive/uncalibrated mode)

Common interface: `validate() -> List[str]` (self-consistency checks), `summary() -> str` (one-line), `to_dict()` / `to_json()` / `to_csv_row()` (serialization), plus computed properties `overall_status` (worst per-policy status), `best_policy`, `filter_rate`, `refuse_level_policies`, `is_calibrated`.

```python
from cje import analyze_dataset
from cje.diagnostics import Status

results = analyze_dataset(fresh_draws_dir="responses/")
diagnostics = results.diagnostics

if diagnostics.overall_status == Status.CRITICAL:
    print(diagnostics.summary())

# Per-policy coverage badges: any REFUSE-LEVEL card means do NOT ship
# level (absolute) claims for that policy — rankings may stand.
if diagnostics.boundary_cards:
    for policy, card in diagnostics.boundary_cards.items():
        print(f"{policy}: {card['status']} (out-of-range={card['out_of_range']:.1%})")
```

## Status System and Gates

`gates.py` is the single source of truth for thresholds, so a given number receives the same verdict everywhere.

- `Status` enum: `GOOD` / `WARNING` / `CRITICAL`.
- `worst_status(*statuses)` combines statuses (None entries ignored).
- `OUT_OF_RANGE_REFUSE_THRESHOLD = 0.05` — the coverage-badge gate (below).

**Gate failures are loud but non-destructive**: estimates are still returned with warnings, CRITICAL statuses, and per-policy records in `result.metadata["reliability_gates"]` (`{"flagged", "refused", "refuse_level_claims", "reasons"}`). The `cje analyze` CLI consults those gates before crowning a winner — a flagged argmax prints "⚠️ Best by point estimate: X (UNRELIABLE — see diagnostics)" followed by the best reliable policy.

## Coverage Badge (`boundary_card`, the REFUSE-LEVEL gate)

Judge scores outside the oracle calibration range are the primary identification threat to *level* claims: the calibrator must extrapolate there, and no data exists to check the extrapolation. `boundary_card(S_policy, S_oracle, R_policy, R_min, R_max, ...)` implements the paper's badge (arXiv:2512.11150) with three signals:

- **out_of_range ≥ 5%** (`OUT_OF_RANGE_REFUSE_THRESHOLD`) → status `REFUSE-LEVEL`: do not ship level (absolute) claims; rankings may stand
- **saturation ≥ 20%** (calibrated rewards piled near the oracle reward bounds) or estimator gap ≥ 0.10 (a cross-estimator signal not computed by the 0.4.x pipeline) → `CAUTION`
- otherwise → `OK`

The returned `BoundaryCard` dataclass carries `status`, `out_of_range`, `saturation`, `estimator_gap`, `partial_id_width` (a conservative partial-identification band under monotonicity), and a human-readable `note`.

**Wiring**: `CalibratedDirectEstimator.estimate()` computes a card per policy automatically, grading each policy's fresh-draw judge scores against the oracle S-range the reward calibrator recorded at fit time. Cards land in `diagnostics.boundary_cards` and `result.metadata["boundary_cards"]`; a REFUSE-LEVEL card triggers a loud warning, sets that policy's status to CRITICAL, and flags it in `result.metadata["reliability_gates"]` (`refuse_level_claims`), which demotes it in the CLI's best-policy announcement.

**Fixing a REFUSE-LEVEL badge**: collect oracle labels covering the missing score range (the warning names the range), then re-run.

## Transportability Audit

**Use case:** test whether a calibrator fitted on policy A / era 1 can be safely reused for policy B / era 2.

`audit_transportability(calibrator, probe_samples, bins=10, group_label=None, alpha=0.05)` runs a simple unbiasedness test on a small oracle-labeled probe slice (typically 40–60 rows):

1. Compute the mean residual δ̂ = E[Y − f̂(S)] on the probe.
2. Construct the parametric (1−α) CI: δ̂ ± t_{1−α/2, n−1}·SE (t critical values, not z — at 40–60 probe rows the z interval under-covers and inflates the audit's false-alarm rate).
3. Classify:
   - **PASS**: 0 ∈ CI → calibrator is unbiased on the target (action: `none`)
   - **WARN**: 0 ∉ CI and |δ̂| < 0.05 → small but detectable bias (action: `monitor`)
   - **FAIL**: 0 ∉ CI and |δ̂| ≥ 0.05 → clear systematic bias (action: `refit_two_stage`)

The per-decile residuals in the result are **display-only** (4–6 rows per bin at recommended probe sizes — never gate on them). Auditing K policies at per-test α inflates the family-wise false-alarm rate; a BH correction is a planned option, not implemented.

```python
import json
from cje import analyze_dataset
from cje.diagnostics import audit_transportability, plot_transport_comparison

# analyze_dataset automatically fits and exposes the calibrator
results = analyze_dataset(fresh_draws_dir="responses/")

# Probe: 40-60 target samples, plain dicts with judge_score + oracle_label
probe = [json.loads(line) for line in open("gpt4_mini_probe.jsonl")]

diag = audit_transportability(
    results.calibrator,
    probe,
    group_label="policy:gpt-4-mini",
)
print(diag.summary())
# Transport: PASS | Group: policy:gpt-4-mini | N=50 | δ̂: +0.012 (CI: [-0.008, +0.032])

diag.plot()  # residuals by score decile (requires the viz extra)

# Compare several policies at once (forest plot):
# audits = {"clone": diag_clone, "premium": diag_premium}
# fig = plot_transport_comparison(audits)
```

`TransportDiagnostics` carries `status`, `delta_hat`, `delta_ci`, `delta_se`, `decile_residuals`/`decile_counts` (for the plot), `coverage`, `recommended_action`, `n_probe`, and `group_label`, plus `summary()`, `to_dict()`, and `plot()`.

**Probe sampling**: sample randomly (or randomly within score strata) — labeling only "interesting" cases biases the audit. Common uses: policy change, temporal drift, domain shift, judge/rubric updates.

**Inspecting individual residuals** when an audit fails:

```python
from cje.diagnostics import compute_residuals

samples = compute_residuals(results.calibrator, probe)  # sorted: worst overestimates first
for s in samples[:3]:
    print(f"residual={s['residual']:+.2f}  judge={s['judge_score']:.2f} "
          f"→ calibrated={s['calibrated']:.2f}  oracle={s['oracle_label']:.2f}")
```

`sort_by="residual"` (default) puts the samples that most fooled the judge first; `"abs_residual"` sorts by error magnitude; `None` preserves input order.

## Robust Inference

CJE's uncertainty has two independent sources: **evaluation sampling** (which prompts you drew) and **calibrator uncertainty** (the judge→oracle map was learned from a finite label budget). `robust_inference.py` provides the two inference engines `CalibratedDirectEstimator` dispatches between; both account for both sources.

**1. Cluster bootstrap with calibrator refit (the default)** — `cluster_bootstrap_direct_with_refit(eval_table, calibrator_factory, n_bootstrap=2000, ...)`. Each replicate resamples prompt clusters (shared across policies, preserving the paired design) and refits the calibrator on the replicate's oracle subset, capturing the calibration/evaluation covariance analytic SEs miss. By default it uses the augmented estimator θ̂_aug = mean(f̂_full(S)) + mean(Y − f̂_oof(S)), an AIPW-style debiasing of the plug-in. Returns percentile CIs plus the `bootstrap_matrix` for paired contrasts. `calibration_policy_idx` restricts calibrator fitting to one policy's labels (transport-aware bootstrap: the residual term then absorbs transport bias on the other policies).

**2. Cluster-robust SE + oracle jackknife** — `cluster_robust_se(data, cluster_ids, statistic_fn, ...)` computes CRV1 sandwich SEs with t-based CIs (df = G−1 clusters). The estimator augments this with the delete-one-oracle-fold jackknife variance (`oracle_jackknife_variance` in `cje.estimators.base_estimator`), added once, so calibration uncertainty is never dropped. This is the automatic fallback when the evaluation draws carry no oracle labels at all (calibration-data-only runs) — there the calibration/evaluation covariance is exactly zero and the additive decomposition is exact, recorded in `result.metadata["inference"]`.

Supporting pieces: `build_direct_eval_table(fresh_draws_per_policy, covariate_names=None)` builds the long-format `DirectEvalTable` the bootstrap resamples; `make_calibrator_factory(mode, ...)` produces fresh `JudgeCalibrator` instances per replicate (mode fixed to the full-data selection, never "auto"); `compare_policies_bootstrap(bootstrap_result, policy_a, policy_b)` computes paired contrasts from the stored bootstrap matrix (same resampled clusters on both sides → tighter CIs than independence).

```python
from cje.diagnostics import (
    build_direct_eval_table,
    cluster_bootstrap_direct_with_refit,
    compare_policies_bootstrap,
    make_calibrator_factory,
)

# fresh_draws = {"policy_a": FreshDrawDataset, "policy_b": FreshDrawDataset}
# eval_table = build_direct_eval_table(fresh_draws)
# boot = cluster_bootstrap_direct_with_refit(
#     eval_table, make_calibrator_factory("monotone"), n_bootstrap=2000
# )
# contrast = compare_policies_bootstrap(boot, policy_a=0, policy_b=1)
```

Most users never call these directly — `analyze_dataset` / `CalibratedDirectEstimator` route through them and record the SE decomposition in `result.metadata["se_components"]` and the method actually used in `result.metadata["inference"]`.

## Budget Planning

How many prompts and how many oracle labels? Total variance decomposes as σ²_eval/n + σ²_cal/m; given per-unit costs, the Square Root Allocation Law gives the optimal split (paper Appendix F).

```python
from cje.diagnostics import CostModel, fit_variance_model, plan_evaluation, plan_for_mde

# model = fit_variance_model(base_pilot_data)      # from pilot fresh draws
# cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
# plan = plan_evaluation(budget=5000, variance_model=model, cost_model=cost)
# plan = plan_for_mde(target_mde=0.01, variance_model=model, cost_model=cost)
```

No pilot data yet? `simulate_variance_model(r2=...)` builds a variance model from expected judge quality (`correlation_to_r2` converts a Pearson correlation), and `simulate_planning` / `simulate_planning_sweep` wrap the exploration loop. The [planning notebook](../../examples/cje_planning.ipynb) walks through the full workflow, and `plot_planning_dashboard` (viz extra) visualizes the tradeoffs.

## Display

`format_diagnostic_comparison(diag1, diag2, label1="Run 1", label2="Run 2")` renders a side-by-side text table of two `DirectDiagnostics` objects (sample counts, calibration RMSE, per-policy estimates) — useful for before/after comparisons across runs.

## Key Design Decisions

1. **Diagnostics are data, not behavior** — dataclasses with computed properties, serializable via `to_dict`/`to_json`/`to_csv_row`.
2. **Push-based flow** — created during estimation, not on demand; every result is born audited.
3. **Warn loudly, never silently** — gate failures warn, set CRITICAL statuses, and are recorded in `metadata["reliability_gates"]`; numeric estimates are still returned so you can inspect them.
4. **One threshold source** — every surface imports gate thresholds from `gates.py`.
5. **Refuse levels, keep rankings** — the coverage badge distinguishes level (absolute) claims, which extrapolation invalidates, from rankings, which often survive.

## Common Issues

**"REFUSE-LEVEL" badge** — a policy's judge scores fall outside the oracle calibration range. Collect oracle labels covering that range (the warning names it); don't ship absolute numbers for that policy meanwhile.

**Transport audit FAIL** — the calibrator doesn't hold on the probe group. Collect 100–200 target-group labels and re-run with the labels pooled in; escalate to a refit with covariates if decile residuals trend with score (see PLAYBOOK Sections 3 and 5).

**Poor calibration (`calibration_rmse` high, `calibration_coverage` low)** — the judge doesn't predict the oracle well. Increase oracle coverage, improve the judge, or add calibration covariates.

## Summary

The 0.4.0 diagnostics system covers the two failure modes a Direct-mode evaluation actually has: calibration that doesn't cover a policy's scores (boundary cards) and calibration that doesn't transport (transport audit) — backed by inference that accounts for the calibrator being learned from finite labels (robust_inference) and planning tools to buy the right data (planning). Always check `results.diagnostics.summary()` before trusting estimates.
