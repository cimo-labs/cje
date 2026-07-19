# CJE Diagnostics System

For the operational runbook that combines diagnostics + action policy + budgeting, see [`PLAYBOOK.md`](../../PLAYBOOK.md).

## Overview

Diagnostics answer the question the estimates alone cannot: **should you trust this result?** CJE follows a **push-based architecture** — the estimator computes diagnostics during estimation and attaches them to results, so every `analyze_dataset(...)` run arrives with its own audit trail. Failing gates qualify the output with loud warnings and CRITICAL statuses; they do not silently replace the highest point estimate with a different policy.

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
├── robust_inference.py     # Bootstrap + cluster-robust inference
├── planning.py             # Budget optimization (Square Root Allocation Law)
├── simulation_planning.py  # Simulation-based planning (no pilot data required)
└── README.md               # This documentation
```

## DirectDiagnostics

The single diagnostics class since 0.4.0, attached to results as `results.diagnostics`. The former `IPSDiagnostics` compatibility alias is removed in 0.6.0; use `DirectDiagnostics`. `DRDiagnostics` and `CJEDiagnostics` were removed with the OPE estimators.

Field groups:

- **Identification**: `estimator_type` ("Direct"), `method` (`calibrated_direct` | `direct_oracle` | `naive_direct`, `_bootstrap` suffix when bootstrap inference ran), `policies`
- **Sample counts**: `n_samples_total`, `n_samples_valid`, per-policy `n_samples_used`
- **Results**: `estimates`, `standard_errors` (per-policy dicts)
- **Status**: `status_per_policy` (`Status.WARNING` for a CAUTION coverage badge, `Status.CRITICAL` when the badge refuses level claims; the card→status ladder is canonical in `gates.py`)
- **Coverage badges**: `boundary_cards` — serialized per-policy `BoundaryCard` dicts, including the `oracle_s_range` they were graded against
- **Calibration quality**: `calibration_rmse`, `calibration_coverage` (P(|pred − oracle| < 0.1)), `n_oracle_labels` (all None in naive/uncalibrated mode)

Common interface: `validate() -> List[str]` (self-consistency checks), `summary() -> str` (one-line), `to_dict()` / `to_json()` (serialization), plus computed properties `overall_status` (worst per-policy status), `best_policy`, `filter_rate`, `refuse_level_policies`, `is_calibrated`.

```python
from cje import analyze_dataset
from cje.diagnostics import Status

results = analyze_dataset(fresh_draws_dir="responses/")
diagnostics = results.diagnostics

if diagnostics.overall_status == Status.CRITICAL:
    print(diagnostics.summary())

# Per-policy scalar-support badges: any REFUSE-LEVEL card means do NOT ship
# level (absolute) claims for that policy. This check alone does not certify
# rankings or residual transport.
if diagnostics.boundary_cards:
    for policy, card in diagnostics.boundary_cards.items():
        print(f"{policy}: {card['status']} (out-of-range={card['out_of_range']:.1%})")
```

## Status System and Gates

`gates.py` is the single source of truth for thresholds and status mappings, so a given number receives the same verdict everywhere.

- `Status` enum: `GOOD` / `WARNING` / `CRITICAL`.
- `worst_status(*statuses)` combines statuses (None entries ignored).
- `OUT_OF_RANGE_REFUSE_THRESHOLD = 0.05` — the coverage-badge gate (below).
- `SATURATION_CAUTION_THRESHOLD = 0.20` — calibrated rewards piled near the oracle reward bounds → the badge's CAUTION signal.
- `TRANSPORT_FAIL_DELTA_THRESHOLD = 0.05` — deprecated compatibility constant; current residual audits require an explicit `delta_max` and do not use this threshold.

**Canonical status ladder** — every badge/audit verdict maps to a `Status` through these two dicts (both importable from `cje.diagnostics`):

| Mapping | Verdict → Status |
|---|---|
| `BOUNDARY_CARD_STATUS_TO_STATUS` | OK → GOOD, CAUTION / INCONCLUSIVE → WARNING, REFUSE-LEVEL → CRITICAL |
| `TRANSPORT_STATUS_TO_STATUS` | PASS → GOOD, FAIL → CRITICAL, INCONCLUSIVE / NOT_GRADED → WARNING, NOT_CHECKED → GOOD (no probe supplied is informational, not a defect — the per-policy audit record still says NOT_CHECKED; `WARN` is legacy-only) |

A CAUTION boundary card yields `Status.WARNING` for that policy; WARNING does not flag the policy or refuse level claims — only REFUSE-LEVEL (CRITICAL) does.

**Gate failures are loud but non-destructive**: estimates are still returned with warnings, CRITICAL statuses, and per-policy records in `result.metadata["reliability_gates"]` (`{"flagged", "refused", "refuse_level_claims", "reasons"}`). The `cje analyze` CLI keeps the highest point estimate visible and prints its `LIMITED` warning and diagnostics beside it.

## Coverage Badge (`boundary_card`, the REFUSE-LEVEL gate)

Judge scores outside the oracle calibration range are the primary identification threat to *level* claims: the calibrator must extrapolate there, and no data exists to check the extrapolation. `boundary_card(S_policy, S_oracle, R_policy, R_min, R_max, ...)` implements the paper's badge (arXiv:2512.11150) with three signals:

- **out_of_range ≥ 5%** (`OUT_OF_RANGE_REFUSE_THRESHOLD`) → status `REFUSE-LEVEL`: do not ship level (absolute) claims from that fit; the scalar check does not establish ranking validity
- **saturation ≥ 20%** (`SATURATION_CAUTION_THRESHOLD`; calibrated rewards piled near the oracle reward bounds) → `CAUTION`
- otherwise → `OK`

The returned `BoundaryCard` dataclass carries `status`, `out_of_range`, `saturation`, `partial_id_width` (a conservative partial-identification band under monotonicity), and a human-readable `note`.

**Wiring**: `CalibratedDirectEstimator.estimate()` computes a card per policy automatically, grading each policy's fresh-draw judge scores against the oracle S-range the reward calibrator recorded at fit time. Cards land in `diagnostics.boundary_cards` and `result.metadata["boundary_cards"]`; for a calibrator-dependent point route, REFUSE-LEVEL triggers a loud warning, sets that policy's status to CRITICAL, and flags it in `result.metadata["reliability_gates"]` (`refuse_level_claims`). A fully observed `direct_oracle` estimate does not use the calibrator, so its card remains descriptive with `applies_to_current_estimate=false` and cannot gate that estimate.

**Fixing a REFUSE-LEVEL badge**: collect oracle labels covering the missing score range (the warning names the range), then re-run.

## Transportability Audit

**Use case:** test whether a calibrator fitted on policy A / era 1 can be safely reused for policy B / era 2.

`audit_transportability(calibrator, probe_samples, bins=10, group_label=None, alpha=0.05, *, delta_max=None, cluster_ids=None, sample_weights=None, covariates=None, family_size=1, min_effective_clusters=20.0)` runs a practical-equivalence test on held-out oracle-labeled probes (rows that were **not** used to fit the calibrator):

1. Compute the (weighted) mean residual δ̂ = E[Y − f̂(S, X)] on the probe.
2. Construct the CI with a prompt-cluster sandwich SE and a finite-cluster t critical value at level 1 − α/`family_size` — a Bonferroni adjustment across every policy/group audit used in the same decision, so the K intervals are simultaneous at level 1 − α (`simultaneous_confidence_level` reports 1 − α; `per_audit_confidence_level` reports 1 − α/K).
3. Grade against the predeclared margin `delta_max`:
   - **PASS**: the CI lies wholly inside `[-delta_max, +delta_max]` (action: none). PASS additionally requires at least `min_effective_clusters` (default 20) Kish-effective clusters — cluster-robust intervals under-cover below that.
   - **FAIL**: the CI lies wholly outside the margin (action: do not reuse; collect target labels and refit). FAIL is graded even below the effective-cluster floor — an interval entirely beyond the margin is decisive evidence of unacceptable bias, not low power, so an under-sized probe cannot defeat the hard gate.
   - **INCONCLUSIVE**: the CI overlaps a margin boundary, or the probe has fewer than `min_effective_clusters` effective clusters without being decisively outside (action: collect more independent probe clusters).
   - **NOT_GRADED**: no `delta_max` was declared — the residual estimate and CI are descriptive and can never PASS or FAIL. Calling without a margin emits a `FutureWarning` for one release cycle, because 0.5.x graded the same call PASS/WARN/FAIL under a zero-null test.

`NOT_CHECKED` is the fifth state, reserved for high-level `analyze_dataset` results when no independent probe was supplied for a policy; the low-level audit never fabricates it. The 0.5.x `WARN` status is removed — legacy diagnostics constructed with `WARN` normalize to `INCONCLUSIVE` (`reason_code="legacy_warn"`).

**Units**: `delta_max` (and `delta_hat`/`delta_ci`) are in the units of the probe `oracle_label` values — this audit grades `oracle_label − calibrator.predict(...)` with no rescaling. For audits wired through `analyze_dataset(transport=TransportAuditConfig(...))`, probe labels are converted to the result OUTPUT scale first, so those margins are in output units (the units of `result.estimates`).

The per-decile residuals in the result are **display-only** — never gate on them.

```python
import json
from cje import analyze_dataset
from cje.diagnostics import audit_transportability, plot_transport_comparison

# analyze_dataset automatically fits and exposes the calibrator
results = analyze_dataset(fresh_draws_dir="responses/")

# Probe: held-out target rows (plain dicts with judge_score + oracle_label),
# probability sampled, with >= 20 effective independent prompt clusters
probe = [json.loads(line) for line in open("gpt56_mini_probe.jsonl")]

diag = audit_transportability(
    results.calibrator,
    probe,
    group_label="policy:gpt-5.6-mini",
    delta_max=0.05,  # predeclared practical margin, probe oracle-label units
    family_size=2,   # all policy/group audits used in this decision
)
print(diag.summary())
# Residual transport: PASS | Group: policy:gpt-5.6-mini | N=60 (60 clusters) | delta: +0.012 (CI: [-0.008, +0.032]) | margin: +/-0.050

diag.plot()  # residuals by score decile (requires the viz extra)

# Compare several policies at once (forest plot):
# audits = {"clone": diag_clone, "premium": diag_premium}
# fig = plot_transport_comparison(audits)
```

`TransportDiagnostics` carries `status`, `delta_hat`, `delta_ci`, `delta_se`, `delta_max`, `n_clusters`, `effective_clusters`, `alpha`, `family_size`, `simultaneous_confidence_level`, `per_audit_confidence_level`, `ci_half_width`, `min_margin_for_pass`, `detectable_bias_80`, `recommended_action`, `reason_code`, `weighted`, `decile_residuals`/`decile_counts` (for the plot; `probe_bin_occupancy` with deprecated alias `coverage`), `n_probe`, and `group_label`, plus `summary()`, `to_dict()`, and `plot()`.

**Probe sampling**: sample randomly (or randomly within score strata) — labeling only "interesting" cases biases the audit. Pass `cluster_ids` when rows share an independence unit, and `sample_weights` (inverse inclusion probabilities) for unequal-probability designs. Common uses: policy change, temporal drift, domain shift, judge/rubric updates.

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

**1. Cluster bootstrap with calibrator refit (the default)** — `cluster_bootstrap_direct_with_refit(eval_table, calibrator_factory, n_bootstrap=2000, ...)`. Each replicate draws positive exponential mean-one weights per prompt cluster (a Bayesian cluster bootstrap; the same cluster weights apply to every policy, preserving the paired design — `bootstrap_scheme: "positive_exponential_cluster_weights"` in the returned dict) and refits the calibrator under those weights, capturing the calibration/evaluation covariance analytic SEs miss. Every replicate counts: no replicate is discarded or retried, so the scheme never conditions on "easy" bootstrap worlds. By default it uses the augmented estimator θ̂_aug = mean(f̂_full(S)) + mean(Y − f̂_oof(S)), an AIPW-style debiasing of the plug-in. Returns percentile CIs plus the `bootstrap_matrix` for paired contrasts. `calibration_policy_idx` restricts calibrator fitting to one policy's labels (transport-aware bootstrap: the residual term then absorbs transport bias on the other policies).

**2. Cluster-robust SE + oracle jackknife** — `cluster_robust_se(data, cluster_ids, statistic_fn, ...)` computes CRV1 sandwich SEs with t-based CIs (df = G−1 clusters). The estimator augments this with the delete-one-oracle-fold jackknife variance, added once, so calibration uncertainty is never dropped. The shared OUA recipes live in `cje.diagnostics.robust_inference`: `oracle_jackknife_estimates` (the leave-one-oracle-fold loop), `oracle_jackknife_variance`, and `combine_cluster_and_oracle` (the SE/df combining rule) — one implementation used by both `CalibratedDirectEstimator` and `calibrated_mean_ci`. This is the automatic fallback when the evaluation draws carry no oracle labels at all (calibration-data-only runs) — there the calibration/evaluation covariance is exactly zero and the additive decomposition is exact, recorded in `result.metadata["inference"]`.

Supporting pieces: `build_direct_eval_table(fresh_draws_per_policy, covariate_names=None)` builds the long-format `DirectEvalTable` the bootstrap reweights; `make_calibrator_factory(mode, ...)` produces fresh `JudgeCalibrator` instances per replicate (passing `"auto"` re-runs mode selection in every weighted replicate, so model-selection uncertainty enters the interval). The returned `bootstrap_matrix` supports paired contrasts directly (same resampled clusters on both sides → tighter CIs than independence): `diff = boot["bootstrap_matrix"][:, a] - boot["bootstrap_matrix"][:, b]`.

```python
from cje.diagnostics import (
    build_direct_eval_table,
    cluster_bootstrap_direct_with_refit,
    make_calibrator_factory,
)

# fresh_draws = {"policy_a": FreshDrawDataset, "policy_b": FreshDrawDataset}
# eval_table = build_direct_eval_table(fresh_draws)
# boot = cluster_bootstrap_direct_with_refit(
#     eval_table, make_calibrator_factory("monotone"), n_bootstrap=2000
# )
# diff = boot["bootstrap_matrix"][:, 0] - boot["bootstrap_matrix"][:, 1]
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

No pilot data yet? `simulate_variance_model(r2=...)` builds a variance model from expected judge quality (`correlation_to_r2` converts a Pearson correlation), and `simulate_planning` wraps the exploration loop. The [planning notebook](../../examples/cje_planning.ipynb) walks through the full workflow, and `plot_planning_dashboard` (viz extra) visualizes the tradeoffs.

## Key Design Decisions

1. **Diagnostics are data, not behavior** — dataclasses with computed properties, serializable via `to_dict`/`to_json`.
2. **Push-based flow** — created during estimation, not on demand; every result is born audited.
3. **Warn loudly, never silently** — gate failures warn, set CRITICAL statuses, and are recorded in `metadata["reliability_gates"]`; numeric estimates are still returned so you can inspect them.
4. **One threshold source** — every surface imports gate thresholds from `gates.py`.
5. **Refuse levels; never certify rankings from a scalar badge** — the coverage badge restricts level (absolute) claims, which extrapolation invalidates. It does not establish that rankings survive: ranking claims need the paired comparisons plus separate residual/transport evidence.

## Common Issues

**"REFUSE-LEVEL" badge** — a policy's judge scores fall outside the oracle calibration range. Collect oracle labels covering that range (the warning names it); don't ship absolute numbers for that policy meanwhile.

**Transport audit FAIL** — the calibrator doesn't hold on the probe group. Collect 100–200 target-group labels and re-run with the labels pooled in; escalate to a refit with covariates if decile residuals trend with score (see PLAYBOOK Sections 3 and 5).

**Poor calibration (`calibration_rmse` high, `calibration_coverage` low)** — the judge doesn't predict the oracle well. Increase oracle coverage, improve the judge, or add calibration covariates.

## Summary

The 0.4.0 diagnostics system covers the two failure modes a Direct-mode evaluation actually has: calibration that doesn't cover a policy's scores (boundary cards) and calibration that doesn't transport (transport audit) — backed by inference that accounts for the calibrator being learned from finite labels (robust_inference) and planning tools to buy the right data (planning). Always check `results.diagnostics.summary()` before trusting estimates.
