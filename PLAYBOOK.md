# CJE Playbook: Audits, Drift Response, and Label Budgeting

Use this playbook to compare observed policy responses, audit a reused judge calibration and plan future human reviews. Choose the analysis from the sampling design.

- For scored responses from each policy, use `analyze_dataset`. If human review was randomized at unequal rates, pass the known review probability for every evaluation row through `label_design="known_propensity"` and `label_propensities`. Historical labels used to fit calibration have their own acquisition design. [External calibration inputs and weights](cje/interface/README.md#dedicated-calibration-data)
- For paired responses sampled from a complete, fixed cluster list, use `weighted_paired_audit` when traffic scoring and subsequent human review were independent Bernoulli draws across clusters, with both probabilities known for every cluster. A selected-click export alone cannot support this audit. [Weighted paired audit](guides/weighted-paired-audit.md) · [Missing scores, outcomes and votes](guides/rater-and-missing-outcomes.md)
- For a future collection, use Section 6 to size an evaluation. If reviews will be allocated between a binary judge’s fails and passes, plan quotas for the intended human outcome. [Binary review planning](guides/evaluation-planning.md#plan-reviews-of-a-binary-judge)

---

## Canonical Operational Loop

1. Define the human outcome, target population and minimum candidate improvement required for deployment. Set the sampling design and stopping rule before collection.
2. Score responses with one fixed judge and rubric. Acquire human outcomes under the planned probability design.
3. Fit the judge-to-outcome mapping and estimate the paired candidate-minus-baseline difference.
4. At the planned look, read the observed paired interval, `results.best_policy().decision_note` and the reliability gates. Check the sampling assumptions and any calibration transfer the comparison relies on.
5. Deploy the candidate only if the comparison supports it and its lower bound clears the deployment threshold after allowing for any residual bias covered by a transport audit. If the interval straddles the threshold, report the decision as unresolved. Use a new planned sample or a prespecified sequential method for another look.
6. Monitor with new oracle labels and residuals
7. Residual-equivalence gate: is the simultaneous CI wholly inside a predeclared practical bias margin?
   - Yes (`PASS`) -> keep monitoring (Step 6)
   - No (`FAIL`) -> inspect failure patterns, collect target labels, then return to Step 1
   - Unresolved (`INCONCLUSIVE`) -> plan another independent audit or narrow the claim

Two-loop framing:
- **Inner calibration loop:** Steps 1-4
- **Outer monitoring loop:** Steps 5-7

This playbook’s sections below implement those same steps using CJE APIs.

---

## 1) Baseline Evaluation Run

Run your main analysis with the default calibration-aware jackknife inference.

```python
from cje import analyze_dataset

results = analyze_dataset(
    fresh_draws_dir="responses/current_batch",
    estimator="direct",
)

print(results.estimates)
print(results.standard_errors)
```

Notes:
- The default combines prompt-cluster-robust sampling variance with the
  delete-one-oracle-fold jackknife variance and a t-based interval. An
  approximate Welch–Satterthwaite df weights the two sources by their realized
  variance shares. When representative labels on the evaluation rows correct
  a policy's estimate, the correction is a mean over the labeled prompts, so
  the interval takes its df from them (at most `n_labeled − q`, with `q = 1`
  for the default weight and 2 for the tuned weight; lower when the
  oracle-jackknife term is large, as `oracle_df_cap_applied` reports) and
  scales their share of the sampling variance by `n_labeled / (n_labeled − q)`.
  With a handful of labeled prompts that interval is very wide, and with one
  it is unavailable.
- Refit bootstrap inference remains available explicitly with
  `estimator_config={"inference_method": "bootstrap", "n_bootstrap": 2000}`.

### Pairwise comparisons

For the calibrated run, use `results.compare_policies("candidate", "baseline")` to obtain the signed difference and its interval. The default analytic path combines the paired influence-function sampling SE
with the oracle-jackknife variance of the difference (`method: "paired_if_oua"`). On an
explicit bootstrap run it instead uses the joint (B × P) replicate matrix
(`method: "paired_bootstrap"`). For many-pair
sweeps use `results.compare_all_policies(adjust="bh")` (Benjamini-Hochberg-adjusted
p-values).

---

## 2) Run a Transport Audit (Probe Protocol)

Use oracle-labeled probes that were not used to fit the calibrator. Sample them according to a documented probability design on each deployment-relevant policy/group. At least 20 effective independent clusters are required for `PASS`; a decisive interval can still grade `FAIL` below that floor; the needed sample size is otherwise determined by the desired CI width and practical margin. Pass `TransportAuditConfig` to `analyze_dataset` to store every policy's state and merge an observed `FAIL` into the result gate:

```python
import json
from cje import TransportAuditConfig, analyze_dataset

probe = [json.loads(line) for line in open("probes/policy_gpt56mini.jsonl")]
transport = TransportAuditConfig(
    probes_by_policy={"candidate": probe},
    delta_max_by_policy={"candidate": 0.03},  # OUTPUT units (units of results.estimates)
    family_size=4,
)
results = analyze_dataset(fresh_draws_data=draws, transport=transport)
```

`results.calibrator` is the fitted public-unit calibration facade returned by `analyze_dataset`. The lower-level form is useful when the result already exists:

```python
import json
from cje.diagnostics import audit_transportability

probe = [json.loads(line) for line in open("probes/policy_gpt56mini.jsonl")]
diag = audit_transportability(
    calibrator=results.calibrator,
    probe_samples=probe,
    group_label="policy:candidate",
    delta_max=0.03,  # practical mean-bias margin, probe oracle-label units
    family_size=4,   # all policy/group audits used in this decision
)

print(diag.summary())
# Residual transport: PASS/FAIL/INCONCLUSIVE/NOT_GRADED | Group: ...
```

### Audit Thresholds

Predeclare `delta_max` from the smallest mean bias that would change the operational decision. The classifier uses a prompt-clustered CI for `delta = E[Y - f(S, X)]`, Bonferroni-adjusted by `family_size`:

- `PASS`: the entire simultaneous CI lies inside `[-delta_max, +delta_max]` — requires at least 20 effective clusters.
- `FAIL`: the entire simultaneous CI lies outside that interval on either side. FAIL is graded even below the effective-cluster floor: a decisive out-of-margin interval is evidence of unacceptable bias, not low power, so an under-sized probe cannot defeat the hard gate.
- `INCONCLUSIVE`: the CI overlaps a margin boundary, or there are fewer than 20 effective clusters without the CI being decisively outside.
- `NOT_GRADED`: probes were evaluated but no practical margin was declared — the residual estimate and CI are descriptive only, and the audit emits a `UserWarning` prompting you to declare one.
- `NOT_CHECKED`: no independent probe was supplied for that policy.

Pass `cluster_ids` when rows share an independence unit, `sample_weights` for unequal-probability probes, and the same fitted covariates used by the calibrator. Score-bin occupancy and decile residual plots are descriptive only; they never determine the verdict.

The automatic boundary card is not this audit. It checks scalar judge-score range extrapolation only and cannot establish residual transport, covariate support, or ranking validity.

---

## 3) Decision Policy After Audit

### PASS

Interpretation:
- The simultaneous CI establishes mean residual bias within the declared practical margin for this probe population and audit family.

Action:
- Proceed only with the claim and population covered by the declared audit.
- Keep normal monitoring cadence.
- Optionally merge probe labels into calibration history for the next cycle.

### INCONCLUSIVE

Interpretation:
- The evidence does not resolve the declared margin, or the effective cluster count is below 20.

Action:
- Collect more independent probability-sampled probes.
- Inspect residual structure by decile.
- Narrow the target population or relax the claim only with substantive justification.

### NOT_GRADED

Interpretation:
- No practical margin was supplied, so the residual estimate and CI are descriptive.

Action:
- Declare the decision-relevant margin before treating the audit as a gate.

### FAIL

Interpretation:
- Mean residual bias is established outside the declared margin for the audited population.

Action:
- Follow the correction protocol in Section 5 (collect labels → EIF correction → escalate to refit if needed).
- Treat this as Step 7 "drift detected" → return to Step 1 of the loop.

---

## 4) Incorporate New Audit Data After an Audit Cycle

After recording the held-out audit result, probe labels may become calibration data for a later cycle. Do not use the same rows both to fit a calibrator and to claim an independent audit of that fit.

Recommended pattern:
1. Append validated probe labels to a maintained calibration dataset.
2. If the historical and current labels support a pooled fit, pass `calibration_data_path="data/oracle_history.jsonl"` and `combine_oracle_sources=True`. To fit exclusively on historical labels sampled at known unequal rates, use `combine_oracle_sources=False` and name their positive relative fit-weight field in `calibration_weight_field`. Current labels can still correct their own policies’ estimates.

---

## 5) Correct for Failed Audits

### Step A: Collect more target-policy oracle labels

- Sample 100-200 labels across score deciles (not only difficult prompts).
- Keep labeling random within predeclared strata, retain inclusion probabilities, and use analysis weights when probabilities differ. The sampling design supports the argument; a KS score-balance check cannot prove ignorability.

### Step B: Apply policy-specific EIF correction (default)

Attach sampled target-policy `oracle_label` values to their matching evaluation responses and rerun `analyze_dataset`. Labels supplied only through `TransportAuditConfig` diagnose the calibration map without changing the estimate. The augmented estimator (`use_augmented_estimator=True`, already the default) applies the policy-specific mean residual correction when the labeling design supports it.

To preserve an external calibration fit, keep `calibration_data_path` and set `combine_oracle_sources=False`. This prevents those evaluation labels from refitting calibration while still allowing residual augmentation. Check `metadata["point_estimator"]["routes"]` for `augmented` and use the recomputed intervals. The [runnable guide](guides/audit-correction.md) demonstrates the complete workflow and label budget.

Retain the original audit as evidence about the old map. The same labels are no longer an independent validation sample for the corrected estimate, and correction does not establish transport to another policy or a future cycle.

### Step C: Escalate to refit if residuals show structural drift

If decile residuals trend with score or covariates (not just a level shift), refit with `include_response_length=True` and/or `calibration_covariates`.

**How to decide:** Plot `Y - f_old(S)` by score decile. Flat = Step B suffices. Trending = Step C needed.

For the full analysis of correction strategies, see [Post-Audit Drift Correction](https://cimolabs.com/research/offset-vs-refit).

---

## 6) Label Budgeting (How Many Oracle Labels?)

For a calibrated evaluation, the planner uses the following variance and cost model. To size the held-out probes needed to assess calibration transfer, use `plan_transport_audits` with a predeclared residual margin. [Transport audit budget](guides/audit-budget-planning.md)

- total variance `= sigma2_eval / n + sigma2_cal / m`
- costs `B = c_S * n + c_Y * m`
- optimize with square-root allocation law.

```python
from cje.data.fresh_draws import load_fresh_draws_auto
from cje.diagnostics import CostModel, fit_variance_model, plan_evaluation, plan_for_mde

# Fit variance model from pilot data (recommended)
base_pilot = load_fresh_draws_auto("responses/pilot", "base")
variance_model = fit_variance_model(
    base_pilot,
    n_replicates=150,  # extra-stable fit, ~20s; the default (50) gives R2~0.85 in a few seconds
    verbose=True,
)

# No pilot yet? Use a scenario-specific simulation instead. simulate_planning
# bundles the fitted variance model, the plan, and a scenario_fingerprint that
# records the simulated DGP (simulate_variance_model alone returns only a
# FittedVarianceModel, with no fingerprint):
# from cje.diagnostics import simulate_planning
# sim = simulate_planning(r2=0.7, budget=5000, cost_model=cost, verbose=True)
# plan, fingerprint = sim.plan, sim.scenario_fingerprint

cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)

# Fixed budget -> achievable sensitivity
plan = plan_evaluation(budget=5000, variance_model=variance_model, cost_model=cost)
print(plan.summary())
print(plan.n_samples, plan.m_oracle, plan.mde)

# Target sensitivity -> required budget
plan_target = plan_for_mde(
    target_mde=0.01,
    variance_model=variance_model,
    cost_model=cost,
)
print(plan_target.total_cost)
```

> **Note:** `fit_variance_model` measures variance components with the same analytic
> cluster-robust + oracle-jackknife instrument used by the default analysis path.

### Label budgeting rules

- Sample oracle labels randomly (or randomly within predeclared strata), record inclusion probabilities, and weight unequal-probability samples.
- Use real dollar costs in `CostModel`.
- Refit the variance model periodically with new pilot/production evidence.
- If audit `FAIL` frequency rises, revisit the target population and calibration design before increasing `m_oracle` and rerunning planning.

### Planning caveats

- MDE assumes independent policies. When shared-prompt outcomes have positive
  covariance, paired evaluation typically detects smaller differences, so that
  pairing assumption is conservative; negative covariance can reverse the
  direction.
- MDE and power use asymptotic-normal critical values. The final analytic CI uses a t critical value. When the oracle labels sit on the evaluated policy's rows (the augmented correction), its df is at most `m − 1` per policy (`n_labeled − 1` with the default weight), so planned power is optimistic at small `m`: about 77% instead of 80% at `m = 30` and about 70% at `m = 10`, when the realized SE equals the planned one. Otherwise the df comes from the realized evaluation/calibration variance shares. Confirm the achieved interval after collection.
- Reported variance shares are specific to the returned allocation: `(sigma2_eval/n) / V` and `(sigma2_cal/m) / V`. Raw fitted coefficients are not variance shares.
- Simulation planning is specific to its synthetic data-generating process. Keep the `scenario_fingerprint` returned by `simulate_planning(...)` (the simulation-planning entry point, which bundles the plan with its fingerprint — `simulate_variance_model` alone returns only the fitted variance model), vary plausible inputs, and do not present a single simulated budget as an empirical guarantee.
- Variance components are measured with the analytic cluster-robust + OUA instrument, which tracked the realized SE of the production estimator within ~5% at every allocation in a pilot-scale validation grid (instrument experiment 2026-07-07, R=400 replicates/cell). Since the labeled-cluster interval (issue #60) the instrument also scales the labeled prompts' share of the variance by `m / (m − 1)`; in its coupled measurement design it runs about 6–10% above the realized SD at 15–20 labels per policy (3–7% before; directional checks), which raises planned budgets by about 5%.

---

## 7) Suggested Operational Cadence

Per evaluation cycle:
1. Run `analyze_dataset(...)` with the default calibration-aware jackknife inference.
2. Run transport audit on each deployment-relevant target policy.
3. Route by `PASS` / `FAIL` / `INCONCLUSIVE` / `NOT_GRADED` using Section 3.
4. Merge accepted audit labels into calibration history.
5. Re-run planning quarterly (or after judge/oracle regime changes).

---

## 8) Minimal Checklist

- Inference: default `inference_method="cluster_robust"` + `oua_jackknife=True`
- Debiasing: `use_augmented_estimator=True`
- Audit margin: predeclare `delta_max` — probe oracle-label units for the low-level audit, OUTPUT units (the units of `results.estimates`) for `TransportAuditConfig` margins
- Probe design: held out, probability sampled, at least 20 effective independent clusters; size for the desired CI width
- FAIL response: collect target labels under a documented sampling design, re-run with pooled labels, and escalate to refit if residuals are score- or covariate-conditional
- Planning: `fit_variance_model` + `plan_evaluation` / `plan_for_mde`
