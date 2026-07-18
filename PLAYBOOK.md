# CJE Playbook: Audits, Drift Response, and Label Budgeting

This is the operational runbook for:
- running transport audits,
- deciding what to do when audits pass vs fail,
- planning oracle label budgets.

It is written against the current CJE API in this repo. CJE 0.4.0+ is Direct-mode only; this runbook is the whole operating loop. (Off-policy IPS/DR workflows live on the frozen 0.3.x line: `pip install "cje-eval==0.3.*"`.)

---

## Canonical Operational Loop

![CJE operational loop: design metrics, sample, fit, precision gate, deploy, monitor, drift gate](images/cje_loop.svg)

The recommended loop is:
1. Design metrics (choose/adjust `S` and `Y`)
2. Sample (judge scores + oracle labels)
3. Fit judge→oracle mapping with uncertainty
4. Precision gate: precise enough?
   - No -> collect more `Y` labels, return to Step 2
   - Yes -> deploy
5. Deploy
6. Monitor with new oracle labels and residuals
7. Residual-equivalence gate: is the simultaneous CI wholly inside a predeclared practical bias margin?
   - Yes (`PASS`) -> keep monitoring (Step 6)
   - No (`FAIL`) -> inspect failure patterns, collect target labels, then return to Step 1
   - Unresolved (`INCONCLUSIVE`) -> collect more independent probes or narrow the claim

Two-loop framing:
- **Inner calibration loop:** Steps 1-4
- **Outer monitoring loop:** Steps 5-7

This playbook’s sections below implement those same steps using CJE APIs.

---

## 1) Baseline Evaluation Run

Run your main analysis with bootstrap inference (default in Direct mode).

```python
from cje import analyze_dataset

results = analyze_dataset(
    fresh_draws_dir="responses/current_batch",
    estimator="direct",
    estimator_config={
        "inference_method": "bootstrap",
        "n_bootstrap": 2000,
        "use_augmented_estimator": True,
    },
)

print(results.estimates)
print(results.standard_errors)
```

Notes:
- `bootstrap` + `use_augmented_estimator=True` is the production default.

### Pairwise comparisons

Use `results.compare_policies(i, j)` for any "is A better than B?" claim — never eyeballed
CI overlap. On the bootstrap path it performs paired inference over the (B × P) replicate
matrix (`method: "paired_bootstrap"` in the returned dict), so the difference SE includes
calibrator-refit noise; pre-0.5.1 influence-function difference SEs were anti-conservative
on near-tie pairs (~90% false significance in the pre-registered benchmark). For many-pair
sweeps use `results.compare_all_policies(adjust="bh")` (Benjamini-Hochberg-adjusted
p-values).

---

## 2) Run a Transport Audit (Probe Protocol)

Use oracle-labeled probes that were not used to fit the calibrator. Sample them according to a documented probability design on each deployment-relevant policy/group. At least 20 effective independent clusters are required to grade the audit; the needed sample size is otherwise determined by the desired CI width and practical margin. Pass `TransportAuditConfig` to `analyze_dataset` to store every policy's state and merge an observed `FAIL` into the result gate:

```python
import json
from cje import TransportAuditConfig, analyze_dataset

probe = [json.loads(line) for line in open("probes/policy_gpt56mini.jsonl")]
transport = TransportAuditConfig(
    probes_by_policy={"gpt-5.6-mini": probe},
    delta_max_by_policy={"gpt-5.6-mini": 0.03},
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
    group_label="policy:gpt-5.6-mini",
    delta_max=0.03,  # practical mean-bias margin in public oracle units
    family_size=4,   # all policy/group audits used in this decision
)

print(diag.summary())
# Residual transport: PASS/FAIL/INCONCLUSIVE/NOT_GRADED | Group: ...
```

### Audit Thresholds

Predeclare `delta_max` from the smallest mean bias that would change the operational decision. The classifier uses a prompt-clustered CI for `delta = E[Y - f(S, X)]`, Bonferroni-adjusted by `family_size`:

- `PASS`: the entire simultaneous CI lies inside `[-delta_max, +delta_max]`.
- `FAIL`: the entire simultaneous CI lies outside that interval on either side.
- `INCONCLUSIVE`: the CI overlaps a margin boundary, or there are fewer than 20 effective clusters.
- `NOT_GRADED`: probes were evaluated but no practical margin was declared.
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
2. Re-run the Section 1 analysis adding `calibration_data_path="data/oracle_history.jsonl"` and `combine_oracle_sources=True` to pool all high-quality labels.

---

## 5) Correct for Failed Audits

### Step A: Collect more target-policy oracle labels

- Sample 100-200 labels across score deciles (not only difficult prompts).
- Keep labeling random within predeclared strata, retain inclusion probabilities, and use analysis weights when probabilities differ. The sampling design supports the argument; a KS score-balance check cannot prove ignorability.

### Step B: Apply policy-specific EIF correction (default)

Re-run `analyze_dataset` with the expanded oracle labels pooled in (same pattern as Section 4). The augmented estimator (`use_augmented_estimator=True`) estimates a policy-specific mean residual correction under the stated sampling assumptions. It does not by itself establish transport to a different policy or future cycle.

### Step C: Escalate to refit if residuals show structural drift

If decile residuals trend with score or covariates (not just a level shift), refit with `include_response_length=True` and/or `calibration_covariates`.

**How to decide:** Plot `Y - f_old(S)` by score decile. Flat = Step B suffices. Trending = Step C needed.

For the full analysis of correction strategies, see [Post-Audit Drift Correction](https://cimolabs.com/research/offset-vs-refit).

---

## 6) Label Budgeting (How Many Oracle Labels?)

Budgeting uses the planning model:
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
    n_replicates=150,  # extra-stable fit, ~20s since 0.5.1; the default (50) gives R2~0.85 in a few seconds
    verbose=True,
)

# No pilot yet? Use a scenario-specific simulation instead:
# from cje.diagnostics import simulate_variance_model
# variance_model = simulate_variance_model(r2=0.7, verbose=True)

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

### Label budgeting rules

- Sample oracle labels randomly (or randomly within predeclared strata), record inclusion probabilities, and weight unequal-probability samples.
- Use real dollar costs in `CostModel`.
- Refit the variance model periodically with new pilot/production evidence.
- If audit `FAIL` frequency rises, revisit the target population and calibration design before increasing `m_oracle` and rerunning planning.

### Planning caveats

- MDE assumes independent policies; paired evals on a shared prompt set typically detect smaller differences (plans are conservative).
- Reported variance shares are specific to the returned allocation: `(sigma2_eval/n) / V` and `(sigma2_cal/m) / V`. Raw fitted coefficients are not variance shares.
- Simulation planning is specific to its synthetic data-generating process. Keep the returned `scenario_fingerprint`, vary plausible inputs, and do not present a single simulated budget as an empirical guarantee.
- Variance components are measured with the analytic cluster-robust + OUA instrument, which tracked the realized SE of the production estimator within ~5% at every allocation in a pilot-scale validation grid (instrument experiment 2026-07-07, R=400 replicates/cell). Budgets planned with pre-0.5.1 versions used a bootstrap instrument that ran 15-29% hot at pilot-sized label counts — those older budgets were inflated upper bounds; re-running planning on the same pilot will typically return ~15-20% smaller budgets.

---

## 7) Suggested Operational Cadence

Per evaluation cycle:
1. Run `analyze_dataset(...)` with bootstrap inference.
2. Run transport audit on each deployment-relevant target policy.
3. Route by `PASS` / `FAIL` / `INCONCLUSIVE` / `NOT_GRADED` using Section 3.
4. Merge accepted audit labels into calibration history.
5. Re-run planning quarterly (or after judge/oracle regime changes).

---

## 8) Minimal Checklist

- Inference: `inference_method="bootstrap"`, `n_bootstrap=2000`
- Debiasing: `use_augmented_estimator=True`
- Audit margin: predeclare `delta_max` in public oracle units
- Probe design: held out, probability sampled, at least 20 effective independent clusters; size for the desired CI width
- FAIL response: collect target labels under a documented sampling design, re-run with pooled labels, and escalate to refit if residuals are score- or covariate-conditional
- Planning: `fit_variance_model` + `plan_evaluation` / `plan_for_mde`
