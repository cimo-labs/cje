# CJE Playbook: Audits, Drift Response, and Label Budgeting

This is the operational runbook for:
- running transport audits,
- deciding what to do when audits pass vs fail,
- planning oracle label budgets.

It is written against the current CJE API in this repo.

Advanced note: CJE supports IPS/DR variants for counterfactual OPE, but this runbook intentionally focuses on the default Direct-mode operating loop.

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
7. Drift gate: CI on mean residual excludes 0?
   - No -> keep monitoring (Step 6)
   - Yes -> inspect failure patterns, improve judge/oracle, then return to Step 1

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
        "use_multipolicy_eif": False,  # conservative default
    },
)

print(results.estimates)
print(results.standard_errors)
```

Notes:
- `bootstrap` + `use_augmented_estimator=True` is the production default.
- Keep `use_multipolicy_eif=False` unless you have evidence of shared calibration curves across policies.

---

## 2) Run a Transport Audit (Probe Protocol)

Use a small oracle-labeled probe slice (typically 40-60 rows) on the target policy/group. `results.calibrator` is the fitted calibration model returned by `analyze_dataset` in Section 1.

```python
import json
from cje.diagnostics import audit_transportability

probe = [json.loads(line) for line in open("probes/policy_gpt4mini.jsonl")]
diag = audit_transportability(
    calibrator=results.calibrator,
    probe_samples=probe,
    group_label="policy:gpt-4-mini",
)

print(diag.summary())
# Transport: PASS/WARN/FAIL | Group: ... | N=... | δ̂: ... | Action: ...
```

### Audit Thresholds

The classifier uses:
- `PASS`: `0` is inside the 95% CI for `δ̂ = E[Y - f(S)]`
- `WARN`: CI excludes 0, and `abs(δ̂) < 0.05`
- `FAIL`: CI excludes 0, and `abs(δ̂) >= 0.05`

---

## 3) Decision Policy After Audit

### PASS

Interpretation:
- No statistically detectable mean bias on the probe group (`0` inside CI).

Action:
- Reuse calibrator for current cycle (equivalent to Step 7 "No drift").
- Keep normal monitoring cadence.
- Optionally merge probe labels into calibration history for the next cycle.

### WARN

Interpretation:
- Detectable but small bias.

Action:
- Increase probe size next cycle.
- Inspect residual structure by decile.
- Consider anchoring updates through pooled recalibration if WARN persists.

### FAIL

Interpretation:
- Clear bias; do not rely on unchanged calibrator for high-stakes decisions.

Action:
- Follow the correction protocol in Section 5 (collect labels → EIF correction → escalate to refit if needed).
- Treat this as Step 7 "drift detected" → return to Step 1 of the loop.

---

## 4) Incorporate New Audit Data When Audits PASS

When probe results pass, treat probe labels as additional calibration signal for future runs.

Recommended pattern:
1. Append validated probe labels to a maintained calibration dataset.
2. Re-run the Section 1 analysis adding `calibration_data_path="data/oracle_history.jsonl"` and `combine_oracle_sources=True` to pool all high-quality labels.

---

## 5) Correct for Failed Audits

### Step A: Collect more target-policy oracle labels

- Sample 100-200 labels across score deciles (not only difficult prompts).
- Keep labeling random within strata to preserve ignorability.

### Step B: Apply policy-specific EIF correction (default)

Re-run `analyze_dataset` with the expanded oracle labels pooled in (same pattern as Section 4). The augmented estimator (`use_augmented_estimator=True`) automatically applies per-policy residual correction using the new labels — this fixes mean drift without refitting the calibrator.

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
variance_model = fit_variance_model(base_pilot, n_replicates=150, verbose=True)

# No pilot yet? Use simulation instead:
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

- Always sample oracle labels randomly (or randomly within strata).
- Use real dollar costs in `CostModel`.
- Refit the variance model periodically with new pilot/production evidence.
- If audit FAIL frequency rises, increase planned `m_oracle` and rerun planning.

---

## 7) Suggested Operational Cadence

Per evaluation cycle:
1. Run `analyze_dataset(...)` with bootstrap inference.
2. Run transport audit on each deployment-relevant target policy.
3. Route by PASS/WARN/FAIL using Section 3.
4. Merge accepted audit labels into calibration history.
5. Re-run planning quarterly (or after judge/oracle regime changes).

---

## 8) Minimal Checklist

- Inference: `inference_method="bootstrap"`, `n_bootstrap=2000`
- Debiasing: `use_augmented_estimator=True`
- Probe size: 40-60 oracle labels per target group
- FAIL response: collect 100-200 target labels, re-run with pooled labels (EIF correction is automatic); escalate to refit if residuals are score-conditional
- Planning: `fit_variance_model` + `plan_evaluation` / `plan_for_mde`
