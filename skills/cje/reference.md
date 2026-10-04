# CJE reference (for agents)

Load this file when you need full signatures, the planning API, the CLI, or troubleshooting.
The workflow and hard rules live in `SKILL.md`; this file is detail only.

## Data formats

**Record schema** (one dict per response, any policy):

| Field | Required | Notes |
|---|---|---|
| `judge_score` | yes | Any bounded scale (0–1, 0–100, Likert 1–5). Auto-normalized; results return in the original scale (`metadata["normalization"]`). |
| `oracle_label` | no | Ground-truth on the labeled slice. `None`/`NaN`/missing = unlabeled. Same scale conventions. |
| `prompt_id` | no | Enables paired within-prompt comparisons across policies (lower-variance). Auto-generated from a hash of `prompt` if absent. |
| `response` | no | Only needed for `include_response_length=True`. |
| `metadata` | no | Dict; fields here are usable as `calibration_covariates`. |

Logprob fields from 0.3.x logged data are accepted and ignored.

**Three ways to supply data to `analyze_dataset`:**

1. `fresh_draws_data={policy_name: [records]}`: in-memory, the default choice when you reshaped the user's data yourself.
2. `fresh_draws_dir="responses/"`: one JSONL per policy, named `{policy}_responses.jsonl` (also accepted: `{policy}.jsonl`). Policy name comes from the filename; keep names identical everywhere. A single JSONL file path (records grouped by `target_policy`) also works here.
3. `calibration_data_path="labeled.jsonl"`: a separate judge+oracle file (e.g. historical labeled logs) used as the calibration source. Values default to [0, 1]; for other scales declare `calibration_judge_scale=(lo, hi)` / `calibration_oracle_scale=(lo, hi)` (external calibration data never infers its scale from observations). Out-of-range files raise a hard error naming the observed range. With `combine_oracle_sources=True` (default) any `oracle_label`s in the fresh draws are pooled with it; `metadata["oracle_sources"]` reports provenance and cross-source conflicts.

Field names differ in the user's data? Pass `judge_field="score"`, `oracle_field="human_rating"` instead of renaming.

## `analyze_dataset`

All arguments are keyword-only. Provide exactly ONE evaluation source
(`fresh_draws_data` or `fresh_draws_dir`; passing both raises):

```python
from cje import analyze_dataset

results = analyze_dataset(
    fresh_draws_data=None,        # {policy: [records]}  (or use fresh_draws_dir=...)
    fresh_draws_dir=None,
    calibration_data_path=None,   # separate judge+oracle JSONL
    combine_oracle_sources=True,
    estimator="auto",             # "auto"/"direct"/"calibrated-direct": same estimator (see below)
    judge_field="judge_score",
    oracle_field="oracle_label",
    calibration_covariates=None,  # e.g. ["domain"] — numeric fields from record metadata
    include_response_length=False,  # auto word-count covariate (needs "response")
    estimator_config=None,        # default jackknife; bootstrap explicit or via auto.
                                  # n_bootstrap/bootstrap_seed alone select bootstrap (warns) —
                                  # set inference_method explicitly to avoid surprises
    verbose=False,
    fresh_judge_scale=None,       # declared (min, max) for evaluation judge scores
    fresh_oracle_scale=None,      # declared scale for oracle labels in the fresh draws
    calibration_judge_scale=None,   # declared scale for the calibration file's judge scores
    calibration_oracle_scale=None,  # declared scale for the calibration file's oracle labels
    output_scale=None,            # display axis only — never changes the estimand label
    strict=False,                 # retained for compatibility; invalid records raise by default
    on_invalid=None,              # default "error" (loud); "drop" filters with counted logging
    label_design="representative",  # or "known_propensity" / "targeted_unknown"
    label_propensities=None,      # per-policy inclusion probs for "known_propensity"
    transport=None,               # TransportAuditConfig with held-out probes (below)
)
```

**Wiring transport audits into the analysis:** probes are held-out oracle-labeled
rows in the same public units and field names as the call; margins are in OUTPUT
units (the units of `results.estimates`):

```python
from cje import TransportAuditConfig, analyze_dataset

transport = TransportAuditConfig(
    probes_by_policy={"candidate": probe_rows},   # {policy: [records]}, held out
    delta_max_by_policy={"candidate": 0.03},      # OUTPUT units; omitting -> NOT_GRADED
    bins=10,
    alpha=0.05,
    family_size=None,             # defaults to the number of configured probes
    min_effective_clusters=20.0,
)
results = analyze_dataset(fresh_draws_data=draws, transport=transport)
```

`estimator` values are aliases of the one calibrated estimator: whether calibration
runs is driven solely by oracle-label availability (0 labels → the loud `naive_direct`
fallback), never by this parameter; its only observable effect is which name lands in
`metadata["estimator"]` (`"auto"` records `"direct"`).

**`EstimationResult`:**

- `.estimates` (np.ndarray, order matches `metadata["target_policies"]`), `.standard_errors`
- `.ci(alpha=0.05)` → list of `(lo, hi)` per policy; `.confidence_interval()` → `(lo_array, hi_array)`
- `.compare_policies(i, j, alpha=0.05)` → dict with difference, SE, CI, p-value; use this for
  pairwise claims. The `method` key names the inference basis, best-first: `"paired_bootstrap"`
  (bootstrap runs: paired inference over the replicate matrix; the difference SE includes
  calibrator noise, honest on near-tie pairs; sign-test p-value floored at 2/(B+1)),
  `"paired_if_oua"` (cluster-robust runs: t-test from the stored pairwise SE + oracle-jackknife
  difference variance), `"paired_if_legacy"` (only for deserialized results from older releases
  that stored unpaired IF z-tests), `"independent_conservative"` (no pairing info). `gate_flagged` lists any policy in
  the pair with a flagged reliability gate; a difference CI cannot repair a biased input
  (e.g. after a transport-audit FAIL), so treat such comparisons per the gates discipline
- `.compare_all_policies(alpha=0.05, adjust=None)` → list of comparison dicts for every (i < j)
  pair with `policy1`/`policy2` names; `adjust="bh"` adds Benjamini-Hochberg
  `p_adjusted`/`significant_adjusted` for many-pair audits
- `.bootstrap_samples` → (B, P) bootstrap replicate matrix on bootstrap runs (columns follow
  `metadata["target_policies"]`); powers paired comparisons, omitted from default portable
  JSON export (`to_dict(detail="full")` retains it)
- `.best_policy()` → PolicyVerdict (name, index, estimate, flagged, all_flagged, runner_up,
  runner_up_reasons); defaults to `reliable_only=True`: a gate-flagged argmax is demoted to the
  best gate-passing policy, loudly (the demoted argmax travels as `runner_up` with its gate
  reasons, a warning is logged, and `summary()` prints both). Pass `reliable_only=False` for the
  raw argmax with `flagged=True`. If everything is flagged, the argmax returns with
  `all_flagged=True`; do not crown it
- `.calibrator` → fitted calibrator when calibration is required; complete oracle coverage may return `None`
- `.metadata["transport_audits"]` → per-policy PASS / FAIL / INCONCLUSIVE / NOT_GRADED / NOT_CHECKED records when using `TransportAuditConfig`; FAIL adds a hard result gate only when the current estimate depends on that calibrator
- `.summary()` → compact text report (per-policy estimate + 95% CI + gate flags, best-policy line)
- `.gates` → `Dict[str, GateResult]` (typed view of `metadata["reliability_gates"]`); `.target_policies`
- `.metadata` keys: `target_policies`, `reliability_gates` (`{policy: {"flagged": bool, ...}}`),
  `boundary_cards`, `normalization`, `oracle_sources`, `bootstrap_ci`, `pairwise_inference`
  (cluster-robust runs: per-pair difference SE/df with pairing basis), `inference` (SE basis,
  selection reason, coupling), `degrees_of_freedom` (per-policy df + `t_critical`;
  Welch–Satterthwaite effective df when the oracle jackknife applies; `df_method:
  "labelled_clusters"` for a policy corrected by representative labels, whose df is
  `n_labelled_clusters − fitted_parameters` capped by that Welch df; cite these if asked how
  a CI was computed), `inference_unavailable_policies` / `inference_unavailable_reasons`
  (policies without an interval: one prompt cluster, or one labeled prompt under weight one)
- `.diagnostics` (DirectDiagnostics): `overall_status` (GOOD/WARNING/CRITICAL), `status_per_policy`,
  `boundary_cards`, `refuse_level_policies`, `calibration_rmse`, `n_oracle_labels`, `.summary()`

## Array API (single-sample primitive)

```python
from cje import calibrated_mean_ci, transport_audit

result = calibrated_mean_ci(
    judge_scores,          # (n,) array
    oracle_labels,         # (n,) array; NaN = unlabeled (or pass oracle_mask=)
    cluster_ids=None,      # cluster by prompt when there are multiple draws per prompt
    covariates=None,       # (n, d) matrix for two-stage calibration
    alpha=0.05,
    n_folds=5,             # calibration needs >=4 independent labeled clusters; folds auto-reduce
    inference="cluster_robust",  # default jackknife path | "bootstrap" | "auto"
    seed=42,
)
# result: estimate, se, ci, n, n_oracle, method, calibrator, diagnostics, .summary()
```

This is the ppi-style bottom layer for ONE sample of judge scores. Multi-policy comparisons
belong in `analyze_dataset` (paired, gate-aware).
Complete oracle coverage uses the direct oracle mean without a calibrator, so the four-cluster
calibration floor does not apply. With one independent cluster, inference remains unavailable.
For refit-bootstrap intervals, set `inference="bootstrap", n_bootstrap=2000` instead.
Supplying `n_bootstrap` without an inference choice selects bootstrap with a compatibility warning.

**Transport audit:** before reusing `result.calibrator` (or `results.calibrator`) on new
data (check it is not `None` first; complete oracle coverage fits no calibrator):

```python
diag = transport_audit(
    probe_scores,
    probe_labels,
    calibrator,
    group_label="policy:gpt-5.6-mini",
    delta_max=0.03,
    cluster_ids=prompt_ids,
    family_size=n_groups,
)
print(diag.summary())
```

- Probe: held out, probability sampled, and at least 20 effective independent clusters; size
  for the desired CI width. Below 20 effective clusters PASS is withheld (INCONCLUSIVE), but a
  CI wholly outside the margin still grades FAIL.
- `TransportDiagnostics`: `status` (PASS/FAIL/INCONCLUSIVE/NOT_GRADED), `delta_hat` (mean
  residual), simultaneous `delta_ci`, `effective_clusters`, `recommended_action`, `.summary()`,
  and `.plot()` (viz extra).
- `delta_max` is predeclared. Units: probe oracle-label units for this low-level audit;
  OUTPUT units (units of `results.estimates`) for `TransportAuditConfig` margins. Omitting it
  gives NOT_GRADED (never PASS/FAIL) and emits a UserWarning prompting you to declare a
  margin. Pass analysis weights for unequal
  sampling probabilities and `family_size` for all groups used in the decision.
- `decile_residuals` and probe-bin occupancy are display-only; never gate on them.

## Planning: "how many labels do I need?"

Use this for a future evaluation, not as a requirement for analyzing existing data.
`fit_variance_model` takes one base-policy `FreshDrawDataset`, not the raw policy-to-records
dictionary accepted by `analyze_dataset`.

Example with pilot judge scores and oracle labels already on a declared [0, 1] scale:

```python
from cje import CostModel, fit_variance_model, plan_evaluation, plan_for_mde
from cje.data.fresh_draws import fresh_draws_from_dict

# pilot_rows contains real base-policy records; missing oracle labels stay missing.
pilots, _ = fresh_draws_from_dict({"base": pilot_rows}, auto_normalize=False)
vm = fit_variance_model(pilots["base"], n_replicates=50, seed=42)
print(vm.summary())
if not vm.fit_ok:
    raise ValueError("Unreliable pilot variance fit; inspect the pilot before planning.")

cost = CostModel(surrogate_cost=0.01, oracle_cost=2.00)
budget_plan = plan_evaluation(
    budget=500.0, variance_model=vm, cost_model=cost,
    m_min=30, power=0.80, alpha=0.05,
)
effect_plan = plan_for_mde(
    target_mde=0.02, variance_model=vm, cost_model=cost,
    m_min=30, power=0.80, alpha=0.05,
)
print(budget_plan.summary())
print(effect_plan.summary())
```

Here `0.02` means two percentage points on the declared [0, 1] oracle scale, not a relative
2% change. If converting other scales for planning, save that mapping and express the target
effect in the same units as the fitted variance model.

**Pilot requirements.** Use probability-sampled prompts and randomly sampled oracle labels
from the population where calibration will be learned. Roughly 200+ independent prompts with
100+ oracle labels is a starting recommendation, not a sufficiency guarantee. Both labeled
and unlabeled observations must support variation in the sample-size/label-count grid.
Insufficient grids raise `ValueError`; poor fits can instead warn and return `fit_ok=False`.
Inspect the sampling design and fit warnings, not only whether the call returned.
The 10–25-label calibration starter loop is a different task.

**Budget scope.** Supply explicit costs in the same units as the budget. The modeled cost is
`n_samples * surrogate_cost + m_oracle * oracle_cost`; it does not automatically total the
extra scores for every candidate, pilot collection, response generation, or held-out transport
probes. State what the budget includes. The default minimum is 30 labels; an infeasible budget
raises rather than authorizing a smaller reliable evaluation.

**Interpretation.** `EvaluationPlan` reports `n_samples`, `m_oracle`, `total_cost`, `mde`,
`power`, and `alpha`; `.to_dict()` preserves these and the planning assumptions.
`.power_to_detect(effect_size)` reports projected normal-theory power.
Pairwise MDE uses independent-policy variance (`sqrt(2) * se_level`) and asymptotic-normal
critical values. Positive shared-prompt covariance makes the independence assumption
conservative, but the planner does not fit the actual paired-difference variance.
After collection, inspect the realized finite-sample interval and pairwise inference.
A target power is not demonstrated power, and a good variance fit does not establish random
label selection or calibration transport.

**No suitable pilot?** `simulate_variance_model` can support explicitly hypothetical
sensitivity scenarios. Its `r2` is isotonic R², not correlation; use
`correlation_to_r2(rho)` when starting from a judge–oracle correlation. Do not present
simulated assumptions as observed pilot evidence.

**Worked example.** [`scripts/planning_example.py`](https://github.com/cimo-labs/cje/blob/main/skills/cje/scripts/planning_example.py) demonstrates
pilot → allocation → held-out comparison and saved audit artifacts using bundled data.
Run it from a repository checkout; its fixed demonstration frame does not establish coverage
or power on a user's population. The example's stated sampling and transport limitations
remain part of the result.

## Audit budget: can the residual audit resolve?

Use `plan_transport_audits` for residual-equivalence audit power, separate from evaluation
precision/MDE planning above. Choose the margin before seeing audit outcomes. The inputs
below are hypothetical residual standard deviations, not standard errors:

```python
from cje import AuditScenario, CostModel, plan_transport_audits

plan = plan_transport_audits(
    {
        "baseline": AuditScenario(0.10, 0.05, available_clusters=130),
        "candidate": AuditScenario(0.20, 0.05, available_clusters=130),
    },
    calibration_labels=40,
    cost_model=CostModel(oracle_cost=2.0),
    budget=1000,
    power=0.80,
    alpha=0.05,
)
print(plan.summary())
```

This plans 63 and 245 independent audit clusters, totaling 348 labels including calibration,
at a cost of 696. The candidate's 130 available clusters are insufficient. `power` targets all
supplied audits passing, under the declared Gaussian independent-cluster residual model and
family adjustment. This is not an observed audit or a `PASS`. Examine plausible bias/variance
scenarios; weighted or unequal-contribution clusters need a justified model beyond this
calculator. Count all underlying ratings with `labels_per_cluster`, and separate new annotation
cost from total acquired-label cost. Save `plan.to_dict()`.
See the [audit budget guide](https://github.com/cimo-labs/cje/blob/main/guides/audit-budget-planning.md)
for assumptions, feasibility limits, and cost scope.

## Correction: use representative target labels

A probe passed only through `TransportAuditConfig` diagnoses the calibration map; it does not
alter estimates. To correct the current target estimate:

1. Attach each probability-sampled `oracle_label` to its exact evaluation response, preserving
   unlabeled rows. A label in a calibration-history file alone is not a target residual.
2. For a fixed external fit, keep `calibration_data_path` and set
   `combine_oracle_sources=False`. The default augmented estimator can then add the estimated
   target mean residual without refitting calibration on the correction labels.
3. Declare `label_design="representative"` only for a justified representative slice. For
   known unequal inclusion probabilities, use `label_design="known_propensity"` and
   `label_propensities`: one vector per policy with a positive probability for **every**
   evaluation row, in row order, including unlabeled rows. For targeted labels with unknown
   probabilities, use `"targeted_unknown"`; CJE keeps an explicit plug-in route.
4. Inspect `results.metadata["point_estimator"]["routes"]` to confirm `augmented` where
   applicable. Use the recomputed SEs, CIs, and paired comparisons, not an old interval shifted
   by the correction. Retain the original audit as evidence about the unchanged map.
5. Do not call reused correction labels independent validation of the corrected estimate.
   Correction targets the sampled population; a future cycle or new fit needs its own evidence.
6. The correction weights the calibrated prediction at one by default (the 0.8.x estimator).
   `estimator_config={"correction_weight": "tuned"}` (or `correction_weight="tuned"` in
   `calibrated_mean_ci`) opts into the PPI++ power-tuned weight for representative designs;
   it falls back to one below 20 labeled prompts, for rare or constant labeled outcomes, and
   for known propensities. Its gain is asymptotic: at 20 to 60 labels its pooled realised error
   was within about half a percent of weight one on held-out benchmarks. Its interval counts
   the slope as a fitted parameter (df `n_labeled − 2`) but omits its delta-method variance;
   in simulations with a separate calibration set it covered within about a quarter of a point
   of weight one at 20 to 30 labels. Read the weight and its reason in
   `metadata["point_estimator"]`.
7. The corrected interval takes its df from the labeled prompts (`n_labeled − 1` with weight
   one), so a few labels give a wide interval and one label gives none (`compare_policies`
   refuses its pairs; `compare_all_policies` raises). The adjustment applies to the default
   analytic path only: bootstrap and `"auto"` percentile intervals still under-cover at about
   10 to 20 labeled prompts.

The [correction guide](https://github.com/cimo-labs/cje/blob/main/guides/audit-correction.md)
and [runnable example](https://github.com/cimo-labs/cje/blob/main/examples/audit_correction.py)
show the complete audit-to-correction workflow. Refit pre-0.8.0 saved two-stage calibrators
before use; retain package version, fit provenance and inputs with the run.

## Production outcomes and ranking

Historical judge/outcome pairs can train calibration through `calibration_data_path`. Logs
can also supply observed responses for the policies and population being evaluated, when
sampling and dependence are accounted for. Neither is counterfactual estimation of outputs
that another policy never generated. Define the outcome, join the exact response, record
judge/rubric versions, and justify selection and transport. Available feedback is not
necessarily representative; do not fabricate inclusion probabilities for organic feedback.

For two policies, the oracle difference equals the calibrated-prediction difference plus
the difference in their mean residuals. A shared offset can cancel, so a failed level audit
alone does not prove the ordering wrong. Individual monotonicity and a scalar support badge
also do not prove ordering survives a shift. Use the paired comparison plus evidence about
residual differences; there is no validated automatic label-free reuse gate in CJE.

## Langfuse

Use [`cje.bridges.langfuse.prepare`](https://github.com/cimo-labs/cje/blob/main/scripts/langfuse_cje/README.md)
for exported experiments before constructing custom joins. The pure converter ships in the
wheel; its GET-only online exporter runs from a source checkout with `httpx`. Declare the
project, dataset population/version, two experiments, score selectors and scales. Preserve
unlabeled responses and verify exact response identity and human-label provenance. A prepared
export is not an estimate, an audit pass, or evidence of representative sampling.

## Audit exports

Save the executable script and retained inputs alongside the exported result. `plan.to_dict()`
preserves the allocation; `results.to_dict()` uses the default portable result format.
Also save the policy intervals and pairwise comparisons at the alpha actually requested,
plus diagnostics and all gate/transport states.

Portable result JSON stores paired comparison results at alpha 0.05 without the large
bootstrap matrix. `results.to_dict(detail="full")` retains bootstrap/influence arrays when
available; `detail="summary"` does not preserve paired-inference capability.
Neither JSON form retains a fitted calibrator for prediction. Refit from retained inputs
when a rerun or calibrator is needed; do not silently substitute independent inference
because exported state is missing.

## CLI

```text
cje validate PATH [-v]                      # check a fresh-draws dir/file; exit 0 = ready
cje analyze PATH [--calibration-data F]     # per-policy estimates + 95% CIs
            [--estimator-config JSON] [-o results.json]
            [--judge-field NAME] [--oracle-field NAME]
            [--transport-probe POLICY=FILE] [--transport-margin POLICY=DELTA]
            [--transport-family-size N] [--transport-alpha A]
            [--transport-min-clusters K] [--transport-bins B]
```

`--transport-probe`/`--transport-margin` repeat per policy; margins are in oracle/output
units (the units of the printed estimates).

`cje analyze` surfaces the highest point estimate together with any diagnostic limitations; it
does not silently substitute a different winner. Run `cje validate` first on user-provided
directories.

## Diagnostics glossary

| Signal | Values | What to tell the user |
|---|---|---|
| `overall_status` | GOOD / WARNING / CRITICAL | CRITICAL: results shipped with explicit caveats only |
| Boundary card | OK / CAUTION / REFUSE-LEVEL | Scalar score-range support only. REFUSE-LEVEL: no absolute level claim from this fit; it does not establish ranking validity |
| `reliability_gates[p]["flagged"]` | bool | Surface the point estimate with the limitation; do not substitute another policy silently |
| Transport `status` | PASS / FAIL / INCONCLUSIVE / NOT_GRADED / NOT_CHECKED | Equivalence verdict for the declared residual margin; interpret only for the audited population and family. NOT_CHECKED = no probe was supplied for that policy; never treat it as a pass |

## Troubleshooting

| Symptom | Meaning / fix |
|---|---|
| `Only N independent oracle-labeled prompt clusters are available (<4 ...); returning the UNCALIBRATED raw-judge tier` warning (`analyze_dataset`, 1–3 labels) | Below the 4-cluster floor no calibrator can be fit; the run returns the flagged `naive_direct` tier. Treat as blocked; run the labeling loop (SKILL.md §Labeling); never invent labels. |
| `ValueError: Too few unique oracle prompt clusters (N) for cross-fitted calibration. Need at least 4 independent prompt clusters ...` (`calibrated_mean_ci`) | Partial coverage requires calibration and at least four independent labeled clusters. Repeated labels within one prompt do not create independent folds. Complete coverage uses the direct oracle mean without calibration. |
| `No oracle labels found` → `method="naive_direct"` | 0 labels: raw judge means with a loud warning. Never report these as the answer; treat as blocked and run the labeling loop. |
| `reducing calibration folds from 5 to K` warning | 4–9 independent labeled clusters: valid but noisier. Recommend ≥10 independent labeled clusters to the user. |
| `ImportError: ... pip install "cje-eval[viz]"` | Plotting needs the viz extra; estimates work without it. |
| Scores on 0–100 / Likert | Pass as-is; auto-normalized, results returned in the original scale. |
| `... outside [0, 1]` error on a calibration file | `calibration_data_path` defaults to [0, 1]. Declare `calibration_judge_scale`/`calibration_oracle_scale`, rescale the file, or pass the data via `fresh_draws_data` (auto-normalizes). |
| `TypeError: ... 'logged_data_path'` / `calibrated-ips` errors | `analyze_dataset` has no `logged_data_path` parameter and no IPS/DR estimators. Logged judge+oracle data works via `calibration_data_path`. For IPS/DR pin `pip install "cje-eval==0.3.*"` (Python ≤3.12). |
| `UserWarning: ... audits without delta_max are NOT_GRADED` | Declare a practical margin (`delta_max=`); no-margin audits can never PASS or FAIL. |
| `results.calibrator is None` | Complete oracle coverage: the estimate is the direct oracle mean; no calibrator was fit. Check for `None` before a transport audit. |
| `ImportError`/`ValueError` naming a replacement (e.g. `BaseCJEEstimator`, `calibrate_from_raw_data`) | Consolidated API: the error message names the current entry point; use it. |
| Python version | CJE requires Python 3.10–3.13. |
