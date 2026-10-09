# CJE reference (for agents)

Load this file when you need full signatures, the planning API, the CLI, or troubleshooting.
The workflow and hard rules live in `SKILL.md`; this file is detail only.

## Data formats

**Record schema** (one dict per response, any policy):

| Field | Required | Notes |
|---|---|---|
| `judge_score` | yes | Any bounded scale (0–1, 0–100, Likert 1–5). Auto-normalized; results return in the original scale (`metadata["normalization"]`). |
| `oracle_label` | no | Ground-truth on the labeled slice. `None` (JSON `null`) or missing = unlabeled. Convert pandas blanks first (`df.astype(object).where(df.notna(), None)`): through 0.9.1 records reject `NaN`; from 0.9.2 a float `NaN` is read as unlabeled (one INFO log) and other invalid labels still raise. The array APIs always use `NaN` for unlabeled. Same scale conventions. |
| `prompt_id` | no | Enables paired within-prompt comparisons across policies (lower-variance). Auto-generated from a hash of `prompt` if absent. |
| `row_id` | no | Unique response ID within a policy/source (e.g. the run ID; not the `prompt_id`). Exact repeats are deduplicated with a warning; repeats with differing fields raise. Without it, re-exported duplicates count as separate rows. |
| `observation_id` | no | Stable ID of the judged response. Set it on evaluation rows, calibration rows and transport probes: `analyze_dataset` then raises on a probe that was also used to fit the calibrator and reports cross-source label conflicts. Does not change estimates. |
| `response` | no | Only needed for `include_response_length=True`. |
| `metadata` | no | Dict; fields here are usable as `calibration_covariates`. |

Logprob fields from 0.3.x logged data are accepted and ignored.

**Three ways to supply data to `analyze_dataset`:**

1. `fresh_draws_data={policy_name: [records]}`: in-memory, the default choice when you reshaped the user's data yourself.
2. `fresh_draws_dir="responses/"`: one JSONL per policy, named `{policy}_responses.jsonl` (also accepted: `{policy}.jsonl`). Policy name comes from the filename; keep names identical everywhere. A single JSONL file path (records grouped by `target_policy`) also works here.
3. `calibration_data_path="labeled.jsonl"`: a separate judge+oracle file (e.g. historical labeled logs) used as the calibration source. Values default to [0, 1]; for other scales declare `calibration_judge_scale=(lo, hi)` / `calibration_oracle_scale=(lo, hi)` (external calibration data never infers its scale from observations). Out-of-range files raise a hard error naming the observed range. With `combine_oracle_sources=True` (default) any `oracle_label`s in the fresh draws are pooled with it; `metadata["oracle_sources"]` reports provenance and cross-source conflicts.

Field names differ in the user's data? Pass `judge_field="score"`, `oracle_field="human_rating"` instead of renaming.

## Your own data

Mark an unlabeled row with `oracle_label: None` or leave the field out. From a CSV:

```python
import csv
from collections import defaultdict
from cje import analyze_dataset

draws = defaultdict(list)
with open("evals.csv") as f:  # prompt_id, variant, judge_score, human_rating (blank = unlabeled)
    for row in csv.DictReader(f):
        draws[row["variant"]].append({
            "prompt_id": row["prompt_id"],
            "judge_score": float(row["judge_score"]),
            "oracle_label": float(row["human_rating"]) if (row["human_rating"] or "").strip() else None,
        })
results = analyze_dataset(fresh_draws_data=dict(draws))
```

With pandas, convert blanks before building records: `df = df.astype(object).where(df.notna(), None)`.
Through 0.9.1 records reject `NaN`; from 0.9.2 a `NaN` label is read as unlabeled.

- **Where to put labels.** Comparing a few variants whose responses you can all rate? Label a
  random sample of each (start with 20 or more per variant, ideally on the same randomly chosen
  prompts). Each estimate is then corrected by its own labels
  (`metadata["point_estimator"]["routes"]` shows `augmented`), so a judge that favors one
  variant's style does not carry into the difference; the intervals are wider than with every
  label on one policy. Put all labels on one policy only when labeling the others is
  impractical; a policy without labels of its own is then not decision-ready until its
  transport audit PASSes (SKILL.md hard rule 5).
- **Unequal sampling.** The default `label_design="representative"` treats each policy's labeled
  rows as a simple random sample of that policy. If sampling rates differ (for example, you
  oversampled some score ranges), pass `label_design="known_propensity",
  label_propensities={policy: probs, ...}`: for every policy in the call, an inclusion
  probability in (0, 1] for every row in input order, labeled or not. Otherwise the estimate and
  its interval can be biased, with no warning. Never hand-pick responses to label; CJE cannot
  detect it.

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
    on_invalid=None,              # default "error" (loud); "drop" filters with counted logging.
                                  # Through 0.9.1 a NaN oracle_label is invalid, so "drop"
                                  # deletes those unlabeled rows: convert NaN to None instead
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

- `.estimates` (np.ndarray, order matches `metadata["target_policies"]`, which is sorted by policy name, not input dict order), `.standard_errors`
- `.ci(alpha=0.05)` → list of `(lo, hi)` per policy; `.confidence_interval()` → `(lo_array, hi_array)`
- `.compare_policies(i, j, alpha=0.05)` → dict with difference, SE, CI, p-value; use this for
  pairwise claims. `i`/`j` are integer indices into `metadata["target_policies"]`, which is
  sorted by name, not your dict's order; `difference` = estimate[i] − estimate[j]. From 0.9.2
  `i`/`j` may also be policy names and the dict carries `policy1`/`policy2`; through 0.9.1
  names raise `TypeError` and the dict carries no names, so report pairs from
  `compare_all_policies()`. From 0.9.2 the dict also carries `transport_unverified` (the pair's
  policies whose estimate relies on an unverified calibration transfer: no labels of their own,
  or a plug-in route that does not correct with them, and no `PASS` audit) and
  `conditional_on_transport` (true when that list is non-empty: the CI and p-value then assume
  the calibration transfers; None for results that do not record label provenance, such as
  results saved before 0.9.2 or built by calling an estimator directly). `significant` keeps its meaning, `p_value < alpha`. The `method` key names the inference basis, best-first: `"paired_bootstrap"`
  (bootstrap runs: paired inference over the replicate matrix; the difference SE includes
  calibrator noise, honest on near-tie pairs; sign-test p-value floored at 2/(B+1)),
  `"paired_if_oua"` (cluster-robust runs: t-test from the stored pairwise SE + oracle-jackknife
  difference variance), `"paired_if_legacy"` (only for deserialized results from older releases
  that stored unpaired IF z-tests), `"independent_conservative"` (no pairing info). `gate_flagged` lists any policy in
  the pair with a flagged reliability gate; a difference CI cannot repair a biased input
  (e.g. after a transport-audit FAIL), so treat such comparisons per the gates discipline
- `.compare_all_policies(alpha=0.05, adjust=None)` → list of comparison dicts for every (i < j)
  pair with `policy1`/`policy2` names (`difference` = `policy1` − `policy2`); `adjust="bh"` adds
  Benjamini-Hochberg `p_adjusted`/`significant_adjusted` for many-pair audits, while `p_value`,
  `significant` and the CIs stay unadjusted
- `.bootstrap_samples` → (B, P) bootstrap replicate matrix on bootstrap runs (columns follow
  `metadata["target_policies"]`); powers paired comparisons, omitted from default portable
  JSON export (`to_dict(detail="full")` retains it)
- `.best_policy()` → PolicyVerdict (name, index, estimate, flagged, all_flagged, runner_up,
  runner_up_reasons); defaults to `reliable_only=True`: a gate-flagged argmax is demoted to the
  best gate-passing policy, loudly (the demoted argmax travels as `runner_up` with its gate
  reasons, a warning is logged, and `summary()` prints both). Pass `reliable_only=False` for the
  raw argmax with `flagged=True`. If everything is flagged, the argmax returns with
  `all_flagged=True`; do not crown it. A demotion is not a comparison: no test is applied (with
  two policies it simply returns the unflagged one). When labels come from one baseline policy,
  a better candidate whose judge scores exceed the labeled range is exactly what triggers the
  flag; clear it, then use `compare_all_policies()`. From 0.9.2 the verdict also carries
  `decision_ready` and `decision_note` (why, and what to do). `decision_ready` is True only when
  the winner beats every other usable policy in the paired comparison (each 95% CI above 0), no
  policy in the comparison is transport-unverified or gate-flagged, the winner was not reached by
  demoting a flagged leader, and each difference clears any PASS audit's `delta_max` it relies
  on. The per-pair CIs are unadjusted on purpose: requiring every pair is an intersection-union
  test, so `decision_ready=True` can sit next to a BH `p_adjusted` above 0.05. Results saved
  before 0.9.2, or built by calling an estimator directly, are never decision-ready
- `.calibrator` → fitted calibrator when calibration is required; complete oracle coverage may return `None`
- `.metadata["transport_audits"]` → per-policy PASS / FAIL / INCONCLUSIVE / NOT_GRADED / NOT_CHECKED records when using `TransportAuditConfig`; FAIL adds a hard result gate only when the current estimate depends on that calibrator
- `.summary()` → compact text report (per-policy estimate + 95% CI + gate flags, best-policy line).
  From 0.9.2, with two or more policies, it adds each pair as `a - b: +0.038  95% CI [-0.027,
  +0.102]  p=0.22` (unadjusted; a BH note with three or more policies), "No reliable winner:
  every paired CI includes 0" when true, labels the best-policy line "(point estimate, not a
  test)", and prints one named line per transport-unverified policy (status at least WARNING),
  marking it `[borrowed calibration]` (no own labels) or `[uncorrected calibration]` (a plug-in
  route that does not correct with its own labels). Pair markers: `[gate-flagged: <policy>]`;
  `[borrowed calibration: <policy>]` (no own labels), `[uncorrected calibration: <policy>]`
  (plug-in route) or `[unverified calibration: <policies>]` (a mix of the two); and
  `[within transport margin <m>]` when the CI excludes 0 only inside the summed `delta_max` of
  the PASS audits the difference relies on. "No decision-ready winner" is printed when every
  printed pair whose CI excludes 0 involves a transport-unverified policy or lies within such a
  margin. With more than 10 pairs only those with the best point estimate are printed
  (`compare_all_policies()` lists all). `cje analyze` prints the same lines
- `.gates` → `Dict[str, GateResult]` (typed view of `metadata["reliability_gates"]`); `.target_policies`
- `.metadata` keys: `target_policies`, `reliability_gates` (`{policy: {"flagged": bool, ...}}`),
  `boundary_cards`, `normalization`, `oracle_sources`, `bootstrap_ci`, `pairwise_inference`
  (cluster-robust runs: per-pair difference SE/df with pairing basis), `inference` (SE basis,
  selection reason, coupling), `degrees_of_freedom` (per-policy df + `t_critical`;
  Welch–Satterthwaite effective df when the oracle jackknife applies; `df_method:
  "labelled_clusters"` for a policy corrected by representative labels, whose df is
  `n_labelled_clusters − fitted_parameters` capped by that Welch df and by the unadjusted
  interval's Welch df, so it never narrows; cite these if asked how
  a CI was computed), `inference_unavailable_policies` / `inference_unavailable_reasons`
  (policies without an interval: one prompt cluster, or one labeled prompt under weight one),
  `calibration_info` when a calibrator was fitted (`selected_mode`,
  `covariates`, `covariates_used`, `n_folds`, `n_folds_without_covariates`). From 0.9.2:
  `own_oracle_labels_by_policy` (`{policy: count of that policy's own non-missing oracle labels
  used}`), `calibration_label_sources` (what the calibration was fit on: policy names, plus
  `"calibration_data"` for `calibration_data_path`; empty when no calibrator was fit),
  `transport_unverified` (sorted policies with no own labels, or on a plug-in route that does
  not correct with them, whose transport audit is not `PASS`; empty when no calibrator was
  fit), and `cje_version`
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
belong in `analyze_dataset` (paired, gate-aware). When partial coverage requires calibration,
`result.calibrator` predicts in the judge and oracle units you supplied, and
`result.diagnostics["boundary_card"]` carries the scalar score-support badge.
Complete oracle coverage uses the direct oracle mean without a calibrator, so the four-cluster
calibration floor does not apply. With one independent cluster, inference remains unavailable.
For refit-bootstrap intervals, set `inference="bootstrap", n_bootstrap=2000` instead.
Supplying `n_bootstrap` without an inference choice selects bootstrap with a compatibility warning.

- **Labels must be finite and in [0, 1]** (here and in `JudgeCalibrator.fit_cv`); anything else
  raises `ValueError`. Rescale a bounded scale with `(y - lo) / (hi - lo)` and map results back
  (estimate and CI with `lo + (hi - lo) * value`, SE times `hi - lo`), or use `analyze_dataset`
  with a declared oracle scale.
- **Covariates need at least 20 labelled rows.** Below that the calibrator falls back to
  judge-score-only monotone calibration (`diagnostics["calibration"]["selected_mode"] ==
  "monotone"`); a cross-fitting fold whose training complement has fewer than 20 labelled rows
  ignores them too (about 25 labels with 5 folds avoids both). Both emit a `UserWarning`; read
  `diagnostics["calibration"]["covariates_used"]` and `["n_folds_without_covariates"]`. With
  every row labelled no calibrator is fitted and covariates are ignored, with a warning.
- **Representative labels are assumed.** The calibrator is fitted on the labelled rows and the
  residual correction averages them unweighted, so stratified or oversampled labels (equal
  quotas per bucket, rare buckets oversampled) bias the estimate; there is no strata or weights
  argument. For strata defined before labelling, call `calibrated_mean_ci` once per stratum h
  and combine with population shares `W_h = N_h / N`: estimate `sum_h W_h * mu_h`, standard
  error `sqrt(sum_h W_h**2 * SE_h**2)`, normal interval. Each stratum then needs its own labelled
  slice (at least four labelled clusters). For known unequal inclusion probabilities use
  `analyze_dataset(label_design="known_propensity", label_propensities=...)`.

**Transport audit:** before reusing `result.calibrator` (or `results.calibrator`) on new
data (check it is not `None` first; complete oracle coverage fits no calibrator):

```python
diag = transport_audit(
    probe_scores,
    probe_labels,
    calibrator,
    group_label="policy:candidate",
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
2% change. Whether a judge saves labels at all depends on how well the calibrated prediction
tracks the label within each policy; measure that on the pilot first
([label-savings guide](https://github.com/cimo-labs/cje/blob/main/guides/evaluation-planning.md)). If converting other scales for planning, save that mapping and express the target
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
   the slope as a fitted parameter (df at most `n_labeled − 2`) but omits its delta-method variance;
   in simulations with a separate calibration set it covered within about a quarter of a point
   of weight one at 20 to 30 labels. Read the weight and its reason in
   `metadata["point_estimator"]`.
7. The corrected interval takes its df from the labeled prompts (at most `n_labeled − 1` with
   weight one; lower when the oracle-jackknife term is large, as `oracle_df_cap_applied`
   reports), so a few labels give a wide interval and one label gives none (`compare_policies`
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
judge/rubric versions, and justify selection and transport. Document how feedback was
selected, the target population, dependence clusters, and any change in the judge or outcome
process. Available feedback is not necessarily representative; do not fabricate inclusion
probabilities for organic feedback. Production outcomes can reduce **new annotation cost**
when their meaning and response identity match the evaluation; report incremental annotation
cost separately from total label acquisition cost, and audit transport before relying on reuse
across populations or time.

For two policies, the oracle difference equals the calibrated-prediction difference plus
the difference in their mean residuals. A shared offset can cancel, so a failed level audit
alone does not prove the ordering wrong. Individual monotonicity and a scalar support badge
also do not prove ordering survives a shift. Use the paired comparison plus evidence about
residual differences; there is no validated automatic label-free reuse gate in CJE.
(A level claim concerns a policy's mean oracle outcome; a ranking concerns the oracle
difference between two policies.)

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
cje --version                               # installed cje-eval version (0.9.2+)
cje skill [--reference | --path]            # print the bundled SKILL.md / reference.md / their folder (0.9.2+)
cje validate PATH [-v]                      # check a fresh-draws dir/file; exit 0 = ready
cje analyze PATH [--calibration-data F]     # estimates, 95% CIs and the summary() caveat and paired lines
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
directories. It checks fields, finite judge scores, declared scales and label counts; it does not
flag policy-name variants, duplicate rows, prompt overlap across policies, or whether labels were
randomly sampled, so exit 0 does not mean the design is sound.

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
| `ValueError: oracle_labels must lie in [0, 1] ...` (`calibrated_mean_ci`, `JudgeCalibrator.fit_cv`) | Array-level labels are not rescaled for you (they used to be clipped silently in `fit_cv`). Rescale with `(y - lo) / (hi - lo)` and map results back, or use `analyze_dataset` with a declared scale. |
| `UserWarning: Covariates were supplied but ...` | Fewer than 20 labelled rows for the full model or for some folds' training complements; those fits ignore the covariates. Label about 25+ rows (5 folds) or drop the covariates; see `covariates_used` / `n_folds_without_covariates`. |
| `... outside [0, 1]` error on a calibration file | `calibration_data_path` defaults to [0, 1]. Declare `calibration_judge_scale`/`calibration_oracle_scale`, rescale the file, or pass the data via `fresh_draws_data` (auto-normalizes). |
| `Oracle field 'oracle_label' must be finite` (0.9.1 and earlier) | A `NaN` label, usually a pandas blank. Convert with `df.astype(object).where(df.notna(), None)`; never `on_invalid="drop"`, which deletes those unlabeled rows. 0.9.2 reads a `NaN` label as unlabeled. |
| WARNING naming policies with no oracle labels of their own (0.9.2+) | They borrow the named labeled policy's calibration: not decision-ready until their own random labels or a transport `PASS` (SKILL.md hard rule 5). |
| Warnings about colliding policy names, repeated `(policy, prompt_id)` rows, or labels on only one policy (0.9.2+) | Informational; nothing is changed. Merge name variants only after the user confirms, add a unique `row_id` to deduplicate re-exports, and see "Where to put labels" above. |
| `TypeError: ... 'logged_data_path'` / `calibrated-ips` errors | `analyze_dataset` has no `logged_data_path` parameter and no IPS/DR estimators. Logged judge+oracle data works via `calibration_data_path`. For IPS/DR pin `pip install "cje-eval==0.3.*"` (Python ≤3.12). |
| `UserWarning: ... audits without delta_max are NOT_GRADED` | Declare a practical margin (`delta_max=`); no-margin audits can never PASS or FAIL. |
| `results.calibrator is None` | Complete oracle coverage: the estimate is the direct oracle mean; no calibrator was fit. Check for `None` before a transport audit. |
| `ImportError`/`ValueError` naming a replacement (e.g. `BaseCJEEstimator`, `calibrate_from_raw_data`) | Consolidated API: the error message names the current entry point; use it. |
| Python version / `cje.__version__` starts with 0.5 | CJE requires Python 3.10–3.13. On Python 3.9 a bare `pip install cje-eval` silently installs the legacy 0.5 line; install `"cje-eval>=0.9.2"` on 3.10–3.13 so pip fails loudly instead. |
