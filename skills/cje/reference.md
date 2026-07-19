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

1. `fresh_draws_data={policy_name: [records]}` — in-memory, the default choice when you reshaped the user's data yourself.
2. `fresh_draws_dir="responses/"` — one JSONL per policy, named `{policy}_responses.jsonl` (also accepted: `{policy}.jsonl`). Policy name comes from the filename; keep names identical everywhere. A single JSONL file path (records grouped by `target_policy`) also works here.
3. `calibration_data_path="labeled.jsonl"` — a separate judge+oracle file (e.g. historical labeled logs) used as the calibration source. Values default to [0, 1]; for other scales declare `calibration_judge_scale=(lo, hi)` / `calibration_oracle_scale=(lo, hi)` (external calibration data never infers its scale from observations). Out-of-range files raise a hard error naming the observed range. With `combine_oracle_sources=True` (default) any `oracle_label`s in the fresh draws are pooled with it; `metadata["oracle_sources"]` reports provenance and cross-source conflicts.

Field names differ in the user's data? Pass `judge_field="score"`, `oracle_field="human_rating"` instead of renaming.

## `analyze_dataset`

All arguments are keyword-only. Provide exactly ONE evaluation source
(`fresh_draws_data` or `fresh_draws_dir` — passing both raises):

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
    calibration_covariates=None,  # e.g. ["domain"] — fields from record metadata
    include_response_length=False,  # auto word-count covariate (needs "response")
    estimator_config=None,        # e.g. {"n_bootstrap": 4000}
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

**Wiring transport audits into the analysis** — probes are held-out oracle-labeled
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
fallback), never by this parameter — its only observable effect is which name lands in
`metadata["estimator"]` (`"auto"` records `"direct"`).

**`EstimationResult`:**

- `.estimates` (np.ndarray, order matches `metadata["target_policies"]`), `.standard_errors`
- `.ci(alpha=0.05)` → list of `(lo, hi)` per policy; `.confidence_interval()` → `(lo_array, hi_array)`
- `.compare_policies(i, j, alpha=0.05)` → dict with difference, SE, CI, p-value — use this for
  pairwise claims. The `method` key names the inference basis, best-first: `"paired_bootstrap"`
  (bootstrap runs: paired inference over the replicate matrix — the difference SE includes
  calibrator noise, honest on near-tie pairs; sign-test p-value floored at 2/(B+1)),
  `"paired_if_oua"` (cluster-robust runs: t-test from the stored pairwise SE + oracle-jackknife
  difference variance), `"paired_if_legacy"` (pre-0.5.1 IF z-test, only for deserialized older
  results), `"independent_conservative"` (no pairing info). `gate_flagged` lists any policy in
  the pair with a flagged reliability gate — a difference CI cannot repair a biased input
  (e.g. after a transport-audit FAIL), so treat such comparisons per the gates discipline
- `.compare_all_policies(alpha=0.05, adjust=None)` → list of comparison dicts for every (i < j)
  pair with `policy1`/`policy2` names; `adjust="bh"` adds Benjamini-Hochberg
  `p_adjusted`/`significant_adjusted` for many-pair audits
- `.bootstrap_samples` → (B, P) bootstrap replicate matrix on bootstrap runs (columns follow
  `metadata["target_policies"]`); powers the paired comparisons, omitted from JSON export
- `.best_policy()` → PolicyVerdict (name, index, estimate, flagged, all_flagged, runner_up,
  runner_up_reasons); defaults to `reliable_only=True`: a gate-flagged argmax is demoted to the
  best gate-passing policy, loudly (the demoted argmax travels as `runner_up` with its gate
  reasons, a warning is logged, and `summary()` prints both). Pass `reliable_only=False` for the
  raw argmax with `flagged=True`. If everything is flagged, the argmax returns with
  `all_flagged=True` — do not crown it
- `.calibrator` → fitted calibrator when calibration is required; complete oracle coverage may return `None`
- `.metadata["transport_audits"]` → per-policy PASS / FAIL / INCONCLUSIVE / NOT_GRADED / NOT_CHECKED records when using `TransportAuditConfig`; FAIL adds a hard result gate only when the current estimate depends on that calibrator
- `.summary()` → compact text report (per-policy estimate + 95% CI + gate flags, best-policy line)
- `.gates` → `Dict[str, GateResult]` (typed view of `metadata["reliability_gates"]`); `.target_policies`
- `.metadata` keys: `target_policies`, `reliability_gates` (`{policy: {"flagged": bool, ...}}`),
  `boundary_cards`, `normalization`, `oracle_sources`, `bootstrap_ci`, `pairwise_inference`
  (cluster-robust runs: per-pair difference SE/df with pairing basis)
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
    n_folds=5,             # 4-9 labels -> folds auto-reduce with a warning (noisier); <4 raises
    inference="auto",      # "bootstrap" (default when calibrated) | "cluster_robust"
    n_bootstrap=2000,
    seed=42,
)
# result: estimate, se, ci, n, n_oracle, method, calibrator, diagnostics, .summary()
```

This is the ppi-style bottom layer for ONE sample of judge scores. Multi-policy comparisons
belong in `analyze_dataset` (paired, gate-aware).

**Transport audit** — before reusing `result.calibrator` (or `results.calibrator`) on new
data (check it is not `None` first — complete oracle coverage fits no calibrator):

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
  gives NOT_GRADED (never PASS/FAIL) and emits a FutureWarning for one release cycle — 0.5.x
  graded the same call PASS/WARN/FAIL under a zero-null test. Pass analysis weights for unequal
  sampling probabilities and `family_size` for all groups used in the decision.
- `decile_residuals` and probe-bin occupancy are display-only; never gate on them.

## Planning: "how many labels do I need?"

With pilot data (a `FreshDrawDataset` or the draws from a first run):

```python
from cje import CostModel, fit_variance_model, plan_evaluation, plan_for_mde

vm = fit_variance_model(fresh_draws)                     # decomposes eval vs calibration variance
cost = CostModel(surrogate_cost=0.01, oracle_cost=2.00)  # $/judge-score, $/oracle-label
plan = plan_evaluation(budget=500.0, variance_model=vm, cost_model=cost)  # or plan_for_mde(0.02, ...)
print(plan.summary())   # n_samples, m_oracle, MDE at 80% power
```

`fit_variance_model` needs a real pilot — roughly 200+ prompts with 100+ randomly sampled oracle
labels (below that it raises "Grid has insufficient variation") — and runs in seconds since 0.5.1
(the default `n_replicates=50` gives a stable fit in a few seconds; `n_replicates=150` for an extra-stable fit, ~20s).
Always pass explicit costs: budget and costs must be in the same units (dollars in ⇒ dollars
out), and plans floor at `m_min=30` oracle labels regardless of budget. Since 0.5.1 the variance
components come from the analytic cluster-robust + OUA instrument (within ~5% of realized SE on
the validation grid); budgets planned with pre-0.5.1 versions were ~15-20% inflated at pilot scale.

No pilot data yet? `from cje import simulate_variance_model, correlation_to_r2` — the `r2`
parameter is the **isotonic R², not the correlation**: call
`simulate_variance_model(r2=correlation_to_r2(rho))` if what you have is a judge–oracle
correlation ρ. Treat simulated plans as rough — prefer the pilot path once any real labels exist.

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
| Transport `status` | PASS / FAIL / INCONCLUSIVE / NOT_GRADED / NOT_CHECKED | Equivalence verdict for the declared residual margin; interpret only for the audited population and family. NOT_CHECKED = no probe was supplied for that policy — never treat it as a pass |

## Troubleshooting

| Symptom | Meaning / fix |
|---|---|
| `Too few oracle samples (N) for 5-fold CV. Need at least 10 (2 per fold).` | Hard floor (1–3 labels). Run the labeling loop (SKILL.md §Labeling); never invent labels. |
| `No oracle labels found` → `method="naive_direct"` | 0 labels: raw judge means with a loud warning. Never report these as the answer — treat as blocked and run the labeling loop. |
| `reducing calibration folds from 5 to K` warning | 4–9 labels: valid but noisier. Recommend ≥10 labels to the user. |
| `ImportError: ... pip install "cje-eval[viz]"` | Plotting needs the viz extra; estimates work without it. |
| Scores on 0–100 / Likert | Pass as-is — auto-normalized, results returned in the original scale. |
| `... outside [0, 1]` error on a calibration file | `calibration_data_path` defaults to [0, 1]. Declare `calibration_judge_scale`/`calibration_oracle_scale`, rescale the file, or pass the data via `fresh_draws_data` (auto-normalizes). |
| `TypeError: ... 'logged_data_path'` / `calibrated-ips` errors | OPE removed in 0.4.0; the dead `logged_data_path` kwarg was removed entirely in 0.6.0 (plain TypeError). Logged judge+oracle data still works via `calibration_data_path`. For IPS/DR pin `pip install "cje-eval==0.3.*"` (Python ≤3.12). |
| `FutureWarning: ... audits without delta_max are NOT_GRADED` | Declare a practical margin (`delta_max=`) — no-margin audits can never PASS or FAIL since 0.6.0. |
| `results.calibrator is None` | Complete oracle coverage: the estimate is the direct oracle mean; no calibrator was fit. Check for `None` before a transport audit (see MIGRATING-0.6.md §6). |
| Python version | 0.6.0 requires 3.10–3.13 (3.9 was supported through 0.5.x). |

## Migrating from 0.5.x (0.6.0)

0.6.0 is a breaking correctness release — full guide with before/after snippets in
[`MIGRATING-0.6.md`](https://github.com/cimo-labs/cje/blob/main/MIGRATING-0.6.md) at the
repo root. Headlines:

- **Python 3.10+** (3.9 dropped).
- **`analyze_dataset` is keyword-only**; `logged_data_path` now raises a plain TypeError;
  provide exactly one of `fresh_draws_data`/`fresh_draws_dir`.
- **Transport audits regraded**: `delta_max` required to grade; statuses are
  PASS/FAIL/INCONCLUSIVE/NOT_GRADED/NOT_CHECKED (WARN removed); no-margin calls warn for
  one release cycle. Update any `status == "PASS"` / `status != "FAIL"` gate code.
- **Full oracle coverage** routes to the direct oracle mean and `results.calibrator is None`.
- **`best_policy()` default unchanged** (`reliable_only=True`), demotion is now loud
  (`runner_up_reasons`, logged warning, both winners in `summary()`).
- **Ingestion is loud by default** (`on_invalid="error"`); `prompt_id` auto-generation
  from a `prompt` hash is retained.
- **Expect numeric drift vs 0.5.x**: new balanced-cluster calibration folds, exponential-
  weight bootstrap, full-coverage routing. Same data, defensibly different numbers.
- `IPSDiagnostics` removed — use `DirectDiagnostics`.

## Migrating from 0.4.x

0.5.0 is a consolidation release: same statistics on the default paths, smaller API.

- **`results.best_policy()` returns a `PolicyVerdict`** (was a naive argmax `int`).
  A gate-flagged argmax is demoted and the best gate-passing policy wins (recorded in
  `runner_up`). Use `verdict.index` for the old integer and `verdict.name` for the name.
- **`estimator_config` unknown keys now raise** a ValueError listing the valid keys
  (`oua_jackknife`, `inference_method`, `n_bootstrap`, `bootstrap_seed`,
  `use_augmented_estimator`, `paired_comparison`). Typos no longer pass silently.
- **Calibration files must be in [0, 1]** — out-of-range `calibration_data_path`
  values raise a hard error instead of being silently filtered. Rescale, or pass the
  data in-memory via `fresh_draws_data` (auto-normalizes any bounded scale).
- **New typed accessors** on `EstimationResult`: `.summary()` (text report),
  `.best_policy()` → PolicyVerdict, `.gates` → `Dict[str, GateResult]`,
  `.target_policies`; metadata mirrors are unchanged and stay the serialized form.
- **Removed names** (each raises an ImportError/ValueError pointing at the
  replacement): `BaseCJEEstimator` (merged into `CalibratedDirectEstimator`);
  `calibrate_from_raw_data`/`calibrate_judge_scores` (use `calibrate_dataset`,
  `JudgeCalibrator.fit_cv`, or `calibrated_mean_ci`); `JudgeCalibrator.fit_transform`
  (use `fit_cv`); `compare_policies_bootstrap` (use `results.compare_policies`);
  `plot_calibration_comparison`; `EstimationResult.plan_allocation` (use
  `fit_variance_model` + `plan_evaluation`); `simulate_planning_sweep` (loop
  `simulate_planning`); `export_results_csv` (JSON export stays);
  `AnalysisService`/`AnalysisConfig`/`create_estimator` (call `analyze_dataset`);
  `calibrate_dataset(enable_cross_fit=False)` (cross-fitting is the only mode).
