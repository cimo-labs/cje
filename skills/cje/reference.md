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
3. `calibration_data_path="labeled.jsonl"` — a separate judge+oracle file (e.g. historical labeled logs) used as the calibration source. With `combine_oracle_sources=True` (default) any `oracle_label`s in the fresh draws are pooled with it; `metadata["oracle_sources"]` reports provenance and cross-source conflicts.

Field names differ in the user's data? Pass `judge_field="score"`, `oracle_field="human_rating"` instead of renaming.

## `analyze_dataset`

```python
from cje import analyze_dataset

results = analyze_dataset(
    fresh_draws_data=None,        # {policy: [records]}  (or use fresh_draws_dir=...)
    fresh_draws_dir=None,
    calibration_data_path=None,   # separate judge+oracle JSONL
    combine_oracle_sources=True,
    estimator="auto",             # "auto" -> calibrated-direct; also "direct" (no calibration)
    judge_field="judge_score",
    oracle_field="oracle_label",
    calibration_covariates=None,  # e.g. ["domain"] — fields from record metadata
    estimator_config=None,        # e.g. {"n_bootstrap": 4000}
    verbose=False,
)
```

**`EstimationResult`:**

- `.estimates` (np.ndarray, order matches `metadata["target_policies"]`), `.standard_errors`
- `.ci(alpha=0.05)` → list of `(lo, hi)` per policy; `.confidence_interval()` → `(lo_array, hi_array)`
- `.compare_policies(i, j, alpha=0.05)` → dict with difference, SE, CI, p-value — use this for pairwise claims
- `.best_policy()` → PolicyVerdict (name, index, estimate, flagged, all_flagged, runner_up); gate-aware — a flagged argmax is demoted to `runner_up` and the best reliable policy wins
- `.calibrator` → fitted calibrator, reusable in `transport_audit`
- `.metadata` keys: `target_policies`, `reliability_gates` (`{policy: {"flagged": bool, ...}}`),
  `boundary_cards`, `normalization`, `oracle_sources`, `bootstrap_ci`
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
    n_folds=5,             # 10-19 labels -> folds auto-reduce with a warning (noisier)
    inference="auto",      # "bootstrap" (default when calibrated) | "cluster_robust"
    n_bootstrap=2000,
    seed=42,
)
# result: estimate, se, ci, n, n_oracle, method, calibrator, diagnostics, .summary()
```

This is the ppi-style bottom layer for ONE sample of judge scores. Multi-policy comparisons
belong in `analyze_dataset` (paired, gate-aware).

**Transport audit** — before reusing `result.calibrator` (or `results.calibrator`) on new data:

```python
diag = transport_audit(probe_scores, probe_labels, calibrator, group_label="policy:gpt-5.6-mini")
print(diag.summary())
```

- Probe: 40–60 fresh oracle labels from the new setting.
- `TransportDiagnostics`: `status` (PASS/WARN/FAIL), `delta_hat` (mean residual), `delta_ci`
  (t-based, df = n_probe − 1), `recommended_action`, `n_probe`, `.summary()`, `.plot()` (viz extra).
- `decile_residuals` are display-only (4–6 rows per bin at recommended probe sizes) — never gate on them.

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
labels (below that it raises "Grid has insufficient variation") — and takes about a minute.
Always pass explicit costs: budget and costs must be in the same units (dollars in ⇒ dollars
out), and plans floor at `m_min=30` oracle labels regardless of budget.

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
```

`cje analyze` is gate-aware: a flagged point-estimate winner prints `⚠️ ... (UNRELIABLE)` and the
trophy 🏆 goes to the best *reliable* policy. Run `cje validate` first on user-provided directories.

## Diagnostics glossary

| Signal | Values | What to tell the user |
|---|---|---|
| `overall_status` | GOOD / WARNING / CRITICAL | CRITICAL: results shipped with explicit caveats only |
| Boundary card | OK / CAUTION / REFUSE-LEVEL | REFUSE-LEVEL (≥5% of a policy's judge mass outside the oracle calibration range): no absolute numbers for that policy; rankings may stand; collect labels covering the missing score range |
| `reliability_gates[p]["flagged"]` | bool | Flagged winner → report the best reliable policy instead |
| Transport `status` | PASS / WARN / FAIL | PASS: reuse. WARN: rankings only, say why. FAIL: do not reuse; follow `recommended_action` |

## Troubleshooting

| Symptom | Meaning / fix |
|---|---|
| `Too few oracle samples (N). Need at least 10.` | Hard floor (1–9 labels). Run the labeling loop (SKILL.md §Labeling); never invent labels. |
| `No oracle labels found` → `method="naive_direct"` | 0 labels: raw judge means with a loud warning. Never report these as the answer — treat as blocked and run the labeling loop. |
| `reducing calibration folds from 5 to K` warning | 10–19 labels: valid but noisier. Recommend more labels to the user. |
| `ImportError: ... pip install "cje-eval[viz]"` | Plotting needs the viz extra; estimates work without it. |
| Scores on 0–100 / Likert | Pass as-is — auto-normalized, results returned in the original scale. |
| `logged_data_path` / `calibrated-ips` errors | OPE removed in 0.4.0. Logged judge+oracle data still works via `calibration_data_path`. For IPS/DR pin `pip install "cje-eval==0.3.*"` (Python ≤3.12). |
| Python version | 0.4.x supports 3.9–3.13. |
