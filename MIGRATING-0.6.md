# Migrating to CJE 0.6

CJE 0.6.0 revised several API contracts around transport auditing, result
routing, and ingestion. If you were using a 0.5.x release, this page lists
what changed; everything else behaves as the current docs describe. Full
detail and rationale live in the [CHANGELOG](CHANGELOG.md).

## Transport audit regrade

0.5.x graded `audit_transportability` / `transport_audit` with a zero-null
significance test. 0.6.0 grades a predeclared practical-equivalence margin
(`delta_max`) with a prompt-clustered, Bonferroni-adjusted CI:

| 0.5.x status | 0.6.0 without `delta_max` | 0.6.0 with `delta_max` |
|---|---|---|
| `PASS` (0 ∈ CI) | `NOT_GRADED` + `UserWarning` | `PASS` only if the CI is wholly inside `[-delta_max, +delta_max]` (and ≥ 20 effective clusters) |
| `WARN` (0 ∉ CI, small δ̂) | `NOT_GRADED` + `UserWarning` | usually `INCONCLUSIVE`; `WARN` no longer exists (legacy serialized values normalize to `INCONCLUSIVE`) |
| `FAIL` (0 ∉ CI, δ̂ ≥ 0.05) | `NOT_GRADED` + `UserWarning` | `FAIL` only if the CI is wholly outside the margin (graded even below the cluster floor) |

Without `delta_max`, `status == "PASS"` never fires and `status != "FAIL"`
always passes — silently, the dangerous direction for a drift monitor.
Declare the margin in every monitor. Units: probe `oracle_label` units for
the low-level audit; OUTPUT units (the units of `results.estimates`) for
margins passed via `TransportAuditConfig`.

## Other breaking changes

- **Python 3.10+** — 3.9 is dropped; 3.10–3.13 are supported and tested.
- **`analyze_dataset` is keyword-only** (snippet below), and exactly one
  evaluation source (`fresh_draws_dir` or `fresh_draws_data`) must be
  provided — passing both raises.
- **`logged_data_path` and `IPSDiagnostics` are removed.** Passing
  `logged_data_path` raises a plain `TypeError`; logged judge+oracle rows
  work via `calibration_data_path`. Import `DirectDiagnostics` instead of
  `IPSDiagnostics`. IPS/DR workflows stay on the frozen 0.3.x line:
  `pip install "cje-eval==0.3.*"` (Python ≤ 3.12).
- **`best_policy()` returns the best reliable policy** (default
  `reliable_only=True`, unchanged) and the demotion is now loud: the
  demoted raw argmax travels as `runner_up` with `runner_up_reasons`, a
  warning is logged, and `summary()` prints both winners.
  `reliable_only=False` returns the raw argmax marked `flagged`.
- **Ingestion errors by default**: invalid records raise with file/line
  context on every path; pass `on_invalid="drop"` to filter with counted
  logging. Explicit duplicate `(prompt_id, draw_idx)` rows always raise.
- **`calibrated_mean_ci` returns `calibrator=None` at complete oracle
  coverage** — the estimate is the direct oracle mean and no calibrator is
  fit. Check for `None` before reusing `result.calibrator`; the reuse
  pattern is documented in
  [`cje/interface/README.md`](cje/interface/README.md#transport-audits).
- **Expect numeric drift vs 0.5.x** at identical seeds: balanced-cluster
  calibration folds, an exponential-weight (Bayesian) bootstrap, and
  direct-oracle routing at full coverage. Same data, defensibly different
  numbers — not a regression.

## Keyword-only `analyze_dataset`

```python
from cje import analyze_dataset

# 0.5.x — positional path tolerated in the signature
results = analyze_dataset("responses/")                  # 0.6.0: TypeError

# 0.6.0 — keyword-only
results = analyze_dataset(fresh_draws_dir="responses/")
```
