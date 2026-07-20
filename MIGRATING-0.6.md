# Migrating to CJE 0.6.0 (from 0.5.x)

0.6.0 is a correctness and claim-calibration release. It makes scale handling
explicit, separates scalar score support from residual transport, and replaces
zero-null transport testing with a predeclared practical-equivalence contract.
This guide covers every change a 0.5.x user can hit, with before/after
snippets. The full list is in the [CHANGELOG](CHANGELOG.md).

Quick checklist:

- [ ] Python ≥ 3.10 (3.9 is dropped)
- [ ] All `analyze_dataset(...)` calls are keyword-only (no positional args)
- [ ] No `logged_data_path=` anywhere (use `calibration_data_path=`)
- [ ] No `IPSDiagnostics` imports (use `DirectDiagnostics`)
- [ ] Transport monitors pass `delta_max=` and handle the new status vocabulary
- [ ] Code that reuses `result.calibrator` checks for `None` first
- [ ] You have re-read what `best_policy()` returns (default unchanged, now louder)
- [ ] You expect small numeric drift vs 0.5.x (folds, bootstrap scheme, full-coverage routing)

---

## 1) Python 3.10+ required

0.6.0 supports Python 3.10–3.13 (`python = ">=3.10,<3.14"`). Python 3.9 is
dropped — the locked scientific stack (numpy 2.2 / scipy 1.15) has no 3.9
support, and CI now actually runs every declared interpreter (3.10–3.13). If
you are stuck on 3.9, stay on 0.5.x.

## 2) `analyze_dataset` is keyword-only

```python
# 0.5.x — positional first argument was already an error path, but tolerated in signature
results = analyze_dataset(fresh_draws_dir="responses/")   # keyword calls: unchanged

# 0.6.0 — ONLY keyword arguments are accepted
results = analyze_dataset(fresh_draws_dir="responses/")   # fine
results = analyze_dataset("responses/")                   # TypeError
```

Every documented 0.5.x call pattern was already keyword-based, so working code
is unlikely to break. Note one call-shape tightening: provide exactly **one**
evaluation source — passing both `fresh_draws_dir` and `fresh_draws_data`
now raises (`0.5.x` silently preferred `fresh_draws_data`).

## 3) `logged_data_path` is fully removed from the API

0.4–0.5 kept a dead `logged_data_path` parameter solely to raise a curated
migration message. 0.6.0 removes the parameter, so the API now raises a plain
`TypeError: unexpected keyword argument 'logged_data_path'`:

```python
# 0.5.x
analyze_dataset(logged_data_path="logged.jsonl")   # ValueError with migration recipe

# 0.6.0
analyze_dataset(logged_data_path="logged.jsonl")   # TypeError

# The migration itself is unchanged:
analyze_dataset(
    fresh_draws_dir="responses/",            # evaluation data (fresh draws)
    calibration_data_path="logged.jsonl",    # old logged judge+oracle rows still work here
)
```

The CLI and the loaders still detect logged-data-shaped files (logprob fields,
no `target_policy`) and print the full guidance. IPS/DR workflows remain on
the frozen 0.3.x line: `pip install "cje-eval==0.3.*"` (Python ≤ 3.12).

## 4) `IPSDiagnostics` is removed

The deprecated alias retained through 0.5.x is gone:

```python
# 0.5.x
from cje.diagnostics import IPSDiagnostics   # deprecated alias of DirectDiagnostics

# 0.6.0
from cje.diagnostics import IPSDiagnostics   # ImportError naming the replacement
from cje.diagnostics import DirectDiagnostics
```

## 5) Transport audits are regraded — update every monitor

0.5.x graded `audit_transportability` / `transport_audit` with a zero-null
significance test and a hard-coded 0.05 cutoff. 0.6.0 grades a predeclared
**practical-equivalence margin** (`delta_max`) with a prompt-clustered,
Bonferroni-adjusted CI. What each 0.5.x verdict becomes:

| 0.5.x (zero-null, no margin needed) | 0.6.0 without `delta_max` | 0.6.0 with `delta_max` |
|---|---|---|
| `PASS` — 0 ∈ CI | `NOT_GRADED` (+ `FutureWarning`) | `PASS` only if the CI is wholly inside `[-delta_max, +delta_max]` (and ≥ 20 effective clusters) |
| `WARN` — 0 ∉ CI, small δ̂ | `NOT_GRADED` (+ `FutureWarning`) | usually `INCONCLUSIVE` (CI overlaps a margin boundary); `WARN` no longer exists |
| `FAIL` — 0 ∉ CI, δ̂ ≥ 0.05 | `NOT_GRADED` (+ `FutureWarning`) | `FAIL` only if the CI is wholly outside the margin (graded even below the cluster floor) |

Consequences for 0.5.x gate code:

- `status == "PASS"` **never fires** without `delta_max` (silent fail-closed).
- `status != "FAIL"` **always passes** without `delta_max` (silent fail-open —
  the dangerous direction for a drift monitor).
- For one release cycle, calling the audit without `delta_max` emits a
  `FutureWarning` naming this behavior change. Do not filter it; pass a margin.

```python
# 0.5.x
diag = audit_transportability(calibrator, probe)          # PASS / WARN / FAIL

# 0.6.0
diag = audit_transportability(
    calibrator,
    probe,
    delta_max=0.05,      # smallest mean bias that would change your decision
    family_size=3,       # all policy/group audits used in the same decision
)
# PASS / FAIL / INCONCLUSIVE / NOT_GRADED  (+ NOT_CHECKED at the analyze_dataset level)
```

New knobs: `cluster_ids` (independence units; defaults to `prompt_id`),
`sample_weights` (unequal-probability probes), `covariates`, `family_size`
(Bonferroni), `min_effective_clusters` (default 20 — withholds PASS, never
FAIL). Deserialized legacy `WARN` diagnostics normalize to `INCONCLUSIVE`.

**Unit contract**: the low-level audit grades in probe `oracle_label` units.
Margins passed through `analyze_dataset(transport=TransportAuditConfig(...))`
are in OUTPUT units — the units of `result.estimates`.

## 6) `calibrator=None` at complete oracle coverage

When every row is oracle-labeled, 0.6.0 reports the direct oracle mean and
fits no calibrator: `result.calibrator is None` (0.5.x returned a fitted
calibrator it did not need). The documented calibrator-reuse workflow must
check for `None`:

```python
result = calibrated_mean_ci(scores, labels)

if result.calibrator is None:
    # Complete oracle coverage: the estimate never used a calibrator, so there
    # is no judge->oracle map to transport-audit. To grade calibrator reuse for
    # a FUTURE partially-labeled run, fit on a partial slice explicitly:
    partial = labels.copy()
    partial[held_out_idx] = float("nan")          # mask a slice to force calibration
    fit = calibrated_mean_ci(scores, partial)
    audit = transport_audit(probe_scores, probe_labels, fit.calibrator, delta_max=0.05)
else:
    audit = transport_audit(probe_scores, probe_labels, result.calibrator, delta_max=0.05)
```

The same applies to `analyze_dataset`: fully observed policies route to their
oracle mean (`direct_oracle`), and `results.calibrator` can be `None` when no
policy needed calibration.

## 7) `best_policy()` — default unchanged, divergence now loud

The 0.5.0 safety default is **kept**: `best_policy()` defaults to
`reliable_only=True`, demoting a gate-flagged raw argmax and returning the
best gate-passing policy. New in 0.6.0, the demotion is never silent:

- the demoted raw argmax travels as `verdict.runner_up` with its gate reasons
  in the new `verdict.runner_up_reasons`,
- a warning is logged at demotion time,
- `result.summary()` prints the raw point-estimate winner with its
  limitations **and** the returned reliable winner
  (`Best reliable policy: ... — raw argmax ... was flagged (...)`).

Pass `reliable_only=False` for the raw argmax with `flagged=True` attached.
If every policy is flagged, the argmax returns with `all_flagged=True` — do
not crown it.

## 8) Ingestion is loud by default; `prompt_id` auto-generation retained

Invalid records **raise** by default on every ingestion path
(`analyze_dataset`, `fresh_draws_dir`/single-file/in-memory loading, and the
calibration-data loaders), with file/line context. To filter instead, opt in
explicitly — drops are counted and logged, never silent:

```python
results = analyze_dataset(fresh_draws_dir="responses/", on_invalid="drop")
```

Two 0.6.0 strictness additions to know about. Top-level/metadata field
conflicts are per-record validity errors like malformed fields or corrupt
JSON lines: they raise by default and are dropped-with-counts under
`on_invalid="drop"`. Explicit duplicate `(prompt_id, draw_idx)` rows are
conflicting row identities, not invalid records — they **always raise**,
regardless of `on_invalid`, with both conflicting rows identified. Omit
`draw_idx` to auto-assign sequential indices per prompt; `on_invalid`
governs only per-record validity errors.

`prompt_id` remains **optional** in fresh-draw records, as in 0.5.x: when
missing, it is auto-generated from a hash of the `prompt` field consistently
across the directory, single-file, and in-memory paths (paired comparisons
need real shared prompt ids, so supply them when you have them).

## 9) Calibration fold assignment changed (numeric drift vs 0.5.x)

0.5.x assigned calibration folds with the canonical `hash(prompt_id) % k`
(`get_folds_for_prompts`). 0.6.0 assigns whole oracle **prompt clusters** to
folds by a seeded-blake2b sort with round-robin assignment (balanced: fold
sizes differ by at most one cluster, so small oracle slices cannot produce
empty folds), and resolves the fold count from unique labeled clusters rather
than the raw label count (`calibration_info["n_folds"]` now records the count
actually used).

Consequences:

- Cross-fitted quantities (OOF predictions, OUA jackknife, SEs) differ
  numerically from 0.5.x at identical seeds. This is expected drift, not a bug.
- Fold membership now depends on the whole oracle cluster set: the 0.5.x
  "same `prompt_id` → same fold regardless of the rest of the data" stability
  property no longer holds for calibration folds. Do not use
  `get_fold`/`get_folds_for_prompts` (still exported) to predict calibration
  fold assignment — e.g. to construct held-out probes; hold rows out
  explicitly instead.

## 10) Reading 0.6.0 results: units and claim tiers

Results now say exactly what estimand they carry and in what units:

- `results.units` (`ResultUnits`): `estimand` is `"oracle_mean"`,
  `"judge_mean"`, or `"mixed"` — derived from the calibration state alone. A
  declared `output_scale` changes only the display axis
  (`normalization.results_scale`), never the estimand label, and is ignored
  (with a warning) for mixed runs.
- `metadata["claim_tier_by_policy"]`: per policy, `"DIRECT_ORACLE_MEAN"`
  (fully labeled, no calibrator), `"CALIBRATED_ORACLE_MEAN"`, or
  `"RAW_JUDGE_MEAN"`; `metadata["claim_tier"]` is the run-level tier
  (`"MIXED"` when tiers differ without calibration).
- `metadata["transport_audits"]`: per-policy
  `PASS / FAIL / INCONCLUSIVE / NOT_GRADED / NOT_CHECKED` records when
  `TransportAuditConfig` is supplied; only an observed `FAIL` adds a hard
  reliability gate.

Report numbers with their tier. A `RAW_JUDGE_MEAN` estimate is a judge-scale
quantity — never present it as an oracle-scale level.

## 11) Other numeric changes vs 0.5.x (expected drift)

Re-running identical 0.5.x inputs on 0.6.0 will produce different numbers for
several defensible reasons; none is a regression:

- **Bootstrap scheme replaced**: multinomial cluster resampling with
  resample-until-valid retries → positive exponential mean-one cluster weights
  (Bayesian bootstrap) with **no** retries (retrying conditioned on easy
  bootstrap worlds and biased SEs down). Same-seed CIs differ; the new scheme
  is validated against analytic SEs and Monte-Carlo coverage in the test suite.
- **Full-coverage routing**: fully labeled policies report the raw weighted
  oracle mean (`direct_oracle`) instead of a calibrated-plus-residual estimate.
- **Weighted ECDF ties** use proper group midranks, slightly shifting
  two-stage fits on data with tied (e.g. integer) judge scores.
- **`power_to_detect`** now evaluates both rejection tails: power is symmetric
  in the effect sign and slightly higher than 0.5.x's one-tail approximation.
- **Bootstrap refits multiply base weights** rather than replacing them, and
  representative-label augmentation uses the weighted ratio functional.
