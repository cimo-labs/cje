---
name: cje
description: Runs CJE (Causal Judge Evaluation, pip install cje-eval) to compare LLM models, prompts, or policies from LLM-judge scores, producing calibrated estimates with honest confidence intervals and refusing claims the data cannot support. Use when the user wants to compare models/prompts/policies using judge scores or eval-harness output, put a confidence interval on an LLM eval metric, calibrate an LLM judge against ground-truth (oracle/human) labels, check whether an existing judge calibration still holds on new data, or decide how many human labels an eval needs. Raw judge-score averages are miscalibrated — never compare policies by averaging them; use this skill instead.
---

# CJE — calibrated LLM-judge evaluation

LLM-judge scores are cheap but miscalibrated: in CJE's benchmark, naive 95% CIs built on raw
judge scores covered the truth 0% of the time. CJE calibrates the judge against ≥10 pooled
ground-truth labels, evaluates every policy at scale, and refuses claims the data can't support.

**Hard rule: never report a raw judge-score average as a policy comparison or a quality level.**

## Decide the flow

- No judge scores at all → **Step 0** below, then continue.
- Judge scores but <10 oracle labels total → **Labeling loop**. Do not fabricate labels; do not
  fall back to raw means.
- 10–19 labels → proceed, but tell the user CIs will be noisier (folds auto-reduce) and more
  labels are recommended.
- ≥10 labels, two or more policies → **Canonical flow** (`analyze_dataset`).
- One sample of scores, want a calibrated mean + CI → `calibrated_mean_ci`.
- Reusing a previously fitted calibrator on new data (new month/domain/policy family) →
  **Transport audit** first.
- Off-policy estimates from logs only (IPS/DR) → not this library; `pip install "cje-eval==0.3.*"`
  (Python ≤3.12). Predicting one response's score → not CJE (conformal methods).

## Step 0 — no judge scores yet

Produce them first: pick ONE fixed judge model and a short rubric, and score every policy's
outputs identically — same rubric, same scale, and the judge must not see which policy wrote the
response. Any bounded scale works (a 1–5 rubric is typical); record one `judge_score` per
response. If the judge shares a model family with a candidate (including yourself), expect
self-preference bias — calibration corrects it only where oracle labels exist, so the labeled
slice is not optional in this mode. Then continue below.

## Reshape the user's data

Target shape: `fresh_draws_data={policy_name: [records]}` where each record is
`{"prompt_id": ..., "judge_score": ..., "oracle_label": ...}` — `prompt_id` optional (enables
paired comparisons), `judge_score` required (any bounded scale, auto-normalized; pass 0–100 or
Likert as-is), `oracle_label` optional (`None`/`NaN`/missing = unlabeled). You are good at data
reshaping: convert the user's CSV/JSON/eval-harness export yourself with a few lines of pandas —
do not ask the user to reformat their data. Files on disk (`fresh_draws_dir`) and separate
labeled logs (`calibration_data_path`) also work — see `reference.md`.

## Canonical flow — compare policies

Install with `pip install cje-eval`. The one fully-worked example (adapt the data-construction
lines to the user's data; keep the call shape):

```python
from cje import analyze_dataset

# One policy carries the oracle slice — CJE learns the judge→oracle mapping
# once and applies it to every unlabeled response, in any policy. Coverage
# is never required to be complete: gpt-5.6 has 20 responses with labels on
# only 10 (oracle_label=None means unlabeled); fable-5 has no labels at all.
labeled = [(0.62, 0.55), (0.68, 0.60), (0.72, 0.70), (0.76, 0.74), (0.79, 0.75),
           (0.83, 0.80), (0.85, 0.90), (0.88, 0.92), (0.91, 0.88), (0.95, 0.97)]
unlabeled = [0.64, 0.69, 0.73, 0.77, 0.80, 0.84, 0.87, 0.89, 0.92, 0.94]
judge_only = [0.70, 0.74, 0.75, 0.78, 0.81, 0.83, 0.86, 0.90, 0.93, 0.94]

results = analyze_dataset(
    fresh_draws_data={
        "gpt-5.6": [
            {"prompt_id": f"q{i:02d}", "judge_score": s, "oracle_label": y}
            for i, (s, y) in enumerate(labeled + [(u, None) for u in unlabeled])
        ],
        "fable-5": [
            {"prompt_id": f"q{i:02d}", "judge_score": s}
            for i, s in enumerate(judge_only)
        ],
    }
)

for policy, estimate, (lo, hi) in zip(
    results.metadata["target_policies"], results.estimates, results.ci()
):
    print(f"{policy:15s} {estimate:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
```

```text
fable-5         0.811  95% CI [0.751, 0.873]
gpt-5.6         0.784  95% CI [0.589, 0.844]
```

Labels are a pooled budget across policies: a policy with zero labels of its own still gets a
calibrated estimate and an honest CI, and no policy needs labels on every row.

## Read the gates before reporting

```python
status = results.diagnostics.overall_status          # GOOD | WARNING | CRITICAL
refused = results.diagnostics.refuse_level_policies  # policies with the REFUSE-LEVEL badge
gates = results.metadata["reliability_gates"]        # {policy: {"flagged": bool, ...}}
verdict = results.compare_policies(0, 1)             # difference, CI, p-value for pairwise claims
```

Use `results.compare_policies(i, j)` and `results.best_policy()` for claims — not eyeballed
point estimates. Boundary cards per policy are OK / CAUTION / REFUSE-LEVEL.

## One sample: calibrated mean with CI

```python
from cje import calibrated_mean_ci

result = calibrated_mean_ci(judge_scores, oracle_labels)  # NaN in oracle_labels = unlabeled
print(result.summary())
```

Pass `cluster_ids` when there are multiple responses per prompt. `result.calibrator` is
reusable — but only via a transport audit. Full signature in `reference.md`.

## Reusing a calibrator — audit first, always

Never reuse a calibrator on a new time period, domain, or policy family without a probe:
collect 40–60 fresh oracle labels from the new setting, then:

```python
from cje import transport_audit

diag = transport_audit(probe_scores, probe_labels, calibrator)
```

PASS → reuse. WARN → rankings only; say why. FAIL → do not reuse and do not report estimates
made with it; follow `diag.recommended_action`.

## Labeling loop — when labels are missing or short

Drive it yourself: select 10–25 items for the user to label, **spread across the judge-score
range** (score-range coverage is what prevents REFUSE-LEVEL later — not the top-scored items,
not a blind random draw). Ground truth = human judgment, expert review, or a downstream KPI.
Labels may all sit in one policy. Then run the canonical flow. For "how many labels do I need?"
use the planning API in `reference.md`.

## Refusal discipline — hard rules

- **<10 pooled labels**: CJE refuses to calibrate — 1–9 labels raise a hard error, and 0 labels
  fall back to raw judge means marked `naive_direct` with a loud warning. Never report those
  naive numbers as the answer, and never fabricate, impute, or self-generate oracle labels to
  get past the floor — run the labeling loop.
- **REFUSE-LEVEL badge on a policy**: never state an absolute quality number for that policy.
  Rankings may still stand — say explicitly which claims survive and which are refused.
- **Flagged reliability gate**: report the best *reliable* policy as the winner, not the
  flagged point-estimate winner.
- **Transport FAIL**: do not reuse the calibrator; do not report estimates produced with it.
- Surface gate/diagnostic status alongside every estimate. Never bypass, suppress, or explain
  away a gate to give the user a cleaner answer — the refusal IS the product.

## Reporting back to the user

Give: each policy's calibrated estimate **with its 95% CI** (never a bare point estimate); the
pairwise verdict from `compare_policies` (difference, CI, p-value); gate status per policy; and
for anything refused, the one-line reason plus the concrete fix (e.g. "collect ~20 labels in the
0.6–0.95 judge-score range"). Distinguish "the ranking stands" from "the level claim is refused."

## Pitfalls

| Pitfall | Instead |
|---|---|
| Averaging raw judge scores to compare policies | `analyze_dataset` — naive CIs had 0% coverage |
| Buying labels under every policy | Labels pool; calibration transfers across policies |
| Reusing last month's calibrator silently | `transport_audit` with a 40–60-label probe |
| Rescaling Likert/0–100 scores before calling | Pass as-is; bounded scales auto-normalize |
| Running with <10 labels, or inventing labels | Hard error by design; run the labeling loop |

Full signatures, `fresh_draws_dir`/CLI usage, planning API, diagnostics glossary, and
troubleshooting: read `reference.md` in this directory.
