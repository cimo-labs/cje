---
name: cje
description: Use CJE (pip install cje-eval) to compare policies from judge scores and oracle labels, report calibrated means and paired differences with uncertainty, audit or correct calibration reuse, and plan evaluation or audit-label budgets. Use for eval-harness exports, production judge/outcome records, and estimating a candidate model or prompt from its generated responses before shipping it; off-policy IPS/DR reweighting of another policy's logged responses is outside current CJE.
---

# CJE: calibrated LLM-judge evaluation

LLM-judge scores are cheap but can be miscalibrated: in CJE's Chatbot Arena benchmark, naive
95% CIs built on raw judge scores covered the truth 0% of the time. CJE calibrates the judge
against a pooled slice
of ground-truth labels (≥10 independent labeled prompt clusters recommended; 4 is the calibration floor), evaluates every policy at scale,
and refuses claims the data can't support.

**Hard rule: never report a raw judge-score average as a policy comparison or a quality level.**

## Decide the flow

`reference.md` sits next to this file; if you fetched this file by URL, fetch the same URL ending in `reference.md`.

- Labels already exist (eval export, audit, production feedback) → first establish how each was
  chosen (**Reshape the user's data**); count only probability-sampled labels toward the 4- and
  10-cluster thresholds below.
- Planning labels to resolve a residual transport audit → **Audit budget** in `reference.md`;
  use `plan_transport_audits`, not the evaluation-power planner.
- Planning a future evaluation (sample size, label budget, detectable effect, or power) →
  **Planning flow**. Existing data and only a comparison requested? Use the analysis routes
  below; planning is optional, not a prerequisite.
- No judge scores at all → **Step 0** below, then continue.
- Complete oracle coverage for one sample → `calibrated_mean_ci` returns the direct oracle
  mean without fitting a calibrator. The four-cluster calibration floor does not apply;
  one independent cluster still cannot support a confidence interval.
- Partial oracle coverage with <4 independent labeled prompt clusters → **Labeling loop**. Do not fabricate labels; do not
  fall back to raw means (0 labels runs only as a loudly-flagged `naive_direct` fallback, and
  1–3 labels fall back to the same loudly-flagged UNCALIBRATED `naive_direct` tier in
  `analyze_dataset`; a calibrator cannot be fit below 4 independent labeled clusters. Treat
  the run as blocked and never report those numbers. `calibrated_mean_ci` raises a
  `ValueError` below this floor when calibration is needed).
- 4–9 independent labeled prompt clusters → runs, but calibration folds auto-reduce with a warning and CIs are noisier.
  Report results as provisional and run the labeling loop toward ≥10.
- ≥10 independent labeled prompt clusters, two or more policies → **Canonical flow** (`analyze_dataset`).
- One sample of scores, want a calibrated mean + CI → `calibrated_mean_ci`.
- Reusing a previously fitted calibrator on new data (new month/domain/policy family) →
  check fit/version provenance, then **Transport audit**.
- A transport audit failed and representative target labels are available → **Correction**
  in `reference.md`. Audit-only probes do not change the estimate.
- Existing production judge/outcome records → **Production outcomes** in `reference.md`;
  distinguish calibration data, observed evaluation responses, and off-policy estimation from logs.
- A candidate model or prompt not yet shipped → generate its responses on representative prompts
  and use the **Canonical flow** against the current policy. The calibrated judge stands in for
  the outcome only as far as the calibration carries over: grade that with a held-out probe of the
  candidate's labelled responses (**Reusing a calibrator**), or report its `NOT_CHECKED` limitation.
- Reweighting another policy's logged responses by logged propensities (IPS, or DR on top of
  target-policy fresh draws) → not this library; `pip install "cje-eval==0.3.*"`
  (Python ≤3.12). Predicting one response's score → not CJE (conformal methods).

## Planning flow: size a future evaluation

1. Establish the comparison, target population, practical effect size and its units, desired
   power, significance level, and budget. Record judge-score and oracle-label costs explicitly;
   distinguish the planner's allocation from additional candidate-policy, pilot, and transport-
   probe costs.
2. Use a probability-sampled pilot from the base policy where calibration will be learned.
   Variance fitting typically needs roughly 200+ independent prompts and 100+ randomly sampled
   oracle labels, with enough labeled and unlabeled data to vary both sample size and label
   count. The 10–25-label loop below starts calibration; it is not a sufficient planning pilot.
   With no suitable pilot, request one. Simulated variance models are optional sensitivity
   scenarios, clearly labeled as assumption-based, not evidence from the user's evaluation.
3. Follow `reference.md` §Planning: fit the variance model; inspect fit quality and warnings;
   then use `plan_evaluation` for a fixed budget or `plan_for_mde` for a target effect.
   A failed fit, `fit_ok=False`, or unresolved pilot sampling problem means the allocation is
   unreliable: explain what pilot evidence is missing rather than presenting a firm plan.
4. Report planned samples and labels, modeled cost, projected MDE/power, and assumptions.
   The planner uses independent-policy variance and asymptotic-normal critical values.
   Positive shared-prompt covariance makes the independence assumption conservative, but this
   is not a design-specific paired-power calculation or a guaranteed finite-sample result.
5. Save the plan before collection. Once data arrive, run the canonical analysis and read its
   realized intervals, comparisons, and gates. Report departures from the plan; do not describe
   projected power as achieved power.

Planning does not establish representative labeling or calibration transport. Passing
diagnostics cannot prove every assumption behind a statistical claim.

## Step 0: no judge scores yet

Produce them first: pick ONE fixed judge model and a short rubric, and score every policy's
outputs identically; same rubric, same scale, and the judge must not see which policy wrote the
response. Any bounded scale works (a 1–5 rubric is typical); record one `judge_score` per
response. If the judge shares a model family with some candidates but not others (including
yourself), expect asymmetric self-preference bias; it favors those candidates in the ranking,
and calibration corrects it only where oracle labels exist. Make sure labels or held-out
probes cover the advantaged policies. Then continue below.

## Reshape the user's data

Target shape: `fresh_draws_data={policy_name: [records]}` where each record is
`{"prompt_id": ..., "judge_score": ..., "oracle_label": ...}`; each record requires `prompt_id`
or nonempty `prompt` text from which CJE can derive it. Reuse the same prompt ID across
policies and repeated responses to preserve pairing and dependence clusters; a response ID
is a separate identity. `judge_score` is required (any bounded scale, auto-normalized; pass 0–100 or
Likert as-is), `oracle_label` optional: `None` (JSON `null`) or a missing key = unlabeled. Records reject `NaN`
(`Oracle field 'oracle_label' must be finite`), including the `NaN` pandas puts in blank cells, so
convert first (`df.astype(object).where(df.notna(), None)`); `NaN` means unlabeled only in the
array APIs. Do not pass `on_invalid="drop"` to get past that error: it deletes every unlabeled row. For Langfuse, use
the existing bridge before writing custom joins; see **Langfuse** in `reference.md`. For other
formats, convert CSV/JSON exports while preserving response identity, missing labels, and
sampling provenance. Files on disk (`fresh_draws_dir`) and separate labeled logs
(`calibration_data_path`) also work; see `reference.md`.

Before analysis, check what CJE cannot:

- **How each existing label was chosen.** Read any label-source column and ask the user. Attach
  `oracle_label` only for labels from a probability sample of that policy's rows: simple random
  (the default `label_design="representative"`) or random within strata at recorded rates
  (`label_design="known_propensity"` with `label_propensities`, a probability in (0, 1] for every
  row of every policy). Leave out labels chosen for the response's quality (complaints, failure
  reviews, judge–human disagreements, "interesting" cases): they pass every gate without warning
  and can reverse a comparison. `label_design="targeted_unknown"` does not repair them. If
  provenance is unknown, use only labels known to be random, or run the labeling loop.
- **Policy names and duplicate rows.** Print each policy with its row count. Names are used
  exactly, so `GPT-5.6`, `gpt-5.6` and `gpt-5.6 ` are three policies; merge variants only after
  the user confirms. Re-exported duplicates count twice unless each record has a unique `row_id`
  (e.g. the harness run ID, never the `prompt_id`): exact repeats of a `row_id` are then dropped
  with a warning, and conflicting repeats raise.
- **Prompt coverage and repeats.** Each policy's estimate averages its own rows, so policies that
  answered different prompts are compared over different prompt mixes (comparisons report
  `basis: "prompt_cluster_partial_overlap"`), and a prompt with k responses counts k times. Score
  refusals and errors as that policy's responses rather than dropping them; restrict to shared
  prompts only when prompts are missing for reasons unrelated to the policy, and report what you
  dropped.

## Canonical flow: compare policies

Install with `pip install cje-eval` (Python 3.10–3.13). Record the package version and the
version that fitted any reused calibrator. **When upgrading to 0.8.0, refit pre-0.8.0 saved two-stage calibrators
from retained inputs before reuse** to rebuild empirical-rank boundaries with corrected
arithmetic. If fit provenance is unknown, refit from retained inputs rather than assuming
compatibility. See [the release notes](https://github.com/cimo-labs/cje/releases/tag/v0.8.0).

Adapt this synthetic example to the user's data; it demonstrates the API, not sufficient
power for a real evaluation:

```python
from cje import analyze_dataset

# Synthetic data: two policies, gpt-5.6 vs fable-5, each answered the same
# 20 prompts. A separate fixed judge model scored all 40 responses; human
# raters labeled 10 of gpt-5.6's (None = not labeled).
judge_scores = {
    "gpt-5.6": [0.62, 0.68, 0.72, 0.76, 0.79, 0.83, 0.85, 0.88, 0.91, 0.95,
                0.64, 0.69, 0.73, 0.77, 0.80, 0.84, 0.87, 0.89, 0.92, 0.94],
    "fable-5": [0.70, 0.74, 0.75, 0.78, 0.81, 0.83, 0.86, 0.90, 0.93, 0.94,
                0.72, 0.76, 0.79, 0.80, 0.84, 0.85, 0.88, 0.89, 0.91, 0.95],
}
human_labels = [0.55, 0.60, 0.70, 0.74, 0.75, 0.80, 0.90, 0.92, 0.88, 0.97,
                None, None, None, None, None, None, None, None, None, None]

draws = {
    "gpt-5.6": [
        {"prompt_id": f"q{i:02d}", "judge_score": s, "oracle_label": y}
        for i, (s, y) in enumerate(zip(judge_scores["gpt-5.6"], human_labels))
    ],
    "fable-5": [
        {"prompt_id": f"q{i:02d}", "judge_score": s}
        for i, s in enumerate(judge_scores["fable-5"])
    ],
}
results = analyze_dataset(fresh_draws_data=draws)
print(results.summary())

# Is fable-5 better? Test the paired difference on the shared prompts;
# don't compare the two intervals by eye.
for c in results.compare_all_policies():
    print(f"{c['policy1']} - {c['policy2']}: {c['difference']:+.3f}  "
          f"95% CI [{c['ci_lower']:+.3f}, {c['ci_upper']:+.3f}]  p={c['p_value']:.2f}")
```

`summary()` reports each policy's calibrated estimate with a 95% CI, the best policy, and
its limitations. A policy with no calibration labels of its own still depends on the
shared-calibration assumption; grade residual transport separately with held-out oracle
probes and a predeclared practical margin. For programmatic access use `results.estimates`,
`results.ci()`, and `results.metadata["target_policies"]` (sorted by policy name, not input order;
`estimates`, `ci()` and `compare_policies` indices all follow it).

## Read the gates before reporting

```python
status = results.diagnostics.overall_status          # GOOD | WARNING | CRITICAL
refused = results.diagnostics.refuse_level_policies  # policies with the REFUSE-LEVEL badge
gates = results.metadata["reliability_gates"]        # {policy: {"flagged": bool, ...}}
pairs = results.compare_all_policies()               # every pair, named: difference = policy1 - policy2
pols = results.target_policies                       # sorted by name, NOT your dict order
verdict = results.compare_policies(pols.index("candidate"), pols.index("base"))  # candidate - base
```

Use `results.compare_policies(i, j)` for pairwise claims and surface the highest point estimate
with its diagnostics rather than silently substituting another policy. Do not rely on eyeballed
point estimates. The default analytic path combines the paired sampling SE with the
oracle-jackknife variance of the difference (`method: "paired_if_oua"`); an explicit bootstrap
run uses the joint replicate matrix (`method: "paired_bootstrap"`). Report the `method` key's
basis if asked how the p-value was computed. For
many-pair audits use `results.compare_all_policies(adjust="bh")` (adds Benjamini-Hochberg
`p_adjusted`/`significant_adjusted`). Boundary cards per policy are OK / CAUTION / REFUSE-LEVEL.

## One sample: calibrated mean with CI

```python
from cje import calibrated_mean_ci

result = calibrated_mean_ci(judge_scores, oracle_labels)  # NaN in oracle_labels = unlabeled
print(result.summary())
```

Pass `cluster_ids` when there are multiple responses per prompt. With partial oracle
coverage, check that `result.calibrator` is not `None` before reusing it for a transport
audit; complete coverage reports the direct oracle mean without fitting a calibrator.
Full signature in `reference.md`.

## Reusing a calibrator

Check version/fit provenance as above. Before relying on calibration transport for a new
time period, domain, or policy family, audit with held-out, probability-sampled probes.
Without probes, retain `NOT_CHECKED` as an unresolved assumption, not an observed failure.
Use at least 20 effective independent clusters and size the probe
for the desired interval width. For high-level analyses, pass the probes with the run so the
state is preserved in results and a `FAIL` augments the gate when the estimate depends on that map:

```python
from cje import TransportAuditConfig

transport = TransportAuditConfig(
    probes_by_policy={"candidate": held_out_probe_rows},
    delta_max_by_policy={"candidate": 0.03},  # units of results.estimates (label scale); 0.03 suits 0-1 labels
    family_size=n_groups,
)
results = analyze_dataset(fresh_draws_data=draws, transport=transport)
```

For an already fitted calibrator, use the array primitive:

```python
from cje import transport_audit

diag = transport_audit(
    probe_scores,
    probe_labels,
    calibrator,
    delta_max=0.03,
    cluster_ids=prompt_ids,
    family_size=n_groups,
)
```

Probes must be labels the calibrator was not fit on: never pass a row whose `oracle_label` is
already in the evaluation draws or `calibration_data_path`. The calibrator reproduces the mean of
its own training labels, so reused rows give a residual near zero by construction and the
audit cannot catch a failure. Give every record an `observation_id` (the response or run ID) on draws and probes
alike, and `analyze_dataset` raises on overlap; `transport_audit` cannot check this. Ask the user
for the margin before seeing probe results: the largest mean calibration error they would accept
for this decision, in the units of `results.estimates`. `0.03` above is a placeholder for 0–1
labels; on a 0–10 scale it makes `PASS` nearly unreachable. If the user cannot name a margin,
omit it and report `NOT_GRADED`.

`PASS` means the simultaneous residual CI is wholly inside the declared margin. `FAIL` means
it is wholly outside (graded even below the 20-effective-cluster floor; a policy cannot
escape a FAIL by supplying too small a probe). Boundary overlap or too few effective clusters is `INCONCLUSIVE`;
no margin is `NOT_GRADED` and can never PASS or FAIL. These verdicts do not replace the
separate scalar support card. A policy without a supplied probe is recorded as `NOT_CHECKED`
rather than silently treated as a pass.

## Labeling loop: when labels are missing or short

Drive it yourself: select 10–25 items for the user to label; **random within judge-score
strata**, so the slice stays a probability sample while covering the score range
(score-range coverage is what prevents REFUSE-LEVEL later; labeling only the top-scored
items is the classic mistake; see `label_design` in `reference.md` if strata are sampled
unevenly). Ground truth = human judgment, expert review, or a downstream KPI.
A trusted stronger model can also serve; the estimate then targets that model's judgment, so
say so when reporting. Labels may all sit in one policy, but then every other policy's estimate, and any difference
involving it, rests on that calibration carrying over, which the `compare_policies` p-value does
not cover. When the user can rate every policy's responses, spread the random slice across
policies. Otherwise, before naming a winner, put a held-out probe (at least 20 effective prompt
clusters, kept out of `oracle_label`) on each policy in the deciding comparison that has no labels
of its own. Then run the canonical flow.
For prospective sample-size or label-budget decisions, use the separate **Planning flow**
above; do not treat this starter batch as a sufficient variance-fitting pilot.

## Reporting limitations

- **Too few pooled labels**: 0 labels fall back to raw judge means marked `naive_direct` with a
  loud warning; never report those naive numbers as the answer. 1–3 labels also fall back to
  the same loudly-flagged UNCALIBRATED `naive_direct` tier in `analyze_dataset` (a calibrator
  cannot be fit below 4 independent labeled clusters; all policies gate-FLAGGED); treat the
  run as blocked and never report those numbers; `calibrated_mean_ci` raises a `ValueError`
  for <4 independent labeled clusters when calibration is needed. Complete oracle coverage
  in the array API instead returns the direct oracle mean without calibration. With 4–9
  independent labeled clusters, folds reduce; report the CIs as noisier and provisional,
  and recommend ≥10. Never fabricate, impute, or self-generate oracle labels to
  get past the floor; run the labeling loop.
- **REFUSE-LEVEL badge on a policy**: never state an absolute quality number for that policy.
  The scalar badge does not establish ranking validity; use the paired comparison and separate
  residual/covariate evidence for any ranking claim.
- **Flagged diagnostic evidence**: still surface the highest point estimate, with its limitation
  adjacent. Do not silently substitute a different policy estimand.
- **Transport FAIL**: keep the requested point estimate visible with the failed assumption;
  do not base a decision on the unchanged calibration map. Consider representative target-label
  correction (reference §Correction). A failed level audit alone does not disprove a ranking;
  a ranking claim needs evidence about the difference in policy mean residuals.
- Surface gate/diagnostic status alongside every estimate. Never bypass, suppress, or explain
  away a gate to give the user a cleaner answer.

## Reporting back to the user

Give: each policy's calibrated estimate **with its 95% CI** (never a bare point estimate); the
pairwise verdict from `compare_policies` (difference, CI, p-value); gate status per policy; and
for any limited claim, the one-line reason plus the concrete fix (e.g. "collect labels in the
0.6–0.95 judge-score range"). If a policy in the headline comparison has no labels of its own
and its transport audit is `NOT_CHECKED`, `INCONCLUSIVE` or `NOT_GRADED`, call the difference
provisional (it assumes the calibration fits both policies equally), say so before the p-value,
and recommend a held-out probe. Never infer that a ranking survives from a scalar support badge. Overlapping marginal CIs
do not establish equivalence; use the paired difference and a predeclared practical margin
for an equivalence claim. Plan the analysis sample/stopping rule before collection; repeated
looks require an appropriate sequential design.

## Save an auditable run

Leave an executable analysis script and an output directory, not only a chat summary. Save:

- Inputs or retained locations plus checksums; prompt/cluster identifiers; sampling and label-
  selection design (including inclusion probabilities when applicable); label source and
  target meaning; judge/rubric versions; and any excluded records.
- Python and `cje-eval` versions, exact configuration and random seeds, declared score scales,
  effect/power/alpha/cost settings, and sampling/calibration assumptions marked known or
  unresolved. Preserve the plan, fitted variance components, fit quality, and warnings.
- `plan.to_dict()` when planning; `results.to_dict()` for the comparison; explicit requested-
  alpha policy intervals and pairwise results; diagnostics, gates, and transport verdicts.
  Do not discard a warning or failed gate when saving the run.
- The rerun command, environment requirements, and a short readout of what the evidence does
  and does not support. See `reference.md` for result-export limits.

These are files the agent assembles using existing APIs, not a new CJE audit API. Keep source
inputs available for reruns; a checksum alone is not the data. “Auditable” means the evidence
and analysis can be inspected and reproduced, not that their assumptions are guaranteed.

## Pitfalls

| Pitfall | Instead |
|---|---|
| Averaging raw judge scores to compare policies | `analyze_dataset`; naive CIs had 0% coverage in the Arena benchmark |
| Putting every label on one policy when each could be labeled | With a few policies, label a random slice of each (routes become `augmented`); with many, pool labels and grade the transfer with held-out probes before relying on it |
| Reusing last month's calibrator silently | Held-out `transport_audit` with an explicit margin and at least 20 effective clusters |
| Rescaling Likert/0–100 scores before calling | Pass as-is; bounded scales auto-normalize |
| Fitting calibration with <4 independent labeled prompt clusters, or inventing labels | Treat the flagged `naive_direct` fallback as blocked and run the labeling loop. The array API raises when partial coverage needs calibration; complete oracle coverage can use the direct oracle mean. |

Full signatures, `fresh_draws_dir`/CLI usage, planning API, diagnostics glossary, and
troubleshooting: read `reference.md` in this directory.
