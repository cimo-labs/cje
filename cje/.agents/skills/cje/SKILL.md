---
name: cje
description: Use CJE (pip install cje-eval) to compare policies from judge scores and oracle labels, report calibrated means and paired differences with uncertainty, audit or correct calibration reuse, and plan evaluation or audit-label budgets. Use for eval-harness exports, production judge/outcome records, and estimating a candidate model or prompt from its generated responses before shipping it; off-policy IPS/DR reweighting of another policy's logged responses is outside current CJE.
---

# CJE: calibrated LLM-judge evaluation

**Stop checks.** If your tool cut this output short or saved it to a file, open that file and
read it whole before anything else. These decide the verdict line before any number:

- Judge scores only, or fewer than 4 independently labeled prompts: "Blocked: no ground-truth
  labels yet"; report no number, ranking or pick (hard rule 1).
- The only labels are ratings users chose to give (stars, thumbs, CSAT): they count as zero
  labels; "Undecided: the only labels are self-selected" (hard rule 2).
- Judge models, versions or rubrics differ unevenly by policy: "Not decision-grade" (rule 3).
- A policy with no labels of its own and no transport `PASS`: "Undecided", with no winner and
  no lean (hard rule 5).
- Decide from the paired difference, never from raw judge means or overlapping intervals.

## Step 0: install, then print the version

```bash
python --version                     # must be 3.10–3.13
pip install -U "cje-eval>=0.9.2" pandas
python -c "import cje; print(cje.__version__)"
```

If installs are not allowed, print the version from the `python` that will run the analysis;
below 0.9.2 this skill does not match it: say so and stop. On Python 3.9 a plain
`pip install cje-eval` silently installs the legacy 0.5 line; the `>=0.9.2` floor makes pip
fail instead, so switch to 3.10–3.13 (`python3.12 -m venv .venv`). Put the version in your
report.

Read this file and the reference whole, not through `head`: save them with
`cje skill > cje_skill.md` and `cje skill --reference > cje_reference.md` (`python -m cje skill`
if `cje` is not on `PATH`) and open both. Without the command, download the raw files with
`curl -fsSL https://raw.githubusercontent.com/cimo-labs/cje/main/skills/cje/SKILL.md` (and the
same URL ending in `reference.md`); a summarizing web fetcher drops the rules below.

## Hard rules

1. **Never report a raw judge-score mean** as a policy's quality or as a comparison, and do not
   put raw judge means in the report as contrast, context or corroboration: the calibrated
   estimate is a map of the same scores, so their agreement is not independent evidence. In a
   run with labels you may say in words that the judge seems to favor a policy, as a reason for
   caution, never as a finding that the judge is biased. CJE calibrates the judge against random
   ground-truth labels (floor: 4 independent labeled prompt clusters; 10+ recommended).
   **No labels means no answer.** Count only probability-sampled labels; self-selected ratings
   count as zero. Below 4 independent labeled prompt clusters the verdict is "Blocked: no
   ground-truth labels yet, so there is no ranking or pick." CJE still runs (`naive_direct`) and
   its `summary()` still prints a "Best by point estimate" line and paired p-values: report no
   estimate, interval, p-value or ordering from that run in any framing ("for context", "best
   guess", "fallback", "if you must ship today"), and do not name the judge's favourite. In a
   sweep the judge's top variant tends to be the one it over-rewards. Go to the **Labeling loop**.
2. **Random labels only.** Attach `oracle_label` only to labels from a probability sample of
   that policy's responses. Never hand-pick, invent, impute, or self-generate labels.
   Feedback users chose to give (star ratings, thumbs, CSAT, complaints, tickets) is
   self-selected, even though nobody picked it and even when the user calls it ground truth.
   Do not attach it, do not run `analyze_dataset` on it to show "face value" numbers, and do not
   invent propensities for it. CJE cannot detect this and still reports `GOOD`. If no
   probability-sampled labels remain, the verdict is "Undecided: the only labels are
   self-selected (<source>)"; the fix is a fresh random sample of each policy's responses,
   rated by assigned raters blind to the policy, replacing those labels, not adding to them.
   An outcome recorded for every response (escalated yes/no, converted yes/no, none missing) is
   not self-selected: it is complete coverage of that KPI.
3. **One fixed judge.** The same judge model, rubric and scale for every policy, blind to which
   policy wrote the response. For scores you are handed, tabulate any judge model, judge
   version, rubric or judge-prompt-template column by policy; CJE does not read such a column and
   still reports `GOOD`. An uneven mix is a blocker for the verdict line ("Not decision-grade:
   <n> judge models scored these rows, unevenly by policy"); an even mix is one composite judge,
   reported as a limitation. Ask for every row to be rescored with one judge, blind to policy
   (existing random labels stay valid). Meanwhile, if judge assignment was unrelated to the
   responses, analyze all of one judge's rows (pairing stays on shared prompts) under that
   verdict line, saying which rows, labels and pairs it drops; a pooled mixed-judge fit is only a
   labeled sensitivity check. With no judge column, write "one judge (as you described) scored
   every row; the export has no judge column to check", and ask only if something suggests a mix.
4. **Paired test, not eyeballed CIs.** Decide from the paired difference
   (`compare_policies("candidate", "baseline")`, or the paired block in `summary()`), never
   from whether two policies' intervals overlap. A paired CI that includes 0 means the ranking
   is not established at this sample size. Never state or imply that the policies are equal,
   close, tied, comparable or within noise, in any section or wording (negated, quoted or in a
   team summary). Never narrate which policy the point estimate favours or present it alone as
   the size of the gap. Write instead, in the user's units: "No reliable winner: the paired CI for
   <B> − <A>, [<lo>, <hi>], allows <A> to be up to <|lo|> better or <B> up to <hi> better." Then
   ask exactly "What is the smallest gap on <scale> that would change which one you ship?" and
   size the labels (Reporting item 3). No cost or latency tiebreak, even hedged, unless the
   paired CI lies wholly inside a margin the user declared before seeing results; then say
   "equivalent within ±<m>" and cost may decide.
5. **No winner and no lean on borrowed calibration.** A policy with no labels of its own whose
   transport audit is not `PASS` is never named the winner, said to lead or be ahead, or called
   "significant", whatever the p-value: its estimate and every difference involving it assume
   another policy's calibration transfers, which the CI and p-value do not cover, and the true
   difference can have either sign. The first line: "Undecided: <B> has no
   labels of its own, so the paired difference <d> [<lo>, <hi>] assumes <A>'s calibration
   carries over to <B>; the true difference can have either sign." Do not narrate which estimate
   is higher; say the interval and p-value measure sampling noise only, not whether <A>'s
   calibration fits <B>. A raw judge gap toward the unlabeled policy, said in words without
   raw means, is a red flag (the judge
   may favor its style, and the borrowed calibration cannot see that), not support. Then size
   what would settle it:
   - Any random labels on that policy's own responses (same prompts, raters blind to policy)
     remove the borrowed flag and correct its estimate; about 20 or more are needed before its
     interval is useful. That is a floor, not a precision target: the rerun may still say "not
     established". Ask for the user's practical margin and size against it, never against the
     borrowed difference; do not frame new labels as confirming a win.
   - A held-out probe sized with `plan_transport_audits` is for keeping a calibrator across
     runs; it needs new random labels on that policy's responses too (at least 20 effective
     prompt clusters, usually more), so it is not a label-free shortcut.

   The same holds for a policy whose own labels go unused on a plug-in route
   (`use_augmented_estimator=False`, `label_design="targeted_unknown"`). `summary()` marks these
   `[borrowed calibration]` or `[uncorrected calibration]`, and
   `results.metadata["transport_unverified"]` lists them.
6. **Surface every gate, badge and audit state** next to the estimate it qualifies, glossed in
   plain words. Never bypass, suppress (including by turning logging off), or explain one away
   to give a cleaner answer.

## Reporting back to the user

Write for a busy reader, in this order, even for a "quick answer":

1. **Verdict, one line, before any number.** One of: "<A> beats <B>"; "No reliable winner: the
   paired CI includes 0" (hard rule 4); "Undecided: <B> has no labels of its own and its
   transport is `NOT_CHECKED`" (also `INCONCLUSIVE`, `NOT_GRADED` or `FAIL`; hard rule 5);
   "Blocked: no ground-truth labels yet" (hard rule 1); "Undecided: the only labels are
   self-selected" (hard rule 2); "Not decision-grade: <precondition>" (hard rule 3); with a
   flagged gate, the `decision_note` ("Not decision-ready: ...", hard rule 6). Once nothing
   above blocks the verdict (own labels on both sides or a transport `PASS`, no flagged gate),
   compare a ship threshold the user named with the paired CI, not with 0: lower bound above it
   means ship, a CI straddling it means not shown to clear the bar, an upper bound below it means
   shown not to clear it. Report the difference once, as candidate − baseline, taking
   `difference`, `ci_lower` and `ci_upper` from one `compare_policies("candidate", "baseline")`
   call; `summary()` prints pairs in name order, so never relabel one of its lines.
2. **Size and uncertainty.** The paired difference with its 95% CI and p-value (say whether
   adjusted; with one pair, that it is the only comparison), then each policy's calibrated
   estimate **with its 95% CI**, never a bare point estimate or an unlabeled ±. Name intervals
   with `dict(zip(results.target_policies, results.ci()))`: with two policies,
   `lo, hi = results.ci()` unpacks the two policies' intervals, not a lower and upper bound.
3. **What would make it firm.** New labels: a random sample of that policy's responses on shared
   prompts, never rows someone picked; 20+ for a policy with none of its own (hard rule 5). When
   every policy has labels and the paired CI includes 0, ask for the smallest gap g that matters
   and give a number: as a rough lower bound, m_needed ≈ m_now × (2.8 × SE_now / g)² labels per
   policy for 80% power at two-sided α = 0.05 (SE_now = the pair's `se_difference`, or its CI
   width / 4). Report m_needed − m_now more labels, capped at the unlabeled rows left; say
   plainly when g is out of reach without new prompts, and that the rerun can still come back
   not established. A boundary flag needs more random labels within judge-score strata that
   reach the range the warning names (unequal rates: `known_propensity`). For a real plan, use
   the **Planning flow**.
4. **Limitations, on every run, even when everything passes.** One line each:
   - Status and gates per policy, each glossed in a clause the user could paste into a decision
     doc: `GOOD` "the estimator's own checks passed; they do not test how labels were chosen or
     which judge scored"; boundary card OK "the labeled rows cover the judge-score range the
     estimate uses"; `decision_ready` (its `decision_note` in your words, as candidate − baseline)
     "the top policy beats every
     other in the paired test and no gate blocks it"; df "set by the number of labeled prompts,
     which limit precision". `NOT_CHECKED` on a policy with its own labels (route `augmented` or
     `direct_oracle`): "not needed: its estimate is corrected by its own random labels", and
     nothing more about
     transport; on a borrowed policy: "nobody has checked that <A>'s calibration fits <B>".
   - Labels: how they were chosen, credited to the user when they said it ("the random <n> you
     described; label rates are even across judge-score fifths, which fits that but cannot prove
     it"). Do not re-ask facts the user stated and the data agree with.
   - Judge: which judge scored the rows (hard rule 3). The `cje-eval` version. In the last line,
     the saved analysis script and results file. Leave method keys such as `paired_if_oua` out.

Plan the analysis sample and stopping rule before collection; repeated looks require an
appropriate sequential design.

## Decide the flow

- Labels already exist (eval export, audit, production feedback) → first establish how each was
  chosen (**Reshape the user's data**); count only probability-sampled labels toward the 4- and
  10-cluster thresholds below.
- Judge scores only, or fewer than 4 independent labeled prompt clusters → **Blocked** (hard
  rule 1). 0 labels and 1–3 labels both return the loudly flagged UNCALIBRATED `naive_direct`
  tier (no calibrator can be fit; all policies gate-FLAGGED); `calibrated_mean_ci` raises a
  `ValueError` when calibration is needed. You may run `analyze_dataset` to read its warnings,
  but the report contains no estimate, interval, p-value or ordering from that run. Then go to
  the **Labeling loop**; never fabricate labels to pass the floor.
- Only self-selected labels (ratings, thumbs, complaints) → **Undecided** (hard rule 2); propose
  a random labeled sample.
- 4–9 independent labeled prompt clusters → runs with fewer calibration folds and noisier CIs;
  report results as provisional and label toward ≥10.
- ≥10 independent labeled prompt clusters, two or more policies → **Canonical flow**
  (`analyze_dataset`). One sample → `calibrated_mean_ci`; with complete oracle coverage it
  returns the direct oracle mean without fitting a calibrator (one cluster still cannot
  support a CI).
- Planning a future evaluation → **Planning flow**; labels for a residual transport audit →
  **Audit budget** in `reference.md`. With existing data and only a comparison asked, skip it.
- No judge scores at all → **Produce judge scores** below, then continue.
- Reusing a previously fitted calibrator on new data (new month/domain/policy family) → check
  fit/version provenance, then **Reusing a calibrator**. A failed audit with representative
  target labels available → **Correction** in `reference.md`; audit-only probes do not change
  the estimate.
- Existing production judge/outcome records → **Production outcomes** in `reference.md`.
- A candidate model or prompt not yet shipped → generate its responses on representative
  prompts and use the **Canonical flow** against the current policy, with a random slice of the
  candidate's own responses labeled (or a held-out probe graded `PASS`); otherwise its
  comparison is undecided (hard rule 5).
- Reweighting another policy's logged responses (IPS/DR) → not this library
  (`pip install "cje-eval==0.3.*"`, Python ≤3.12). Predicting one response's score → not CJE.

## Planning flow: size a future evaluation

1. Establish the comparison, target population, practical effect size and its units, power,
   significance level, and budget. Record judge-score and label costs; say what the budget
   covers (response generation, the pilot and transport probes are separate).
2. Use a probability-sampled pilot from the base policy where calibration will be learned:
   roughly 200+ independent prompts and 100+ random labels (say so when a pilot only just meets
   this). The labeling loop below is not a planning pilot. With no suitable pilot, request one.
3. Follow `reference.md` §Planning: convert the pilot with the rubric's declared endpoints
   (never its observed min and max), fit the variance model, then use `plan_evaluation` for a
   fixed budget or `plan_for_mde` for a target effect. Keep logging on for every fit,
   sensitivity refits included, and report `fit_ok` and any "measurement cells were clamped"
   warning. A failed fit, `fit_ok=False`, or an unresolved pilot sampling problem means no firm
   plan: explain what pilot evidence is missing.
4. Open with a one-line budget verdict ("Does not fit: <total> across <k+1> policies against
   <budget>; <budget> buys an MDE of <x>"). The planner sizes ONE policy: 2 candidates plus a
   baseline are 3 policies, so plan at `budget=B/3` and multiply responses, labels and cost by 3.
   Report per-policy counts, the total, projected MDE and power in the user's units, the pilot's
   size, fit warnings, and the assumptions (independent-policy variance, asymptotic-normal
   critical values: optimistic at small label counts, not a paired-design calculation). With k
   comparisons and family-wise control, also show the plan at `alpha=0.05/k`. Report the
   equal-allocation result first; label any reallocation as your own.
5. Save the plan before collection. Once data arrive, run the canonical analysis and report
   departures from the plan; never describe projected power as achieved power. Planning does not
   establish representative labeling or calibration transport.

## Produce judge scores

Pick ONE fixed judge model and a short rubric, and score every policy's outputs identically;
same rubric, same scale, and the judge must not see which policy wrote the response. Any bounded
scale works (a 1–5 rubric is typical); record one `judge_score` per response. A judge that
shares a model family with some candidates (or with you) favors them; calibration corrects that
only where labels exist, so label or probe the advantaged policies. Then continue below.

## Reshape the user's data

Target shape: `fresh_draws_data={policy_name: [records]}`, each record
`{"prompt_id": ..., "judge_score": ..., "oracle_label": ...}` (`prompt_id`, or `prompt` text it
can be derived from, is required). Reuse the same prompt ID across policies and repeated
responses to keep pairing and clusters. Pass any bounded scale as-is and declare the endpoints
the user states with `fresh_judge_scale=(lo, hi)` and `fresh_oracle_scale=(lo, hi)`, never the
observed range, and describe results on that scale. Undeclared, values outside [0, 1] are
normalized by their observed min and max; estimates, CIs and differences come back in label
units either way. Declaring validates the range, sets the `Calibration coverage (±t)` tolerance
(10% of the range) and fixes what [0, 1] means for planning. `oracle_label` `None` or missing =
unlabeled: convert pandas blanks first (`df.astype(object).where(df.notna(), None)`) and never
pass `on_invalid="drop"` to get past a validation error (it deletes rows). Langfuse, CSV,
`fresh_draws_dir` and `calibration_data_path` recipes are in `reference.md`; keep response
identity, missing labels and sampling provenance.

Before analysis, check what CJE cannot:

- **How each existing label was chosen.** If the user has not said, read any label-source
  column and ask; labels nobody can account for count as not random. Attach `oracle_label`
  only for labels from a probability sample of that
  policy's rows: simple random (the default `label_design="representative"`) or random within
  strata at recorded rates (`label_design="known_propensity"` with `label_propensities`, a
  probability in (0, 1] for every row of every policy). Leave out labels chosen for the
  response's quality (complaints, failure reviews, judge–human disagreements, "interesting"
  cases): they pass every gate without warning and can reverse a comparison;
  `label_design="targeted_unknown"` does not repair them. Self-selected user feedback is never a
  probability sample (hard rule 2). One-line check: the label rate by judge-score quintile within
  each policy, with a chi-square test, not by eye. Rates that track the judge score beyond chance
  mean selection. Different rates across policies are fine when each policy's labels were drawn
  at random from its own rows; ask why they differ. Even rates do not prove randomness.
- **Judge identity.** Tabulate the judge columns by policy (hard rule 3).
- **Policy names, duplicates and coverage.** Print each policy with its row count; names are
  exact (`production` and `production ` are two policies), so merge variants only after the user
  confirms. Give each record a unique `row_id` (the harness run ID, never the `prompt_id`) or
  re-exported duplicates count twice. Policies that answered different prompts are compared over
  different prompt mixes, and a prompt with k responses counts k times; score refusals and
  errors as that policy's responses. Act on `analyze_dataset`'s warnings; they change nothing.
  Details: **Your own data** in `reference.md`.

## Canonical flow: compare policies

Install as in **Step 0** and record the package version. Adapt this synthetic example to the
user's data; it demonstrates the API, not sufficient power for a real evaluation:

```python
from cje import analyze_dataset

# Synthetic data: two policies, production vs candidate, each answered the same
# 20 prompts. A separate fixed judge model scored all 40 responses; human
# raters labeled 10 of production's (None = not labeled).
judge_scores = {
    "production": [0.62, 0.68, 0.72, 0.76, 0.79, 0.83, 0.85, 0.88, 0.91, 0.95,
                0.64, 0.69, 0.73, 0.77, 0.80, 0.84, 0.87, 0.89, 0.92, 0.94],
    "candidate": [0.70, 0.74, 0.75, 0.78, 0.81, 0.83, 0.86, 0.90, 0.93, 0.94,
                0.72, 0.76, 0.79, 0.80, 0.84, 0.85, 0.88, 0.89, 0.91, 0.95],
}
human_labels = [0.55, 0.60, 0.70, 0.74, 0.75, 0.80, 0.90, 0.92, 0.88, 0.97,
                None, None, None, None, None, None, None, None, None, None]

draws = {
    "production": [
        {"prompt_id": f"q{i:02d}", "judge_score": s, "oracle_label": y}
        for i, (s, y) in enumerate(zip(judge_scores["production"], human_labels))
    ],
    "candidate": [
        {"prompt_id": f"q{i:02d}", "judge_score": s}
        for i, s in enumerate(judge_scores["candidate"])
    ],
}
results = analyze_dataset(fresh_draws_data=draws)

# Estimates, then the paired test of candidate vs production on the shared
# prompts (decide from that, not by comparing the two intervals by eye).
print(results.summary())
```

`summary()` prints each policy's calibrated estimate with a 95% CI, the best by point estimate
(not a test), the paired differences with unadjusted CIs and p-values, their `[gate-flagged]`,
`[borrowed calibration]` or `[within transport margin]` marks, and "No reliable winner" or "No
decision-ready winner" when every printed pair supports that. The quickstart's `candidate` is
the borrowed case: by hard rule 5 its comparison is undecided. `results.target_policies` is
sorted by name, not input order; `estimates`, `ci()` and integer indices follow it.

## Read the gates before reporting

```python
status = results.diagnostics.overall_status          # GOOD | WARNING | CRITICAL
refused = results.diagnostics.refuse_level_policies  # policies with the REFUSE-LEVEL badge
gates = results.metadata["reliability_gates"]        # {policy: {"flagged": bool, ...}}
pols = results.target_policies                       # sorted by name, NOT your dict order
cis = dict(zip(pols, results.ci()))                  # {policy: (lo, hi)}
routes = dict(zip(pols, results.metadata["point_estimator"]["routes"]))  # augmented = own labels
pairs = results.compare_all_policies()               # difference = policy1 - policy2
verdict = results.compare_policies("candidate", "base")  # candidate - base
best = results.best_policy()                         # .decision_ready, .decision_note
```

`decision_ready` is True only when the winner beats every other usable policy in the paired
comparison (by more than any PASS audit's `delta_max` it relies on), with no transport-unverified
or gate-flagged policy involved and no gate demotion; its per-pair tests are unadjusted on
purpose (an intersection-union test; for many-pair audits use `compare_all_policies(adjust="bh")`).
It covers CJE's statistical checks only: write "decision-ready on CJE's checks, assuming <random
labels, one judge, blind raters>". A demotion is never a decision, and `significant` (only
`p_value < alpha`) never overrides hard rule 5.

`overall_status` GOOD and unflagged gates mean the estimator's diagnostics passed, not that
labels were random, one judge scored every row, or a decision is supported; never write "ready
for decision" or "no data-quality problem" from them. Boundary cards are OK / CAUTION /
REFUSE-LEVEL: a scalar badge never shows that a ranking survives, and a REFUSE-LEVEL policy gets
no absolute quality number. In a calibrated run with a flagged gate, still surface the highest
point estimate with its limitation adjacent and never substitute another policy (hard rules 1
and 5 still apply). After a transport `FAIL`, base no
decision on the unchanged map; a failed level audit alone does not disprove a ranking
(**Correction** in `reference.md`).

## One sample: calibrated mean with CI

```python
from cje import calibrated_mean_ci

result = calibrated_mean_ci(judge_scores, oracle_labels)  # NaN in oracle_labels = unlabeled
print(result.summary())
```

Pass `cluster_ids` when there are multiple responses per prompt. Check that `result.calibrator`
is not `None` before reusing it; complete coverage fits none. Full signature in `reference.md`.

## Reusing a calibrator

Refit from retained inputs any two-stage calibrator saved before 0.8.0 or of unknown
provenance. Before relying on a calibrator for a new period, domain or policy family, audit it
with held-out, probability-sampled probes (at least 20 effective independent clusters), via
`TransportAuditConfig` in `analyze_dataset` or `transport_audit` (code in `reference.md`).
Probes must be labels the calibrator was not fit on; give every record an `observation_id` so
`analyze_dataset` raises on overlap. Ask the user for the margin
`delta_max` before seeing probe results, in the units of `results.estimates`; with none, report
`NOT_GRADED`. `PASS` = the simultaneous residual CI is wholly inside the margin; `FAIL` = wholly
outside (graded even below 20 clusters); otherwise `INCONCLUSIVE`. A policy without a probe is
`NOT_CHECKED`: an unresolved assumption, never a pass and never an observed failure.

## Labeling loop: when labels are missing or short

Drive it yourself. To compare policies, pick 20+ prompts by simple random sampling and label
every policy's response on them (4 variants × 20 = 80 ratings; default `label_design`),
including variants the judge scores low: judge bias can make a low-scored variant the true
best. Give raters the response only (no policy name, no judge score), shuffled. For one policy,
or if you stratify, sample at random within that policy's judge-score strata and declare
unequal rates with `label_design="known_propensity"`; covering the score range prevents
REFUSE-LEVEL later, and labeling only the top-scored items is the classic mistake. Ground truth
= human judgment, expert review, or a downstream KPI; a trusted stronger model can serve, and
the estimate then targets that model's judgment, so say so. 20 per policy is a floor: the rerun
may still say "not established". Labels may all sit in one policy, but then hard rule 5
applies. Then run the canonical flow.

## Save an auditable run

Leave an executable analysis script and an output directory, not only a chat summary: inputs
or their locations with checksums, the label-selection design, judge and rubric versions,
excluded records, Python and `cje-eval` versions, seeds, declared scales, `results.to_dict()`
or `plan.to_dict()`, and every warning and gate state. Full list in `reference.md` §Audit
exports.

## Pitfalls

| Pitfall | Instead |
|---|---|
| Putting every label on one policy when each could be labeled | Label a random slice of each (routes become `augmented`); with many policies, pool labels and grade the transfer with held-out probes |
| Reusing last month's calibrator silently | Held-out transport audit with an explicit margin |

Full signatures, CSV/pandas recipes, `fresh_draws_dir`/CLI usage, planning API, diagnostics
glossary, and troubleshooting: read `reference.md` (`cje skill --reference`).
