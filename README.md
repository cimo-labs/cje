<div align="left">
  <img src="https://raw.githubusercontent.com/cimo-labs/cje/main/images/CJE_logo.jpg" alt="CJE Logo" width="250">
</div>

# CJE: Causal Judge Evaluation

**Reuse an informative judge and available outcome labels to reduce the labeling needed for policy evaluation.** Because raw judge scores can be biased, CJE calibrates them against ground-truth labels, estimates policy means and paired differences, and reports uncertainty under explicit sampling and calibration-reuse assumptions. The savings come from applying that calibration to many responses and, where supported, across policies or evaluation cycles.

[![arXiv](https://img.shields.io/badge/arXiv-2512.11150-b31b1b.svg)](https://arxiv.org/abs/2512.11150)
[![Dataset](https://img.shields.io/badge/HF-Dataset-yellow)](https://huggingface.co/datasets/elandy/cje-chatbot-arena)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_core_demo.ipynb)
[![Docs](https://img.shields.io/badge/docs-cimolabs.com-blue)](https://cimolabs.com/cje)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)](https://www.python.org/downloads/)
[![Tests](https://github.com/cimo-labs/cje/actions/workflows/ci.yml/badge.svg)](https://github.com/cimo-labs/cje/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/cimo-labs/cje/blob/main/LICENSE)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/cje-eval?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/cje-eval)

## 60 seconds

```bash
pip install cje-eval   # Python 3.10–3.13
```

**Upgrading?** Read the [CHANGELOG](https://github.com/cimo-labs/cje/blob/main/CHANGELOG.md) first. Two-stage calibrators saved before 0.8.0 need refitting from retained inputs.

You need, for each **policy** (model, system prompt, or agent version) you compare: its responses to a shared set of prompts, one `judge_score` per response from **one fixed LLM judge** (same rubric for every policy), and ground-truth labels (`oracle_label`: human ratings, expert reviews, or observed outcomes) on a **randomly sampled** subset of responses. Each record is one judged response: `{"prompt_id", "judge_score", "oracle_label" (optional)}`; the API calls these records *fresh draws*. Responses that share a `prompt_id` are paired across policies and treated as one cluster for uncertainty, so reuse the same IDs. Judge and label scales may differ (0–1, 0–100, Likert); calibrated estimates come back on the label scale.

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
print(results.summary())

# Is candidate better? Test the paired difference on the shared prompts;
# don't compare the two intervals by eye.
for c in results.compare_all_policies():
    print(f"{c['policy1']} - {c['policy2']}: {c['difference']:+.3f}  "
          f"95% CI [{c['ci_lower']:+.3f}, {c['ci_upper']:+.3f}]  p={c['p_value']:.2f}")
```

```text
CJE Estimation Results (method: calibrated_direct)
  candidate   0.824  95% CI [0.766, 0.882]
  production  0.786  95% CI [0.696, 0.876]
Best by point estimate: candidate
Limitations: residual transport NOT_CHECKED
Status: warning
candidate - production: +0.038  95% CI [-0.027, +0.102]  p=0.22
```

`Best by point estimate` ranks point estimates; it is not a test. The paired interval includes 0 (p = 0.22), so these 20 synthetic prompts do not show that candidate is better. `compare_all_policies()` names each pair (`difference` = `policy1` − `policy2`; for many pairs add `adjust="bh"` and read `p_adjusted`; `p_value`, `significant` and the CIs stay unadjusted); `results.compare_policies(i, j)` takes integer indices into `results.target_policies`, which is sorted by name, not your dict's order.

`candidate` has no labels of its own, so it borrows production's calibration. Its interval covers prompt sampling and calibration-fit uncertainty, but not the chance that the calibration is off for candidate's responses; that is why it is narrower than production's, which its own 10 labels correct. This unaudited reuse, which the difference inherits, is what `NOT_CHECKED` marks: a [held-out audit](#guardrails-claims-cje-refuses-to-make) grades it, and labeling a random slice of candidate's own responses removes the reliance ([Your own data](#your-own-data)).

→ [Colab tutorial](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_core_demo.ipynb) on Chatbot Arena prompts (GPT-5 labels stand in for human ratings) · [API reference](https://github.com/cimo-labs/cje/blob/main/cje/interface/README.md#api-reference) · [Overview](https://cimolabs.com/cje)

### Your own data

Mark an unlabeled row with `oracle_label: None` or leave the field out. Records reject `NaN`; only the array API below uses `NaN` for unlabeled. From a CSV or DataFrame:

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
            "oracle_label": float(row["human_rating"]) if row["human_rating"] else None,
        })
results = analyze_dataset(fresh_draws_data=dict(draws))
```

With pandas, convert blanks before building records: `df = df.astype(object).where(df.notna(), None)`.

- **Where to put labels.** Comparing a few variants whose responses you can all rate? Label a random sample of each (start with 20 or more per variant, ideally on the same randomly chosen prompts). Each estimate is then corrected by its own labels (`metadata["point_estimator"]["routes"]` shows `augmented`), so a judge that favors one variant's style does not carry into the difference; the intervals are wider than with every label on one policy. Put all labels on one policy only when labeling the others is impractical.
- **Unequal sampling.** The default `label_design="representative"` treats each policy's labeled rows as a simple random sample of that policy. If sampling rates differ (for example, you oversampled some score ranges), pass `label_design="known_propensity", label_propensities={policy: probs, ...}`: for every policy in the call, an inclusion probability in (0, 1] for every row in input order, labeled or not. Otherwise the estimate and its interval can be biased, with no warning. Never hand-pick responses to label; CJE cannot detect it.

## Use CJE from your AI agent

You don't have to learn the API yourself. [`skills/cje/`](https://github.com/cimo-labs/cje/tree/main/skills/cje) teaches a coding agent to reshape eval data, plan labels, calibrate, compare, and report diagnostics. It's plain Markdown: any agent can use it, and agents with [skill support](https://github.com/cimo-labs/cje/tree/main/skills/cje#install) load it natively. Or just paste:

```text
Read https://raw.githubusercontent.com/cimo-labs/cje/main/skills/cje/SKILL.md,
then use CJE to compare the policies in my eval data. When it points to reference.md,
fetch https://raw.githubusercontent.com/cimo-labs/cje/main/skills/cje/reference.md.
```

## Is CJE the right tool?

| Your situation | Use |
|---|---|
| Rank/compare policies using an LLM judge, with some ground-truth labels | **CJE** |
| Estimate how a new model or prompt would score before shipping it (ahead of, not instead of, an A/B test) | **CJE**, when the outcome can be labelled offline (raters, experts): score its responses on prompts sampled from real traffic. This measures offline quality on those prompts, not online effects such as changes in user behavior. Its estimate stays `NOT_CHECKED` until held-out labels on its own responses grade the calibration reuse |
| One dataset, labels sampled from it, want a CI on its mean | `calibrated_mean_ci`, a prediction-powered mean ([array API](#the-array-api)) |
| Evaluate **many** policies without labeling under each | **CJE**. Labels pool across policies; audit that reuse with held-out probes. With only a few policies, label a random slice of each instead ([Your own data](#your-own-data)) |
| No ground-truth labels yet (or labels on fewer than 4 prompts) | **Label a random slice first.** CJE still runs, but returns only the raw judge mean, marked `naive_direct` / `UNCALIBRATED` and gate-`FLAGGED` |
| Predict how a *specific response* will score | Per-item prediction (e.g. conformal methods) |
| Estimate a policy by reweighting another policy's logged responses (importance sampling or doubly robust OPE) | The frozen `cje-eval==0.3.*` line ([why](#why-direct-mode-only-no-ipsdr)) |

## How it works

1. **Calibrate**: learn the judge → oracle mapping on the labeled slice (isotonic regression, unless the default auto mode's cross-validation prefers a two-stage variant that is isotonic in a learned index of the score, so it need not be monotone in the raw score; two-stage whenever covariates are used; cross-fitted by prompt).
2. **Evaluate**: average each policy's calibrated scores; when a policy has random labels of its own, correct that average by their mean residual (label minus calibrated score; the `augmented` route). Compare policies on the same prompts.
3. **Diagnose**: automatically report scalar score-range support, and optionally run a held-out residual equivalence audit with a predeclared practical margin.

Confidence intervals include finite-label calibration uncertainty on supported inference paths. Each estimate targets the policy's mean oracle label over the prompt population your prompts were sampled from, assuming (1) labels are a probability sample of the responses they describe, (2) the judge and rubric stay fixed, and (3) for a policy without labels of its own, the reused calibration has zero mean error on its responses (transport). CJE cannot check (1) or (2); it reports (3) as `NOT_CHECKED` until a held-out audit grades it. [Estimator details](https://github.com/cimo-labs/cje/blob/main/cje/estimators/README.md).

<div align="center">
  <img src="https://raw.githubusercontent.com/cimo-labs/cje/main/images/forest_plot_n1000_oracle25.png" alt="Forest plot of calibrated estimates with 95% CIs for four Chatbot Arena policies (base, clone, parallel_universe_prompt, unhelpful) next to mean labels on held-out responses; the unhelpful policy's estimate sits far above its held-out mean and its transport audit fails" width="80%">
  <br><em>The Chatbot Arena sample in <code>examples/arena_sample</code> (GPT-5 labels stand in for human ratings). The calibration is fit on base-policy labels; diamonds are mean labels on 50 held-out responses per policy (for base, on its own labels). The calibration overstates the deliberately unhelpful policy, and its transport audit fails. Made by <code>scripts/make_readme_forest_plot.py</code>.</em>
</div>

## Validation against reference labels

- **HealthBench Consensus (29,511 response–criterion records)**: a custom confidence-augmented regrade found judge overconfidence of 24.5 (gpt-4o-mini) and 13.0 (Claude Haiku 4.5) percentage points against strict positive physician majority, with ties coded not met (14.4 and 3.0 points, respectively, against mean physician agreement). One seeded retrospective replay exposed 5% of aggregate labels (1,454 endpoints, each based on 2–5 physician grades); calibrated estimates were within 1.4–2.1 points of the full aggregate endpoint. This was not prospective annotation or repeated-split validation. [Read the full audit →](https://cimolabs.com/research/healthbench-judge-audit)
- **Chatbot Arena (4,961 prompts, 5 policies; GPT-5 ratings stand in for human labels, so this is a model-reference study, not human validation)**: 99% pairwise ranking accuracy in the headline 5%-oracle configuration, and 94% averaged across configurations with a response-length covariate (92% without it, the `analyze_dataset` default). The arXiv v3 cost model gives a 14× reduction against full labeling under that reference. Nominal 95% intervals on raw judge means covered the reference mean 0% of the time; the calibration-aware intervals the library uses by default covered about 95% (93.9% Direct, 95.6% with the covariate) in a [corrected rerun](https://github.com/cimo-labs/cje-arena-experiments/blob/main/erratum_rerun/DELTAS.md) of the paper's experiments. A deliberately unhelpful policy, which the judge already ranks last but whose level the base-policy calibration overstates, fails the residual transport audit and is flagged. [Paper →](https://arxiv.org/abs/2512.11150)

**How many labels will a judge save you?** The 14× above comes from reusing one calibration across policies, which holds only if the calibration carries over. Labels that correct a policy's own estimate save less: the calibrated judge acts as a control variate, cutting the labels needed at equal precision by about 1 − Var(label − calibrated prediction) / Var(label) within that policy. That is at most the squared within-policy correlation between label and calibrated prediction, and at the default weight a calibration that fits the policy poorly can save nothing or even cost precision ([weight options](https://github.com/cimo-labs/cje/blob/main/guides/audit-correction.md#weight-the-prediction-or-not)). Agreement on easy, lopsided comparisons does not count. Before planning around savings, label a pilot of a few hundred random responses with the outcome you will actually report, measure that share, and size the budget with the [planning notebook](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_planning.ipynb); when it is below about 0.10, budget labels as if there were no judge and use the judge for triage and for ordering clear differences.

## Guardrails: claims CJE refuses to make

Diagnostics never act silently; every estimate ships with its limitations attached.

**Score-support badge (automatic).** Each policy gets a scalar badge checking only whether its judge scores extrapolate beyond the labeled score range (not residual bias, covariate shift, or ranking validity). When at least 5% of scores land outside it, the estimate carries `REFUSE-LEVEL`:

```text
REFUSE-LEVEL for policy 'candidate': 88.3% of fresh-draw judge scores fall
outside the oracle calibration range [0.161, 0.595]. Do not report level
(absolute) claims for this policy from this fit. Collect oracle labels covering
the missing score range.
```

The warning prints the range on CJE's internal 0–1 judge scale; `results.metadata["boundary_cards"][policy]["oracle_s_range"]` gives it in your judge's units. Expect the badge at small label counts: with m random labels, about 2/(m+1) of responses fall outside the labeled range by chance (in simulation it fired in about 90% of runs at 10 labels, 70% at 20 and 40% at 40). To clear it, label more responses at random within judge-score strata that include the extremes, declaring any unequal rates with `label_design="known_propensity"`; hand-picked out-of-range labels clear the badge but, treated as a random sample, bias the estimate.

**Residual transport audit (opt-in).** Reusing a calibration map on another policy, time period, or domain is an assumption. Grade it with held-out oracle probes that were not used to fit the calibrator (calibration labels reused as probes give a near-zero residual by construction; give draws and probes an `observation_id` and CJE rejects the overlap), plus a predeclared practical margin in the units of `results.estimates`:

```python
from cje import TransportAuditConfig

transport = TransportAuditConfig(
    probes_by_policy={"candidate": held_out_probe_rows},  # same record shape as draws, oracle_label filled
    delta_max_by_policy={"candidate": 0.03},  # OUTPUT units (units of results.estimates)
)
results = analyze_dataset(fresh_draws_data=draws, transport=transport)
print(results.metadata["transport_audits"]["candidate"]["status"])
```

`PASS` requires the simultaneous residual CI (Bonferroni across the audited policies) to lie wholly inside `[-delta_max, +delta_max]`; wholly outside is `FAIL`; overlap is `INCONCLUSIVE`; omitting the margin is `NOT_GRADED`. Fewer than 20 effective (Kish-weighted) prompt clusters withholds `PASS` but can still grade `FAIL`; a policy cannot escape a `FAIL` by supplying too small a probe. Policies without probes stay `NOT_CHECKED`. Among these audit states, only an observed `FAIL` hard-flags a policy whose estimate depends on that map; every other unresolved state remains visible as a limitation without suppressing the estimate. A `PASS` does not change any estimate or interval. For an already fitted calibrator, `cje.diagnostics.audit_transportability(results.calibrator, probe_rows, delta_max=...)` and its array twin `transport_audit(probe_scores, probe_labels, results.calibrator, delta_max=...)` run the same audit directly (pass `family_size=` the number of audited policies to get the same Bonferroni adjustment; their default is 1); they return a standalone diagnostic and do not flag `results`.

**Use audit labels for correction.** Probes only grade. To correct an estimate, attach probability-sampled labels to their matching evaluation responses (the `augmented` route) and use the recomputed intervals; those labels then no longer validate it independently ([audit-to-correction guide](https://github.com/cimo-labs/cje/blob/main/guides/audit-correction.md), with a runnable example).

**Plan audit labels before collecting them.** `plan_transport_audits` sizes independent audit probes and the total label budget under declared residual assumptions; it is a Gaussian planning model, not an observed audit ([audit budget guide](https://github.com/cimo-labs/cje/blob/main/guides/audit-budget-planning.md)).

**Reliability-aware winner.** `results.best_policy()` skips a gate-flagged argmax and returns the best gate-passing policy, loudly ([output and fields](https://github.com/cimo-labs/cje/blob/main/cje/interface/README.md#command-line-interface)); `reliable_only=False` returns the raw argmax, marked `flagged`. A demotion is not a comparison: no test is applied (with two policies, it simply returns the unflagged one). When labels come from one baseline policy, a better candidate whose judge scores exceed the labeled range is exactly what triggers the flag; clear it, then use `compare_all_policies()`.

## Levels, rankings, and production outcomes

A level claim concerns a policy's mean oracle outcome; a ranking concerns the oracle difference between two policies, which equals the calibrated-prediction difference plus the difference in their mean residuals. A common residual offset cancels, so a failed level audit does not by itself prove the ordering wrong. Conversely, individual-score monotonicity or a scalar support badge does not certify policy ordering under a shift. Use the paired comparison (`results.compare_all_policies()`, shown in the [quickstart](#60-seconds)) with evidence about the residual difference; CJE has no validated automatic label-free reuse gate.

Production outcomes can reduce **new annotation cost** when their meaning and response identity match the evaluation. Document how feedback was selected, the target population, dependence clusters, and any change in the judge or outcome process. Organic feedback is not automatically representative. Report incremental annotation cost separately from total label acquisition cost, and audit transport before relying on reuse across populations or time.

## The array API

`calibrated_mean_ci` is the library's bottom layer: a ppi_py-style primitive that takes NumPy arrays for one sample (judge scores on any bounded scale; labels in [0, 1] on an equal-probability random slice, `NaN` elsewhere) and returns a calibrated mean and confidence interval. It has no weights argument, so use `analyze_dataset` with `label_design="known_propensity"` for stratified labels, and for multi-policy comparisons. [Inference details, bootstrap, and calibrator reuse →](https://github.com/cimo-labs/cje/blob/main/cje/interface/README.md#array-api-calibrated_mean_ci)

```python
import numpy as np
from cje import calibrated_mean_ci

rng = np.random.default_rng(0)
scores = rng.uniform(size=400)                      # judge scores for every sample
labels = np.full(400, np.nan)                       # NaN = unlabeled
labeled = rng.choice(400, size=100, replace=False)  # oracle slice (25%)
labels[labeled] = np.clip(scores[labeled] + rng.normal(0, 0.1, size=100), 0, 1)

result = calibrated_mean_ci(scores, labels)
print(result.summary())
```

```text
Calibrated mean: 0.5316 (SE 0.0174, CI [0.4970, 0.5663], n=400, n_oracle=100, cluster_robust)
```

When partial oracle coverage requires calibration, `result.calibrator` predicts in the same public judge and oracle units supplied by the caller; complete oracle coverage returns the direct oracle mean with `result.calibrator is None`. Grade any fitted calibrator's reuse on an independent probe with `transport_audit(..., delta_max=<practical margin>)`; `result.diagnostics["boundary_card"]` carries the separate scalar score-support badge when calibration is fitted.

## Documentation

| Resource | Description |
|----------|-------------|
| **[Interactive Tutorial](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_core_demo.ipynb)** | Walk through a complete example in Colab; no setup required |
| **[Agent Skill](https://github.com/cimo-labs/cje/tree/main/skills/cje)** | Teach any coding agent to run CJE correctly |
| **[CJE in 3 Minutes](https://youtu.be/VbSYrby8iaQ)** | Video: why raw judge scores mislead and how CJE fixes it |
| **[Technical Walkthrough](https://youtu.be/r0dinGsPuqY)** | Video: calibration, evaluation, and transport auditing pipeline |
| **[Operational Playbook](https://github.com/cimo-labs/cje/blob/main/PLAYBOOK.md)** | End-to-end runbook: audits, drift correction, label budgeting |
| **[Migration Guide](https://github.com/cimo-labs/cje/blob/main/MIGRATING-0.6.md)** | Upgrading from 0.5.x or earlier: what changed and how to adapt |
| **[Planning Notebook](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_planning.ipynb)** | Choose sample sizes and an oracle-label budget from your judge's quality and per-call costs (no data needed), with optional pilot-data refinement |
| **[API Reference](https://github.com/cimo-labs/cje/blob/main/cje/interface/README.md)** | `analyze_dataset()` parameters, `EstimationResult` methods, the `cje analyze` / `cje validate` CLI |
| **[Website](https://cimolabs.com/cje)** | Overview, when CJE fits, research notes |

**Bridges:** Use the [Langfuse experiment bridge](https://github.com/cimo-labs/cje/blob/main/scripts/langfuse_cje/README.md) to preserve response identity and label provenance before analysis. Already running evals in [Promptfoo, TruLens, LangSmith, or OpenCompass](https://github.com/cimo-labs/cje/blob/main/scripts/cje_bridges/README.md)? Convert those outputs into CJE format with one command.

**Module deep dives:** [Calibration](https://github.com/cimo-labs/cje/blob/main/cje/calibration/README.md) · [Diagnostics](https://github.com/cimo-labs/cje/blob/main/cje/diagnostics/README.md) · [Estimators](https://github.com/cimo-labs/cje/blob/main/cje/estimators/README.md) · [Interface/API](https://github.com/cimo-labs/cje/blob/main/cje/interface/README.md) · [Data formats](https://github.com/cimo-labs/cje/blob/main/cje/data/README.md)

## Why Direct mode only (no IPS/DR)?

CJE is **Direct-mode only**: fresh draws, calibrated judge, audits. There is no off-policy machinery: no importance-sampling or doubly-robust estimators (`calibrated-ips`, `dr-cpo`, `mrdr`, `tmle`, `stacked-dr`), teacher forcing, SIMCal weight stabilization, or overlap diagnostics. Our own paper's results drove that design: for realistic LLM policy pairs, importance weighting failed even when ESS looked healthy (target-typicality coverage 0.19–0.49, far below the 0.70 gate), and the best DR stack merely matched Direct mode's accuracy at ~12× the compute.

- **Need IPS/DR from logged propensities?** Pin the frozen OPE line: `pip install "cje-eval==0.3.*"` (maintained on the `0.3.x` branch; docs at the `v0.3.0` tag; requires Python <=3.12).
- **Have old logged data with `judge_score` + `oracle_label`?** It works as the calibration source: `analyze_dataset(fresh_draws_dir=..., calibration_data_path="logged.jsonl")`. Its values are read as [0, 1] unless you declare `calibration_judge_scale=(lo, hi)` / `calibration_oracle_scale=(lo, hi)`.

Full version history in the [CHANGELOG](https://github.com/cimo-labs/cje/blob/main/CHANGELOG.md); two-stage calibrators saved before 0.8.0 need refitting from retained inputs.

## Development

```bash
git clone https://github.com/cimo-labs/cje.git
cd cje && poetry install && make test
```

## Citation

If you use CJE in your research, please cite:

```bibtex
@misc{landesberg2025causaljudgeevaluationcalibrated,
  title={Causal Judge Evaluation: Calibrated Surrogate Metrics for LLM Systems},
  author={Eddie Landesberg and Manjari Narayan},
  year={2025},
  eprint={2512.11150},
  archivePrefix={arXiv},
  primaryClass={stat.ME},
  url={https://arxiv.org/abs/2512.11150},
}
```

## License

MIT. See [LICENSE](https://github.com/cimo-labs/cje/blob/main/LICENSE) for details.
