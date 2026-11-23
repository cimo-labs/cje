# Weighting a paired audit of a fixed traffic frame

Use `weighted_paired_audit` when you can enumerate the target prompts, users or documents and know exactly how they were sampled. One row represents one independent cluster. First sample clusters for traffic scoring, then sample from those clusters for paired human review. Each draw must be Bernoulli, independent across clusters. Both policies are judged and, when selected for human review, labeled together. The target is a prespecified weighted average of the paired outcomes over **this frame**.

Historical calibration labels may have been collected under unequal probabilities. A separate fit can weight them by their inverse acquisition probabilities. In the current frame, each cluster has a traffic inclusion probability and a conditional label inclusion probability. Optional known target masses specify the frame average you want. For example, if the target is two groups with fixed shares, divide each group's target share by its total frame count to get its per-cluster mass.

This function takes a fixed paired prediction as input, so it can report an interval even when very few clusters receive human review. If you only have selected click logs, recover the assignment frame and the sampling probabilities before using it. With a known frame and bounded outcomes, missing human reviews can still leave informative limits on the mean even when their inclusion probabilities are unknown. [Rater panels, judge timeouts and missing-outcome bounds](rater-and-missing-outcomes.md)

## A complete synthetic example

The example compares two policies on the same prompts. The outcome is the challenger-minus-incumbent human pass difference. The independent historical recruitment deliberately favors disputed items. Its sampling probabilities and the current audit probabilities come from the respective randomization code.

```python
import numpy as np

from cje.calibration import JudgeCalibrator
from cje.weighted_audit import weighted_paired_audit

rng = np.random.default_rng(27)

# Separate historical experiment. Fit once, before drawing the current audit.
score = rng.uniform(size=900)
disputed = rng.random(900) < 0.25
historical_label_probability = np.where(disputed, 0.8, 0.2)
historical_chosen = rng.random(900) < historical_label_probability
historical_y = rng.random(900) < np.clip(0.1 + 0.65 * score + 0.15 * disputed, 0, 1)
calibrator = JudgeCalibrator(calibration_mode="monotone")
calibrator.fit_cv(
    score[historical_chosen],
    oracle_labels=historical_y[historical_chosen].astype(float),
    prompt_ids=[f"historical-{i}" for i in np.flatnonzero(historical_chosen)],
    sample_weight=1 / historical_label_probability[historical_chosen],
    quiet=True,
)

# Known design for every current cluster. Each cluster has one paired output.
n = 600
group = np.arange(n) % 2
q = np.where(group == 0, 0.8, 0.45)  # Probability of a scored traffic pair
p = np.where(group == 0, 0.4, 0.7)   # Probability of a labeled pair GIVEN scoring
traffic = rng.random(n) < q
audited = traffic & (rng.random(n) < p)

# Synthetic potential responses for the example; estimation sees only the masks.
base_score = rng.uniform(size=n)
candidate_score = np.clip(base_score + np.where(group == 0, 0.12, -0.08), 0, 1)
base_pass = rng.random(n) < 0.1 + 0.65 * base_score
candidate_pass = rng.random(n) < 0.1 + 0.65 * candidate_score
pred = np.full(n, np.nan)
pred[traffic] = (
    calibrator.predict(candidate_score[traffic]) - calibrator.predict(base_score[traffic])
)
out = np.full(n, np.nan)
out[audited] = candidate_pass[audited].astype(float) - base_pass[audited]

audit = weighted_paired_audit(
    cluster_ids=np.arange(n),
    traffic_selected=traffic,
    label_selected=audited,
    traffic_inclusion=q,
    label_inclusion=p,
    predictions=pred,
    outcomes=out,
    outcome_bounds=(-1, 1),
    target_weights=np.where(group == 0, 3, 1),  # 75/25 across the equally sized groups
)
print(audit.estimate, audit.ci, audit.outcome_identification_bounds)
print(audit.n_labeled, audit.label_effective_n)
```

Pass one row per **frame** cluster, including unselected rows. Give each nonselected field a missing value. A judge timeout or unavailable human rating on a selected row is an observation failure. Correct the missingness design or define the endpoint before estimating. Quota sampling without replacement, voluntary responses, estimated traffic shares, and repeated decisions about whether to keep sampling require different uncertainty calculations. Inclusion probabilities inferred from observed label rates cannot establish how an acquisition process worked. If a current label probability depends on a score, the input still requires the true conditional probability for every potential cluster, even those skipped at stage one.

## Point estimate and interval

For cluster `i`, let `w` be its fixed mass normalized across the full frame, `q` its traffic probability, `p` its conditional label probability, `m` its fixed paired prediction and `y` its fixed paired outcome. Let `I` indicate a scored pair and `R` a pair labeled after scoring. Sum the following contributions over clusters.

```text
target contribution     w y
estimated contribution  w [ I m/q + I R (y − m)/(q p) ]
```

Since E[I]=q and E[IR]=qp, the estimate is design-unbiased for the weighted frame mean even if the fixed prediction is biased. Calibration weights and a good predictor may improve precision. Conditioning on an independently trained map allows any such fit, including a poor one. Using these same audit labels to fit a predictor breaches that independence. For two [0,1] outcomes, a paired difference has bounds [−1,1]. A paired indicator that the candidate introduces a failure has bounds [0,1] and requires its own auxiliary prediction. If you want a direct estimator, use a fixed zero prediction when zero is inside your declared bounds.

The design variance of each unweighted contribution is `(1−q)y²/q + (1−p)(y−m)²/(qp)`. Write `M=max(|lo|,|hi|)` and `S=hi−lo` for the declared support. Bound variance over the frame by summing `w²[(1−q)M²/q + (1−p)S²/(qp)]`. A bound on the largest absolute centered contribution comes from the three possible outcomes of the sampling design:

| Sampling outcome | Bound for the absolute centered weighted contribution |
|---|---|
| No traffic, possible when q<1 | wM |
| Traffic only, possible when p<1 | w[(1−q)M/q + S/q] |
| Both stages | w[(1−q)M/q + (1−p)S/(qp)] |

Using the resulting maximum B and variance bound V, the two-sided Bernstein radius is `B x/3 + sqrt(2 V x + (B x/3)²)`, where `x=log(2/alpha)`. This covers with at least the requested probability over repeated sampling under the fixed design and specified support, at a single predeclared look. Bounds use design probabilities for **all** frame clusters. [Bernstein's inequality for bounded variables](https://academic.oup.com/book/26549/chapter-abstract/195070651)

Observed labels also give an always-valid interval. Start with their weighted sum and assign every remaining cluster the lower or upper outcome endpoint. CJE intersects this interval with the concentration interval when the intersection is nonempty, falling back to the observed-outcome bounds if they are disjoint. If no labels were observed the interval is the full declared support. If every cluster was labeled, the interval is the known frame mean. The raw design statistic can fall outside these bounds in a particular sample, especially under large inverse weights. Its reported value stays untruncated to retain its sampling expectation.

The interval can be wide. A replay of a public WMT translation comparison sampled 170 documents with unequal probabilities at both stages, averaging about 52 paired document reviews. The underlying rate of new major errors was 8.04%. On 20,000 samples the design interval covered in every replay, with mean full width 48.9 percentage points. Bounds using the observed labels alone averaged 69.1 points. An ordinary estimated-variance normal interval in this design averaged 10.8 points wide and covered 89.0% of the time. These are properties of that retrospective sampling experiment. The package reports an interval for this fixed-frame sampling target. The [replay regression test](../cje/tests/test_weighted_paired_audit.py) retains the 170 documents as 42 groups with identical sampling inputs and outcomes. [WMT24 evaluation](https://aclanthology.org/2024.wmt-1.2/)

`traffic_effective_n` and `label_effective_n` summarize concentration of the realized weights w/q and w/(qp). Their definition is the squared sum of weights divided by the sum of squared weights. They supplement raw counts, labeled target mass and the minimum joint inclusion probability. A large effective count among observed clusters can coexist with poor representation of a rare group.
