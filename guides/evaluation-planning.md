# How many labels will a judge save you?

A calibrated judge saves labels in two different ways, and they are worth very different amounts. Measure which one applies before planning a budget around it.

## Reusing one calibration across policies

The Chatbot Arena study's 14× lower total cost than full labeling (the arXiv v3 cost model, oracle labels plus judge calls, with GPT-5 ratings standing in for human labels) is 8.8× for a single policy; fitting one calibration and applying it to policies that have no labels of their own raises it to 14×. That extra saving holds only if the calibration carries over to those policies' responses. CJE reports the reuse as `NOT_CHECKED` until held-out labels on each such policy grade it; [plan that audit](audit-budget-planning.md) before relying on the saving.

## Correcting a policy's own estimate

Labels that correct a policy's own estimate (the `augmented` route) save less. The calibrated judge acts as a control variate, cutting the labels needed at equal precision by about

    1 − Var(label − calibrated prediction) / Var(label)

within that policy. That share is at most the squared within-policy correlation between label and calibrated prediction. At the default weight, a calibration that fits the policy poorly can save nothing or even cost precision ([weight options](audit-correction.md#weight-the-prediction-or-not)). Agreement on easy, lopsided comparisons does not count toward it.

## Measure before you plan

1. Label a pilot of a few hundred random responses with the outcome you will actually report.
2. Measure the share above on the pilot.
3. Size the budget with the [planning notebook](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_planning.ipynb), or in code with `fit_variance_model`, `plan_evaluation` and `plan_for_mde` ([reference](../skills/cje/reference.md#planning-how-many-labels-do-i-need)).

When the share is below about 0.10, budget labels as if there were no judge. Use the observed paired comparison and its gates to assess the ordering.

Planned power is a projection under the planner's assumptions, not achieved power; after collection, read the realized intervals and paired comparisons.

## Plan reviews of a binary judge

A judge that passes 99% of responses has a 60.5% chance of yielding zero judge failures in 50 randomly selected responses. Use random sampling *within* the observed fail and pass groups to measure the difference in their human pass rates (the calibration slope). To measure the overall human pass rate, the two group means instead contribute in proportion to their traffic shares.

```python
from math import sqrt
from cje import plan_binary_reviews

plan = plan_binary_reviews(
    objective="slope", judge_fail_share=0.01, budget=500, target_se=0.05,
    fail_sd=sqrt(0.6 * 0.4), pass_sd=sqrt(0.98 * 0.02),
    available_fail=150, available_pass=1000,
)
print(plan.fail_reviews, plan.pass_reviews, plan.meets_target)  # 124 35 True
```

This assumes a human pass rate of 60% among judge failures and 98% among passes to set the within-group SDs. With a 50-review budget, those assumptions assign 39 reviews to failures for the slope and two for the mean. A target SE of 0.05 denotes five points of standard error. At a 1% judge-fail rate, finding 124 eligible failures takes 12,400 scored responses in expectation. Supply real availability caps before recruiting. If a feasible quota cannot reach the target within budget, the result reports the best modeled SE available and `meets_target=False`. A stratum with fewer than two available clusters cannot support this design.

The units are independent prompts or other prespecified independent clusters, with one bounded human outcome in [0,1] per unit. Each review has the same cost. Use the defaults of 0.5 for conservative SDs when prior evidence is weak. Randomize within each stratum and retain the selection mechanism. The calculation treats the traffic share as known and targets future independent units. Analyze fixed review quotas with a stratified variance calculation. If each cluster entered traffic and then human review by independent Bernoulli draws across a complete frame, follow the [weighted paired audit](weighted-paired-audit.md).
