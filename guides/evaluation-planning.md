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

When the share is below about 0.10, budget labels as if there were no judge, and use the judge for triage and for ordering clear differences.

Planned power is a projection under the planner's assumptions, not achieved power; after collection, read the realized intervals and paired comparisons.
