# Plan the labels needed for a transport audit

An audit can remain inconclusive even when the fitted calibration map is useful. Before acquiring labels, choose a practical residual margin and check whether the audit can resolve it with the independent units available.

```python
from cje import AuditScenario, CostModel, plan_transport_audits

plan = plan_transport_audits(
    {
        "baseline": AuditScenario(0.10, 0.05, available_clusters=130),
        "candidate": AuditScenario(0.20, 0.05, available_clusters=130),
    },
    calibration_labels=40,
    cost_model=CostModel(oracle_cost=2.0),
    budget=1000,
    power=0.80,
    alpha=0.05,
)
print(plan.summary())
```

Under these hypothetical assumptions, the baseline needs 63 independent audit units and the candidate needs 245. The complete label count is 40 calibration plus 308 audit labels, costing 696 at 2 per label. The monetary budget is sufficient, but the candidate has only 130 available units. Its modeled probability of passing with those units is about 44%.

These numbers are a planning calculation, not evidence that either calibration map transports. No observed audit is run and no `PASS` status is produced.

## Choose defensible inputs

- `residual_sd` is the standard deviation of independent cluster-level residuals, in the reported oracle units. It is not a standard error. A pilot estimate can be unstable; examine several plausible values.
- `delta_max` is the largest acceptable absolute mean residual. Choose it for the decision before inspecting audit outcomes.
- `assumed_bias` defaults to zero. Plan sensitivity to plausible drift. A mean at or beyond the margin with positive residual variance cannot attain the requested power by acquiring more labels.
- `available_clusters` counts eligible independent units after excluding calibration units. Repeated ratings or several outputs from one prompt do not create additional independent units.
- `labels_per_cluster` counts the underlying human ratings needed per independent unit. For example, an equally weighted document outcome that averages several segment ratings still costs all those ratings. A fractional value represents an anticipated mean cost.
- `calibration_labels` includes acquired calibration and pilot ratings whose cost belongs to the evaluation. Reusing an existing calibration set does not make those ratings free in the total-cost comparison.

The cost scope is human ratings only, using `CostModel.oracle_cost`. Add judge execution and other costs separately. Shared prompts across policies can reduce variance, but each separately rated response still consumes a label.

## What the model calculates

For independent, identically distributed Gaussian cluster residuals, the sample mean and sample variance are independent. The planner integrates the probability that the two-sided Student t interval lies wholly inside the practical margin, including the randomness of the sample variance. It finds the smallest integer sample size within the requested search bounds that reaches the modeled power.

`power` targets **all supplied audits passing**. For K policies, each is planned at `1 - (1 - power) / K`. A union bound gives the requested family power without requiring audits to be independent across policies. Each audit interval uses `alpha / family_size`, matching the Bonferroni adjustment in `transport_audit`. By default the family contains every supplied policy. If a larger family is declared, omitted members receive neither a power guarantee nor a cost allocation from this call.

The default minimum of 20 independent units matches the observed audit's support floor. Availability is a feasibility check; the calculation does not apply a finite-population correction. Weighted audits and unequal cluster contributions require a justified sampling model beyond this calculator. Estimated variance, nonnormal residuals, clustering errors, or drift can make achieved power differ from the plan.

If the search limit is too small, `required_clusters` is `None` with reason `search_limit_reached`. If the assumed mean prevents the power target, the reason is `assumed_mean_outside_margin`. An unresolved policy makes the total required budget unknown. It is not reported as a finite affordable plan.

Use `plan.to_dict()` to retain the assumptions, requirements, availability checks, and cost scope alongside the eventual observed audit.
