# Results on the fixed prompt pool

The runner's default is 100 repetitions across six conditions. Results below vary label selection on one fixed pool of 600 matched prompts; they do not estimate population-level performance. No model calls are made by the runner.

|Condition|Target|Bias|RMSE|Coverage|Mean CI half-width|Quintile pass|
|---|---|---:|---:|---:|---:|---:|
|random|base|-0.00146|0.02639|94%|0.05828|96%|
|random|premium-base|+0.00094|0.03323|99%|0.07782|—|
|flagged_symmetric|base|-0.23936|0.24001|0%|0.04736|99%|
|flagged_symmetric|premium-base|+0.00094|0.02557|99%|0.05838|—|
|targeted_unknown|base|-0.23523|0.23554|0%|0.04151|99%|
|targeted_unknown|premium-base|-0.00617|0.00630|100%|0.01060|—|
|known_propensity_symmetric|base|+0.00332|0.02245|100%|0.10785|99%|
|known_propensity_symmetric|premium-base|+0.00010|0.03007|100%|0.15004|—|
|flagged_differential|base|-0.14381|0.14505|0%|0.05063|99%|
|flagged_differential|premium-base|-0.17040|0.17200|0%|0.05906|—|
|known_propensity_differential|base|+0.00007|0.01940|100%|0.08632|100%|
|known_propensity_differential|premium-base|+0.00226|0.03783|100%|0.13858|—|

The fully labeled pool's true premium-minus-base difference is +0.00927. Under the predeclared differential flagged-only stress condition, the estimated difference has the opposite sign in all 100 repetitions. Its paired confidence interval is entirely below zero in every repetition (largest upper bound −0.037547), and `compare_policies` marks the difference significant with no gate flagged. In my rerun, the overall status still reads `GOOD` in 38 of 100 repetitions. This is a constructed oracle-assisted stress case, not evidence about observed failure-report behavior.

The symmetric known-propensity intervals are about 1.85 times wider for the base mean and 1.93 times wider for the paired difference than random sampling. CJE's variance calculation uses each row's inclusion probability and gives no credit for fixed counts within strata. Since the strata here are built from oracle residuals, the intervals are valid but conservative; this is not attributed as a library defect. With independently sampled labels at the same probabilities, the maintainer found coverage of 94–96%, with half-widths within 5% of RMSE, while RMSE more than doubled.

The score-quintile chi-square is a custom diagnostic, not a built-in CJE gate. In the symmetric flagged-only base condition it passes in 99% of repetitions despite bias of −0.23936. Passing this check is failure to detect unequal score-bin label rates, not evidence that outcome-dependent selection is absent. Full contingency tables and run-level results are written to `results/runs.json` when you run it.
