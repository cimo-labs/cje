# Measured fixed-pool results

100 repetitions; six conditions; 43.67 seconds. Fresh namespace 20261024. No model/API calls.

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

The fixed-pool true premium−base difference is +0.00927. The differential flagged-only mean estimated difference is −0.16114; all 100 estimates change sign. The sign change is specific to the predeclared stress mechanism, not evidence about observed failure reports.

Known-propensity symmetric intervals are about 1.85× wider for the base mean and 1.93× wider for the paired difference than random intervals. This is a result to discuss with the maintainer, not an attributed library defect.

For symmetric flagged-only samples, the custom score-quintile screen passes for 99% of base repetitions despite mean bias −0.23936. Full chi-square statistics and contingency tables are in results/runs.json.

All 100 sampled masks are distinct in each condition/policy. Restricted support and the targeted calibration route explain why low variation should not be interpreted as general precision; the phrase near-identical does not mean identical masks.

Acceptance checks: 600 estimator results; independent arithmetic recomputation for all 18 summary rows; 60-label and 600-row contingency margins; matched symmetric masks across the two analysis routes; four-field-only data; paired prompt order. All passed. Notebook schema validated. All code cells executed in IPython in-process; all 600 result records matched the original run exactly. A separate Jupyter kernel could not start because this environment forbids TCP and IPC socket binding.

AI-assisted implementation and analysis. No independent human review. The original 200-repetition pilot was preserved.
