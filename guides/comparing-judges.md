# Comparing judges

Use `compare_judges` when several candidate judges score the same responses and one outcome is labelled on a shared, random subset of them. The question it answers is which judge to use for this outcome and how many labels each one saves. That is a different question from comparing policies scored by one judge.

Do not pass the judges to `analyze_dataset` as if they were policies. It fits one calibrator to every policy's (score, label) pairs and detects the judge-score scale jointly across policies, so neither judge gets its own calibration map. Both "policies" estimate the mean of the same outcome on the same rows, so `compare_policies` reports a difference of about zero by construction, which says nothing about which judge is better.

## Example

```python
import numpy as np
from cje import compare_judges

rng = np.random.default_rng(1)
n = 2000
sigmoid = lambda x: 1 / (1 + np.exp(-x))
quality = rng.normal(size=n)                                  # latent quality
outcome = (rng.uniform(size=n) < sigmoid(3 * quality - 0.5)).astype(float)
lenient = sigmoid(1.2 * (quality + rng.normal(0, 0.9, n)) + 2.0)   # scores on [0, 1]
strict = np.clip(np.round(5.5 + 2 * (quality + rng.normal(0, 0.5, n))), 1, 10)  # 1-10
labelled = rng.choice(n, 400, replace=False)                  # one random slice
labels = np.full(n, np.nan)                                   # NaN = unlabelled
labels[labelled] = outcome[labelled]
prompt_ids = [f"p{i}" for i in range(n)]

cmp = compare_judges(
    {"lenient": lenient, "strict": strict},  # each judge on its own scale
    labels,                                  # one outcome, shared by both judges
    prompt_ids,                              # required: fold and resampling unit
    judge_scales={"lenient": (0, 1), "strict": (1, 10)},
    n_unlabeled=1600,                        # unlabelled rows in the planned run
    n_bootstrap=100,                         # default 2000; kept small here
)
print(cmp.summary())
```

Each judge gets its own cross-fitted `JudgeCalibrator`, chosen as in `calibrated_mean_ci` (two-stage with covariates, otherwise auto-selected). The folds are a seeded hash over the labelled prompt ids, so both judges are cross-fitted on the same folds and their out-of-fold predictions are paired. There is no scale detection across judges. Calibration does not change under increasing affine rescaling, so a declared scale only checks that each judge's scores lie in its range.

## Reading the result

- `cmp.table` has one row per judge: labelled rows and prompt clusters, the selected calibration mode, out-of-fold RMSE, out-of-fold R² pooled and within policy, and the variance components `Var(f)` (calibrated prediction) and `Var(Y - f)` (out-of-fold residual).
- `cmp.pairwise` compares each judge J with the reference (the first judge unless `reference=` says otherwise). It reports the differences in within-policy R² and RMSE, and the label multiplier `(1 - R²_ref) / (1 - R²_J)`. The multiplier is the number of labels the reference needs per label of J for the same interval width on a policy mean, when unlabelled rows are plentiful. Above 1, J saves labels.
- With `n_unlabeled`, two finite-N fields appear. `variance_ratio_at_n` is the predicted ratio of `Var(f)/N + Var(Y - f)/n` between the two judges, at the observed labelled rows per policy `n` and `N = n + n_unlabeled`. `label_multiplier_at_n` is the multiplier at that `N`. It is capped where the reference would have to label every row.
- Every interval comes from a paired prompt-cluster bootstrap. Each replicate draws one positive weight per prompt cluster, shares it across judges, and refits every judge with its selected mode. A judge's row does not depend on which other judges are in the call. `cmp.calibrators` holds the full-sample fits for `transport_audit`, and `cmp.to_dict()` is JSON-safe.

The R² that sets label savings is computed within a policy. With `policy_ids`, an R² pooled over policies also credits a judge for tracking which policy produced a response, and the per-policy correction cannot use that. `r2_pooled` is reported for contrast, but the multiplier uses `r2_within`.

## Assumptions and limits

- The labels are a representative random sample of the rows, the same rows for every judge. Stratified, oversampled or targeted labels are not supported.
- The multiplier concerns policy means (levels) corrected at weight one, not differences between policies.
- Labelled rows are treated as independent. With several labelled rows per prompt and judge errors shared within a prompt, the multiplier in labelled prompts can differ from this row-level ratio. The intervals still resample prompts.
- One calibrator per judge is fitted across all policies, as `analyze_dataset` fits one.
- Folds hash the cluster-id strings, so results move with the seed and with how the ids are spelled. To reproduce a judge's row with `calibrated_mean_ci` or `JudgeCalibrator.fit_cv`, pass identical `cluster_ids` (or None in both) and the same seed.
- Choosing the best of many judges on the same labels flatters the winner. The intervals hold for each pair separately, not for the selection.
- The default 2,000 replicates mean 2,000 calibrator fits per judge: about 40 seconds for two judges at 800 labels.
