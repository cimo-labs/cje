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

- `cmp.table` has one row per judge: labelled rows and prompt clusters, the selected calibration mode, out-of-fold RMSE, out-of-fold R² pooled and within policy, the variance components `Var(f)` (calibrated prediction) and `Var(Y - f)` (out-of-fold residual), and their prompt-clustered versions `var_f_clustered`, `var_residual_clustered` and `cov_f_residual_clustered`, which the multipliers use. The clustered versions sum the rows of each prompt within a policy before squaring, so they equal `Var(f)` and `Var(Y - f)` when every prompt has one row per policy.
- `cmp.pairwise` compares each judge J with the reference (the first judge unless `reference=` says otherwise). It reports the differences in within-policy R² and RMSE, and the label multiplier `var_residual_clustered[ref] / var_residual_clustered[J]`: the labelled rows the reference needs per labelled row of J for the same interval width on a policy mean, when unlabelled rows are plentiful. Above 1, J saves labels. With one labelled row per prompt and policy it is `(1 - R²_ref) / (1 - R²_J)`.
- With `n_unlabeled`, two finite-N fields appear. `variance_ratio_at_n` is the predicted ratio of `V = max(Var(f) + 2 Cov(f, Y - f), 0)/N + Var(Y - f)/n` (clustered components) between the two judges, where `n` is the mean labelled rows per labelled policy and `N = n + n_unlabeled`. The covariance term is there because the estimate averages `f` over all rows and the residual over the labelled rows among them. `label_multiplier_at_n` is the multiplier at that `N`. It is capped where the reference would have to label every row. With `n_unlabeled=0` every row is labelled, the estimate is the label mean whatever the judge, and both fields are 1.
- Every interval comes from a paired prompt-cluster bootstrap. Each replicate draws one positive weight per prompt cluster, shares it across judges, and refits every judge with its selected mode. A judge's row does not depend on which other judges are in the call. `cmp.calibrators` holds the full-sample fits for `transport_audit`, and `cmp.to_dict()` is JSON-safe.

The R² that sets label savings is computed within a policy. With `policy_ids`, an R² pooled over policies also credits a judge for tracking which policy produced a response, and the per-policy correction cannot use that. `r2_pooled` is reported for contrast, but the multiplier uses the within-policy residual variance, as `r2_within` does.

## Assumptions and limits

- The labels are a representative random sample of the rows, the same rows for every judge. Stratified, oversampled or targeted labels are not supported.
- The multiplier concerns policy means (levels) corrected at weight one, not differences between policies.
- Prompts, not rows, are the independent units. With several labelled draws per prompt and judge errors shared within a prompt, a row-level ratio misstates the label need: in a simulation with four labelled draws per prompt, `(1 - R²_ref) / (1 - R²_J)` predicted about 1.00 where the realised variance ratio was 1.24 to 1.44, while the prompt-clustered multiplier predicted it within 5%. The multipliers count labelled rows and assume that added labels come in prompts labelled like the observed ones (the same labelled rows per prompt); `diagnostics["max_labelled_rows_per_prompt"]` shows how many there are, and `summary()` says so when prompts hold several rows.
- With `policy_ids`, the variance components are pooled within policy and `n` is the mean labelled rows per labelled policy, so the finite-N fields describe a policy with average labelling. When labelled rows per policy differ by more than 1.5 times (`diagnostics["labelled_rows_by_policy"]`), they fit neither extreme and a warning says so; run `compare_judges` on one policy's rows for a per-policy prediction.
- One calibrator per judge is fitted across all policies, as `analyze_dataset` fits one.
- Folds hash the cluster-id strings, so results move with the seed and with how the ids are spelled. To reproduce a judge's row with `calibrated_mean_ci` or `JudgeCalibrator.fit_cv`, pass identical `cluster_ids` (or None in both) and the same seed.
- Choosing the best of many judges on the same labels flatters the winner. The intervals hold for each pair separately, not for the selection.
- The default 2,000 replicates mean 2,000 calibrator fits per judge: about 40 seconds for two judges at 800 labels.
