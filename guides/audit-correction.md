# Use audit labels to correct a policy estimate

A transport audit measures the mean residual of the current calibration map. Passing labels through `TransportAuditConfig` does not change an estimate. CJE's existing augmented estimator can use those labels for bias correction when they are attached to their matching evaluation rows and their sampling design supports the correction.

Run the complete synthetic example from this checkout:

```sh
poetry run python examples/audit_correction.py
```

The [example](../examples/audit_correction.py) gives the candidate a +0.15 change in human outcomes that its judge scores miss. It fits calibration on 40 separate anchor labels, audits a uniform sample of 60 evaluation prompts per policy, and then reuses those 120 labels to correct the current estimates. The total is 160 human labels, including calibration.

## Audit first, then correct

1. Fit on the separate calibration file and keep evaluation rows judge-only. Pass the held-out labels through `TransportAuditConfig` to record the audit of that map. The example verifies that both point-estimator routes are `plug_in`.
2. Copy the evaluation records and attach each sampled `oracle_label` to its exact response. Preserve missing labels on the other rows. A label in a calibration-history file alone is not an evaluation residual observation.
3. Rerun with `combine_oracle_sources=False` to keep the same external calibration fit. `use_augmented_estimator=True` is already the default; the example specifies it explicitly and verifies that both routes become `augmented`.
4. Use the new standard errors and confidence intervals. The correction changes the estimator's sampling variance and its covariance across policies. Do not shift an old interval while keeping its width.

For a representative labeled slice, the correction is:

```text
corrected mean = mean calibrated prediction across evaluation rows
               + mean(human label - calibrated prediction) on the labeled slice
```

The result records the applied correction, label design and route in `metadata["point_estimator"]`. Inspect this metadata before comparing methods; requesting augmentation does not prove it was applicable to the supplied data.

## Weight the prediction, or not

The formula above applies the calibrated prediction at weight one. That is only optimal when the prediction's spread within the policy matches its covariance with the human label. A binary judge, or a calibration map that transfers imperfectly to the evaluated policy, over-corrects at weight one, and the corrected estimate can then be less precise than the labeled mean alone. `estimator_config={"correction_weight": "tuned"}` estimates the weight from the labeled slice as the power-tuned weight of PPI++ (the least-squares slope of labels on predictions, clipped to [0, 1]):

```text
corrected mean = w * mean calibrated prediction across evaluation rows
               + mean(human label - w * calibrated prediction) on the labeled slice
```

Weight one is the default and reproduces 0.8.x. With `w` tuned, the correction's variance is minimised asymptotically, so it cannot be worse than the labeled mean alone in large samples; with few labels the weight's own estimation noise, which the analytic standard error does not carry, can cost as much as it saves. On 45 held-out settings from six public benchmarks the tuned weight's realised error was within about half a percent of weight one at 20 to 60 labels per policy, and clearly ahead only where weight one over-corrects: a binary judge, or a calibration map that transfers poorly to the evaluated policy. Opt in there, or with 60 or more labeled prompts per policy.

The tuned rule falls back to weight one, and says why in `metadata["point_estimator"]["correction_weight_reasons"]`, when a policy has fewer than 20 labeled prompts (`too_few_labelled_clusters`), when fewer than 5 labeled prompts differ from the most common outcome (`rare_outcome`; with identical labels a weight of zero would make the standard error exactly zero), when the predictions are constant, and always in the known-propensity design (`known_propensity_fixed_one`), whose Horvitz-Thompson form is uncentred. The per-policy weights are in `metadata["point_estimator"]["correction_weights"]` (NaN where no correction applies); the bootstrap and the oracle-fold jackknife re-estimate the weight in every replicate. The weight optimises each policy's level, not paired differences. It is the power-tuned weight of PPI++ (Angelopoulos, Duchi and Zrnic, 2023).

When every labeled outcome of a policy is identical, CJE logs a warning under either weight: the correction and its interval then rest on no outcome variation, and a rare outcome needs more labels, not a different weight.

## Keep the statistical roles explicit

`label_design="representative"` requires a justified representative sample, such as the uniform prompt sample used here. For unequal-probability sampling, use the supported label-design and propensity inputs. With targeted labels and unknown inclusion probabilities, CJE retains an explicit plug-in route rather than treating those labels as representative.

Shared prompts are independent units for this example; individual responses are the units that consume human labels. Repeated ratings or multiple responses from the same prompt are not additional independent prompts. The same sampled prompt indices across policies preserve pairing for the comparison.

The original audit remains a statement about the unchanged calibration map. An original `FAIL` does not prevent representative target labels from supporting a residual-corrected estimate. Once those labels estimate a correction, they are not also an independent validation sample for that corrected estimate. The example keeps the old audit record and does not pass the reused slice as a new transport audit.

Correction concerns the sampled target population. It does not establish transport to a future cycle or another policy. If you later refit using these labels, reserve a separate sample for an independent audit of that new fit.

## Numerical consistency

Two-stage calibration transforms smooth predictions into empirical ranks. A tiny rounding change before a rank boundary can cause a large change after calibration. The smooth prediction sum now uses the same arithmetic order for training, full-model prediction and held-out-fold prediction, so splitting or reordering a prediction batch does not change the corresponding scores.

Refit persisted two-stage calibrators to build their rank boundaries with the same arithmetic. Predictions at learned rank boundaries can differ from older versions; the batch-invariance regression tests cover weighted fits, covariates and fold inference.
