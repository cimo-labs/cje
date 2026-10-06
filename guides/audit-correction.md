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
4. Use the new standard errors and confidence intervals. The correction changes the estimator's sampling variance and its covariance across policies. Do not shift an old interval while keeping its width. Because the correction is a mean over the labeled prompts, the analytic interval takes its degrees of freedom from them (at most `n_labeled − 1` with weight one, here 59 per policy, and lower when the oracle-jackknife term is large, as `oracle_df_cap_applied` reports; `metadata["degrees_of_freedom"][policy]["df_method"] == "labelled_clusters"`), so a handful of labels gives a wide interval and a single label gives none. Bootstrap and `"auto"` percentile intervals do not have this adjustment and under-cover at about 10 to 20 labeled prompts.

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

Weight one is the default and reproduces 0.8.x. With `w` tuned, the correction's variance is minimised asymptotically, so it cannot be worse than the labeled mean alone in large samples; with few labels the weight's own estimation noise can cost as much as it saves. The analytic interval counts the slope as a second fitted parameter (at most `n_labeled − 2` degrees of freedom, and the labeled prompts' variance scaled by `n_labeled / (n_labeled − 2)`) but omits its delta-method variance. On 45 held-out settings from five public benchmarks the tuned weight's pooled realised error was within about half a percent of weight one at 20 to 60 labels per policy; single settings ranged from about 3% worse at 20 labels to clearly better where weight one over-corrects, mostly a calibration map that transfers poorly to the evaluated policy. In simulations with a separately sampled calibration set, the tuned interval's coverage then matched weight one's to within about a quarter of a point (pooled gaps of 0.00, −0.26 and −0.22 points at 20, 25 and 30 labels). Opt in where weight one visibly over-corrects, or with 60 or more labeled prompts per policy.

The tuned rule falls back to weight one, and says why in `metadata["point_estimator"]["correction_weight_reasons"]`, when a policy has fewer than 20 labeled prompts (`too_few_labelled_clusters`), when fewer than 5 labeled prompts differ from the most common outcome (`rare_outcome`; with identical labels the tuned slope is zero, which made the standard error exactly zero), when the predictions are constant, and always in the known-propensity design (`known_propensity_fixed_one`), whose Horvitz-Thompson form is uncentred. The per-policy weights are in `metadata["point_estimator"]["correction_weights"]` (NaN where no correction applies); the bootstrap re-estimates the weight in every replicate, and the oracle-fold jackknife recomputes it on each fold's predictions. The weight optimises each policy's level, not paired differences. It is the power-tuned weight of PPI++ (Angelopoulos, Duchi and Zrnic, 2023).

When the correction is applied and every labeled outcome of a policy is identical, CJE logs a warning under either weight: the correction and its interval then rest on no outcome variation (if the calibration was fitted on the same labels, the standard error is zero whatever the weight), and a rare outcome needs more labels, not a different weight.

## Keep the statistical roles explicit

`label_design="representative"` requires a justified representative sample, such as the uniform prompt sample used here. For unequal-probability sampling, use the supported label-design and propensity inputs. With targeted labels and unknown inclusion probabilities, CJE retains an explicit plug-in route rather than treating those labels as representative.

Shared prompts are independent units for this example; individual responses are the units that consume human labels. Repeated ratings or multiple responses from the same prompt are not additional independent prompts. The same sampled prompt indices across policies preserve pairing for the comparison.

The original audit remains a statement about the unchanged calibration map. An original `FAIL` does not prevent representative target labels from supporting a residual-corrected estimate. Once those labels estimate a correction, they are not also an independent validation sample for that corrected estimate. The example keeps the old audit record and does not pass the reused slice as a new transport audit.

If you attach the labels and pass the audit probes in the same `analyze_dataset` call, the gates follow the same rule. Each augmented policy gets a correction design check in `metadata["correction_checks"][policy]`: at least 20 effective labelled prompts (`effective_labelled_prompts`), under known propensities a design effective sample size of at least 20 (`design_effective_n`) and no unlabelled row declared at propensity 1 (`unlabelled_certain_rows`; propensity 1 means "always labelled", not "no reweighting"), labelled outcomes that are not all identical, and labelled rows that reproduce the policy's judge-score mean and spread under the declared design (`balance_t`, each below 3 in absolute value). When it passes (`passed`; otherwise `failed` lists the codes), a transport `FAIL` or a `REFUSE-LEVEL` badge is recorded with `gate_exemption: "residual_corrected"` and a note, and does not flag the policy, set `CRITICAL` or drop it from `best_policy()`. When it fails, the policy keeps the gates exactly as in 0.9.1, and a balance, design-size or propensity-1 failure logs a warning even without any audit. The check catches labels chosen by judge score; it cannot detect selection that is unrelated to the score, so the design you declare still has to be true. Probes are checked as before: a probe that carries a calibration row's `observation_id` or `(source_id, row_id)` raises, so keep the fit external with `combine_oracle_sources=False` and audit with held-out probes.

Correction concerns the sampled target population. It does not establish transport to a future cycle or another policy. If you later refit using these labels, reserve a separate sample for an independent audit of that new fit.

## Numerical consistency

Two-stage calibration transforms smooth predictions into empirical ranks. A tiny rounding change before a rank boundary can cause a large change after calibration. The smooth prediction sum now uses the same arithmetic order for training, full-model prediction and held-out-fold prediction, so splitting or reordering a prediction batch does not change the corresponding scores.

Refit persisted two-stage calibrators to build their rank boundaries with the same arithmetic. Predictions at learned rank boundaries can differ from older versions; the batch-invariance regression tests cover weighted fits, covariates and fold inference.
