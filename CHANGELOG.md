# Changelog

## Unreleased

- **Fix OUA jackknife variance understated by a factor of K** in `base_estimator._apply_oua_jackknife` and `stacking._apply_stacked_oua` (mean over folds instead of the delete-one-fold jackknife sum, paper Alg. 6). With the default K=5, the calibration-uncertainty contribution to reported standard errors was ~2.24x too small for calibrated-ips, calibrated direct (jackknife path), TMLE, MRDR, and stacked-dr. All OUA sites now share one `oracle_jackknife_variance()` helper. Reported CIs widen by design.
- **Fix `calibration_data_path` replacing the evaluation dataset**: in IPS/DR mode the estimator was built from the calibration (or combined-oracle) pool — whose fabricated logprobs made every policy's estimate identical with a perfect ESS — instead of the logged data. The calibration pool is now used only to fit the calibrator, whose rewards are applied to the logged evaluation dataset before estimation. Regression tests pin distinct per-policy estimates and logged-length influence functions.
- **Fix TMLE collapsing to the untargeted plug-in**: the fluctuation was solved on logged data but never propagated to fresh draws, so the solved score equation cancelled the IPS correction exactly and `TMLEEstimator` returned the direct-method plug-in with a DR-type SE (no double robustness). The targeting step now uses a score-indexed clever covariate h(S)=E[w|S] (fit per fold; exact for S-calibrated weights) and applies the targeted outcome model to fresh draws per draw. Regression tests pin double robustness under deliberate outcome-model misspecification and first-order agreement with DR-CPO.
- **Fix OUA jackknife predictions for two-stage calibrators**: leave-one-fold reward predictions in `CalibratedIPS`, `DREstimator`, and `CalibratedDirectEstimator` fed raw judge scores to per-fold isotonic models that expect the ECDF rank index, corrupting jackknife replicates whenever `two_stage` calibration was active (auto-selectable by default; always active with covariates). All jackknife paths now route through `reward_calibrator.predict_oof(...)`, and `CalibratedIPS` gained covariate support in its jackknife.

## 0.2.25

- Remove the experimental multi-policy EIF implementation and standardize Direct-mode bootstrap on per-policy residual correction.
- Keep a temporary compatibility shim so legacy `use_multipolicy_eif=False` configs warn and are ignored, while `True` now fails fast with a clear error.
- Remove stale docs and tests for the retired multi-policy EIF path.

## 0.2.24

- Speed up planning variance fitting by using a lighter internal bootstrap during repeated measurement loops.
- Fix `fit_variance_model(...)` so `oracle_fraction_grid` actually controls the calibration grid.
- Make notebook tests execute the local checkout rather than reinstalling a published package inside the test kernel.
- Validate the planning workflow, planning notebook, simulation planning suite, and packaged wheel before release prep.
