# Changelog

## Unreleased

- Fix the README and interface-README Quick Start examples, which crashed verbatim because they provided fewer than the 10 oracle labels required by 5-fold cross-fitted calibration; document the 10-label minimum next to the labeling guidance.
- Reduce calibration folds gracefully (with a warning) in Direct mode when 4-9 oracle labels are available instead of failing; below 4 labels, the calibration error now says exactly how to fix it.
- Handle degenerate fold assignments in FlexibleCalibrator (all oracle samples hashed into one fold) by falling back to fitting on all oracle samples for that fold.
- Reduce full-data calibration folds for small oracle slices in cluster-bootstrap inference so Direct-mode point estimates no longer come back NaN with fewer than 10 labels.

## 0.2.25

- Remove the experimental multi-policy EIF implementation and standardize Direct-mode bootstrap on per-policy residual correction.
- Keep a temporary compatibility shim so legacy `use_multipolicy_eif=False` configs warn and are ignored, while `True` now fails fast with a clear error.
- Remove stale docs and tests for the retired multi-policy EIF path.

## 0.2.24

- Speed up planning variance fitting by using a lighter internal bootstrap during repeated measurement loops.
- Fix `fit_variance_model(...)` so `oracle_fraction_grid` actually controls the calibration grid.
- Make notebook tests execute the local checkout rather than reinstalling a published package inside the test kernel.
- Validate the planning workflow, planning notebook, simulation planning suite, and packaged wheel before release prep.
