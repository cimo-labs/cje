# Changelog

## Unreleased

- Remove the experimental multi-policy EIF implementation and standardize Direct-mode bootstrap on per-policy residual correction.
- Keep a temporary compatibility shim so legacy `use_multipolicy_eif=False` configs warn and are ignored, while `True` now fails fast with a clear error.
- Remove stale docs and tests for the retired multi-policy EIF path.

## 0.2.24

- Speed up planning variance fitting by using a lighter internal bootstrap during repeated measurement loops.
- Fix `fit_variance_model(...)` so `oracle_fraction_grid` actually controls the calibration grid.
- Make notebook tests execute the local checkout rather than reinstalling a published package inside the test kernel.
- Validate the planning workflow, planning notebook, simulation planning suite, and packaged wheel before release prep.
