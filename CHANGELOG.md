# Changelog

## 0.2.24

- Speed up planning variance fitting by using a lighter internal bootstrap during repeated measurement loops.
- Fix `fit_variance_model(...)` so `oracle_fraction_grid` actually controls the calibration grid.
- Make notebook tests execute the local checkout rather than reinstalling a published package inside the test kernel.
- Validate the planning workflow, planning notebook, simulation planning suite, and packaged wheel before release prep.
