# Experiments

Repository-level experiments live here and are intentionally not part of the
installable `cje` Python package.

## Current Studies

- `offset_vs_refit/`: transport-drift simulation study for first-moment
  correction strategies (global offset, policy-specific offset, refit).

## Research setup

From the repository root, install the optional research dependencies (including
pandas) before running these scripts or their tests:

```bash
poetry install --with research --extras viz
poetry run pytest -q experiments/offset_vs_refit/test_offset_vs_refit_simulation.py
```

The `viz` extra is needed for plots; omit it when using `--no-plots`. The
[study README](offset_vs_refit/README.md) gives the run commands and explains the
required predeclared transport margin and the limits of the simulation's claims.

## Why This Folder Exists

- Keeps PyPI package scope focused on stable APIs.
- Avoids shipping research scripts and experiment artifacts in wheels.
- Makes it clear these studies are reproducibility assets, not runtime library
  interfaces.
