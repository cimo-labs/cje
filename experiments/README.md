# Experiments

Repository-level experiments live here and are intentionally not part of the
installable `cje` Python package.

## Current Studies

- `offset_vs_refit/`: transport-drift simulation study for first-moment
  correction strategies (global offset, policy-specific offset, refit).

## Why This Folder Exists

- Keeps PyPI package scope focused on stable APIs.
- Avoids shipping research scripts and experiment artifacts in wheels.
- Makes it clear these studies are reproducibility assets, not runtime library
  interfaces.
