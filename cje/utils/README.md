# CJE Utils Module

## Overview

Utility functions for working with CJE results: exporting `EstimationResult` objects to standard formats and aggregating/analyzing result files across runs.

## When to Use

### Use **Export Utilities** when:
- You need to save estimation results for reporting
- You want JSON or CSV output formats
- You need to share results with non-Python tools

### Use the **Aggregation CLIs** when:
- You're comparing results across many experiments
- You want a single CSV of per-policy estimates + diagnostics from a directory of result JSONs

## File Structure

```
utils/
├── __init__.py                  # Backward-compat re-exports of plot functions
├── export.py                    # JSON/CSV export functions
├── aggregate_diagnostics.py     # CLI: aggregate result JSONs into one CSV
└── analyze_diagnostics.py       # CLI: correlations / quality heuristics on the CSV
```

## Common Interface

### Export Results

```python
from cje.utils.export import export_results_json, export_results_csv

# results = analyze_dataset(...)

# JSON with full details (this is what `cje analyze -o out.json` writes)
# export_results_json(results, "results/analysis.json",
#                     include_diagnostics=True, include_metadata=True)

# CSV for spreadsheets
# export_results_csv(results, "results/summary.csv", include_ci=True)
```

The JSON export includes per-policy estimates/SEs/CIs, the serialized
diagnostics (including boundary cards), and metadata.

### CLI Tools

```bash
# Aggregate result JSONs (file or directory) into a single CSV
python -m cje.utils.aggregate_diagnostics --input results_dir/ --output aggregated.csv

# Correlation matrix + "do not ship" heuristic counts on the aggregated CSV
python -m cje.utils.analyze_diagnostics --input aggregated.csv --corr correlation_matrix.csv
```

Aggregation extracts one row per (file, policy): estimate, SE, CI bounds,
sample counts, and boundary-card fields. Parsing is best-effort — malformed
files are skipped, not fatal.

## Key Design Decisions

1. **Graceful serialization** — numpy arrays → lists, NaN → null (JSON) or empty (CSV), complex objects → strings; export never fails on serialization errors.
2. **Best-effort aggregation** — cross-experiment tooling keeps going past malformed inputs so one bad file doesn't sink a dashboard build.
3. **Plot re-exports are legacy** — `cje.utils` re-exports `plot_calibration_comparison`/`plot_policy_estimates` for backward compatibility; import from `cje` or `cje.visualization` in new code.

## Common Issues

### "Can't serialize object to JSON"
The export functions handle most types, but custom objects may need:
```python
# Add to metadata as strings
# results.metadata["custom_obj"] = str(my_custom_object)
```

## Summary

Small, practical helpers for the last mile of a CJE workflow: exporting results for reports and rolling many runs up into one analyzable table.
