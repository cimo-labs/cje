# CJE Utils Module

## Overview

Utility functions for working with CJE results: exporting `EstimationResult` objects to JSON.

## When to Use

### Use **Export Utilities** when:
- You need to save estimation results for reporting
- You need to share results with non-Python tools

## File Structure

```
utils/
├── __init__.py                  # (no re-exports)
└── export.py                    # JSON export function
```

## Common Interface

### Export Results

```python
from cje.utils.export import export_results_json

# results = analyze_dataset(...)

# JSON with full details (this is what `cje analyze -o out.json` writes)
# export_results_json(results, "results/analysis.json",
#                     include_diagnostics=True, include_metadata=True)
```

The JSON export includes per-policy estimates/SEs/CIs, the serialized
diagnostics (including boundary cards), and metadata.

## Key Design Decisions

1. **Graceful serialization** — numpy arrays → lists, NaN → null, complex objects → strings; export never fails on serialization errors.
2. **One format** — `export_results_csv` and the aggregation CLIs (`aggregate_diagnostics`, `analyze_diagnostics`) were removed in 0.5.0 (unused); the JSON export is the canonical serialized form.
3. **Plots live in cje.visualization** — the 0.3.x-era matplotlib re-exports from `cje.utils` were removed in 0.5.0.

## Common Issues

### "Can't serialize object to JSON"
The export functions handle most types, but custom objects may need:
```python
# Add to metadata as strings
# results.metadata["custom_obj"] = str(my_custom_object)
```

## Summary

A small helper for the last mile of a CJE workflow: exporting results to JSON for reports and downstream tools.
