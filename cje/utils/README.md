# CJE Utils Module

## Overview

Utility functions for export and analysis in CJE. This module provides practical tools for saving estimation results and debugging extreme weight issues.

## When to Use

### Use **Export Utilities** when:
- You need to save estimation results for reporting
- You want JSON or CSV output formats
- You need to share results with non-Python tools
- You're creating reproducible analysis pipelines

### Use **Extreme Weights Analysis** when:
- Debugging weight explosion issues
- Understanding which samples dominate estimates
- Identifying problematic log probability ratios
- Generating diagnostic reports for stakeholders

## File Structure

```
utils/
├── __init__.py                  # Re-exports and backward compatibility
├── export.py                    # JSON/CSV export functions
├── extreme_weights_analysis.py  # Weight debugging and reporting
├── aggregate_diagnostics.py     # CLI: Aggregate multiple result JSONs to CSV
└── analyze_diagnostics.py       # CLI: Statistical analysis of aggregated results
```

## Core Concepts

### 1. Result Export
Converts EstimationResult objects to standard formats:
- **JSON**: Hierarchical format with metadata and diagnostics
- **CSV**: Tabular format for spreadsheet analysis
- Handles numpy arrays, NaN values, and complex nested structures

### 2. Extreme Weights Analysis
Deep dive into importance weight behavior:
- Identifies samples with highest/lowest weights
- Tracks consistently extreme samples across policies
- Computes ESS and weight statistics
- Generates both JSON and text reports

### 3. Diagnostics Aggregation (CLI)
Aggregate multiple CJE result JSON files into a single CSV for cross-experiment analysis:
- Extracts core fields (policy, estimate, SE, CI bounds)
- Extracts diagnostics (ESS, tail indices, Hellinger affinity)
- Best-effort parsing - continues on malformed files

### 4. Diagnostics Analysis (CLI)
Statistical analysis of aggregated diagnostics:
- Correlation analysis across diagnostic fields
- Identifies quality issues using heuristics
- Outputs correlation matrix as CSV


## Common Interface

### Export Results

```python
from cje.utils import export_results_json, export_results_csv

# After running estimation
result = estimator.fit_and_estimate()

# Export to JSON with full details
export_results_json(
    result,
    "results/analysis.json",
    include_diagnostics=True,
    include_metadata=True
)

# Export to CSV for Excel
export_results_csv(
    result,
    "results/summary.csv",
    include_ci=True
)
```

### Analyze Extreme Weights

```python
from cje.utils import analyze_extreme_weights

# Debug weight issues
json_report, text_report = analyze_extreme_weights(
    dataset=dataset,
    sampler=sampler,
    raw_weights_dict=raw_weights,
    calibrated_weights_dict=calibrated_weights,
    n_extreme=10,  # Top/bottom 10 samples
    output_dir=Path("diagnostics/")
)

# Reports saved to diagnostics/extreme_weights_analysis.{json,txt}
print(text_report)  # Human-readable summary
```


### CLI Tools

```bash
# Aggregate multiple result JSONs into a single CSV
python -m cje.utils.aggregate_diagnostics --input results_dir/ --output aggregated.csv

# Analyze correlations across aggregated diagnostics
python -m cje.utils.analyze_diagnostics --input aggregated.csv --corr correlation_matrix.csv
```

The aggregation workflow is useful for:
- Comparing results across multiple experiments
- Identifying patterns in diagnostic metrics
- Building dashboards from multiple runs


## Key Design Decisions

### 1. **Graceful Serialization**
Export functions handle complex types:
- Numpy arrays → lists
- NaN → null (JSON) or empty (CSV)
- Complex objects → string representations
- Never fails on serialization errors

### 2. **Comprehensive Weight Analysis**
Extreme weights analysis provides multiple views:
- Per-policy statistics
- Cross-policy patterns
- Sample-level details
- Both JSON (programmatic) and text (human) formats


## Common Issues

### "Can't serialize object to JSON"
The export functions handle most types, but custom objects may need:
```python
# Add to metadata as strings
result.metadata["custom_obj"] = str(my_custom_object)
```

### "Extreme weights report too large"
Limit number of samples analyzed:
```python
analyze_extreme_weights(..., n_extreme=5)  # Only top/bottom 5
```

## Performance

- **Export**: O(n_policies) - Fast even for large results
- **Extreme weights**: O(n_samples × n_policies) - Can be slow for large datasets

For large datasets:
- Export in batches if memory constrained
- Analyze subset of policies for extreme weights

## Summary

The utils module provides essential tools for CJE workflows: exporting results for reporting and debugging weight issues through detailed analysis. These utilities handle the practical aspects of working with CJE results in production environments.