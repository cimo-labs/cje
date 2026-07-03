# CJE Visualization Module

## Overview

The visualization module provides diagnostic plots for understanding and validating CJE analysis results. It offers calibration assessment, policy estimate comparisons, and budget-planning dashboards to help practitioners audit assumptions and interpret results.

## When to Use

### Use **Calibration Plots** when:
- You want to visualize judge → oracle calibration
- You need to assess calibration quality (ECE, RMSE)
- You're comparing before/after calibration alignment
- You want to understand calibration transformations

### Use **Estimate Plots** when:
- You need to compare policy performance
- You want confidence intervals visualized
- You have oracle ground truth for validation
- You need publication-ready forest plots

### Use **Planning Dashboard** when:
- You need to determine budget for target MDE
- You want to visualize cost-sensitivity tradeoffs
- You're planning sample sizes before running experiments
- You need to communicate budget/precision tradeoffs

> **Note:** For transportability diagnostics, use `cje.diagnostics`:
> - `diag.plot()` for single-policy decile bars
> - `plot_transport_comparison(results_dict)` for multi-policy forest plot

## File Structure

```
visualization/
├── __init__.py              # Public API
├── calibration.py           # Calibration transformation and reliability plots
├── estimates.py             # Policy performance forest plots
└── planning.py              # Budget planning dashboard (MDE vs budget)
```

## Core Concepts

### 1. Calibration Assessment
Visual tools for judge calibration quality:
- **Transformation curves**: Visualize f: judge → oracle mapping
- **Reliability diagrams**: Bin-wise calibration alignment
- **Improvement metrics**: ECE and RMSE before/after calibration

### 2. Estimate Visualization
Clear presentation of final results:
- **Forest plots**: Point estimates with confidence intervals
- **Policy comparison**: Visual ranking and uncertainty
- **Oracle validation**: Compare estimates to ground truth when available

### 3. Planning Dashboard
Budget optimization visualization (3 panels):
- **MDE vs Budget**: What precision can you achieve at each budget level
- **Power Curve**: Statistical power to detect various effect sizes
- **Cost Sensitivity**: How oracle cost ratio affects optimal allocation

## Common Interface

All visualization functions follow consistent patterns and are available in two ways:

```python
# Option 1: Import directly from main cje namespace (recommended)
from cje import (
    plot_policy_estimates,
    plot_calibration_comparison,
    plot_planning_dashboard,
)

# Option 2: Import from visualization module (also works)
from cje.visualization import (
    plot_calibration_comparison,
    plot_policy_estimates,
    plot_planning_dashboard,
)

# Calibration comparison
fig = plot_calibration_comparison(
    judge_scores=judge_scores,
    oracle_labels=oracle_labels,
    calibrated_scores=calibrated_scores,
    save_path="diagnostics/calibration.png"
)

# Policy estimates - Option 1: Direct function call
fig = plot_policy_estimates(
    estimates={"clone": 0.74, "parallel_universe_prompt": 0.76, "unhelpful": 0.17},
    standard_errors={"clone": 0.02, "parallel_universe_prompt": 0.03, "unhelpful": 0.01},
    oracle_values={"clone": 0.74, "parallel_universe_prompt": 0.77, "unhelpful": 0.18}
)

# Policy estimates - Option 2: Convenience method from EstimationResult
from cje import analyze_dataset

result = analyze_dataset(fresh_draws_dir="responses/")
fig = result.plot_estimates(
    base_policy_stats={"mean": 0.72, "se": 0.01},
    save_path="estimates.png"
)

# Planning dashboard (budget optimization)
from cje.visualization import plot_planning_dashboard
from cje.diagnostics import fit_variance_model, CostModel

# Fit variance model from base policy pilot data (where calibration is learned)
variance_model = fit_variance_model(pilot_data)

# Specify your actual costs
cost_model = CostModel(surrogate_cost=0.01, oracle_cost=0.16)

# Generate 3-panel dashboard: MDE vs Budget, Power Curve, Cost Sensitivity
fig = plot_planning_dashboard(variance_model, cost_model)

# Transportability diagnostics (from cje.diagnostics, not visualization)
from cje.diagnostics import audit_transportability, plot_transport_comparison

results = {}
for policy in ["clone", "premium"]:
    probe = load_probe(policy)  # List[dict] with judge_score, oracle_label
    results[policy] = audit_transportability(calibrator, probe, group_label=policy)

fig = plot_transport_comparison(results)  # Forest plot
results["clone"].plot()  # Single-policy decile bars
```

**Jupyter notebooks:** `EstimationResult` objects automatically display as formatted HTML tables when evaluated in a cell.

## Key Design Decisions

### 1. **Multi-Panel Dashboards**
Complex diagnostics are organized into focused panels:
- Each panel answers one specific question
- Panels are visually connected but independently interpretable
- Summary metrics accompany visual diagnostics

### 2. **Automatic Metric Computation**
Visualizations compute and display key metrics:
- Calibration errors (ECE, RMSE)
- No need for separate metric calculation

### 3. **Save Options**
All plots support optional saving:
- Automatic file extension handling
- High DPI for publication quality
- Consistent naming conventions

## Common Issues

### "No matplotlib backend"
Most plots require the optional visualization dependencies:
```bash
pip install "cje-eval[viz]"
```

If you're in a headless environment, force a non-GUI backend:
```bash
export MPLBACKEND=Agg
```

If you're on Linux and want interactive plots, install a GUI backend (e.g. `python3-tk` or Qt bindings) and then install matplotlib (or just use the `cje-eval[viz]` extra above).

### "Missing diagnostics object"
Ensure estimator was run with diagnostics enabled:
```python
result = estimator.fit_and_estimate()
```

## Performance

- **Calibration plots**: O(n_samples × n_bins) for binning operations
- **Memory**: Plots create temporary copies for sorting/binning

For large datasets (>100k samples), consider:
- Sampling for scatter plots
- Reducing bin counts
- Pre-computing metrics

## Summary

The visualization module transforms statistical diagnostics into interpretable visual insights. It helps practitioners validate assumptions, diagnose issues, and communicate results effectively through focused diagnostic plots and planning dashboards.
