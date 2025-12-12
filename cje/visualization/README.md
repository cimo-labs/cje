# CJE Visualization Module

## Overview

The visualization module provides comprehensive diagnostic plots for understanding and validating CJE analysis results. It offers specialized dashboards for weight diagnostics, doubly robust diagnostics, calibration assessment, and policy estimate comparisons to help practitioners audit assumptions and interpret results.

## When to Use

### Use **Weight Dashboards** when:
- You need to diagnose weight explosion or concentration
- You want to understand effective sample size (ESS) issues
- You're comparing raw vs calibrated weight behaviors
- You need to identify which samples dominate estimates

### Use **DR Dashboard** when:
- You're using doubly robust estimators
- You need to check orthogonality assumptions
- You want to understand DM vs IPS contributions
- You need to diagnose influence function tail behavior

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

> **Note:** For transportability diagnostics, use `cje.diagnostics`:
> - `diag.plot()` for single-policy decile bars
> - `plot_transport_comparison(results_dict)` for multi-policy forest plot

## File Structure

```
visualization/
├── __init__.py              # Public API with backward-compatible aliases
├── calibration.py           # Calibration transformation and reliability plots
├── dr_dashboards.py         # Doubly robust diagnostic visualizations
├── estimates.py             # Policy performance forest plots
└── weight_dashboards.py     # Weight diagnostic dashboards (summary & detailed)
```

## Core Concepts

### 1. Weight Diagnostics
Comprehensive analysis of importance weight behavior:
- **ESS tracking**: Monitor effective sample size degradation
- **Tail analysis**: CCDF plots to identify heavy tails
- **Concentration metrics**: How many samples contribute X% of weight
- **Calibration impact**: Compare raw vs calibrated distributions
- **Judge correlation**: Optional analysis of weight-judge score relationships

### 2. DR Diagnostics
Specialized plots for doubly robust estimation:
- **Component analysis**: Direct method vs IPS correction contributions
- **Orthogonality checks**: Score function mean ± 2SE for validity
- **Influence functions**: EIF tail behavior and stability

### 3. Calibration Assessment
Visual tools for judge calibration quality:
- **Transformation curves**: Visualize f: judge → oracle mapping
- **Reliability diagrams**: Bin-wise calibration alignment
- **Improvement metrics**: ECE and RMSE before/after calibration

### 4. Estimate Visualization
Clear presentation of final results:
- **Forest plots**: Point estimates with confidence intervals
- **Policy comparison**: Visual ranking and uncertainty
- **Oracle validation**: Compare estimates to ground truth when available

## Common Interface

All visualization functions follow consistent patterns and are available in two ways:

```python
# Option 1: Import directly from main cje namespace (recommended)
from cje import (
    plot_policy_estimates,
    plot_calibration_comparison,
    plot_weight_dashboard_summary,
    plot_weight_dashboard_detailed,
    plot_dr_dashboard,
)

# Option 2: Import from visualization module (also works)
from cje.visualization import (
    plot_weight_dashboard_summary,
    plot_weight_dashboard_detailed,
    plot_dr_dashboard,
    plot_calibration_comparison,
    plot_policy_estimates,
)

# Weight diagnostics - summary dashboard (6 panels)
fig, metrics = plot_weight_dashboard_summary(
    raw_weights_dict=raw_weights,
    calibrated_weights_dict=calibrated_weights,
    save_path="diagnostics/weights_summary.png"
)

# Weight diagnostics - detailed per-policy view
fig, metrics = plot_weight_dashboard_detailed(
    raw_weights_dict=raw_weights,
    calibrated_weights_dict=calibrated_weights,
    judge_scores=judge_scores,  # Optional for correlation analysis
    save_path="diagnostics/weights_detailed.png"
)

# DR diagnostics (requires DR estimation result)
fig, summary = plot_dr_dashboard(
    estimation_result=dr_result,
    figsize=(15, 5)
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

### 2. **Dual Dashboard Approach**
Two complementary weight visualizations:
- **Summary dashboard**: 6-panel overview across all policies
- **Detailed dashboard**: Per-policy analysis with judge score correlation
- Each serves distinct analysis needs with clear naming

### 3. **Automatic Metric Computation**
Visualizations compute and display key metrics:
- ESS and effective sample percentages
- Calibration errors (ECE, RMSE)
- Weight concentration statistics
- No need for separate metric calculation

### 4. **Save Options**
All plots support optional saving:
- Automatic file extension handling
- High DPI for publication quality
- Consistent naming conventions

## Common Issues

### "No matplotlib backend"
Install matplotlib with GUI support:
```bash
pip install matplotlib[gui]
```

### "Figure too small for content"
Adjust figsize parameter:
```python
plot_weight_dashboard_summary(..., figsize=(16, 14))
```

### "Missing diagnostics object"
Ensure estimator was run with diagnostics enabled:
```python
result = estimator.fit_and_estimate(compute_diagnostics=True)
```

## Performance

- **Weight dashboards**: O(n_samples × n_policies) for metric computation
- **DR dashboards**: O(n_samples) for influence function analysis  
- **Calibration plots**: O(n_samples × n_bins) for binning operations
- **Memory**: Dashboards create temporary copies for sorting/binning

For large datasets (>100k samples), consider:
- Sampling for scatter plots
- Reducing bin counts
- Pre-computing metrics
- Using summary dashboard instead of detailed for initial analysis

## Summary

The visualization module transforms complex statistical diagnostics into interpretable visual insights. It helps practitioners validate assumptions, diagnose issues, and communicate results effectively through carefully designed multi-panel dashboards and focused diagnostic plots.