# CJE Visualization Module

## Overview

Diagnostic plots for understanding and validating CJE results: judge-calibration assessment, policy-estimate forest plots, and the budget-planning dashboard. All of it is optional — the core library runs without matplotlib.

## When to Use

### Use **Calibration Plots** when:
- You want to visualize the judge → oracle mapping
- You need to assess calibration quality (ECE, RMSE) before/after calibration

### Use **Estimate Plots** when:
- You need to compare policy performance with confidence intervals
- You want publication-ready forest plots (with optional oracle ground truth)

### Use the **Planning Dashboard** when:
- You're choosing sample sizes and oracle-label budgets before an experiment
- You need to communicate budget/precision tradeoffs

> **Note:** Transportability plots live in `cje.diagnostics`, not here:
> `diag.plot()` for single-policy decile bars and
> `plot_transport_comparison(audits_dict)` for a multi-policy forest plot.

## Installation

```bash
pip install "cje-eval[viz]"   # adds matplotlib + seaborn
```

Without the extra, `import cje` still works with no warnings; *accessing* any
`plot_*` name raises an actionable hint instead of an `AttributeError`:

```text
ImportError: plot_policy_estimates requires the viz extra. Install with: pip install "cje-eval[viz]"
```

The same lazy behavior applies to `cje.visualization` and the
`results.plot_estimates(...)` convenience method.

## File Structure

```
visualization/
├── __init__.py              # Public API (lazy ImportError hint on no-viz installs)
├── calibration.py           # Calibration transformation and reliability plots
├── estimates.py             # Policy performance forest plots
└── planning.py              # Budget planning dashboard (MDE vs budget)
```

## Common Interface

```python
# Option 1: Import directly from the main cje namespace (recommended)
from cje import (
    plot_policy_estimates,
    plot_calibration_comparison,
    plot_planning_dashboard,
)

# Option 2: Import from the visualization module (also works)
from cje.visualization import (
    plot_calibration_comparison,
    plot_policy_estimates,
    plot_planning_dashboard,
)

# Calibration comparison (before/after alignment, ECE/RMSE annotations)
fig = plot_calibration_comparison(
    judge_scores=judge_scores,
    oracle_labels=oracle_labels,
    calibrated_scores=calibrated_scores,
    save_path="diagnostics/calibration.png",
)

# Policy estimates — direct function call
fig = plot_policy_estimates(
    estimates={"clone": 0.74, "parallel_universe_prompt": 0.76, "unhelpful": 0.17},
    standard_errors={"clone": 0.02, "parallel_universe_prompt": 0.03, "unhelpful": 0.01},
    oracle_values={"clone": 0.74, "parallel_universe_prompt": 0.77, "unhelpful": 0.18},
)

# Policy estimates — convenience method on EstimationResult
from cje import analyze_dataset

result = analyze_dataset(fresh_draws_dir="responses/")
fig = result.plot_estimates(save_path="estimates.png")

# Planning dashboard (3 panels: MDE vs budget, power curve, cost sensitivity)
from cje.diagnostics import CostModel, fit_variance_model

# variance_model = fit_variance_model(pilot_data)   # from base-policy pilot draws
# cost_model = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
# fig = plot_planning_dashboard(variance_model, cost_model)
```

**Jupyter notebooks:** `EstimationResult` objects display as formatted HTML
tables automatically when evaluated in a cell.

## Key Design Decisions

1. **Optional by construction** — plots resolve lazily; a no-viz install pays no import cost and gets a pip-install hint on access.
2. **Automatic metric computation** — calibration plots compute and display ECE/RMSE; no separate metric step.
3. **Save options everywhere** — every plot takes `save_path` and writes high-DPI output.

## Common Issues

### "No matplotlib backend"
In headless environments force a non-GUI backend:
```bash
export MPLBACKEND=Agg
```

### `ImportError: ... requires the viz extra`
Working as intended — install with `pip install "cje-eval[viz]"`.

## Summary

Three focused plot families — calibration quality, policy estimates, budget planning — that turn CJE's statistics into reviewable pictures, shipped as an optional extra with a loud install hint when missing.
