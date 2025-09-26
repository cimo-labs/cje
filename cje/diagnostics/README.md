# CJE Diagnostics System

## Overview

The CJE diagnostics system provides comprehensive monitoring and validation of causal inference assumptions. It follows a **push-based architecture** where estimators compute diagnostics during estimation and attach them to results.

## Core Architecture

The diagnostics system is now consolidated into a single cohesive module at `cje/diagnostics/`:

```
cje/diagnostics/
├── __init__.py          # Public API exports
├── models.py            # Data models (IPSDiagnostics, DRDiagnostics, Status, GateState)
├── weights.py           # Weight diagnostic computations (ESS, Hill, etc.)
├── overlap.py           # Overlap metrics (Hellinger affinity, auto-tuning, σ(S) floors)
├── dr.py                # DR-specific diagnostics
├── stability.py         # Stability and drift detection
├── display.py           # Display and formatting utilities
├── robust_inference.py  # Robust standard errors and inference
└── README.md           # This documentation
```

### Three-Layer Architecture

```
┌─────────────────┐
│  Data Models    │  models.py: Immutable dataclasses
│                 │  (IPSDiagnostics, DRDiagnostics)
└────────┬────────┘
         │
┌────────▼────────┐
│  Computation    │  weights.py, dr.py, stability.py:
│                 │  Pure functions for metric computation
└────────┬────────┘
         │
┌────────▼────────┐
│  Integration    │  Estimators import and use diagnostics
│                 │  during their estimate() methods
└─────────────────┘
```

### Key Design Principles

1. **Diagnostics are data, not behavior** - Dataclasses with computed properties
2. **Push-based flow** - Created during estimation, not on-demand
3. **Fail-fast with NaN** - Critical issues return NaN estimates, not exceptions
4. **Hierarchical status** - Multiple layers of safety checks
5. **Self-describing** - Objects know how to validate, summarize, and serialize themselves

## Diagnostic Classes

The system provides three main diagnostic classes that share a common interface:

### Common Interface

All diagnostic classes provide these methods:
- `validate() -> List[str]` - Self-consistency checks, returns list of issues
- `summary() -> str` - Human-readable one-line summary
- `to_dict() -> Dict` - Full serialization including enums as strings
- `to_json(indent=2) -> str` - JSON export with configurable formatting
- `to_csv_row() -> Dict` - Flat dictionary for tabular analysis

Computed properties (via `@property`):
- `filter_rate` - Fraction of samples filtered out
- `best_policy` - Policy with highest estimate
- `overall_status` - Aggregate health status
- Additional class-specific properties

### IPSDiagnostics

Base diagnostics for importance sampling estimators. Key field groups:

**Identification**: `estimator_type`, `method`, `policies`  
**Sample counts**: `n_samples_total`, `n_samples_valid`, `n_samples_used`  
**Results**: `estimates`, `standard_errors` (per policy)  
**Weight metrics**: `weight_ess`, `ess_per_policy`, `max_weight_per_policy`  
**Tail behavior**: `tail_indices` (Hill estimator results)  
**Status**: `weight_status`, `status_per_policy`  
**Calibration**: `calibration_rmse`, `calibration_r2`, `n_oracle_labels`

### DRDiagnostics

Extends IPSDiagnostics with doubly robust specific metrics:

**Cross-fitting**: `dr_cross_fitted`, `dr_n_folds`
**Outcome model**: `outcome_r2_range`, `outcome_rmse_mean`
**Influence functions**: `worst_if_tail_ratio`, `influence_functions`
**Decompositions**: `dr_diagnostics_per_policy`, `dm_ips_decompositions`
**Orthogonality**: `orthogonality_scores`


## Status System

The diagnostic system uses a **three-tier hierarchy**:

### 1. Computed Status (Informational)
Each diagnostic object computes an `overall_status` based on its metrics. This is purely informational and shown in displays but doesn't prevent estimation.

The `Status` enum has three values:
- `GOOD` - All metrics within acceptable ranges
- `WARNING` - Some concerning metrics but results usable
- `CRITICAL` - Severe issues detected

The `GateState` enum extends this with:
- `REFUSE` - Overlap too poor for any reliable estimation

Status computation varies by diagnostic class and combines multiple factors like ESS, tail indices, and calibration quality.

### 2. Validation Warnings  
The `validate()` method checks for logical inconsistencies:
- Impossible values (ESS > 1.0, R² > 1.0)
- Inconsistent counts (n_valid > n_total)
- Extreme metrics that suggest problems

Returns a list of issue descriptions. Empty list means all checks pass.

### 3. Refusal Gates (Optional)
Estimators can optionally refuse to provide estimates when diagnostics indicate unreliable results. By default, estimators **warn** and still provide estimates. When `refuse_unreliable=True`, they return `NaN` for unreliable policies.

Gate criteria use combinations of ESS, weight concentration, and coefficient of variation. These thresholds are more conservative than status levels and are estimator-specific.

## Key Diagnostic Metrics

### Hellinger Affinity (Bhattacharyya Coefficient)
Measures structural overlap between policies. **Cannot be improved by calibration.**
- **Affinity > 50%**: Good overlap
- **Affinity 35-50%**: Marginal overlap  
- **Affinity 20-35%**: Poor overlap (calibration might help)
- **Affinity < 20%**: Catastrophic mismatch (refuse estimation)

Key insight: Hellinger tells us whether to give up, ESS tells us how hard to try.

### Effective Sample Size (ESS)
Measures how many "effective" samples remain after weighting. **Can be improved by calibration.**
- **ESS > 30%**: Good overlap
- **ESS 10-30%**: Moderate overlap issues  
- **ESS < 10%**: Severe overlap problems

### Auto-Tuned ESS Thresholds
Instead of fixed thresholds, compute based on desired CI width using variance bounds for bounded rewards [0,1]:
```python
# For bounded rewards: Var(V_IPS) ≤ 1/(4n·ESS_fraction)  
# 95% CI halfwidth: ≈ 1.96/(2√(n·ESS_fraction))
# Solving: ESS_fraction ≥ (1.96/2)²/(n·target²) = 0.9604/(n·target²)
threshold = 0.9604 / (n * target_ci_halfwidth²)
```
For n=10,000 and ±1% target: threshold = 96%  
For n=100,000 and ±1% target: threshold = 9.6%

### Hill Tail Index
Estimates tail behavior of importance weights (k = 5% of samples).
- **α ≥ 2**: Finite variance, acceptable
- **α ∈ [1, 2)**: Infinite variance, WARNING
- **α < 1**: Infinite mean, CRITICAL

### Calibration R²
Measures judge-to-oracle calibration quality.
- **R² ≥ 0.5**: Good calibration
- **R² ∈ [0, 0.5)**: Moderate calibration
- **R² < 0**: Poor calibration

### Weight Concentration
Fraction of samples with near-zero weight.
- **< 50%**: Acceptable
- **50-85%**: Concerning
- **> 85%**: Critical

## Usage Examples

### Basic Diagnostics Check
```python
from cje import analyze_dataset

results = analyze_dataset("data.jsonl", estimator="calibrated-ips")
diagnostics = results.diagnostics

# Check overall health
if diagnostics.overall_status == Status.CRITICAL:
    print("⚠️ Critical issues detected!")
    print(diagnostics.summary())
```

### Detailed Analysis
```python
# Check per-policy metrics
for policy in diagnostics.policies:
    print(f"{policy}: ESS={diagnostics.ess_per_policy[policy]:.1%}")
    if diagnostics.hellinger_per_policy:
        print(f"  Hellinger affinity={diagnostics.hellinger_per_policy[policy]:.1%}")

# For DR estimators
if isinstance(diagnostics, DRDiagnostics):
    min_r2, max_r2 = diagnostics.outcome_r2_range
    print(f"Outcome R² range: [{min_r2:.3f}, {max_r2:.3f}]")
```

### Using Overlap Metrics
```python
from cje.diagnostics.overlap import compute_overlap_metrics, diagnose_overlap_problems

# Analyze overlap for a specific policy
weights = estimator.get_raw_weights("target_policy")
metrics = compute_overlap_metrics(
    weights,
    target_ci_halfwidth=0.01,  # Want ±1% CI
    auto_tune_threshold=True
)

# Get diagnosis and recommendations
should_proceed, explanation = diagnose_overlap_problems(metrics)
print(explanation)

# Check if calibration would help
if metrics.can_calibrate:
    print("SIMCal calibration could improve ESS")
else:
    print("Overlap too poor for calibration to help")
```


### Export for Analysis
```python
# Export to pandas for further analysis
import pandas as pd

df = pd.DataFrame(diagnostics.to_csv_row(), index=[0])
df.to_csv("diagnostics.csv")

# Or as JSON
with open("diagnostics.json", "w") as f:
    f.write(diagnostics.to_json())
```

## Diagnostic Gates

The system implements automatic gates that refuse estimation when critical issues are detected:

### CalibratedIPS Gates
The estimator refuses to provide estimates (returns NaN) when:
- ESS < 30% (less than 30% effective sample size)
- raw_near_zero > 85% (more than 85% of raw weights near zero)  
- top_5pct_weight > 30% AND cv_weights > 2.0 (high concentration with high variability)

### DR Estimator Gates
DR estimators inherit IPS gates and add warnings (but continue) when:
- Outcome model R² < 0 (indicates misspecification)
- Influence function tail ratio > 100 (heavy-tailed influence functions)

## Visualization

Weight diagnostics are displayed automatically when running `analyze_dataset.py`:
```
Weight Summary
----------------------------------------------------------------------
Policy                             ESS   Max Weight Status    
----------------------------------------------------------------------
clone                             45.2%      12.3456 GOOD      
parallel_universe_prompt          38.7%      18.9012 WARNING   
----------------------------------------------------------------------
```

Display utilities in `display.py` format diagnostics for tables and comparisons.

## Interpreting Diagnostics

### When to Trust Results

✅ **High Confidence**:
- Overall status: GOOD
- ESS > 50%
- Hill index > 2.5
- Calibration R² > 0.8
- DR: Balanced DM/IPS contributions

⚠️ **Use with Caution**:
- Overall status: WARNING
- ESS 20-50%
- Hill index 2.0-2.5
- Calibration R² 0.5-0.8
- DR: One component dominates

🔴 **Do Not Trust**:
- Overall status: CRITICAL
- ESS < 20%
- Hill index < 2.0
- Calibration R² < 0.5
- DR: Negative R² values

### Common Issues and Solutions

**Problem**: Low ESS (< 30%)
- **Cause**: Poor overlap between policies
- **Solution**: Use DR estimators with fresh draws

**Problem**: Heavy tails (Hill index < 2)
- **Cause**: Extreme importance weights
- **Solution**: Tighten variance cap in SIMCal

**Problem**: Poor calibration (R² < 0.5)
- **Cause**: Judge doesn't predict oracle well
- **Solution**: Increase oracle coverage or improve judge

**Problem**: Negative outcome model R²
- **Cause**: Model misspecification
- **Solution**: Check for distribution shift, add features

## Implementation Notes

### Memory Considerations
- Diagnostics store summary statistics, not raw data
- Influence functions stored in `EstimationResult.influence_functions`
- Can be large for many samples - consider memory when processing large datasets

### Adding New Metrics
1. Extend the dataclass in `models.py`
2. Add computation function to appropriate module
3. Call in estimator's `_build_diagnostics()`
4. Update `summary()` and `to_dict()` methods

## Advanced Topics

### Influence Function Analysis
```python
# Access influence functions (always stored)
for policy, ifs in results.influence_functions.items():
    z_scores = np.abs((ifs - np.mean(ifs)) / np.std(ifs))
    n_outliers = np.sum(z_scores > 3)
    print(f"{policy}: {n_outliers} influential points")
```

### Drift Detection
The Kendall-τ drift test is available but not integrated (Unix philosophy - you orchestrate):
```python
from cje.diagnostics import kendall_tau_drift
drift_result = kendall_tau_drift(historical_scores, current_scores)
if drift_result["tau"] < 0.5:
    print("Drift detected!")
```

## References

- **ESS**: Effective Sample Size in Importance Sampling (Kong, 1992)
- **Hill Estimator**: Hill (1975), "A Simple General Approach to Inference About the Tail of a Distribution"
- **Influence Functions**: Bickel et al. (1993), "Efficient and Adaptive Estimation"
- **TMLE Diagnostics**: van der Laan & Rose (2011), "Targeted Learning"

## Summary

The CJE diagnostics system provides:
- **Comprehensive monitoring** of all causal inference assumptions
- **Automatic safety gates** to prevent unreliable estimates
- **Clear status indicators** (GOOD/WARNING/CRITICAL)
- **Detailed metrics** for debugging issues
- **Export capabilities** for further analysis
- **Integration with visualization** for intuitive understanding

Always check diagnostics before trusting results!