# CJE Diagnostics System

## Overview

The CJE diagnostics system provides comprehensive monitoring and validation of causal inference assumptions. It follows a **push-based architecture** where estimators compute diagnostics during estimation and attach them to results.

## Core Architecture

The diagnostics system is now consolidated into a single cohesive module at `cje/diagnostics/`:

```
cje/diagnostics/
├── __init__.py          # Public API exports
├── models.py            # Data models (IPSDiagnostics, DRDiagnostics, CJEDiagnostics, Status, GateState)
├── weights.py           # Weight diagnostic computations (ESS, Hill, etc.)
├── overlap.py           # Overlap metrics (Hellinger affinity, auto-tuning, σ(S) floors)
├── dr.py                # DR-specific diagnostics
├── transport.py         # Transportability auditing
├── display.py           # Display and formatting utilities
├── robust_inference.py  # Robust standard errors and inference
└── README.md            # This documentation
```

### Three-Layer Architecture

```
┌─────────────────┐
│  Data Models    │  models.py: Immutable dataclasses
│                 │  (IPSDiagnostics, DRDiagnostics)
└────────┬────────┘
         │
┌────────▼────────┐
│  Computation    │  weights.py, dr.py, overlap.py, transport.py:
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

### CJEDiagnostics

Unified diagnostics for paper-ready reporting. Simplifies IPSDiagnostics and DRDiagnostics into a single class focused on two key questions:

1. **Can we make level claims?** (identification/coverage risk)
2. **Are CIs honest?** (sampling/variance risk)

**Key fields**:
- `coverage_risk`: "low", "medium", "high", or "critical"
- `variance_risk`: "low", "medium", "high", or "critical"
- `refuse_level_claims`: Boolean gate for point estimates
- `refuse_inference`: Boolean gate for CI reliability

**Key properties**:
- `can_make_level_claims` - Whether point estimates are reliable
- `has_honest_inference` - Whether confidence intervals are reliable
- `overall_risk` - Combined risk assessment

**Factory methods**:
```python
from cje.diagnostics import CJEDiagnostics, IPSDiagnostics, DRDiagnostics

# Convert from IPS diagnostics
unified = CJEDiagnostics.from_ips_diagnostics(ips_diag)

# Convert from DR diagnostics
unified = CJEDiagnostics.from_dr_diagnostics(dr_diag)

# Check risk levels
print(unified.summary())
# Output: "CalibratedIPS (calibrated) | N=950/1000 | Coverage: low | Variance: medium | ESS=45.2%"

if unified.can_make_level_claims:
    print("Safe to compare policy levels")
if unified.has_honest_inference:
    print("Confidence intervals are reliable")
```


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

### TTC (Target-Typicality Coverage)
TTC = Hellinger affinity = E[√w]. Measures what fraction of the target distribution's mass falls in regions where the logger has meaningful coverage.

**Decision rule:** If TTC < 0.7, logs-only IPS will fail regardless of weight stabilization.
- **TTC > 0.7**: Good overlap, IPS may work
- **TTC 0.3-0.7**: Marginal, consider DR methods
- **TTC < 0.3**: Poor overlap, IPS will fail

```python
from cje.diagnostics import compute_ttc, compute_cle_diagnostics

# Quick TTC check
ttc = compute_ttc(importance_weights)
if ttc < 0.7:
    print("Use Direct or DR methods instead of IPS")

# Full CLE diagnostics
cle = compute_cle_diagnostics(weights, target_logprobs)
print(cle.summary())
# CLE: POOR | TTC=12.5% (IPS will fail) | coverage_penalty=2.31 | shape_mismatch=25.4 | cle_factor=58.7x
```

### CLE (Coverage-Limited Efficiency) Diagnostics
The CLE bound explains why IPS can fail even with high ESS:

```
SE(Ψ̂) ≥ (σ_T · α) / √(β·n) · √(1 + χ²)
```

Components:
- **α** = target mass on target-typical region T
- **β** = logger mass on T (logger coverage)
- **Coverage penalty α/√β**: Explodes when logger rarely visits target-typical regions
- **Shape mismatch √(1+χ²)**: Inflates variance even with good coverage
- **CLE factor**: Combined inflation = coverage_penalty × shape_mismatch

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

## Transportability Audit

**Use case:** Test if a calibrator fitted on policy A/era 1 can be safely reused for policy B/era 2.

**How it works:**
1. Computes mean residual: δ̂ = E[Y - f̂(S)] on target probe
2. Constructs 95% CI: δ̂ ± 1.96 × SE
3. Tests unbiasedness: Is 0 ∈ CI?
4. Checks regional residuals by decile (catches non-uniform drift)
5. Verifies coverage of score range

**Basic Usage:**

```python
import json
from sklearn.isotonic import IsotonicRegression
from cje.diagnostics import audit_transportability, plot_transport_comparison

# Step 1: Fit calibrator on source (e.g., base policy with oracle labels)
source_records = [json.loads(line) for line in open("base_responses.jsonl")]
source_with_oracle = [r for r in source_records if r.get("oracle_label") is not None]
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(
    [r["judge_score"] for r in source_with_oracle],
    [r["oracle_label"] for r in source_with_oracle]
)

# Step 2: Load 40-60 probe samples on target (just dicts with judge_score + oracle_label)
probe = [json.loads(line) for line in open("gpt4_mini_probe.jsonl")]

# Step 3: Test transportability
diag = audit_transportability(
    calibrator,
    probe,  # List[dict] - no wrapper needed!
    group_label="policy:gpt-4-mini"
)

# Step 4: Check result
print(diag.summary())
# Output: "Transport: PASS | Group: policy:gpt-4-mini | N=50 | δ̂: +0.012 (CI: [-0.008, +0.032])"

# Step 5: Visualize
diag.plot()  # Decile-level residuals

# Or compare multiple policies at once:
# results = {"clone": diag_clone, "premium": diag_premium}
# fig = plot_transport_comparison(results)

# Step 6: Take action
if diag.status == "PASS":
    # δ̂ CI includes 0 → calibrator is unbiased on target
    print("✅ Calibrator transports - safe to reuse")

elif diag.status == "WARN":
    # δ̂ CI excludes 0 but |δ̂| < 0.05 → small bias
    print(f"⚠️ Small bias detected: δ̂={diag.delta_hat:+.3f}")
    print(f"Action: {diag.recommended_action}")  # "monitor"
    # Consider mean anchoring or collecting more probe labels

elif diag.status == "FAIL":
    # |δ̂| ≥ 0.05 → clear systematic bias
    print(f"🔴 Clear drift detected: δ̂={diag.delta_hat:+.3f}")
    print(f"Action: {diag.recommended_action}")  # "refit_two_stage"
    # Collect more oracle labels on target and refit calibrator
```

**Status Classification:**

Tests H₀: E[Y - f̂(S)] = 0 for target policy using parametric 95% CI.

- **PASS**: 0 ∈ CI → calibrator is unbiased on target
- **WARN**: 0 ∉ CI but |δ̂| < 0.05 → small but detectable bias
- **FAIL**: 0 ∉ CI and |δ̂| ≥ 0.05 → large systematic bias

**Recommended Actions:**
- `none` - Safe to reuse calibrator as-is
- `monitor` - Small bias detected, monitor or collect more probe labels
- `refit_two_stage` - Large bias, refit with pooled data or use two-stage calibration

**Probe Sampling Strategy:**

For best coverage, stratify by risk index:

```python
# Stratified sampling for better score range coverage
probe = stratified_sample(
    target_data,
    strata_field="risk_index_decile",
    n_per_stratum=5,  # 50 total for 10 deciles
    oracle_label_required=True
)
```

**Common Use Cases:**

1. **Policy change**: GPT-4 → GPT-4-mini
2. **Temporal shift**: Q1 logs → Q2 logs (without timestamps)
3. **Domain shift**: Customer support → Sales queries
4. **Judge update**: After rubric or model changes
5. **Cross-user**: Power users → Novice users

**Example: Cross-Era Transport**

```python
import json
from sklearn.isotonic import IsotonicRegression
from cje.diagnostics import audit_transportability

# Fit on Q1 data
q1_records = [json.loads(line) for line in open("q1_logs.jsonl")]
q1_with_oracle = [r for r in q1_records if r.get("oracle_label") is not None]
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit([r["judge_score"] for r in q1_with_oracle], [r["oracle_label"] for r in q1_with_oracle])

# Test on Q2 with 50 probe labels
q2_probe = [json.loads(line) for line in open("q2_probe.jsonl")]
diag = audit_transportability(calibrator, q2_probe, group_label="era:Q2-2025")

print(diag.summary())
# Output: "Transport: FAIL | Group: era:Q2-2025 | N=50 | δ̂: -0.082 (CI: [-0.114, -0.050]) | Action: refit_two_stage"

# Interpretation: Judge systematically under-predicts by ~8% in Q2
# Action: Collect 100-200 Q2 oracle labels and refit
```

**Inspecting Individual Residuals:**

When calibration fails, inspect which samples have the worst errors:

```python
from cje.diagnostics import compute_residuals

# Compute residuals for each sample (sorted by worst overestimate first)
samples = compute_residuals(calibrator, probe_data)

# Inspect worst overestimates (where the judge was most fooled)
for s in samples[:3]:
    print(f"Residual: {s['residual']:.2f}")
    print(f"  Judge: {s['judge_score']:.2f} → Calibrated: {s['calibrated']:.2f}")
    print(f"  Oracle: {s['oracle_label']:.2f}")
    print(f"  Response: {s['response'][:100]}...")
```

Sort options:
- `sort_by="residual"` (default): Worst overestimates first (most negative residuals)
- `sort_by="abs_residual"`: Biggest errors first (regardless of direction)
- `sort_by=None`: Preserve original order

## Uncertainty Quantification: Two-Component Structure

CJE's uncertainty quantification properly accounts for **two independent sources of variance**:

1. **Main sampling uncertainty** (from the eval log/prompts) → cluster-robust SEs
2. **Calibrator uncertainty** (from the oracle slice) → OUA jackknife

Under the standard product-sample setup (oracle and eval datasets are independent), these components are **additive**:

```
Var_total = Var_CR + Var_OUA
```

This two-component structure is critical for honest inference. Ignoring clustering on the eval side can cause severe undercoverage (e.g., 86.9% instead of 95%), while ignoring calibrator uncertainty understates risk when oracle slices are small.

### Component 1: Cluster-Robust Variance (Eval-Side Dependence)

**What to cluster:** Any structure that creates dependence between rows in your evaluation data:
- **User/session clustering**: Multiple prompts from the same user or session
- **Time blocks**: Prompts collected in temporal batches
- **Conversation threads**: Multi-turn dialogues
- **Paired designs**: DM contrasts where the same prompt is used for multiple policies

**Why it matters:** Standard i.i.d. SEs assume independent samples. When prompts cluster (e.g., 5 prompts per user), the **effective** sample size is closer to the number of clusters (users), not the number of rows (prompts). Ignoring this inflates precision artificially.

**Computing cluster-robust variance:**

All CJE estimators are means of **per-row influence contributions**:
- **DM level** for policy π: `ψ_i = f(S_i^π) - V_hat`
- **DM paired contrast** (π vs π'): `ψ_i = [f(S_i^π) - f(S_i^π')] - Delta_hat`
- **IPS**: `ψ_i = w_tilde_i * R_i - V_hat`
- **DR**: `ψ_i = g_π(X_i) + w_tilde_i * (R_i - q_hat(X_i, A_i)) - V_hat`

Aggregate to cluster sums:
```
Ψ_g = Σ_{i: c(i)=g} ψ_i
```

Small-sample corrected cluster-robust variance:
```
Var_CR = (G/(G-1)) * (1/n²) * Σ_{g=1}^G Ψ_g²
```

where G is the number of clusters and n is the total number of rows.

**Edge cases:**
- **Few clusters (G < 15)**: Prefer wild-cluster bootstrap over asymptotic CR1
- **Two-way dependence** (e.g., user × day): Use two-way clustering formula (sum one-way variances, subtract intersection)
- **Time series**: Use moving-block bootstrap with block length ≈ n^(1/3)

### Component 2: Oracle Uncertainty Aware (OUA) Jackknife

**What it captures:** Uncertainty from learning the calibration function f(S) on a finite oracle slice.

**Why it matters:** When oracle size is small (common: 5-10% coverage), treating the calibrator as fixed drastically understates total uncertainty. OUA properly accounts for this.

**Computing OUA variance:**

1. Split oracle labels into K folds (same folds used for cross-fitting)
2. For each fold k:
   - Refit calibrator f^(-k) on oracle \ fold_k (let auto mode selection re-decide mono vs two-stage)
   - Recompute calibrated rewards R^(-k) = f^(-k)(S)
   - **For IPS/DR**: Re-select SIMCal weights (selection depends on R)
   - **For DR**: Refit outcome model (depends on R)
   - Compute point estimate θ_hat^(-k)

3. Jackknife variance:
```
Var_OUA = ((K-1)/K) * Σ_{k=1}^K (θ_hat^(-k) - θ_bar)²
```
where `θ_bar = (1/K) Σ_k θ_hat^(-k)`

**Key principle:** OUA captures **oracle-only** uncertainty. It holds the eval log fixed and only refits components that depend on calibrator outputs. This maintains independence with Var_CR.

### Combining the Components

Because oracle and eval are independent samples, the cross-term vanishes asymptotically:

```
Var_total = Var_CR + Var_OUA
SE_total = √Var_total
```

**Critical value selection:**

- **Large samples** (G ≥ 30, K ≥ 5): Use normal quantile (1.96 for 95% CI)
- **Small clusters or oracle**: Use Satterthwaite effective degrees of freedom:

```
df_eff = (Var_CR + Var_OUA)² / (Var_CR²/df_CR + Var_OUA²/df_OUA)

where df_CR = G - 1, df_OUA = K - 1
```

Then use t_{1-α/2, df_eff} critical value (e.g., qt(0.975, df_eff) in R).

### What to Report

For complete transparency, always report:

1. **Point estimate** with 95% CI using SE_total
2. **OUA share**: `Var_OUA / Var_total` (shows which component dominates)
3. **Cluster structure**: Number of clusters G, cluster size distribution
4. **Comparison**: i.i.d. SE vs cluster-robust SE (shows dependence penalty)

Example output:
```
Policy: gpt-4-mini
Estimate: 0.756 [0.712, 0.800]  (95% CI, df=42)

Variance decomposition:
  Cluster-robust (eval): 0.0012  (72%)
  OUA (oracle):          0.0005  (28%)
  Total:                 0.0017

Cluster structure: G=45 clusters (mean size=22, range [5, 67])
Standard errors: i.i.d.=0.032, cluster-robust=0.041 (28% inflation)
```

### Implementation Pattern (Pseudocode)

```python
# Component 1: Cluster-robust variance on eval log
def compute_cluster_robust_variance(psi, cluster_ids):
    """
    psi: per-row influence contributions (n,)
    cluster_ids: cluster identifier for each row (n,)
    Returns: Var_CR
    """
    clusters = pd.DataFrame({"psi": psi, "cluster": cluster_ids})
    cluster_sums = clusters.groupby("cluster")["psi"].sum()  # Ψ_g

    G = len(cluster_sums)
    n = len(psi)

    Var_CR = (G / (G - 1)) * (cluster_sums ** 2).sum() / (n ** 2)
    return Var_CR, G

# Component 2: OUA jackknife (oracle-only refits)
def compute_oua_variance(dataset, oracle_folds, estimator_config):
    """
    Refit calibrator K times, holding eval log fixed
    Returns: Var_OUA, K
    """
    K = len(oracle_folds)
    estimates = []

    for k in range(K):
        # Refit calibrator on oracle \ fold_k
        f_minus_k = fit_autocal(oracle_minus_fold=k)  # auto mode selection

        # Recompute calibrated rewards
        R_minus_k = f_minus_k(dataset.judge_scores)

        # For IPS/DR: re-select SIMCal (depends on R)
        if mode in ["ips", "dr"]:
            w_tilde = simcal_select(raw_weights, index=S, rewards=R_minus_k)

        # For DR: refit outcome model (depends on R)
        if mode == "dr":
            g_hat = crossfit_critic(X, A, R_minus_k)

        # Compute point estimate with refit components
        theta_k = compute_estimate(R_minus_k, w_tilde, g_hat)
        estimates.append(theta_k)

    theta_bar = np.mean(estimates)
    Var_OUA = ((K - 1) / K) * np.sum((estimates - theta_bar) ** 2)

    return Var_OUA, K

# Combine and construct CI
def construct_ci(estimate, Var_CR, Var_OUA, df_CR, df_OUA, alpha=0.05):
    """
    estimate: point estimate
    Var_CR, df_CR: cluster-robust component
    Var_OUA, df_OUA: oracle uncertainty component
    Returns: (lower, upper, df_eff)
    """
    Var_total = Var_CR + Var_OUA
    SE_total = np.sqrt(Var_total)

    # Satterthwaite effective df
    if Var_CR > 0 and Var_OUA > 0:
        df_eff = (Var_total ** 2) / (
            (Var_CR ** 2 / df_CR) + (Var_OUA ** 2 / df_OUA)
        )
    elif Var_CR > 0:
        df_eff = df_CR
    else:
        df_eff = df_OUA

    # Critical value
    if df_eff >= 30:
        crit = 1.96  # normal approximation
    else:
        from scipy.stats import t
        crit = t.ppf(1 - alpha/2, df_eff)

    lower = estimate - crit * SE_total
    upper = estimate + crit * SE_total

    return lower, upper, df_eff
```

### Usage Example

```python
from cje import analyze_dataset

# Analyze with cluster-robust inference
result = analyze_dataset(
    fresh_draws_dir="responses/",
    cluster_id_field="user_id",  # Enable cluster-robust SEs
    n_oracle_folds=5              # OUA jackknife folds (default)
)

# Access two-component variance decomposition
print(f"Cluster-robust variance: {result.var_cluster_robust:.4f}")
print(f"OUA variance: {result.var_oua:.4f}")
print(f"OUA share: {result.oua_share:.1%}")

# Confidence intervals use SE_total = √(Var_CR + Var_OUA)
for policy, est in result.estimates.items():
    se = result.standard_errors[policy]  # SE_total
    ci_lower, ci_upper = est - 1.96*se, est + 1.96*se
    print(f"{policy}: {est:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
```

### Common Pitfalls

**❌ Using i.i.d. SEs with clustered data**
- Result: Severe undercoverage (86.9% instead of 95% observed empirically)
- Fix: Always specify `cluster_id_field` when rows have dependence

**❌ Ignoring OUA with small oracle slices**
- Result: CIs too narrow, overconfidence in precision
- Fix: CJE computes OUA by default; check OUA share in reports

**❌ Not reporting which component dominates**
- Result: Unclear whether to add more prompts or more labels
- Fix: Always report OUA share—if >50%, labels are the bottleneck

**❌ Mixing cluster levels (e.g., clustering by session but pairing by user)**
- Result: Incorrect variance, wrong CIs
- Fix: Match cluster definition to the dependence structure; for nested clusters, use the coarsest level

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