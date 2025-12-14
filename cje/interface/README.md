# CJE Interface

Simple, reliable LLM evaluation with automatic mode selection and AutoCal-R calibration.

## Quick Start

CJE automatically selects the best mode and estimator for your data:

```python
from cje import analyze_dataset

# Mode 1: Direct (simplest - compare policies on eval set)
results = analyze_dataset(fresh_draws_dir="responses/")

# Mode 2: IPS (counterfactual with logged data)
results = analyze_dataset(logged_data_path="logs.jsonl")  # Auto-selects IPS mode

# Mode 3: DR (most accurate - both logged data and fresh draws)
results = analyze_dataset(
    logged_data_path="logs.jsonl",
    fresh_draws_dir="responses/"  # Auto-selects DR mode
)

# Print results
print(f"Policy value: {results.estimates[0]:.3f} ± {1.96*results.standard_errors[0]:.3f}")
```

## Three Analysis Modes

| Mode | Data Needed | Estimand | When to Use |
|------|-------------|----------|-------------|
| **Direct** | Fresh draws only | On-policy comparison | Simplest setup, no logprobs needed |
| **IPS** | Logged data with logprobs | Counterfactual deployment | Have production logs, want fast estimates |
| **DR** | Both logged + fresh draws | Counterfactual (most accurate) | High-stakes decisions, maximum accuracy |

### Automatic Mode Selection

Use `estimator="auto"` (default) and CJE will:
1. Detect the **mode** based on your data (Direct/IPS/DR) using the 3-rule system
2. Select the best **estimator** for that mode:
   - **Direct mode** → `direct` estimator
   - **IPS mode** → `calibrated-ips` estimator (IPS with variance-reduced weights via SIMCal)
   - **DR mode** → `stacked-dr` estimator (ensemble of DR-CPO, TMLE, MRDR, OC-DR-CPO, TR-CPO-E)

**Note:** In the paper, "Calibrated DR" refers to DR mode, which defaults to `stacked-dr` in the implementation. Stacked DR is an optimal convex combination of multiple DR estimators that typically outperforms any single variant.

### How Mode Detection Works

When you use `estimator="auto"` (the default), CJE automatically detects the mode using a **simple 3-rule system** based on available data:

**Decision rules:**
1. **fresh_draws + logged_data** → DR mode (doubly robust - best accuracy)
2. **logged_data only** → IPS mode (importance sampling - counterfactual)
3. **fresh_draws only** → Direct mode (on-policy comparison)

**Automatic filtering:** If your logged data has incomplete logprobs, CJE will:
- Automatically filter to only samples with complete logprobs
- Warn you about coverage (what % of samples were usable)
- Proceed with the filtered subset

A sample has "complete logprobs" if:
- `base_policy_logprob` is not None
- `target_policy_logprobs[policy]` exists for ALL target policies (not None)

Example: If you have 1000 logged samples but only 400 have complete logprobs:
```python
# CJE filters to 400 valid samples, warns about 40% coverage
# With fresh draws → DR mode using 400 samples
# Without fresh draws → IPS mode using 400 samples (with low coverage warning)
```

**Mode selection metadata:** Results include `result.metadata["mode_selection"]` with:
- `mode`: Selected mode ("dr", "ips", or "direct")
- `estimator`: Actual estimator used (e.g., "stacked-dr")
- `logprob_coverage`: Coverage fraction
- `has_fresh_draws`: Whether fresh draws were provided
- `has_logged_data`: Whether logged data was provided
- `reason`: Human-readable explanation of selection

**Overriding automatic selection:**
You can explicitly choose a mode/estimator instead of using `"auto"`:
```python
# Force IPS mode even with fresh draws available
results = analyze_dataset(
    logged_data_path="logs.jsonl",
    fresh_draws_dir="responses/",
    estimator="calibrated-ips"  # Explicitly choose IPS instead of auto DR
)

# Force Direct mode instead of auto-selected DR
results = analyze_dataset(
    logged_data_path="logs.jsonl",
    fresh_draws_dir="responses/",
    estimator="direct"  # Use Direct mode for on-policy comparison
)

# Choose specific DR variant
results = analyze_dataset(
    logged_data_path="logs.jsonl",
    fresh_draws_dir="responses/",
    estimator="tmle"  # Use TMLE instead of default stacked-dr
)
```

### What are fresh draws?
Fresh draws are new responses from your target policies evaluated by the judge. For Direct mode, these are your only data source. For DR mode, they supplement logged data for better accuracy.

**Format:** One JSONL file per policy in `fresh_draws_dir`. Policy name is inferred from filename.

**Example:** `responses/clone_responses.jsonl` → policy name is `"clone"`

### ⚠️ Policy Name Matching (IPS/DR Modes)

**CRITICAL:** Policy names must match EXACTLY between logged data and fresh draw filenames.

**How policies are detected:**
1. **From logged data:** Policy names = keys in `target_policy_logprobs` dict
2. **From fresh draws:** Policy names = extracted from `{policy}_responses.jsonl` filenames

**Example - Correct matching:**
```json
// logs.jsonl (defines policies via keys)
{
  "target_policy_logprobs": {
    "clone": -14.7,
    "premium": -18.3,
    "unhelpful": -42.1
  }
}
```

```
responses/  (filenames must match keys exactly)
├── clone_responses.jsonl      ✅ matches "clone"
├── premium_responses.jsonl    ✅ matches "premium"
└── unhelpful_responses.jsonl  ✅ matches "unhelpful"
```

**Common mistake - Name mismatch:**
```json
// Logged data has "gpt-4"
{"target_policy_logprobs": {"gpt-4": -14.7}}

// But file is named "gpt4_responses.jsonl"
❌ Error: No fresh draw file found for policy 'gpt-4'
```

**Fix:** Use identical names everywhere. If logged data has `"gpt-4"`, file must be `gpt-4_responses.jsonl`.

## Common Workflows

### Basic Analysis (Direct Mode)
```python
from cje import analyze_dataset

# Simplest workflow - just fresh draws
results = analyze_dataset(fresh_draws_dir="responses/")

# Get estimates for each policy
for i, policy in enumerate(results.metadata["target_policies"]):
    print(f"{policy}: {results.estimates[i]:.3f} ± {1.96*results.standard_errors[i]:.3f}")

# Note: Direct mode auto-discovers policies from filenames
print(f"Found policies: {results.metadata['target_policies']}")
```

### IPS Analysis (With Logged Data)
```python
# Analyze logged production data
results = analyze_dataset(logged_data_path="logs.jsonl", estimator="calibrated-ips")

# Check reliability (important for IPS!)
if results.diagnostics.weight_ess < 0.1:
    print("⚠️ Low effective sample size - consider using DR mode with fresh draws")

# Get estimates
for i, policy in enumerate(results.metadata["target_policies"]):
    print(f"{policy}: {results.estimates[i]:.3f} ± {1.96*results.standard_errors[i]:.3f}")
```

### DR Analysis (Maximum Accuracy)
```python
# Combine logged data with fresh draws for best accuracy
results = analyze_dataset(
    logged_data_path="production_logs.jsonl",
    fresh_draws_dir="responses/",
    estimator="stacked-dr"  # or "auto"
)

# Compare policies using built-in method
baseline_idx = 0
for i in range(1, len(results.estimates)):
    comparison = results.compare_policies(i, baseline_idx)
    sig = "*" if comparison["significant"] else ""
    print(f"Policy {i} vs baseline: {comparison['difference']:+.3f} (p={comparison['p_value']:.3f}) {sig}")
```

### Export Results
```python
# Save to JSON
results = analyze_dataset("logs.jsonl")
with open("results.json", "w") as f:
    json.dump({
        "estimates": results.estimates.tolist(),
        "standard_errors": results.standard_errors.tolist(),
        "ess": results.diagnostics.weight_ess if results.diagnostics else None
    }, f)
```

### Visualize Results
```python
from cje import analyze_dataset, plot_policy_estimates

# Run analysis
results = analyze_dataset(fresh_draws_dir="responses/")

# Option 1: Quick plot with convenience method
results.plot_estimates(
    base_policy_stats={"mean": 0.72, "se": 0.01},
    save_path="estimates.png"
)

# Option 2: Direct import for more control
plot_policy_estimates(
    estimates={"policy_a": 0.75, "policy_b": 0.68},
    standard_errors={"policy_a": 0.02, "policy_b": 0.03},
    oracle_values={"policy_a": 0.74, "policy_b": 0.69}  # Optional
)
```

**Jupyter notebooks:** Results auto-display as formatted HTML tables when evaluated in a cell.

See [`cje/visualization/README.md`](../visualization/README.md) for all available visualizations.

## Command Line Interface

```bash
# Basic usage
python -m cje analyze logs.jsonl

# With fresh draws (for robust estimation)
python -m cje analyze logs.jsonl --fresh-draws-dir responses/

# Fast mode (no fresh draws)
python -m cje analyze logs.jsonl --estimator calibrated-ips

# Save results
python -m cje analyze logs.jsonl -o results.json

# Validate data format
python -m cje validate logs.jsonl --verbose
```

## Data Format

### Direct Mode (fresh draws only):

**File naming:** One file per policy with pattern `{policy}_responses.jsonl`

**Example structure:**
```
responses/
├── clone_responses.jsonl
├── premium_responses.jsonl
└── unhelpful_responses.jsonl
```

**Record format** (inside each file):
```json
{
  "prompt_id": "arena_0",
  "judge_score": 0.85,        // Required: judge evaluation
  "oracle_label": 0.86,       // Optional: ground truth for AutoCal-R
  "prompt": "User question",  // Optional: for reference
  "response": "Model response" // Optional: for reference
}
```

**Note:** Policy name is inferred from filename (e.g., `clone_responses.jsonl` → policy `"clone"`). Do NOT include a `"policy"` field in the records.

**AutoCal-R in Direct mode**: If any fresh draws have `oracle_label`, Direct mode automatically applies AutoCal-R to learn judge→oracle calibration and uses calibrated rewards. Otherwise, uses raw judge scores. More oracle labels = better calibration (5-10% is often sufficient).

### IPS/DR Modes (logged data):
```json
{
  "prompt": "User question here",
  "response": "Model response here",
  "base_policy_logprob": -14.7,
  "target_policy_logprobs": {
    "clone": -14.7,
    "parallel_universe_prompt": -18.3,
    "unhelpful": -42.1
  },
  "judge_score": 0.85,        // Required
  "oracle_label": 0.86        // Optional (for calibration, 5-10% is enough)
}
```

Note: `judge_score` and `oracle_label` can be at top-level (preferred) or in `metadata` (backward compatible).

**Working example:** See [`examples/arena_sample/`](../../examples/arena_sample/) for complete dataset examples.

## Troubleshooting

### "ValueError: Estimator 'stacked-dr' requires fresh draws"
**Solution**: Either provide fresh draws or use calibrated-ips:
```python
# Option 1: Provide fresh draws
analyze_dataset("logs.jsonl", fresh_draws_dir="path/to/responses/")

# Option 2: Use calibrated-ips (no fresh draws needed)
analyze_dataset("logs.jsonl", estimator="calibrated-ips")
```

### "Low effective sample size" warning
**Cause**: Policies are very different from logging policy.
**Solutions**:
- Collect more data
- Use tighter variance cap (advanced)
- Consider if policies are too different for reliable estimation

### Missing judge scores
**Error**: "Judge field 'judge_score' not found"
**Solution**: Ensure your data has `judge_score` field:
```python
# Check your data
import json
with open("logs.jsonl") as f:
    sample = json.loads(f.readline())
    print(sample.get("judge_score"))  # Should not be None
```

### "Insufficient data" or no logprob coverage
**Error**: "No samples have complete logprobs and no fresh draws provided"

**Cause**: None of your samples have complete logprobs (both `base_policy_logprob` and all `target_policy_logprobs`)

**Check your coverage:**
```python
import json

with open("logs.jsonl") as f:
    samples = [json.loads(line) for line in f]

n_valid = 0
for s in samples:
    has_base = s.get("base_policy_logprob") is not None
    has_all_targets = all(
        s.get("target_policy_logprobs", {}).get(p) is not None
        for p in ["clone", "parallel_universe_prompt"]  # Your target policies
    )
    if has_base and has_all_targets:
        n_valid += 1

print(f"Logprob coverage: {n_valid}/{len(samples)} = {n_valid/len(samples):.1%}")
```

**Solutions:**
1. **Compute missing logprobs** using `cje/teacher_forcing/` (see README section on "Generating Log Probabilities")
2. **Provide fresh draws** to use Direct mode (no logprobs needed)

**Low coverage warning:** If you see "⚠️ Low coverage", CJE will automatically filter to valid samples and proceed, but results may be less reliable with very few samples.

## API Reference

### `analyze_dataset()`

```python
def analyze_dataset(
    logged_data_path: Optional[str] = None,
    fresh_draws_dir: Optional[str] = None,
    calibration_data_path: Optional[str] = None,
    combine_oracle_sources: bool = True,
    estimator: str = "auto",
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
    calibration_covariates: Optional[List[str]] = None,
    include_response_length: bool = False,
    estimator_config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> EstimationResult:
```

**Parameters:**
- `logged_data_path`: Path to JSONL file with logged data (optional for Direct mode)
- `fresh_draws_dir`: Directory with fresh draw response files
- `calibration_data_path`: Path to dedicated calibration dataset with oracle labels. Used to learn judge→oracle mapping separately from evaluation data.
- `combine_oracle_sources`: Pool oracle labels from all sources (calibration + logged + fresh) for maximum data efficiency. Default: `True`. Set `False` to use only calibration_data_path.
- `estimator`: Estimator name or "auto" for automatic selection
  - Use "auto" (default) for automatic mode selection
  - Manual: `direct`, `calibrated-ips`, `stacked-dr`, `dr-cpo`, `tmle`, `mrdr`, etc.
- `judge_field`: Metadata field with judge scores (default: "judge_score")
- `oracle_field`: Metadata field with oracle labels (default: "oracle_label")
- `calibration_covariates`: Optional list of metadata field names to use as covariates in two-stage reward calibration (e.g., `["domain", "difficulty"]`). Helps handle confounding where judge scores at fixed S have different oracle outcomes based on observable features. Only works with two-stage or auto calibration mode.
- `include_response_length`: If True, automatically includes response length (word count) as a covariate. Computed as `len(response.split())`. Requires all samples to have a `response` field. If True, `response_length` is prepended to `calibration_covariates`. Convenient for handling length bias.
- `verbose`: Print detailed progress

**Returns:**
- `EstimationResult` with:
  - `.estimates`: Policy value estimates (numpy array)
  - `.standard_errors`: Standard errors for each estimate
  - `.diagnostics`: Diagnostic metrics (ESS, overlap quality, etc.)
  - `.metadata`: Mode, estimator, data sources (see additional fields below)

**Additional metadata fields** (when using calibration_data_path):
- `metadata["oracle_sources"]`: Breakdown of oracle labels by source (calibration_data, logged_data, fresh_draws)
- `metadata["oracle_sources"]["distribution_mismatch"]`: KS test results comparing calibration vs. evaluation distributions

**At least one of `logged_data_path` or `fresh_draws_dir` must be provided.**

### CLI Commands

#### `analyze` - Run analysis
```bash
python -m cje analyze <dataset> [options]

Options:
  --estimator {stacked-dr,calibrated-ips,raw-ips,dr-cpo,oc-dr-cpo,tr-cpo,tr-cpo-e,orthogonalized-ips,mrdr,tmle}
  --fresh-draws-dir DIR     Directory with fresh draws
  --output FILE            Save results to JSON
  --verbose               Detailed output
  --quiet                Minimal output
```

#### `validate` - Check data format
```bash
python -m cje validate <dataset> [options]

Options:
  --verbose              Show detailed statistics
```

## Advanced Usage

### Dedicated Calibration Sets

Use a separate high-quality calibration dataset to learn the judge→oracle mapping:

```python
# Learn calibration from curated oracle set, apply to evaluation data
results = analyze_dataset(
    logged_data_path="production_logs.jsonl",      # 10K samples, 100 with oracle labels
    calibration_data_path="human_labels.jsonl",     # 500 samples, all with high-quality oracle labels
    estimator="calibrated-ips"
)

# Check oracle source breakdown
print(results.metadata["oracle_sources"])
# {
#   "calibration_data": {"n_oracle": 500, "coverage": 1.0},
#   "logged_data": {"n_oracle": 100, "coverage": 0.01},
#   "total_oracle": 600,  # Auto-combined for efficiency
#   "priority_order": ["calibration_data", "fresh_draws", "logged_data"]
# }
```

**Key features**:
- **Auto-combining** (default): Pools oracle labels from calibration_data + logged_data + fresh_draws for maximum data efficiency
- **Priority ordering**: calibration_data (highest) > fresh_draws > logged_data (lowest)
- **Conflict detection**: Warns if duplicate prompt_ids have different oracle values (>5% difference)

**Use cases**:
1. **Curated calibration sets**: You have expensive human labels in a separate file
2. **Distribution mismatch**: Your logged data has different characteristics than your eval set
3. **Temporal separation**: Oracle labels were collected at a different time

**Disable combining** to use only calibration data:
```python
results = analyze_dataset(
    logged_data_path="eval_data.jsonl",
    calibration_data_path="oracle_labels.jsonl",
    combine_oracle_sources=False,  # Use ONLY calibration data for learning f̂
    estimator="calibrated-ips"
)
```

**Metadata outputs**:
- `oracle_sources`: Breakdown of oracle labels by source
- `distribution_mismatch`: KS test comparing calibration vs. evaluation judge score distributions

### Transportability Auditing

Test if a calibrator fitted on one policy/era can safely transport to another using a cheap probe protocol (40-60 oracle labels):

```python
import json
from cje import analyze_dataset
from cje.diagnostics import audit_transportability, plot_transport_comparison

# analyze_dataset automatically fits and exposes the calibrator
results = analyze_dataset(fresh_draws_dir="responses/")

# Test transport to new policy with 50-sample probe
# Just load as list of dicts - no special wrapper needed!
probe = [json.loads(line) for line in open("target_policy_probe.jsonl")]
diag = audit_transportability(
    results.calibrator,  # Calibrator from analysis
    probe,  # List[dict] with judge_score and oracle_label
    group_label="policy:gpt-4-mini"
)

# Check result
print(diag.summary())
# Transport: PASS | Group: policy:gpt-4-mini | N=50 | δ̂: +0.012 (CI: [-0.008, +0.032])

# Visualize single policy
diag.plot()  # Shows decile-level residuals

# Compare multiple policies
audits = {}
for policy in ["clone", "premium", "unhelpful"]:
    probe = [json.loads(line) for line in open(f"{policy}_probe.jsonl")]
    audits[policy] = audit_transportability(results.calibrator, probe, group_label=f"policy:{policy}")

fig = plot_transport_comparison(audits, title="Transportability Audit")

# Handle failures
if diag.status == "FAIL":
    if diag.recommended_action == "refit_two_stage":
        # Regional miscalibration - need full refit
        print("⚠️ Calibrator does not transport. Collect more oracle labels and refit.")
```

**When to audit transport:**
- Applying calibrator to different policy than training data
- Reusing calibrator across time periods (e.g., Q1 → Q2)
- After judge model updates or prompt changes
- When distribution shift is suspected

**Traffic-light interpretation:**
- **PASS** (green): Safe to reuse calibrator
- **WARN** (orange): Marginal issues, monitor or consider mean anchoring
- **FAIL** (red): Must refit or apply corrections

**How it works:**
1. Computes global mean residual δ̂ = E[Y - f(S)] and 95% CI
2. Checks regional residuals by risk-index deciles
3. Verifies coverage of probe within calibrator's training range
4. Returns actionable recommendations based on failure mode

**Probe protocol:**
- 40-60 oracle labels recommended (cheap validation)
- Stratify by risk index for better coverage
- Can pool across multiple target policies for efficiency

**See also:** `cje/diagnostics/README.md` for details on transportability diagnostics.

### Covariate Support

Handle judge bias using observable features as covariates in two-stage calibration:

```python
# Example 1: Include response_length covariate (auto-computed)
results = analyze_dataset(
    logged_data_path="logs.jsonl",
    include_response_length=True,  # Auto-compute response length
    estimator="calibrated-ips"
)

# Example 2: Add custom metadata covariates with response_length
# Assumes your data has "domain" and "difficulty" in metadata
results = analyze_dataset(
    logged_data_path="logs.jsonl",
    include_response_length=True,
    calibration_covariates=["domain", "difficulty"],  # Additional covariates
    estimator="stacked-dr",
    fresh_draws_dir="responses/"
)
# Effective covariates: ["response_length", "domain", "difficulty"]

# Example 3: Judge score only (default - no covariates)
results = analyze_dataset(
    logged_data_path="logs.jsonl",
    estimator="calibrated-ips"
)

# Example 4: Custom covariates without response_length
results = analyze_dataset(
    logged_data_path="logs.jsonl",
    calibration_covariates=["domain"],  # Only domain, no response_length
    estimator="calibrated-ips"
)
```

**When to use covariates:**
- **Length bias**: Judge scores vary by response length at fixed oracle quality
- **Domain effects**: Judge miscalibration differs across domains (e.g., math vs. creative writing)
- **Task heterogeneity**: Observable features predict judge-oracle disagreement

**How it works:**
1. Two-stage calibration learns g(S, X_cov) → rank → isotonic
2. Covariates help handle non-monotone patterns in judge scores
3. DR estimators automatically use covariates in outcome models
4. All modes (Direct, IPS, DR) support covariate-adjusted calibration

**Requirements:**
- Covariate fields must exist in `sample.metadata` for all samples
- When using `include_response_length=True`, all samples must have a `"response"` field
- Covariates work with two-stage or auto calibration mode (not monotone-only)

**See also:** `cje/calibration/README.md` for details on two-stage calibration with covariates.

### Custom Configuration
```python
results = analyze_dataset(
    "logs.jsonl",
    estimator="dr-cpo",
    estimator_config={
        "n_folds": 10,
        "use_calibrated_weights": True,
    },
    fresh_draws_dir="responses/"
)
```

### Hydra Support
For complex configurations, use Hydra:
```bash
python -m cje.interface.hydra_entry \
  dataset=logs.jsonl \
  estimator=stacked-dr \
  fresh_draws_dir=responses/ \
  estimator_config.n_folds=10
```

## Summary

**Three modes, three use cases:**

1. **Direct Mode** (`fresh_draws_dir` only)
   - Simplest setup - no logprobs needed
   - On-policy comparison: "Which policy is best on this eval set?"
   - Auto-discovers policies from filenames

2. **IPS Mode** (`logged_data_path` only)
   - Fast counterfactual estimates from logged data
   - Check `diagnostics.weight_ess` for reliability
   - Use when you can't generate fresh draws

3. **DR Mode** (both `logged_data_path` + `fresh_draws_dir`)
   - Maximum accuracy combining IPS and outcome modeling
   - Recommended for production decisions
   - Robust to model misspecification

**Best practice:** Use `estimator="auto"` and let CJE choose the right mode.

For more details, see the [examples](../../examples/) and full documentation.
