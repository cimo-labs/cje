# CJE Interface

Simple, reliable LLM evaluation with automatic mode selection.

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
1. Detect the **mode** based on your data (Direct/IPS/DR)
2. Select the best **estimator** for that mode:
   - **Direct mode** → `direct` estimator
   - **IPS mode** → `calibrated-ips` estimator (default for IPS)
   - **DR mode** → `stacked-dr` estimator (default for DR)

### What are fresh draws?
Fresh draws are new responses from your target policies evaluated by the judge. For Direct mode, these are your only data source. For DR mode, they supplement logged data for better accuracy.

Format: JSONL files per policy in a directory (e.g., `responses/gpt4_responses.jsonl`)

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
```json
{
  "prompt_id": "0",
  "prompt": "User question",
  "response": "Model response",
  "policy": "gpt4",           // Optional if using separate files per policy
  "judge_score": 0.85,        // Required
  "oracle_label": 0.90        // Optional (for calibration - 50% coverage recommended)
}
```
Store as: `responses/gpt4_responses.jsonl`, `responses/claude_responses.jsonl`, etc.

**Calibration in Direct mode**: If 50% or more of fresh draws have `oracle_label`, Direct mode will automatically learn judge→oracle calibration and apply calibrated rewards. Otherwise, uses raw judge scores.

### IPS/DR Modes (logged data):
```json
{
  "prompt": "User question here",
  "response": "Model response here",
  "base_policy_logprob": -35.7,
  "target_policy_logprobs": {"policy_a": -33.1, "policy_b": -34.2},
  "judge_score": 0.85,        // Required
  "oracle_label": 0.90        // Optional (for calibration, 5-10% is enough)
}
```

Note: `judge_score` and `oracle_label` can be at top-level (preferred) or in `metadata` (backward compatible).

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
**Solution**: Ensure your data has `metadata.judge_score` field:
```python
# Check your data
import json
with open("logs.jsonl") as f:
    sample = json.loads(f.readline())
    print(sample.get("metadata", {}).get("judge_score"))  # Should not be None
```

## API Reference

### `analyze_dataset()`

```python
def analyze_dataset(
    logged_data_path: Optional[str] = None,  # NEW: Optional (for Direct mode)
    fresh_draws_dir: Optional[str] = None,
    estimator: str = "auto",  # NEW: Auto mode selection
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
    estimator_config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> EstimationResult:
```

**Parameters:**
- `logged_data_path`: Path to JSONL file with logged data (optional for Direct mode)
- `fresh_draws_dir`: Directory with fresh draw response files
- `estimator`: Estimator name or "auto" for automatic selection
  - Use "auto" (default) for automatic mode selection
  - Manual: `direct`, `calibrated-ips`, `stacked-dr`, `dr-cpo`, `tmle`, `mrdr`, etc.
- `judge_field`: Metadata field with judge scores (default: "judge_score")
- `oracle_field`: Metadata field with oracle labels (default: "oracle_label")
- `verbose`: Print detailed progress

**Returns:**
- `EstimationResult` with:
  - `.estimates`: Policy value estimates (numpy array)
  - `.standard_errors`: Standard errors for each estimate
  - `.diagnostics`: Diagnostic metrics (ESS, overlap quality, etc.)
  - `.metadata`: Mode, estimator, data sources, etc.

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
