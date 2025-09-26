# CJE Interface

Simple, reliable off-policy evaluation for LLM systems.

## Quick Start

```python
from cje import analyze_dataset

# Most robust analysis (default)
results = analyze_dataset(
    "your_logs.jsonl",
    fresh_draws_dir="responses/"  # Required for stacked-dr
)
print(f"Policy value: {results.estimates[0]:.3f} ± {1.96*results.standard_errors[0]:.3f}")

# Fast analysis (no fresh draws needed)  
results = analyze_dataset(
    "your_logs.jsonl",
    estimator="calibrated-ips"
)
```

## Choosing an Estimator

| Your Situation | Use This | Command |
|---------------|----------|---------|
| **Have fresh draws** → Most robust | `stacked-dr` (default) | `analyze_dataset("data.jsonl", fresh_draws_dir="responses/")` |
| **No fresh draws** → Fast & good | `calibrated-ips` | `analyze_dataset("data.jsonl", estimator="calibrated-ips")` |
| **Need triple robustness** → Robust to all errors | `tr-cpo-e` | `analyze_dataset("data.jsonl", estimator="tr-cpo-e", fresh_draws_dir="responses/")` |
| **Want orthogonalized IPS** → Robust calibration | `orthogonalized-ips` | `analyze_dataset("data.jsonl", estimator="orthogonalized-ips")` |
| **Debugging** → Baseline | `raw-ips` | `analyze_dataset("data.jsonl", estimator="raw-ips")` |

### What are fresh draws?
Fresh draws are new responses from your target policy π' that have been scored by the judge. Required for doubly-robust (DR) methods. Store as JSONL files in a directory, one per policy.

## Common Workflows

### Basic Analysis
```python
from cje import analyze_dataset

# Analyze with automatic defaults
results = analyze_dataset("logs.jsonl")

# Check reliability
if results.diagnostics.weight_ess < 0.1:
    print("⚠️ Low effective sample size - results may be unreliable")

# Get estimates for each policy
for i, policy in enumerate(results.metadata["target_policies"]):
    print(f"{policy}: {results.estimates[i]:.3f} ± {1.96*results.standard_errors[i]:.3f}")
```

### Comparing Policies
```python
# Run robust analysis
results = analyze_dataset(
    "production_logs.jsonl",
    fresh_draws_dir="fresh_responses/"
)

# Compare policies
baseline_idx = 0
for i in range(1, len(results.estimates)):
    diff = results.estimates[i] - results.estimates[baseline_idx]
    # Note: This is a simplified comparison - proper inference would account for correlation
    print(f"Policy {i} vs baseline: {diff:+.3f}")
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

Minimal required fields:
```json
{
  "prompt": "User question here",
  "response": "Model response here", 
  "base_policy_logprob": -35.7,
  "target_policy_logprobs": {"policy_a": -33.1, "policy_b": -34.2},
  "metadata": {
    "judge_score": 0.85,      // Required
    "oracle_label": 0.90       // Optional (for calibration)
  }
}
```

Fresh draws format (same structure, in separate files per policy):
- `responses/policy_a_responses.jsonl`
- `responses/policy_b_responses.jsonl`

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
    dataset_path: str,
    estimator: str = "stacked-dr",  # Default: most robust
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label", 
    estimator_config: Optional[Dict[str, Any]] = None,
    fresh_draws_dir: Optional[str] = None,
    verbose: bool = False,
) -> EstimationResult:
```

**Parameters:**
- `dataset_path`: Path to JSONL file with logged data
- `estimator`: One of: stacked-dr, calibrated-ips, raw-ips, dr-cpo, oc-dr-cpo, tr-cpo, orthogonalized-ips, mrdr, tmle
- `fresh_draws_dir`: Directory with fresh draw responses (required for DR methods)
- `verbose`: Print progress messages

**Returns:**
- `EstimationResult` with `.estimates`, `.standard_errors`, `.diagnostics`, `.metadata`

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

1. **Default to `stacked-dr`** with fresh draws for most robust results
2. **Use `calibrated-ips`** when you need speed or don't have fresh draws
3. **Check diagnostics** especially `weight_ess` for reliability
4. **Fresh draws required** for all DR methods (stacked-dr, dr-cpo, mrdr, tmle)

For more details, see the [full documentation](https://causal-judge-evaluation.readthedocs.io).
