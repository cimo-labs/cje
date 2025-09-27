# CJE - Engineering Documentation

## Interface

### Primary Function
```python
from cje import analyze_dataset

result = analyze_dataset(
    dataset_path: str,                    # Required: Path to JSONL file
    estimator: str = "stacked-dr",        # Estimator choice (see below)
    judge_field: str = "judge_score",     # Where to find judge scores
    oracle_field: str = "oracle_label",   # Where to find oracle labels
    estimator_config: Dict = None,        # Estimator-specific config
    fresh_draws_dir: str = None,          # Path to fresh draws (required for DR)
    verbose: bool = False                 # Progress logging
) -> EstimationResult
```

### Return Type
```python
@dataclass
class EstimationResult:
    estimates: np.ndarray           # Shape: [n_policies], values in [0,1]
    standard_errors: np.ndarray     # Shape: [n_policies], SE estimates
    n_samples_used: Dict[str, int]  # Samples per policy after filtering
    method: str                     # Estimator name used
    influence_functions: Dict        # Per-sample contributions (advanced)
    diagnostics: IPSDiagnostics      # Health metrics (see below)
    robust_standard_errors: np.ndarray    # Cluster-robust SEs
    robust_confidence_intervals: List     # 95% CIs
    metadata: Dict                        # Run metadata
```

## Input Data Format

### Required JSONL Structure
```json
{
  "prompt": "string",                    // Required: Input text
  "response": "string",                 // Required: Generated output
  "base_policy_logprob": -12.34,        // Required: Log P(response|prompt) under logging policy
  "target_policy_logprobs": {           // Required: Log probs under target policies
    "policy_a": -11.23,
    "policy_b": -13.45
  },
  "metadata": {                         // Required: Contains scores
    "judge_score": 0.75,               // Required: Judge's score in [0,1]
    "oracle_label": 0.80               // Optional: Ground truth (for some samples)
  }
}
```

### Data Validation Rules
- All log probabilities must be ≤ 0 (negative or zero)
- Judge scores must be in [0, 1]
- Oracle labels must be in [0, 1] when present
- Missing log probs → sample skipped with warning
- At least 10% samples need oracle labels for calibration

## Output Usage

### Basic Usage
```python
results = analyze_dataset("data.jsonl", estimator="calibrated-ips")

# Get point estimate for first policy
estimate = results.estimates[0]  # e.g., 0.723

# Get 95% confidence interval
lower, upper = results.robust_confidence_intervals[0]  # e.g., (0.701, 0.745)

# Check reliability
if results.diagnostics.weight_ess < 0.1:
    print("Warning: Low effective sample size")
```

### Diagnostics Structure
```python
diagnostics.weight_ess          # Effective sample size (0-1, higher=better)
diagnostics.weight_status       # Status enum: GOOD/MARGINAL/CRITICAL
diagnostics.overlap_quality     # "good"/"marginal"/"poor"
diagnostics.n_samples_valid     # Samples with valid weights
diagnostics.n_samples_total     # Total input samples
diagnostics.summary()           # Human-readable summary
```

## Error Scenarios

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: DR estimators require fresh draws` | Using stacked-dr without fresh_draws_dir | Use `estimator="calibrated-ips"` or provide fresh draws |
| `ValueError: No oracle labels found` | Missing oracle_label in metadata | Add oracle labels to 5-10% of samples |
| `ValueError: Judge field 'X' not found` | Wrong judge_field name | Check your metadata structure |
| `NaN in estimates` | Catastrophic weight distribution | Check policy overlap, use DR methods |
| `FileNotFoundError` | Invalid dataset_path | Verify file exists |

## Estimator Selection

### Quick Decision Tree
```
Do you have fresh draws (new responses from target policy)?
├─ YES → Use "stacked-dr" (default, most robust)
└─ NO  → Use "calibrated-ips" (fast, reliable)
```

### Performance Characteristics

| Estimator | Speed | Memory | Robustness | Requirements |
|-----------|-------|--------|------------|--------------|
| calibrated-ips | Fast (10K/sec) | O(n) | Good | Judge scores only |
| stacked-dr | Slow (1K/sec) | O(n²) | Excellent | Fresh draws required |
| raw-ips | Fastest | O(n) | Poor | Baseline only |

## Dependencies

### Required
```
python >= 3.9
numpy >= 1.21
scipy >= 1.7
scikit-learn >= 1.0
pydantic >= 2.0
```

### Installation
```bash
pip install -e .
# or
poetry install
```

## Validation

### Quick Test
```python
# Minimal working example
from cje import analyze_dataset
import json

# Create test data
test_data = [
    {
        "prompt": f"Question {i}",
        "response": f"Answer {i}",
        "base_policy_logprob": -10.0,
        "target_policy_logprobs": {"target": -9.5},
        "metadata": {
            "judge_score": 0.5 + i*0.01,
            "oracle_label": 0.6 if i < 10 else None
        }
    }
    for i in range(100)
]

with open("test.jsonl", "w") as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n")

# Should work
result = analyze_dataset("test.jsonl", estimator="calibrated-ips")
assert 0 <= result.estimates[0] <= 1
print(f"Success: estimate = {result.estimates[0]:.3f}")
```

## CLI Interface

### Basic Commands
```bash
# Analyze with calibrated-ips (no fresh draws needed)
python -m cje analyze data.jsonl --estimator calibrated-ips -o results.json

# Analyze with stacked-dr (requires fresh draws)
python -m cje analyze data.jsonl --fresh-draws-dir responses/ -o results.json

# Validate data format
python -m cje validate data.jsonl
```

### Output Format (results.json)
```json
{
  "estimates": {"policy_a": 0.723, "policy_b": 0.691},
  "standard_errors": {"policy_a": 0.012, "policy_b": 0.014},
  "confidence_intervals": {
    "policy_a": [0.699, 0.747],
    "policy_b": [0.664, 0.718]
  },
  "diagnostics": {
    "ess": 0.423,
    "n_samples": 10000,
    "status": "GOOD"
  }
}
```

## Memory and Performance

### Scaling Characteristics
- **Memory**: O(n) for IPS methods, O(n²) for DR with cross-fitting
- **Time**: Linear in n_samples, quadratic in n_policies
- **Practical limits**:
  - IPS: 1M samples feasible
  - DR: 100K samples recommended max

### Performance Tips
1. Filter data before CJE if possible
2. Use `calibrated-ips` for initial exploration
3. Subsample for quick iteration
4. Monitor `diagnostics.weight_ess` - low values mean slow convergence

## Advanced Features

### Custom Estimator Config
```python
results = analyze_dataset(
    "data.jsonl",
    estimator="stacked-dr",
    estimator_config={
        "n_folds": 10,          # More folds = better calibration
        "use_iic": False,       # Disable IIC (on by default)
        "oua_jackknife": True,  # Oracle uncertainty (on by default)
    }
)
```

### Accessing Influence Functions
```python
# Per-sample contributions to estimate
inf_funcs = results.influence_functions["policy_a"]  # shape: [n_samples]
# These sum to zero and have variance equal to SE²
```

### Fresh Draws Format
Same as input data but represents new responses from target policy:
```json
{
  "prompt_id": "p123",           // Must match original prompt
  "response": "New answer...",   // Fresh sample from target
  "judge_score": 0.83,          // Judge evaluation of new response
  "draw_idx": 0                  // Index if multiple draws per prompt
}
```

## Common Patterns

### Multi-Policy Comparison
```python
results = analyze_dataset("data.jsonl", estimator="calibrated-ips")
policies = results.metadata["target_policies"]
best_idx = results.estimates.argmax()
print(f"Best policy: {policies[best_idx]} ({results.estimates[best_idx]:.3f})")
```

### Reliability Gating
```python
def get_reliable_estimate(data_path):
    results = analyze_dataset(data_path, estimator="calibrated-ips")
    if results.diagnostics.weight_ess < 0.05:
        raise ValueError("Insufficient overlap for reliable estimation")
    return results.estimates[0]
```

### Batch Processing
```python
import glob
for file in glob.glob("experiments/*.jsonl"):
    try:
        results = analyze_dataset(file, estimator="calibrated-ips")
        print(f"{file}: {results.estimates[0]:.3f}")
    except Exception as e:
        print(f"{file}: Failed - {e}")
```

## Module Documentation

Each subdirectory in `cje/` contains a developer-oriented README with implementation details:

- **`cje/estimators/README.md`** - Estimator hierarchy, implementation details, adding new estimators
- **`cje/diagnostics/README.md`** - Diagnostic system architecture, adding new metrics, gate thresholds
- **`cje/data/README.md`** - Data models, validation pipeline, custom data sources
- **`cje/calibration/README.md`** - Calibration algorithms, SIMCal implementation, isotonic regression
- **`cje/interface/README.md`** - High-level API, CLI implementation, service architecture
- **`cje/utils/README.md`** - Utility functions, export formats
- **`cje/visualization/README.md`** - Plotting utilities, diagnostic visualizations
- **`cje/teacher_forcing/README.md`** - Fresh draw generation

These READMEs are not user-facing documentation but provide valuable technical context for developers working with or extending the codebase.