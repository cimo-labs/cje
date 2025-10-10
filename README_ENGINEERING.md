# CJE - Engineering Documentation

## Interface

### Primary Function
```python
from cje import analyze_dataset

result = analyze_dataset(
    logged_data_path: str = None,              # Path to logged data JSONL (for IPS/DR)
    fresh_draws_dir: str = None,               # Path to fresh draws (for DR/Direct)
    calibration_data_path: str = None,         # Optional: Dedicated calibration set
    combine_oracle_sources: bool = True,       # Auto-pool oracle labels (default: True)
    timestamp_field: str = None,               # Field for temporal drift detection
    check_drift: bool = False,                 # Enable drift detection
    estimator: str = "auto",                   # Estimator choice (see below)
    judge_field: str = "judge_score",          # Where to find judge scores
    oracle_field: str = "oracle_label",        # Where to find oracle labels
    estimator_config: Dict = None,             # Estimator-specific config
    verbose: bool = False                      # Progress logging
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

### For Direct Mode (fresh draws only)

**File structure:** One JSONL file per policy in `fresh_draws_dir`
```
responses/
├── policy_a_responses.jsonl
├── policy_b_responses.jsonl
```

**Record format:**
```json
{
  "prompt_id": "eval_0",              // Required: Prompt identifier
  "judge_score": 0.85,                // Required: Judge evaluation in [0,1]
  "oracle_label": 0.86,               // Optional: Ground truth for AutoCal-R
  "prompt": "What is 2+2?",           // Optional: For reference
  "response": "4"                     // Optional: For reference
}
```

**Notes:**
- Policy name inferred from filename. Do NOT include `"policy"` field in records.
- `prompt_id` is optional - auto-generated from `prompt` text hash if missing

### For IPS/DR Modes (logged data)

**Required JSONL Structure:**
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

### Computing Log Probabilities

CJE includes Fireworks API integration for teacher-forced log probability computation:

```python
from cje.teacher_forcing import compute_teacher_forced_logprob

# For any model available on Fireworks
result = compute_teacher_forced_logprob(
    prompt="What is the capital of France?",
    response="The capital of France is Paris.",
    model="accounts/fireworks/models/llama-v3p2-3b-instruct"
)
if result.status == "success":
    logprob = result.value  # The log probability
```

The teacher forcing module handles:
- Automatic chat template detection and application
- Proper tokenization and log probability extraction
- Fallback mechanisms for robustness
- Support for all major model families (Llama, Qwen, Mistral, etc.)

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

### Dedicated Calibration Sets
```python
# Learn judge→oracle mapping from separate high-quality oracle set
results = analyze_dataset(
    logged_data_path="production_logs.jsonl",      # 10K samples, 100 oracle labels
    calibration_data_path="human_labels.jsonl",    # 500 samples, 500 oracle labels
    combine_oracle_sources=True,                   # Pool all 600 oracle labels (default)
    estimator="calibrated-ips"
)

# Check oracle source breakdown
oracle_meta = results.metadata["oracle_sources"]
print(f"Total oracle labels: {oracle_meta['total_oracle']}")
print(f"From calibration: {oracle_meta['calibration_data']['n_oracle']}")
print(f"From logged: {oracle_meta['logged_data']['n_oracle']}")
```

Priority ordering when combining: `calibration_data > fresh_draws > logged_data`

### Temporal Drift Detection
```python
# Monitor judge stability over time
results = analyze_dataset(
    logged_data_path="logs_q1_q2.jsonl",
    timestamp_field="created_at",  # Unix timestamp or ISO string
    check_drift=True
)

# Check for drift
drift = results.metadata["drift_diagnostics"]
if drift["drift_detection"]["has_drift"]:
    print(f"⚠️ Drift detected at batch transitions: {drift['drift_detection']['drift_points']}")
```

See `cje/interface/README.md` for complete documentation of these features.

### Custom Estimator Config
```python
results = analyze_dataset(
    logged_data_path="data.jsonl",
    estimator="stacked-dr",
    fresh_draws_dir="responses/",
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

### Cluster-Robust Inference

CJE's uncertainty quantification properly accounts for **two independent sources of variance**:

1. **Main sampling uncertainty** (eval log/prompts) → cluster-robust SEs
2. **Calibrator uncertainty** (oracle slice) → OUA jackknife

These components are **additive**: `Var_total = Var_CR + Var_OUA`

#### Enabling Cluster-Robust SEs

```python
# Analyze with cluster-robust inference (e.g., multiple prompts per user)
results = analyze_dataset(
    fresh_draws_dir="responses/",
    cluster_id_field="user_id"  # Enable cluster-robust SEs
)

# Access variance decomposition
var_cr = results.metadata["cluster_robust_variance"]     # Eval-side clustering
var_oua = results.metadata["oua_variance"]               # Oracle uncertainty
var_total = var_cr + var_oua

oua_share = var_oua / var_total  # What fraction is oracle uncertainty?

# Confidence intervals automatically use SE_total = √(Var_CR + Var_OUA)
for policy, est in results.estimates.items():
    se = results.standard_errors[policy]  # SE_total
    ci_lower, ci_upper = est - 1.96*se, est + 1.96*se
    print(f"{policy}: {est:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
```

#### What to Cluster On

Specify `cluster_id_field` when your evaluation data has dependence structure:

- **User/session clustering**: Multiple prompts from same user (`cluster_id_field="user_id"`)
- **Time blocks**: Prompts in temporal batches (`cluster_id_field="batch_id"`)
- **Conversation threads**: Multi-turn dialogues (`cluster_id_field="conversation_id"`)
- **Paired designs**: DM contrasts using same prompts for multiple policies

**Required:** Your data must have the specified field. CJE will group by this field and compute cluster-robust variance.

#### Data Format with Clustering

**For fresh draws:**
```json
{
  "prompt_id": "eval_0",
  "judge_score": 0.85,
  "user_id": "user_123",        // Cluster identifier
  "session_id": "sess_456"      // Alternative cluster identifier
}
```

**For logged data:**
```json
{
  "prompt": "What is 2+2?",
  "response": "4",
  "base_policy_logprob": -14.7,
  "target_policy_logprobs": {"policy_a": -13.2},
  "metadata": {
    "judge_score": 0.85,
    "user_id": "user_123"       // Cluster identifier
  }
}
```

#### When to Use Cluster-Robust SEs

**✅ Always use when:**
- Multiple prompts per user/session (user_id, session_id)
- Temporal batches (batch_id, day, hour)
- Paired DM contrasts (same prompt evaluated by multiple policies)
- Conversation threads (conversation_id, thread_id)

**❌ Not needed when:**
- Each prompt is from a different user (i.i.d. sampling)
- Data is truly independent draws

**Impact:** Ignoring clustering can cause severe undercoverage (e.g., 86.9% instead of 95% observed empirically).

#### Interpreting Results

```python
results = analyze_dataset(
    fresh_draws_dir="responses/",
    cluster_id_field="user_id"
)

# Cluster structure
n_clusters = results.metadata["n_clusters"]        # Number of clusters (G)
cluster_sizes = results.metadata["cluster_sizes"]  # Distribution

# Variance decomposition
se_iid = results.metadata["se_iid"]               # Naive (ignoring clustering)
se_cr = results.metadata["se_cluster_robust"]     # Cluster-robust
inflation = (se_cr / se_iid - 1) * 100            # Percent inflation

print(f"Clusters: {n_clusters} (mean size: {cluster_sizes['mean']:.1f})")
print(f"SE inflation from clustering: {inflation:.1f}%")

# OUA share diagnostic
oua_share = results.metadata["oua_share"]
if oua_share > 0.5:
    print("⚠️ Labels are the bottleneck - add more oracle labels")
else:
    print("✓ Prompts dominate uncertainty - sufficient oracle labels")
```

#### API Parameters

```python
analyze_dataset(
    # Data sources
    logged_data_path: str = None,
    fresh_draws_dir: str = None,

    # Cluster-robust inference
    cluster_id_field: str = None,          # Field name for cluster IDs
                                           # Set to enable cluster-robust SEs

    # Oracle uncertainty (always enabled by default)
    n_oracle_folds: int = 5,               # OUA jackknife folds

    # Other parameters...
    estimator: str = "auto",
    verbose: bool = False
)
```

#### Advanced: Satterthwaite Degrees of Freedom

With few clusters or small oracle slices, CJE automatically uses Satterthwaite effective degrees of freedom instead of normal approximation:

```
df_eff = (Var_CR + Var_OUA)² / (Var_CR²/(G-1) + Var_OUA²/(K-1))
```

Then uses t-distribution critical value for CIs. This is automatic - no configuration needed.

```python
# Check which critical value was used
df_eff = results.metadata.get("df_effective", None)
if df_eff is not None and df_eff < 30:
    print(f"Using t({df_eff:.1f}) critical value (small sample)")
else:
    print("Using normal (z) critical value (large sample)")
```

### Fresh Draws Format

**File structure:** One JSONL file per policy with pattern `{policy}_responses.jsonl`

**Example:**
```
responses/
├── clone_responses.jsonl
├── premium_responses.jsonl
└── unhelpful_responses.jsonl
```

**Record format** (inside each file):
```json
{
  "prompt_id": "p123",           // Required: Must match logged data prompts (for DR)
  "judge_score": 0.83,          // Required: Judge evaluation
  "oracle_label": 0.86,         // Optional: Ground truth for AutoCal-R
  "response": "New answer...",   // Optional: Fresh sample from target
  "draw_idx": 0                  // Optional: Index if multiple draws per prompt
}
```

**CRITICAL:**
- Policy name inferred from filename (e.g., `clone_responses.jsonl` → policy `"clone"`)
- Do NOT include `"policy"` field in records
- For DR mode: Policy names must match exactly with `target_policy_logprobs` keys in logged data

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