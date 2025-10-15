# CJE Data Module

## Overview

The data module handles all data loading, validation, and preparation for CJE analysis. It provides type-safe data models using Pydantic, flexible data loading through factory patterns, and comprehensive validation to ensure data quality before estimation.

## When to Use

### Use **Dataset** when:
- You need a type-safe container for CJE data
- You're passing data between modules
- You want automatic validation

### Use **PrecomputedSampler** when:
- You have data with rewards ready for estimation
- You need importance weight computation
- You're feeding data to estimators

### Use **DatasetFactory** when:
- Loading data from JSONL files
- Converting raw dictionaries to typed Datasets
- You need flexible data loading patterns

### Use **FreshDrawDataset** when:
- You have fresh samples for DR estimation
- You need to organize per-policy fresh draws
- You're using DR/TMLE estimators

## File Structure

```
data/
├── __init__.py           # Public API exports
├── models.py             # Pydantic data models (Sample, Dataset, etc.)
├── loaders.py            # Data loading utilities (DatasetLoader, DataSource)
├── factory.py            # Factory pattern for Dataset creation
├── precomputed_sampler.py # Sampler wrapper for estimators
├── fresh_draws.py        # Fresh draw models for DR
├── folds.py              # Unified fold management for cross-validation
├── validation.py         # Data validation functions
└── reward_utils.py       # Reward manipulation utilities
```

## Core Concepts

### 1. Type-Safe Data Models
All data flows through Pydantic models with automatic validation:
- **Sample**: Single observation with prompt, response, rewards, and log probs
- **Dataset**: Collection of samples with target policies
- **EstimationResult**: Output from estimators with estimates and diagnostics

### 2. Factory Pattern
DatasetFactory provides a clean interface for loading data from various sources while maintaining flexibility through dependency injection.

### 3. Validation Layers
Data is validated at multiple levels:
- Pydantic field validation (types, ranges)
- Structural validation (required fields exist)
- Semantic validation (policies in data match declared targets)

## Common Interface

### Loading Data
```python
from cje.data import DatasetFactory

# From JSONL file
factory = DatasetFactory()
dataset = factory.create_from_jsonl("data.jsonl")

# From raw dictionaries
data = [{"prompt": "...", "response": "...", ...}, ...]
dataset = factory.create_from_data(data)
```

### Using PrecomputedSampler
```python
from cje.data import PrecomputedSampler

# Create sampler (requires rewards)
sampler = PrecomputedSampler(dataset)

# Or directly from JSONL
sampler = PrecomputedSampler.from_jsonl("calibrated_data.jsonl")

# Access data
n_samples = sampler.n_valid_samples
policies = sampler.target_policies

# Check oracle coverage (affects OUA jackknife when < 1.0)
oracle_coverage = sampler.oracle_coverage  # Float in [0, 1]: fraction with oracle labels
```

### Working with EstimationResult
```python
from cje import analyze_dataset

# Get results
result = analyze_dataset(fresh_draws_dir="responses/")

# Access core data
estimates = result.estimates  # numpy array
standard_errors = result.standard_errors  # numpy array
policies = result.metadata["target_policies"]  # list of policy names

# Get confidence intervals
ci_lower, ci_upper = result.confidence_interval(alpha=0.05)
# Or as list of tuples
cis = result.ci()  # [(lower, upper), ...]

# Compare policies
comparison = result.compare_policies(idx1=0, idx2=1)
print(f"Difference: {comparison['difference']:.3f} (p={comparison['p_value']:.3f})")

# Export
result_dict = result.to_dict()  # Full export with CIs, diagnostics
```

**New visualization features:**
```python
# Quick plotting (convenience method)
result.plot_estimates(
    base_policy_stats={"mean": 0.72, "se": 0.01},
    save_path="estimates.png"
)

# Jupyter notebook display
result  # Auto-displays as formatted HTML table
```

See [`visualization/README.md`](../visualization/README.md) for all visualization options.

### Data Validation
```python
from cje.data import validate_cje_data

# Check if data has required fields
is_valid, issues = validate_cje_data(
    data,
    judge_field="judge_score",
    oracle_field="oracle_label"
)
if not is_valid:
    for issue in issues:
        print(f"Issue: {issue}")
```

## Data Format

### Required Fields

Every sample must have:
- `prompt_id`: Unique identifier (optional - auto-generated from prompt hash if missing)
- `prompt`: Input text/context
- `response`: Generated output
- `base_policy_logprob`: Log probability under logging policy
- `target_policy_logprobs`: Dict of log probs for target policies

**prompt_id auto-generation:** If `prompt_id` is missing, CJE automatically generates one:
1. Checks top-level field, then `metadata.prompt_id`
2. If still missing, generates `prompt_{hash}` from SHA256 hash of prompt text
3. If no prompt text either, uses record index: `sample_{idx:06d}` (logged) or `fresh_{policy}_{idx:06d}` (fresh draws)

⚠️ **Warning:** Index-based IDs (fallback #3) are fragile and will trigger a warning. They break with reordering and won't align across datasets for DR mode. Always provide either `prompt_id` or `prompt` for stable, reproducible IDs.

### Optional Fields
- `reward`: Calibrated reward in [0, 1] (required for PrecomputedSampler)
- `metadata`: Dict containing additional fields like:
  - `judge_score`: Raw judge evaluation
  - `oracle_label`: Ground truth label

### Example JSONL Entry
```json
{
  "prompt_id": "arena_001",
  "prompt": "What is machine learning?",
  "response": "Machine learning is a subset of AI...",
  "base_policy_logprob": -14.7,
  "target_policy_logprobs": {
    "clone": -14.7,
    "parallel_universe_prompt": -18.3,
    "unhelpful": -42.1
  },
  "judge_score": 0.85,
  "oracle_label": 0.86
}
```

See `examples/arena_sample/` for a complete working example with logged data and fresh draws.

### Fresh Draws for Direct Mode

Direct Mode (on-policy evaluation) requires **one JSONL file per policy** in `fresh_draws_dir`:

**Directory structure:**
```
responses/
├── model_a_responses.jsonl
├── model_b_responses.jsonl
└── model_c_responses.jsonl
```

**File naming patterns** (searched in order):
1. `{policy}_responses.jsonl` (preferred)
2. `responses/{policy}_responses.jsonl`
3. `{policy}_fresh.jsonl`
4. `fresh_draws/{policy}.jsonl`

**Each file contains records (policy inferred from filename):**
```json
{"prompt_id": "eval_0", "judge_score": 0.85}
{"prompt_id": "eval_1", "judge_score": 0.72, "oracle_label": 0.70}
```

**Fields:**
- `prompt_id`: Identifies the prompt (optional - auto-generated from `prompt` field if missing)
- `judge_score`: Required - judge evaluation in [0, 1]
- `oracle_label`: Optional - ground truth for AutoCal-R calibration
- `draw_idx`: Optional - defaults to 0 (for multiple draws per prompt)
- `prompt`: Optional - used to auto-generate `prompt_id` if not provided
- `response`: Optional - the actual generated text
- `metadata`: Optional - dict for storing per-response covariates (auto-populated)

**Note:** The `target_policy` field is **NOT needed** - it's inferred from the filename!

**Response-level covariates:**

Fresh draws can include response-level covariates for calibration and outcome modeling:

```python
from cje import analyze_dataset

# Auto-compute response_length covariate
result = analyze_dataset(
    fresh_draws_dir="responses/",
    include_response_length=True,  # Computes word count from response field
    estimator="direct"
)

# Or specify manual covariates from metadata
result = analyze_dataset(
    fresh_draws_dir="responses/",
    calibration_covariates=["custom_feature"],  # Must exist in metadata
    estimator="direct"
)
```

The `compute_response_covariates()` utility automatically computes specified covariates and stores them in `FreshDrawSample.metadata`:

```python
from cje.data.fresh_draws import compute_response_covariates, load_fresh_draws_auto

# Load fresh draws
fresh_draws = load_fresh_draws_auto(fresh_draws_dir, policy="model_a")

# Compute covariates (modifies FreshDrawSample.metadata in-place)
fresh_draws = compute_response_covariates(
    fresh_draws,
    covariate_names=["response_length"]
)

# Now fresh_draws[0].metadata["response_length"] contains word count
```

**Important:** `response_length` is response-specific (varies per response), unlike prompt-level covariates which can be tiled across fresh draws. Fresh draw metadata is the authoritative source for response-level features.

### Policy Name Matching (IPS/DR Modes)

When using DR mode (combining logged data + fresh draws), policy names must match exactly:

**Policy discovery process:**
1. **From logged data**: Extracts policy names from `target_policy_logprobs` dict keys
2. **From fresh draws**: Extracts policy names from filenames (`{policy}_responses.jsonl`)
3. **Matching**: For each policy in logged data, loads `{policy}_responses.jsonl`

**Example:**
```python
# Logged data defines policies
{"target_policy_logprobs": {"clone": -14.7, "premium": -18.3}}
# Discovers: ["clone", "premium"]

# CJE then searches for:
# - responses/clone_responses.jsonl
# - responses/premium_responses.jsonl

# If filenames don't match → FileNotFoundError
```

**Common pitfall:** Names with special characters (e.g., `"gpt-4"` vs `"gpt4"`, `"model_v2"` vs `"model-v2"`). Use identical names everywhere!

## Key Design Decisions

### 1. **Pydantic for Type Safety**
We use Pydantic models instead of plain dictionaries to:
- Catch errors early through validation
- Provide clear interfaces with IDE support
- Ensure data consistency across the pipeline

### 2. **Factory Pattern for Flexibility**
DatasetFactory separates data loading concerns from the Dataset model, allowing:
- Easy extension with new data sources
- Testability through dependency injection
- Clean separation of concerns

### 3. **Rewards as Optional**
Rewards are optional in the base Dataset but required for PrecomputedSampler because:
- Data may arrive uncalibrated (needs calibration first)
- Different estimators have different requirements
- Flexibility in pipeline design

### 4. **Metadata as Catch-All**
Non-core fields go into metadata automatically, allowing:
- Preservation of all input data
- Extension without schema changes

### 5. **Oracle Coverage Detection**
PrecomputedSampler.oracle_coverage property enables:
- Automatic OUA jackknife activation when coverage < 100%
- Oracle uncertainty included in standard_errors at partial coverage
- Graceful handling of partial oracle labels
- Backward compatibility

### 6. **Validation at Multiple Levels**
We validate at Pydantic, structural, and semantic levels to:
- Catch issues early before expensive computation
- Provide helpful error messages
- Ensure estimation reliability

## Common Issues and Solutions

### Issue: "PrecomputedSampler requires all samples to have rewards"
**Cause**: Trying to use uncalibrated data with PrecomputedSampler
**Solution**: 
```python
from cje.calibration import calibrate_dataset

# Calibrate first
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    judge_field="judge_score",
    oracle_field="oracle_label"
)
# Then create sampler
sampler = PrecomputedSampler(calibrated_dataset)
```

### Issue: "Log probability must be <= 0"
**Cause**: Invalid log probabilities (positive values)
**Solution**: Ensure log probs are actual log values (negative or zero)

### Issue: Missing target_policy_logprobs
**Cause**: Data doesn't have log probs for declared target policies
**Solution**: Either compute missing log probs or remove policies from target list

### Issue: Inconsistent data types in metadata
**Cause**: Mixed types in metadata fields across samples
**Solution**: Ensure consistent types or handle in preprocessing

## Performance

### Memory Considerations
- Datasets are fully loaded into memory
- For very large datasets (>1GB), consider streaming approaches
- Influence functions in EstimationResult can be large (n_samples × n_policies)
- PrecomputedSampler maintains both original and formatted data

### Optimization Tips
- `PrecomputedSampler.n_valid_samples` shows actual samples after filtering
- Invalid samples are automatically filtered during formatting
- Judge scores are accessed via `get_judge_scores()` for weight calibration

## Fold Management

The `folds` module provides unified cross-validation fold assignment across all CJE components:

### Core Functions
```python
from cje.data.folds import get_fold, get_folds_for_dataset

# Get fold for single prompt
fold = get_fold("prompt_123", n_folds=5, seed=42)  # Returns 0-4

# Get folds for entire dataset
folds = get_folds_for_dataset(dataset, n_folds=5, seed=42)

# Balanced oracle distribution (for calibration)
from cje.data.folds import get_folds_with_oracle_balance
oracle_mask = np.array([s.oracle_label is not None for s in dataset.samples])
balanced_folds = get_folds_with_oracle_balance(prompt_ids, oracle_mask, n_folds=5)
```

### Key Properties
- **Deterministic**: `hash(prompt_id) % n_folds` ensures reproducibility
- **Filtering-proof**: Based on stable prompt_id, not array indices
- **Fresh-draw compatible**: Same prompt_id → same fold always
- **Cross-component consistent**: All estimators use same fold system

**Note**: Folds are computed on-demand using `hash(prompt_id) % n_folds`. The fold configuration (n_folds, fold_seed) is stored in dataset metadata for reproducibility.

## Advanced Topics

### Custom Data Sources
Implement the DataSource protocol:
```python
from typing import List, Dict, Any

class CustomDataSource:
    def load(self) -> List[Dict[str, Any]]:
        # Your loading logic
        return data
        
# Use with factory
factory = DatasetFactory()
source = CustomDataSource()
dataset = factory.loader.load_from_source(source, target_policies=["clone", "parallel_universe_prompt"])
```

### Fresh Draws for DR
```python
from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample

# Create fresh draws
samples = [
    FreshDrawSample(
        prompt_id="arena_1",
        target_policy="clone",
        judge_score=0.85,
        oracle_label=0.86,
        draw_idx=0
    ),
    # ... more samples
]

fresh_dataset = FreshDrawDataset(
    target_policy="clone",
    draws_per_prompt=5,
    samples=samples
)
```

### Custom Validation
```python
def validate_custom_requirements(data: List[Dict]) -> Tuple[bool, List[str]]:
    issues = []
    
    # Your validation logic
    for record in data:
        if "custom_field" not in record:
            issues.append("Missing custom_field")
    
    return len(issues) == 0, issues
```

## Summary

The data module provides a robust foundation for CJE analysis through type-safe models, flexible loading patterns, and comprehensive validation. It ensures data quality early in the pipeline while maintaining flexibility for different use cases and data sources.