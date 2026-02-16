# CJE Interface

Simple, reliable LLM evaluation with automatic mode selection and AutoCal-R calibration.

For an end-to-end operational workflow (audits, drift response, label budgeting), see [`PLAYBOOK.md`](../../PLAYBOOK.md).

## Quick Start

CJE’s default workflow is Direct mode on fresh draws:

```python
from cje import analyze_dataset

# Default: Direct mode (on-policy comparison on your eval prompts)
results = analyze_dataset(fresh_draws_dir="responses/")

# Print results
print(f"Policy value: {results.estimates[0]:.3f} ± {1.96*results.standard_errors[0]:.3f}")
```

## Which Mode Should I Use?

**Use Direct mode.** It's simple, reliable, and works for the vast majority of LLM evaluation tasks.

*Footnote (advanced): IPS/DR variants are supported for explicit counterfactual OPE workflows. They require strong overlap and additional diagnostics. See `cje/estimators/README.md` for internals.*

### What are fresh draws?
Fresh draws are new responses from your target policies evaluated by the judge.

**Format:** One JSONL file per policy in `fresh_draws_dir`. Policy name is inferred from filename.

**Example:** `responses/clone_responses.jsonl` → policy name is `"clone"`

## Common Workflows

### Basic Analysis (Direct Mode)
```python
from cje import analyze_dataset

# Simplest workflow - just fresh draws from files
results = analyze_dataset(fresh_draws_dir="responses/")

# Alternative: In-memory data (no file I/O needed)
results = analyze_dataset(
    fresh_draws_data={
        "policy_a": [
            {"prompt_id": "q1", "judge_score": 0.85, "oracle_label": 0.9},
            {"prompt_id": "q2", "judge_score": 0.72},  # oracle_label optional
        ],
        "policy_b": [
            {"prompt_id": "q1", "judge_score": 0.70, "oracle_label": 0.75},
            {"prompt_id": "q2", "judge_score": 0.82},
        ],
    }
)

# Get estimates for each policy
for i, policy in enumerate(results.metadata["target_policies"]):
    print(f"{policy}: {results.estimates[i]:.3f} ± {1.96*results.standard_errors[i]:.3f}")

# Note: Direct mode auto-discovers policies from filenames
print(f"Found policies: {results.metadata['target_policies']}")
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
# Direct-style run with fresh draws
python -m cje analyze logs.jsonl --fresh-draws-dir responses/ --estimator direct

# Save results
python -m cje analyze logs.jsonl --fresh-draws-dir responses/ --estimator direct -o results.json

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

*Footnote (advanced): IPS/DR logged-data schema and counterfactual diagnostics are documented in `cje/estimators/README.md`.*

**Working example:** See [`examples/arena_sample/`](../../examples/arena_sample/) for complete dataset examples.

## Troubleshooting

*Footnote (advanced): IPS/DR-specific troubleshooting (ESS, overlap, logprob coverage) lives in `cje/estimators/README.md`.*

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

## API Reference

### `analyze_dataset()`

```python
def analyze_dataset(
    logged_data_path: Optional[str] = None,
    fresh_draws_dir: Optional[str] = None,
    fresh_draws_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
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
- `fresh_draws_data`: In-memory alternative to `fresh_draws_dir`. Dict mapping policy names to lists of records. Each record needs: `prompt_id`, `judge_score`. Optional: `oracle_label`, `response`. Example: `{"policy_a": [{"prompt_id": "1", "judge_score": 0.8}, ...], ...}`
- `calibration_data_path`: Path to dedicated calibration dataset with oracle labels. Used to learn judge→oracle mapping separately from evaluation data.
- `combine_oracle_sources`: Pool oracle labels from all sources (calibration + logged + fresh) for maximum data efficiency. Default: `True`. Set `False` to use only calibration_data_path.
- `estimator`: Estimator name or "auto" for automatic selection
  - Recommended: `direct` (or `auto` with fresh draws)
  - *Footnote (advanced): `calibrated-ips`, `stacked-dr`, `dr-cpo`, `tmle`, `mrdr`, etc. for counterfactual OPE.*
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
  - `.calibrator`: Fitted calibrator for transportability audits
  - `.plan_allocation(budget, cost_model, fresh_draws)`: Plan optimal oracle/surrogate allocation for production (requires pilot fresh draws)
  - `.metadata`: Mode, estimator, data sources (see additional fields below)

**Additional metadata fields** (when using calibration_data_path):
- `metadata["oracle_sources"]`: Breakdown of oracle labels by source (calibration_data, logged_data, fresh_draws)
- `metadata["oracle_sources"]["distribution_mismatch"]`: KS test results comparing calibration vs. evaluation distributions

**At least one of `logged_data_path`, `fresh_draws_dir`, or `fresh_draws_data` must be provided.**

### CLI Commands

#### `analyze` - Run analysis
```bash
python -m cje analyze <dataset> [options]

Options:
  --estimator {stacked-dr,calibrated-ips,calibrated-direct,direct,raw-ips,dr-cpo,mrdr,tmle}
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
    fresh_draws_dir="responses/",                  # Evaluation set
    calibration_data_path="human_labels.jsonl",     # 500 samples, all with high-quality oracle labels
    estimator="direct"
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
    fresh_draws_dir="responses/",
    calibration_data_path="oracle_labels.jsonl",
    combine_oracle_sources=False,  # Use ONLY calibration data for learning f̂
    estimator="direct"
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
    fresh_draws_dir="responses/",
    include_response_length=True,  # Auto-compute response length
    estimator="direct"
)

# Example 2: Add custom metadata covariates with response_length
# Assumes your data has "domain" and "difficulty" in metadata
results = analyze_dataset(
    fresh_draws_dir="responses/",
    include_response_length=True,
    calibration_covariates=["domain", "difficulty"],  # Additional covariates
    estimator="direct",
)
# Effective covariates: ["response_length", "domain", "difficulty"]

# Example 3: Judge score only (default - no covariates)
results = analyze_dataset(
    fresh_draws_dir="responses/",
    estimator="direct"
)

# Example 4: Custom covariates without response_length
results = analyze_dataset(
    fresh_draws_dir="responses/",
    calibration_covariates=["domain"],  # Only domain, no response_length
    estimator="direct"
)
```

**When to use covariates:**
- **Length bias**: Judge scores vary by response length at fixed oracle quality
- **Domain effects**: Judge miscalibration differs across domains (e.g., math vs. creative writing)
- **Task heterogeneity**: Observable features predict judge-oracle disagreement

**How it works:**
1. Two-stage calibration learns g(S, X_cov) → rank → isotonic
2. Covariates help handle non-monotone patterns in judge scores
3. Direct mode uses covariates in the calibration mapping
4. *Footnote (advanced): IPS/DR variants also support covariate-adjusted calibration*

**Requirements:**
- Covariate fields must exist in `sample.metadata` for all samples
- When using `include_response_length=True`, all samples must have a `"response"` field
- Covariates work with two-stage or auto calibration mode (not monotone-only)

**See also:** `cje/calibration/README.md` for details on two-stage calibration with covariates.

### Direct Mode Inference Methods

Direct mode supports multiple inference methods for computing standard errors:

```python
# Recommended default: bootstrap + per-policy augmented correction (θ̂_aug)
results = analyze_dataset(
    fresh_draws_dir="responses/"
)

# Explicit bootstrap config (equivalent defaults)
results = analyze_dataset(
    fresh_draws_dir="responses/",
    estimator_config={
        "inference_method": "bootstrap",
        "use_augmented_estimator": True,
        "use_multipolicy_eif": False,  # Conservative default
        "n_bootstrap": 2000
    }
)

# Cluster-robust only (fastest, for large samples)
results = analyze_dataset(
    fresh_draws_dir="responses/",
    estimator_config={"inference_method": "cluster_robust"}
)
```

**Inference methods:**
| Method | Coverage | Description |
|--------|----------|-------------|
| `bootstrap` (default) | **~95%** | Cluster bootstrap with θ̂_aug + calibrator refit |
| `cluster_robust` | ~22-55% | Standard cluster-robust SEs (fast, ignores calibration uncertainty) |
| `auto` | varies | Uses cluster_robust; switches to bootstrap when coupling detected |

**Separate flag:** `oua_jackknife` is not an `inference_method` value.
Set `estimator_config={"oua_jackknife": True}` to add oracle jackknife augmentation.

**Bootstrap with θ̂_aug** is recommended for valid confidence intervals. It uses a per-policy AIPW-style bias correction (`θ̂_aug = plug-in + residual correction`) and refits the calibrator on each replicate to capture calibration/evaluation covariance.
Treat a single global offset as a diagnostic baseline, not as a default production correction.

**When to use bootstrap (recommended for all cases):**
- Always, if you need valid confidence intervals
- Few evaluation prompts (< 20)
- Calibration and evaluation data overlap

**Transport-aware bootstrap** (`calibration_policy` option):
When evaluating multiple policies where calibration was learned on one base policy, use `calibration_policy` to enable transport-aware bias correction:

```python
results = analyze_dataset(
    fresh_draws_dir="responses/",
    estimator_config={
        "inference_method": "bootstrap",
        "calibration_policy": "base",  # Fit calibrator only on base policy
    }
)
```

This separates calibration (base policy only) from residual corrections (all policies). When the calibrator doesn't transport to target policies, the residual term in θ̂_aug captures the bias. See `diagnostics/README.md` for details.

### Custom Configuration
```python
results = analyze_dataset(
    fresh_draws_dir="responses/",
    estimator="direct",
    estimator_config={
        "inference_method": "bootstrap",
        "n_bootstrap": 2000,
    },
)
```

### Hydra Support
For complex configurations, use Hydra:
```bash
python -m cje.interface.hydra_entry \
  dataset=logs.jsonl \
  estimator=direct \
  fresh_draws_dir=responses/ \
  estimator_config.inference_method=bootstrap
```

## Summary

**Default mode: Direct.**

1. Use `fresh_draws_dir` (or `fresh_draws_data`) as your primary input.
2. Use bootstrap inference for reliable confidence intervals.
3. Run transport audits on small probe slices over time.

*Footnote (advanced): IPS/DR modes remain available for counterfactual OPE with strong overlap assumptions and diagnostics.*

For more details, see the [examples](../../examples/) and full documentation.
