# CJE Project Documentation - All README Files
Generated on: Sat Sep 27 15:04:01 PDT 2025
=


# ============================================================================
# FILE: README.md
# ============================================================================

<div align="left">
  <img src="CJE_logo.jpg" alt="CJE Logo" width="250">
</div>

# CJE - Causal Judge Evaluation

[![Docs](https://img.shields.io/badge/docs-cimo--labs.com-blue)](https://cimo-labs.com/cje)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue)](https://www.python.org/downloads/)
[![codecov](https://codecov.io/gh/cimo-labs/cje/branch/main/graph/badge.svg)](https://codecov.io/gh/cimo-labs/cje)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Off-policy evaluation for LLMs that actually works.** Get unbiased estimates of how your new model will perform before deployment.

## Why CJE?

🎯 **Problem**: LLM-as-judge scores are biased - they tell you about your current model, not your next one
✅ **Solution**: CJE uses causal inference to debias these scores for reliable policy evaluation

## Installation

```bash
pip install cje-eval
```

For development:
```bash
git clone https://github.com/fondutech/causal-judge-evaluation.git
cd causal-judge-evaluation
poetry install  # or pip install -e .
```

## Quick Start

```python
from cje import analyze_dataset

# Get unbiased estimate with confidence intervals
result = analyze_dataset("your_data.jsonl", estimator="calibrated-ips")
print(f"Policy value: {result.estimates[0]:.3f} ± {result.standard_errors[0]:.3f}")
```

For production use with fresh samples (most accurate):
```python
# With fresh draws from target policy (best accuracy)
result = analyze_dataset("logs.jsonl", estimator="stacked-dr",
                        fresh_draws_dir="responses/")
```

CLI usage:
```bash
# Quick evaluation
python -m cje analyze data.jsonl --estimator calibrated-ips -o results.json

# Production evaluation with fresh samples
python -m cje analyze data.jsonl --estimator stacked-dr --fresh-draws-dir responses/
```

## How It Works

CJE transforms biased judge scores into unbiased policy estimates:

```
Your Data → Judge Calibration → Importance Weighting → Unbiased Estimate + CI
(logs.jsonl)  (maps judge→truth)   (reweights samples)    (with diagnostics)
```

## When to Use CJE

✅ **Perfect for:**
- A/B testing LLMs before deployment
- Evaluating multiple model variants
- Reusing existing data for new evaluations
- High-stakes decisions needing confidence intervals

❌ **Not for:**
- Online learning (CJE is offline)
- Real-time scoring (CJE is batch)
- Small samples (<1000 examples)

## Data Requirements

CJE expects JSONL with these fields:

```json
{
  "prompt": "What is 2+2?",
  "response": "4",
  "base_policy_logprob": -2.3,               // Log P(response|prompt) for current model
  "target_policy_logprobs": {"gpt4": -1.8},  // Same for model(s) to evaluate
  "metadata": {
    "judge_score": 0.9,                      // Your LLM judge's score
    "oracle_label": 1.0                      // Ground truth (5-10% labeled is enough)
  }
}
```

## Choosing an Estimator

- **`calibrated-ips`** (default for quick start): Fast, reliable, no fresh samples needed
- **`stacked-dr`** (recommended for production): Most accurate, requires fresh samples from target
- **Individual estimators** (`dr-cpo`, `tmle`, `mrdr`): For research and debugging

See the documentation for estimator details.

## Documentation

📚 **Getting Started**
- [5-Minute Quickstart](QUICKSTART.md) - First analysis step-by-step
- [Examples](examples/) - Working code samples
- Full documentation coming soon on cimo-labs.com

🔧 **For Engineers**
- [Engineering Guide](README_ENGINEERING.md) - Interface specs and patterns
- [Arena Experiment](cje/experiments/arena_10k_simplified/) - Production pipeline example
- **Module READMEs** - Each subdirectory in `cje/` contains a developer-oriented README:
  - `cje/estimators/README.md` - Estimator implementations and hierarchy
  - `cje/diagnostics/README.md` - Diagnostic system architecture
  - `cje/data/README.md` - Data models and validation
  - `cje/calibration/README.md` - Calibration methods
  - `cje/interface/README.md` - High-level API details

📊 **Additional Resources**
- API Reference - Coming soon
- Mathematical Foundations - Coming soon
- Troubleshooting Guide - Coming soon

## Development

```bash
make install  # Install with Poetry
make test     # Run tests
make format   # Auto-format code
make lint     # Check code quality
```

## Support

- 🐛 [Issues](https://github.com/fondutech/causal-judge-evaluation/issues)
- 💬 [Discussions](https://github.com/fondutech/causal-judge-evaluation/discussions)

## License

MIT - See [LICENSE](LICENSE) for details.

---
**Ready to start?** → [5-Minute Quickstart](QUICKSTART.md)


# ============================================================================
# FILE: QUICKSTART.md
# ============================================================================

# CJE Quickstart - 5 Minutes to First Results

This guide gets you from zero to your first unbiased LLM evaluation in 5 minutes.

## Prerequisites

✅ Python 3.9+ installed
✅ 1000+ examples with judge scores
✅ Log probabilities from your models

## Step 1: Install (30 seconds)

```bash
pip install cje-eval
```

## Step 2: Understand Your Data (1 minute)

CJE needs data in JSONL format with these fields:

```json
{
  "prompt": "What is machine learning?",
  "response": "Machine learning is...",
  "base_policy_logprob": -45.2,      // Log P(response|prompt) for current model
  "target_policy_logprobs": {         // Same for models you want to evaluate
    "gpt4": -42.1,
    "claude": -44.3
  },
  "metadata": {
    "judge_score": 0.82,              // Your LLM judge's score
    "oracle_label": 0.90              // Ground truth (only need 5-10% labeled)
  }
}
```

Don't have this format? See the data preparation section below.

## Step 3: Run Your First Analysis (30 seconds)

```python
from cje import analyze_dataset

# Simplest possible usage
result = analyze_dataset("your_data.jsonl", estimator="calibrated-ips")

# View results
print(f"GPT-4 estimated value: {result.estimates[0]:.3f}")
print(f"95% CI: [{result.robust_confidence_intervals[0][0]:.3f}, "
      f"{result.robust_confidence_intervals[0][1]:.3f}]")
```

## Step 4: Understand the Output (2 minutes)

You'll see something like:
```
GPT-4 estimated value: 0.731
95% CI: [0.712, 0.750]

Diagnostics Summary:
- Effective Sample Size: 43.2% (GOOD)
- Weight concentration: Low
- Calibration R²: 0.83
```

What this means:
- **Estimated value**: GPT-4 would score 73.1% if deployed
- **Confidence interval**: We're 95% sure it's between 71.2% and 75.0%
- **ESS 43.2%**: Good overlap between policies (>30% is good)

## Step 5: Check Reliability (1 minute)

```python
# Always check diagnostics before trusting results
diagnostics = result.diagnostics

if diagnostics.overall_status.value == "CRITICAL":
    print("⚠️ Results may be unreliable!")
    print(diagnostics.summary())
else:
    print("✅ Results are reliable")

# For detailed analysis
print(f"ESS: {diagnostics.weight_ess:.1%}")
for policy, ess in diagnostics.ess_per_policy.items():
    print(f"  {policy}: {ess:.1%}")
```

## Common Issues & Solutions

### "Low ESS warning"
Your policies are very different. Solutions:
1. Use doubly-robust estimation (requires fresh samples)
2. Collect more diverse training data
3. Use a less different target policy

### "No oracle labels found"
Add ground truth to 5-10% of samples:
```python
# Before running CJE, add some labels
import random
labeled_sample_ids = random.sample(all_ids, k=int(0.1 * len(all_ids)))
# Add oracle_label to metadata for these samples
```

### "ValueError: DR estimators require fresh draws"
Generate new samples from target policy:
```bash
python -m cje.teacher_forcing generate --model gpt4 --n-samples 1000
```

## Next Steps - Choose Your Path

### 🎯 "I just want better estimates"
→ Choose the Right Estimator
```python
# If you have fresh samples from GPT-4:
result = analyze_dataset("data.jsonl", estimator="stacked-dr",
                         fresh_draws_dir="gpt4_responses/")
```

### 📊 "I need to compare multiple models"
→ Compare Multiple Policies
```python
results = analyze_dataset("data.jsonl", estimator="calibrated-ips")
best_policy = results.diagnostics.best_policy
print(f"Best policy: {best_policy}")
```

### 🔬 "I want to understand the theory"
→ Mathematical Foundations (documentation coming soon)
- Why importance sampling works
- SIMCal calibration details
- Doubly robust estimation

### 🏭 "I'm deploying to production"
→ Production Configuration
```python
# Production configuration
result = analyze_dataset(
    "data.jsonl",
    estimator="calibrated-ips",
    refuse_unreliable=True,  # Return NaN if unreliable
    verbose=True             # Log progress
)
```

## Complete Example

Here's everything together:

```python
from cje import analyze_dataset

# Load and analyze
result = analyze_dataset(
    "llm_judge_data.jsonl",
    estimator="calibrated-ips",
    judge_field="judge_score",
    oracle_field="human_label"
)

# Check reliability
if result.diagnostics.weight_ess < 0.1:
    print("Warning: Low ESS, consider using DR methods")

# Report results
for i, policy in enumerate(result.metadata["target_policies"]):
    estimate = result.estimates[i]
    se = result.robust_standard_errors[i]
    ci_low, ci_high = result.robust_confidence_intervals[i]

    print(f"\n{policy}:")
    print(f"  Estimate: {estimate:.3f} ± {se:.3f}")
    print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  ESS: {result.diagnostics.ess_per_policy[policy]:.1%}")
```

## FAQ

**Q: How much ground truth do I need?**
A: 5-10% of samples with oracle labels is usually sufficient.

**Q: Can I use this without log probabilities?**
A: No, log probs are essential for importance weighting.

**Q: How do I get log probabilities?**
A: Most APIs provide them: OpenAI (`logprobs=True`), Anthropic (`log_probs=true`), etc.

**Q: What if I don't have fresh samples?**
A: Use `calibrated-ips` - it's robust without fresh samples. DR methods need them.

---

🎉 **Congrats!** You've run your first unbiased LLM evaluation.

**Ready for more?**
- Understanding diagnostics - See diagnostics.summary()
- Generating fresh samples - Use teacher_forcing module
- [Engineering Guide](README_ENGINEERING.md) - Interface specs and patterns
- Research paper - Coming soon

**For Developers:**
Each module in `cje/` has a developer-oriented README with implementation details:
- `cje/estimators/README.md` - Estimator implementations
- `cje/diagnostics/README.md` - Diagnostic architecture
- `cje/data/README.md` - Data models and validation
- `cje/calibration/README.md` - Calibration methods

# ============================================================================
# FILE: README_ENGINEERING.md
# ============================================================================

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

# ============================================================================
# FILE: cje/calibration/README.md
# ============================================================================

# CJE Calibration Module

## Overview

The calibration module implements the core mathematical machinery that enables unbiased causal inference from judge-based evaluations. It provides three distinct calibration approaches that work together to transform raw logged data into reliable policy value estimates with controlled variance.

## When to Use Each Calibration

### Use **Reward Calibration** when:
- You have judge scores and some oracle labels
- You want to map judge scores → oracle scale
- You're using any estimation method

### Use **Weight Calibration** (SIMCal) when:
- Importance weights have high variance
- You want to stabilize IPS estimates
- You're using CalibratedIPS estimator

### Use **Cross-Fitted Models** when:
- You're using DR estimators
- You need orthogonality guarantees
- You have enough data for stable folds

## File Structure

```
calibration/
├── __init__.py          # Public API exports
├── dataset.py           # High-level dataset calibration workflows
├── flexible_calibrator.py # Flexible calibration for non-monotone relationships
├── isotonic.py          # Core isotonic regression and variance control
├── judge.py             # Judge score calibration to oracle labels
├── oracle_slice.py      # Oracle slice configuration (deprecated)
├── simcal.py            # Stacked SIMCal implementation
└── iic.py               # Isotonic Influence Control for variance reduction
```

## Core Concepts

### 1. Judge Score Calibration
Maps cheap LLM judge scores to expensive oracle labels. Default is 'auto' mode which automatically selects between:
- **Monotone calibration**: Standard isotonic regression (when relationship is monotone)
- **Flexible calibration**: Two-stage g(S)→isotonic for non-monotone relationships

Auto mode detects non-monotonicity by comparing regional performance and selects the appropriate method. The selected mode is stored in metadata for transparency.

### 2. Weight Calibration (SIMCal)
Stabilizes importance weights through score-indexed monotone projection:
- Projects weights to be monotone with an ordering index
- Enforces variance constraints via blending
- Maintains mean-1 property for unbiasedness

### 3. Cross-Fitted Models
For doubly robust methods, provides out-of-fold predictions to maintain orthogonality between nuisance functions.
Stacking relies on the component estimators' influence functions and does not re-fit nuisances at the stack level.

### 4. Oracle Uncertainty Quantification (Two Approaches)
When we calibrate judge scores using only a subset of oracle labels (e.g., 10% coverage), the calibration function f̂ itself has uncertainty. We handle this through two complementary mechanisms:

**Oracle Uncertainty Augmentation (OUA)**: The default approach that uses fold-jackknife to add a **variance** component to CIs, accounting for calibration-induced uncertainty. Used by all Cal-IPS/DR estimators.

**Oracle Slice Augmentation**: An optional point-estimate **bias correction** term `(L/π_L)m̂(S)(Y-f̂(S))` used **only** in TR-CPO under MAR with fitted π_L(S), or optionally as an MCAR engineering fallback (off by default).

### 5. Isotonic Influence Control (IIC)
A variance reduction technique that residualizes influence functions against judge scores. By fitting E[φ|S] using spline or isotonic regression and computing residuals φ̃ = φ - Ê[φ|S], IIC reduces variance without changing the estimand. This is "free" variance reduction that can be enabled in estimators that support it (CalibratedIPS, OrthogonalizedIPS, DR-CPO, and all DR variants). It is disabled by default to preserve standard methodology.

## Module Descriptions

### `dataset.py` - Dataset Calibration Workflows
High-level functions that orchestrate the calibration process for entire datasets:
- `calibrate_dataset()`: Transforms Dataset objects with judge scores into calibrated rewards
- `calibrate_from_raw_data()`: Works with raw dictionaries for pipeline integration
- Handles both standard and cross-fitted calibration
- Preserves metadata and adds calibration diagnostics

### `judge.py` - Judge Calibration
Implements calibration from judge scores to oracle labels with auto mode selection:
- `JudgeCalibrator`: Main calibration class with flexible mode support
- `fit_transform()`: Standard calibration on oracle subset
- `fit_cv()`: Cross-fitted calibration for DR methods
- `index()`: Returns transformation for outcome models (S for monotone, g(S) for two-stage)
- `CalibrationResult`: Container for calibrated scores and diagnostics
- Auto mode (default): Automatically selects monotone or flexible calibration
- Supports partial labeling (oracle coverage)

### `flexible_calibrator.py` - Non-Monotone Calibration
Handles non-monotone judge→oracle relationships via two-stage approach:
- `FlexibleCalibrator`: Implements g(S)→isotonic calibration
- First stage: Learn smooth transformation g(S) using splines
- Second stage: Apply isotonic regression on g(S)
- `index()`: Exposes the transformation T=g(S) for outcome models
- Per-fold ECDF for consistent rank transformation
- Auto selection based on regional performance comparison

**Mode Selection Logic:**
- Compares monotone vs two-stage using 1-SE rule
- Checks performance across score regions (low/mid/high)
- Selects two-stage if better in ≥2/3 regions or significantly better overall
- Optimized to skip two-stage training when monotone is clearly sufficient

**Technical Details:**
- ECDF-based ranking prevents distribution leakage between folds
- Minimum 5 spline knots to avoid underfitting
- Fallback to monotone for small samples (<20)
- Clipping to [0,1] ensures valid reward range

### `isotonic.py` - Isotonic Weight Calibration
Core mathematical operations for weight calibration:
- `calibrate_to_target_mean()`: Main entry point for weight calibration
- `_pav_mean1_projection_sorted()`: Pool Adjacent Violators with mean preservation
- `_variance_safe_blend_closed_form()`: Optimal blending for variance control
- Uses "exact" mode (bisection) for consistency
- Handles ordering by arbitrary index (e.g., judge scores)

### `simcal.py` - Stacked SIMCal
Advanced weight calibration through stacking:
- `SIMCalibrator`: Combines {baseline, increasing, decreasing} candidates
- Out-of-fold (OOF) influence function minimization
- Quadratic program on simplex for optimal mixture
- Uniform blending for ESS/variance constraints
- Configurable via `SimcalConfig` dataclass
- **New**: Supports fit/predict separation for honest inference
  - `fit()`: Learn isotonic models and mixture weights on training data
  - `predict()`: Apply learned calibration to new data with score clipping
  - `fit_transform()`: Backward-compatible single-pass method

### `oracle_slice.py` - Oracle Slice Augmentation
Implements the optional point-estimate bias correction (used primarily in TR-CPO):
- **Problem**: We use f̂(S) everywhere but only have true Y on oracle subset  
- **Solution**: Add augmentation term `(L/π_L) * m̂(S) * (Y - f̂(S))` where:
  - L indicates oracle label presence, π_L = labeling propensity
  - m̂(S) = E[W|S] estimated via isotonic regression
  - Unbiased correction for proxy-truth gap under MAR/MCAR
- **Usage**: Enabled in TR-CPO for MAR setting; optional MCAR fallback (off by default)
- **Note**: This is separate from OUA jackknife variance (the default uncertainty method)

### `iic.py` - Isotonic Influence Control
Advanced variance reduction through influence function residualization:

**Core Mechanism:**
- `IsotonicInfluenceControl`: Main class that residualizes influence functions against judge scores
- Fits E[φ|S] using either spline regression (default) or isotonic regression
- Returns residuals φ̃ = φ - Ê[φ|S] with guaranteed variance reduction
- **Critical**: Centers fitted values to preserve mean exactly (E[φ̃] = E[φ] = 0)

**Implementation Features:**
- **Flexible regression modes**:
  - Spline regression (default): Cubic splines with configurable knots for smooth fits
  - Isotonic regression: Monotone fit with automatic direction selection via Spearman correlation
- **Cross-fitting support**: Uses same folds as reward calibration for consistency
- **Automatic fallback**: Handles edge cases (insufficient data, non-finite values) gracefully
- **Comprehensive diagnostics**: R², variance reduction ratio, ESS improvement, regression type used

**Configuration via `IICConfig`:**
- `use_splines`: Enable spline regression (default=True, more flexible than isotonic)
- `n_knots`: Number of spline knots (default=8)
- `spline_degree`: Degree of spline polynomials (default=3 for cubic)
- `use_cross_fit`: Apply fold-honest fitting (default=True)
- `min_samples_for_iic`: Minimum samples required (default=50)

**Key Properties:**
- **Variance-only**: Point estimates remain unchanged, only standard errors are reduced
- **Guaranteed improvement**: Var(φ̃) ≤ Var(φ) by construction
- **Typical reductions**: 5-20% SE reduction depending on R²(φ|S)
- **Free lunch**: No additional data or assumptions required
- **Disabled by default**: IIC must be explicitly enabled with use_iic=True

**Why it works**: Influence functions often correlate with judge scores because both relate to outcome quality. By removing the predictable component E[φ|S], we eliminate systematic variation while preserving the estimand.

## Key Design Decisions

### 1. **Separation of Concerns**
Each calibration type is isolated with clear interfaces:
- Reward calibration doesn't know about weights
- Weight calibration doesn't know about rewards
- Outcome models are separate from both

### 2. **Mean Preservation**
Calibrations preserve means for unbiased estimation:
- Isotonic preserves the **slice sample mean** exactly, and the **population mean asymptotically** under J₁ (representative slice)
- Weight projections preserve the **sample** mean-one exactly (Hájek normalization)
- Critical for unbiased estimation

### 3. **Variance Control**
Multiple mechanisms for variance reduction:
- **Isotonic projection**: Can reduce variance when weights correlate with ordering index
- **Variance cap**: Explicit upper bound on weight variance via blending
- **ESS floor**: Minimum effective sample size constraint
- **Baseline shrinkage**: Small bias for large variance reduction

### 4. **Cross-Fitting Support**
Built-in support for cross-fitted calibration:
- Prevents overfitting in DR methods
- Maintains orthogonality between nuisance functions
- Uses unified fold system from `cje.data.folds` for consistency
- Fold assignments computed deterministically from prompt_id

### 5. **Numerical Robustness**
Careful handling of edge cases:
- Zero weights: Fallback to uniform
- Constant weights: Return target mean
- Sparse weights: Relaxed tolerance
- Numerical precision: Multiple safety checks


## Mathematical Foundations

### Isotonic Regression (PAV Algorithm)
Finds the best-fitting monotone function: `min ||f(x) - y||²` subject to monotonicity.
- **Time**: O(n log n) 
- **Property**: When ordered by uncorrelated index, produces nearly constant weights

### Mean-Preserving Projection  
Ensures calibrated weights have exactly mean=1 via bisection on Lagrange multipliers.
- **Why**: Critical for unbiased estimation (E[W] = 1)
- **Implementation**: ~30-40 PAV calls for exact solution

### Variance-Safe Blending
Optimally blends raw and calibrated weights to satisfy variance constraints:
```
w_final = (1-α)·raw + α·calibrated
where Var(w_final) ≤ ρ·Var(raw)
```
- **Solution**: Closed-form via quadratic formula

### Stacked SIMCal
Combines K=3 candidates by minimizing OOF influence variance:
```
min_π π'Σπ s.t. π ≥ 0, Σπ = 1
```
- **Candidates**: {baseline, increasing, decreasing}
- **Solution**: Quadratic program on simplex

## Usage Patterns

### Basic Reward Calibration
```python
from cje.calibration import calibrate_dataset

# Calibrate judge scores to oracle labels (auto mode by default)
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    judge_field="judge_score",
    oracle_field="oracle_label",
    calibration_mode="auto",  # Or "monotone", "two_stage"
    random_seed=42  # For reproducibility
)

# Access calibration quality metrics and metadata
print(f"RMSE: {cal_result.calibration_rmse:.3f}")
print(f"Coverage: {cal_result.coverage_at_01:.1%}")
print(f"Selected mode: {calibrated_dataset.metadata.get('calibration_info', {}).get('selected_mode')}")
```

### Weight Calibration (Direct)
```python
from cje.calibration import calibrate_to_target_mean

# Calibrate weights with variance control
calibrated_weights, info = calibrate_to_target_mean(
    raw_weights,
    target_mean=1.0,
    enforce_variance_nonincrease=True,
    ordering_index=judge_scores,  # Order by judge scores
    return_diagnostics=True
)

print(f"Variance reduction: {info['var_before']/info['var_after']:.2f}x")
```

### Stacked SIMCal
```python
from cje.calibration import SIMCalibrator, SimcalConfig

# Configure stacked calibration
config = SimcalConfig(
    ess_floor=0.2,      # Minimum 20% ESS
    var_cap=1.0,        # No variance increase
    include_baseline=False,
)

# Run calibration
calibrator = SIMCalibrator(config)
calibrated, info = calibrator.transform(
    weights, 
    judge_scores,
    rewards=rewards  # For IPS influence functions
)

print(f"Mixture: {info['mixture_weights']}")
print(f"ESS improvement: {info['ess_after']/info['ess_before']:.2f}x")
```

### Cross-Fitted Calibration (for DR)
```python
from cje.calibration import JudgeCalibrator

# Fit with cross-validation for DR methods
calibrator = JudgeCalibrator()
result = calibrator.fit_cv(
    judge_scores,
    oracle_labels,
    oracle_mask,
    n_folds=5
)

# Get out-of-fold predictions
oof_predictions = calibrator.predict_oof(judge_scores, fold_ids)
```

### Isotonic Influence Control (IIC)
```python
from cje.calibration import IsotonicInfluenceControl, IICConfig

# Configure IIC with spline regression
config = IICConfig(
    use_splines=True,      # Use flexible splines instead of isotonic
    n_knots=8,            # Number of spline knots
    spline_degree=3,      # Cubic splines
    use_cross_fit=True,   # Fold-honest fitting
    compute_diagnostics=True
)

# Apply IIC to reduce influence function variance
iic = IsotonicInfluenceControl(config)
residualized_if, diagnostics = iic.residualize(
    influence=influence_function,
    judge_scores=judge_scores,
    policy="target_policy",
    fold_ids=fold_ids  # Optional: for cross-fitting
)

print(f"R²(φ|S): {diagnostics['r_squared']:.3f}")
print(f"Variance reduction: {diagnostics['var_reduction']:.1%}")
print(f"Regression type: {diagnostics['regression_type']}")

# IIC can be enabled in estimators that support it (disabled by default)
from cje import CalibratedIPS
estimator = CalibratedIPS(sampler, use_iic=True)  # Enable IIC for variance reduction
```

### Oracle Uncertainty (Default: OUA Jackknife)
```python
from cje import CalibratedIPS

# Default: OUA jackknife for oracle uncertainty (recommended)
estimator = CalibratedIPS(sampler, oua_jackknife=True)  # Default
result = estimator.fit_and_estimate()
# Result has both standard_errors and robust_standard_errors

# Optional: Enable bias correction augmentation (engineering fallback)
from cje.calibration import OracleSliceConfig
oracle_config = OracleSliceConfig(
    enable_augmentation=True,
    enable_cross_fit=True,
    min_pi=0.01,
    use_mar=False  # MCAR assumption
)

estimator = CalibratedIPS(
    sampler,
    oracle_slice_config=oracle_config
)

# The augmentation automatically adjusts standard errors
# to account for calibration uncertainty
result = estimator.fit_and_estimate()

# Check oracle uncertainty via OUA jackknife (if enabled)
if result.robust_standard_errors is not None:
    print(f"Standard SE: {result.standard_errors[0]:.4f}")
    print(f"OUA-adjusted SE: {result.robust_standard_errors[0]:.4f}")
    oracle_var = result.robust_standard_errors[0]**2 - result.standard_errors[0]**2
    print(f"Oracle uncertainty contribution: {oracle_var:.6f}")
```

## Configuration Options

### SimcalConfig Parameters
- `ess_floor`: Minimum ESS as fraction (e.g., 0.2 = 20%)
- `var_cap`: Maximum variance (e.g., 1.0 = no increase)
- `include_baseline`: Include raw weights in stack
- `baseline_shrink`: Shrinkage toward baseline (0-1)
- `ridge_lambda`: Ridge regularization for covariance
- `n_folds`: Number of OOF folds if not provided

### Calibration Modes
- **Auto** (default): Automatically selects between monotone and two-stage based on performance
- **Monotone**: Standard isotonic regression (forces monotone relationship)
- **Two-stage**: Flexible g(S)→isotonic for non-monotone relationships
- **Cross-fitted**: K-fold models for DR orthogonality (enable_cross_fit=True)
- **Projection mode**: Always uses "exact" (bisection) for consistency

## Implementation Details

### Ordering Index Flexibility
The `ordering_index` parameter in isotonic calibration allows weights to be monotone in any score:
- **None**: Sort by raw weights (backward compatibility)
- **Judge scores**: Align with human evaluation
- **Calibrated rewards**: Align with outcome models (for DR)

When the ordering index is uncorrelated with weights, isotonic projection produces nearly constant weights - this is expected and provides stabilization.

### Tie Handling
When the ordering index has ties (common with discrete judge scores):
1. Pool weights within tied groups (average)
2. Apply isotonic regression to pooled values
3. Assign same calibrated weight to all tied samples

### Numerical Tolerances
- `EPS = 1e-12`: Machine epsilon for comparisons
- `MEAN_TOL = 1e-10`: Tolerance for mean preservation
- `VAR_TOL = 1.001`: Allow 0.1% slack on variance cap

### Memory Efficiency
- Isotonic regression is O(n log n) time, O(n) space
- Stacked calibration builds K=3 candidates
- Cross-fitting stores K models but applies one at a time

## Common Issues and Solutions

### Issue: "Judge field 'reward' not allowed"
**Cause**: Trying to use 'reward' as judge field to avoid confusion  
**Solution**: Use a different field name in metadata (e.g., 'judge_score')

### Issue: Low calibration R² (< 0.3)
**Cause**: Judge scores poorly predict oracle labels  
**Solution**: 
- Increase oracle coverage (aim for >10%)
- Improve judge prompt/model
- Consider using a different judge
- Check if oracle labels are noisy

### Issue: Nearly constant calibrated weights
**Cause**: Ordering index uncorrelated with importance ratios  
**Solution**: This is expected and actually good - provides maximum variance stabilization

### Issue: Variance cap not satisfied exactly
**Cause**: Numerical precision or infeasible constraint  
**Solution**: Check info dict for 'feasible' flag and 'note' field

### Issue: ESS floor conflicts with variance cap
**Cause**: ESS implies tighter variance constraint than specified  
**Solution**: ESS constraint will dominate (warning issued)

### Issue: Very low oracle coverage (<5%)
**Cause**: Too few labeled samples for reliable calibration
**Solution**: 
- Collect more oracle labels
- Consider using judge scores directly (uncalibrated)
- Use bootstrapping to assess calibration uncertainty

## Testing

The calibration module has comprehensive test coverage:
- `test_stacked_simcal.py`: Stacked SIMCal functionality
- Integration tests verify calibration in full pipeline
- Edge case tests for degenerate inputs

Run tests:
```bash
poetry run pytest cje/tests/ -k calibration
```

## Performance Considerations

### Computational Complexity
- **Isotonic regression**: O(n log n) via PAV
- **Exact projection**: ~30-40 PAV calls (still O(n log n))
- **Stacked SIMCal**: O(nK²) time, O(K²) memory (K=3 candidates)
- **Cross-fitting**: K × isotonic regression cost


### When to Use Each Method

**Use standard calibration when:**
- You have sufficient oracle labels (>100)
- Not using DR methods
- Speed is critical

**Use cross-fitted calibration when:**
- Using DR estimators
- Need orthogonality guarantees
- Have enough data for stable fold models

**Use stacked SIMCal when:**
- Weights have high variance
- Multiple candidate projections make sense
- OOF validation is feasible


## Advanced Topics

### Bootstrapping Calibration Uncertainty
```python
# For low oracle coverage scenarios
n_bootstrap = 100
calibrations = []
for _ in range(n_bootstrap):
    idx = np.random.choice(n_oracle, n_oracle, replace=True)
    cal = JudgeCalibrator()
    result = cal.fit_transform(judge_scores[idx], oracle_labels[idx])
    calibrations.append(result.calibrated_scores)
```

### Debugging SIMCal
```python
# Check intermediate steps
calibrated, info = calibrator.transform(weights, scores, rewards=rewards)
print(f"Mixture weights: {info['mixture_weights']}")
print(f"Variance reduction: {info['var_before']/info['var_after']:.2f}x")
```


## References

- **Isotonic Regression**: Robertson et al. (1988), "Order Restricted Statistical Inference"
- **PAV Algorithm**: Ayer et al. (1955), "An Empirical Distribution Function for Sampling with Incomplete Information"  
- **Majorization**: Marshall & Olkin (1979), "Inequalities: Theory of Majorization"
- **SIMCal**: CJE paper (2025), "Surrogate-Indexed Monotone Calibration"
- **Cross-fitting**: Chernozhukov et al. (2018), "Double/Debiased Machine Learning"

## Summary

The calibration module provides three essential transformations for causal inference: mapping judge scores to oracle labels, stabilizing importance weights through SIMCal, and enabling cross-fitted models for DR methods. Each calibration type maintains mean preservation for unbiased estimation while controlling variance through different mechanisms.


# ============================================================================
# FILE: cje/data/README.md
# ============================================================================

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
- `prompt_id`: Unique identifier (checked in top-level, then metadata, auto-generated from prompt hash if missing)
- `prompt`: Input text/context
- `response`: Generated output
- `base_policy_logprob`: Log probability under logging policy
- `target_policy_logprobs`: Dict of log probs for target policies

### Optional Fields
- `reward`: Calibrated reward in [0, 1] (required for PrecomputedSampler)
- `metadata`: Dict containing additional fields like:
  - `judge_score`: Raw judge evaluation
  - `oracle_label`: Ground truth label

### Example JSONL Entry
```json
{
  "prompt_id": "example_001",
  "prompt": "What is machine learning?",
  "response": "Machine learning is a subset of AI...",
  "base_policy_logprob": -45.67,
  "target_policy_logprobs": {
    "gpt4": -42.31,
    "claude": -44.89
  },
  "reward": 0.85,
  "metadata": {
    "judge_score": 0.82,
    "oracle_label": 0.90
  }
}
```

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
- Honest confidence intervals via robust_standard_errors
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
oracle_mask = np.array([s.metadata.get("oracle_label") is not None for s in dataset.samples])
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
dataset = factory.loader.load_from_source(source, target_policies=["gpt4"])
```

### Fresh Draws for DR
```python
from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample

# Create fresh draws
samples = [
    FreshDrawSample(
        prompt_id="p1",
        target_policy="gpt4",
        judge_score=0.9,
        draw_idx=0
    ),
    # ... more samples
]

fresh_dataset = FreshDrawDataset(
    target_policy="gpt4",
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

# ============================================================================
# FILE: cje/diagnostics/README.md
# ============================================================================

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

# ============================================================================
# FILE: cje/estimators/README.md
# ============================================================================

# CJE Estimators

## Overview

Causal inference methods for unbiased off-policy evaluation of LLMs, transforming logged data and importance weights into reliable policy value estimates with proper uncertainty quantification.

## Estimator Hierarchy

```
BaseCJEEstimator (abstract)
├── CalibratedIPS              # IPS with optional SIMCal weight calibration
│   └── OrthogonalizedCalibratedIPS  # OC-IPS with robustness to calibration errors
├── StackedDREstimator         # Optimal stacking of DR estimators
└── DREstimator                # Doubly robust base (abstract)
    ├── DRCPOEstimator         # Basic DR with CPO
    ├── OrthogonalizedCalibratedDRCPO  # OC-DR-CPO with first-order insensitivity
    ├── MRDREstimator          # Multiple robust DR
    ├── TMLEEstimator          # Targeted maximum likelihood
    └── TRCPOEstimator         # Triply robust CPO
```

## Core Concepts

### 1. Importance Sampling (IPS)
Foundation of off-policy evaluation. Reweights logged data to estimate performance under new policies using importance weights W = π_target/π_base.

### 2. SIMCal Weight Calibration
Stabilizes importance weights through monotone projection with variance control. Independent of reward calibration. CalibratedIPS now uses outer CV by default (`use_outer_cv=True`) for honest inference accounting for weight learning uncertainty.

### 3. Doubly Robust (DR) Estimation
Combines direct method (outcome model) with IPS correction. Provides two chances to get the estimate right - if either the outcome model OR the weights are correct, DR is consistent.

### 4. Multiple Robustness (MRDR)
Achieves robustness to outcome model misspecification, propensity score misspecification, and both simultaneously through cross-fitting.

### 5. Targeted Learning (TMLE)
Optimally combines outcome models and importance weights through targeted fluctuation to achieve optimal asymptotic efficiency.

### 6. Estimator Stacking
Forms optimal convex combination of DR estimators by minimizing combined influence function variance. Uses oracle IC approach (w₀ᵀφ(Z)) with ridge regularization for numerical stability.

### 7. Orthogonalized Estimators
Achieve first-order insensitivity to nuisance estimation errors:
- **OC-IPS**: Robust to errors in f̂(S) and m̂(S)
- **OC-DR-CPO**: Additionally robust to q̂(X,A) errors

### 8. Triply Robust (TR-CPO)
Robust to weight calibration, reward calibration, and outcome model errors simultaneously. TR-CPO-E variant (recommended) uses m̂(S)=E[W|S] for variance reduction.

## File Structure

```
estimators/
├── base_estimator.py               # Abstract base
├── calibrated_ips.py              # IPS with optional SIMCal
├── orthogonalized_ips.py          # OC-IPS
├── stacking.py                    # Optimal stacking
├── dr_base.py                     # DR base + DRCPOEstimator
├── orthogonalized_calibrated_dr.py # OC-DR-CPO
├── mrdr.py                        # Multiple robust DR
├── tmle.py                        # TMLE
├── tr_cpo.py                      # Triply robust CPO
└── outcome_models.py              # Outcome models
```

## Common Interface

All estimators follow the same pattern:

```python
from cje import CalibratedIPS, PrecomputedSampler
from cje.calibration import calibrate_dataset

# 1. Calibrate dataset (if using reward calibration)
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    enable_cross_fit=True,  # Required for DR methods
    calibration_mode='auto'  # Auto-selects monotone or two-stage
)

# 2. Create sampler with data
sampler = PrecomputedSampler(calibrated_dataset)

# 3. Initialize estimator
# For IPS:
estimator = CalibratedIPS(sampler)
# For DR (requires fresh draws):
estimator = StackedDREstimator(sampler)

# 4. Fit and estimate
result = estimator.fit_and_estimate()

# 5. Access results
estimates = result.estimates           # Point estimates
std_errors = result.standard_errors    # Standard errors
diagnostics = result.diagnostics       # Health metrics
influence = result.influence_functions # For inference
```

## Default Recommendation

**Use StackedDREstimator** - Combines multiple DR methods (DR-CPO, TMLE, MRDR, OC-DR-CPO, TR-CPO-E) via optimal weighting to minimize variance. Requires fresh draws. Provides modest improvements (1-5% SE reduction) over best single method.

## Refusal Gates

CalibratedIPS includes safety mechanisms that detect when estimates would be unreliable due to poor overlap:

1. **ESS < 30%**: Over 70% of data effectively ignored
2. **Raw near-zero > 85%**: Severe distribution mismatch that calibration may mask
3. **Top 5% concentration > 30% with CV > 2.0**: Few outliers dominate estimate

Default behavior: Provides estimates with warnings. Set `refuse_unreliable=True` to return NaN for unreliable policies.

```python
# Default: Warn but estimate
estimator = CalibratedIPS(sampler)

# Strict mode: Return NaN for unreliable
estimator = CalibratedIPS(sampler, refuse_unreliable=True)
```


## Key Design Decisions

1. **Transparency**: Default to warnings over silent failures
2. **Influence Functions**: Always computed for proper inference
3. **Diagnostics**: Automatically attached to all results
4. **Modularity**: DR estimators compose CalibratedIPS for weights

## Outcome Models

- **IsotonicOutcomeModel**: Monotonic regression with judge scores, no parametric assumptions
- **LinearOutcomeModel**: Simple linear regression baseline, fast and stable
- **CalibratorBackedOutcomeModel**: Uses same calibrator as rewards for consistency
- **WeightedIsotonicOutcomeModel**: Policy-specific models for MRDR with omega weights ("w", "w2", or "snips")

## Fresh Draws

DR estimators auto-load fresh draws from:
- `data/{policy}_responses.jsonl`
- `data/responses/{policy}_responses.jsonl`
- `data/{policy}_fresh.jsonl`
- `data/fresh_draws/{policy}.jsonl`

Or add manually:
```python
estimator.add_fresh_draws('policy', FreshDrawDataset(samples=[...]))
```



## Standard Errors and Uncertainty Quantification

### Three Types of Standard Errors

1. **`standard_errors`**: Base uncertainty from sampling (includes MC variance for DR estimators)
2. **`robust_standard_errors`**: Adds oracle uncertainty from finite calibration sample
3. **Method-specific robust SEs**: Some estimators add additional robustness adjustments

### IPS Standard Errors
```python
# Base SE from influence functions
standard_errors = np.std(influence_functions, ddof=1) / np.sqrt(n)

# Robust SE adds oracle uncertainty (only when oracle_coverage < 100%)
robust_standard_errors = np.sqrt(standard_errors² + oracle_variance)
```

### DR Standard Errors (with Monte Carlo Variance)
```python
# Base IF variance + MC variance from finite fresh draws
standard_errors = np.sqrt(if_variance/n + mc_variance)

# Robust SE adds oracle uncertainty on top
robust_standard_errors = np.sqrt(standard_errors² + oracle_variance)
```

**Important**: For DR estimators, `standard_errors` already includes MC variance. Check `mc_variance_included: True` in metadata.

### Automatic MC Variance Handling
When only one fresh draw per prompt (M=1), DR estimators automatically use a conservative upper bound:
- Total variance across single draws bounds within-prompt variance
- Capped at 0.25 for binary [0,1] outcomes
- Mixed cases (some M≥2, some M=1) combine exact computation with upper bound


## Advanced Features

### Stacked DR Configuration
```python
StackedDREstimator(
    sampler,
    estimators=['dr-cpo', 'tmle', 'mrdr', 'oc-dr-cpo', 'tr-cpo-e'],
    covariance_regularization=1e-4,  # Ridge regularization strength
    n_folds=20                       # Cross-fitting folds
)
```
Uses regularized covariance estimation to handle highly correlated component estimators.

### Oracle Uncertainty Augmentation (OUA)
All estimators support OUA via delete-one-fold jackknife to account for calibrator uncertainty from finite oracle samples. **Note: OUA is automatically skipped at 100% oracle coverage** since there's no oracle uncertainty when all samples have ground truth labels.

```python
# Enabled by default
estimator = CalibratedIPS(sampler, oua_jackknife=True)

# Access OUA-adjusted standard errors
result = estimator.fit_and_estimate()
robust_ses = result.robust_standard_errors  # Includes oracle uncertainty

# At 100% oracle coverage: robust_ses == standard_errors (no OUA applied)
# At <100% coverage: robust_ses >= standard_errors (OUA adds uncertainty)
```

### Honest Inference with Outer CV
CalibratedIPS uses outer cross-validation by default (`use_outer_cv=True`) to account for weight learning uncertainty:
```python
# Default: Outer CV enabled
estimator = CalibratedIPS(sampler)  # use_outer_cv=True by default

# Customize settings
estimator = CalibratedIPS(
    sampler,
    n_outer_folds=10,       # More folds for stability
    honest_iic=True         # Apply honest IIC for variance reduction
)
```

### Custom Estimators
Inherit from `BaseCJEEstimator` or `DREstimator` and implement `fit()` and `estimate()`. Always compute and store influence functions in `_influence_functions`.



## Common Issues

- **NaN estimates**: Check ESS in diagnostics. Likely poor overlap - try DR methods with fresh draws
- **Low ESS**: Policies too different. Consider collecting more diverse base data
- **DR fails**: All DR methods require fresh draws. Generate them first
- **Underestimated SEs**: Ensure `use_outer_cv=True` for honest inference (enabled by default)
- **Different results between runs**: Set random seeds for reproducibility in cross-fitting

## Implementation Notes

### Cross-Fitting
DR estimators use k-fold cross-fitting for orthogonality:
- Unified fold system via `cje.data.folds` (deterministic: `hash(prompt_id) % k`)
- Each fold gets predictions from model trained on other folds
- Prevents overfitting in outcome models

### Weight Caching
Estimators cache computed weights to avoid recomputation across policies.

### Influence Functions
Always computed and stored for proper inference, policy comparison, and diagnostics.

## References

- **IPS**: Horvitz & Thompson (1952)
- **Doubly Robust**: Robins et al. (1994)
- **TMLE**: van der Laan & Rubin (2006)
- **SIMCal**: Score-indexed monotone calibration (2024)

## Summary

Comprehensive toolkit for causal inference on LLM outputs, from simple importance sampling to sophisticated multiply-robust methods. All estimators follow the same interface, compute influence functions, and provide transparent diagnostics for reliability assessment. **StackedDREstimator is recommended for production use** when fresh draws are available.


# ============================================================================
# FILE: cje/interface/README.md
# ============================================================================

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

For more details, see the full documentation (coming soon on cimo-labs.com).


# ============================================================================
# FILE: cje/teacher_forcing/README.md
# ============================================================================

# CJE Teacher Forcing Module

## Overview

The teacher forcing module computes log probabilities log P(response|prompt) for importance weight calculation in CJE. It provides robust, production-ready implementations with automatic fallback mechanisms and support for various chat templates.

## When to Use

### Use **compute_teacher_forced_logprob** when:
- You need raw log P(response|prompt) for completion-style inputs
- You're working directly with the Fireworks API
- You want fine control over the computation method

### Use **compute_chat_logprob** when:
- You have chat-formatted conversations
- You need automatic template detection for Fireworks models
- You want to score assistant replies in multi-turn dialogues

### Use **Template configs** when:
- Working with specific model families (Llama, HuggingFace)
- Converting between chat and completion formats
- Ensuring correct tokenization boundaries

## File Structure

```
teacher_forcing/
├── __init__.py              # Public API exports
├── api/
│   ├── __init__.py
│   └── fireworks.py         # Fireworks API integration
├── chat.py                  # Chat conversation utilities
└── templates/               # Chat template configurations
    ├── __init__.py
    ├── base.py              # Abstract base class
    ├── fireworks.py         # Fireworks model templates
    ├── huggingface.py       # HuggingFace templates
    └── llama.py             # Llama-specific templates
```

## Core Concepts

### 1. Teacher Forcing Method
Computes log P(response|prompt) by feeding the concatenated prompt+response to the model and extracting token-level log probabilities. This avoids sampling bias from autoregressive generation.

### 2. One-Call vs Two-Call Approaches
- **One-call**: Uses byte counting to find prompt/response boundary (~89% of cases)
- **Two-call**: Fallback using difference of two API calls (100% reliability)

### 3. Chat Templates
Different models use different formatting for chat conversations. Templates handle:
- Role markers (user/assistant/system)
- Special tokens (<|begin_of_text|>, <|eot_id|>)
- Proper tokenization boundaries

## Common Interface

### Basic Teacher Forcing
```python
from cje.teacher_forcing import compute_teacher_forced_logprob

result = compute_teacher_forced_logprob(
    prompt="What is machine learning?",
    response="Machine learning is a subset of AI...",
    model="accounts/fireworks/models/llama-v3p2-3b-instruct",
    temperature=1.0
)

if result.is_valid:
    print(f"Log probability: {result.value}")
    print(f"Method used: {result.metadata['method']}")
```

### Chat Conversations
```python
from cje.teacher_forcing import compute_chat_logprob

chat = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."}
]

result = compute_chat_logprob(
    chat=chat,
    model="accounts/fireworks/models/llama-v3p2-3b-instruct"
)

# Computes log P("The answer is 4." | user message + template)
```

### Custom Templates
```python
from cje.teacher_forcing import (
    HuggingFaceTemplateConfig,
    Llama3TemplateConfig,
    convert_chat_to_completions
)

# For HuggingFace models
hf_config = HuggingFaceTemplateConfig("meta-llama/Llama-3.2-3B-Instruct")

# For Llama 3 models with explicit template
llama3_config = Llama3TemplateConfig()

# Convert chat to completion format
prompt_only, prompt_plus_reply = convert_chat_to_completions(chat, hf_config)
```

## Implementation Details

### Byte Counting Algorithm
The one-call approach uses UTF-8 byte counting to find the exact boundary between prompt and response tokens:

```python
def find_boundary_by_bytes_safe(tokens, prompt, reconstructed_text):
    prompt_bytes = prompt.encode("utf-8", errors="surrogatepass")
    running = b""
    
    for idx, tok in enumerate(tokens):
        tok_bytes = tok.encode("utf-8", errors="surrogatepass")
        running += tok_bytes
        
        if len(running) == len(prompt_bytes):
            return True, idx + 1, "exact_match"
        elif len(running) > len(prompt_bytes):
            # Token spans boundary - need fallback
            return False, None, "boundary_spans_token"
```

### Two-Call Fallback
When byte counting fails (e.g., token spans boundary), the system automatically falls back to:
1. Call 1: Get log P(prompt)
2. Call 2: Get log P(prompt + response)
3. Result: log P(response|prompt) = Call 2 - Call 1

This ensures 100% reliability at the cost of an extra API call.

## Key Design Decisions

### 1. **Automatic Fallback**
Rather than failing when byte counting doesn't work, the system transparently falls back to the two-call method. This ensures reliability while optimizing for efficiency.

### 2. **Template Abstraction**
Chat templates are abstracted into configuration classes, allowing easy extension for new model families without changing core logic.

### 3. **Explicit Error Handling**
All failure modes return structured `LogProbResult` objects with clear status codes and error messages, never exceptions or magic values.

### 4. **UTF-8 Safety**
Uses `surrogatepass` error handling to deal with edge cases in tokenization, ensuring robustness with multilingual text.

### 5. **Diagnostic Metadata**
Every result includes metadata about the computation method, token counts, and failure reasons for debugging and monitoring.

## Common Issues and Solutions

### Issue: "boundary_spans_token" in metadata
**Cause**: A single token contains both prompt and response text
**Solution**: System automatically uses two-call fallback

### Issue: "echo_mismatch" error
**Cause**: API normalized whitespace or line endings differently
**Solution**: Check prompt formatting, system will use fallback

### Issue: High API latency
**Cause**: Two-call fallback doubles API requests
**Solution**: Ensure prompts don't have trailing whitespace, use shorter prompts when possible

### Issue: Template not found for model
**Cause**: Using non-Fireworks model without explicit template
**Solution**: Provide explicit `HuggingFaceTemplateConfig` or `Llama3TemplateConfig`

## Performance

### Typical Metrics
- **One-call success rate**: ~89% of requests
- **API latency**: 200-400ms (one-call), 400-800ms (two-call) 
- **Token limit**: Handles up to model's context length

### Optimization Tips
- Remove trailing whitespace from prompts
- Keep prompts under 10K characters when possible
- Reuse template configs across multiple calls
- Batch requests when computing multiple log probabilities

## Advanced Usage

### Force Two-Call Method
```python
# Skip byte counting attempt
result = compute_teacher_forced_logprob(
    prompt=prompt,
    response=response,
    model=model,
    force_two_call=True  # Always use two-call
)
```

### Custom API Configuration
```python
result = compute_teacher_forced_logprob(
    prompt=prompt,
    response=response,
    model=model,
    api_key="your-api-key",
    api_base="https://custom-endpoint.com"
)
```

### System Prompts in Chat
```python
result = compute_chat_logprob(
    chat=chat,
    model=model,
    system_prompt="You are a helpful assistant."
)
```

## Summary

The teacher forcing module provides reliable computation of log probabilities for CJE's importance weights. With automatic fallback, comprehensive template support, and production-ready error handling, it ensures accurate weight calculation across diverse models and use cases.

# ============================================================================
# FILE: cje/tests/README.md
# ============================================================================

# CJE Test Suite

## Overview

The CJE test suite has been radically simplified to focus on end-to-end testing with real data. We've reduced from 28 test files (238 tests) to 7 core test files (~80 tests) while maintaining comprehensive coverage of critical functionality.

## File Structure

```
tests/
├── conftest.py                    # Shared fixtures and arena data loaders
├── run_all_tests.py              # Test runner script
│
├── E2E Tests                    
│   ├── test_e2e_estimators.py    # Complete pipelines for all estimators
│   ├── test_e2e_features.py      # IIC, SIMCal, cross-fitting
│   └── test_interface_integration.py # High-level API testing
│
├── Core Tests
│   ├── test_infrastructure.py    # Critical infrastructure and edge cases
│   ├── test_unified_folds.py     # Comprehensive fold management
│   └── test_mc_variance.py       # Monte Carlo variance testing
│
└── data/                          # Test datasets
    ├── arena_sample/              # Real Arena 10K subset (100 samples)
    │   ├── dataset.jsonl          # Main dataset with judge scores
    │   └── responses/             # Fresh draws for DR estimation
    └── *.jsonl                    # Synthetic test data for edge cases
```

## Core Concepts

### 1. End-to-End Focus
Instead of testing individual functions, we test complete pipelines:
- Load data → Calibrate → Create sampler → Estimate → Validate results
- All E2E tests use real Arena data for authentic testing
- Tests verify user-visible outcomes, not implementation details

### 2. Arena Sample Data
Real subset from Arena 10K evaluation:
- 100 samples with actual judge scores and oracle labels
- 4 target policies: clone, premium, parallel_universe_prompt, unhelpful
- Fresh draws for each policy enabling DR estimation
- Ground truth for validation

### 3. Fixture Architecture
Shared fixtures in `conftest.py` provide consistent test data:
- **arena_sample**: Real 100-sample Arena dataset
- **arena_fresh_draws**: Filtered fresh draws matching dataset prompts
- **arena_calibrated**: Pre-calibrated Arena dataset
- **synthetic datasets**: Edge case testing (NaN, extreme weights)

### 4. Test Philosophy
- **Real Data Priority**: Use arena sample for integration tests
- **Complete Workflows**: Test what users actually do
- **Fast Feedback**: Most tests run in < 1 second
- **Clear Intent**: Each test has one clear purpose

## Running Tests

```bash
# Run all tests
poetry run pytest cje/tests/

# Run E2E tests only (recommended for quick validation)
poetry run pytest cje/tests/test_e2e*.py -q

# Run specific test files
poetry run pytest cje/tests/test_e2e_estimators.py -v
poetry run pytest cje/tests/test_unified_folds.py

# Run with markers
poetry run pytest cje/tests -m e2e
poetry run pytest cje/tests -m "not slow"

# With coverage
poetry run pytest --cov=cje --cov-report=html cje/tests/

# Quick health check (single E2E test)
poetry run pytest cje/tests/test_e2e_estimators.py::TestE2EEstimators::test_calibrated_ips_pipeline -v
```

## Writing New Tests

When adding tests, follow these guidelines:

1. **Prefer E2E tests** - Test complete workflows
2. **Use arena data** - Real data finds real bugs
3. **Keep it focused** - Each test should have one clear purpose
4. **Document intent** - Clear test names and docstrings

```python
def test_new_feature_workflow(arena_sample):
    """Test that new feature improves estimates."""
    # 1. Calibrate dataset
    calibrated, cal_result = calibrate_dataset(
        arena_sample,
        judge_field="judge_score",
        oracle_field="oracle_label"
    )
    
    # 2. Create sampler
    sampler = PrecomputedSampler(calibrated)
    
    # 3. Run estimation with new feature
    estimator = YourEstimator(sampler, new_feature=True)
    results = estimator.fit_and_estimate()
    
    # 4. Validate results
    assert len(results.estimates) == 4  # 4 policies
    assert all(0 <= e <= 1 for e in results.estimates)
    # Test that new feature had expected effect
    assert results.metadata["new_feature_applied"] == True
```

## Key Design Decisions

### 1. **Simplified Test Suite**
Reduced from 238 tests to ~80 focused tests:
- 73% reduction in test count
- Comprehensive coverage maintained
- Faster execution and easier maintenance
- Focus on integration over unit testing

### 2. **Real Data Testing**
Arena sample data provides ground truth validation:
- Catches regressions in production scenarios
- Tests all estimators with same data
- Reveals integration issues unit tests miss

### 3. **E2E Testing Priority**
Complete workflows over isolated functions:
- Test what users actually do
- Catch integration bugs
- Validate full pipelines
- Ensure components work together

### 4. **Unified Fold System**
Consistent cross-validation across all components:
- Hash-based fold assignment from prompt_id
- Prevents data leakage
- Ensures reproducibility
- Single source of truth (`data/folds.py`)

## Common Issues

### "FileNotFoundError for test data"
Ensure running from project root:
```bash
cd /path/to/causal-judge-evaluation
poetry run pytest cje/tests/
```

### "Slow test execution"
Skip slow tests during development:
```bash
poetry run pytest -m "not slow" cje/tests/
```

### "Import errors"
Install package in development mode:
```bash
poetry install
# or
pip install -e .
```

## Performance

- **E2E tests**: < 2 seconds each
- **Infrastructure tests**: < 1 second each
- **Full suite**: ~15 seconds for all tests

Test execution tips:
- Use `-x` to stop on first failure
- Use `-k pattern` to run tests matching pattern
- Use `--lf` to run last failed tests
- Use `-q` for quiet output during development
- Run E2E tests first for quick validation

## Summary

The CJE test suite has been transformed from 238 scattered unit tests to ~80 focused tests that test real workflows with real data. This simplified approach catches more integration issues, runs faster, and is easier to maintain while providing comprehensive coverage of all estimators, calibration methods, and diagnostic tools.

# ============================================================================
# FILE: cje/utils/README.md
# ============================================================================

# CJE Utils Module

## Overview

Utility functions for export and analysis in CJE. This module provides practical tools for saving estimation results and debugging extreme weight issues.

## When to Use

### Use **Export Utilities** when:
- You need to save estimation results for reporting
- You want JSON or CSV output formats
- You need to share results with non-Python tools
- You're creating reproducible analysis pipelines

### Use **Extreme Weights Analysis** when:
- Debugging weight explosion issues
- Understanding which samples dominate estimates
- Identifying problematic log probability ratios
- Generating diagnostic reports for stakeholders

## File Structure

```
utils/
├── __init__.py                  # Re-exports and backward compatibility
├── export.py                    # JSON/CSV export functions
└── extreme_weights_analysis.py # Weight debugging and reporting
```

## Core Concepts

### 1. Result Export
Converts EstimationResult objects to standard formats:
- **JSON**: Hierarchical format with metadata and diagnostics
- **CSV**: Tabular format for spreadsheet analysis
- Handles numpy arrays, NaN values, and complex nested structures

### 2. Extreme Weights Analysis
Deep dive into importance weight behavior:
- Identifies samples with highest/lowest weights
- Tracks consistently extreme samples across policies
- Computes ESS and weight statistics
- Generates both JSON and text reports


## Common Interface

### Export Results

```python
from cje.utils import export_results_json, export_results_csv

# After running estimation
result = estimator.fit_and_estimate()

# Export to JSON with full details
export_results_json(
    result,
    "results/analysis.json",
    include_diagnostics=True,
    include_metadata=True
)

# Export to CSV for Excel
export_results_csv(
    result,
    "results/summary.csv",
    include_ci=True
)
```

### Analyze Extreme Weights

```python
from cje.utils import analyze_extreme_weights

# Debug weight issues
json_report, text_report = analyze_extreme_weights(
    dataset=dataset,
    sampler=sampler,
    raw_weights_dict=raw_weights,
    calibrated_weights_dict=calibrated_weights,
    n_extreme=10,  # Top/bottom 10 samples
    output_dir=Path("diagnostics/")
)

# Reports saved to diagnostics/extreme_weights_analysis.{json,txt}
print(text_report)  # Human-readable summary
```


## Key Design Decisions

### 1. **Graceful Serialization**
Export functions handle complex types:
- Numpy arrays → lists
- NaN → null (JSON) or empty (CSV)
- Complex objects → string representations
- Never fails on serialization errors

### 2. **Comprehensive Weight Analysis**
Extreme weights analysis provides multiple views:
- Per-policy statistics
- Cross-policy patterns
- Sample-level details
- Both JSON (programmatic) and text (human) formats


## Common Issues

### "Can't serialize object to JSON"
The export functions handle most types, but custom objects may need:
```python
# Add to metadata as strings
result.metadata["custom_obj"] = str(my_custom_object)
```

### "Extreme weights report too large"
Limit number of samples analyzed:
```python
analyze_extreme_weights(..., n_extreme=5)  # Only top/bottom 5
```

## Performance

- **Export**: O(n_policies) - Fast even for large results
- **Extreme weights**: O(n_samples × n_policies) - Can be slow for large datasets

For large datasets:
- Export in batches if memory constrained
- Analyze subset of policies for extreme weights

## Summary

The utils module provides essential tools for CJE workflows: exporting results for reporting and debugging weight issues through detailed analysis. These utilities handle the practical aspects of working with CJE results in production environments.

# ============================================================================
# FILE: cje/visualization/README.md
# ============================================================================

# CJE Visualization Module

## Overview

The visualization module provides comprehensive diagnostic plots for understanding and validating CJE analysis results. It offers specialized dashboards for weight diagnostics, doubly robust diagnostics, calibration assessment, and policy estimate comparisons to help practitioners audit assumptions and interpret results.

## When to Use

### Use **Weight Dashboards** when:
- You need to diagnose weight explosion or concentration
- You want to understand effective sample size (ESS) issues
- You're comparing raw vs calibrated weight behaviors
- You need to identify which samples dominate estimates

### Use **DR Dashboard** when:
- You're using doubly robust estimators
- You need to check orthogonality assumptions
- You want to understand DM vs IPS contributions
- You need to diagnose influence function tail behavior

### Use **Calibration Plots** when:
- You want to visualize judge → oracle calibration
- You need to assess calibration quality (ECE, RMSE)
- You're comparing before/after calibration alignment
- You want to understand calibration transformations

### Use **Estimate Plots** when:
- You need to compare policy performance
- You want confidence intervals visualized
- You have oracle ground truth for validation
- You need publication-ready forest plots

## File Structure

```
visualization/
├── __init__.py              # Public API with backward-compatible aliases
├── calibration.py           # Calibration transformation and reliability plots
├── dr_dashboards.py         # Doubly robust diagnostic visualizations
├── estimates.py             # Policy performance forest plots
└── weight_dashboards.py     # Weight diagnostic dashboards (summary & detailed)
```

## Core Concepts

### 1. Weight Diagnostics
Comprehensive analysis of importance weight behavior:
- **ESS tracking**: Monitor effective sample size degradation
- **Tail analysis**: CCDF plots to identify heavy tails
- **Concentration metrics**: How many samples contribute X% of weight
- **Calibration impact**: Compare raw vs calibrated distributions
- **Judge correlation**: Optional analysis of weight-judge score relationships

### 2. DR Diagnostics
Specialized plots for doubly robust estimation:
- **Component analysis**: Direct method vs IPS correction contributions
- **Orthogonality checks**: Score function mean ± 2SE for validity
- **Influence functions**: EIF tail behavior and stability

### 3. Calibration Assessment
Visual tools for judge calibration quality:
- **Transformation curves**: Visualize f: judge → oracle mapping
- **Reliability diagrams**: Bin-wise calibration alignment
- **Improvement metrics**: ECE and RMSE before/after calibration

### 4. Estimate Visualization
Clear presentation of final results:
- **Forest plots**: Point estimates with confidence intervals
- **Policy comparison**: Visual ranking and uncertainty
- **Oracle validation**: Compare estimates to ground truth when available

## Common Interface

All visualization functions follow consistent patterns:

```python
from cje.visualization import (
    plot_weight_dashboard_summary,
    plot_weight_dashboard_detailed,
    plot_dr_dashboard,
    plot_calibration_comparison,
    plot_policy_estimates
)

# Weight diagnostics - summary dashboard (6 panels)
fig, metrics = plot_weight_dashboard_summary(
    raw_weights_dict=raw_weights,
    calibrated_weights_dict=calibrated_weights,
    save_path="diagnostics/weights_summary.png"
)

# Weight diagnostics - detailed per-policy view
fig, metrics = plot_weight_dashboard_detailed(
    raw_weights_dict=raw_weights,
    calibrated_weights_dict=calibrated_weights,
    judge_scores=judge_scores,  # Optional for correlation analysis
    save_path="diagnostics/weights_detailed.png"
)

# DR diagnostics (requires DR estimation result)
fig, summary = plot_dr_dashboard(
    estimation_result=dr_result,
    figsize=(15, 5)
)

# Calibration comparison
fig = plot_calibration_comparison(
    judge_scores=judge_scores,
    oracle_labels=oracle_labels,
    calibrated_scores=calibrated_scores,
    save_path="diagnostics/calibration.png"
)

# Policy estimates
fig = plot_policy_estimates(
    estimates={"policy_a": 0.75, "policy_b": 0.82},
    standard_errors={"policy_a": 0.02, "policy_b": 0.03},
    oracle_values={"policy_a": 0.74, "policy_b": 0.85}
)
```

## Key Design Decisions

### 1. **Multi-Panel Dashboards**
Complex diagnostics are organized into focused panels:
- Each panel answers one specific question
- Panels are visually connected but independently interpretable
- Summary metrics accompany visual diagnostics

### 2. **Dual Dashboard Approach**
Two complementary weight visualizations:
- **Summary dashboard**: 6-panel overview across all policies
- **Detailed dashboard**: Per-policy analysis with judge score correlation
- Each serves distinct analysis needs with clear naming

### 3. **Automatic Metric Computation**
Visualizations compute and display key metrics:
- ESS and effective sample percentages
- Calibration errors (ECE, RMSE)
- Weight concentration statistics
- No need for separate metric calculation

### 4. **Save Options**
All plots support optional saving:
- Automatic file extension handling
- High DPI for publication quality
- Consistent naming conventions

## Common Issues

### "No matplotlib backend"
Install matplotlib with GUI support:
```bash
pip install matplotlib[gui]
```

### "Figure too small for content"
Adjust figsize parameter:
```python
plot_weight_dashboard_summary(..., figsize=(16, 14))
```

### "Missing diagnostics object"
Ensure estimator was run with diagnostics enabled:
```python
result = estimator.fit_and_estimate(compute_diagnostics=True)
```

## Performance

- **Weight dashboards**: O(n_samples × n_policies) for metric computation
- **DR dashboards**: O(n_samples) for influence function analysis  
- **Calibration plots**: O(n_samples × n_bins) for binning operations
- **Memory**: Dashboards create temporary copies for sorting/binning

For large datasets (>100k samples), consider:
- Sampling for scatter plots
- Reducing bin counts
- Pre-computing metrics
- Using summary dashboard instead of detailed for initial analysis

## Summary

The visualization module transforms complex statistical diagnostics into interpretable visual insights. It helps practitioners validate assumptions, diagnose issues, and communicate results effectively through carefully designed multi-panel dashboards and focused diagnostic plots.

# ============================================================================
# END OF DOCUMENTATION
# ============================================================================

# Summary:
# - Total README files: 12
# - Total lines of documentation: 5452
# - Modules documented: 14
