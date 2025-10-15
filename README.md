<div align="left">
  <img src="CJE_logo.jpg" alt="CJE Logo" width="250">
</div>

# CJE - Causal Judge Evaluation

[![Docs](https://img.shields.io/badge/docs-cimolabs.com-blue)](https://cimolabs.com/cje)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-green)](https://github.com/cimo-labs/cje/actions)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Turn noisy LLM-judge scores into unbiased policy estimates with valid confidence intervals.**

CJE calibrates judge scores using a small oracle slice (5-10% coverage), then delivers statistically rigorous estimates.

## How It Works

```
Judge scores + small oracle slice (5-10%) → Calibrate to oracle scale
                                          → Unbiased policy estimates
                                          → Valid 95% confidence intervals
```

**What you get:**
- **Unbiased estimates**: Judge scores mapped to oracle outcome scale via isotonic regression (unbiased, de-noising)
- **Small label budget**: 5-10% oracle coverage often sufficient
- **Valid CIs**: Account for both sampling uncertainty and calibration uncertainty (when oracle coverage is <100%)


See [`cje/calibration/README.md`](cje/calibration/README.md#why-isotonic-regression-for-reward-calibration) for technical details.

## Calibration Methods

CJE provides two calibration modes for mapping judge scores to oracle outcomes:

### Monotone (Default)
Standard isotonic regression enforces: *higher judge score → no worse expected outcome*. Simple, stable, works well when the judge-oracle relationship is already monotone.

### Two-Stage (Flexible)
Learns smooth transformation g(S) → rank → isotonic. Handles non-monotone patterns (e.g., length bias, regional miscalibration) while maintaining final monotonicity guarantee.

<div align="center">
  <img src="two_stage_comparison.png" alt="Calibration Comparison" width="100%">
</div>

**When to use two-stage:**
- Regional miscalibration (monotone works well at low/high but poorly at mid-range)
- Length bias (judge gives same score to different-quality responses based on length)
- Non-monotone empirical E[Oracle|Judge] relationship

**Auto mode:** CJE automatically selects the better method via cross-validation (1-SE rule).

```python
# Let CJE choose automatically (default)
result = analyze_dataset(fresh_draws_dir="responses/")

# Or force a specific mode
result = analyze_dataset(
    fresh_draws_dir="responses/",
    calibration_mode="two_stage"  # or "monotone"
)
```

## Installation

```bash
pip install cje-eval
```

## 🚀 Try it Now - Interactive Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_arena_demo.ipynb)

## Quick Start

### Minimal Example

```python
from cje import analyze_dataset

# Compare policies on an eval set
result = analyze_dataset(fresh_draws_dir="responses/")

# Get estimates with confidence intervals
for policy, est, se in zip(
    result.metadata["target_policies"],
    result.estimates,
    result.standard_errors
):
    print(f"{policy}: {est:.3f} ± {1.96*se:.3f}")
```

### Data Format

**Directory structure:** One JSONL file per policy
```
responses/
├── model_a_responses.jsonl
└── model_b_responses.jsonl
```

**Minimal record** (inside each file):
```json
{"prompt_id": "eval_0", "judge_score": 0.85}
{"prompt_id": "eval_1", "judge_score": 0.72}
```

**With calibration** (add oracle labels to 5-10% of samples):
```json
{"prompt_id": "eval_0", "judge_score": 0.85, "oracle_label": 0.86}
{"prompt_id": "eval_1", "judge_score": 0.72}
```

CJE automatically:
- Discovers policies from filenames (`model_a_responses.jsonl` → policy `"model_a"`)
- Applies AutoCal-R when oracle labels are present
- Uses cluster-robust SEs for paired comparisons (when same prompts across policies)
- Returns unbiased estimates with valid 95% CIs

### Paired Comparisons

When comparing policies on the **same prompts** (paired design), CJE automatically uses cluster-robust standard errors:

```python
# Both files must have matching prompt_ids for pairing
result = analyze_dataset(fresh_draws_dir="responses/")

# CJE automatically clusters by prompt for valid inference
if result.metadata.get("prompts_aligned"):
    print("✓ Paired comparison - using cluster-robust SEs")
```

**Why it matters:** Paired designs have correlated outcomes across policies (same prompt evaluated by multiple models). Standard SEs would understate uncertainty. CJE automatically accounts for this by clustering by `prompt_id`.

## Beyond Direct Mode

CJE also supports **IPS** (counterfactual inference from logs) and **DR** (doubly robust with fresh draws). These require log probabilities from your models.

```python
# IPS: Estimate "what if we deployed policy X?" from existing logs
result = analyze_dataset(logged_data_path="logs.jsonl")

# DR: Combine logged data + fresh draws for maximum accuracy
result = analyze_dataset(
    logged_data_path="logs.jsonl",
    fresh_draws_dir="responses/"
)
```

**For IPS/DR data formats and API details:** Run `help(analyze_dataset)` or see [`cje/interface/`](cje/interface/) module docs.

## Visualization

CJE provides diagnostic plots for understanding and validating results:

```python
from cje import analyze_dataset, plot_policy_estimates

# Run analysis
result = analyze_dataset(fresh_draws_dir="responses/")

# Quick plot with convenience method
result.plot_estimates(save_path="estimates.png")

# Or use visualization functions directly for more control
plot_policy_estimates(
    estimates={"policy_a": 0.75, "policy_b": 0.68},
    standard_errors={"policy_a": 0.02, "policy_b": 0.03},
    oracle_values={"policy_a": 0.74, "policy_b": 0.69}  # Optional
)
```

**Available visualizations:**
- `plot_policy_estimates` - Forest plots with confidence intervals
- `plot_calibration_comparison` - Judge→oracle calibration curves
- `plot_weight_dashboard_summary` - Weight diagnostics for IPS/DR
- `plot_weight_dashboard_detailed` - Per-policy weight analysis
- `plot_dr_dashboard` - Doubly robust diagnostics

**Jupyter notebooks:** Results automatically display as formatted tables when evaluated in a cell.

See [`cje/visualization/README.md`](cje/visualization/README.md) for complete guide.

## Documentation

📚 **Getting Started**
- [Interactive Demo](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_arena_demo.ipynb) - Try in your browser
- [Examples](examples/) - Working code samples

🔧 **For Engineers**
- [Calibration Methods](cje/calibration/README.md) - AutoCal-R, isotonic regression, two-stage fallback
- [Diagnostics System](cje/diagnostics/README.md) - Uncertainty quantification, OUA, transportability tests
- [Estimators](cje/estimators/README.md) - Direct, IPS, DR implementations
- [Interface/API](cje/interface/README.md) - `analyze_dataset` implementation and mode selection

📖 **Theory**
- [Playbook](docs/playbook/) - Mathematical foundations, assumptions, diagnostics

## Development

```bash
git clone https://github.com/cimo-labs/cje.git
cd cje
poetry install
make test
```

## Support

- 🐛 [Issues](https://github.com/cimo-labs/cje/issues)
- 💬 [Discussions](https://github.com/cimo-labs/cje/discussions)

## License

MIT - See [LICENSE](LICENSE) for details.
