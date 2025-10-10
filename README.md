<div align="left">
  <img src="CJE_logo.jpg" alt="CJE Logo" width="250">
</div>

# CJE - Causal Judge Evaluation

[![Docs](https://img.shields.io/badge/docs-cimolabs.com-blue)](https://cimolabs.com/cje)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-green)](https://github.com/cimo-labs/cje/actions)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Turn noisy LLM-judge scores into unbiased policy estimates with valid confidence intervals.**

CJE calibrates judge scores using a small oracle slice (5-10% coverage), then delivers statistically rigorous estimates. As simple as comparing responses, as powerful as A/B testing.

## How It Works

```
Judge scores + small oracle slice (5-10%) → Calibrate to oracle scale
                                          → Unbiased policy estimates
                                          → Valid 95% confidence intervals
```

**What you get:**
- **Unbiased estimates**: Judge scores mapped to oracle outcome scale (preserves mean, enforces "higher score → no worse outcome")
- **Valid CIs**: Account for both sampling uncertainty and calibration uncertainty
- **Small label budget**: 5-10% oracle coverage often sufficient

**Automatic handling:**
- Paired comparisons: Clusters by prompt when comparing policies on same eval set
- Oracle uncertainty: Jackknife procedure captures calibration risk

See [`cje/calibration/README.md`](cje/calibration/README.md#why-isotonic-regression-for-reward-calibration) for technical details.

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

See [Engineering Guide](README_ENGINEERING.md) for IPS/DR data formats and log probability computation.

## Documentation

📚 **Getting Started**
- [5-Minute Quickstart](QUICKSTART.md) - First analysis step-by-step
- [Interactive Demo](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_arena_demo.ipynb) - Try in your browser
- [Examples](examples/) - Working code samples

🔧 **For Engineers**
- [Engineering Guide](README_ENGINEERING.md) - Complete API reference, data formats, uncertainty quantification
- [Calibration Methods](cje/calibration/README.md) - AutoCal-R, isotonic regression, two-stage fallback
- [Diagnostics System](cje/diagnostics/README.md) - Uncertainty quantification, OUA, paired inference
- [Estimators](cje/estimators/README.md) - Direct, IPS, DR implementations

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

---
**Ready to start?** → [5-Minute Quickstart](QUICKSTART.md)
