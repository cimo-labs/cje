<div align="left">
  <img src="CJE_logo.jpg" alt="CJE Logo" width="250">
</div>

# CJE - Causal Judge Evaluation

[![Docs](https://img.shields.io/badge/docs-cimo--labs.com-blue)](https://cimo-labs.com/cje)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-green)](https://github.com/cimo-labs/cje/actions)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Evaluate LLM policies with statistical rigor - as simple as comparing responses, as powerful as A/B testing.**

CJE turns your LLM-judge evaluations into reliable estimates with confidence intervals. Compare policies head-to-head, or estimate counterfactual deployment value from logged data.

## Why CJE?

🎯 **Problem**: LLM-judge scores are noisy and biased
✅ **Solution**: Automatic calibration (AutoCal-R) learns judge→oracle mapping to debias scores and provide reliable estimates with confidence intervals

**Three modes, one interface:**
- **Direct mode**: Compare policies on an eval set (simplest - no logprobs needed)
- **IPS mode**: Estimate counterfactual value from logged data (reuse existing logs)
- **DR mode**: Combine both for maximum accuracy (doubly robust)

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

## 🚀 Try it Now - Interactive Demo

**No installation required!** Try CJE in your browser with real Arena data:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_arena_demo.ipynb)

The notebook demonstrates all three analysis modes (IPS, DR, Direct) with step-by-step explanations and diagnostics interpretation.

## Quick Start

**Simplest workflow - Direct mode (no logprobs needed):**

```python
from cje import analyze_dataset

# Compare policies on an eval set
result = analyze_dataset(fresh_draws_dir="responses/")

# Get estimates with confidence intervals
for i, policy in enumerate(result.metadata["target_policies"]):
    est = result.estimates[i]
    se = result.standard_errors[i]
    print(f"{policy}: {est:.3f} ± {1.96*se:.3f}")
```

Your `responses/` directory just needs JSONL files like:
```json
{"prompt_id": "eval_0", "policy": "model_a", "judge_score": 0.85}
{"prompt_id": "eval_0", "policy": "model_b", "judge_score": 0.72}
```

That's it! CJE handles the rest - auto-discovers policies, applies calibration if oracle labels are present, and returns reliable estimates.

**Advanced: Reuse logged data (IPS/DR modes)**

If you have production logs with log probabilities, CJE can estimate counterfactual deployment value:

```python
# IPS mode: Use logged data only
result = analyze_dataset(logged_data_path="logs.jsonl")

# DR mode: Combine logged data + fresh draws (most accurate)
result = analyze_dataset(
    logged_data_path="logs.jsonl",
    fresh_draws_dir="responses/"
)
```

See [Data Requirements](#data-requirements) for IPS/DR data format and [Teacher Forcing](#generating-log-probabilities) for computing logprobs.

## Three Analysis Modes

CJE automatically selects the best mode based on your data:

| Mode | Data | What it tells you | Best for |
|------|------|-------------------|----------|
| **Direct** | Responses from each policy | Which policy is best on this eval set? | Quick comparisons, A/B testing |
| **IPS** | Logged data with logprobs | What if we deployed policy X? (counterfactual) | Reusing existing logs, fast iteration |
| **DR** | Both logged + responses | Counterfactual value (most accurate) | High-stakes decisions, maximum accuracy |

**Automatic mode selection:**
- `fresh_draws_dir` only → Direct mode
- `logged_data_path` only → IPS mode (importance sampling)
- Both → DR mode (doubly robust)

**Direct mode** is the simplest - just provide responses from each policy with judge scores. No logprobs needed!

**IPS/DR modes** enable counterfactual inference: "What would happen if we deployed this policy?" This requires log probabilities from your models. See [Generating Log Probabilities](#generating-log-probabilities) below for Fireworks API integration.

## When to Use CJE

✅ **Use CJE when you need:**
- Statistical rigor (confidence intervals, p-values)
- Debiased judge scores (automatic calibration)
- Policy comparisons or counterfactual estimates
- To reuse logged data for new evaluations

❌ **Don't use CJE for:**
- Online learning (CJE is offline/batch)
- Real-time scoring (use raw judge for that)
- Very small samples (<100 examples)

## Data Requirements

Requirements depend on which mode you're using:

### For Direct Mode (fresh draws only):
```json
{
  "prompt_id": "arena_0",
  "prompt": "What is 2+2?",
  "response": "4",
  "policy": "clone",
  "judge_score": 0.85,                       // Required: judge evaluation
  "oracle_label": 0.86                       // Optional: ground truth (enables AutoCal-R)
}
```

**AutoCal-R**: If any fresh draws have `oracle_label`, Direct mode automatically applies AutoCal-R to learn judge→oracle calibration and uses calibrated rewards. More oracle labels = better calibration (5-10% is often sufficient).

### For IPS/DR Modes (logged data):
```json
{
  "prompt": "What is 2+2?",
  "response": "4",
  "base_policy_logprob": -14.7,              // Required: log P(response|prompt) for logging policy
  "target_policy_logprobs": {                // Required: same for policies to evaluate
    "clone": -14.7,
    "parallel_universe_prompt": -18.3,
    "unhelpful": -42.1
  },
  "judge_score": 0.85,                       // Required: judge evaluation
  "oracle_label": 0.86                       // Optional: ground truth (5-10% is enough for calibration)
}
```

**Key difference:** Direct mode doesn't need logprobs! Just responses from each policy with judge scores (and optionally oracle labels for AutoCal-R calibration).

**Working example:** See [`examples/arena_sample/`](examples/arena_sample/) for complete dataset examples with logged data and fresh draws.

### Generating Log Probabilities

**For IPS/DR modes, you need log probabilities.** CJE includes built-in Fireworks API integration:

```python
from cje.teacher_forcing import compute_teacher_forced_logprob

# Compute log P(response|prompt) for any model on Fireworks
result = compute_teacher_forced_logprob(
    prompt="What is 2+2?",
    response="4",
    model="accounts/fireworks/models/llama-v3p2-3b-instruct"
)
if result.status == "success":
    logprob = result.value  # e.g., -2.3
```

This handles chat templates, tokenization, and API calls automatically. Supports all Fireworks models.

**Don't have Fireworks access?** Direct mode doesn't need logprobs - just use `fresh_draws_dir` with judge scores.

See [`cje/teacher_forcing/README.md`](cje/teacher_forcing/README.md) for batch processing and advanced options.

## Advanced: Choosing an Estimator

**Most users:** Use `estimator="auto"` (the default). CJE auto-selects the best estimator for your mode.

**For researchers:** You can specify estimators explicitly:
- `direct`: On-policy comparison (no counterfactual inference)
- `calibrated-ips`: IPS with variance-reduced weights (SIMCal)
- `stacked-dr`: Ensemble of DR estimators (recommended for production)
- Individual DR variants: `dr-cpo`, `tmle`, `mrdr`, `oc-dr-cpo`, `tr-cpo-e`

See [`cje/estimators/README.md`](cje/estimators/README.md) for technical details on each estimator.

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
