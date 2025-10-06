<div align="left">
  <img src="CJE_logo.jpg" alt="CJE Logo" width="250">
</div>

# CJE - Causal Judge Evaluation

[![Docs](https://img.shields.io/badge/docs-cimo--labs.com-blue)](https://cimo-labs.com/cje)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-green)](https://github.com/cimo-labs/cje/actions)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**What if your AI evals looked like A/B tests with reliable confidence intervals and causal guarantees?**

CJE makes it possible. Get unbiased estimates of how your new model will perform before deployment, with the statistical rigor you'd expect from production experimentation.

## Why CJE?

🎯 **Problem**: Your LLM-judge scores are noisy, biased, and untrustworthy
✅ **Solution**: CJE uses **AutoCal-R** (Automatic Calibration for Rewards) and causal inference to debias them, giving you reliable estimates with confidence intervals without compromising on judge flexibility

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

CJE automatically selects the right mode based on your data:

```python
from cje import analyze_dataset

# Mode 1: Direct (simplest - just fresh draws)
result = analyze_dataset(fresh_draws_dir="responses/")
print(f"Policy value: {result.estimates[0]:.3f} ± {result.standard_errors[0]:.3f}")

# Mode 2: IPS (logged data with logprobs)
result = analyze_dataset(logged_data_path="logs.jsonl")  # Auto-selects IPS mode

# Mode 3: DR (logged data + fresh draws - most accurate)
result = analyze_dataset(
    logged_data_path="logs.jsonl",
    fresh_draws_dir="responses/"  # Auto-selects DR mode
)
```

CLI usage:
```bash
# Direct mode (fresh draws only)
python -m cje analyze --fresh-draws-dir responses/

# IPS mode (logged data) - auto-selects calibrated-ips
python -m cje analyze logs.jsonl

# DR mode (both) - auto-selects stacked-dr
python -m cje analyze logs.jsonl --fresh-draws-dir responses/
```

## Three Analysis Modes

CJE automatically selects the best mode based on your data. Mode selection follows a simple **4-rule system** based on logprob coverage (fraction of samples with complete logprobs) and data sources:

1. **fresh_draws + coverage ≥50%** → DR mode (doubly robust - most accurate)
2. **no fresh_draws + coverage ≥50%** → IPS mode (importance sampling - counterfactual)
3. **fresh_draws + coverage <50%** → Direct mode (on-policy comparison)
4. **no fresh_draws + coverage <50%** → Error (insufficient data)

See [Interface README](cje/interface/README.md#how-mode-detection-works) for details.

### 1. **Direct Mode** (Fresh draws only)
- **Use when:** You have responses from target policies, no logprobs needed
- **Estimand:** "Which policy performs best on this eval set?" (on-policy comparison)
- **Data needed:** Fresh responses with judge scores
- **Example:** Comparing 3 model variants on 1000 prompts

### 2. **IPS Mode** (Logged data with logprobs)
- **Use when:** You have logged data with importance weights, no fresh draws
- **Estimand:** "What would happen if we deployed this policy?" (counterfactual)
- **Data needed:** Logged responses with base/target logprobs
- **Example:** Evaluating a new model on production traffic logs

### 3. **DR Mode** (Both logged data and fresh draws)
- **Use when:** You want maximum accuracy and have both
- **Estimand:** Counterfactual deployment value (most accurate)
- **Data needed:** Both logged data and fresh draws
- **Default estimator:** `stacked-dr` (ensemble of DR-CPO, TMLE, MRDR, OC-DR-CPO, TR-CPO-E)
- **Example:** High-stakes A/B decision for model deployment

**Note:** The paper's "Calibrated DR" refers to DR mode, which defaults to `stacked-dr` - an optimal convex combination of multiple DR estimators for robustness and tighter intervals.

## When to Use CJE

✅ **Perfect for:**
- Comparing LLM policies before deployment
- Evaluating multiple model variants
- Reusing existing data for new evaluations
- High-stakes decisions needing confidence intervals

❌ **Not for:**
- Online learning (CJE is offline)
- Real-time scoring (CJE is batch)
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
  "oracle_label": 0.86                       // Optional: ground truth (50% coverage enables calibration)
}
```

**AutoCal-R**: If 50%+ of fresh draws have `oracle_label`, Direct mode automatically applies AutoCal-R to learn judge→oracle calibration and uses calibrated rewards.

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

CJE includes built-in **Fireworks API integration** for computing teacher-forced log probabilities:

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

This handles chat templates, tokenization, and API calls automatically. See `cje/teacher_forcing/` for details.

## Choosing an Estimator

**Most users should use `estimator="auto"`** (the default) - CJE will automatically select:
- **`direct`** when you only provide `fresh_draws_dir`
- **`calibrated-ips`** when you only provide `logged_data_path`
- **`stacked-dr`** when you provide both

**You can override automatic selection** by specifying an estimator explicitly:
```python
# Use IPS even with fresh draws available
analyze_dataset("logs.jsonl", fresh_draws_dir="responses/", estimator="calibrated-ips")

# Use Direct mode for on-policy comparison instead of DR
analyze_dataset("logs.jsonl", fresh_draws_dir="responses/", estimator="direct")
```

**Manual estimator options:**
- **`direct`**: On-policy comparison (Direct mode - no counterfactual inference)
- **`calibrated-ips`**: IPS with SIMCal weight stabilization (IPS mode default)
- **`stacked-dr`**: Ensemble of DR estimators (DR mode default - recommended for production)
  - Optimally combines: DR-CPO, TMLE, MRDR, OC-DR-CPO, TR-CPO-E
  - Provides robustness and tighter confidence intervals
- **Individual DR estimators**: `dr-cpo`, `tmle`, `mrdr`, `oc-dr-cpo`, `tr-cpo`, `tr-cpo-e` (for research)

**Paper terminology:** "Calibrated DR" in the paper = DR mode with `stacked-dr` estimator in the code.

See the [examples](examples/) for mode-specific workflows.

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
