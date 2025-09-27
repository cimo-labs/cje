<div align="left">
  <img src="CJE_logo_v3.jpg" alt="CJE Logo" width="250">
</div>

# CJE - Causal Judge Evaluation

[![Docs](https://img.shields.io/badge/docs-cimo--labs.com-blue)](https://cimo-labs.com/cje)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue)](https://www.python.org/downloads/)
[![codecov](https://codecov.io/gh/fondutech/causal-judge-evaluation/branch/main/graph/badge.svg)](https://codecov.io/gh/fondutech/causal-judge-evaluation)
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
