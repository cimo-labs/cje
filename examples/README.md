# CJE Examples

Two entry points to get started with CJE:

## 🚀 Interactive Tutorial (Recommended)

**Try CJE in your browser - no installation required:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_tutorial.ipynb)

The Colab notebook (`cje_tutorial.ipynb`) provides a complete walkthrough:
- **Setup**: Install CJE and download Arena sample data
- **Inspect Data**: Understand the dataset structure
- **Three Modes**: Direct, IPS, and DR estimation
- **Policy Selection**: Statistical comparison with confidence intervals
- **Diagnostics**: Check reliability and transportability

**What you'll learn:**
- When to use each mode (Direct vs IPS vs DR)
- How AutoCal-R calibrates judge scores to oracle labels
- How SIMCal stabilizes importance weights
- How to interpret diagnostics (ESS, transportability tests)
- How to select the best policy with proper statistical inference

## 🐍 Python Quickstart

**Want a copy-paste script?** Use `quickstart.py`:

```bash
# From the repo root
poetry run python examples/quickstart.py
```

15-line script showing the most common workflow:
- Load Arena sample data (logged responses + fresh draws)
- Run doubly robust analysis (most accurate mode)
- Get policy estimates with 95% confidence intervals

## Dataset: Arena Sample

Both examples use real data from the [LMSYS Chatbot Arena](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations):

- **1000 prompts** from Arena conversations
- **3 LLM policies** (clone, parallel_universe_prompt, unhelpful)
- **Oracle labels** (GPT-5) at 50% coverage for calibration
- **Judge scores** (GPT-4.1-nano) for all responses
- **Fresh draws** for doubly robust estimation

See `arena_sample/README.md` for details.

## What is CJE?

**Causal Judge Evaluation (CJE)** transforms LLM-as-judge scores into causally interpretable policy value estimates:

1. **AutoCal-R**: Calibrates cheap judge scores (GPT-4.1-nano) to expensive oracle labels (GPT-5) using isotonic regression
2. **SIMCal-W**: Stabilizes importance weights via surrogate-indexed monotone projection
3. **Three Modes**: Direct (on-policy), IPS (off-policy), DR (doubly robust)
4. **Valid Inference**: Confidence intervals account for sampling, calibration, and Monte Carlo uncertainty

**Key insight**: Judge scores are correlational (E[judge | policy]). CJE makes them causal (E[oracle | policy]).

## Next Steps

- **Documentation**: See [cje/interface/README.md](../cje/interface/README.md) for API details
- **Main README**: See [README.md](../README.md) for installation and overview

## Running Locally

```bash
# Install CJE
pip install cje-eval

# Run quickstart
python examples/quickstart.py

# Or open the notebook in Jupyter
jupyter notebook examples/cje_tutorial.ipynb
```
