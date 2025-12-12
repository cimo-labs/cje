# CJE Examples

Two interactive notebooks to get started with CJE:

## 🎯 Core Demo: Understanding CJE from First Principles

**Learn why calibration matters and how to detect drift:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_core_demo.ipynb)

**The complete CJE workflow** with [`cje_core_demo.ipynb`](cje_core_demo.ipynb):
- **The Problem**: See why naive S (judge) scores don't match Y (oracle) outcomes
- **Calibration**: Build an isotonic S→Y mapping from scratch
- **Transportability**: Test if calibration transfers to new policies
- **Residual Analysis**: Find samples where calibration fails (and why)
- **Drift Detection**: Simulate and detect temporal drift in production
- **Library Usage**: Use `cje-eval` for production with forest plots and statistical tests

**Perfect for**: Understanding the concepts, building intuition, production monitoring

---

## 📚 Advanced Tutorial (Off-Policy Evaluation)

**Want counterfactual deployment estimates?**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_advanced.ipynb)

**Learn IPS and DR modes** with [`cje_advanced.ipynb`](cje_advanced.ipynb):
- **IPS Mode**: Reuse logged data with importance sampling
- **DR Mode**: Doubly robust estimation (most accurate)
- **Diagnostics**: ESS, overlap, weight analysis
- **Mode Comparison**: When to use Direct vs IPS vs DR

**Perfect for**: Production deployment decisions, logged data reuse, maximum accuracy

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
- **4 LLM policies** (base, clone, parallel_universe_prompt, unhelpful)
- **Judge scores** (GPT-4.1-nano) for all responses
- **Oracle labels** (GPT-5) for calibration:
  - Logged data: 48% oracle coverage across all policies
  - Fresh draws: 48% oracle coverage for base policy only
- **Fresh draws** for all policies for doubly robust estimation

**Why oracle labels in base fresh draws?** This enables Direct mode examples to demonstrate AutoCal-R. When running Direct mode with only fresh draws, the base policy serves as the source of oracle labels for learning the calibration function that maps judge scores → oracle labels.

See `arena_sample/README.md` for details.

## What is CJE?

**Causal Judge Evaluation (CJE)** transforms LLM-as-judge scores into calibrated policy estimates:

- **AutoCal-R**: Calibrates cheap judge scores to expensive oracle labels (only 5-10% oracle coverage needed!)
- **Three Modes**: Direct (on-policy), IPS (off-policy), DR (doubly robust)
- **Valid Inference**: Confidence intervals account for all uncertainty (sampling, calibration, Monte Carlo)

**Key insight**: Raw judge scores are correlational. CJE makes them causal through calibration.

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
jupyter notebook examples/cje_core_demo.ipynb
```
