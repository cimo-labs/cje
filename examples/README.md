# CJE Examples

Focused examples showing CJE's three analysis modes. Each example is ~20 lines and teaches one concept.

## Running Examples

```bash
# From the repo root
poetry run python examples/01_understanding_modes.py
poetry run python examples/02_minimal.py
poetry run python examples/03_with_fresh_draws.py
poetry run python examples/04_comparing_policies.py
poetry run python examples/05_checking_reliability.py
```

## What Each Example Shows

### 1. Understanding Modes (`01_understanding_modes.py`)
Comprehensive overview of CJE's three analysis modes: Direct, IPS, and DR.

**Key concept:** When to use each mode and what they estimate.

### 2. Minimal Usage (`02_minimal.py`)
The absolute simplest CJE workflow - load data and get estimates.

**Key concept:** Getting started with CJE in ~10 lines of code.

### 3. With Fresh Draws (`03_with_fresh_draws.py`)
Shows how to use doubly-robust estimation with fresh draws from target policies.

**Key concept:** Fresh draws enable more accurate estimates via DR methods.

### 4. Comparing Policies (`04_comparing_policies.py`)
Find the best policy and compare against a baseline using proper statistical inference.

**Key concept:** Policy selection and comparison with significance testing.

### 5. Checking Reliability (`05_checking_reliability.py`)
Use diagnostics to assess whether your estimates are trustworthy.

**Key concept:** Effective Sample Size (ESS) and reliability assessment for IPS mode.

## Data

All examples use the arena sample dataset included in `cje/tests/data/arena_sample/`:
- 100 samples from Arena 10K evaluation
- 4 target policies: clone, premium, parallel_universe_prompt, unhelpful
- Judge scores and oracle labels for calibration
- Fresh draws for doubly-robust estimation

## Interactive Demo

**🚀 Try CJE in your browser:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_arena_demo.ipynb)

The Colab notebook (`cje_arena_demo.ipynb`) provides an interactive walkthrough of CJE's three modes using the Arena sample data. No installation required - runs entirely in your browser!

## Next Steps

- **Main README:** [README.md](../README.md) for Quick Start and overview
- **API details:** Run `help(analyze_dataset)` or see [cje/interface/README.md](../cje/interface/README.md)
- **Production example:** [cje/experiments/arena_10k_simplified/](../cje/experiments/arena_10k_simplified/) for complete pipeline
