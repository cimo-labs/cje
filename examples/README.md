# CJE Examples

Focused examples showing common CJE workflows. Each example is ~20 lines and teaches one concept.

## Running Examples

```bash
# From the repo root
poetry run python examples/01_minimal.py
poetry run python examples/02_with_fresh_draws.py
poetry run python examples/03_comparing_policies.py
poetry run python examples/04_checking_reliability.py
```

## What Each Example Shows

### 1. Minimal Usage (`01_minimal.py`)
The absolute simplest CJE workflow - load data and get estimates.

**Key concept:** Getting started with CJE in ~10 lines of code.

### 2. With Fresh Draws (`02_with_fresh_draws.py`)
Shows how to use doubly-robust estimation with fresh draws from target policies.

**Key concept:** Fresh draws enable more robust estimates via DR methods.

### 3. Comparing Policies (`03_comparing_policies.py`)
Find the best policy and compare against a baseline.

**Key concept:** Policy selection and comparison.

### 4. Checking Reliability (`04_checking_reliability.py`)
Use diagnostics to assess whether your estimates are trustworthy.

**Key concept:** Effective Sample Size (ESS) and reliability assessment.

## Data

All examples use the arena sample dataset included in `cje/tests/data/arena_sample/`:
- 100 samples from Arena 10K evaluation
- 4 target policies: clone, premium, parallel_universe_prompt, unhelpful
- Judge scores and oracle labels for calibration
- Fresh draws for doubly-robust estimation

## Next Steps

- **Quick tutorial:** See [QUICKSTART.md](../QUICKSTART.md) for step-by-step guide
- **Full documentation:** [README_ENGINEERING.md](../README_ENGINEERING.md) for detailed API
- **Production example:** [cje/experiments/arena_10k_simplified/](../cje/experiments/arena_10k_simplified/) for complete pipeline
