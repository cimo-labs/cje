# CJE Examples

## Interactive Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_core_demo.ipynb)

**Start here:** [`cje_core_demo.ipynb`](cje_core_demo.ipynb) walks through the complete workflow:

1. **The Problem** — Why raw judge scores lie (and how to see it)
2. **Calibration** — Build a judge→oracle mapping from scratch
3. **Compare Prompts** — Rank prompt variants with valid confidence intervals
4. **Monitor Drift** — Detect when calibration breaks down over time

No setup required — runs entirely in Google Colab with real Chatbot Arena data.

## Python Quickstart

**Want a copy-paste script?** Use `quickstart.py`:

```bash
# From the repo root
poetry run python examples/quickstart.py
```

15-line script showing the most common workflow:
- Load Arena sample data (logged responses + fresh draws)
- Run doubly robust analysis (most accurate mode)
- Get policy estimates with 95% confidence intervals

## Dataset

Examples use real data from [LMSYS Chatbot Arena](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations):
- 1000 prompts, multiple response variants
- Judge scores (GPT-4.1-nano) + oracle labels (GPT-5) for calibration

See `arena_sample/README.md` for details.

## Advanced: Off-Policy Evaluation

For IPS/DR modes (reusing logged data without new inference), see [`cje_advanced.ipynb`](cje_advanced.ipynb).
