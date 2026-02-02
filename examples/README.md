# CJE Examples

## Notebooks

### Core Tutorial
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_core_demo.ipynb)

**Start here:** [`cje_core_demo.ipynb`](cje_core_demo.ipynb) — Compare policies, check calibration transfers, monitor drift.

1. **Compare Policies** — One-line analysis with `analyze_dataset()`
2. **Check If Calibration Transfers** — Test on held-out data with `audit_transportability()`
3. **Inspect What's Fooling the Judge** — Dig into worst residuals with `compute_residuals()`
4. **Monitor Calibration Over Time** — Detect drift before it breaks your metrics

No setup required — runs entirely in Google Colab with real Chatbot Arena data.

### Budget Planning
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_planning.ipynb)

**Optimize costs:** [`cje_planning.ipynb`](cje_planning.ipynb) — How many samples? How many oracle labels? What's the minimum detectable effect?

1. **Fit Variance Model** — Learn σ²_eval and σ²_cal from pilot data
2. **Budget-Constrained Planning** — "I have $X, what MDE can I detect?"
3. **MDE-Constrained Planning** — "I need to detect X%, what's the cost?"
4. **Visualize Tradeoffs** — Interactive dashboard for budget vs precision

### Advanced: Off-Policy Evaluation

For IPS/DR modes (reusing logged data without new inference), see [`cje_advanced.ipynb`](cje_advanced.ipynb).

## Dataset

Examples use a curated Chatbot Arena-derived dataset:
- HF dataset: https://huggingface.co/datasets/elandy/cje-chatbot-arena
- The repo also includes a ready-to-run sample under `examples/arena_sample/`.

See `arena_sample/README.md` for details.
