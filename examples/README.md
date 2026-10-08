# CJE Examples

## Notebooks

### Core Tutorial
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_core_demo.ipynb)

**Start here:** [`cje_core_demo.ipynb`](cje_core_demo.ipynb) — Compare policies, check calibration transfers, monitor drift.

1. **Compare Policies** — Hold out audit labels, then analyze with `analyze_dataset()`
2. **Know When Not to Trust the Levels** — Per-policy coverage badge (`boundary_cards`); all four real policies come out OK, and a small simulated example shows what REFUSE-LEVEL looks like
3. **Check If Calibration Transfers** — Test on held-out data with `audit_transportability()`
4. **Inspect What the Calibration Gets Wrong** — Worst residuals (oracle minus calibrated) with `compute_residuals()`
5. **Monitor Calibration Over Time** — Detect drift before it breaks your metrics

No setup required — runs entirely in Google Colab on Chatbot Arena prompts and responses (GPT-5 labels stand in for human ratings).

### Budget Planning
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_planning.ipynb)

**Optimize costs:** [`cje_planning.ipynb`](cje_planning.ipynb) — How many samples? How many oracle labels? What's the minimum detectable effect?

1. **Quick Planning (no data)** — Judge quality (R²) + per-call costs → allocation and MDE with `simulate_planning()`
2. **Understanding the Tradeoffs** — How judge quality shifts the split between samples and oracle labels
3. **Budget vs MDE** — `plan_evaluation()` ("I have $X, what MDE?") and `plan_for_mde()` ("I need X%, what's the cost?"), plus a static planning dashboard (`plot_planning_dashboard`)
4. **Refine with Pilot Data (optional, off by default)** — `fit_variance_model()` learns σ²_eval and σ²_cal from labeled pilot data

### Off-Policy Evaluation (0.3.x line)

CJE is Direct-mode only; IPS/DR off-policy evaluation lives on the frozen 0.3.x line (`pip install "cje-eval==0.3.*"`). [`cje_advanced.ipynb`](cje_advanced.ipynb) is a stub pointing there; the full OPE notebook lives at the [v0.3.0 tag](https://github.com/cimo-labs/cje/blob/v0.3.0/examples/cje_advanced.ipynb).

## Audit labels and bias correction

[`audit_correction.py`](audit_correction.py) is a complete synthetic example of auditing a fixed calibration map and then using the same probability-sampled labels for residual correction. It prints the estimator routes, corrections, new intervals, original audit and complete human-label count. See the [guide](../guides/audit-correction.md) for label roles and sampling assumptions.

## Dataset

Examples use a curated Chatbot Arena-derived dataset:
- HF dataset: https://huggingface.co/datasets/elandy/cje-chatbot-arena
- The repo also includes a ready-to-run sample under `examples/arena_sample/`.

See `arena_sample/README.md` for details.
