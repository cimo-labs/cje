<div align="left">
  <img src="CJE_logo.jpg" alt="CJE Logo" width="250">
</div>

# CJE - Causal Judge Evaluation

**Your LLM judge scores are lying. CJE calibrates them to what actually matters.**

[![Docs](https://img.shields.io/badge/docs-cimolabs.com-blue)](https://cimolabs.com/cje)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-green)](https://github.com/cimo-labs/cje/actions)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/cje-eval?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/cje-eval)

<div align="center">
  <img src="cje_pipeline.jpg" alt="CJE Pipeline" width="85%">
</div>

---

## Quick Start

```bash
pip install cje-eval
```

```python
from cje import analyze_dataset

# Point to your response files (one JSONL per policy)
results = analyze_dataset(fresh_draws_dir="data/responses/")

# Get calibrated estimates with valid confidence intervals
results.plot_estimates(save_path="ranking.png")
```

**Data format** (one JSONL file per policy):
```json
{"prompt_id": "1", "judge_score": 0.85, "oracle_label": 0.9}
{"prompt_id": "2", "judge_score": 0.72}
```

Only 5-25% of samples need oracle labels. CJE learns the judge→oracle mapping and applies it everywhere.

---

## Why You Need This

Raw LLM judge scores suffer from systematic biases that make your metrics unreliable:

- **Preference inversion**: Higher scores often predict *lower* real-world quality
- **Invalid confidence intervals**: Standard error bars yield 0% coverage
- **Scale arbitrariness**: Is "4.2" actually better than "4.0"?

CJE fixes this by treating your judge as a sensor that must be calibrated against ground truth.

[**Read the full explanation →**](https://cimolabs.com/blog/metrics-lying)

---

## The Proof

We benchmarked 14 estimators on 5,000 real Chatbot Arena prompts using GPT-5 as oracle:

<div align="center">
  <img src="forest_plot_n1000_oracle25.png" alt="CJE Calibration Accuracy" width="80%">
</div>

| Method | Result |
|:-------|:-------|
| Raw Judges | **0% CI coverage** — error bars were mathematical lies |
| CJE (Direct + Two-Stage) | **94% ranking accuracy**, valid 95% CIs, **16x cost reduction** |

[**Read the full Arena Experiment →**](https://cimolabs.com/blog/arena-experiment)

---

## Evaluation Modes

CJE automatically selects the right estimator based on your data:

| Mode | Data Required | Use Case |
|:-----|:--------------|:---------|
| **Direct** | Fresh responses | A/B testing — compare policies on the same prompts |
| **IPS** | Logs + logprobs | Historical analysis — reweight old logs without new inference |
| **DR** | Logs + fresh draws | Production deploy — doubly robust for minimum variance |

---

## Verifying Calibration Transfers

Before trusting calibration on a new policy, verify it transfers:

```python
from cje.diagnostics import audit_transportability

# Test with a small probe (50 oracle labels)
probe = [{"judge_score": 0.8, "oracle_label": 0.75}, ...]
diag = audit_transportability(calibrator, probe, group_label="new_policy")

print(diag.summary())
# Transport: PASS | N=50 | δ̂: +0.012 (CI: [-0.03, +0.05])

diag.plot()  # Visualize decile-level residuals
```

<div align="center">
  <img src="transportability_audit.png" alt="Transportability Audit" width="70%">
</div>

---

## Tutorials

- [**Core Demo**](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_core_demo.ipynb) — Full workflow from first principles
- [**Advanced (IPS/DR)**](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_advanced.ipynb) — Off-policy evaluation with logged data

---

## Documentation

**Technical Guides**
- [Calibration Methods](cje/calibration/README.md) — AutoCal-R, isotonic regression, two-stage
- [Diagnostics System](cje/diagnostics/README.md) — Uncertainty quantification, transportability
- [Estimators](cje/estimators/README.md) — Direct, IPS, DR implementations
- [Interface/API](cje/interface/README.md) — `analyze_dataset` implementation

**Examples & Data**
- [Examples Folder](examples/) — Working code samples
- [Arena Sample Data](examples/arena_sample/README.md) — Real-world test data

---

## Development

```bash
git clone https://github.com/cimo-labs/cje.git
cd cje && poetry install && make test
```

## Support

- [Issues](https://github.com/cimo-labs/cje/issues)
- [Discussions](https://github.com/cimo-labs/cje/discussions)

## License

MIT — See [LICENSE](LICENSE) for details.
