<div align="left">
  <img src="CJE_logo.jpg" alt="CJE Logo" width="250">
</div>

# CJE - Causal Judge Evaluation

**Your LLM judge scores are lying. CJE calibrates them to what actually matters.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_core_demo.ipynb)
[![Docs](https://img.shields.io/badge/docs-cimolabs.com-blue)](https://cimolabs.com/cje)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-green)](https://github.com/cimo-labs/cje/actions)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/cje-eval?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/cje-eval)

We ran 16,000+ tests on Chatbot Arena data. **Without calibration, 95% confidence intervals captured the true value 0% of the time.** With CJE: 99% ranking accuracy using just 5% oracle labels, at 14× lower cost.

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
results.plot_estimates(
    policy_labels={"prompt_v1": "Conversational tone", ...},
    save_path="ranking.png"
)
```

**Data format** (one JSONL file per policy):
```json
{"prompt_id": "1", "judge_score": 0.85, "oracle_label": 0.9}
{"prompt_id": "2", "judge_score": 0.72}
```

Only 5-25% of samples need oracle labels. CJE learns the judge→oracle mapping and applies it everywhere.

---

## Why You Need This

Uncalibrated LLM-as-judge evaluation has two systematic failure modes:

| Failure Mode | What Happens | Evidence |
|:-------------|:-------------|:---------|
| **Invalid confidence intervals** | Your error bars don't work | "95% confident" was actually 0% accurate |
| **Hidden scale distortion** | Judge scores ≠ oracle scores | Calibration cut prediction error by 72% |

With 0% CI coverage, you can't trust any A/B test conclusion. Rankings improve too (91% → 99%), but the uncertainty problem is universal.

**CJE fixes both** by treating your judge as a sensor that must be calibrated against ground truth, then propagating calibration uncertainty into valid confidence intervals.

[**Read the full explanation →**](https://cimolabs.com/blog/metrics-lying)

---

## The Results

We tested on 5,000 Chatbot Arena prompts with GPT-5 as the oracle (ground truth) and GPT-4.1-nano as the cheap judge:

| Without CJE | With CJE |
|:------------|:---------|
| Rankings correct 91% of the time | Rankings correct 99% of the time |
| Error bars contain truth 0% of the time | Error bars contain truth 87% of the time |
| Need 100% oracle labels | Need only 5% oracle labels |
| Full labeling cost | **14× cheaper** |

Label ~250 samples with your oracle (human raters, downstream KPIs, expensive model). CJE learns the judge→oracle mapping and applies it to everything else.

**Already using an expensive model for evals?** Switch to a 10-30× cheaper judge + CJE calibration. Same accuracy, fraction of the inference cost.

<div align="center">
  <img src="forest_plot_n1000_oracle25.png" alt="CJE Output Example" width="80%">
  <br><em>Example output: comparing prompt variants with calibrated confidence intervals</em>
</div>

[**Read the full Arena Experiment →**](https://www.cimolabs.com/research/arena-experiment) ・ [**Paper (Zenodo)**](https://zenodo.org/records/17903629)

---

## Monitoring Calibration Over Time

Calibration can drift. Periodically verify it still holds with a small probe:

```python
from cje import analyze_dataset
from cje.diagnostics import audit_transportability

# results.calibrator is automatically fitted during analysis
results = analyze_dataset(fresh_draws_dir="responses/")

# Check if calibration still works on this week's data (50+ oracle labels)
diag = audit_transportability(results.calibrator, this_week_samples)
print(diag.summary())
# Status: PASS | Samples: 48 | Mean error: +0.007 (CI: -0.05 to +0.06)
```

<div align="center">
  <img src="transportability_audit.png" alt="Temporal Monitoring" width="70%">
</div>

PASS means your calibration is still valid. FAIL means something changed — investigate or recalibrate.

---

## Try It Now

**[Open the interactive tutorial in Google Colab →](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_core_demo.ipynb)**

Walk through a complete example: compare prompt variants, check if calibration transfers, inspect what's fooling the judge, and monitor drift over time. No setup required.

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
