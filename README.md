<div align="left">
  <img src="images/CJE_logo.jpg" alt="CJE Logo" width="250">
</div>

# CJE - Causal Judge Evaluation

**Your LLM judge scores are noisy and nebulous. CJE calibrates them to what actually matters.**

[![arXiv](https://img.shields.io/badge/arXiv-2512.11150-b31b1b.svg)](https://arxiv.org/abs/2512.11150)
[![Dataset](https://img.shields.io/badge/HF-Dataset-yellow)](https://huggingface.co/datasets/elandy/cje-chatbot-arena)
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

# Your evaluation data - one list per policy variant
results = analyze_dataset(
    fresh_draws_data={
        "prompt_v1": [
            {"prompt_id": "1", "judge_score": 0.85, "oracle_label": 0.9},
            {"prompt_id": "2", "judge_score": 0.72, "oracle_label": 0.7},
            {"prompt_id": "3", "judge_score": 0.68},  # oracle_label optional (5-25% needed)
        ],
        "prompt_v2": [
            {"prompt_id": "1", "judge_score": 0.78, "oracle_label": 0.82},
            {"prompt_id": "2", "judge_score": 0.81, "oracle_label": 0.79},
            {"prompt_id": "3", "judge_score": 0.75},
        ],
    }
)

# Or from files: analyze_dataset(fresh_draws_dir="responses/")

results.plot_estimates(save_path="ranking.png")
```

CJE learns the judge→oracle mapping from the labeled samples and applies it everywhere.

---

## Why You Need This

**LLM-as-judge gives you rankings. CJE gives you certainty.**

Without calibration, you know prompt A scored higher than B—but you don't know:
- Is the difference real or noise?
- How big is the improvement, actually?
- Have I tested enough samples?
- Will this hold next week?

CJE answers all of these. Label 5% of samples with your oracle (human raters, latest SOTA model or AI agent, downstream metric). CJE learns the calibration and applies it everywhere—giving you trustworthy magnitudes, valid confidence intervals, and drift detection.

**The result:** Make decisions faster, spend less on labeling, and defend your conclusions with real statistics.

[**Read the full explanation →**](https://cimolabs.com/blog/metrics-lying)

---

## The Results

We tested on 5,000 Chatbot Arena prompts with GPT-5 as the oracle (ground truth) and GPT-4.1-nano as the cheap judge:

**CJE achieves 99% ranking accuracy using only 5% oracle labels—matching full-oracle performance at 14× lower cost.**

Label ~250 samples with your oracle (human raters, downstream KPIs, expensive model). CJE learns the judge→oracle mapping and applies it to everything else. Without calibration, error bars contained the true value 0% of the time. With CJE: ~95%.

**Already using an expensive model for evals?** Switch to a 10-30× cheaper judge + CJE calibration. Same accuracy, fraction of the inference cost.

<div align="center">
  <img src="images/forest_plot_n1000_oracle25.png" alt="CJE Output Example" width="80%">
  <br><em>Example output: comparing prompt variants with calibrated confidence intervals</em>
</div>

[**Read the full Arena Experiment →**](https://www.cimolabs.com/research/arena-experiment)

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
  <img src="images/transportability_audit.png" alt="Temporal Monitoring" width="70%">
</div>

PASS means your calibration is still valid. FAIL means something changed — investigate or recalibrate.

---

## Try It Now

**[Open the interactive tutorial in Google Colab →](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_core_demo.ipynb)**

Walk through a complete example: compare prompt variants, check if calibration transfers, inspect what's fooling the judge, and monitor drift over time. No setup required.

---

## Documentation

**Video Walkthroughs**
- [CJE Technical Walkthrough](https://youtu.be/r0dinGsPuqY) — Pipeline deep dive: calibration, evaluation, and transport auditing
- [CJE in 3 Minutes](https://youtu.be/VbSYrby8iaQ) — Quick intro: why raw judge scores mislead and how CJE fixes it

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

## Citation

If you use CJE in your research, please cite:

```bibtex
@misc{landesberg2025causaljudgeevaluationcalibrated,
  title={Causal Judge Evaluation: Calibrated Surrogate Metrics for LLM Systems},
  author={Eddie Landesberg},
  year={2025},
  eprint={2512.11150},
  archivePrefix={arXiv},
  primaryClass={stat.ME},
  url={https://arxiv.org/abs/2512.11150},
}
```

## License

MIT — See [LICENSE](LICENSE) for details.
