<div align="left">
  <img src="images/CJE_logo.jpg" alt="CJE Logo" width="250">
</div>

# CJE — Causal Judge Evaluation

**LLM-judge scores are cheap, plentiful, and miscalibrated. In our benchmark, naive 95% confidence intervals built on raw judge scores contained the true value 0% of the time.** CJE calibrates your judge against a small slice of ground truth (5–25% of samples), evaluates your policies at scale, and reports uncertainty you can defend — including telling you when *not* to trust the result.

[![arXiv](https://img.shields.io/badge/arXiv-2512.11150-b31b1b.svg)](https://arxiv.org/abs/2512.11150)
[![Dataset](https://img.shields.io/badge/HF-Dataset-yellow)](https://huggingface.co/datasets/elandy/cje-chatbot-arena)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_core_demo.ipynb)
[![Docs](https://img.shields.io/badge/docs-cimolabs.com-blue)](https://cimolabs.com/cje)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.13-blue)](https://www.python.org/downloads/)
[![Tests](https://github.com/cimo-labs/cje/actions/workflows/ci.yml/badge.svg)](https://github.com/cimo-labs/cje/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/cje-eval?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/cje-eval)

## 60 seconds

```bash
pip install cje-eval
```

Generate responses from each candidate policy on a shared prompt set, judge everything, and attach ground-truth labels (`oracle_label`) to the slice you can afford — human raters, expert review, a downstream KPI. Any bounded scale works (0–1, 0–100, Likert). CJE needs at least 10 labeled rows pooled across policies:

```python
from cje import analyze_dataset

results = analyze_dataset(
    fresh_draws_data={
        "gpt-4o": [
            {"prompt_id": "q01", "judge_score": 0.85, "oracle_label": 0.90},
            {"prompt_id": "q02", "judge_score": 0.72, "oracle_label": 0.70},
            {"prompt_id": "q03", "judge_score": 0.91, "oracle_label": 0.88},
            {"prompt_id": "q04", "judge_score": 0.64, "oracle_label": 0.55},
            {"prompt_id": "q05", "judge_score": 0.77, "oracle_label": 0.74},
            {"prompt_id": "q06", "judge_score": 0.88, "oracle_label": 0.92},
            {"prompt_id": "q07", "judge_score": 0.68},
            {"prompt_id": "q08", "judge_score": 0.79},
        ],
        "claude-sonnet": [
            {"prompt_id": "q01", "judge_score": 0.78, "oracle_label": 0.82},
            {"prompt_id": "q02", "judge_score": 0.81, "oracle_label": 0.79},
            {"prompt_id": "q03", "judge_score": 0.86, "oracle_label": 0.84},
            {"prompt_id": "q04", "judge_score": 0.70, "oracle_label": 0.66},
            {"prompt_id": "q05", "judge_score": 0.74, "oracle_label": 0.71},
            {"prompt_id": "q06", "judge_score": 0.93, "oracle_label": 0.90},
            {"prompt_id": "q07", "judge_score": 0.75},
            {"prompt_id": "q08", "judge_score": 0.83},
        ],
    }
)

for policy, estimate, (lo, hi) in zip(
    results.metadata["target_policies"], results.estimates, results.ci()
):
    print(f"{policy:15s} {estimate:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
```

```text
claude-sonnet   0.773  95% CI [0.686, 0.884]
gpt-4o          0.745  95% CI [0.595, 0.909]
```

**And when the data can't support an answer, CJE says so** instead of handing you a confident number. Here a candidate policy's judge scores land mostly *outside* the range where the calibrator saw oracle labels — the run emits the paper's coverage badge and refuses level claims for that policy:

```text
REFUSE-LEVEL for policy 'candidate': 88.3% of fresh-draw judge scores fall
outside the oracle calibration range [0.161, 0.595]. Do not ship level
(absolute) claims for this policy; rankings may stand. Collect oracle labels
covering the missing score range.
```

The `cje analyze` CLI takes the same gates into account before crowning a winner:

```text
⚠️ Best by point estimate: candidate (UNRELIABLE — see diagnostics)
🏆 Best reliable policy: baseline
```

→ [Runnable Colab with real data](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_core_demo.ipynb) · [Full docs](https://cimolabs.com/cje)

## Is CJE the right tool?

| Your situation | Use |
|---|---|
| Rank/compare policies using an LLM judge, with some ground-truth labels | **CJE** |
| One dataset, labels sampled from it, want a CI on its mean | PPI works; CJE's `calibrated_mean_ci` is the same primitive and adds the transport audit + coverage badge |
| Evaluate **many** policies without labeling under each — and know when calibration reuse breaks | **CJE** (the transport audit is the point) |
| Predict how a *specific response* will score | Not CJE — per-item prediction (conformal methods) |
| Off-policy estimates from logs only (importance weighting / doubly robust) | `pip install "cje-eval==0.3.*"` — the frozen OPE line; 0.4.x is Direct-mode only (see [Notes on 0.4.0](#notes-on-040)) |

## How it works

1. **Calibrate**: learn the judge → oracle mapping on the labeled slice (isotonic, two-stage when needed; mean-preserving by construction; cross-fitted).
2. **Evaluate**: score every policy's fresh responses through the calibrated judge and compare policies on the same prompts.
3. **Audit & refuse**: a transport audit per policy (does the calibration still hold on this policy's outputs?) and a coverage badge for level claims (were there oracle labels where this policy's scores live?). Failing gates change the output — they are not footnotes.

Confidence intervals account for the judge being *learned from a finite label budget* (calibration-aware inference), not just sampling noise.

<div align="center">
  <img src="images/forest_plot_n1000_oracle25.png" alt="CJE forest plot showing calibrated policy estimates with confidence intervals" width="80%">
  <br><em>Calibrated estimates with 95% CIs (valid under the calibration and transport checks CJE runs by default)</em>
</div>

## Validation on real ground truth

- **HealthBench (physician labels, n=29,511)**: two LLM judges were overconfident by 24.5 and 13.0 points and disagreed with each other by up to 73 points on specific criteria categories. Calibrated on 5% physician labels (~1,400 records), both converged to the physician ground truth. [Read the full audit →](https://cimolabs.com/blog/healthbench-judge-audit)
- **Chatbot Arena (4,961 prompts, 5 policies)**: 99% pairwise ranking accuracy at a 5% oracle fraction — 14× cheaper than labeling everything, with ~95% CI coverage vs 0% for naive judge-score CIs. An adversarial policy that fools the judge is correctly flagged by the transport audit. [Paper →](https://arxiv.org/abs/2512.11150)

## The array API

`calibrated_mean_ci` is the library's bottom layer — a ppi_py-style primitive that takes plain numpy arrays and returns a calibrated mean with an honest CI. Use it when you have one sample of judge scores and a partial oracle slice; use `analyze_dataset` for multi-policy comparisons.

```python
import numpy as np
from cje import calibrated_mean_ci

rng = np.random.default_rng(0)
scores = rng.uniform(size=400)                      # judge scores for every sample
labels = np.full(400, np.nan)                       # NaN = unlabeled
labeled = rng.choice(400, size=100, replace=False)  # oracle slice (25%)
labels[labeled] = np.clip(scores[labeled] + rng.normal(0, 0.1, size=100), 0, 1)

result = calibrated_mean_ci(scores, labels)
print(result.summary())
```

```text
Calibrated mean: 0.5316 (SE 0.0175, CI [0.4965, 0.5649], n=400, n_oracle=100, bootstrap)
```

`result.calibrator` is reusable: `transport_audit(probe_scores, probe_labels, result.calibrator)` checks whether the calibration still holds on a new slice. `result.diagnostics["boundary_card"]` carries the coverage badge.

## Documentation

| Resource | Description |
|----------|-------------|
| **[Interactive Tutorial](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_core_demo.ipynb)** | Walk through a complete example in Colab — no setup required |
| **[CJE in 3 Minutes](https://youtu.be/VbSYrby8iaQ)** | Video: why raw judge scores mislead and how CJE fixes it |
| **[Technical Walkthrough](https://youtu.be/r0dinGsPuqY)** | Video: calibration, evaluation, and transport auditing pipeline |
| **[Operational Playbook](PLAYBOOK.md)** | End-to-end runbook: audits, drift correction, label budgeting |
| **[Planning Notebook](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_planning.ipynb)** | Optimize your evaluation budget with pilot data |
| **[Full Docs](https://cimolabs.com/cje)** | Installation, assumptions, API reference, research notes |

**Bridges:** Already running evals in [Promptfoo, TruLens, LangSmith, or OpenCompass](scripts/cje_bridges/README.md)? Convert those outputs into CJE format with one command.

**Module deep dives:** [Calibration](cje/calibration/README.md) · [Diagnostics](cje/diagnostics/README.md) · [Estimators](cje/estimators/README.md) · [Interface/API](cje/interface/README.md) · [Data formats](cje/data/README.md)

## Notes on 0.4.0

0.4.0 is a **breaking release: CJE is now Direct-mode only.** The off-policy machinery — importance-sampling and doubly-robust estimators (`calibrated-ips`, `dr-cpo`, `mrdr`, `tmle`, `stacked-dr`), teacher forcing, SIMCal weight stabilization, and the overlap diagnostics — has been removed. Our own paper's results drove the cut: for realistic LLM policy pairs, importance weighting failed even when ESS looked healthy (target-typicality coverage 0.19–0.49, far below the 0.70 gate), and the best DR stack merely matched Direct mode's accuracy at ~12× the compute. Direct mode — fresh draws, calibrated judge, audits — is what the evidence supports, so it is now the whole product.

- **Need IPS/DR from logged propensities?** Pin the frozen OPE line: `pip install "cje-eval==0.3.*"` (maintained on the `0.3.x` branch; docs at the `v0.3.0` tag).
- **Have old logged data with `judge_score` + `oracle_label`?** It still works as the calibration source: `analyze_dataset(fresh_draws_dir=..., calibration_data_path="logged.jsonl")`.
- Removed entry points raise migration errors that say exactly this.

Full details in the [CHANGELOG](CHANGELOG.md).

## Development

```bash
git clone https://github.com/cimo-labs/cje.git
cd cje && poetry install && make test
```

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
