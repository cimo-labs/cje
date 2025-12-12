# CJE Documentation

**Find the right evaluation mode in 30 seconds.**

## Installation

```bash
pip install cje-eval
```

---

## Which Mode Should I Use?

Answer these questions:

```
                    ┌─────────────────────────────┐
                    │  Do you have logged data    │
                    │  with logprobs?             │
                    └─────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              │ NO                              │ YES
              ▼                                 ▼
    ┌─────────────────┐               ┌─────────────────────┐
    │  DIRECT MODE    │               │ Do you also have    │
    │  (Recommended)  │               │ fresh responses?    │
    └─────────────────┘               └─────────────────────┘
                                               │
                              ┌────────────────┴────────────────┐
                              │ NO                              │ YES
                              ▼                                 ▼
                    ┌─────────────────┐               ┌─────────────────┐
                    │   IPS MODE      │               │   DR MODE       │
                    │   (Fast)        │               │   (Best)        │
                    └─────────────────┘               └─────────────────┘
```

---

## The Three Modes

| Mode | What You Need | Best For | Accuracy | Speed |
|:-----|:--------------|:---------|:--------:|:-----:|
| **[Direct](modes/direct.md)** | Fresh responses + judge scores | A/B testing, leaderboards | ★★★★☆ | ★★★★★ |
| **[IPS](modes/ips.md)** | Logged data + logprobs | "What if we had used Model B?" | ★★★☆☆ | ★★★★★ |
| **[DR](modes/dr.md)** | Both logged + fresh | Production deployment decisions | ★★★★★ | ★★★☆☆ |

---

## Data Checklist

Before you start, verify you have:

### Minimum Requirements (All Modes)
- [ ] **Prompts** — The inputs you're evaluating
- [ ] **Responses** — LLM outputs for each prompt
- [ ] **Judge scores** — `judge_score` field (0-1) from your LLM judge

### For Calibration (Recommended)
- [ ] **Oracle labels** — `oracle_label` field for 5-25% of samples
  - *This is what makes CJE work. Without it, you're just averaging biased scores.*

### For IPS/DR Modes
- [ ] **Log probabilities** — `base_policy_logprob` and `target_policy_logprobs`
  - *Required for counterfactual estimation*

---

## Quick Decision Guide

### "I just want to compare Model A vs Model B"
→ **[Direct Mode](modes/direct.md)** — Generate responses, score them, done.

### "I have 100k logs. What if we had used a different model?"
→ **[IPS Mode](modes/ips.md)** — Re-weight your existing logs. No new inference needed.

### "This is a high-stakes production decision"
→ **[DR Mode](modes/dr.md)** — Maximum accuracy. Combines logs + fresh responses.

### "I don't understand calibration / my CIs seem weird"
→ **[Calibration Guide](calibration.md)** — How AutoCal-R fixes judge bias.

### "CJE returned NaN / my results look wrong"
→ **[Diagnostics Guide](diagnostics.md)** — Debug overlap, ESS, and refusal gates.

---

## Tutorials

| Tutorial | Time | What You'll Learn |
|:---------|:-----|:------------------|
| [**Quick Start**](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_tutorial.ipynb) | 5 min | Direct Mode basics |
| [**Advanced**](https://colab.research.google.com/github/cimo-labs/cje/blob/main/examples/cje_advanced.ipynb) | 15 min | IPS & DR modes |

---

## Reference Documentation

For implementation details and API specifications:

- [Calibration Module](../cje/calibration/README.md) — AutoCal-R, isotonic regression, SIMCal-W
- [Estimators Module](../cje/estimators/README.md) — IPS, DR, TMLE, stacking
- [Diagnostics Module](../cje/diagnostics/README.md) — ESS, overlap metrics, gates
- [Interface Module](../cje/interface/README.md) — `analyze_dataset()` API
