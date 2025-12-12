# IPS Mode: Counterfactual Estimation from Logs

**Use Case:** "I have 100k logged responses. What if we had used a different model?"

**Best For:**
- Historical analysis without new inference
- Quick counterfactual estimates
- Exploring "what if" scenarios from existing data

---

## When to Use IPS Mode

✅ **Use IPS Mode when:**
- You have logged data with log probabilities
- You want fast counterfactual estimates (no new inference)
- The target policy is *similar* to the logging policy

❌ **Don't use IPS Mode when:**
- Target and logging policies are very different (overlap issues)
- You need maximum accuracy → Use [DR Mode](dr.md)
- You don't have logprobs → Use [Direct Mode](direct.md)

---

## ⚠️ The Overlap Problem

> **This is the #1 failure mode of IPS. Read this section carefully.**

IPS works by re-weighting logged data:

```
Weight = P(response | target_policy) / P(response | logging_policy)
```

**The Problem:** If the target policy would *never* produce the logged response, the weight is ~0. If it would produce it *much more often*, the weight explodes.

### Signs of Poor Overlap

| Symptom | Meaning |
|:--------|:--------|
| ESS < 10% | 90% of your data is effectively useless |
| A few weights > 100 | Estimate dominated by handful of samples |
| CJE returns NaN | Overlap so bad we refuse to estimate |

### The Rule of Thumb

> **If your target policy is a different model family (e.g., GPT-4 → Llama-7B), IPS will fail.**

Use [DR Mode](dr.md) instead, which combines IPS with fresh draws to fix overlap issues.

---

## Data Format

### JSONL Schema

```json
{
  "prompt_id": "log_001",
  "prompt": "What is the capital of France?",
  "response": "Paris is the capital of France.",
  "base_policy_logprob": -12.5,
  "target_policy_logprobs": {
    "policy_a": -14.2,
    "policy_b": -18.7,
    "policy_c": -45.3
  },
  "judge_score": 0.85,
  "oracle_label": 0.82
}
```

| Field | Required | Type | Description |
|:------|:--------:|:-----|:------------|
| `prompt_id` | ✅ | string | Unique identifier |
| `base_policy_logprob` | ✅ | float ≤ 0 | Log prob under logging policy |
| `target_policy_logprobs` | ✅ | dict | Log probs under each target policy |
| `judge_score` | ✅ | float [0,1] | LLM judge evaluation |
| `oracle_label` | ❌ | float [0,1] | Ground truth (5-25% recommended) |

### Getting Log Probabilities

You need `log P(response | model)` for both the logging policy and each target policy.

**For open models (Llama, Mistral, etc.):**
```python
# Teacher forcing: compute log prob of existing response
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b")

inputs = tokenizer(prompt + response, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    logprob = -outputs.loss.item() * inputs["input_ids"].shape[1]
```

**For closed APIs:** Many APIs (OpenAI, Anthropic) return logprobs on request.

See the [Teacher Forcing Module](../../cje/teacher_forcing/README.md) for utilities.

---

## How It Works

### The IPS Estimator

```
V̂_IPS(π) = (1/n) Σᵢ wᵢ · Rᵢ

where wᵢ = exp(logP_target - logP_base) = P(response | target) / P(response | base)
```

### SIMCal-W: Weight Stabilization

Raw importance weights can have extreme variance. CJE applies **SIMCal-W** (Surrogate-Indexed Monotone Calibration for Weights):

1. Project weights onto monotone functions of judge score
2. Cap variance to prevent explosion
3. Preserve mean=1 for unbiasedness

This typically improves ESS by 10-50x.

---

## Code Examples

### Basic Usage

```python
from cje import analyze_dataset

results = analyze_dataset(
    logged_data_path="data/logs.jsonl",
    estimator="calibrated-ips"  # or "auto"
)

# Check diagnostics FIRST
print(f"ESS: {results.diagnostics.weight_ess:.1%}")
if results.diagnostics.weight_ess < 0.1:
    print("⚠️ WARNING: Very low ESS. Results may be unreliable.")
```

### Checking Overlap Quality

```python
# Per-policy ESS
for policy, ess in results.diagnostics.ess_per_policy.items():
    status = "✓" if ess > 0.1 else "⚠️"
    print(f"{status} {policy}: ESS = {ess:.1%}")

# Weight distribution
for policy, max_w in results.diagnostics.max_weight_per_policy.items():
    status = "✓" if max_w < 100 else "⚠️"
    print(f"{status} {policy}: max weight = {max_w:.1f}")
```

### When IPS Fails

```python
import numpy as np

for i, policy in enumerate(results.metadata['target_policies']):
    est = results.estimates[i]
    if np.isnan(est):
        print(f"❌ {policy}: CJE refused to estimate (overlap too poor)")
        print(f"   Consider using DR mode with fresh draws for this policy")
```

---

## Diagnostics Deep Dive

### Effective Sample Size (ESS)

```
ESS = (Σwᵢ)² / Σwᵢ²
ESS_fraction = ESS / n
```

| ESS Fraction | Interpretation |
|:-------------|:---------------|
| > 30% | Good overlap |
| 10-30% | Marginal, check results carefully |
| < 10% | Poor overlap, consider DR mode |
| < 1% | Catastrophic, results meaningless |

### Hill Tail Index

Measures how "heavy-tailed" your weights are:

| Tail Index | Meaning |
|:-----------|:--------|
| α > 2 | Finite variance (good) |
| 1 < α < 2 | Infinite variance (warning) |
| α < 1 | Infinite mean (critical) |

---

## Limitations

> ⚠️ **Goodhart Watch:** IPS is seductive because it's fast. But speed is worthless if the estimate is garbage.

**Always check ESS before trusting IPS results.**

Common failure modes:

1. **Model family mismatch:** GPT-4 logs → Llama estimates = disaster
2. **Temperature mismatch:** Temperature 0 logs → Temperature 1 estimates = poor overlap
3. **Prompt changes:** Different system prompts = different distributions

---

## When to Give Up on IPS

If after checking diagnostics you see:
- ESS < 5% for most policies
- Max weights > 1000
- CJE returning NaN

**The fix:** Use [DR Mode](dr.md) with fresh draws. DR is robust to overlap issues because it falls back to a direct estimator when weights fail.

---

## Reference

For implementation details:
- [Calibration Module](../../cje/calibration/README.md) — SIMCal-W weight stabilization
- [Estimators Module](../../cje/estimators/README.md) — CalibratedIPS implementation
- [Diagnostics Module](../../cje/diagnostics/README.md) — ESS, Hill index, overlap metrics
