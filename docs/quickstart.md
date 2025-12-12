# Quick Start: Compare Two Policies in 5 Minutes

**Scenario:** You have responses from Model A and Model B. You want to know which is better.

This guide shows the fastest path to a statistically valid answer.

---

## Step 1: Prepare Your Data

Create a folder with one JSONL file per policy:

```
data/responses/
├── model_a_responses.jsonl
└── model_b_responses.jsonl
```

Each file contains one JSON object per line:

```json
{"prompt_id": "q1", "judge_score": 0.85, "oracle_label": 0.82}
{"prompt_id": "q2", "judge_score": 0.72}
{"prompt_id": "q3", "judge_score": 0.91, "oracle_label": 0.88}
```

### Required Fields

| Field | Type | Description |
|:------|:-----|:------------|
| `prompt_id` | string | Unique identifier for each prompt |
| `judge_score` | float (0-1) | Your LLM judge's evaluation |

### Optional (But Recommended)

| Field | Type | Description |
|:------|:-----|:------------|
| `oracle_label` | float (0-1) | Ground truth label (only need 5-25% of samples) |
| `response` | string | The actual response text |
| `prompt` | string | The original prompt |

> **Why oracle labels?** Without them, CJE just averages your (biased) judge scores. With them, CJE learns to *correct* the judge's bias. Even 50 labels can make a big difference.

---

## Step 2: Run the Analysis

```python
from cje import analyze_dataset

# Point to your data folder
results = analyze_dataset(fresh_draws_dir="data/responses/")

# Print results
for policy, est, se in zip(
    results.metadata["target_policies"],
    results.estimates,
    results.standard_errors
):
    print(f"{policy}: {est:.3f} ± {1.96*se:.3f}")
```

**Example output:**
```
model_a: 0.742 ± 0.024
model_b: 0.698 ± 0.027
```

---

## Step 3: Visualize the Results

```python
# Generate a forest plot
fig = results.plot_estimates(save_path="comparison.png")
```

This produces a plot showing:
- Point estimates (dots)
- 95% confidence intervals (horizontal bars)
- The best policy highlighted

---

## Step 4: Make the Decision

### Reading the Forest Plot

```
model_a  ●━━━━━━━━━━━━●━━━━━━━━━━━━●
model_b       ●━━━━━━━━━━━━●━━━━━━━━━━━━●
         0.65       0.70       0.75       0.80
```

**Interpretation:**
- If confidence intervals **don't overlap** → Statistically significant difference
- If confidence intervals **overlap** → Cannot confidently distinguish

### Formal Statistical Test

```python
# Compare model_a (index 0) vs model_b (index 1)
comparison = results.compare_policies(0, 1)

print(f"Difference: {comparison['difference']:+.3f}")
print(f"p-value: {comparison['p_value']:.3f}")
print(f"Significant: {comparison['significant']}")
```

---

## What Just Happened?

Behind the scenes, CJE:

1. **Detected policies** from your filenames (`model_a`, `model_b`)
2. **Found oracle labels** and applied AutoCal-R calibration
3. **Computed paired estimates** clustering by `prompt_id` (removing prompt difficulty as a confounder)
4. **Generated honest CIs** that account for both sampling error and calibration uncertainty

---

## Common Questions

### "I don't have oracle labels. Can I still use CJE?"

Yes, but your estimates will just be averages of the (biased) judge scores. CJE will still:
- Cluster by `prompt_id` for paired comparisons
- Provide standard errors

To unlock the full power of CJE (bias correction), add oracle labels to 5-25% of your samples.

### "How do I get oracle labels?"

Common approaches:
- **Human evaluation** on a random sample
- **Stronger LLM** (e.g., GPT-4 or Claude) on a subset
- **Ground truth** if available (e.g., correctness on math problems)

### "My results say NaN"

CJE refuses to give an estimate when the data is too unreliable. See the [Diagnostics Guide](diagnostics.md) for debugging.

---

## Next Steps

- **More control?** See [Direct Mode Deep Dive](modes/direct.md)
- **Have logs with logprobs?** See [IPS Mode](modes/ips.md)
- **High-stakes decision?** See [DR Mode](modes/dr.md)
- **Weird results?** See [Diagnostics](diagnostics.md)
