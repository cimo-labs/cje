# Direct Mode: A/B Testing Without Logprobs

**Use Case:** Compare policies on a fixed evaluation set using fresh responses.

**Best For:**
- A/B testing
- Model leaderboards
- Evaluating closed-source APIs (no logprob access)
- Quick policy comparison

---

## When to Use Direct Mode

✅ **Use Direct Mode when:**
- You can generate fresh responses from each policy
- You don't have (or don't need) historical logs
- You want the simplest possible setup

❌ **Don't use Direct Mode when:**
- You want counterfactual estimates ("What if we had deployed X?")
- You have valuable logged data you want to leverage → Use [IPS](ips.md) or [DR](dr.md)

---

## Data Format

### Directory Structure

```
fresh_draws_dir/
├── policy_a_responses.jsonl
├── policy_b_responses.jsonl
└── policy_c_responses.jsonl
```

Policy names are extracted from filenames: `{policy}_responses.jsonl` → policy `"policy_a"`

### JSONL Schema

Each line is a JSON object:

```json
{
  "prompt_id": "eval_001",
  "judge_score": 0.85,
  "oracle_label": 0.82,
  "response": "The answer is 42 because...",
  "prompt": "What is the meaning of life?"
}
```

| Field | Required | Type | Description |
|:------|:--------:|:-----|:------------|
| `prompt_id` | ✅ | string | Unique prompt identifier (must match across policies) |
| `judge_score` | ✅ | float [0,1] | LLM judge evaluation |
| `oracle_label` | ❌ | float [0,1] | Ground truth (5-25% of samples recommended) |
| `response` | ❌ | string | Generated response text |
| `prompt` | ❌ | string | Original prompt |

> **Critical:** `prompt_id` must be consistent across policy files. This enables paired estimation.

---

## How It Works

### The Estimation Formula

For each policy π:

```
V̂(π) = (1/n) Σᵢ f̂(Sᵢᵖ)
```

Where:
- `Sᵢᵖ` = judge score for policy π on prompt i
- `f̂(S)` = calibrated mapping from judge score to oracle (learned via AutoCal-R)

### Why Pairing Matters

CJE clusters observations by `prompt_id`. This matters because:

**Without pairing:**
- "Model A scored 0.75, Model B scored 0.72"
- But maybe Model A got easier prompts!

**With pairing:**
- "On the *same* prompts, Model A beat Model B by 0.03"
- Prompt difficulty cancels out

This is why Direct Mode requires the same prompts across policies.

---

## Code Examples

### Basic Usage

```python
from cje import analyze_dataset

results = analyze_dataset(
    fresh_draws_dir="data/responses/",
    estimator="direct"  # or "auto" (will auto-detect)
)

# Results
print(f"Policies: {results.metadata['target_policies']}")
print(f"Estimates: {results.estimates}")
print(f"Standard Errors: {results.standard_errors}")
```

### With Covariates (Length Bias Correction)

If your judge is biased by response length:

```python
results = analyze_dataset(
    fresh_draws_dir="data/responses/",
    include_response_length=True,  # Auto-compute word count
    calibration_covariates=["domain"]  # Add other covariates
)
```

This triggers two-stage calibration: `g(S, length, domain) → rank → isotonic`

### Comparing Policies

```python
# Find the best policy
best_idx = results.best_policy()
print(f"Best: {results.metadata['target_policies'][best_idx]}")

# Statistical comparison
comparison = results.compare_policies(0, 1)  # Policy 0 vs Policy 1
print(f"Difference: {comparison['difference']:+.3f}")
print(f"p-value: {comparison['p_value']:.3f}")
print(f"Significant (α=0.05): {comparison['significant']}")
```

### Visualization

```python
# Forest plot
fig = results.plot_estimates(save_path="comparison.png")

# Or with more control
from cje import plot_policy_estimates

plot_policy_estimates(
    estimates=dict(zip(results.metadata['target_policies'], results.estimates)),
    standard_errors=dict(zip(results.metadata['target_policies'], results.standard_errors)),
    figsize=(10, 6)
)
```

---

## Understanding the Output

### Estimates

The calibrated policy value. Higher = better (assuming oracle_label encodes quality).

### Standard Errors

Account for:
1. **Sampling variance** — Finite number of prompts
2. **Calibration uncertainty** — Oracle labels are sparse (OUA jackknife)
3. **Clustering** — Correlation across policies on the same prompt

### Confidence Intervals

CJE uses t-critical values (not z) to account for finite degrees of freedom:

```python
ci_lower, ci_upper = results.confidence_interval(alpha=0.05)
# Or
cis = results.ci()  # List of (lower, upper) tuples
```

---

## Limitations

> ⚠️ **Y* Check:** Direct Mode evaluates performance *on this specific prompt set*.

It does NOT answer:
- "What would happen if we deployed this in production?"
- "How would this perform on different prompts?"

For counterfactual questions, use [IPS](ips.md) or [DR](dr.md).

---

## Troubleshooting

### "Only one policy found"

Check your filenames. CJE expects `{policy}_responses.jsonl`.

### "prompt_id mismatch"

Ensure all policies have the same prompt_ids. CJE requires paired data.

### "No oracle labels found"

Calibration won't happen. Results are just averaged judge scores.
Add oracle labels to 5-25% of samples for bias correction.

### "Very wide confidence intervals"

Possible causes:
- Too few prompts (need more data)
- Low oracle coverage (add more labels)
- High judge variance (consider a better judge)

---

## Reference

For implementation details, see [Estimators Module](../../cje/estimators/README.md).
