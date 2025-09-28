# CJE Quickstart - 5 Minutes to First Results

This guide gets you from zero to your first unbiased LLM evaluation in 5 minutes.

## Prerequisites

✅ Python 3.9+ installed
✅ 1000+ examples with judge scores
✅ Log probabilities from your models

## Step 1: Install (30 seconds)

```bash
pip install cje-eval
```

## Step 2: Understand Your Data (1 minute)

CJE needs data in JSONL format with these fields:

```json
{
  "prompt": "What is machine learning?",
  "response": "Machine learning is...",
  "base_policy_logprob": -45.2,      // Log P(response|prompt) for current model
  "target_policy_logprobs": {         // Same for models you want to evaluate
    "gpt4": -42.1,
    "claude": -44.3
  },
  "metadata": {
    "judge_score": 0.82,              // Your LLM judge's score
    "oracle_label": 0.90              // Ground truth (only need 5-10% labeled)
  }
}
```

Don't have this format? See the data preparation section below.

## Step 3: Run Your First Analysis (30 seconds)

```python
from cje import analyze_dataset

# Simplest possible usage
result = analyze_dataset("your_data.jsonl", estimator="calibrated-ips")

# View results
print(f"GPT-4 estimated value: {result.estimates[0]:.3f}")
print(f"95% CI: [{result.robust_confidence_intervals[0][0]:.3f}, "
      f"{result.robust_confidence_intervals[0][1]:.3f}]")
```

## Step 4: Understand the Output (2 minutes)

You'll see something like:
```
GPT-4 estimated value: 0.731
95% CI: [0.712, 0.750]

Diagnostics Summary:
- Effective Sample Size: 43.2% (GOOD)
- Weight concentration: Low
- Calibration R²: 0.83
```

What this means:
- **Estimated value**: GPT-4 would score 73.1% if deployed
- **Confidence interval**: We're 95% sure it's between 71.2% and 75.0%
- **ESS 43.2%**: Good overlap between policies (>30% is good)

## Step 5: Check Reliability (1 minute)

```python
# Always check diagnostics before trusting results
diagnostics = result.diagnostics

if diagnostics.overall_status.value == "CRITICAL":
    print("⚠️ Results may be unreliable!")
    print(diagnostics.summary())
else:
    print("✅ Results are reliable")

# For detailed analysis
print(f"ESS: {diagnostics.weight_ess:.1%}")
for policy, ess in diagnostics.ess_per_policy.items():
    print(f"  {policy}: {ess:.1%}")
```

## Common Issues & Solutions

### "Low ESS warning"
Your policies are very different. Solutions:
1. Use doubly-robust estimation (requires fresh samples)
2. Collect more diverse training data
3. Use a less different target policy

### "No oracle labels found"
Add ground truth to 5-10% of samples:
```python
# Before running CJE, add some labels
import random
labeled_sample_ids = random.sample(all_ids, k=int(0.1 * len(all_ids)))
# Add oracle_label to metadata for these samples
```

### "ValueError: DR estimators require fresh draws"
Generate new samples from target policy using CJE's Fireworks integration:
```python
from cje.teacher_forcing import compute_teacher_forced_logprob

# Generate log probabilities for your target model
for sample in your_data:
    logprob = compute_teacher_forced_logprob(
        prompt=sample["prompt"],
        response=sample["response"],
        model="accounts/fireworks/models/llama-v3p2-3b-instruct"
    )
```

## Next Steps - Choose Your Path

### 🎯 "I just want better estimates"
→ Choose the Right Estimator
```python
# If you have fresh samples from GPT-4:
result = analyze_dataset("data.jsonl", estimator="stacked-dr",
                         fresh_draws_dir="gpt4_responses/")
```

### 📊 "I need to compare multiple models"
→ Compare Multiple Policies
```python
results = analyze_dataset("data.jsonl", estimator="calibrated-ips")
best_policy = results.diagnostics.best_policy
print(f"Best policy: {best_policy}")
```

### 🔬 "I want to understand the theory"
→ Mathematical Foundations (documentation coming soon)
- Why importance sampling works
- SIMCal calibration details
- Doubly robust estimation

### 🏭 "I'm deploying to production"
→ Production Configuration
```python
# Production configuration
result = analyze_dataset(
    "data.jsonl",
    estimator="calibrated-ips",
    refuse_unreliable=True,  # Return NaN if unreliable
    verbose=True             # Log progress
)
```

## Complete Example

Here's everything together:

```python
from cje import analyze_dataset

# Load and analyze
result = analyze_dataset(
    "llm_judge_data.jsonl",
    estimator="calibrated-ips",
    judge_field="judge_score",
    oracle_field="human_label"
)

# Check reliability
if result.diagnostics.weight_ess < 0.1:
    print("Warning: Low ESS, consider using DR methods")

# Report results
for i, policy in enumerate(result.metadata["target_policies"]):
    estimate = result.estimates[i]
    se = result.robust_standard_errors[i]
    ci_low, ci_high = result.robust_confidence_intervals[i]

    print(f"\n{policy}:")
    print(f"  Estimate: {estimate:.3f} ± {se:.3f}")
    print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  ESS: {result.diagnostics.ess_per_policy[policy]:.1%}")
```

## FAQ

**Q: How much ground truth do I need?**
A: 5-10% of samples with oracle labels is usually sufficient.

**Q: Can I use this without log probabilities?**
A: No, log probs are essential for importance weighting.

**Q: How do I get log probabilities?**
A: Most APIs provide them: OpenAI (`logprobs=True`), Anthropic (`log_probs=true`), etc.

**Q: What if I don't have fresh samples?**
A: Use `calibrated-ips` - it's robust without fresh samples. DR methods need them.

---

🎉 **Congrats!** You've run your first unbiased LLM evaluation.

**Ready for more?**
- Understanding diagnostics - See diagnostics.summary()
- Generating fresh samples - Use teacher_forcing module
- [Engineering Guide](README_ENGINEERING.md) - Interface specs and patterns
- Research paper - Coming soon

**For Developers:**
Each module in `cje/` has a developer-oriented README with implementation details:
- `cje/estimators/README.md` - Estimator implementations
- `cje/diagnostics/README.md` - Diagnostic architecture
- `cje/data/README.md` - Data models and validation
- `cje/calibration/README.md` - Calibration methods