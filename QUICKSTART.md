# CJE Quickstart - 5 Minutes to First Results

This guide gets you from zero to your first unbiased LLM evaluation in 5 minutes.

## Prerequisites

**Minimal (Direct mode):**
✅ Python 3.9+ installed
✅ Responses from policies you want to compare
✅ Judge scores for each response

**Advanced (IPS/DR modes):**
✅ Log probabilities from your models
✅ 5-10% samples with oracle labels (for AutoCal-R calibration)

## Step 1: Install (30 seconds)

```bash
pip install cje-eval
```

## Step 2: Prepare Your Data (1 minute)

**Simplest: Direct mode (no logprobs needed)**

Create **one JSONL file per policy** in a `responses/` directory:

**`responses/model_a_responses.jsonl`:**
```json
{"prompt_id": "eval_0", "judge_score": 0.85}
{"prompt_id": "eval_1", "judge_score": 0.91}
```

**`responses/model_b_responses.jsonl`:**
```json
{"prompt_id": "eval_0", "judge_score": 0.72}
{"prompt_id": "eval_1", "judge_score": 0.88}
```

The policy name is inferred from the filename (no `policy` field needed).

**Advanced: IPS/DR modes (with logprobs)**

For counterfactual inference, you need log probabilities:

```json
{
  "prompt": "What is 2+2?",
  "response": "4",
  "base_policy_logprob": -45.2,
  "target_policy_logprobs": {"model_a": -42.1, "model_b": -44.3},
  "judge_score": 0.82,
  "oracle_label": 0.90  // Optional: ground truth (5-10% coverage)
}
```

## Step 3: Run Your First Analysis (30 seconds)

**Direct mode (simplest):**

```python
from cje import analyze_dataset

# Compare policies on eval set
result = analyze_dataset(fresh_draws_dir="responses/")

# View results with confidence intervals
for i, policy in enumerate(result.metadata["target_policies"]):
    est = result.estimates[i]
    se = result.standard_errors[i]
    print(f"{policy}: {est:.3f} ± {1.96*se:.3f}")
```

**IPS mode (with logprobs):**

```python
# Estimate counterfactual deployment value
result = analyze_dataset(logged_data_path="logs.jsonl")
print(f"Model A estimated value: {result.estimates[0]:.3f}")
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

## Step 5: The Detect → Fix → Re-run Workflow (2 minutes)

CJE provides estimates with diagnostics, even when data quality isn't perfect. Use diagnostics to decide if you need to improve and re-run:

```python
# Check diagnostics (CJE provides estimates regardless)
diagnostics = result.diagnostics

# Inspect overall health
print(f"Status: {diagnostics.overall_status.value}")
print(f"ESS: {diagnostics.weight_ess:.1%}")

# Check per-policy diagnostics
for policy, ess in diagnostics.ess_per_policy.items():
    print(f"  {policy}: {ess:.1%}")

if diagnostics.overall_status.value == "CRITICAL":
    print("\n⚠️ Diagnostics suggest improvements needed")
    print("Current estimates provided, but consider fixes below before production use")
    print(diagnostics.summary())
else:
    print("\n✅ Diagnostics look good - estimates are reliable")
```

**Workflow:**
1. **Run analysis** - Get estimates + diagnostics
2. **Check diagnostics** - Review health metrics
3. **If issues detected** - Apply fixes below and re-run
4. **Ship with confidence** - Use improved estimates

## Common Issues & How to Fix Them

### Issue: "Low ESS" (Effective Sample Size < 30%)

**What it means:** Policies are very different from logging policy

**Fixes to try (in order):**
```python
# Fix 1: Use DR mode with fresh draws (usually best)
result = analyze_dataset(
    "logs.jsonl",
    fresh_draws_dir="fresh_draws/",
    estimator="stacked-dr"
)

# Fix 2: Restrict to overlapping cohort
# Filter data to prompts where policies have similar behavior

# Fix 3: Collect more diverse base policy data
# Use multiple base policies or sample more broadly
```

### Issue: "No oracle labels found"

**What it means:** AutoCal-R can't calibrate judge scores to ground truth without oracle labels

**Fix:**
```python
# Add ground truth to 5-10% of samples for AutoCal-R
import random
labeled_sample_ids = random.sample(all_ids, k=int(0.1 * len(all_ids)))
# Label these samples and add oracle_label field
# Re-run analysis - AutoCal-R will automatically calibrate

# Generate log probabilities for your target model
for sample in your_data:
    result = compute_teacher_forced_logprob(
        prompt=sample["prompt"],
        response=sample["response"],
        model="accounts/fireworks/models/llama-v3p2-3b-instruct"
    )
    if result.status == "success":
        sample["target_logprob"] = result.value
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
cis = result.ci()
for i, policy in enumerate(result.metadata["target_policies"]):
    estimate = result.estimates[i]
    ci_low, ci_high = cis[i]

    print(f"\n{policy}:")
    print(f"  Estimate: {estimate:.3f}")
    print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  ESS: {result.diagnostics.ess_per_policy[policy]:.1%}")
```

## FAQ

**Q: Can I use CJE without log probabilities?**
A: Yes! Use Direct mode - just provide responses from each policy with judge scores. No logprobs needed.

**Q: How much ground truth do I need?**
A: 5-10% of samples with oracle labels is usually sufficient for AutoCal-R to learn accurate calibration.

**Q: How do I get log probabilities?**
A: CJE has built-in Fireworks API integration. See [Teacher Forcing](cje/teacher_forcing/README.md) for details.

**Q: When should I use IPS vs DR vs Direct mode?**
A: Direct = simplest (compare policies). IPS = counterfactual (logged data only). DR = most accurate (both logged + fresh).

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