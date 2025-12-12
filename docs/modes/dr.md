# DR Mode: The Gold Standard

**Use Case:** High-stakes production decisions requiring maximum accuracy.

**Best For:**
- Deployment decisions
- When IPS has poor overlap
- Combining historical logs with fresh evaluation data
- Getting the tightest possible confidence intervals

---

## When to Use DR Mode

✅ **Use DR Mode when:**
- This decision matters (production deployment, significant resource allocation)
- You have both logged data AND can generate fresh responses
- IPS gave poor ESS (< 30%)
- You want the best possible estimate

❌ **Don't use DR Mode when:**
- You only have fresh draws → Use [Direct Mode](direct.md)
- You only have logs → Use [IPS Mode](ips.md)
- Speed matters more than accuracy

---

## Why DR Wins

**Doubly Robust (DR)** combines the best of both worlds:

| Component | What It Provides | Weakness |
|:----------|:-----------------|:---------|
| **IPS** | Unbiased if weights correct | High variance when overlap is poor |
| **Direct (Outcome Model)** | Low variance | Biased if model misspecified |
| **DR** | Unbiased if *either* is correct | Needs both data sources |

### The Magic Property

DR is consistent if **either**:
- The importance weights are correct, OR
- The outcome model is correct

You get two chances to be right. In practice, this makes DR far more robust than IPS alone.

---

## Data Requirements

DR needs **two data sources**:

### 1. Logged Data (with logprobs)

Same format as IPS mode:

```json
{
  "prompt_id": "log_001",
  "base_policy_logprob": -12.5,
  "target_policy_logprobs": {"policy_a": -14.2, "policy_b": -18.7},
  "judge_score": 0.85,
  "oracle_label": 0.82
}
```

### 2. Fresh Draws (per policy)

Same format as Direct mode:

```
fresh_draws_dir/
├── policy_a_responses.jsonl
└── policy_b_responses.jsonl
```

Each file:
```json
{"prompt_id": "fresh_001", "judge_score": 0.78, "oracle_label": 0.75}
```

### Policy Name Matching

> ⚠️ **Critical:** Policy names must match EXACTLY between logged data and fresh draw filenames.

```python
# In logged data
"target_policy_logprobs": {"gpt-4": -14.2, "claude-3": -18.7}

# Fresh draw files must be named exactly:
# - gpt-4_responses.jsonl  ✓
# - claude-3_responses.jsonl  ✓
# - gpt4_responses.jsonl  ✗ (won't match)
```

---

## How It Works

### The DR Estimator

```
V̂_DR(π) = (1/n) Σᵢ [q̂(Xᵢ) + wᵢ(Rᵢ - q̂(Xᵢ))]

where:
  q̂(X) = outcome model (predicts reward from covariates)
  wᵢ = importance weight
  Rᵢ = observed reward
```

**Intuition:**
1. Start with outcome model prediction: `q̂(X)`
2. Add IPS correction for the residual: `wᵢ(Rᵢ - q̂(X))`
3. If outcome model is perfect, correction is zero
4. If weights are perfect, correction fixes any model bias

### Cross-Fitting

To prevent overfitting, CJE uses K-fold cross-fitting:
1. Split data into K folds
2. Train outcome model on K-1 folds
3. Predict on held-out fold
4. Repeat for all folds

This maintains the theoretical guarantees of DR.

### Stacked DR (Default)

CJE's `stacked-dr` estimator combines 5 DR variants:
- DR-CPO (basic)
- TMLE (targeted learning)
- MRDR (multiply robust)
- OC-DR-CPO (orthogonalized)
- TR-CPO-E (triply robust)

The optimal combination is learned via influence function stacking, typically giving 1-5% SE reduction over the best single method.

---

## Code Examples

### Basic Usage

```python
from cje import analyze_dataset

results = analyze_dataset(
    logged_data_path="data/logs.jsonl",
    fresh_draws_dir="data/responses/",
    estimator="stacked-dr"  # or "auto" (will auto-select DR)
)

print(f"Mode: {results.metadata['mode']}")  # "dr"
print(f"Estimator: {results.metadata['estimator']}")  # "stacked-dr"
```

### Checking Data Integration

```python
# Verify both sources loaded
print(f"Logged samples: {results.metadata.get('n_logged', 0)}")
print(f"Fresh draws loaded: {results.metadata.get('fresh_draws_loaded', [])}")

# Check policy matching
logged_policies = set(results.metadata.get('logged_policies', []))
fresh_policies = set(results.metadata.get('target_policies', []))
if logged_policies != fresh_policies:
    print(f"⚠️ Policy mismatch!")
    print(f"   In logs: {logged_policies}")
    print(f"   In fresh: {fresh_policies}")
```

### Specific DR Variants

```python
# Use a specific DR estimator instead of stacked
results = analyze_dataset(
    logged_data_path="data/logs.jsonl",
    fresh_draws_dir="data/responses/",
    estimator="tmle"  # or "dr-cpo", "mrdr", "oc-dr-cpo", "tr-cpo-e"
)
```

---

## Diagnostics

DR inherits IPS diagnostics plus additional checks:

### Outcome Model Quality

```python
if hasattr(results.diagnostics, 'outcome_r2_range'):
    min_r2, max_r2 = results.diagnostics.outcome_r2_range
    print(f"Outcome model R²: [{min_r2:.3f}, {max_r2:.3f}]")

    if min_r2 < 0:
        print("⚠️ Negative R² indicates model misspecification")
```

### Orthogonality Check

For OC-DR and TR-CPO variants:

```python
if hasattr(results.diagnostics, 'orthogonality_scores'):
    for policy, score in results.diagnostics.orthogonality_scores.items():
        print(f"{policy}: orthogonality score = {score:.4f}")
        # Should be close to 0 with CI containing 0
```

### DM vs IPS Decomposition

```python
if hasattr(results.diagnostics, 'dm_ips_decompositions'):
    for policy, decomp in results.diagnostics.dm_ips_decompositions.items():
        dm_share = decomp['dm_contribution'] / decomp['total']
        print(f"{policy}: {dm_share:.0%} from outcome model, {1-dm_share:.0%} from IPS")
```

---

## When DR Outperforms

### Scenario 1: Poor IPS Overlap

```
IPS ESS: 5% → High variance
DR with decent outcome model → Much lower variance
```

DR's outcome model provides a stable baseline; IPS only corrects residuals.

### Scenario 2: Biased Outcome Model

```
Outcome model: Systematic bias
IPS weights: Correct
DR → IPS correction fixes the bias
```

### Scenario 3: Both Partially Wrong

```
Outcome model: 70% accurate
IPS weights: Noisy but unbiased
DR → Combines strengths of both
```

---

## Computational Cost

DR is more expensive than IPS or Direct:

| Operation | Cost |
|:----------|:-----|
| Load logged data | O(n) |
| Load fresh draws | O(m × policies) |
| Cross-fit outcome model | O(K × model_training) |
| Compute estimates | O(n) |
| Stacking (5 estimators) | 5× single DR |

For large datasets (>100k samples), consider:
- Reducing `n_folds` (default 20 → 5)
- Using a single DR variant instead of stacking

---

## Limitations

> DR requires **both** logged data AND fresh draws. If you only have one, use Direct or IPS.

DR can still fail if:
- Overlap is catastrophically bad (ESS < 1%)
- Outcome model is severely misspecified AND weights are wrong
- Data quality issues (missing logprobs, mismatched policies)

---

## Reference

For implementation details:
- [Estimators Module](../../cje/estimators/README.md) — DR estimator implementations
- [Calibration Module](../../cje/calibration/README.md) — Outcome model fitting
- [Diagnostics Module](../../cje/diagnostics/README.md) — DR-specific diagnostics
