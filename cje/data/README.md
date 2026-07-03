# CJE Data Module

## Overview

The data module handles data loading, validation, and preparation for Direct-mode CJE analysis. It provides type-safe Pydantic models, loud loaders (fail with file/line context, never silently shrink your data), raw-record validation, and automatic label-scale normalization.

## When to Use

### Use **fresh draws** (the primary input) when:
- You have judge-scored responses from each policy you want to compare
- Records live in per-policy JSONL files (`fresh_draws_dir`) or in memory (`fresh_draws_data`)

### Use **Dataset / load_dataset_from_jsonl** when:
- You're loading a calibration file (judge + oracle pairs) for `calibration_data_path`
- You need a type-safe container to pass to `calibrate_dataset`

### Use **validate_direct_data** when:
- You want to check raw parsed JSONL records before running analysis
- (Or just run `cje validate PATH` — same code path)

### Use **fresh_draws_from_dict** when:
- You have fresh draws in memory and want `FreshDrawDataset` objects without file I/O

## File Structure

```
data/
├── __init__.py           # Public API exports
├── models.py             # Pydantic models (Sample, Dataset, EstimationResult)
├── loaders.py            # DatasetLoader, FreshDrawLoader, DataSource protocol
├── factory.py            # DatasetFactory (dependency-injected loading)
├── fresh_draws.py        # FreshDrawSample/FreshDrawDataset + loading utilities
├── folds.py              # Unified hash-based cross-validation folds
├── normalization.py      # Auto-normalization of bounded label scales
├── validation.py         # validate_direct_data on raw records
└── reward_utils.py       # Reward manipulation utilities
```

## Data Format

### Fresh-draw records (evaluation data)

The minimal record is two fields:

```json
{"prompt_id": "eval_0", "judge_score": 0.85}
{"prompt_id": "eval_1", "judge_score": 0.72, "oracle_label": 0.70}
```

**Fields:**
- `prompt_id`: Identifies the prompt (optional — auto-generated from a `prompt` field's hash if missing; integer ids are coerced to strings)
- `judge_score`: **Required** — judge evaluation on any bounded scale
- `oracle_label`: Optional — ground truth for reward calibration (label 5–25% of rows; CJE needs ≥ 10 labeled rows pooled across policies)
- `response`: Optional — the generated text (required only when `include_response_length=True`)
- `draw_idx`: Optional — defaults to 0 (for multiple draws per prompt)
- `fold_id`: Optional — CV fold override
- `target_policy`: Optional in per-policy files (inferred from the filename); **required** per record in a single combined JSONL file
- `metadata`: Optional dict for per-response covariates

**Logprob fields are ignored.** 0.3.x-era logged data carries `base_policy_logprob` / `target_policy_logprobs`; in 0.4.0 (Direct-mode only) these are present-and-ignored, with an INFO note. Off-policy estimation lives on the frozen 0.3.x line (`pip install "cje-eval==0.3.*"`).

### Fresh-draws directory layout

One JSONL file per policy; the policy name comes from the filename:

```
responses/
├── model_a_responses.jsonl
├── model_b_responses.jsonl
└── model_c_responses.jsonl
```

File patterns searched per policy (in order): `{policy}_responses.jsonl`, `responses/{policy}_responses.jsonl`, `{policy}_fresh.jsonl`, `fresh_draws/{policy}.jsonl`.

### Calibration files (`calibration_data_path`)

A calibration file needs only judge + oracle pairs:

```json
{"prompt_id": "calib_0", "judge_score": 0.42, "oracle_label": 0.50}
{"prompt_id": "calib_1", "judge_score": 0.81, "oracle_label": 0.77}
```

No `prompt`, `response`, or logprob fields are required (they're accepted and ignored). Your old 0.3.x logged data works here as-is — the judge/oracle pairs are used to learn the judge→oracle mapping and everything else is ignored.

## Core Concepts

### 1. Type-Safe Data Models
- **Sample**: single observation (`prompt_id`, optional `prompt`/`response`, `judge_score`, `oracle_label`, metadata). Judge scores and oracle labels are validated to [0, 1] after normalization.
- **Dataset**: samples + `target_policies` (may be **empty** for calibration-only datasets, which carry no policy information).
- **FreshDrawSample / FreshDrawDataset**: per-policy fresh draws for estimation.
- **EstimationResult**: estimates, standard errors, diagnostics, metadata, and the fitted calibrator.

### 2. Loud Loading
Loaders fail with context rather than fabricate or silently skip:
- `FreshDrawLoader.load_from_jsonl` and `load_fresh_draws_auto` raise `ValueError` with `file:line` context on invalid records (blank lines are skipped, not errors).
- `DatasetLoader` filters invalid records with a counted warning and raises if **every** record was invalid.

### 3. Validation on Raw Records
`validate_direct_data` checks records exactly as `json.loads` produced them — no Dataset round-trip — so what it blesses is what the loaders accept.

## Common Interface

### Loading fresh draws

```python
from cje.data.fresh_draws import load_fresh_draws_auto

fresh_draws = load_fresh_draws_auto("responses/", policy="model_a")
print(fresh_draws.n_samples)
```

Or from memory:

```python
from cje.data import fresh_draws_from_dict

datasets, norm_info = fresh_draws_from_dict({
    "policy_a": [
        {"prompt_id": "q1", "judge_score": 0.85, "oracle_label": 0.9},
        {"prompt_id": "q2", "judge_score": 0.72},  # oracle_label optional
    ],
    "policy_b": [
        {"prompt_id": "q1", "judge_score": 0.70, "oracle_label": 0.75},
        {"prompt_id": "q2", "judge_score": 0.82},
    ],
})
datasets["policy_a"].n_samples   # 2
datasets["policy_a"].target_policy  # "policy_a"
```

(Most users should just pass `fresh_draws_data=...` to `analyze_dataset` and let it do this.)

### Loading a calibration file

```python
from cje import load_dataset_from_jsonl

dataset = load_dataset_from_jsonl("calibration.jsonl")
# Minimal judge+oracle files load fine: target_policies is empty
print(dataset.n_samples, dataset.target_policies)
```

### Working with EstimationResult

```python
from cje import analyze_dataset

result = analyze_dataset(fresh_draws_dir="responses/")

# Core data
estimates = result.estimates                    # numpy array
standard_errors = result.standard_errors        # numpy array
policies = result.metadata["target_policies"]   # list of policy names

# Confidence intervals (percentile bootstrap by default; t-based with
# finite-sample df under cluster_robust — df metadata only exists there)
ci_lower, ci_upper = result.confidence_interval(alpha=0.05)
cis = result.ci()  # [(lower, upper), ...]

# Compare two policies (paired when prompts are shared)
comparison = result.compare_policies(0, 1)
print(f"Difference: {comparison['difference']:.3f} (p={comparison['p_value']:.3f})")

# Export
result_dict = result.to_dict()

# Quick plotting (requires pip install "cje-eval[viz]")
result.plot_estimates(save_path="estimates.png")
```

Budget planning from a pilot:

```python
from cje import CostModel
from cje.data.fresh_draws import load_fresh_draws_auto

pilot_data = load_fresh_draws_auto("responses/", "base")
cost_model = CostModel(surrogate_cost=0.01, oracle_cost=0.16)  # real dollar costs
allocation = result.plan_allocation(budget=5000, cost_model=cost_model, fresh_draws=pilot_data)
print(allocation.summary())
```

In Jupyter, a result auto-displays as a formatted HTML table.

### Validating data

```python
import json
from cje.data import validate_direct_data

records = [json.loads(line) for line in open("responses/model_a_responses.jsonl")]
is_valid, findings = validate_direct_data(
    records, judge_field="judge_score", oracle_field="oracle_label"
)
for finding in findings:
    print(finding)  # entries starting with "Note:" are informational
```

Checks: `prompt_id` present and judge scores numeric over a leading window; oracle-label counts pooled across policies (0 or <10 labels is an issue, 10–49 an informational note); per-policy consistency when `target_policy` is present. Logprob fields earn only the note "logprob fields present and ignored (Direct mode)". The CLI equivalent is `cje validate PATH`.

### Response-level covariates

```python
from cje import analyze_dataset

# Auto-compute response_length (word count; requires a "response" field)
result = analyze_dataset(fresh_draws_dir="responses/", include_response_length=True)

# Or use custom metadata fields
result = analyze_dataset(fresh_draws_dir="responses/", calibration_covariates=["domain"])
```

Under the hood, `compute_response_covariates()` fills `FreshDrawSample.metadata`:

```python
from cje.data.fresh_draws import compute_response_covariates, load_fresh_draws_auto

fresh_draws = load_fresh_draws_auto("responses/", policy="model_a")
fresh_draws = compute_response_covariates(fresh_draws, covariate_names=["response_length"])
```

### Auto-normalization (label scale compatibility)

Any bounded scale works — 0–100, Likert 1–5, etc.:

```python
from cje import analyze_dataset

results = analyze_dataset(
    fresh_draws_data={
        "gpt-4o": [
            {"prompt_id": "1", "judge_score": 85, "oracle_label": 78},
            {"prompt_id": "2", "judge_score": 72, "oracle_label": 65},
        ],
    }
)

# Results come back in the ORIGINAL oracle scale (0-100 here, not [0,1])
print(results.metadata.get("normalization"))
```

If all values are already in [0, 1], nothing is transformed. Otherwise the range is auto-detected, data is normalized internally, and estimates/SEs/bootstrap CIs are inverse-transformed back to the original scale (recorded in `metadata["normalization"]`). Disable with `fresh_draws_from_dict(data, auto_normalize=False)`.

## Fold Management

`folds.py` provides one deterministic fold system used everywhere:

```python
import numpy as np
from cje.data.folds import get_fold, get_folds_for_dataset, get_folds_with_oracle_balance

fold = get_fold("prompt_123", n_folds=5, seed=42)          # 0-4, stable
folds = get_folds_for_dataset(dataset, n_folds=5, seed=42)
```

- **Deterministic**: `hash(prompt_id) % n_folds` — reproducible across runs
- **Filtering-proof**: based on stable prompt_ids, not array indices
- **Consistent**: calibration and estimation always agree on fold assignment

## Key Design Decisions

1. **Pydantic for type safety** — errors are caught at load time with clear messages, not deep inside estimation.
2. **Minimal required schema** — `prompt_id` + `judge_score` is a valid record; calibration files need only judge + oracle pairs. Everything else is optional, and unknown fields are preserved in `metadata`.
3. **Fail or filter, never fill** — loaders never fabricate missing values; invalid fresh-draw lines raise with file/line context.
4. **Logprobs ignored, not rejected** — 0.3.x logged data remains loadable (as a calibration source) without editing files.
5. **Validation mirrors loading** — `validate_direct_data` reads fields in the same top-level-then-metadata order as the loaders, so validation verdicts match loader behavior.

## Common Issues

### "No valid samples could be created from data"
Every record failed validation (the warnings above the error give per-record reasons). Common cause: judge scores outside the declared scale.

### "Invalid fresh draw record at file.jsonl:LINE"
A record is malformed (bad JSON, missing `judge_score`, out-of-range value). The message includes the exact file and line.

### "Inconsistent draws per prompt"
Some prompts have more draws than others for a policy. This is a warning, not an error — estimation handles unbalanced draws.

### Policy name mismatches
Policy names come from filenames (`{policy}_responses.jsonl`). Use identical names everywhere (`"gpt-4"` vs `"gpt4"` are different policies).

## Summary

The data module is the Direct-mode data boundary: minimal JSONL schemas in (fresh draws per policy, judge+oracle calibration pairs), type-safe models out, with loud failures, raw-record validation, and automatic scale handling in between.
