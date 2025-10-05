# Arena Sample Dataset

This directory contains a real-world sample dataset from Chatbot Arena for demonstrating CJE.

## Contents

- `logged_data.jsonl` - Logged data from base/production policy (1000 samples)
  - Logged responses from the base policy
  - Judge scores and oracle labels for calibration
  - Log probabilities under target policies (for importance weighting)

- `fresh_draws/` - Teacher-forced responses from target policies (for DR/Direct estimation)
  - `clone_responses.jsonl` - Responses from clone policy (1000 draws, 50% oracle coverage)
  - `parallel_universe_prompt_responses.jsonl` - Alternative prompt formulation (1000 draws, 50% oracle coverage)
  - `unhelpful_responses.jsonl` - Intentionally poor responses (1000 draws, 50% oracle coverage)
  - All fresh draws share the same 1000 prompts (matching logged_data.jsonl)

## Format

### Logged Data (`logged_data.jsonl`)

Each line is a JSON object with:
```json
{
  "prompt": "User question",
  "response": "LLM response from base policy",
  "prompt_id": "unique_identifier",
  "base_policy_logprob": -60.88,
  "target_policy_logprobs": {
    "clone": -60.88,
    "parallel_universe_prompt": -59.75
  },
  "judge_score": 0.85,
  "oracle_label": 0.7,
  "metadata": {
    "prompt_id": "arena_123"
  }
}
```

### Fresh Draws (`fresh_draws/*.jsonl`)

Each line contains a teacher-forced response from a target policy:
```json
{
  "prompt_id": "arena_0",
  "response": "Fresh response text",
  "policy": "clone",
  "judge_score": 0.85,
  "oracle_label": 0.86,  // 50% of samples have oracle labels for calibration
  "metadata": {
    "judge_model": "gpt-4"
  }
}
```

**Oracle coverage**: 50% of samples in each file have `oracle_label` values. This enables:
- Direct mode to learn calibration without logged data
- Cross-validation during calibration
- Proper uncertainty quantification

## Usage

### In Examples

```python
from pathlib import Path
from cje import analyze_dataset

# IPS mode: logged data only
LOGGED_DATA = Path(__file__).parent / "arena_sample" / "logged_data.jsonl"
results = analyze_dataset(logged_data_path=str(LOGGED_DATA), estimator="calibrated-ips")

# DR mode: logged data + fresh draws
FRESH_DRAWS = Path(__file__).parent / "arena_sample" / "fresh_draws"
results = analyze_dataset(
    logged_data_path=str(LOGGED_DATA),
    fresh_draws_dir=str(FRESH_DRAWS),
    estimator="stacked-dr"
)

# Direct mode: fresh draws only (learns calibration from oracle labels in fresh draws)
results = analyze_dataset(fresh_draws_dir=str(FRESH_DRAWS), estimator="auto")
```

### In Tests

The test suite references this data via fixtures in `cje/tests/conftest.py`:
- `arena_sample` - Full 1000-sample dataset
- `arena_sample_small` - First 20 samples
- `arena_fresh_draws` - Fresh draw responses (clone and parallel_universe_prompt)
- `arena_calibrated` - Pre-calibrated version

## Data Source

This is a sample from the Chatbot Arena project, demonstrating real-world LLM evaluation data.
