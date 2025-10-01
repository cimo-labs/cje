# Arena Sample Dataset

This directory contains a real-world sample dataset from Chatbot Arena for demonstrating CJE.

## Contents

- `dataset.jsonl` - Main dataset with 1000 samples
  - Logged responses from a base policy
  - Judge scores and oracle labels for calibration
  - Log probabilities for 2 target policies

- `responses/` - Fresh draws for doubly-robust estimation
  - `clone_responses.jsonl` - Responses from clone policy (1000 draws)
  - `parallel_universe_prompt_responses.jsonl` - Alternative prompt formulation (1000 draws)

## Format

### Main Dataset (`dataset.jsonl`)

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
  "metadata": {
    "judge_score": 0.85,
    "oracle_label": 0.7
  }
}
```

### Fresh Draws (`responses/*.jsonl`)

Each line contains a fresh response from a target policy:
```json
{
  "prompt_id": "arena_0",
  "response": "Fresh response text",
  "policy": "clone",
  "metadata": {
    "judge_score": 0.85
  }
}
```

## Usage

### In Examples

```python
from pathlib import Path
from cje import analyze_dataset

DATA_PATH = Path(__file__).parent / "arena_sample" / "dataset.jsonl"
results = analyze_dataset(str(DATA_PATH), estimator="calibrated-ips")
```

### In Tests

The test suite references this data via fixtures in `cje/tests/conftest.py`:
- `arena_sample` - Full 1000-sample dataset
- `arena_sample_small` - First 20 samples
- `arena_fresh_draws` - Fresh draw responses (clone and parallel_universe_prompt)
- `arena_calibrated` - Pre-calibrated version

## Data Source

This is a sample from the Chatbot Arena project, demonstrating real-world LLM evaluation data.
