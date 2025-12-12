# Arena Sample Dataset

This directory contains a real-world sample dataset from Chatbot Arena for demonstrating CJE.

## Contents

- `logged_data.jsonl` - Logged data from base/production policy (1000 samples)
  - Logged responses from the base policy
  - Judge scores and oracle labels for calibration
  - Log probabilities under target policies (for importance weighting)

- `fresh_draws/` - Fresh responses from all policies (for estimation)
  - `base_responses.jsonl` - Base policy (1000 samples, 480 with oracle for calibration)
  - `clone_responses.jsonl` - Clone policy (1000 samples, no oracle)
  - `parallel_universe_prompt_responses.jsonl` - Alternative prompt (1000 samples, no oracle)
  - `unhelpful_responses.jsonl` - Adversarial policy (1000 samples, no oracle)

- `probe_slice/` - Small oracle-labeled samples for transportability testing
  - `clone_probe.jsonl` - 50 samples with oracle labels
  - `parallel_universe_prompt_probe.jsonl` - 50 samples with oracle labels
  - `unhelpful_probe.jsonl` - 50 samples with oracle labels

## Data Structure

The data is structured to demonstrate the CJE workflow:

1. **Calibration training**: Uses oracle labels from `base_responses.jsonl` (~48% coverage)
2. **Policy estimation**: Uses all samples in `fresh_draws/` (judge scores only for target policies)
3. **Transportability testing**: Uses separate `probe_slice/` to verify calibration transfers

This separation ensures calibration is trained only on base policy data, and probe samples are held out for transportability validation.

## Format

### Fresh Draws (`fresh_draws/*.jsonl`)

```json
{
  "prompt_id": "arena_0",
  "prompt": "User question",
  "response": "Model response",
  "judge_score": 0.85,
  "oracle_label": 0.86,  // Only in base_responses.jsonl
  "draw_idx": 0
}
```

### Probe Slice (`probe_slice/*.jsonl`)

```json
{
  "prompt_id": "arena_916",
  "prompt": "User question",
  "response": "Model response",
  "judge_score": 0.1,
  "oracle_label": 0.0,  // All probe samples have oracle labels
  "draw_idx": 0
}
```

### Logged Data (`logged_data.jsonl`)

```json
{
  "prompt": "User question",
  "response": "Base policy response",
  "base_policy_logprob": -60.88,
  "target_policy_logprobs": {
    "clone": -60.88,
    "parallel_universe_prompt": -59.75,
    "unhelpful": -120.5
  },
  "judge_score": 0.85,
  "oracle_label": 0.7,
  "metadata": {
    "prompt_id": "arena_123"
  }
}
```

## Usage

### Direct Mode (Recommended for demos)

```python
from cje import analyze_dataset

# CJE automatically uses oracle labels from base for calibration
results = analyze_dataset(
    fresh_draws_dir="arena_sample/fresh_draws",
    estimator="auto"
)

# Results include calibrated estimates for all policies
for policy, est, se in zip(
    results.metadata["target_policies"],
    results.estimates,
    results.standard_errors
):
    print(f"{policy}: {est:.3f} ± {1.96*se:.3f}")
```

### Transportability Testing

```python
import json
from cje.diagnostics import audit_transportability, plot_transport_comparison

# Load probe as list of dicts (no wrapper needed!)
probe = [json.loads(line) for line in open("arena_sample/probe_slice/unhelpful_probe.jsonl")]

# Run canonical transportability audit
diag = audit_transportability(calibrator, probe, group_label="policy:unhelpful")
print(diag.summary())
# Transport: FAIL | Group: policy:unhelpful | N=50 | δ̂: -0.275 (CI: [-0.320, -0.231]) | Action: refit_two_stage

# Visualize
diag.plot()  # Decile-level residuals

# Or compare all policies at once
results = {}
for policy in ["clone", "parallel_universe_prompt", "unhelpful"]:
    probe = [json.loads(line) for line in open(f"arena_sample/probe_slice/{policy}_probe.jsonl")]
    results[policy] = audit_transportability(calibrator, probe, group_label=f"policy:{policy}")

fig = plot_transport_comparison(results)
```

### IPS/DR Modes

```python
# IPS mode: logged data only
results = analyze_dataset(
    logged_data_path="arena_sample/logged_data.jsonl",
    estimator="calibrated-ips"
)

# DR mode: logged data + fresh draws (most accurate)
results = analyze_dataset(
    logged_data_path="arena_sample/logged_data.jsonl",
    fresh_draws_dir="arena_sample/fresh_draws",
    estimator="stacked-dr"
)
```

## Data Source

This is a sample from the Chatbot Arena project, demonstrating real-world LLM evaluation data. Judge scores are from GPT-4.1-nano, oracle labels are from GPT-5.
