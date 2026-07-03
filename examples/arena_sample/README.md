# Arena Sample Dataset

This directory contains a real-world sample dataset from Chatbot Arena for demonstrating CJE.

## Contents

- `fresh_draws/` - Fresh responses from all policies (the evaluation data)
  - `base_responses.jsonl` - Base policy (1000 samples, 480 with oracle labels for calibration)
  - `clone_responses.jsonl` - Clone policy (1000 samples, no oracle)
  - `parallel_universe_prompt_responses.jsonl` - Alternative system prompt (1000 samples, no oracle)
  - `unhelpful_responses.jsonl` - Adversarial policy that fools the judge (1000 samples, no oracle)

- `probe_slice/` - Small oracle-labeled samples for transportability testing
  - `clone_probe.jsonl` - 50 samples with oracle labels
  - `parallel_universe_prompt_probe.jsonl` - 50 samples with oracle labels
  - `unhelpful_probe.jsonl` - 50 samples with oracle labels

- `logged_data.jsonl` - 0.3.x-era logged dataset from the base policy (1000 samples). Its judge scores and oracle labels make it usable in 0.4.0 as a **calibration source** (`--calibration-data`); its logprob fields are ignored (off-policy estimation lives on the 0.3.x line: `pip install "cje-eval==0.3.*"`).

## Data Structure

The data is structured to demonstrate the CJE workflow:

1. **Calibration training**: Uses oracle labels from `base_responses.jsonl` (~48% coverage)
2. **Policy estimation**: Uses all samples in `fresh_draws/` (judge scores only for target policies)
3. **Transportability testing**: Uses the held-out `probe_slice/` to verify calibration transfers

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

Judge + oracle pairs from the base policy, plus 0.3.x-era logprob fields that 0.4.0 ignores:

```json
{
  "prompt": "User question",
  "response": "Base policy response",
  "base_policy_logprob": -60.88,     // ignored in 0.4.0
  "target_policy_logprobs": { ... }, // ignored in 0.4.0
  "judge_score": 0.85,
  "oracle_label": 0.7,
  "metadata": {
    "prompt_id": "arena_123"
  }
}
```

## Usage

### Direct-mode analysis

```bash
cje validate examples/arena_sample/fresh_draws
cje analyze examples/arena_sample/fresh_draws
```

Or from Python:

```python
from cje import analyze_dataset

# CJE automatically uses the oracle labels in base_responses.jsonl for calibration
results = analyze_dataset(fresh_draws_dir="examples/arena_sample/fresh_draws")

for policy, est, se in zip(
    results.metadata["target_policies"],
    results.estimates,
    results.standard_errors
):
    print(f"{policy}: {est:.3f} ± {1.96*se:.3f}")
```

### Using the logged data as a calibration source

```bash
cje analyze examples/arena_sample/fresh_draws --calibration-data examples/arena_sample/logged_data.jsonl
```

### Transportability Testing

```python
import json
from cje import analyze_dataset
from cje.diagnostics import audit_transportability, plot_transport_comparison

# Get calibrator from analysis
results = analyze_dataset(fresh_draws_dir="examples/arena_sample/fresh_draws")

# Load probe as list of dicts (no wrapper needed!)
probe = [json.loads(line) for line in open("examples/arena_sample/probe_slice/unhelpful_probe.jsonl")]

# Run canonical transportability audit
diag = audit_transportability(results.calibrator, probe, group_label="policy:unhelpful")
print(diag.summary())
# Transport: FAIL | Group: policy:unhelpful | N=50 | δ̂: -0.275 (CI: [-0.320, -0.231]) | Action: refit_two_stage

# Visualize
diag.plot()  # Decile-level residuals

# Or compare all policies at once
audits = {}
for policy in ["clone", "parallel_universe_prompt", "unhelpful"]:
    probe = [json.loads(line) for line in open(f"examples/arena_sample/probe_slice/{policy}_probe.jsonl")]
    audits[policy] = audit_transportability(results.calibrator, probe, group_label=f"policy:{policy}")

fig = plot_transport_comparison(audits)
```

The adversarial `unhelpful` policy is the point of this dataset: its judge scores look plausible, but the transport audit catches that the calibration learned on base-policy data does not hold for it.

## Data Source

This is a sample from the Chatbot Arena project, demonstrating real-world LLM evaluation data. Judge scores are from GPT-4.1-nano, oracle labels are from GPT-5.
