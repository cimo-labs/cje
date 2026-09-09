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
3. **Transportability testing**: Reserves probe prompt clusters before fitting, then uses their oracle labels to check calibration transfers

The probe files contain additional oracle labels for selected target-policy evaluation responses. Their prompt IDs can also appear in the labeled base-policy data, so separate files alone do **not** ensure independent calibration and validation. Exclude every reserved probe prompt ID from all calibration/evaluation inputs before fitting, as shown below. If using `logged_data.jsonl` as a calibration source, exclude those IDs there too.

The core notebook also reserves baseline monitoring rows before fitting. It splits the adversarial probe between policy audits and the three-week monitoring simulation, excludes monitoring IDs from the other policy probes, and uses different prompt clusters in each week. Small slices may correctly produce INCONCLUSIVE audits.

## Format

### Fresh Draws (`fresh_draws/*.jsonl`)

```json
{
  "prompt_id": "arena_0",
  "prompt": "User question",
  "response": "Model response",
  "judge_score": 0.85,
  "oracle_label": 0.86,
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
  "oracle_label": 0.0,
  "draw_idx": 0
}
```

### Logged Data (`logged_data.jsonl`)

Judge + oracle pairs from the base policy, plus 0.3.x-era logprob fields that 0.4.0 ignores:

```json
{
  "prompt": "User question",
  "response": "Base policy response",
  "base_policy_logprob": -60.88,
  "target_policy_logprobs": {"clone": -58.0},
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
ci_lower, ci_upper = results.confidence_interval()

for policy, est, se, lo, hi in zip(
    results.metadata["target_policies"],
    results.estimates,
    results.standard_errors,
    ci_lower,
    ci_upper,
):
    print(f"{policy}: {est:.3f} (SE {se:.3f}, 95% CI [{lo:.3f}, {hi:.3f}])")
```

### Using the logged data as a calibration source

```bash
cje analyze examples/arena_sample/fresh_draws --calibration-data examples/arena_sample/logged_data.jsonl
```

### Transportability Testing

```python
import json
from pathlib import Path
from cje import analyze_dataset
from cje.diagnostics import audit_transportability, plot_transport_comparison

root = Path("examples/arena_sample")
policies = ["base", "clone", "parallel_universe_prompt", "unhelpful"]

def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]

probes = {
    policy: read_jsonl(root / "probe_slice" / f"{policy}_probe.jsonl")
    for policy in policies if policy != "base"
}
heldout_ids = {row["prompt_id"] for rows in probes.values() for row in rows}
evaluation_data = {
    policy: [
        row for row in read_jsonl(root / "fresh_draws" / f"{policy}_responses.jsonl")
        if row["prompt_id"] not in heldout_ids
    ]
    for policy in policies
}
results = analyze_dataset(fresh_draws_data=evaluation_data)

# Predeclare one three-policy audit family and a practical margin in oracle units.
audits = {
    policy: audit_transportability(
        results.calibrator, rows, group_label=f"policy:{policy}",
        delta_max=0.05, family_size=len(probes),
    )
    for policy, rows in probes.items()
}
for audit in audits.values():
    print(audit.summary())

fig = plot_transport_comparison(audits)
audits["unhelpful"].plot()  # Residuals by score bin
```

The adversarial `unhelpful` policy is the point of this dataset: its judge scores look plausible, but the transport audit catches that the calibration learned on base-policy data does not hold for it.

## Data Source

This is a sample from the Chatbot Arena project, demonstrating real-world LLM evaluation data. Judge scores are from GPT-4.1-nano, oracle labels are from GPT-5.
