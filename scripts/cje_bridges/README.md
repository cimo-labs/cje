# CJE bridges (Promptfoo, TruLens, LangSmith, OpenCompass)

This folder is a **thin convenience wrapper** around the standalone converters in this repo.

Goal: make it easy to go from “I already have eval results in tool X” → **CJE `fresh_draws_data` JSON** (plus an oracle labeling template).

> These converters are included in the GitHub repo under `scripts/` and are not shipped as part of the PyPI wheel.
> Run the commands below from the repo root (after `git clone https://github.com/cimo-labs/cje.git && cd cje`).

Refs:
- CJE paper: https://arxiv.org/abs/2512.11150
- CJE package: `pip install cje-eval`

---

## Quickstart

### Promptfoo → CJE

```bash
python3 scripts/cje_bridges/convert.py promptfoo results.json \
  --out cje_fresh_draws_data.json \
  --label-template oracle_label_template.csv
```

After you fill in `oracle_label_template.csv`, re-run to embed oracle labels in the JSON:

```bash
python3 scripts/cje_bridges/convert.py promptfoo results.json \
  --oracle-labels oracle_label_template.csv \
  --out cje_fresh_draws_data_with_oracle.json \
  --no-label-template
```

This supports Promptfoo’s common JSON shapes, including `promptfoo export` wrapper output.

Full help:
```bash
python3 scripts/promptfoo_cje/promptfoo_to_cje.py --help
```

### TruLens → CJE

Install TruLens first:
```bash
pip install trulens
```

Then:
```bash
python3 scripts/cje_bridges/convert.py trulens \
  --database-url sqlite:///default.sqlite \
  --judge-col "Answer Relevance" \
  --out cje_fresh_draws_data.json \
  --label-template oracle_label_template.csv
```

After you fill in `oracle_label_template.csv`, re-run to embed oracle labels in the JSON:

```bash
python3 scripts/cje_bridges/convert.py trulens \
  --database-url sqlite:///default.sqlite \
  --judge-col "Answer Relevance" \
  --oracle-labels oracle_label_template.csv \
  --out cje_fresh_draws_data_with_oracle.json \
  --no-label-template
```

Full help:
```bash
python3 scripts/trulens_cje/trulens_to_cje.py --help
```

### LangSmith → CJE

Install LangSmith first:
```bash
pip install langsmith
export LANGSMITH_API_KEY=ls_...
# optionally:
# export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Then (example: one LangSmith project per policy/model):
```bash
python3 scripts/cje_bridges/convert.py langsmith \
  --project "my_model_a_project" \
  --project "my_model_b_project" \
  --feedback-key "correctness" \
  --out cje_fresh_draws_data.json \
  --label-template oracle_label_template.csv
```

Notes:
- By default we use `reference_example_id` as `prompt_id` when available, which helps align runs
  across policies when they were generated from the same dataset.
- If you already logged human labels into LangSmith as feedback, you can pass `--oracle-feedback-key`
  to populate `oracle_label` directly.
- The CSV template is for convenience when collecting labels, but this script currently only ingests
  oracle labels via LangSmith feedback (`--oracle-feedback-key`). If you label outside LangSmith,
  you’ll need to merge labels into the exported JSON yourself or upload them back to LangSmith as feedback.

Full help:
```bash
python3 scripts/langsmith_cje/langsmith_to_cje.py --help
```

### OpenCompass → CJE

OpenCompass supports LLM-as-judge evaluation (e.g. `GenericLLMEvaluator`) and can optionally emit per-sample outputs via `--dump-eval-details`.

In the OpenCompass docs, the `--dump-eval-details` output is a JSON dict with a top-level `details` list; each record commonly includes fields like `origin_prompt` and `prediction` (A/B).

Once you have a per-sample output JSON (typically under `output/.../results/.../*.json`), convert it:

```bash
python3 scripts/cje_bridges/convert.py opencompass path/to/opencompass_results.json \
  --out cje_fresh_draws_data.json \
  --label-template oracle_label_template.csv
```

After you fill in `oracle_label_template.csv`, re-run to embed oracle labels in the JSON:

```bash
python3 scripts/cje_bridges/convert.py opencompass path/to/opencompass_results.json \
  --oracle-labels oracle_label_template.csv \
  --out cje_fresh_draws_data_with_oracle.json \
  --no-label-template
```

Notes:
- OpenCompass JSON schemas vary across evaluators/datasets. This converter is best-effort.
- If your file uses different keys, pass overrides:
  - `--prompt-field <key>`
  - `--prediction-field <key>`

Full help:
```bash
python3 scripts/opencompass_cje/opencompass_to_cje.py --help
```

---

## Why this exists

When teams adopt CJE, the first friction point is almost always **data plumbing**:
- “Our eval runner is Promptfoo, can we use that output?”
- “We’re scoring with TruLens feedback functions, can we calibrate those?”
- “We already evaluate everything in LangSmith — can we export it?”
- “We’re running LLM-as-judge in OpenCompass — can we reuse that output?”

These bridges keep the answer lightweight: yes — export JSON/DB/API → run converter → label an oracle slice → run CJE.

## Label identity and sparse labeling

Each exported response now has two identifiers:

- `prompt_id` identifies the evaluation prompt. It remains shared across policies and repeated draws so CJE can cluster and pair observations correctly.
- `response_id` identifies one response to label. The CSV template and exported JSON retain it. Two responses to the same prompt must have different response IDs, even when their answer text is identical.

For example, a Promptfoo template can contain:

```csv
provider_id,prompt_id,response_id,oracle_label
model_a,question_1,response_1,0
model_a,question_1,response_2,1
model_a,question_2,response_3,
```

Fill the oracle labels for your selected slice and leave the remaining cells blank. Blank or whitespace-only labels remain absent, and numeric `0` remains a label. Keep the identifier columns unchanged. CSV and JSONL imports reject nonfinite values (`NaN`, `Infinity`), booleans, malformed nonblank labels, duplicate label keys, and labels that do not match the exported responses. Errors identify the source file and line.

Promptfoo uses the upstream result `id` when available; TruLens uses `record_id`; LangSmith uses its run ID; OpenCompass uses `response_id` or `record_id` when available. Otherwise a deterministic content hash identifies the response, with an occurrence suffix for identical repeats. Keep the original export: distinct response contents can be reordered, but identical responses without upstream IDs must retain their original order. Re-exporting a changed run or dataset is not a substitute for the source used to generate the label template. Duplicate upstream response IDs fail rather than silently merging records.

### Migrating older label files

Older templates without `response_id` still work when a labeled `(policy, prompt_id)` pair matches exactly one response. A prompt-only label is rejected if that policy has multiple draws for the prompt. Regenerate the template and associate each existing label with the correct response; do not copy one label onto every draw. Duplicate keys are rejected even if they repeat the same numeric label.

The policy column remains `provider_id` for Promptfoo, `policy_id` for TruLens, and `policy_name` for OpenCompass. LangSmith continues to read oracle feedback by run ID using `--oracle-feedback-key`; it does not import edited CSV templates.

## OpenCompass decisions and directory layouts

The decision parser accepts numeric scores, its documented exact tokens, and explicit final decisions such as `Answer: A`, `(B)`, `choice=A`, or `Final answer: [B].`. It drops ambiguous prose such as `This is a tie.`, `Neither A nor B.`, or an explanation mentioning both options. Use `--prediction-field` to select a structured final-decision field when your evaluator provides one. An export where all rows are unparsed returns a nonzero status.

Without `--policy-name`, the JSON filename stem identifies the policy. Directory conversion fails if two files have the same stem, because silently combining `model_a/benchmark.json` and `model_b/benchmark.json` would erase model identity. Convert each model separately with its explicit name:

```bash
python3 scripts/cje_bridges/convert.py opencompass results/model_a \
  --policy-name model_a --out model_a.json --label-template model_a_labels.csv
python3 scripts/cje_bridges/convert.py opencompass results/model_b \
  --policy-name model_b --out model_b.json --label-template model_b_labels.csv
```

Use one `--policy-name` for multiple input files only when they really are responses from one policy. When combining separately converted JSON objects in Python, reject overlapping policy keys rather than overwriting one export.

## Run the converted data through CJE

Bridge JSON files contain an API `fresh_draws_data` dictionary. Load that dictionary directly; the `cje analyze` CLI expects JSONL records or a directory of per-policy JSONL files.

```python
import json
from cje import analyze_dataset

with open("cje_fresh_draws_data_with_oracle.json", encoding="utf-8") as stream:
    fresh_draws = json.load(stream)
results = analyze_dataset(fresh_draws_data=fresh_draws)
print(results.summary())
```

Promptfoo, TruLens, and LangSmith also support `--run-cje`. It returns a nonzero status if oracle labels are absent or the requested analysis fails, so shell and CI callers can detect failure. A successful conversion file may already exist if the subsequent analysis fails. `--run-cje` is a convenience check; inspect the calibration, reliability and transport diagnostics before interpreting estimates as validated oracle-scale comparisons. Conversion itself does not establish that a labeled slice is representative or held out from a transport audit.
