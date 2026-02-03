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
