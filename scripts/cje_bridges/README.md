# CJE bridges (Promptfoo, TruLens, LangSmith)

This folder is a **thin convenience wrapper** around the standalone converters in this repo.

Goal: make it easy to go from “I already have eval results in tool X” → **CJE `fresh_draws_data` JSON** (plus an oracle labeling template).

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

Full help:
```bash
python3 scripts/langsmith_cje/langsmith_to_cje.py --help
```

---

## Why this exists

When teams adopt CJE, the first friction point is almost always **data plumbing**:
- “Our eval runner is Promptfoo, can we use that output?”
- “We’re scoring with TruLens feedback functions, can we calibrate those?”
- “We already evaluate everything in LangSmith — can we export it?”

These bridges keep the answer lightweight: yes — export JSON/DB/API → run converter → label an oracle slice → run CJE.
