# CJE bridges (Promptfoo, TruLens)

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

---

## Why this exists

When teams adopt CJE, the first friction point is almost always **data plumbing**:
- “Our eval runner is Promptfoo, can we use that output?”
- “We’re scoring with TruLens feedback functions, can we calibrate those?”

These bridges keep the answer lightweight: yes — export JSON/DB → run converter → label an oracle slice → run CJE.
