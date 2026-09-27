# Langfuse experiments to CJE

Validate two experiments against the same declared dataset population, then write CJE `fresh_draws_data` and a readiness report. The converter preserves responses without human labels. It does not estimate policy quality or certify a labeling design.

The pure converter ships in the CJE wheel as `cje.bridges.langfuse.prepare`. The companion command in this directory runs from a source checkout and needs `httpx` only for online reads:

```sh
python -m pip install -e .
python -m pip install 'httpx>=0.28.1,<0.29'
```

## Select the runs and scores

Create `config.json` using your project, dataset, experiment, and item IDs:

```json
{
  "project_id": "your-project-id",
  "dataset_id": "your-dataset-id",
  "dataset_name": "your-dataset-name",
  "dataset_version": "2026-09-01T00:00:00Z",
  "from_start_time": "2026-09-01T00:00:00Z",
  "to_start_time": "2026-09-02T00:00:00Z",
  "policies": {"baseline": "baseline-experiment-id", "candidate": "candidate-experiment-id"},
  "expected_prompt_ids": ["dataset-item-1", "dataset-item-2"],
  "judge": {"name": "quality", "source": "EVAL", "config_id": null, "scale": [1, 5]},
  "oracle": {"name": "human-quality", "source": "ANNOTATION", "config_id": "human-rubric-config-id", "scale": [1, 5]}
}
```

Declare the full intended population rather than deriving it from whichever responses happened to export successfully. Each experiment must contain exactly one finished response per declared item. The time window is timezone aware and limits experiment selection; later human annotations are still collected. `config_id: null` deliberately selects scores with no score configuration. Score names, ingestion sources, and configuration IDs must match exactly.

`dataset_version` is the timestamp at which the dataset was frozen, at or before response generation. With `dataset_name`, it enables a separate read of every dataset item at that version. Native experiment exports may contain `experimentItemVersion: null`; these stay null. The converter verifies the separately frozen input and expected output and records `frozen_dataset_content` identity, its content hash, and its manifest hash. It does not invent a native version. If every response has a matching native version, you may omit `dataset_name` and `dataset_version`.

## Export once; resume offline

Supply `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY` through your environment or credential manager. Do not put keys in the configuration or command arguments. Use an HTTPS base URL for the correct region or self-hosted instance.

```sh
python scripts/cje_bridges/convert.py langfuse \
  --config config.json --out-dir export-01 --online
```

The command only sends GET requests. It reads all experiment pages, batches score queries in groups of 50 trace IDs, and exhausts each Scores v3 cursor. It honors numeric `Retry-After` values up to 300 seconds, with at most five attempts. Longer delays or exhausted retries fail visibly. A failed collection restarts that incomplete read; it never labels a partial export complete. Langfuse reads are not an atomic database snapshot: finish the runs and annotations before collecting. Use a new directory to refresh an export.

Complete raw reads are saved before conversion. If validation stops, inspect the error and rerun from the saved reads without `--online`:

```sh
python scripts/cje_bridges/convert.py langfuse \
  --config config.json --out-dir export-01
```

To start from a separately saved export, add `--snapshot snapshot.json`. Supply a missing frozen dataset proof with `--dataset-manifest dataset-manifest.json`, or resume online if the config declares its name and version. A supplied dataset manifest is trusted evidence from the caller; its hash provides content identity, not authentication of the source. Supported manifests have schema 1, project/dataset IDs, `requested_version`, complete ACTIVE `items`, and source `Langfuse dataset-items API at the pinned version` (or the equivalent SDK source described below).

Files are written atomically with private permissions. An identical rerun is a no-op. Changed inputs or results require a new output directory; existing evidence is never overwritten. The command creates:

- `snapshot.json`: complete raw experiment items and separately read scores, with the exact config in `selection` and `schema: 1`.
- `dataset-manifest.json`: frozen dataset content when supplied or collected.
- `oracle-manifest.json`: verified reference manifest for imported human labels, when applicable.
- `fresh-draws.json`: the dictionary accepted by CJE's Python API.
- `readiness.json`: counts, declared scales, hashes, identity and label provenance, and unresolved sampling/independence/audit decisions. `PREPARED` means conversion passed; estimation remains `NOT_RUN`.

## Imported human labels

Langfuse's `source` identifies the ingestion route. An API score named "human" does not establish human authorship. For published or externally collected human ratings, set the oracle selector to `source: "API"`, `label_origin: "imported_human"`, and `manifest_sha256`. Pass `--oracle-manifest human-reference.json`.

The trusted reference manifest must contain:

```json
{
  "schema": 1,
  "label_origin": "human",
  "scale": [1, 5],
  "rubric": "Your documented human rating rubric",
  "aggregation": "One rating per response",
  "source_url": "https://example.org/human-study",
  "source_revision": "immutable-source-revision",
  "source_sha256": "64 lowercase hexadecimal characters",
  "records": [
    {"record_id": "rating-1", "policy": "baseline", "prompt_id": "dataset-item-1",
     "output_sha256": "canonical JSON hash of the exact response output", "value": 4}
  ]
}
```

Compute manifest and output hashes with `cje.bridges.langfuse.digest`: SHA-256 of UTF-8 JSON with sorted keys, compact separators, ASCII escaping, and no nonfinite numbers. Each imported score must carry `metadata.cje_human_import` with the same `manifest_sha256` and its original `record_id`. The converter checks response identity, output hash, value, scale, uniqueness, and completeness against that manifest. Record which people or study produced the source labels; a hash alone cannot prove human authorship. API scores without this evidence are rejected as oracle labels. The companion command does not upload labels.

Native `ANNOTATION` scores retain author and annotation-queue IDs when available. Multiple matching human scores are rejected: predeclare and perform the rater/revision aggregation before conversion. Blank or missing human scores remain absent; zero remains a label. Booleans, strings, nonfinite values, and scores outside the declared scale fail.

## Use the prepared data

```python
import json
from cje import analyze_dataset
from cje.bridges.langfuse import prepare

# Alternatively call prepare(snapshot, config, oracle_manifest=..., dataset_manifest=...)
# directly. It performs no I/O and does not mutate its inputs.
config = json.load(open("config.json", encoding="utf-8"))
fresh = json.load(open("export-01/fresh-draws.json", encoding="utf-8"))

# Before this call, confirm representative labeling and separate any held-out
# transport-audit rows. Do not fit the calibrator on its audit labels.
result = analyze_dataset(
    fresh_draws_data=fresh,
    fresh_judge_scale=tuple(config["judge"]["scale"]),
    fresh_oracle_scale=tuple(config["oracle"]["scale"]),
    output_scale=tuple(config["oracle"]["scale"]),
)
print(result.summary())
```

Inspect CJE's calibration, uncertainty, and transport diagnostics before making a policy decision. Two rows sharing a prompt ID are paired across policies; this does not prove independence among different dataset items. This initial importer supports two policies and observation-level scores. Trace scores, repeated responses per dataset item, ambiguous score joins, partial populations, and silent score aggregation are intentionally rejected. A missing expected answer exported as `""` may match a frozen dataset's `null`; this explicitly represents absence of an expected answer, never a human score.

Dataset manifests exported using SDK `get_dataset(name, version=...)` can also use source `Langfuse SDK get_dataset at the pinned version`. Items must retain their native `id`, `datasetId`, `status`, `input`, `expectedOutput`, `metadata`, `createdAt`, and `updatedAt`. Mixed native-version and content-based identities for the same prompt across policies fail.

API contracts: [Experiments](https://langfuse.com/docs/api-and-data-platform/features/experiments-api), [Scores v3](https://langfuse.com/docs/api-and-data-platform/features/scores-api), [rate limits](https://langfuse.com/faq/all/api-limits), [score provenance](https://langfuse.com/docs/evaluation/scores/data-model).
