from typing import Any
from copy import deepcopy

import pytest

from cje.bridges.langfuse import digest, prepare
from cje.tests.test_langfuse_bridge import snapshot


def imported_snapshot() -> Any:
    data, cfg = snapshot()
    manifest: dict[str, Any] = {
        "schema": 1,
        "label_origin": "human",
        "scale": [1, 5],
        "rubric": "Synthetic contract-test rubric",
        "aggregation": "one test rater",
        "source_url": "https://example.invalid/reference",
        "source_revision": "fixture-v1",
        "source_sha256": "a" * 64,
        "records": [],
    }
    for policy, rows in data["items"].items():
        row = rows[0]
        manifest["records"].append(
            {
                "record_id": f"record-{policy}",
                "policy": policy,
                "prompt_id": row["experimentItemId"],
                "output_sha256": digest(row["output"]),
                "value": 4,
            }
        )
    cfg["oracle"].update(
        source="API", label_origin="imported_human", manifest_sha256=digest(manifest)
    )
    data["selection"] = deepcopy(cfg)
    for score in data["scores"]:
        if score["name"] == "human-coherence":
            policy = score["subject"]["id"].split("-")[1]
            score.update(
                source="API",
                metadata={
                    "cje_human_import": {
                        "manifest_sha256": digest(manifest),
                        "record_id": f"record-{policy}",
                    }
                },
            )
    return data, cfg, manifest


def test_verified_import_keeps_sparse_labels_and_provenance() -> None:
    data, cfg, manifest = imported_snapshot()
    result = prepare(data, cfg, oracle_manifest=manifest)
    assert result["provenance"]["human_label_origin"] == "imported_human"
    for policy, rows in result["fresh_draws_data"].items():
        assert rows[0]["oracle_label"] == 4
        assert (
            rows[0]["metadata"]["human_provenance"]["record_id"] == f"record-{policy}"
        )
        assert "oracle_label" not in rows[1]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d, c, m: d["scores"][-1].update(value=3),
        lambda d, c, m: d["scores"][-1].update(value=True),
        lambda d, c, m: d["scores"][-1].update(metadata={}),
        lambda d, c, m: d["scores"][-1]["metadata"]["cje_human_import"].update(
            record_id="record-A"
        ),
        lambda d, c, m: d["items"]["B"][0].update(output="changed output"),
        lambda d, c, m: d["scores"].pop(),
        lambda d, c, m: m["records"][0].update(value=2),
        lambda d, c, m: m.update(label_origin="model"),
        lambda d, c, m: c["oracle"].update(manifest_sha256="wrong"),
    ],
)
def test_import_rejects_mismatched_reference_or_score(mutation: Any) -> None:
    data, cfg, manifest = imported_snapshot()
    mutation(data, cfg, manifest)
    data["selection"] = deepcopy(cfg)
    with pytest.raises(ValueError):
        prepare(data, cfg, oracle_manifest=manifest)


def test_api_name_alone_cannot_establish_human_origin() -> None:
    data, cfg, _ = imported_snapshot()
    del cfg["oracle"]["label_origin"]
    data["selection"] = deepcopy(cfg)
    with pytest.raises(ValueError, match="ANNOTATION"):
        prepare(data, cfg)


def test_missing_manifest_is_rejected() -> None:
    data, cfg, _ = imported_snapshot()
    with pytest.raises(ValueError, match="pinned reference manifest"):
        prepare(data, cfg)


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m["records"].append(deepcopy(m["records"][0])),
        lambda m: m["records"][0].update(policy="outside"),
        lambda m: m.update(source_sha256="not-a-hash"),
        lambda m: m.update(scale=[0, 1]),
        lambda m: m.update(label_origin="model"),
    ],
)
def test_repinning_cannot_bypass_manifest_contract(change: Any) -> None:
    data, cfg, manifest = imported_snapshot()
    change(manifest)
    cfg["oracle"]["manifest_sha256"] = digest(manifest)
    data["selection"] = deepcopy(cfg)
    with pytest.raises(ValueError):
        prepare(data, cfg, oracle_manifest=manifest)
