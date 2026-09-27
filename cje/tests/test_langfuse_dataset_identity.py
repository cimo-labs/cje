"""Missing native versions require a separate, explicit frozen-content proof."""

from typing import Any

from copy import deepcopy

import pytest

from cje.bridges.langfuse import digest, prepare
from cje.tests.test_langfuse_bridge import STAMP, snapshot


def exported_panel() -> Any:
    data, cfg = snapshot()
    for rows in data["items"].values():
        for row in rows:
            row["experimentItemVersion"] = None
            row["expectedOutput"] = ""
    manifest = {
        "schema": 1,
        "source": "Langfuse SDK get_dataset at the pinned version",
        "project_id": cfg["project_id"],
        "dataset_id": cfg["dataset_id"],
        "requested_version": STAMP,
        "items": [
            {
                "id": row["experimentItemId"],
                "datasetId": cfg["dataset_id"],
                "input": row["input"],
                "expectedOutput": None,
                "metadata": {},
                "status": "ACTIVE",
                "createdAt": STAMP,
                "updatedAt": STAMP,
            }
            for row in data["items"]["A"]
        ],
    }
    return data, cfg, manifest


def test_content_identity_preserves_null_versions_and_raw_export() -> None:
    data, cfg, manifest = exported_panel()
    before = deepcopy(data)
    prepared = prepare(data, cfg, dataset_manifest=manifest)
    assert data == before
    assert prepared["provenance"]["snapshot_sha256"] == digest(before)
    assert prepared["provenance"]["dataset_manifest_sha256"] == digest(manifest)
    for rows in prepared["fresh_draws_data"].values():
        assert all(r["metadata"]["dataset_item_version"] is None for r in rows)
        assert all(
            r["metadata"]["dataset_item_identity"]["basis"] == "frozen_dataset_content"
            for r in rows
        )
        assert "oracle_label" not in rows[1]


def test_null_version_without_dataset_export_still_fails() -> None:
    data, cfg, _ = exported_panel()
    with pytest.raises(ValueError, match="item version"):
        prepare(data, cfg)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda d, c, m: m.update(project_id="foreign"), "ownership"),
        (lambda d, c, m: m.update(source="unverified"), "source"),
        (lambda d, c, m: m["items"].pop(), "population"),
        (lambda d, c, m: m["items"].append(deepcopy(m["items"][0])), "Duplicate"),
        (lambda d, c, m: m["items"][0].update(datasetId="foreign"), "foreign"),
        (lambda d, c, m: m["items"][0].update(status="ARCHIVED"), "inactive"),
        (
            lambda d, c, m: m["items"][0].update(updatedAt="2026-09-02T00:00:00Z"),
            "newer",
        ),
        (
            lambda d, c, m: m.update(requested_version="2026-09-02T00:00:00Z"),
            "after the response",
        ),
        (lambda d, c, m: m.update(requested_version="2026-09-01T00:00:00"), "timezone"),
        (lambda d, c, m: d["items"]["A"][0].update(input="changed"), "input differs"),
        (
            lambda d, c, m: d["items"]["A"][0].update(expectedOutput="invented answer"),
            "Expected output differs",
        ),
        (
            lambda d, c, m: d["items"]["A"][0].update(experimentItemVersion=" "),
            "item version",
        ),
        (
            lambda d, c, m: d["items"]["A"][0].update(experimentItemVersion=STAMP),
            "version differs",
        ),
    ],
)
def test_dataset_content_proof_rejects_changed_or_unfrozen_data(
    mutation: Any, message: Any
) -> None:
    data, cfg, manifest = exported_panel()
    mutation(data, cfg, manifest)
    with pytest.raises(ValueError, match=message):
        prepare(data, cfg, dataset_manifest=manifest)
