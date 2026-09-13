"""Validate raw Langfuse experiment/score exports for CJE.

One response per frozen dataset item per policy. This module performs no I/O,
never infers human authorship from a score name, and does not run estimation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from typing import Any


def stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(stable(value).encode()).hexdigest()


def identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: missing string identity")
    return value


def numeric(score: dict, selector: dict) -> float:
    value = score.get("value")
    if (
        score.get("dataType") != "NUMERIC"
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"score {score.get('id')}: expected a finite NUMERIC value")
    low, high = selector["scale"]
    if not low <= value <= high:
        raise ValueError(f"score {score.get('id')}: outside the declared scale")
    return float(value)


def imported_label_index(manifest: dict | None, config: dict) -> dict:
    """Resolve a caller-pinned reference manifest, never trust score metadata alone."""
    selector = config["oracle"]
    if not isinstance(manifest, dict) or digest(manifest) != selector.get(
        "manifest_sha256"
    ):
        raise ValueError("Imported human labels require the pinned reference manifest")
    if manifest.get("schema") != 1 or manifest.get("label_origin") != "human":
        raise ValueError("Reference manifest must declare human label origin")
    if manifest.get("scale") != selector["scale"]:
        raise ValueError("Reference manifest scale differs from score configuration")
    for key in ("rubric", "aggregation", "source_url", "source_revision"):
        identifier(manifest.get(key), f"reference {key}")
    checksum = identifier(manifest.get("source_sha256"), "reference source checksum")
    if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("Reference source checksum must be SHA-256")
    indexed: dict[str, Any] = {}
    identities: set[tuple[str, str]] = set()
    for record in manifest["records"]:
        record_id = identifier(record.get("record_id"), "reference record ID")
        identity = (record.get("policy"), record.get("prompt_id"))
        if record_id in indexed or identity in identities:
            raise ValueError("Duplicate reference record or response identity")
        if (
            identity[0] not in config["policies"]
            or identity[1] not in config["expected_prompt_ids"]
        ):
            raise ValueError("Reference record is outside the frozen population")
        numeric(
            {"id": record_id, "dataType": "NUMERIC", "value": record["value"]}, selector
        )
        identifier(record.get("output_sha256"), "reference response hash")
        indexed[record_id] = record
        identities.add(identity)
    if not indexed:
        raise ValueError("Imported human reference manifest is empty")
    return indexed


def dataset_content_index(manifest: dict | None, config: dict) -> dict:
    """Validate an explicitly supplied, trusted export of a frozen dataset."""
    if manifest is None:
        return {}
    if (
        manifest.get("schema") != 1
        or manifest.get("source")
        not in {
            "Langfuse SDK get_dataset at the pinned version",
            "Langfuse dataset-items API at the pinned version",
        }
        or manifest.get("project_id") != config["project_id"]
        or manifest.get("dataset_id") != config["dataset_id"]
    ):
        raise ValueError("Frozen dataset export has the wrong source or ownership")
    pinned_at = timestamp(manifest.get("requested_version"), "frozen dataset version")
    if config.get("dataset_version") is not None and pinned_at != timestamp(
        config["dataset_version"], "configured dataset version"
    ):
        raise ValueError("Dataset export differs from the configured version")
    indexed = {}
    for item in manifest["items"]:
        prompt = identifier(item.get("id"), "frozen dataset item")
        if prompt in indexed or item.get("datasetId") != config["dataset_id"]:
            raise ValueError("Duplicate or foreign frozen dataset item")
        if item.get("status") != "ACTIVE" or any(
            k not in item for k in ("input", "expectedOutput", "metadata")
        ):
            raise ValueError("Incomplete or inactive frozen dataset item")
        created = timestamp(item.get("createdAt"), "dataset item creation")
        updated = timestamp(item.get("updatedAt"), "dataset item update")
        if not created <= updated <= pinned_at:
            raise ValueError("Dataset item is newer than the frozen dataset version")
        indexed[prompt] = item
    if set(indexed) != set(config["expected_prompt_ids"]):
        raise ValueError("Frozen dataset export differs from the declared population")
    return indexed


def timestamp(value: Any, context: str) -> datetime:
    parsed = datetime.fromisoformat(identifier(value, context).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{context}: timezone is required")
    return parsed


def validate_selection(config: dict) -> None:
    """Validate run filters consistently for online and offline inputs."""
    start = timestamp(config.get("from_start_time"), "experiment window start")
    end = timestamp(config.get("to_start_time"), "experiment window end")
    if start >= end:
        raise ValueError("Use an increasing experiment time window")
    policies = config["policies"]
    if len(policies) != 2 or len(set(policies.values())) != 2:
        raise ValueError("Expected two distinct experiment IDs")
    for policy, experiment in policies.items():
        identifier(policy, "policy")
        identifier(experiment, "experiment ID")
        if "," in experiment:
            raise ValueError("Each policy must select one experiment ID")
    if (config.get("dataset_name") is None) != (config.get("dataset_version") is None):
        raise ValueError("Declare dataset_name and dataset_version together")


def prepare(
    snapshot: dict,
    config: dict,
    *,
    oracle_manifest: dict | None = None,
    dataset_manifest: dict | None = None,
) -> dict:
    """Require an explicit frozen population and exact observation score joins."""
    validate_selection(config)
    if snapshot.get("schema") != 1:
        raise ValueError("Unsupported snapshot schema")
    if snapshot.get("selection") != config:
        raise ValueError("Snapshot selection differs from the import configuration")
    project = identifier(config["project_id"], "project ID")
    dataset = identifier(config["dataset_id"], "dataset ID")
    policies = config["policies"]
    if (
        set(snapshot["items"]) != set(policies)
        or len(policies) != 2
        or len(set(policies.values())) != 2
    ):
        raise ValueError("Expected the two configured policies")
    for policy, experiment in policies.items():
        identifier(policy, "policy")
        identifier(experiment, "experiment ID")
    expected = config["expected_prompt_ids"]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Declare a nonempty, unique frozen prompt population")
    for prompt in expected:
        identifier(prompt, "expected prompt ID")
    frozen_items = dataset_content_index(dataset_manifest, config)
    dataset_manifest_hash = (
        digest(dataset_manifest) if dataset_manifest is not None else None
    )
    selectors = {k: config[k] for k in ("judge", "oracle")}
    origin = selectors["oracle"].get("label_origin", "native_annotation")
    if origin == "native_annotation":
        if selectors["oracle"]["source"] != "ANNOTATION" or oracle_manifest is not None:
            raise ValueError("Native human labels require ANNOTATION scores")
        imported = {}
    elif origin == "imported_human":
        if selectors["oracle"]["source"] != "API":
            raise ValueError("Imported human labels require API scores")
        imported = imported_label_index(oracle_manifest, config)
    else:
        raise ValueError("Unknown human label origin")
    if selectors["judge"]["source"] not in {"API", "EVAL"}:
        raise ValueError("Judge source must be API or EVAL")
    if tuple(selectors["judge"][k] for k in ("name", "source", "config_id")) == tuple(
        selectors["oracle"][k] for k in ("name", "source", "config_id")
    ):
        raise ValueError("Judge and human score selectors must be distinct")
    for selector in selectors.values():
        identifier(selector["name"], "score name")
        low, high = selector["scale"]
        if (
            any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                for v in (low, high)
            )
            or low >= high
        ):
            raise ValueError("Declare increasing finite score scale bounds")
        if "config_id" not in selector:
            raise ValueError(
                "Declare the score config ID (null only for unconfigured scores)"
            )

    rows: dict[str, list[dict[str, Any]]] = {}
    subject_to_row: dict[tuple[str, str], dict[str, Any]] = {}
    prompt_signatures: dict[str, tuple[str, str]] = {}
    for policy, items in snapshot["items"].items():
        rows[policy] = []
        seen = set()
        for item in items:
            prompt = identifier(item.get("experimentItemId"), "experimentItemId")
            if prompt in seen:
                raise ValueError(
                    f"{policy}: duplicate prompt; repeated draws need a separate export design"
                )
            seen.add(prompt)
            if (
                item.get("experimentId") != policies[policy]
                or item.get("experimentDatasetId") != dataset
            ):
                raise ValueError(f"{policy}: wrong experiment or dataset")
            version = item.get("experimentItemVersion")
            identity = {"basis": "native_version", "version": version}
            if frozen_items:
                assert dataset_manifest is not None
                if prompt not in frozen_items:
                    raise ValueError("Response is outside the frozen dataset export")
                reference = frozen_items[prompt]
                if "expectedOutput" not in item:
                    raise ValueError(
                        "Response is missing frozen expected output evidence"
                    )
                if item.get("input") != reference["input"]:
                    raise ValueError(
                        "Response input differs from frozen dataset content"
                    )
                if item.get("expectedOutput") != reference["expectedOutput"] and not (
                    reference["expectedOutput"] is None
                    and item.get("expectedOutput") == ""
                ):
                    raise ValueError(
                        "Expected output differs from frozen dataset content"
                    )
                if timestamp(
                    dataset_manifest["requested_version"], "frozen dataset version"
                ) > timestamp(item.get("startTime"), "experiment start"):
                    raise ValueError(
                        "Frozen dataset was selected after the response began"
                    )
                if version is None:
                    identity = {
                        "basis": "frozen_dataset_content",
                        "manifest_sha256": dataset_manifest_hash,
                        "requested_version": dataset_manifest["requested_version"],
                        "content_sha256": digest(
                            {
                                k: reference[k]
                                for k in (
                                    "id",
                                    "datasetId",
                                    "status",
                                    "input",
                                    "expectedOutput",
                                    "metadata",
                                )
                            }
                        ),
                    }
            if version is not None or not frozen_items:
                identifier(version, "item version")
            if item.get("endTime") is None or item.get("level") == "ERROR":
                raise ValueError(f"{policy}/{prompt}: unfinished or failed item")
            start = timestamp(item.get("startTime"), "response start")
            end = timestamp(item.get("endTime"), "response end")
            if end < start or not (
                timestamp(config["from_start_time"], "window start")
                <= start
                < timestamp(config["to_start_time"], "window end")
            ):
                raise ValueError(
                    "Response is outside the selected window or has invalid timing"
                )
            if item.get("input") is None or item.get("output") is None:
                raise ValueError(f"{policy}/{prompt}: missing input or actual output")
            signature = (digest(identity), digest(item["input"]))
            if prompt in prompt_signatures and prompt_signatures[prompt] != signature:
                raise ValueError(
                    f"{prompt}: input or dataset item version differs across policies"
                )
            prompt_signatures[prompt] = signature
            subject = (
                identifier(item.get("traceId"), "trace ID"),
                identifier(item.get("id"), "observation ID"),
            )
            if subject in subject_to_row:
                raise ValueError(
                    "One observation cannot represent multiple experiment responses"
                )
            row: dict[str, Any] = {
                "prompt_id": prompt,
                "response_id": digest([project, *subject, item["output"]]),
                "metadata": {
                    "policy": policy,
                    "trace_id": subject[0],
                    "observation_id": subject[1],
                    "dataset_item_version": version,
                    "dataset_item_identity": identity,
                    "output_sha256": digest(item["output"]),
                    "score_ids": {},
                },
            }
            rows[policy].append(row)
            subject_to_row[subject] = row
        if seen != set(expected):
            raise ValueError(
                f"{policy}: export does not match the frozen prompt population"
            )

    seen_scores: set[str] = set()
    used_imports: set[str] = set()
    ignored: Counter[str] = Counter()
    for score in snapshot["scores"]:
        score_id = identifier(score.get("id"), "score ID")
        if score_id in seen_scores:
            raise ValueError("Duplicate score ID across pages; export may have changed")
        seen_scores.add(score_id)
        if score.get("projectId") != project:
            raise ValueError("Score belongs to a different project")
        matches = [
            role
            for role, s in selectors.items()
            if (score.get("name"), score.get("source"), score.get("configId"))
            == (s["name"], s["source"], s["config_id"])
        ]
        if not matches:
            ignored["other_score_configuration"] += 1
            continue
        score_subject = score.get("subject") or {}
        key = (score_subject.get("traceId"), score_subject.get("id"))
        if score_subject.get("kind") != "observation" or key not in subject_to_row:
            raise ValueError(
                f"score {score_id}: selected score does not identify an exported response observation"
            )
        row = subject_to_row[key]
        for role in matches:
            field = "judge_score" if role == "judge" else "oracle_label"
            if field in row:
                raise ValueError(
                    f"{score_id}: multiple {role} scores; predeclare a rater/revision rule"
                )
            row[field] = numeric(score, selectors[role])
            row["metadata"]["score_ids"][role] = score_id
            if role == "oracle":
                if origin == "imported_human":
                    provenance = (score.get("metadata") or {}).get("cje_human_import")
                    if (
                        not isinstance(provenance, dict)
                        or provenance.get("manifest_sha256")
                        != selectors["oracle"]["manifest_sha256"]
                    ):
                        raise ValueError(
                            "Imported score lacks matching reference provenance"
                        )
                    record_id = identifier(
                        provenance.get("record_id"), "human reference record ID"
                    )
                    reference = imported.get(record_id)
                    if reference is None or record_id in used_imports:
                        raise ValueError("Unknown or reused human reference record")
                    if (
                        reference["policy"] != row["metadata"]["policy"]
                        or reference["prompt_id"] != row["prompt_id"]
                        or reference["output_sha256"]
                        != row["metadata"]["output_sha256"]
                        or reference["value"] != row[field]
                    ):
                        raise ValueError(
                            "Imported human label differs from its reference response or value"
                        )
                    used_imports.add(record_id)
                    row["metadata"]["human_provenance"] = dict(provenance)
                else:
                    row["metadata"]["human_provenance"] = {
                        "label_origin": "native_annotation",
                        "author_user_id": score.get("authorUserId"),
                        "queue_id": score.get("queueId"),
                    }
    if used_imports != set(imported):
        raise ValueError(
            "Missing imported human scores from the pinned reference manifest"
        )
    for policy, items in rows.items():
        if any("judge_score" not in row for row in items):
            raise ValueError(
                f"{policy}: missing judge score; no items may be silently dropped"
            )
    return {
        "fresh_draws_data": rows,
        "provenance": {
            "snapshot_sha256": digest(snapshot),
            "dataset_manifest_sha256": dataset_manifest_hash,
            "human_label_origin": origin,
            "human_manifest_sha256": selectors["oracle"].get("manifest_sha256"),
            "ignored_scores": dict(ignored),
            "counts": {
                p: {"responses": len(r), "labels": sum("oracle_label" in x for x in r)}
                for p, r in rows.items()
            },
            "analysis_status": "NOT_RUN: confirm representative labels and predeclared audit allocation",
        },
    }
