"""Synthetic API contract tests, not evidence from a hosted Langfuse project."""

from typing import Any

from copy import deepcopy

import pytest

from cje.bridges.langfuse import prepare

STAMP = "2026-09-01T00:00:00Z"


def config() -> Any:
    return {
        "project_id": "project-test",
        "dataset_id": "dataset-test",
        "from_start_time": STAMP,
        "to_start_time": "2026-09-02T00:00:00Z",
        "policies": {"A": "exp-a", "B": "exp-b"},
        "expected_prompt_ids": ["q0", "q1"],
        "judge": {
            "name": "coherence",
            "source": "EVAL",
            "config_id": None,
            "scale": [1, 5],
        },
        "oracle": {
            "name": "human-coherence",
            "source": "ANNOTATION",
            "config_id": "human-config",
            "scale": [1, 5],
        },
    }


def item(policy: Any, q: Any) -> Any:
    return {
        "id": f"obs-{policy}-{q}",
        "traceId": f"trace-{policy}-{q}",
        "startTime": STAMP,
        "endTime": "2026-09-01T00:01:00Z",
        "level": "DEFAULT",
        "environment": "test",
        "experimentId": f"exp-{policy.lower()}",
        "experimentName": policy,
        "experimentItemId": f"q{q}",
        "experimentDatasetId": "dataset-test",
        "experimentItemVersion": STAMP,
        "input": {"question": f"Question {q}"},
        "output": f"Response {policy} {q}",
        "expectedOutput": "This is not a human score",
    }


def score(row: Any, human: Any = False, value: Any = 3) -> Any:
    return {
        "id": f"{'human' if human else 'judge'}-{row['id']}",
        "projectId": "project-test",
        "name": "human-coherence" if human else "coherence",
        "source": "ANNOTATION" if human else "EVAL",
        "dataType": "NUMERIC",
        "value": value,
        "timestamp": STAMP,
        "environment": "test",
        "createdAt": STAMP,
        "updatedAt": STAMP,
        "configId": "human-config" if human else None,
        "subject": {"kind": "observation", "id": row["id"], "traceId": row["traceId"]},
    }


def snapshot() -> Any:
    cfg = config()
    items = {p: [item(p, q) for q in range(2)] for p in cfg["policies"]}
    scores = [score(r) for rows in items.values() for r in rows]
    scores += [score(items[p][0], human=True, value=4) for p in items]
    return {
        "schema": 1,
        "selection": deepcopy(cfg),
        "items": items,
        "scores": scores,
    }, cfg


def test_exact_joins_keep_unlabeled_items_and_native_schema() -> None:
    data, cfg = snapshot()
    result = prepare(data, cfg)
    for rows in result["fresh_draws_data"].values():
        assert len(rows) == 2
        assert rows[0]["oracle_label"] == 4
        assert "oracle_label" not in rows[1]
        assert rows[0]["response_id"] != rows[1]["response_id"]
    assert result["provenance"]["analysis_status"].startswith("NOT_RUN")


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "3", None, 0, 6])
def test_invalid_numeric_score(value: Any) -> None:
    data, cfg = snapshot()
    data["scores"][0]["value"] = value
    with pytest.raises(ValueError):
        prepare(data, cfg)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda d: d["items"]["A"].pop(), "frozen prompt"),
        (
            lambda d: d["items"]["A"].append(deepcopy(d["items"]["A"][0])),
            "duplicate prompt",
        ),
        (
            lambda d: d["items"]["B"][0].update(input="different input"),
            "input or dataset",
        ),
        (
            lambda d: d["items"]["B"][0].update(
                experimentItemVersion="2026-09-02T00:00:00Z"
            ),
            "item version differs",
        ),
        (
            lambda d: d["items"]["B"][0].update(experimentDatasetId="other"),
            "wrong experiment or dataset",
        ),
        (lambda d: d["items"]["B"][0].update(output=None), "actual output"),
        (lambda d: d["items"]["B"][0].update(endTime=None), "unfinished"),
        (lambda d: d["items"]["B"][0].update(level="ERROR"), "failed item"),
        (lambda d: d["scores"].pop(0), "missing judge"),
        (lambda d: d["scores"].append(deepcopy(d["scores"][0])), "Duplicate score"),
        (lambda d: d["scores"][0].update(projectId="other"), "different project"),
        (
            lambda d: d["scores"][0]["subject"].update(kind="trace"),
            "exported response observation",
        ),
        (
            lambda d: d["scores"][0]["subject"].update(id="different-observation"),
            "exported response observation",
        ),
        (
            lambda d: d["scores"][0]["subject"].update(traceId="different-trace"),
            "exported response observation",
        ),
    ],
)
def test_identity_and_population_failures(mutation: Any, message: Any) -> None:
    data, cfg = snapshot()
    mutation(data)
    with pytest.raises(ValueError, match=message):
        prepare(data, cfg)


def test_two_raters_require_explicit_rule() -> None:
    data, cfg = snapshot()
    extra = deepcopy(data["scores"][-1])
    extra["id"] = "second-human-rating"
    data["scores"].append(extra)
    with pytest.raises(ValueError, match="multiple oracle scores"):
        prepare(data, cfg)


def test_same_name_different_configuration_is_not_a_label() -> None:
    data, cfg = snapshot()
    for s in data["scores"]:
        if s["source"] == "ANNOTATION":
            s["configId"] = "different-human-rubric"
    result = prepare(data, cfg)
    assert all(
        "oracle_label" not in r
        for rows in result["fresh_draws_data"].values()
        for r in rows
    )
    assert result["provenance"]["ignored_scores"] == {"other_score_configuration": 2}


def test_native_cje_accepts_prepared_sparse_panel() -> None:
    from cje import analyze_dataset

    data, cfg = snapshot()
    cfg["expected_prompt_ids"] = [f"q{i}" for i in range(25)]
    data["selection"] = deepcopy(cfg)
    data["items"] = {p: [item(p, i) for i in range(25)] for p in cfg["policies"]}
    data["scores"] = []
    for policy, rows in data["items"].items():
        for i, row in enumerate(rows):
            value = 1 + (i + (policy == "B")) % 5
            data["scores"].append(score(row, value=value))
            if i < 15:
                data["scores"].append(score(row, human=True, value=value))
    prepared = prepare(data, cfg)
    result = analyze_dataset(
        fresh_draws_data=prepared["fresh_draws_data"],
        estimator="direct",
        fresh_judge_scale=(1, 5),
        fresh_oracle_scale=(1, 5),
        output_scale=(1, 5),
    )
    assert set(result.metadata["claim_tier_by_policy"].values()) == {
        "CALIBRATED_ORACLE_MEAN"
    }
    assert len(result.estimates) == 2
    assert result.compare_policies(1, 0)["basis"] == "prompt_cluster_paired"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d, c: d["items"]["A"][0].update(startTime="2026-08-31T23:59:59Z"),
        lambda d, c: d["items"]["A"][0].update(endTime="invalid"),
        lambda d, c: c.update(to_start_time=c["from_start_time"]),
        lambda d, c: c["policies"].update(B="exp-a"),
        lambda d, c: c.update(dataset_name="name-without-version"),
    ],
)
def test_offline_selection_and_timing_cannot_bypass_filters(mutation: Any) -> None:
    data, cfg = snapshot()
    mutation(data, cfg)
    data["selection"] = deepcopy(cfg)
    with pytest.raises(ValueError):
        prepare(data, cfg)
