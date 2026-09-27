"""Companion exporter regressions, including interrupted offline preparation."""

from __future__ import annotations
from typing import Any
import importlib.util
import json
from pathlib import Path
import sys

import httpx
import pytest

from cje.bridges.langfuse import prepare
from cje.tests.test_langfuse_bridge import config, item, score, snapshot
from cje.tests.test_langfuse_dataset_identity import exported_panel

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "langfuse_cje"
sys.path.insert(0, str(SCRIPT))
import transport
from transport import collect, pages, request_with_retry, collect_dataset

spec = importlib.util.spec_from_file_location(
    "langfuse_export_command", SCRIPT / "export.py"
)
assert spec is not None and spec.loader is not None
command = importlib.util.module_from_spec(spec)
spec.loader.exec_module(command)


def test_rate_limit_on_second_page_retries_same_cursor_without_duplicates(
    monkeypatch: Any,
) -> None:
    delays: list[int] = []
    requests: list[Any] = []
    monkeypatch.setattr("transport.time.sleep", delays.append)

    def route(request: Any) -> Any:
        requests.append(dict(request.url.params))
        if len(requests) == 1:
            return httpx.Response(
                200, json={"data": [{"id": "first"}], "meta": {"cursor": "next"}}
            )
        if len(requests) == 2:
            return httpx.Response(429, headers={"Retry-After": "61"})
        return httpx.Response(200, json={"data": [{"id": "second"}], "meta": {}})

    with httpx.Client(
        base_url="https://pilot.invalid", transport=httpx.MockTransport(route)
    ) as client:
        rows = list(pages(client, "api/public/v3/scores", {"traceId": "a,b"}))
    assert rows == [{"id": "first"}, {"id": "second"}]
    assert requests[1] == requests[2]
    assert requests[1]["cursor"] == "next"
    assert delays == [30, 30, 1]


def test_persistent_rate_limit_fails_instead_of_returning_partial_export(
    monkeypatch: Any,
) -> None:
    delays: list[int] = []
    requests: list[Any] = []
    monkeypatch.setattr("transport.time.sleep", delays.append)

    def route(request: Any) -> Any:
        requests.append(request)
        return httpx.Response(429, headers={"Retry-After": "1"})

    with httpx.Client(
        base_url="https://pilot.invalid", transport=httpx.MockTransport(route)
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            list(pages(client, "api/public/v3/scores", {}))
    assert len(requests) == 5
    assert delays == [1, 1, 1, 1]


def test_long_server_delay_is_not_shortened(monkeypatch: Any) -> None:
    delays: list[int] = []
    monkeypatch.setattr("transport.time.sleep", delays.append)
    with httpx.Client(
        base_url="https://pilot.invalid",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(429, headers={"Retry-After": "301"})
        ),
    ) as client:
        assert request_with_retry(client, "GET", "/test").status_code == 429
    assert delays == []


@pytest.mark.parametrize("header", ["invalid", "nan", "-1"])
def test_malformed_retry_header_uses_bounded_backoff(
    monkeypatch: Any, header: Any
) -> None:
    delays: list[int] = []
    calls: list[Any] = []
    monkeypatch.setattr("transport.time.sleep", delays.append)

    def route(request: Any) -> Any:
        calls.append(request)
        return (
            httpx.Response(429, headers={"Retry-After": header})
            if len(calls) == 1
            else httpx.Response(200)
        )

    with httpx.Client(
        base_url="https://pilot.invalid", transport=httpx.MockTransport(route)
    ) as client:
        assert request_with_retry(client, "GET", "/test").status_code == 200
    assert delays == [5]


def test_collect_batches_all_134_traces_into_three_score_reads() -> None:
    cfg = config()
    cfg["expected_prompt_ids"] = [f"q{i}" for i in range(67)]
    groups = {p: [item(p, i) for i in range(67)] for p in cfg["policies"]}
    all_rows = [r for group in groups.values() for r in group]
    scores = [score(r) for r in all_rows]
    batches = []

    def route(request: Any) -> Any:
        params = request.url.params
        if request.url.path == "/api/public/experiment-items":
            rows = [r for r in all_rows if r["experimentId"] == params["experimentId"]]
        else:
            assert request.url.path == "/api/public/v3/scores"
            assert "fromTimestamp" not in params and "toTimestamp" not in params
            batch = params["traceId"].split(",")
            batches.append(batch)
            rows = [s for s in scores if s["subject"]["traceId"] in batch]
        return httpx.Response(200, json={"data": rows, "meta": {}})

    with httpx.Client(
        base_url="https://pilot.invalid", transport=httpx.MockTransport(route)
    ) as client:
        result = collect(client, cfg)
    assert list(map(len, batches)) == [50, 50, 34]
    assert len({t for batch in batches for t in batch}) == 134
    assert {s["id"] for s in result["scores"]} == {s["id"] for s in scores}


def test_pagination_fetches_scores_beyond_embedded_cap_and_uses_get_only() -> None:
    data, cfg = snapshot()
    requests = []
    target = data["items"]["A"][0]
    extras = [dict(score(target), id=f"extra-{i}", name="other") for i in range(101)]
    all_scores = [*extras, *data["scores"]]

    def route(request: Any) -> Any:
        requests.append(request)
        assert request.method == "GET"
        query = request.url.params
        offset = int(query.get("cursor", "0"))
        if request.url.path == "/api/public/experiment-items":
            assert query["fields"] == "core,dataset,io"
            rows = [
                r
                for group in data["items"].values()
                for r in group
                if r["experimentId"] == query["experimentId"]
            ]
            size = 1
        else:
            assert request.url.path == "/api/public/v3/scores"
            assert query["fields"] == "subject,details,annotation"
            assert "fromTimestamp" not in query and "toTimestamp" not in query
            rows = [
                s
                for s in all_scores
                if s["subject"]["traceId"] in query["traceId"].split(",")
            ]
            size = 100
        page = rows[offset : offset + size]
        meta = {"cursor": str(offset + size)} if offset + size < len(rows) else {}
        return httpx.Response(200, json={"data": page, "meta": meta})

    with httpx.Client(
        base_url="https://pilot.invalid/", transport=httpx.MockTransport(route)
    ) as client:
        exported = collect(client, cfg)
    assert len(exported["scores"]) == 107
    result = prepare(exported, cfg)
    assert result["provenance"]["counts"]["A"]["labels"] == 1
    assert any(r.url.params.get("cursor") == "100" for r in requests)


def test_repeated_cursor_and_page_limit_fail() -> None:
    with httpx.Client(
        base_url="https://pilot.invalid/",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"data": [], "meta": {"cursor": "same"}})
        ),
    ) as client:
        with pytest.raises(ValueError, match="repeated pagination"):
            list(pages(client, "items", {}))
        with pytest.raises(ValueError, match="page limit"):
            list(pages(client, "items", {}, max_pages=1))


def test_offline_resume_preserves_export_after_missing_version_proof(
    tmp_path: Any,
) -> None:
    data, cfg, manifest = exported_panel()
    with pytest.raises(ValueError, match="item version"):
        command.run(cfg, tmp_path, snapshot=data)
    raw = (tmp_path / "snapshot.json").read_bytes()
    assert not (tmp_path / "readiness.json").exists()
    result = command.run(cfg, tmp_path, dataset_manifest=manifest)
    assert result["status"] == "PREPARED"
    assert (tmp_path / "snapshot.json").read_bytes() == raw
    assert command.run(cfg, tmp_path) == result
    for path in tmp_path.iterdir():
        assert path.stat().st_mode & 0o077 == 0
    assert len(json.loads((tmp_path / "fresh-draws.json").read_text())["A"]) == 2
    assert result["sampling_design"].startswith("UNKNOWN")
    assert result["provenance"]["analysis_status"].startswith("NOT_RUN")


def test_resume_rejects_changed_config_or_export(tmp_path: Any) -> None:
    data, cfg = snapshot()
    command.run(cfg, tmp_path, snapshot=data)
    data["scores"][0]["value"] = 2
    with pytest.raises(ValueError, match="snapshot.json differs"):
        command.run(cfg, tmp_path, snapshot=data)
    cfg["judge"]["name"] = "another"
    with pytest.raises(ValueError, match="config.json differs"):
        command.run(cfg, tmp_path)


def test_atomic_write_does_not_leave_truncated_file(
    tmp_path: Any, monkeypatch: Any
) -> None:
    def failed_link(*args: Any) -> Any:
        raise OSError("interrupted")

    monkeypatch.setattr(command.os, "link", failed_link)
    with pytest.raises(OSError):
        command.save(tmp_path / "result.json", {"a": 1})
    assert list(tmp_path.iterdir()) == []


def test_offline_cli_needs_no_keys_and_reuses_completed_files(
    tmp_path: Any, monkeypatch: Any, capsys: Any
) -> None:
    data, cfg = snapshot()
    command.save(tmp_path / "input.json", data)
    command.save(tmp_path / "selection.json", cfg)
    for name in ("LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    args = [
        "--config",
        str(tmp_path / "selection.json"),
        "--out-dir",
        str(tmp_path / "result"),
    ]
    assert command.main(args + ["--snapshot", str(tmp_path / "input.json")]) == 0
    assert command.main(args) == 0
    assert "estimation has not run" in capsys.readouterr().out


def test_online_missing_dataset_read_saves_snapshot_before_failure(
    tmp_path: Any,
) -> None:
    data, cfg = snapshot()
    cfg.update(dataset_name="sample", dataset_version=cfg["from_start_time"])
    data["selection"] = cfg.copy()
    with pytest.raises(ValueError, match="Frozen dataset read is missing"):
        command.run(cfg, tmp_path, snapshot=data)
    assert (tmp_path / "snapshot.json").exists()
    assert not (tmp_path / "readiness.json").exists()


def test_frozen_dataset_api_pages_and_version_are_preserved() -> None:
    data, cfg, manifest = exported_panel()
    cfg.update(
        dataset_name="folder/sample", dataset_version=manifest["requested_version"]
    )
    seen = []

    def route(request: Any) -> Any:
        assert request.method == "GET"
        if "/v2/datasets/" in request.url.path:
            return httpx.Response(200, json={"id": cfg["dataset_id"]})
        query = dict(request.url.params)
        assert query["datasetName"] == "folder/sample"
        assert query["version"] == "2026-09-01T00:00:00+00:00"
        page = int(query["page"])
        seen.append(page)
        return httpx.Response(
            200,
            json={
                "data": [manifest["items"][page - 1]],
                "meta": {"page": page, "totalPages": 2},
            },
        )

    with httpx.Client(
        base_url="https://example.invalid", transport=httpx.MockTransport(route)
    ) as client:
        result = collect_dataset(client, cfg)
    assert seen == [1, 2]
    assert result["items"] == manifest["items"]
    data["selection"] = cfg.copy()
    from cje.bridges.langfuse import prepare

    assert prepare(data, cfg, dataset_manifest=result)["fresh_draws_data"]


def test_exporter_never_retries_writes() -> None:
    with pytest.raises(ValueError, match="only permits GET"):
        request_with_retry(None, "POST", "/api/public/scores")
