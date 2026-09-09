"""Offline end-to-end regression tests for bridge exports and label imports."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Iterator

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_bridge(tool: str, args: list[str]) -> int:
    old_argv, old_path = sys.argv[:], sys.path[:]
    try:
        sys.argv = ["convert.py", tool, *args]
        runpy.run_path(
            str(REPO_ROOT / "scripts/cje_bridges/convert.py"), run_name="__main__"
        )
    except SystemExit as exc:
        return int(exc.code or 0)
    finally:
        sys.argv, sys.path[:] = old_argv, old_path
    return 0


def _write_labels(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.suffix == ".jsonl":
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


@dataclass
class BridgeCase:
    tool: str
    args: list[str]
    source_rows: list[dict[str, Any]]
    write_source: Callable[[], None]
    tmp_path: Path

    def export(self, *args: str) -> tuple[int, Path, Path]:
        self.write_source()
        output = self.tmp_path / "converted.json"
        template = self.tmp_path / "labels.csv"
        code = _run_bridge(
            self.tool,
            [
                *self.args,
                "--out",
                str(output),
                "--label-template",
                str(template),
                *args,
            ],
        )
        return code, output, template


@pytest.fixture(params=["promptfoo", "trulens", "opencompass"])
def bridge_case(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> BridgeCase:
    tool = str(request.param)
    input_path = tmp_path / "model.json"
    prompts = ["repeated question", "repeated question", "other question"]
    outputs = ["wrong answer", "right answer", "unlabeled answer"]
    scores = [0.1, 0.9, 0.4]
    rows: list[dict[str, Any]]
    if tool == "promptfoo":
        rows = [
            {
                "provider": {"id": "model"},
                "prompt": {"label": "question"},
                "testCase": {"vars": {"question": prompt}},
                "score": score,
                "response": {"output": output},
            }
            for prompt, output, score in zip(prompts, outputs, scores)
        ]

        def write_source() -> None:
            input_path.write_text(json.dumps({"results": rows}))

        args = [str(input_path)]
    elif tool == "opencompass":
        rows = [
            {"origin_prompt": prompt, "prediction": score, "output": output}
            for prompt, output, score in zip(prompts, outputs, scores)
        ]

        def write_source() -> None:
            input_path.write_text(json.dumps({"details": rows}))

        args = [str(input_path)]
    else:
        rows = [
            {
                "app_version": "model",
                "input": prompt,
                "output": output,
                "Answer Relevance": score,
                "record_id": f"record-{i}",
            }
            for i, (prompt, output, score) in enumerate(zip(prompts, outputs, scores))
        ]

        class Frame:
            columns = list(rows[0])

            def iterrows(self) -> Iterator[tuple[int, dict[str, Any]]]:
                yield from enumerate(rows)

        class Session:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def get_records_and_feedback(
                self, **kwargs: Any
            ) -> tuple[Frame, list[str]]:
                return Frame(), ["Answer Relevance"]

        for name in ("trulens", "trulens.core", "trulens.core.session"):
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
        monkeypatch.setattr(
            sys.modules["trulens.core.session"], "TruSession", Session, raising=False
        )

        def write_source() -> None:
            pass

        args = [
            "--database-url",
            "sqlite:///fixture.sqlite",
            "--judge-col",
            "Answer Relevance",
        ]
    return BridgeCase(tool, args, rows, write_source, tmp_path)


def _template_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("suffix", [".csv", ".jsonl"])
def test_sparse_labels_roundtrip_distinguishes_repeated_responses(
    bridge_case: BridgeCase, suffix: str
) -> None:
    code, output, template = bridge_case.export()
    assert code == 0
    rows = _template_rows(template)
    assert rows[0]["prompt_id"] == rows[1]["prompt_id"]
    assert len({row["response_id"] for row in rows}) == 3
    for row, label in zip(rows, [0, 1, "  "]):
        row["oracle_label"] = label
    labels = bridge_case.tmp_path / f"completed{suffix}"
    _write_labels(labels, rows)

    # Source export reordering must not swap labels on distinct responses.
    bridge_case.source_rows.reverse()
    code, output, _ = bridge_case.export("--oracle-labels", str(labels))
    assert code == 0
    converted = json.loads(output.read_text())["model"]
    assert [row.get("oracle_label") for row in converted] == [None, 1.0, 0.0]
    assert {row["response_id"] for row in converted} == {
        row["response_id"] for row in rows
    }
    assert [row["oracle_label"] for row in _template_rows(template)] == [
        "",
        "1.0",
        "0.0",
    ]

    # New response IDs remain compatible with the actual CJE ingestion API.
    from cje import analyze_dataset

    result = analyze_dataset(fresh_draws_data={"model": converted})
    assert result.estimates.shape == (1,)


def test_legacy_labels_only_apply_to_unique_prompt(
    bridge_case: BridgeCase, capsys: pytest.CaptureFixture[str]
) -> None:
    assert bridge_case.export()[0] == 0
    rows = _template_rows(bridge_case.tmp_path / "labels.csv")
    for row in rows:
        row.pop("response_id")
    labels = bridge_case.tmp_path / "legacy.csv"
    rows[2]["oracle_label"] = 0
    _write_labels(labels, [rows[2]])
    code, output, _ = bridge_case.export("--oracle-labels", str(labels))
    assert code == 0
    assert json.loads(output.read_text())["model"][2]["oracle_label"] == 0

    output.unlink()
    rows[0]["oracle_label"] = 1
    _write_labels(labels, [rows[0]])
    assert bridge_case.export("--oracle-labels", str(labels))[0] != 0
    assert "ambiguous legacy oracle label" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("bad_label", ["NaN", "Infinity", "-Infinity", "not a number"])
def test_nonfinite_or_invalid_labels_fail_with_row_context(
    bridge_case: BridgeCase, bad_label: str, capsys: pytest.CaptureFixture[str]
) -> None:
    _, output, template = bridge_case.export()
    row = _template_rows(template)[0]
    row["oracle_label"] = bad_label
    labels = bridge_case.tmp_path / "bad.csv"
    _write_labels(labels, [row])
    output.unlink()
    assert bridge_case.export("--oracle-labels", str(labels))[0] != 0
    assert "bad.csv:line 2: oracle_label" in capsys.readouterr().err
    assert not output.exists()


def test_boolean_json_label_is_not_numeric(bridge_case: BridgeCase) -> None:
    _, _, template = bridge_case.export()
    row = _template_rows(template)[0]
    row["oracle_label"] = True
    labels = bridge_case.tmp_path / "boolean.jsonl"
    _write_labels(labels, [row])
    assert bridge_case.export("--oracle-labels", str(labels))[0] != 0


def test_dropped_response_does_not_make_legacy_label_unambiguous(
    bridge_case: BridgeCase, capsys: pytest.CaptureFixture[str]
) -> None:
    _, output, template = bridge_case.export()
    row = _template_rows(template)[0]
    row.pop("response_id")
    row["oracle_label"] = 0
    labels = bridge_case.tmp_path / "legacy_with_drop.csv"
    _write_labels(labels, [row])
    score_key = {
        "promptfoo": "score",
        "trulens": "Answer Relevance",
        "opencompass": "prediction",
    }[bridge_case.tool]
    bridge_case.source_rows[0][score_key] = None
    output.unlink()
    assert bridge_case.export("--oracle-labels", str(labels))[0] != 0
    assert "ambiguous legacy oracle label" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("problem", ["duplicate", "unmatched"])
def test_invalid_label_keys_fail_instead_of_losing_labels(
    bridge_case: BridgeCase, problem: str, capsys: pytest.CaptureFixture[str]
) -> None:
    _, output, template = bridge_case.export()
    row = _template_rows(template)[0]
    row["oracle_label"] = 0
    rows = [row]
    if problem == "duplicate":
        rows.append({**row, "oracle_label": 1})
    else:
        row["response_id"] = "does-not-exist"
    labels = bridge_case.tmp_path / "wrong_keys.csv"
    _write_labels(labels, rows)
    output.unlink()
    assert bridge_case.export("--oracle-labels", str(labels))[0] != 0
    error = capsys.readouterr().err
    assert (
        "duplicate oracle-label key"
        if problem == "duplicate"
        else "matches no exported response"
    ) in error
    assert not output.exists()


def test_nonfinite_judge_score_is_not_written(bridge_case: BridgeCase) -> None:
    score_key = {
        "promptfoo": "score",
        "trulens": "Answer Relevance",
        "opencompass": "prediction",
    }[bridge_case.tool]
    bridge_case.source_rows[0][score_key] = float("nan")
    code, output, _ = bridge_case.export()
    assert code != 0
    assert not output.exists()


def test_identical_fallback_draws_still_get_distinct_ids(
    bridge_case: BridgeCase,
) -> None:
    for row in bridge_case.source_rows:
        row.pop("record_id", None)
    bridge_case.source_rows[1] = dict(bridge_case.source_rows[0])
    _, _, template = bridge_case.export()
    rows = _template_rows(template)
    assert rows[0]["response_id"] != rows[1]["response_id"]
    rows[0]["oracle_label"], rows[1]["oracle_label"] = 0, 1
    labels = bridge_case.tmp_path / "identical.csv"
    _write_labels(labels, rows)
    code, output, _ = bridge_case.export("--oracle-labels", str(labels))
    assert code == 0
    assert [
        row["oracle_label"] for row in json.loads(output.read_text())["model"][:2]
    ] == [0, 1]


def test_duplicate_native_response_ids_fail(bridge_case: BridgeCase) -> None:
    id_key = {
        "promptfoo": "id",
        "trulens": "record_id",
        "opencompass": "response_id",
    }[bridge_case.tool]
    for row in bridge_case.source_rows[:2]:
        row[id_key] = "duplicate-id"
    code, output, _ = bridge_case.export()
    assert code != 0
    assert not output.exists()


@pytest.mark.parametrize("bridge_case", ["trulens"], indirect=True)
def test_missing_dataframe_record_ids_use_fallback(bridge_case: BridgeCase) -> None:
    # pandas missing float cells are NaN; a nullable sentinel also refuses
    # conversion to bool. Exercise both contracts without adding pandas to
    # the core test dependencies.
    class NullableMissing:
        def __ne__(self, other: object) -> Any:
            return self

        def __bool__(self) -> bool:
            raise TypeError("boolean value of NA is ambiguous")

    for missing in (float("nan"), NullableMissing()):
        bridge_case.source_rows[1]["record_id"] = missing
        bridge_case.source_rows[2]["record_id"] = missing
        code, output, _ = bridge_case.export()
        assert code == 0
        rows = json.loads(output.read_text())["model"]
        assert rows[0]["response_id"] == "record-0"
        assert len({row["response_id"] for row in rows}) == 3
        assert all(row["response_id"].startswith("response::") for row in rows[1:])


@pytest.mark.parametrize("bridge_case", ["trulens", "opencompass"], indirect=True)
def test_missing_primary_id_still_uses_native_alias(bridge_case: BridgeCase) -> None:
    for i, row in enumerate(bridge_case.source_rows):
        if bridge_case.tool == "trulens":
            row["record_id"] = float("nan")
            row["recordId"] = f"native-{i}"
        else:
            row["response_id"] = None
            row["record_id"] = f"native-{i}"
        row["output"] = "identical answer"
    code, output, template = bridge_case.export()
    assert code == 0
    assert [row["response_id"] for row in json.loads(output.read_text())["model"]] == [
        "native-0",
        "native-1",
        "native-2",
    ]
    rows = _template_rows(template)
    rows[0]["oracle_label"], rows[1]["oracle_label"] = 0, 1
    labels = bridge_case.tmp_path / "aliases.csv"
    _write_labels(labels, rows)
    bridge_case.source_rows.reverse()
    code, output, _ = bridge_case.export("--oracle-labels", str(labels))
    assert code == 0
    assert [
        row.get("oracle_label") for row in json.loads(output.read_text())["model"]
    ] == [None, 1, 0]


@pytest.mark.parametrize("bridge_case", ["promptfoo"], indirect=True)
@pytest.mark.parametrize("outputs", [(3, 2), (True, False), (3, "3"), (None, "null")])
def test_scalar_outputs_keep_response_identity_when_reordered(
    bridge_case: BridgeCase, outputs: tuple[Any, Any]
) -> None:
    for row, value in zip(bridge_case.source_rows[:2], outputs):
        row["response"]["output"] = value
    _, _, template = bridge_case.export()
    rows = _template_rows(template)
    assert rows[0]["output"] != ""
    assert rows[1]["output"] != ""
    assert rows[0]["response_id"] != rows[1]["response_id"]
    rows[0]["oracle_label"], rows[1]["oracle_label"] = 0, 1
    labels = bridge_case.tmp_path / "scalar_labels.csv"
    _write_labels(labels, rows)
    bridge_case.source_rows.reverse()
    code, output, _ = bridge_case.export("--oracle-labels", str(labels))
    assert code == 0
    assert [
        row.get("oracle_label") for row in json.loads(output.read_text())["model"]
    ] == [None, 1, 0]


def _mock_langsmith(
    monkeypatch: pytest.MonkeyPatch, *, bad_oracle: bool = False
) -> None:
    class Client:
        def list_runs(self, **kwargs: Any) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(
                    id=f"run-{i}",
                    reference_example_id="shared-question",
                    inputs={"q": "Q"},
                    outputs={"answer": str(i)},
                )
                for i in range(2)
            ]

        def list_feedback(
            self, run_ids: list[str], feedback_key: list[str]
        ) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(
                    run_id=run_id,
                    score=(
                        float("nan")
                        if bad_oracle and feedback_key == ["human"]
                        else float(i)
                    ),
                    created_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
                )
                for i, run_id in enumerate(run_ids)
            ]

    module = ModuleType("langsmith")
    monkeypatch.setattr(module, "Client", Client, raising=False)
    monkeypatch.setitem(sys.modules, "langsmith", module)


def test_langsmith_uses_run_identity_for_repeated_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_langsmith(monkeypatch)
    output, template = tmp_path / "langsmith.json", tmp_path / "langsmith.csv"
    assert (
        _run_bridge(
            "langsmith",
            [
                "--project",
                "model",
                "--feedback-key",
                "judge",
                "--oracle-feedback-key",
                "human",
                "--out",
                str(output),
                "--label-template",
                str(template),
            ],
        )
        == 0
    )
    rows = json.loads(output.read_text())["model"]
    assert [row["response_id"] for row in rows] == ["run-0", "run-1"]
    assert rows[0]["prompt_id"] == rows[1]["prompt_id"]
    assert [row["oracle_label"] for row in rows] == [0, 1]
    assert [row["response_id"] for row in _template_rows(template)] == [
        "run-0",
        "run-1",
    ]


@pytest.mark.parametrize("bridge_case", ["promptfoo", "trulens"], indirect=True)
def test_run_cje_propagates_analysis_failure(
    bridge_case: BridgeCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import cje

    def fail_analysis(**kwargs: Any) -> None:
        raise RuntimeError("fixture analysis failure")

    monkeypatch.setattr(cje, "analyze_dataset", fail_analysis)
    _, _, template = bridge_case.export()
    row = _template_rows(template)[0]
    row["oracle_label"] = 0
    labels = bridge_case.tmp_path / "for_analysis.csv"
    _write_labels(labels, [row])
    assert bridge_case.export("--oracle-labels", str(labels), "--run-cje")[0] == 1
    assert "fixture analysis failure" in capsys.readouterr().err


def test_langsmith_run_cje_propagates_analysis_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import cje

    _mock_langsmith(monkeypatch)

    def fail_analysis(**kwargs: Any) -> None:
        raise RuntimeError("fixture analysis failure")

    monkeypatch.setattr(cje, "analyze_dataset", fail_analysis)
    assert (
        _run_bridge(
            "langsmith",
            [
                "--project",
                "model",
                "--feedback-key",
                "judge",
                "--oracle-feedback-key",
                "human",
                "--run-cje",
                "--out",
                str(tmp_path / "output.json"),
                "--no-label-template",
            ],
        )
        == 1
    )
    assert "fixture analysis failure" in capsys.readouterr().err


def test_langsmith_rejects_nonfinite_oracle_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_langsmith(monkeypatch, bad_oracle=True)
    output = tmp_path / "nonfinite.json"
    assert (
        _run_bridge(
            "langsmith",
            [
                "--project",
                "model",
                "--feedback-key",
                "judge",
                "--oracle-feedback-key",
                "human",
                "--out",
                str(output),
                "--no-label-template",
            ],
        )
        == 2
    )
    assert not output.exists()
