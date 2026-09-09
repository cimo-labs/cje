from __future__ import annotations

import csv
import json
import runpy
import sys
from pathlib import Path

import pytest


def _run_script(path: Path, argv: list[str]) -> int:
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(path)] + argv
        runpy.run_path(str(path), run_name="__main__")
        return 0
    except SystemExit as e:
        code = e.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    finally:
        sys.argv = old_argv


def test_opencompass_to_cje_converter_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    converter = repo_root / "scripts" / "opencompass_cje" / "opencompass_to_cje.py"
    sample = Path(__file__).resolve().parent / "data" / "opencompass_sample.json"

    out_json = tmp_path / "cje.json"
    out_csv = tmp_path / "oracle_label_template.csv"

    code = _run_script(
        converter,
        [
            str(sample),
            "--out",
            str(out_json),
            "--label-template",
            str(out_csv),
        ],
    )
    assert code == 0

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "opencompass_sample" in payload
    rows = payload["opencompass_sample"]
    assert len(rows) == 3  # last sample missing prompt -> dropped

    assert rows[0]["judge_score"] == 0.0
    assert rows[1]["judge_score"] == 1.0
    assert rows[2]["judge_score"] == 0.7

    # CSV template exists + has expected columns
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "policy_name",
            "prompt_id",
            "response_id",
            "judge_score",
            "prompt",
            "oracle_label",
        ]
        csv_rows = list(reader)
    assert len(csv_rows) == 3


def test_opencompass_format_details_dict_keyed(tmp_path: Path) -> None:
    """Test that the converter handles format_details() dict-keyed-by-string-indices output.

    OpenCompass's OpenICLEvalTask.format_details() returns details as
    {"type": "GEN", "0": {...}, "1": {...}} rather than a list of dicts.
    """
    repo_root = Path(__file__).resolve().parents[2]
    converter = repo_root / "scripts" / "opencompass_cje" / "opencompass_to_cje.py"
    sample = (
        Path(__file__).resolve().parent / "data" / "opencompass_format_details.json"
    )

    out_json = tmp_path / "cje.json"

    code = _run_script(
        converter,
        [
            str(sample),
            "--out",
            str(out_json),
            "--no-label-template",
        ],
    )
    assert code == 0

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "opencompass_format_details" in payload
    rows = payload["opencompass_format_details"]
    assert len(rows) == 3

    # predictions field is used for judge score extraction
    assert rows[0]["judge_score"] == 0.0  # "B" -> 0.0
    assert rows[1]["judge_score"] == 1.0  # "A" -> 1.0
    assert rows[2]["judge_score"] == 0.7  # "0.7" -> 0.7


def test_ppl_format_returns_nonzero_exit(tmp_path: Path) -> None:
    """PPL-type format_details() output has no prompt field.

    The converter should exit non-zero (all rows dropped) rather than
    silently producing empty output.
    """
    repo_root = Path(__file__).resolve().parents[2]
    converter = repo_root / "scripts" / "opencompass_cje" / "opencompass_to_cje.py"
    sample = Path(__file__).resolve().parent / "data" / "opencompass_ppl_format.json"

    out_json = tmp_path / "cje.json"

    code = _run_script(
        converter,
        [
            str(sample),
            "--out",
            str(out_json),
            "--no-label-template",
        ],
    )
    assert code == 1  # non-zero: all rows dropped


def test_unified_wrapper_opencompass(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    wrapper = repo_root / "scripts" / "cje_bridges" / "convert.py"
    sample = Path(__file__).resolve().parent / "data" / "opencompass_sample.json"

    out_json = tmp_path / "cje2.json"

    code = _run_script(
        wrapper,
        [
            "opencompass",
            str(sample),
            "--out",
            str(out_json),
            "--no-label-template",
        ],
    )
    assert code == 0

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "opencompass_sample" in payload
    assert len(payload["opencompass_sample"]) == 3


@pytest.mark.parametrize(
    "verdict",
    ["This is a tie.", "A is poor; B is the winner.", "Neither A nor B."],
)
def test_ambiguous_prose_is_not_scored(tmp_path: Path, verdict: str) -> None:
    wrapper = Path(__file__).resolve().parents[2] / "scripts/cje_bridges/convert.py"
    source, output = tmp_path / "judge.json", tmp_path / "out.json"
    source.write_text(
        json.dumps({"details": [{"origin_prompt": "Question", "prediction": verdict}]})
    )
    assert (
        _run_script(
            wrapper,
            ["opencompass", str(source), "--out", str(output), "--no-label-template"],
        )
        == 1
    )
    assert json.loads(output.read_text()) == {"judge": []}


def test_explicit_verdicts_remain_supported(tmp_path: Path) -> None:
    wrapper = Path(__file__).resolve().parents[2] / "scripts/cje_bridges/convert.py"
    source, output = tmp_path / "judge.json", tmp_path / "out.json"
    verdicts = ["A", "B", "Answer: A", "(B)", "choice=A", "Final answer: [B].", "0.7"]
    source.write_text(
        json.dumps(
            {
                "details": [
                    {"origin_prompt": str(i), "prediction": verdict}
                    for i, verdict in enumerate(verdicts)
                ]
            }
        )
    )
    assert (
        _run_script(
            wrapper,
            ["opencompass", str(source), "--out", str(output), "--no-label-template"],
        )
        == 0
    )
    assert [row["judge_score"] for row in json.loads(output.read_text())["judge"]] == [
        1,
        0,
        1,
        0,
        1,
        0,
        0.7,
    ]


def test_colliding_filenames_do_not_merge_policies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wrapper = Path(__file__).resolve().parents[2] / "scripts/cje_bridges/convert.py"
    source = tmp_path / "results"
    for model, score in [("model_a", 0.2), ("model_b", 0.8)]:
        directory = source / model
        directory.mkdir(parents=True)
        (directory / "benchmark.json").write_text(
            json.dumps(
                {"details": [{"origin_prompt": "Question", "prediction": score}]}
            )
        )
    output = tmp_path / "out.json"
    assert (
        _run_script(
            wrapper,
            ["opencompass", str(source), "--out", str(output), "--no-label-template"],
        )
        == 2
    )
    assert "--policy-name" in capsys.readouterr().err
    assert not output.exists()
    assert (
        _run_script(
            wrapper,
            [
                "opencompass",
                str(source / "model_a"),
                "--policy-name",
                "model_a",
                "--out",
                str(output),
                "--no-label-template",
            ],
        )
        == 0
    )
    assert set(json.loads(output.read_text())) == {"model_a"}
