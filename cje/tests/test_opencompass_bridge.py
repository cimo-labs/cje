from __future__ import annotations

import csv
import json
import runpy
import sys
from pathlib import Path


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
            "judge_score",
            "prompt",
            "oracle_label",
        ]
        csv_rows = list(reader)
    assert len(csv_rows) == 3


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
