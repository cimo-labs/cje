"""Tests for the redesigned `cje validate` (0.4.0).

The 0.2.x-era implementation round-tripped records through the Dataset
model before validating: the round-trip regenerated prompt_ids and
relocated judge/oracle fields, so `cje validate` false-failed ALL valid
data (and crashed outright on fresh-draws files, which have no
target_policy_logprobs). 0.4.0 validates the RAW parsed JSONL via
validate_direct_data: fresh-draws directories and single files exit 0,
per-policy sample + oracle counts are reported, and logged-style files
with judge + oracle pairs validate as calibration sources with an
informational ignored-logprobs note.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from cje.data.validation import NOTE_PREFIX, validate_direct_data
from cje.interface.cli import main

pytestmark = pytest.mark.unit

ARENA_SAMPLE = Path(__file__).parent.parent.parent / "examples" / "arena_sample"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_cli(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["cje", *argv])
    return main()


def _fresh_records(
    policy: str,
    n: int = 30,
    with_oracle: bool = True,
    seed: int = 7,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed + sum(ord(c) for c in policy))
    records = []
    for i in range(n):
        score = float(np.clip(rng.uniform(0.05, 0.95), 0, 1))
        record: Dict[str, Any] = {
            "prompt_id": f"p{i}",
            "judge_score": score,
            "draw_idx": 0,
        }
        if with_oracle:
            record["oracle_label"] = float(np.clip(score + rng.normal(0, 0.05), 0, 1))
        records.append(record)
    return records


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _write_fresh_draws_dir(directory: Path, policies: tuple, **kwargs: Any) -> None:
    directory.mkdir(exist_ok=True)
    for policy in policies:
        _write_jsonl(
            directory / f"{policy}_responses.jsonl", _fresh_records(policy, **kwargs)
        )


# ---------------------------------------------------------------------------
# The CLI surface
# ---------------------------------------------------------------------------


class TestValidateCLI:
    @pytest.mark.skipif(
        not (ARENA_SAMPLE / "fresh_draws").exists(),
        reason="Arena sample data not available",
    )
    def test_arena_fresh_draws_dir_exits_0(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The 0.2.x-era bug: `cje validate` false-failed ALL valid data.
        The bundled arena fresh draws must validate clean."""
        exit_code = _run_cli(monkeypatch, "validate", str(ARENA_SAMPLE / "fresh_draws"))

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "ready" in out
        # Per-policy sample + oracle counts are reported
        assert "base: 1000 samples, 480 oracle labels" in out
        assert "clone: 1000 samples, 0 oracle labels" in out

    @pytest.mark.skipif(
        not (ARENA_SAMPLE / "logged_data.jsonl").exists(),
        reason="Arena sample data not available",
    )
    def test_arena_logged_data_is_a_valid_calibration_source(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A logged-style file (judge + oracle, logprob fields, no
        target_policy) is valid as a calibration source — with the
        informational ignored-logprobs note, not a failure."""
        exit_code = _run_cli(
            monkeypatch, "validate", str(ARENA_SAMPLE / "logged_data.jsonl")
        )

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "logprob fields present and ignored (Direct mode)" in out
        assert "calibration source" in out
        assert "--calibration-data" in out

    def test_synthetic_fresh_draws_dir_exits_0(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        draws = tmp_path / "responses"
        _write_fresh_draws_dir(draws, ("policy_a", "policy_b"), n=30)

        exit_code = _run_cli(monkeypatch, "validate", str(draws))

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "ready" in out
        assert "policy_a: 30 samples, 30 oracle labels" in out

    def test_missing_prompt_id_is_named(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        records = _fresh_records("policy_a", n=20)
        for record in records:
            del record["prompt_id"]
            record["target_policy"] = "policy_a"
        draws_file = tmp_path / "fresh_draws.jsonl"
        _write_jsonl(draws_file, records)

        exit_code = _run_cli(monkeypatch, "validate", str(draws_file))

        assert exit_code == 1
        assert "prompt_id" in capsys.readouterr().out

    def test_non_numeric_judge_score_is_an_issue(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        records = _fresh_records("policy_a", n=20)
        for record in records:
            record["judge_score"] = "great"
            record["target_policy"] = "policy_a"
        draws_file = tmp_path / "fresh_draws.jsonl"
        _write_jsonl(draws_file, records)

        exit_code = _run_cli(monkeypatch, "validate", str(draws_file))

        assert exit_code == 1
        out = capsys.readouterr().out
        assert "judge_score" in out
        assert "non-numeric" in out

    def test_logged_style_synthetic_file_validates_with_note(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        records = _fresh_records("calib", n=60)
        for record in records:
            record["base_policy_logprob"] = -10.0
            record["target_policy_logprobs"] = {"policy_a": -11.0}
        calib_file = tmp_path / "logged_data.jsonl"
        _write_jsonl(calib_file, records)

        exit_code = _run_cli(monkeypatch, "validate", str(calib_file))

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "logprob fields present and ignored (Direct mode)" in out
        assert "calibration source" in out

    def test_blank_lines_are_fine(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        records = _fresh_records("policy_a", n=20)
        for record in records:
            record["target_policy"] = "policy_a"
        draws_file = tmp_path / "fresh_draws.jsonl"
        lines = [json.dumps(r) for r in records]
        lines.insert(5, "")  # blank line mid-file
        draws_file.write_text("\n".join(lines) + "\n\n")  # + trailing blank

        assert _run_cli(monkeypatch, "validate", str(draws_file)) == 0

    def test_invalid_json_line_reports_file_and_line(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        draws_file = tmp_path / "fresh_draws.jsonl"
        draws_file.write_text('{"prompt_id": "p0", "judge_score": 0.5}\n{broken\n')

        exit_code = _run_cli(monkeypatch, "validate", str(draws_file))

        assert exit_code == 1
        err = capsys.readouterr().err
        assert f"{draws_file}:2" in err

    def test_nonexistent_path_errors(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        exit_code = _run_cli(monkeypatch, "validate", str(tmp_path / "no_such_path"))

        assert exit_code == 1
        assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# validate_direct_data (the library function)
# ---------------------------------------------------------------------------


class TestValidateDirectData:
    def test_empty_data_is_invalid(self) -> None:
        is_valid, issues = validate_direct_data([])
        assert not is_valid
        assert issues == ["Data is empty"]

    def test_valid_records_pass(self) -> None:
        records = _fresh_records("policy_a", n=60)
        is_valid, issues = validate_direct_data(records)
        assert is_valid, issues
        assert issues == []

    def test_mixed_target_policy_presence_is_an_issue(self) -> None:
        records = _fresh_records("policy_a", n=20)
        for record in records[:10]:
            record["target_policy"] = "policy_a"
        is_valid, issues = validate_direct_data(records)
        assert not is_valid
        assert any("target_policy" in issue for issue in issues)

    def test_zero_oracle_labels_is_an_issue_pointing_at_calibration_data(
        self,
    ) -> None:
        records = _fresh_records("policy_a", n=30, with_oracle=False)
        is_valid, issues = validate_direct_data(records)
        assert not is_valid
        assert any(
            "No oracle labels" in issue and "calibration" in issue for issue in issues
        )

    def test_under_ten_oracle_labels_is_an_issue(self) -> None:
        records = _fresh_records("policy_a", n=30, with_oracle=False)
        for record in records[:5]:
            record["oracle_label"] = 0.5
        is_valid, issues = validate_direct_data(records)
        assert not is_valid
        assert any("Too few oracle samples (5)" in issue for issue in issues)

    def test_ten_to_fifty_oracle_labels_is_a_note_not_an_issue(self) -> None:
        records = _fresh_records("policy_a", n=100, with_oracle=False)
        for record in records[:20]:
            record["oracle_label"] = 0.5
        is_valid, issues = validate_direct_data(records)
        assert is_valid
        assert any(
            issue.startswith(NOTE_PREFIX) and "20 oracle samples" in issue
            for issue in issues
        )

    def test_oracle_counts_pool_across_policies(self) -> None:
        """Calibration pools oracle labels across policies' draws (the
        arena layout: only one policy's file carries labels)."""
        labeled = _fresh_records("policy_a", n=30, with_oracle=True)
        unlabeled = _fresh_records("policy_b", n=30, with_oracle=False)
        for record in labeled:
            record["target_policy"] = "policy_a"
        for record in unlabeled:
            record["target_policy"] = "policy_b"
        is_valid, issues = validate_direct_data(labeled + unlabeled)
        assert is_valid, issues

    def test_logprob_fields_are_a_note_and_do_not_fail(self) -> None:
        records = _fresh_records("policy_a", n=60)
        for record in records:
            record["base_policy_logprob"] = -10.0
            record["target_policy_logprobs"] = {"policy_a": -11.0}
        is_valid, issues = validate_direct_data(records)
        assert is_valid
        assert any(
            "logprob fields present and ignored (Direct mode)" in issue
            for issue in issues
        )

    def test_judge_score_in_metadata_is_accepted(self) -> None:
        records = []
        for i in range(60):
            records.append(
                {
                    "prompt_id": f"p{i}",
                    "metadata": {"judge_score": 0.5, "oracle_label": 0.5},
                }
            )
        is_valid, issues = validate_direct_data(records)
        assert is_valid, issues

    def test_non_numeric_oracle_label_is_an_issue(self) -> None:
        records = _fresh_records("policy_a", n=30)
        records[3]["oracle_label"] = "good"
        is_valid, issues = validate_direct_data(records)
        assert not is_valid
        assert any(
            "oracle_label" in issue and "non-numeric" in issue for issue in issues
        )
