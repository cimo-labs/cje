"""Tests for the 0.4.0 `cje analyze` surface.

`cje analyze PATH` takes fresh draws (a directory of per-policy response
files or a single JSONL file with target_policy per record) — the logged
dataset positional is gone (see test_migration_errors.py for the exit
behavior of the removed invocations). Also covers the --fresh-draws-dir
alias, --calibration-data, the logprob-ignored INFO note, and the
reliability-aware trophy logic (best_policy_lines), which 0.4.0 keeps
unchanged from 0.3.0.
"""

import json
import logging
import runpy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from cje.data.models import EstimationResult
from cje.diagnostics import DirectDiagnostics, Status
from cje.interface.cli import best_policy_lines, create_parser, main

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_cli(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["cje", *argv])
    return main()


def _fresh_records(
    policy: str,
    n: int = 30,
    with_oracle: bool = False,
    with_logprobs: bool = False,
    with_target_policy: bool = True,
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
        if with_target_policy:
            record["target_policy"] = policy
        if with_oracle:
            record["oracle_label"] = float(np.clip(score + rng.normal(0, 0.05), 0, 1))
        if with_logprobs:
            record["base_policy_logprob"] = -10.0
            record["target_policy_logprobs"] = {policy: -11.0}
        records.append(record)
    return records


def _write_fresh_draws_file(path: Path, policies: tuple, **kwargs: Any) -> None:
    records = [r for p in policies for r in _fresh_records(p, **kwargs)]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _write_fresh_draws_dir(directory: Path, policies: tuple, **kwargs: Any) -> None:
    directory.mkdir(exist_ok=True)
    for policy in policies:
        records = _fresh_records(policy, with_target_policy=False, **kwargs)
        (directory / f"{policy}_responses.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )


def _write_calibration_data(path: Path, n: int = 20, seed: int = 3) -> None:
    """Logged-style data with judge + oracle: the calibration source."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        score = float(np.clip(rng.uniform(0.05, 0.95), 0, 1))
        records.append(
            {
                "prompt_id": f"c{i}",
                "prompt": f"question {i}",
                "response": f"answer {i}",
                "base_policy_logprob": -10.0,
                "target_policy_logprobs": {"policy_a": -11.0},
                "judge_score": score,
                "oracle_label": float(np.clip(score + rng.normal(0, 0.05), 0, 1)),
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


# ---------------------------------------------------------------------------
# The analyze surface
# ---------------------------------------------------------------------------


class TestAnalyzeSurface:
    def test_python_m_cje_entrypoint_help(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["cje", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("cje.__main__", run_name="__main__")

        assert exc_info.value.code == 0
        assert "usage:" in capsys.readouterr().out

    def test_default_estimator_is_calibrated_direct(self) -> None:
        args = create_parser().parse_args(["analyze", "draws/"])
        assert args.estimator == "calibrated-direct"

    def test_scale_and_strict_options_parse(self) -> None:
        args = create_parser().parse_args(
            [
                "analyze",
                "draws/",
                "--fresh-judge-scale",
                "0",
                "100",
                "--calibration-oracle-scale",
                "1",
                "5",
                "--output-scale",
                "0",
                "10",
                "--strict",
            ]
        )
        assert args.fresh_judge_scale == [0.0, 100.0]
        assert args.calibration_oracle_scale == [1.0, 5.0]
        assert args.output_scale == [0.0, 10.0]
        assert args.strict is True
        assert args.on_invalid == "error"  # loud by default

    def test_on_invalid_option_parses(self) -> None:
        args = create_parser().parse_args(["analyze", "draws/", "--on-invalid", "drop"])
        assert args.on_invalid == "drop"

    def test_directory_positional_runs(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        draws = tmp_path / "responses"
        _write_fresh_draws_dir(draws, ("policy_a", "policy_b"), with_oracle=True)

        exit_code = _run_cli(monkeypatch, "analyze", str(draws))

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "policy_a" in out and "policy_b" in out
        assert "Best by point estimate" in out

    def test_file_positional_runs(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        draws_file = tmp_path / "fresh_draws.jsonl"
        _write_fresh_draws_file(draws_file, ("policy_a", "policy_b"), with_oracle=True)

        exit_code = _run_cli(monkeypatch, "analyze", str(draws_file))

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "policy_a" in out and "policy_b" in out

    def test_fresh_draws_dir_flag_is_kept_as_alias(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        draws = tmp_path / "responses"
        _write_fresh_draws_dir(draws, ("policy_a",), with_oracle=True)

        exit_code = _run_cli(monkeypatch, "analyze", "--fresh-draws-dir", str(draws))

        assert exit_code == 0
        assert "policy_a" in capsys.readouterr().out

    def test_calibration_data_option(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Old logged data (judge + oracle) drives calibration for
        judge-only fresh draws — the documented migration path."""
        draws = tmp_path / "responses"
        _write_fresh_draws_dir(draws, ("policy_a", "policy_b"), with_oracle=False)
        calibration = tmp_path / "logged_data.jsonl"
        _write_calibration_data(calibration)

        exit_code = _run_cli(
            monkeypatch,
            "analyze",
            str(draws),
            "--calibration-data",
            str(calibration),
            "--estimator-config",
            '{"inference_method": "cluster_robust"}',
        )

        assert exit_code == 0
        assert "policy_a" in capsys.readouterr().out

    def test_calibration_scale_options_run_end_to_end(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        draws = tmp_path / "responses"
        draws.mkdir()
        fresh = [{"prompt_id": f"p{i}", "judge_score": 100 * i / 19} for i in range(20)]
        (draws / "policy_a_responses.jsonl").write_text(
            "\n".join(json.dumps(record) for record in fresh) + "\n"
        )
        calibration = tmp_path / "calibration.jsonl"
        calibration.write_text(
            "\n".join(
                json.dumps(
                    {
                        "prompt_id": f"c{i}",
                        "judge_score": 100 * i / 19,
                        "oracle_label": 100 * i / 19,
                    }
                )
                for i in range(20)
            )
            + "\n"
        )

        exit_code = _run_cli(
            monkeypatch,
            "analyze",
            str(draws),
            "--calibration-data",
            str(calibration),
            "--fresh-judge-scale",
            "0",
            "100",
            "--calibration-judge-scale",
            "0",
            "100",
            "--calibration-oracle-scale",
            "0",
            "100",
            "--output-scale",
            "0",
            "100",
            "--estimator-config",
            '{"inference_method": "cluster_robust"}',
        )

        assert exit_code == 0
        assert "policy_a" in capsys.readouterr().out

    def test_single_file_invalid_rows_error_by_default_with_line_context(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Loud by default (matching analyze_dataset): an invalid record
        fails the run and names the file and line; --strict is the
        compatible spelling of the same default."""
        records = _fresh_records("policy_a", n=30, with_oracle=True)
        records.append(
            {
                "prompt_id": "missing-policy",
                "judge_score": 0.5,
                "oracle_label": 0.5,
            }
        )
        path = tmp_path / "fresh_draws.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

        assert _run_cli(monkeypatch, "analyze", str(path)) == 1
        err = capsys.readouterr().err
        assert "missing 'target_policy'" in err
        assert f"{path}:31" in err

        assert _run_cli(monkeypatch, "analyze", str(path), "--strict") == 1
        assert "missing 'target_policy'" in capsys.readouterr().err

    def test_single_file_corrupt_json_line_errors_by_default(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        records = _fresh_records("policy_a", n=30, with_oracle=True)
        path = tmp_path / "fresh_draws.jsonl"
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n{not json\n"
        )

        assert _run_cli(monkeypatch, "analyze", str(path)) == 1
        err = capsys.readouterr().err
        assert f"Invalid JSON at {path}:31" in err

    def test_single_file_on_invalid_drop_opts_in_and_reports_count(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        records = _fresh_records("policy_a", n=30, with_oracle=True)
        records.append(
            {
                "prompt_id": "missing-policy",
                "judge_score": 0.5,
                "oracle_label": 0.5,
            }
        )
        path = tmp_path / "fresh_draws.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

        exit_code = _run_cli(monkeypatch, "analyze", str(path), "--on-invalid", "drop")

        assert exit_code == 0
        assert "Dropped 1 invalid record(s)" in capsys.readouterr().out

    def test_strict_conflicts_with_on_invalid_drop(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / "fresh_draws.jsonl"
        _write_fresh_draws_file(path, ("policy_a",), with_oracle=True)

        exit_code = _run_cli(
            monkeypatch, "analyze", str(path), "--strict", "--on-invalid", "drop"
        )

        assert exit_code == 1
        assert "--strict conflicts with --on-invalid drop" in capsys.readouterr().err

    def test_transport_probe_and_margin_options(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        draws = tmp_path / "responses"
        draws.mkdir()
        records = [
            {
                "prompt_id": f"p{i}",
                "judge_score": 0.2 + 0.4 * i / 29,
                "oracle_label": 0.2 + 0.4 * i / 29,
                "draw_idx": 0,
            }
            for i in range(30)
        ]
        (draws / "policy_a_responses.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n"
        )
        probes = [
            {
                "prompt_id": f"probe-{i}",
                "judge_score": 0.2 + 0.4 * i / 29,
                "oracle_label": 0.2 + 0.4 * i / 29,
            }
            for i in range(30)
        ]
        probe_path = tmp_path / "policy_a_probe.jsonl"
        probe_path.write_text("\n".join(json.dumps(record) for record in probes) + "\n")

        exit_code = _run_cli(
            monkeypatch,
            "analyze",
            str(draws),
            "--transport-probe",
            f"policy_a={probe_path}",
            "--transport-margin",
            "policy_a=0.05",
            "--estimator-config",
            '{"inference_method": "cluster_robust"}',
        )

        assert exit_code == 0
        assert "residual transport: PASS" in capsys.readouterr().out

    def test_logprob_fields_in_file_ignored_with_info_line(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        draws_file = tmp_path / "fresh_draws.jsonl"
        _write_fresh_draws_file(
            draws_file,
            ("policy_a", "policy_b"),
            with_oracle=True,
            with_logprobs=True,
        )

        with caplog.at_level(logging.INFO):
            exit_code = _run_cli(monkeypatch, "analyze", str(draws_file))

        assert exit_code == 0
        assert any(
            "logprob fields present and ignored (Direct mode)" == r.message
            for r in caplog.records
        )

    def test_logprob_fields_in_dir_ignored_with_info_line(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        draws = tmp_path / "responses"
        _write_fresh_draws_dir(
            draws, ("policy_a",), with_oracle=True, with_logprobs=True
        )

        with caplog.at_level(logging.INFO):
            exit_code = _run_cli(monkeypatch, "analyze", str(draws))

        assert exit_code == 0
        assert any(
            "logprob fields present and ignored (Direct mode)" == r.message
            for r in caplog.records
        )

    def test_no_logprob_fields_no_info_line(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        draws = tmp_path / "responses"
        _write_fresh_draws_dir(draws, ("policy_a",), with_oracle=True)

        with caplog.at_level(logging.INFO):
            exit_code = _run_cli(monkeypatch, "analyze", str(draws))

        assert exit_code == 0
        # Match the exact note: log messages can echo file paths (e.g. this
        # test's own tmp dir name contains "logprob").
        assert not any(
            "logprob fields present and ignored" in r.message for r in caplog.records
        )

    def test_missing_path_argument_errors(
        self,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        exit_code = _run_cli(monkeypatch, "analyze")

        assert exit_code == 1
        assert "Provide fresh draws" in capsys.readouterr().err

    def test_nonexistent_path_errors(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        exit_code = _run_cli(monkeypatch, "analyze", str(tmp_path / "no_such_path"))

        assert exit_code == 1
        assert "no_such_path" in capsys.readouterr().err

    def test_file_without_target_policy_or_logprobs_errors(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Judge-scored records without target_policy are ambiguous — the
        error must name the missing field, not the migration copy."""
        draws_file = tmp_path / "fresh_draws.jsonl"
        _write_fresh_draws_file(
            draws_file, ("policy_a",), with_target_policy=False, with_oracle=True
        )

        exit_code = _run_cli(monkeypatch, "analyze", str(draws_file))

        assert exit_code == 1
        err = capsys.readouterr().err
        assert "target_policy" in err
        assert "Off-policy evaluation was removed" not in err

    def test_output_writes_results_json(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        draws = tmp_path / "responses"
        _write_fresh_draws_dir(draws, ("policy_a",), with_oracle=True)
        output = tmp_path / "results.json"

        exit_code = _run_cli(
            monkeypatch, "analyze", str(draws), "--output", str(output), "--quiet"
        )

        assert exit_code == 0
        payload = json.loads(output.read_text())
        assert payload  # non-empty results document


# ---------------------------------------------------------------------------
# Reliability-aware trophy logic (kept unchanged from 0.3.0; tests restored
# from the retired test_refusal_gates.py)
# ---------------------------------------------------------------------------


def _make_direct_diagnostics(
    policies: List[str],
    estimates: Dict[str, float],
    standard_errors: Dict[str, float],
    status_per_policy: Dict[str, Status],
) -> DirectDiagnostics:
    return DirectDiagnostics(
        estimator_type="Direct",
        method="calibrated_direct",
        n_samples_total=100,
        n_samples_valid=100,
        policies=policies,
        estimates=estimates,
        standard_errors=standard_errors,
        n_samples_used={p: 100 for p in policies},
        status_per_policy=status_per_policy,
    )


class TestCLIBestPolicy:
    """The CLI must not crown a gate-flagged policy (0.3.0 finding #8)."""

    @staticmethod
    def _make_results(
        estimates: list,
        policies: list,
        gates: Optional[dict] = None,
        diagnostics: Optional[Any] = None,
    ) -> EstimationResult:
        metadata: dict = {"target_policies": policies}
        if gates is not None:
            metadata["reliability_gates"] = gates
        return EstimationResult(
            estimates=np.array(estimates, dtype=float),
            standard_errors=np.full(len(estimates), 0.05),
            n_samples_used={p: 100 for p in policies},
            method="calibrated_direct",
            influence_functions={},
            diagnostics=diagnostics,
            metadata=metadata,
        )

    def test_reliable_argmax_gets_trophy(self) -> None:
        results = self._make_results(
            [0.5, 0.7],
            ["base", "good"],
            gates={
                "base": {"flagged": False, "refused": False, "reasons": []},
                "good": {"flagged": False, "refused": False, "reasons": []},
            },
        )
        lines = best_policy_lines(results)
        assert lines == [
            "Best by point estimate: good",
            "Limitations: residual transport NOT_CHECKED",
        ]

    def test_flagged_argmax_shown_with_reliable_fallback(self) -> None:
        # The verified 0.3.0 repro: adversarial 'unhelpful' wins the raw
        # argmax while flagged UNRELIABLE by the refusal gates. The CLI
        # shows the raw winner qualified, and announces the reliable
        # winner best_policy() returned (reliable_only=True default).
        results = self._make_results(
            [0.771, 0.756, 0.763],
            ["unhelpful", "base", "clone"],
            gates={
                "unhelpful": {
                    "flagged": True,
                    "refused": False,
                    "reasons": ["raw_near_zero=90.2%"],
                },
                "base": {"flagged": False, "refused": False, "reasons": []},
                "clone": {"flagged": False, "refused": False, "reasons": []},
            },
        )
        lines = best_policy_lines(results)
        assert lines[0] == "Best by point estimate: unhelpful"
        assert any("reliability gates flagged" in line for line in lines)
        assert any(
            "Best reliable policy: clone" in line and "reliable_only=False" in line
            for line in lines
        )
        assert any("raw_near_zero=90.2%" in line for line in lines)

    def test_critical_status_demotes_point_winner(self) -> None:
        diag = _make_direct_diagnostics(
            policies=["bad", "ok"],
            estimates={"bad": 0.9, "ok": 0.6},
            standard_errors={"bad": 0.1, "ok": 0.05},
            status_per_policy={"bad": Status.CRITICAL, "ok": Status.GOOD},
        )
        results = self._make_results([0.9, 0.6], ["bad", "ok"], diagnostics=diag)
        lines = best_policy_lines(results)
        assert lines[0] == "Best by point estimate: bad"
        assert any("reliability gates flagged" in line for line in lines)
        assert any("Best reliable policy: ok" in line for line in lines)

    def test_all_flagged_no_winner(self) -> None:
        results = self._make_results(
            [0.9, 0.6],
            ["a", "b"],
            gates={
                "a": {"flagged": True, "refused": False, "reasons": ["ESS=5%"]},
                "b": {"flagged": True, "refused": False, "reasons": ["ESS=8%"]},
            },
        )
        lines = best_policy_lines(results)
        assert any("no policy passed" in line for line in lines)

    def test_all_nan_estimates(self) -> None:
        results = self._make_results([float("nan"), float("nan")], ["a", "b"])
        lines = best_policy_lines(results)
        assert any("every point estimate is NaN" in line for line in lines)
