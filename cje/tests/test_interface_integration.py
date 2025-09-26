"""Integration tests for the high-level interface.

These tests validate that the public interface chooses sensible defaults
and runs end-to-end on real arena sample data.
"""

import os
from pathlib import Path

import pytest

from cje.interface.service import AnalysisService
from cje.interface.config import AnalysisConfig
from cje.interface.analysis import analyze_dataset
from cje.interface.cli import create_parser, run_analysis


pytestmark = [pytest.mark.integration, pytest.mark.uses_arena_sample]


def _arena_paths() -> tuple[Path, Path]:
    """Return (dataset_path, responses_dir) under the repo test data tree."""
    here = Path(__file__).parent
    dataset_path = here / "data" / "arena_sample" / "dataset.jsonl"
    responses_dir = here / "data" / "arena_sample" / "responses"
    if not dataset_path.exists():
        pytest.skip(f"Arena sample not found: {dataset_path}")
    return dataset_path, responses_dir


def test_analyze_dataset_ips_path_works() -> None:
    """analyze_dataset runs with a dataset path and returns valid results (IPS)."""
    dataset_path, _ = _arena_paths()

    results = analyze_dataset(
        dataset_path=str(dataset_path),
        estimator="calibrated-ips",
        verbose=False,
    )

    assert results is not None
    assert "target_policies" in results.metadata
    assert len(results.estimates) == len(results.metadata["target_policies"])
    assert results.method in ("calibrated_ips", "raw_ips")


def test_service_auto_selects_calibrated_ips_without_fresh_draws() -> None:
    """Service chooses calibrated-ips when no fresh draws are provided (auto)."""
    dataset_path, _ = _arena_paths()

    svc = AnalysisService()
    cfg = AnalysisConfig(
        dataset_path=str(dataset_path),
        judge_field="judge_score",
        oracle_field="oracle_label",
        estimator="auto",
        fresh_draws_dir=None,
        estimator_config={},
        verbose=False,
    )
    results = svc.run(cfg)

    assert results.metadata.get("estimator") == "calibrated-ips"
    assert len(results.estimates) > 0


def test_service_auto_selects_stacked_dr_with_fresh_draws() -> None:
    """Service chooses stacked-dr when fresh draws directory is provided (auto)."""
    dataset_path, responses_dir = _arena_paths()

    svc = AnalysisService()
    cfg = AnalysisConfig(
        dataset_path=str(dataset_path),
        judge_field="judge_score",
        oracle_field="oracle_label",
        estimator="auto",
        fresh_draws_dir=str(responses_dir),
        # Disable parallelism in tests to avoid resource contention
        estimator_config={"parallel": False},
        verbose=False,
    )
    results = svc.run(cfg)

    assert results.metadata.get("estimator") == "stacked-dr"
    assert len(results.estimates) > 0


def test_cli_analyze_ips_quiet() -> None:
    """CLI 'analyze' runs with calibrated-ips and returns code 0."""
    dataset_path, _ = _arena_paths()

    parser = create_parser()
    args = parser.parse_args(
        [
            "analyze",
            str(dataset_path),
            "--estimator",
            "calibrated-ips",
            "-q",
        ]
    )

    code = run_analysis(args)
    assert code == 0


def test_cli_analyze_auto_with_fresh_draws_quiet() -> None:
    """CLI 'analyze' defaults to stacked-dr when fresh draws dir is provided."""
    dataset_path, responses_dir = _arena_paths()

    parser = create_parser()
    args = parser.parse_args(
        [
            "analyze",
            str(dataset_path),
            "--fresh-draws-dir",
            str(responses_dir),
            "-q",
        ]
    )

    code = run_analysis(args)
    assert code == 0
