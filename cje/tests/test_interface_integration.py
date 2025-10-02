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
    """Return (dataset_path, responses_dir) from examples directory."""
    here = Path(__file__).parent
    # Point to examples directory (shared with tutorials)
    dataset_path = here.parent.parent / "examples" / "arena_sample" / "dataset.jsonl"
    responses_dir = here.parent.parent / "examples" / "arena_sample" / "responses"
    if not dataset_path.exists():
        pytest.skip(f"Arena sample not found: {dataset_path}")
    return dataset_path, responses_dir


def test_analyze_dataset_ips_path_works() -> None:
    """analyze_dataset runs with a dataset path and returns valid results (IPS)."""
    dataset_path, _ = _arena_paths()

    results = analyze_dataset(
        logged_data_path=str(dataset_path),
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
        logged_data_path=str(dataset_path),
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
        logged_data_path=str(dataset_path),
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


def test_stacked_dr_without_fresh_draws_raises_helpful_error() -> None:
    """Stacked-DR requires fresh draws - ensure clear error message."""
    dataset_path, _ = _arena_paths()

    with pytest.raises(ValueError, match="DR estimators require fresh draws"):
        analyze_dataset(
            logged_data_path=str(dataset_path),
            estimator="stacked-dr",
            fresh_draws_dir=None,  # Missing!
        )


def test_mode_detection_three_modes() -> None:
    """Test that mode detection correctly identifies all three modes."""
    from cje.interface.mode_detection import detect_analysis_mode
    from cje.data.models import Dataset, Sample

    # Case 1: Dataset with logprobs only (IPS mode)
    samples_with_logprobs = [
        Sample(
            prompt_id=f"p{i}",
            prompt="test",
            response="response",
            reward=0.5 + i * 0.05,
            base_policy_logprob=-1.0,
            target_policy_logprobs={"policy_a": -1.5, "policy_b": -2.0},
            metadata={"judge_score": 0.5 + i * 0.05},
        )
        for i in range(10)
    ]
    dataset_ips = Dataset(
        samples=samples_with_logprobs,
        target_policies=["policy_a", "policy_b"],
    )

    mode, explanation = detect_analysis_mode(dataset_ips, fresh_draws_dir=None)
    assert mode == "calibrated-ips"
    assert "IPS mode" in explanation
    assert "100.0% of samples have valid logprobs" in explanation

    # Case 2: Dataset with no logprobs but fresh draws directory (Direct mode with calibration)
    samples_no_logprobs = [
        Sample(
            prompt_id=f"p{i}",
            prompt="test",
            response="response",
            reward=0.5 + i * 0.05,
            base_policy_logprob=None,
            # Include policies in dict but with None values
            target_policy_logprobs={"policy_a": None, "policy_b": None},
            metadata={"judge_score": 0.5 + i * 0.05, "policy": "policy_a"},
        )
        for i in range(10)
    ]
    dataset_no_logprobs = Dataset(
        samples=samples_no_logprobs,
        target_policies=["policy_a", "policy_b"],
    )

    # Dataset with no logprobs but fresh draws should select Direct mode
    dataset_path, responses_dir = _arena_paths()
    mode, explanation = detect_analysis_mode(
        dataset_no_logprobs, fresh_draws_dir=str(responses_dir)
    )
    assert mode == "direct"
    assert "Direct mode with calibration" in explanation

    # Case 3: Dataset with logprobs AND fresh draws directory (DR mode)
    dataset_path, responses_dir = _arena_paths()

    mode, explanation = detect_analysis_mode(
        dataset_ips, fresh_draws_dir=str(responses_dir)
    )
    assert mode == "stacked-dr"
    assert "DR mode" in explanation
    assert "combines importance weighting with outcome models" in explanation


def test_mode_detection_insufficient_data() -> None:
    """Test that mode detection raises clear error when data is insufficient."""
    from cje.interface.mode_detection import detect_analysis_mode
    from cje.data.models import Dataset, Sample

    # Dataset with no logprobs, no rewards, and no fresh draws
    samples_insufficient = [
        Sample(
            prompt_id=f"p{i}",
            prompt="test",
            response="response",
            reward=None,  # No rewards!
            base_policy_logprob=None,
            target_policy_logprobs={"policy_a": None},
            metadata={"judge_score": 0.5},
        )
        for i in range(10)
    ]
    dataset = Dataset(
        samples=samples_insufficient,
        target_policies=["policy_a"],
    )

    with pytest.raises(ValueError, match="Insufficient data"):
        detect_analysis_mode(dataset, fresh_draws_dir=None)


def test_direct_mode_with_explicit_estimator() -> None:
    """Test that direct estimator can be explicitly selected with fresh draws."""
    dataset_path, responses_dir = _arena_paths()

    # Explicitly select direct mode (requires fresh_draws_dir)
    results = analyze_dataset(
        logged_data_path=str(dataset_path),
        fresh_draws_dir=str(responses_dir),
        estimator="direct",
        verbose=False,
    )

    assert results is not None
    assert results.metadata.get("mode") == "direct"
    assert (
        results.metadata.get("estimand") == "on-policy evaluation on provided prompts"
    )
    assert len(results.estimates) > 0
    assert "target_policies" in results.metadata


def test_direct_mode_without_fresh_draws_raises_error() -> None:
    """Test that direct mode requires either fresh_draws_dir or logged_data."""
    # Direct mode with logged data but no fresh draws should error
    dataset_path, _ = _arena_paths()

    with pytest.raises(ValueError, match="Direct mode requires fresh_draws_dir"):
        analyze_dataset(
            logged_data_path=str(dataset_path),
            estimator="direct",
            fresh_draws_dir=None,  # Missing!
            verbose=False,
        )


def test_direct_only_mode_works() -> None:
    """Test that Direct-only mode works with just fresh_draws_dir (no logged data)."""
    _, responses_dir = _arena_paths()

    # Direct-only mode: fresh draws without logged data
    results = analyze_dataset(
        fresh_draws_dir=str(responses_dir),
        estimator="auto",  # Should auto-select "direct"
        verbose=False,
    )

    assert results is not None
    assert results.metadata.get("mode") == "direct"
    assert (
        results.metadata.get("calibration") == "none"
    )  # No calibration without logged data
    assert len(results.estimates) > 0
    assert "target_policies" in results.metadata
