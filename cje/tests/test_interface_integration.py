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
    """Return (dataset_path, fresh_draws_dir) from examples directory."""
    here = Path(__file__).parent
    # Point to examples directory (shared with tutorials)
    dataset_path = (
        here.parent.parent / "examples" / "arena_sample" / "logged_data.jsonl"
    )
    fresh_draws_dir = here.parent.parent / "examples" / "arena_sample" / "fresh_draws"
    if not dataset_path.exists():
        pytest.skip(f"Arena sample not found: {dataset_path}")
    return dataset_path, fresh_draws_dir


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
        calibration_data_path=None,
        combine_oracle_sources=True,
        calibration_covariates=None,
        include_response_length=False,
        estimator_config={},
        verbose=False,
    )
    results = svc.run(cfg)

    assert results.metadata.get("mode") == "ips"
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
        calibration_data_path=None,
        combine_oracle_sources=True,
        calibration_covariates=None,
        include_response_length=False,
        # Disable parallelism in tests to avoid resource contention
        estimator_config={"parallel": False},
        verbose=False,
    )
    results = svc.run(cfg)

    assert results.metadata.get("mode") == "dr"
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
            judge_score=0.5 + i * 0.05,
            oracle_label=None,
            metadata={},
        )
        for i in range(10)
    ]
    dataset_ips = Dataset(
        samples=samples_with_logprobs,
        target_policies=["policy_a", "policy_b"],
    )

    mode, explanation, coverage = detect_analysis_mode(
        dataset_ips, fresh_draws_dir=None
    )
    assert mode == "ips"
    assert "IPS mode" in explanation
    assert "100.0% of samples have valid logprobs" in explanation
    assert coverage == 1.0  # 100% coverage

    # Case 2: Dataset with no logprobs but fresh draws directory (DR mode - degrades to outcome-only)
    samples_no_logprobs = [
        Sample(
            prompt_id=f"p{i}",
            prompt="test",
            response="response",
            reward=0.5 + i * 0.05,
            base_policy_logprob=None,
            # Include policies in dict but with None values
            target_policy_logprobs={"policy_a": None, "policy_b": None},
            judge_score=0.5 + i * 0.05,
            oracle_label=None,
            metadata={"policy": "policy_a"},
        )
        for i in range(10)
    ]
    dataset_no_logprobs = Dataset(
        samples=samples_no_logprobs,
        target_policies=["policy_a", "policy_b"],
    )

    # Dataset with no logprobs but fresh draws should select DR mode (but warn it's outcome-only)
    dataset_path, responses_dir = _arena_paths()
    mode, explanation, coverage = detect_analysis_mode(
        dataset_no_logprobs, fresh_draws_dir=str(responses_dir)
    )
    assert mode == "dr"
    assert "DR mode" in explanation
    assert (
        "No valid logprobs" in explanation or "equivalent to Direct mode" in explanation
    )
    assert coverage == 0.0  # No logprobs

    # Case 3: Dataset with logprobs AND fresh draws directory (DR mode)
    dataset_path, responses_dir = _arena_paths()

    mode, explanation, coverage = detect_analysis_mode(
        dataset_ips, fresh_draws_dir=str(responses_dir)
    )
    assert mode == "dr"
    assert "DR mode" in explanation
    assert "Combining importance weighting with outcome models" in explanation
    assert coverage == 1.0  # 100% coverage


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
            judge_score=0.5,
            oracle_label=None,
            metadata={},
        )
        for i in range(10)
    ]
    dataset = Dataset(
        samples=samples_insufficient,
        target_policies=["policy_a"],
    )

    with pytest.raises(ValueError, match="Insufficient data"):
        detect_analysis_mode(dataset, fresh_draws_dir=None)


def test_mode_selection_metadata_populated() -> None:
    """Test that mode_selection metadata is properly populated in results."""
    dataset_path, responses_dir = _arena_paths()

    # Test auto mode with DR selection
    result = analyze_dataset(
        logged_data_path=str(dataset_path),
        fresh_draws_dir=str(responses_dir),
        estimator="auto",
        verbose=False,
    )

    # Verify mode_selection metadata exists
    assert "mode_selection" in result.metadata
    mode_sel = result.metadata["mode_selection"]

    # Verify all required fields
    assert "mode" in mode_sel
    assert "estimator" in mode_sel
    assert "logprob_coverage" in mode_sel
    assert "has_fresh_draws" in mode_sel
    assert "has_logged_data" in mode_sel
    assert "reason" in mode_sel

    # Verify correct values for DR mode
    assert mode_sel["mode"] == "dr"
    assert mode_sel["estimator"] == "stacked-dr"  # Default for DR mode
    assert mode_sel["has_fresh_draws"] is True
    assert mode_sel["has_logged_data"] is True
    assert mode_sel["logprob_coverage"] is not None  # Should be computed

    # Test IPS mode (no fresh draws)
    result_ips = analyze_dataset(
        logged_data_path=str(dataset_path),
        estimator="auto",
        verbose=False,
    )

    mode_sel_ips = result_ips.metadata["mode_selection"]
    assert mode_sel_ips["mode"] == "ips"
    assert mode_sel_ips["estimator"] == "calibrated-ips"  # Default for IPS mode
    assert mode_sel_ips["has_fresh_draws"] is False
    assert mode_sel_ips["has_logged_data"] is True


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
    # Fresh draws now include base policy with oracle labels (48% coverage)
    # So calibration should be "from_fresh_draws" using AutoCal-R
    assert results.metadata.get("calibration") == "from_fresh_draws"
    assert results.metadata.get("oracle_coverage", 0) > 0
    assert len(results.estimates) > 0
    assert "target_policies" in results.metadata


def test_three_modes_estimate_clone_accurately() -> None:
    """All three modes should accurately estimate clone policy value.

    Clone policy is nearly identical to the base policy (same model, same prompts).
    Ground truth is the mean calibrated reward in the logged data.
    Each mode should estimate within 0.05 of this truth.

    This tests:
    - Estimation accuracy against ground truth for A/A-like scenario
    - Each mode independently produces correct estimates
    - Regression protection for all three mode implementations
    """
    from cje.data import load_dataset_from_jsonl
    import numpy as np

    dataset_path, fresh_draws_dir = _arena_paths()

    # Calculate ground truth: mean calibrated reward in logged data
    # All three modes should use the same calibration (learned from logged data's oracle labels)
    from cje.calibration import calibrate_dataset

    dataset = load_dataset_from_jsonl(str(dataset_path))
    calibrated_dataset, _ = calibrate_dataset(
        dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
        enable_cross_fit=True,
        n_folds=5,
    )
    ground_truth = float(
        np.mean([s.reward for s in calibrated_dataset.samples if s.reward is not None])
    )

    # Run IPS mode (logged data only)
    results_ips = analyze_dataset(
        logged_data_path=str(dataset_path),
        estimator="calibrated-ips",
        verbose=False,
    )

    # Run DR mode (logged + fresh draws)
    results_dr = analyze_dataset(
        logged_data_path=str(dataset_path),
        fresh_draws_dir=str(fresh_draws_dir),
        estimator="stacked-dr",
        verbose=False,
    )

    # Run Direct mode (fresh draws for estimation, logged data for calibration)
    results_direct = analyze_dataset(
        logged_data_path=str(
            dataset_path
        ),  # Use logged data's oracle labels for calibration
        fresh_draws_dir=str(fresh_draws_dir),
        estimator="direct",  # Force direct mode (auto would select DR with both data sources)
        verbose=False,
    )

    # Find clone policy index in each result
    clone_idx_ips = results_ips.metadata["target_policies"].index("clone")
    clone_idx_dr = results_dr.metadata["target_policies"].index("clone")
    clone_idx_direct = results_direct.metadata["target_policies"].index("clone")

    # Extract clone estimates
    clone_ips = float(results_ips.estimates[clone_idx_ips])
    clone_dr = float(results_dr.estimates[clone_idx_dr])
    clone_direct = float(results_direct.estimates[clone_idx_direct])

    # All estimates should be valid
    assert 0 <= clone_ips <= 1, f"IPS estimate {clone_ips} out of range"
    assert 0 <= clone_dr <= 1, f"DR estimate {clone_dr} out of range"
    assert 0 <= clone_direct <= 1, f"Direct estimate {clone_direct} out of range"

    # Each mode should be within 0.05 of ground truth
    assert abs(clone_ips - ground_truth) < 0.05, (
        f"IPS ({clone_ips:.3f}) differs from truth ({ground_truth:.3f}) by "
        f"{abs(clone_ips - ground_truth):.3f}"
    )
    assert abs(clone_dr - ground_truth) < 0.05, (
        f"DR ({clone_dr:.3f}) differs from truth ({ground_truth:.3f}) by "
        f"{abs(clone_dr - ground_truth):.3f}"
    )
    assert abs(clone_direct - ground_truth) < 0.05, (
        f"Direct ({clone_direct:.3f}) differs from truth ({ground_truth:.3f}) by "
        f"{abs(clone_direct - ground_truth):.3f}"
    )

    # IPS should show good overlap for clone (since it's similar to base)
    if results_ips.diagnostics is not None and hasattr(
        results_ips.diagnostics, "ess_per_policy"
    ):
        clone_ess = results_ips.diagnostics.ess_per_policy.get("clone", 0)
        assert (
            clone_ess > 0.5
        ), f"Clone should have decent overlap, got ESS={clone_ess:.1%}"


def test_dr_and_direct_rank_unhelpful_as_worst() -> None:
    """DR and Direct modes should both rank unhelpful as the worst policy.

    The unhelpful policy is intentionally poor and should be ranked lowest among
    all three policies (clone, parallel_universe_prompt, unhelpful).

    This tests:
    - Ranking correctness across all policies
    - Both DR and Direct can distinguish quality differences
    - Sign correctness (not just magnitude accuracy)
    """
    dataset_path, fresh_draws_dir = _arena_paths()

    # Run DR mode (logged + fresh draws)
    results_dr = analyze_dataset(
        logged_data_path=str(dataset_path),
        fresh_draws_dir=str(fresh_draws_dir),
        estimator="stacked-dr",
        verbose=False,
    )

    # Run Direct mode (fresh draws only)
    results_direct = analyze_dataset(
        fresh_draws_dir=str(fresh_draws_dir),
        estimator="auto",
        verbose=False,
    )

    # Get all policy estimates for DR
    policies_dr = results_dr.metadata["target_policies"]
    unhelpful_idx_dr = policies_dr.index("unhelpful")
    unhelpful_dr = float(results_dr.estimates[unhelpful_idx_dr])

    # Get all policy estimates for Direct
    policies_direct = results_direct.metadata["target_policies"]
    unhelpful_idx_direct = policies_direct.index("unhelpful")
    unhelpful_direct = float(results_direct.estimates[unhelpful_idx_direct])

    # DR: unhelpful should be the minimum across all policies
    assert unhelpful_dr == min(results_dr.estimates), (
        f"DR mode: unhelpful ({unhelpful_dr:.3f}) should be lowest, "
        f"but estimates are {[f'{e:.3f}' for e in results_dr.estimates]}"
    )

    # Direct: unhelpful should be the minimum across all policies
    assert unhelpful_direct == min(results_direct.estimates), (
        f"Direct mode: unhelpful ({unhelpful_direct:.3f}) should be lowest, "
        f"but estimates are {[f'{e:.3f}' for e in results_direct.estimates]}"
    )
