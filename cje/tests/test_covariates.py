"""
Tests for calibration covariate functionality.

This test suite validates:
1. Auto-computable covariates (include_response_length flag)
2. Validation and error handling
3. Integration with Direct mode
"""

import json
from pathlib import Path

import pytest

from cje import analyze_dataset
from cje.calibration import calibrate_dataset
from cje.data.models import Dataset


def test_missing_covariate_error_message() -> None:
    """Test that missing covariate produces helpful error with available fields."""
    from cje.data.models import Sample

    samples = []
    for i in range(20):
        samples.append(
            Sample(
                prompt_id=f"prompt_{i}",
                prompt=f"Question {i}",
                response=f"Answer {i}",
                reward=None,
                judge_score=0.5,
                oracle_label=0.6 if i < 10 else None,
                metadata={
                    "domain": 1.0,
                    "difficulty": 0.5,
                },
            )
        )
    dataset = Dataset(samples=samples, target_policies=["target"])

    # Try to use a covariate that doesn't exist
    with pytest.raises(ValueError) as exc_info:
        calibrate_dataset(dataset, covariate_names=["nonexistent_field"])

    # Check that error message includes helpful info
    error_msg = str(exc_info.value)
    assert "nonexistent_field" in error_msg
    assert "Available metadata fields" in error_msg
    assert "Auto-computable covariates" in error_msg

    print("✅ Helpful error message test passed!")


def test_covariates_in_direct_mode(tmp_path: Path) -> None:
    """Test that covariates work in Direct mode."""

    # Create fresh draws directory
    fresh_draws_dir = tmp_path / "fresh_draws"
    fresh_draws_dir.mkdir()

    # Create fresh draw responses with varying lengths
    fresh_draws = []
    for i in range(30):
        response = "short" if i % 2 == 0 else "this is a much longer response here"

        fresh_draws.append(
            {
                "prompt_id": f"prompt_{i}",
                "response": response,
                "judge_score": 0.5 + i * 0.01,
                "oracle_label": 0.6 + i * 0.01 if i < 15 else None,  # 50% oracle
            }
        )

    policy_file = fresh_draws_dir / "policy_a_responses.jsonl"
    with open(policy_file, "w") as f:
        for item in fresh_draws:
            f.write(json.dumps(item) + "\n")

    # Run Direct mode with include_response_length
    results = analyze_dataset(
        fresh_draws_dir=str(fresh_draws_dir),
        include_response_length=True,
        estimator="direct",
        verbose=True,
    )

    assert results is not None
    assert len(results.estimates) == 1
    assert results.metadata["mode"] == "direct"

    print("✅ Covariates in Direct mode test passed!")


def test_covariate_auto_computation_stores_in_metadata() -> None:
    """Test that auto-computed covariates are stored in sample metadata."""

    from cje.data.models import Dataset, Sample

    # Create dataset with responses
    samples = []
    for i in range(10):
        response = "one two three" if i % 2 == 0 else "one two three four five"

        samples.append(
            Sample(
                prompt_id=f"prompt_{i}",
                prompt=f"Question {i}",
                response=response,
                reward=None,
                judge_score=0.5,
                oracle_label=0.6,
                metadata={},
            )
        )

    dataset = Dataset(samples=samples, target_policies=["target"])

    # Calibrate with response_length covariate
    calibrated_dataset, result = calibrate_dataset(
        dataset,
        covariate_names=["response_length"],
    )

    # Check that response_length was computed and stored
    for i, sample in enumerate(calibrated_dataset.samples):
        assert "response_length" in sample.metadata

        # Verify computation is correct
        expected_length = len(sample.response.split())
        assert sample.metadata["response_length"] == float(expected_length)

        if i % 2 == 0:
            assert sample.metadata["response_length"] == 3.0
        else:
            assert sample.metadata["response_length"] == 5.0

    print("✅ Auto-computation stores in metadata test passed!")


def test_covariate_validation_calibration_source(tmp_path: Path) -> None:
    """Test that include_response_length validates the calibration data source."""
    from typing import Any, Dict, List

    # Create fresh draws WITH responses
    fresh_draws_dir = tmp_path / "fresh_draws"
    fresh_draws_dir.mkdir()

    fresh_draws: List[Dict[str, Any]] = [
        {
            "prompt_id": f"prompt_{i}",
            "response": f"Answer {i}",
            "judge_score": 0.5 + i * 0.01,
        }
        for i in range(20)
    ]

    policy_file = fresh_draws_dir / "target_responses.jsonl"
    with open(policy_file, "w") as f:
        for item in fresh_draws:
            f.write(json.dumps(item) + "\n")

    # Create calibration data with response=None (data loader will reject this)
    calibration_data: List[Dict[str, Any]] = [
        {
            "prompt_id": f"calib_{i}",
            "prompt": f"Calib question {i}",
            "response": None,  # None should be rejected
            "base_policy_logprob": -10.0,
            "target_policy_logprobs": {"target": -9.5},
            "judge_score": 0.4,
            "oracle_label": 0.5,
        }
        for i in range(10)
    ]

    calib_path = tmp_path / "calibration.jsonl"
    with open(calib_path, "w") as f:
        for item in calibration_data:
            f.write(json.dumps(item) + "\n")

    # Should fail when trying to use calibration data without response field
    # The data loader will catch this before our validation
    with pytest.raises(ValueError):
        analyze_dataset(
            fresh_draws_dir=str(fresh_draws_dir),
            calibration_data_path=str(calib_path),
            include_response_length=True,
            estimator="direct",
        )

    print("✅ Validation of calibration data source test passed!")


def test_covariate_computation_consistency() -> None:
    """REGRESSION TEST: Ensure calibration and fresh draws compute covariates identically.

    This test catches the bug where calibration used word count but fresh draws
    used log10(character count), causing systematic bias in Direct estimates.

    Bug context: Prior to Oct 2024, this mismatch caused ~0.06 systematic bias
    in all estimators when use_covariates=True.
    """
    from cje.calibration.dataset import AUTO_COMPUTABLE_COVARIATES
    from cje.data.fresh_draws import (
        compute_response_covariates,
        FreshDrawDataset,
        FreshDrawSample,
    )
    from cje.data.models import Sample

    # Test responses with varying lengths
    test_responses = [
        "This is a short response.",  # 5 words, 25 chars
        "This is a much longer response with many more words to demonstrate the issue.",  # 14 words, 77 chars
        "x" * 100,  # 1 word, 100 chars
        "word " * 50,  # 50 words, 250 chars
    ]

    # Get calibration covariate function
    calib_compute_fn, _ = AUTO_COMPUTABLE_COVARIATES["response_length"]

    for response in test_responses:
        # Compute via calibration path
        sample = Sample(
            prompt_id="test",
            prompt="test prompt",
            response=response,
            reward=None,
            judge_score=0.5,
            oracle_label=None,
            metadata={},
        )
        calib_value = calib_compute_fn(sample)

        # Compute via fresh draws path
        fresh_sample = FreshDrawSample(
            prompt_id="test",
            target_policy="test",
            response=response,
            judge_score=0.5,
            oracle_label=None,
            draw_idx=0,
            metadata={},
        )
        fresh_dataset = FreshDrawDataset(target_policy="test", samples=[fresh_sample])
        fresh_with_covs = compute_response_covariates(
            fresh_dataset, covariate_names=["response_length"]
        )
        fresh_value = fresh_with_covs.samples[0].metadata["response_length"]

        # CRITICAL: These must match exactly
        assert calib_value == fresh_value, (
            f"Covariate mismatch for response '{response[:50]}...':\n"
            f"  Calibration computed: {calib_value}\n"
            f"  Fresh draws computed: {fresh_value}\n"
            f"  This causes systematic bias in estimates!"
        )

        # Verify they both use word count
        expected_word_count = len(response.split())
        assert calib_value == float(
            expected_word_count
        ), f"Calibration not using word count! Expected {expected_word_count}, got {calib_value}"
        assert fresh_value == float(
            expected_word_count
        ), f"Fresh draws not using word count! Expected {expected_word_count}, got {fresh_value}"

    print("✅ Covariate computation consistency test passed!")
    print("   Both calibration and fresh draws use word count (len(response.split()))")


@pytest.mark.e2e
@pytest.mark.uses_arena_sample
@pytest.mark.slow
def test_covariates_with_real_arena_data(arena_sample: Dataset) -> None:
    """E2E smoke test: Covariates work end-to-end with real arena sample data."""
    from pathlib import Path

    # Use the real arena sample data as the calibration source
    data_path = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "arena_sample"
        / "logged_data.jsonl"
    )
    fresh_draws_dir = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "arena_sample"
        / "fresh_draws"
    )
    if not fresh_draws_dir.exists():
        pytest.skip(f"Fresh draws not found at {fresh_draws_dir}")

    # Run Direct mode with response_length covariate on real data
    # (combine_oracle_sources=False: combined-source pairs carry empty
    # responses, which would degenerate the response_length covariate)
    results = analyze_dataset(
        fresh_draws_dir=str(fresh_draws_dir),
        calibration_data_path=str(data_path),
        combine_oracle_sources=False,
        include_response_length=True,
        estimator="direct",
        verbose=False,
    )

    # Basic validation
    assert results is not None
    assert len(results.estimates) == 4  # base, clone, parallel, unhelpful
    assert all(0 <= e <= 1 for e in results.estimates)

    print("✅ Arena data E2E smoke test passed!")


if __name__ == "__main__":
    # Run all tests directly
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        print("\n=== Running Covariate Tests ===\n")

        test_missing_covariate_error_message()
        test_covariates_in_direct_mode(tmp_path)
        test_covariate_auto_computation_stores_in_metadata()
        test_covariate_validation_calibration_source(tmp_path)
        test_covariate_computation_consistency()  # REGRESSION TEST

        print("\n=== All Covariate Tests Passed! ===\n")
