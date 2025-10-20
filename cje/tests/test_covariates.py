"""
Tests for calibration covariate functionality.

This test suite validates:
1. Manual covariate specification (calibration_covariates parameter)
2. Auto-computable covariates (include_response_length flag)
3. Validation and error handling
4. Integration across IPS, DR, and Direct modes
"""

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np

from cje import analyze_dataset
from cje.calibration import calibrate_dataset
from cje.data import load_dataset_from_jsonl
from cje.data.models import Dataset


def test_include_response_length_flag(tmp_path: Path) -> None:
    """Test that include_response_length=True auto-computes response_length covariate."""

    # Create data with varying response lengths
    data = []
    for i in range(50):
        # Vary response length: short (3 words) and long (10 words)
        response = (
            f"short answer here"
            if i % 2 == 0
            else f"this is a much longer response with many more words here"
        )

        data.append(
            {
                "prompt_id": f"prompt_{i}",
                "prompt": f"Question {i}",
                "response": response,
                "base_policy_logprob": -10.0,
                "target_policy_logprobs": {"target": -9.5},
                "judge_score": 0.5 + i * 0.01,
                "oracle_label": 0.6 + i * 0.01 if i < 20 else None,  # 40% oracle
            }
        )

    data_path = tmp_path / "data.jsonl"
    with open(data_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    # Run with include_response_length=True
    results = analyze_dataset(
        logged_data_path=str(data_path),
        include_response_length=True,
        estimator="calibrated-ips",
        verbose=True,
    )

    # Should complete successfully
    assert results is not None
    assert len(results.estimates) == 1
    assert results.estimates[0] >= 0 and results.estimates[0] <= 1

    print("✅ include_response_length test passed!")


def test_include_response_length_missing_field_error(tmp_path: Path) -> None:
    """Test that include_response_length=True fails gracefully when response field is None."""

    # Create data with response=None (empty string would pass validation)
    data = []
    for i in range(20):
        data.append(
            {
                "prompt_id": f"prompt_{i}",
                "prompt": f"Question {i}",
                "response": None,  # None should trigger validation error
                "base_policy_logprob": -10.0,
                "target_policy_logprobs": {"target": -9.5},
                "judge_score": 0.5,
                "oracle_label": 0.6 if i < 10 else None,
            }
        )

    data_path = tmp_path / "data.jsonl"
    with open(data_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    # Should raise ValueError with clear message
    # Note: The data loader will reject None responses, which is correct behavior
    with pytest.raises(ValueError):
        analyze_dataset(
            logged_data_path=str(data_path),
            include_response_length=True,
            estimator="calibrated-ips",
        )

    print("✅ Missing response field error test passed!")


def test_manual_covariate_specification(tmp_path: Path) -> None:
    """Test manual covariate specification via calibration_covariates parameter."""

    # Create data with custom covariates
    data = []
    for i in range(50):
        data.append(
            {
                "prompt_id": f"prompt_{i}",
                "prompt": f"Question {i}",
                "response": f"Answer {i}",
                "base_policy_logprob": -10.0,
                "target_policy_logprobs": {"target": -9.5},
                "judge_score": 0.5 + i * 0.01,
                "oracle_label": 0.6 + i * 0.01 if i < 20 else None,
                "metadata": {
                    "domain": float(i % 3),  # 3 domains
                    "difficulty": float(i % 5) / 5.0,  # 5 difficulty levels
                },
            }
        )

    data_path = tmp_path / "data.jsonl"
    with open(data_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    # Run with manual covariates
    results = analyze_dataset(
        logged_data_path=str(data_path),
        calibration_covariates=["domain", "difficulty"],
        estimator="calibrated-ips",
        verbose=True,
    )

    assert results is not None
    assert len(results.estimates) == 1

    print("✅ Manual covariate specification test passed!")


def test_combined_manual_and_auto_covariates(tmp_path: Path) -> None:
    """Test combining include_response_length with manual covariates."""

    # Create data with both response lengths and custom covariates
    data = []
    for i in range(50):
        response = f"short" if i % 2 == 0 else f"this is a longer response"

        data.append(
            {
                "prompt_id": f"prompt_{i}",
                "prompt": f"Question {i}",
                "response": response,
                "base_policy_logprob": -10.0,
                "target_policy_logprobs": {"target": -9.5},
                "judge_score": 0.5 + i * 0.01,
                "oracle_label": 0.6 + i * 0.01 if i < 20 else None,
                "metadata": {
                    "domain": float(i % 3),
                },
            }
        )

    data_path = tmp_path / "data.jsonl"
    with open(data_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    # Combine auto and manual covariates
    results = analyze_dataset(
        logged_data_path=str(data_path),
        include_response_length=True,
        calibration_covariates=["domain"],  # response_length should be prepended
        estimator="calibrated-ips",
        verbose=True,
    )

    assert results is not None
    assert len(results.estimates) == 1

    print("✅ Combined auto + manual covariates test passed!")


def test_missing_covariate_error_message(tmp_path: Path) -> None:
    """Test that missing covariate produces helpful error with available fields."""

    data = []
    for i in range(20):
        data.append(
            {
                "prompt_id": f"prompt_{i}",
                "prompt": f"Question {i}",
                "response": f"Answer {i}",
                "base_policy_logprob": -10.0,
                "target_policy_logprobs": {"target": -9.5},
                "judge_score": 0.5,
                "oracle_label": 0.6 if i < 10 else None,
                "metadata": {
                    "domain": 1.0,
                    "difficulty": 0.5,
                },
            }
        )

    data_path = tmp_path / "data.jsonl"
    with open(data_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    # Try to use a covariate that doesn't exist
    with pytest.raises(ValueError) as exc_info:
        analyze_dataset(
            logged_data_path=str(data_path),
            calibration_covariates=["nonexistent_field"],
            estimator="calibrated-ips",
        )

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
        response = f"short" if i % 2 == 0 else f"this is a much longer response here"

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


def test_covariates_in_dr_mode(tmp_path: Path) -> None:
    """Test that covariates work in DR mode (both calibration and outcome models)."""

    # Create logged data
    logged_data = []
    for i in range(30):
        response = (
            f"short answer"
            if i % 2 == 0
            else f"this is a longer answer with more words"
        )

        logged_data.append(
            {
                "prompt_id": f"prompt_{i}",
                "prompt": f"Question {i}",
                "response": response,
                "base_policy_logprob": -10.0,
                "target_policy_logprobs": {"target": -9.5},
                "judge_score": 0.5 + i * 0.01,
                "oracle_label": 0.6 + i * 0.01 if i < 15 else None,  # 50% oracle
            }
        )

    logged_path = tmp_path / "logged.jsonl"
    with open(logged_path, "w") as f:
        for item in logged_data:
            f.write(json.dumps(item) + "\n")

    # Create fresh draws - MUST match ALL logged prompt_ids for DR
    fresh_draws_dir = tmp_path / "fresh_draws"
    fresh_draws_dir.mkdir()

    fresh_draws = []
    for i in range(30):  # Match all 30 logged prompts
        response = f"fresh short" if i % 2 == 0 else f"fresh longer response here"

        fresh_draws.append(
            {
                "prompt_id": f"prompt_{i}",  # Exact match with logged data
                "response": response,
                "judge_score": 0.55 + i * 0.01,
            }
        )

    policy_file = fresh_draws_dir / "target_responses.jsonl"
    with open(policy_file, "w") as f:
        for item in fresh_draws:
            f.write(json.dumps(item) + "\n")

    # Run DR mode with include_response_length
    results = analyze_dataset(
        logged_data_path=str(logged_path),
        fresh_draws_dir=str(fresh_draws_dir),
        include_response_length=True,
        estimator="stacked-dr",
        verbose=True,
    )

    assert results is not None
    assert len(results.estimates) == 1
    # DR mode should use metadata from mode detection or explicit
    assert results.metadata.get("mode") in ["dr", None]

    # REGRESSION TEST: Ensure all three DR estimators work with covariates
    # This catches bugs where TMLE/MRDR fail to pass covariates to outcome_model.predict()
    # which causes SplineTransformer feature mismatch errors when use_covariates=True
    assert (
        "valid_estimators" in results.metadata
    ), "Missing 'valid_estimators' in metadata - stacking may have failed"
    valid_estimators = results.metadata["valid_estimators"]
    assert set(valid_estimators) == {"dr-cpo", "mrdr", "tmle"}, (
        f"Expected all three DR estimators to work with covariates, "
        f"but got: {valid_estimators}. "
        f"Failed estimators: {results.metadata.get('failed_estimators', [])}"
    )

    # Verify no estimators failed
    failed_estimators = results.metadata.get("failed_estimators", [])
    assert (
        len(failed_estimators) == 0
    ), f"Some estimators failed with covariates: {failed_estimators}"

    # Verify stacking actually used all three estimators
    assert (
        "stacking_weights" in results.metadata
    ), "Missing 'stacking_weights' - stacking may have fallen back to single estimator"

    print("✅ Covariates in DR mode test passed!")


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
                base_policy_logprob=-10.0,
                target_policy_logprobs={"target": -9.5},
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


def test_covariate_validation_all_data_sources(tmp_path: Path) -> None:
    """Test that include_response_length validates ALL data sources (logged, fresh, calibration)."""

    # Create logged data WITH response
    logged_data = [
        {
            "prompt_id": f"prompt_{i}",
            "prompt": f"Question {i}",
            "response": f"Answer {i}",
            "base_policy_logprob": -10.0,
            "target_policy_logprobs": {"target": -9.5},
            "judge_score": 0.5,
            "oracle_label": 0.6 if i < 10 else None,
        }
        for i in range(20)
    ]

    logged_path = tmp_path / "logged.jsonl"
    with open(logged_path, "w") as f:
        for item in logged_data:
            f.write(json.dumps(item) + "\n")

    # Create calibration data with response=None (data loader will reject this)
    calibration_data = [
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
            logged_data_path=str(logged_path),
            calibration_data_path=str(calib_path),
            include_response_length=True,
            estimator="calibrated-ips",
        )

    print("✅ Validation across data sources test passed!")


def test_covariate_computation_consistency() -> None:
    """REGRESSION TEST: Ensure calibration and fresh draws compute covariates identically.

    This test catches the bug where calibration used word count but fresh draws
    used log10(character count), causing systematic bias in DR/Direct estimates.

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
            base_policy_logprob=-1.0,
            target_policy_logprobs={},
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
            fold_id=None,
            metadata={},
        )
        fresh_dataset = FreshDrawDataset(
            target_policy="test", draws_per_prompt=1, samples=[fresh_sample]
        )
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
def test_covariates_with_real_arena_data(arena_sample: Dataset) -> None:
    """E2E smoke test: Covariates work end-to-end with real arena sample data."""
    from pathlib import Path

    # Use the real arena sample data
    data_path = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "arena_sample"
        / "logged_data.jsonl"
    )

    # Run with response_length covariate on real data
    results = analyze_dataset(
        logged_data_path=str(data_path),
        include_response_length=True,
        estimator="calibrated-ips",
        verbose=False,
    )

    # Basic validation
    assert results is not None
    assert len(results.estimates) == 3  # clone, parallel, unhelpful
    assert all(0 <= e <= 1 for e in results.estimates)

    print("✅ Arena data E2E smoke test passed!")


if __name__ == "__main__":
    # Run all tests directly
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        print("\n=== Running Covariate Tests ===\n")

        test_include_response_length_flag(tmp_path)
        test_include_response_length_missing_field_error(tmp_path)
        test_manual_covariate_specification(tmp_path)
        test_combined_manual_and_auto_covariates(tmp_path)
        test_missing_covariate_error_message(tmp_path)
        test_covariates_in_direct_mode(tmp_path)
        test_covariates_in_dr_mode(tmp_path)
        test_covariate_auto_computation_stores_in_metadata()
        test_covariate_validation_all_data_sources(tmp_path)
        test_covariate_computation_consistency()  # REGRESSION TEST

        print("\n=== All Covariate Tests Passed! ===\n")
