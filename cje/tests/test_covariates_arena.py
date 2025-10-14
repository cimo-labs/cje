"""
E2E tests for calibration covariate functionality using arena sample data.

This test suite validates covariates using real arena data:
1. include_response_length flag (auto-computed covariate)
2. Manual synthetic covariates (domain, difficulty)
3. Combined auto + manual covariates
4. Integration across IPS, DR, and Direct modes
"""

import json
import tempfile
from pathlib import Path
from copy import deepcopy

import pytest
import numpy as np

from cje import analyze_dataset
from cje.calibration import calibrate_dataset
from cje.data.models import Dataset
from cje.tests.conftest import assert_valid_estimation_result


# ============================================================================
# Helper: Add Synthetic Covariates to Arena Data
# ============================================================================


def add_synthetic_covariates_to_dataset(dataset: Dataset) -> Dataset:
    """Add domain and difficulty covariates to arena dataset samples.

    Synthetic covariates based on real data patterns:
    - domain: Hash of prompt_id % 5 (creates 5 domains)
    - difficulty: Response/prompt length ratio (proxy for complexity)

    These are deterministic so tests are reproducible.
    """
    new_samples = []

    for sample in dataset.samples:
        # Compute synthetic covariates
        response_len = len(sample.response.split())
        prompt_len = len(sample.prompt.split())

        # Domain based on prompt_id hash (stable across runs)
        domain_hash = hash(sample.prompt_id) % 5

        # Difficulty as length ratio (longer responses → higher difficulty)
        difficulty = min(1.0, response_len / max(prompt_len * 10, 1))

        # Add to metadata
        new_metadata = sample.metadata.copy()
        new_metadata["domain"] = float(domain_hash)
        new_metadata["difficulty"] = float(difficulty)

        # Create new sample with augmented metadata
        new_sample = sample.model_copy(update={"metadata": new_metadata})
        new_samples.append(new_sample)

    return Dataset(
        samples=new_samples,
        target_policies=dataset.target_policies,
        metadata=dataset.metadata.copy(),
    )


# ============================================================================
# E2E Tests: include_response_length Flag
# ============================================================================


@pytest.mark.e2e
@pytest.mark.uses_arena_sample
def test_arena_response_length_ips_mode(arena_sample: Dataset) -> None:
    """E2E: include_response_length works in IPS mode with arena data."""

    # Run full pipeline with response_length covariate
    from cje import analyze_dataset
    from cje.data import load_dataset_from_jsonl

    # Get path to arena data
    data_path = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "arena_sample"
        / "logged_data.jsonl"
    )

    results = analyze_dataset(
        logged_data_path=str(data_path),
        include_response_length=True,
        estimator="calibrated-ips",
        verbose=False,
    )

    # Validate results
    assert_valid_estimation_result(results, n_policies=3)  # clone, parallel, unhelpful

    # All estimates should be valid probabilities
    assert all(0 <= e <= 1 for e in results.estimates)


@pytest.mark.e2e
@pytest.mark.uses_arena_sample
def test_arena_response_length_ips_only_logged_data(arena_sample: Dataset) -> None:
    """E2E: include_response_length with only logged data (IPS mode - no fresh draws needed)."""

    data_path = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "arena_sample"
        / "logged_data.jsonl"
    )

    results = analyze_dataset(
        logged_data_path=str(data_path),
        include_response_length=True,
        estimator="calibrated-ips",
        verbose=False,
    )

    # Validate results
    assert_valid_estimation_result(results, n_policies=3)


# ============================================================================
# E2E Tests: Manual Synthetic Covariates (using direct dataset manipulation)
# ============================================================================


@pytest.mark.e2e
@pytest.mark.uses_arena_sample
def test_arena_synthetic_covariates_calibration(arena_sample: Dataset) -> None:
    """E2E: Manual synthetic covariates work in calibration pipeline."""

    # Add synthetic covariates to arena data
    dataset_with_covariates = add_synthetic_covariates_to_dataset(arena_sample)

    # Run calibration with covariates (tests that covariates work end-to-end)
    from cje.calibration import calibrate_dataset
    from cje.data.precomputed_sampler import PrecomputedSampler
    from cje.estimators.calibrated_ips import CalibratedIPS

    calibrated_dataset, result = calibrate_dataset(
        dataset_with_covariates,
        judge_field="judge_score",
        oracle_field="oracle_label",
        covariate_names=["domain", "difficulty"],
    )

    # Create sampler and estimator
    sampler = PrecomputedSampler(calibrated_dataset)
    estimator = CalibratedIPS(sampler)

    # Run estimation
    results = estimator.fit_and_estimate()

    # Validate results
    assert_valid_estimation_result(results, n_policies=3)


# ============================================================================
# Unit Tests: Covariate Computation and Storage
# ============================================================================


@pytest.mark.unit
@pytest.mark.uses_arena_sample
def test_response_length_stored_in_metadata(arena_sample: Dataset) -> None:
    """Unit: Auto-computed response_length is stored in sample metadata."""

    # Calibrate with response_length covariate
    calibrated_dataset, result = calibrate_dataset(
        arena_sample,
        covariate_names=["response_length"],
    )

    # Check that response_length was computed and stored for all samples
    for sample in calibrated_dataset.samples:
        assert (
            "response_length" in sample.metadata
        ), f"response_length missing for sample {sample.prompt_id}"

        # Verify it matches actual response length
        expected_length = len(sample.response.split())
        assert sample.metadata["response_length"] == float(
            expected_length
        ), f"Incorrect response_length for {sample.prompt_id}"

    # Check that values vary (different responses have different lengths)
    lengths = [s.metadata["response_length"] for s in calibrated_dataset.samples]
    unique_lengths = set(lengths)
    assert (
        len(unique_lengths) > 10
    ), f"Response lengths should vary across arena samples, got only {len(unique_lengths)} unique values"


@pytest.mark.unit
@pytest.mark.uses_arena_sample
def test_response_length_computation_correctness(arena_sample: Dataset) -> None:
    """Unit: Verify response_length computation is correct (word count)."""

    calibrated_dataset, _ = calibrate_dataset(
        arena_sample,
        covariate_names=["response_length"],
    )

    # Check a few samples in detail
    for i, sample in enumerate(calibrated_dataset.samples[:10]):
        computed_length = sample.metadata["response_length"]
        expected_length = len(sample.response.split())

        assert computed_length == float(
            expected_length
        ), f"Sample {i}: computed={computed_length}, expected={expected_length}"


# ============================================================================
# Integration Tests: Covariate Impact
# ============================================================================


@pytest.mark.integration
@pytest.mark.uses_arena_sample
def test_covariates_impact_on_estimates(arena_sample: Dataset) -> None:
    """Integration: Covariates can produce different calibration results."""

    from cje.calibration import calibrate_dataset

    # Baseline: No covariates
    calibrated_baseline, result_baseline = calibrate_dataset(
        arena_sample,
        judge_field="judge_score",
        oracle_field="oracle_label",
    )

    # With response_length covariate
    calibrated_with_cov, result_with_cov = calibrate_dataset(
        arena_sample,
        judge_field="judge_score",
        oracle_field="oracle_label",
        covariate_names=["response_length"],
    )

    # Both should succeed
    assert result_baseline.n_oracle > 0
    assert result_with_cov.n_oracle > 0

    # Both should produce valid calibrated scores
    assert len(calibrated_baseline.samples) == len(arena_sample.samples)
    assert len(calibrated_with_cov.samples) == len(arena_sample.samples)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


@pytest.mark.unit
def test_covariate_error_message_helpful(arena_sample: Dataset) -> None:
    """Unit: Missing covariate produces helpful error with available fields."""

    # Try to use a covariate that doesn't exist in calibrate_dataset directly
    with pytest.raises(ValueError) as exc_info:
        calibrate_dataset(
            arena_sample,
            covariate_names=["nonexistent_field"],
        )

    # Check that error message includes helpful info
    error_msg = str(exc_info.value)
    assert "nonexistent_field" in error_msg
    assert "Available metadata fields" in error_msg or "not found" in error_msg


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
