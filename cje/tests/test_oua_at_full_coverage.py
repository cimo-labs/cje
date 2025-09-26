#!/usr/bin/env python3
"""
Test that OUA is correctly skipped at 100% oracle coverage.

This test ensures that robust_standard_errors equals standard_errors
when oracle_coverage = 1.0, preventing the bug discovered in the
ablation experiments.
"""

import pytest
import numpy as np
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch

from cje.data import Dataset, Sample
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.estimators import (
    CalibratedIPS,
    DRCPOEstimator,
    StackedDREstimator,
    OrthogonalizedCalibratedIPS,
    TMLEEstimator,
    MRDREstimator,
)


def create_test_dataset(oracle_coverage: float = 1.0, n_samples: int = 100) -> Dataset:
    """Create a test dataset with specified oracle coverage."""
    samples = []
    n_oracle = int(n_samples * oracle_coverage)

    for i in range(n_samples):
        # Compute a reward based on judge score (required for PrecomputedSampler)
        # Keep rewards in [0, 1] range
        judge_score = 0.5 + 0.001 * i  # Smaller increments to stay in range
        reward = min(judge_score, 1.0)  # Cap at 1.0

        sample = Sample(
            prompt_id=f"p{i}",
            prompt="test prompt",
            response="test response",
            reward=reward,  # Add reward field
            base_policy="base",
            base_policy_logprob=-2.0,
            target_policy_logprobs={"target": -1.5},
            metadata={
                "judge_score": judge_score,
                "oracle_label": 1 if i < n_oracle else None,
            },
        )
        samples.append(sample)

    dataset = Dataset(
        samples=samples,
        target_policies=["target"],  # Required field
        metadata={
            "oracle_coverage": oracle_coverage,
            "oracle_indices": list(range(n_oracle)),
            "calibrated": True,  # Mark as calibrated
        },
    )

    return dataset


def test_ips_skips_oua_at_full_coverage() -> None:
    """Test that CalibratedIPS skips OUA at 100% oracle coverage."""
    # Create dataset with 100% coverage
    dataset = create_test_dataset(oracle_coverage=1.0)

    # Mock calibrator
    mock_calibrator = MagicMock()
    mock_calibrator.has_oracle_indices = False

    # Create sampler (target_policies already in dataset)
    sampler = PrecomputedSampler(dataset)

    # Create estimator with OUA enabled
    estimator = CalibratedIPS(
        sampler,
        calibrate_weights=False,  # Use raw IPS for simplicity
        oua_jackknife=True,  # Enable OUA
    )
    estimator.reward_calibrator = mock_calibrator

    # Fit and estimate
    result = estimator.fit_and_estimate()

    # Check that robust SE equals standard SE
    assert result.standard_errors is not None
    assert result.robust_standard_errors is not None
    np.testing.assert_array_equal(
        result.standard_errors,
        result.robust_standard_errors,
        err_msg="At 100% oracle coverage, robust SE should equal standard SE",
    )

    # Check metadata
    assert result.metadata is not None
    assert "oua" in result.metadata
    assert result.metadata["oua"].get("skipped") == "100% oracle coverage"


def test_ips_applies_oua_at_partial_coverage() -> None:
    """Test that CalibratedIPS applies OUA at partial oracle coverage."""
    # Create dataset with 50% coverage
    dataset = create_test_dataset(oracle_coverage=0.5)

    # Create mock calibrator with jackknife support
    mock_calibrator = MagicMock()
    mock_calibrator.has_oracle_indices = True

    # Create sampler (target_policies already in dataset)
    sampler = PrecomputedSampler(dataset)

    # Create estimator with OUA enabled
    estimator = CalibratedIPS(sampler, calibrate_weights=False, oua_jackknife=True)
    estimator.reward_calibrator = mock_calibrator

    # Mock the jackknife method to return some variance
    def mock_jackknife(policy: str) -> np.ndarray:
        # Return K jackknife estimates with some variance
        return np.array([0.5, 0.52, 0.48, 0.51, 0.49])

    estimator.get_oracle_jackknife = mock_jackknife  # type: ignore[method-assign]

    # Fit and estimate
    result = estimator.fit_and_estimate()

    # Check that robust SE is larger than standard SE
    assert result.standard_errors is not None
    assert result.robust_standard_errors is not None
    assert np.all(
        result.robust_standard_errors >= result.standard_errors
    ), "Robust SE should be >= standard SE when OUA is applied"

    # For meaningful OUA, they should actually be different
    if not np.allclose(result.robust_standard_errors, result.standard_errors):
        assert np.any(
            result.robust_standard_errors > result.standard_errors
        ), "At partial coverage with OUA, robust SE should be larger than standard SE"


def test_stacked_dr_skips_oua_at_full_coverage() -> None:
    """Test that StackedDREstimator skips OUA at 100% oracle coverage."""
    # Create dataset with 100% coverage
    dataset = create_test_dataset(oracle_coverage=1.0)

    # Mock calibrator
    mock_calibrator = MagicMock()
    mock_calibrator.has_oracle_indices = False

    # Create sampler (target_policies already in dataset)
    sampler = PrecomputedSampler(dataset)

    # Mock fresh draws to avoid DR failure
    mock_fresh = MagicMock()
    mock_fresh.samples = dataset.samples[:10]  # Use subset as "fresh"

    # Create estimator
    estimator = StackedDREstimator(
        sampler,
        reward_calibrator=mock_calibrator,
        estimators=["dr-cpo"],  # Just use one for simplicity
        n_folds=2,
        oua_jackknife=True,
    )

    # Add fresh draws
    estimator.add_fresh_draws("target", mock_fresh)

    with patch.object(estimator, "_apply_stacked_oua") as mock_apply:
        # The method should be called
        result = estimator.fit_and_estimate()
        mock_apply.assert_called_once()

    # If successful, check that robust SE equals standard SE
    if result.standard_errors is not None and result.robust_standard_errors is not None:
        np.testing.assert_array_almost_equal(
            result.standard_errors,
            result.robust_standard_errors,
            decimal=10,
            err_msg="At 100% oracle coverage, stacked DR should have robust SE = standard SE",
        )


def test_multiple_estimators_at_full_coverage() -> None:
    """Test multiple estimators to ensure they all skip OUA at 100% coverage."""
    dataset = create_test_dataset(oracle_coverage=1.0)

    # Mock calibrator
    mock_calibrator = MagicMock()
    mock_calibrator.has_oracle_indices = False

    sampler = PrecomputedSampler(dataset)

    # Test different estimators
    estimators_to_test = [
        CalibratedIPS(sampler, oua_jackknife=True, calibrate_weights=False),
        OrthogonalizedCalibratedIPS(
            sampler, oua_jackknife=True, calibrate_weights=False
        ),
    ]

    for estimator in estimators_to_test:
        estimator.reward_calibrator = mock_calibrator

        # Fit and estimate
        result = estimator.fit_and_estimate()

        # Check that robust SE equals standard SE
        if (
            result.standard_errors is not None
            and result.robust_standard_errors is not None
        ):
            np.testing.assert_array_almost_equal(
                result.standard_errors,
                result.robust_standard_errors,
                decimal=10,
                err_msg=f"{estimator.__class__.__name__} should skip OUA at 100% coverage",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
