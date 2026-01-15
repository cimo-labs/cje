#!/usr/bin/env python3
"""
Test that OUA is correctly skipped at 100% oracle coverage.

This test ensures that oracle uncertainty is not added to standard_errors
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
            base_policy_logprob=-2.0,
            target_policy_logprobs={"target": -1.5},
            judge_score=judge_score,  # Top-level field
            oracle_label=1 if i < n_oracle else None,  # Top-level field
            metadata={},
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

    # Check that standard errors were computed
    assert result.standard_errors is not None

    # Check metadata indicates OUA was skipped
    assert result.metadata is not None
    assert "se_components" in result.metadata
    assert (
        result.metadata["se_components"].get("oracle_uncertainty_skipped")
        == "100% oracle coverage"
    )


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

    # Check that standard errors include oracle uncertainty
    assert result.standard_errors is not None
    assert result.metadata is not None
    assert "se_components" in result.metadata
    assert result.metadata["se_components"].get("includes_oracle_uncertainty") is True

    # Check that oracle variance was actually added
    oracle_variance = result.metadata["se_components"].get(
        "oracle_variance_per_policy", {}
    )
    assert (
        len(oracle_variance) > 0
    ), "Oracle variance should be computed at partial coverage"


def test_stacked_dr_skips_oua_at_full_coverage() -> None:
    """Test that estimators skip OUA at 100% oracle coverage."""
    # At 100% oracle coverage, there's no uncertainty in the calibrator,
    # so OUA should be skipped.

    # Create dataset with 100% coverage
    dataset = create_test_dataset(oracle_coverage=1.0)

    # Mock calibrator
    mock_calibrator = MagicMock()
    mock_calibrator.has_oracle_indices = False  # This indicates 100% coverage

    # Create sampler (target_policies already in dataset)
    sampler = PrecomputedSampler(dataset)

    # Test with CalibratedIPS which handles 100% coverage properly
    estimator = CalibratedIPS(
        sampler,
        calibrate_weights=False,
        oua_jackknife=True,
    )
    estimator.reward_calibrator = mock_calibrator

    # Fit and estimate
    result = estimator.fit_and_estimate()

    # Check that OUA was skipped
    assert result.standard_errors is not None
    assert result.metadata is not None
    assert "se_components" in result.metadata
    assert (
        result.metadata["se_components"].get("oracle_uncertainty_skipped")
        == "100% oracle coverage"
    )


def test_multiple_estimators_at_full_coverage() -> None:
    """Test multiple estimators to ensure they all skip OUA at 100% coverage."""
    dataset = create_test_dataset(oracle_coverage=1.0)

    # Mock calibrator
    mock_calibrator = MagicMock()
    mock_calibrator.has_oracle_indices = False

    sampler = PrecomputedSampler(dataset)

    # Test CalibratedIPS (the primary IPS estimator)
    estimators_to_test = [
        CalibratedIPS(sampler, oua_jackknife=True, calibrate_weights=False),
    ]

    for estimator in estimators_to_test:
        estimator.reward_calibrator = mock_calibrator

        # Fit and estimate
        result = estimator.fit_and_estimate()

        # Check that OUA was skipped
        assert result.standard_errors is not None
        assert result.metadata is not None
        assert "se_components" in result.metadata
        assert (
            result.metadata["se_components"].get("oracle_uncertainty_skipped")
            == "100% oracle coverage"
        ), f"{estimator.__class__.__name__} should skip OUA at 100% coverage"


def test_direct_method_skips_oua_at_full_coverage() -> None:
    """Test that CalibratedDirectEstimator skips OUA at 100% oracle coverage.

    The direct method doesn't use a sampler, so it must check oracle_coverage
    on the calibrator instead.
    """
    from cje.estimators.direct_method import CalibratedDirectEstimator
    from cje.calibration.judge import JudgeCalibrator
    from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample

    # Create test data with 100% oracle coverage
    n_samples = 100
    np.random.seed(42)
    judge_scores = np.random.uniform(0, 1, n_samples)
    oracle_labels = np.random.uniform(0, 1, n_samples)  # All samples have oracle labels

    # Create and fit calibrator (100% coverage = all samples have oracle)
    calibrator = JudgeCalibrator(random_seed=42, calibration_mode="monotone")
    cal_result = calibrator.fit_transform(judge_scores, oracle_labels)

    # Verify calibrator has oracle_coverage set
    assert calibrator.oracle_coverage is not None
    assert calibrator.oracle_coverage >= 1.0, "Oracle coverage should be 100%"

    # Create fresh draws for direct method using FreshDrawDataset
    fresh_draw_samples = [
        FreshDrawSample(
            prompt_id=f"p{i}",
            judge_score=float(judge_scores[i % n_samples]),
            oracle_label=float(oracle_labels[i % n_samples]),
            target_policy="policy_a",
            draw_idx=0,
            response=None,
            fold_id=None,
            metadata={},
        )
        for i in range(50)
    ]
    fresh_draws_dataset = FreshDrawDataset(
        samples=fresh_draw_samples,
        target_policy="policy_a",
        draws_per_prompt=1,
    )

    # Create direct estimator with OUA enabled
    estimator = CalibratedDirectEstimator(
        target_policies=["policy_a"],
        reward_calibrator=calibrator,
        oua_jackknife=True,  # Enable OUA
        inference_method="analytical",  # Use analytical for speed
    )

    # Add fresh draws using the proper method
    estimator.add_fresh_draws("policy_a", fresh_draws_dataset)

    # Fit and estimate
    result = estimator.fit_and_estimate()

    # Check that standard errors were computed
    assert result.standard_errors is not None

    # Check metadata indicates OUA was skipped
    assert result.metadata is not None
    assert "se_components" in result.metadata
    assert (
        result.metadata["se_components"].get("oracle_uncertainty_skipped")
        == "100% oracle coverage"
    ), "Direct method should skip OUA at 100% oracle coverage"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
