#!/usr/bin/env python3
"""
Test that oracle-jackknife inference is correctly skipped at 100% oracle coverage.

This test ensures that calibration uncertainty is not added to
standard_errors when oracle_coverage = 1.0, preventing the bug
discovered in the ablation experiments.
"""

import pytest
import numpy as np


def test_direct_method_skips_oua_at_full_coverage() -> None:
    """Test that CalibratedDirectEstimator skips oracle-jackknife inference at 100% oracle coverage.

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

    # Create direct estimator with oracle-jackknife inference enabled
    estimator = CalibratedDirectEstimator(
        target_policies=["policy_a"],
        reward_calibrator=calibrator,
        oua_jackknife=True,  # Enable calibration-aware inference
        inference_method="analytical",  # Use analytical for speed
    )

    # Add fresh draws using the proper method
    estimator.add_fresh_draws("policy_a", fresh_draws_dataset)

    # Fit and estimate
    result = estimator.fit_and_estimate()

    # Check that standard errors were computed
    assert result.standard_errors is not None

    # Check metadata indicates oracle-jackknife inference was skipped
    assert result.metadata is not None
    assert "se_components" in result.metadata
    assert (
        result.metadata["se_components"].get("oracle_uncertainty_skipped")
        == "100% oracle coverage"
    ), "Direct method should skip oracle-jackknife inference at 100% oracle coverage"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
