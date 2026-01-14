"""End-to-end tests for all CJE estimators using real arena data.

These tests validate complete pipelines from data loading through estimation
and diagnostics using the 100-sample arena dataset.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Any

from cje import load_dataset_from_jsonl
from cje.data.models import Dataset

# Mark all tests in this file as E2E tests
pytestmark = [pytest.mark.e2e, pytest.mark.uses_arena_sample]
from cje.calibration import calibrate_dataset
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.estimators import (
    CalibratedIPS,
    DRCPOEstimator,
    MRDREstimator,
    TMLEEstimator,
    StackedDREstimator,
)


class TestE2EEstimators:
    """Complete pipeline tests for each estimator."""

    def test_calibrated_ips_pipeline(self, arena_sample: Any) -> None:
        """Test CalibratedIPS: load → calibrate → estimate → diagnostics."""
        # 1. Calibrate dataset with partial oracle coverage
        import random

        random.seed(42)
        np.random.seed(42)

        # Mask 50% of oracle labels to simulate realistic scenario
        new_samples = []
        for i, sample in enumerate(arena_sample.samples):
            if i % 2 == 1 and sample.oracle_label is not None:
                new_samples.append(sample.model_copy(update={"oracle_label": None}))
            else:
                new_samples.append(sample)
        arena_sample.samples = new_samples

        calibrated, cal_result = calibrate_dataset(
            arena_sample, judge_field="judge_score", oracle_field="oracle_label"
        )

        assert cal_result is not None
        assert cal_result.n_oracle > 0
        assert cal_result.n_oracle < len(arena_sample.samples)

        # 2. Create sampler
        sampler = PrecomputedSampler(calibrated)
        assert sampler.n_samples == len(calibrated.samples)
        n_policies = len(sampler.target_policies)
        assert n_policies >= 2  # Arena sample has multiple policies

        # 3. Run estimation
        estimator = CalibratedIPS(sampler)
        results = estimator.fit_and_estimate()

        # 4. Validate results
        assert len(results.estimates) == n_policies
        assert all(0 <= e <= 1 for e in results.estimates)
        assert all(se > 0 for se in results.standard_errors)
        assert results.method == "calibrated_ips"

        # 5. Check diagnostics work
        assert results.diagnostics is not None
        summary = results.diagnostics.summary()
        assert "ESS" in summary or "Weight ESS" in summary
        assert "Method" in summary

        # 6. Check specific policies
        for i, policy in enumerate(sampler.target_policies):
            weights = estimator.get_weights(policy)
            assert weights is not None
            assert len(weights) == sampler.n_valid_samples  # Should match valid samples
            assert np.abs(np.mean(weights) - 1.0) < 0.01  # Calibrated to mean 1

            # Check estimates are reasonable
            assert 0 <= results.estimates[i] <= 1
            assert results.standard_errors[i] > 0

    def test_dr_cpo_pipeline(self, arena_sample: Any, arena_fresh_draws: Any) -> None:
        """Test DR-CPO: load → calibrate → add fresh draws → estimate."""
        # 1. Calibrate dataset
        import random

        random.seed(42)
        np.random.seed(42)

        # Use 50% oracle coverage
        new_samples = []
        for i, sample in enumerate(arena_sample.samples):
            if i % 2 == 1 and sample.oracle_label is not None:
                new_samples.append(sample.model_copy(update={"oracle_label": None}))
            else:
                new_samples.append(sample)
        arena_sample.samples = new_samples

        calibrated, cal_result = calibrate_dataset(
            arena_sample,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,
            n_folds=5,
        )

        # 2. Create sampler
        sampler = PrecomputedSampler(calibrated)

        # 3. Create DR estimator and add fresh draws
        estimator = DRCPOEstimator(
            sampler, reward_calibrator=cal_result.calibrator, n_folds=5
        )

        # 4. Add fresh draws manually for each policy
        for policy, fresh_dataset in arena_fresh_draws.items():
            if policy in sampler.target_policies:
                estimator.add_fresh_draws(policy, fresh_dataset)

        # 5. Run estimation
        results = estimator.fit_and_estimate()

        # 6. Validate DR results
        n_policies = len(results.estimates)
        assert len(results.estimates) == n_policies
        assert all(0 <= e <= 1 for e in results.estimates)
        assert results.method == "dr_cpo"

        # 7. Check DR diagnostics
        assert results.diagnostics is not None
        summary = results.diagnostics.summary()
        assert "Weight ESS" in summary
        assert "Outcome R²" in summary or "DR" in summary

        # DR should generally have smaller SEs than IPS
        assert all(se > 0 for se in results.standard_errors)

    def test_mrdr_pipeline(self, arena_sample: Any, arena_fresh_draws: Any) -> None:
        """Test MRDR: multiply robust doubly robust estimation."""
        # 1. Prepare dataset
        import random

        random.seed(42)
        np.random.seed(42)

        new_samples = []
        for i, sample in enumerate(arena_sample.samples):
            if i % 2 == 1 and sample.oracle_label is not None:
                new_samples.append(sample.model_copy(update={"oracle_label": None}))
            else:
                new_samples.append(sample)
        arena_sample.samples = new_samples

        calibrated, cal_result = calibrate_dataset(
            arena_sample,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,
            n_folds=5,
        )

        # 2. Create sampler
        sampler = PrecomputedSampler(calibrated)

        # 3. Create MRDR estimator and add fresh draws
        estimator = MRDREstimator(
            sampler,
            reward_calibrator=cal_result.calibrator,
            n_folds=5,
            omega_mode="w2",  # Test specific MRDR mode
        )

        # 4. Add fresh draws
        for policy, fresh_dataset in arena_fresh_draws.items():
            if policy in sampler.target_policies:
                estimator.add_fresh_draws(policy, fresh_dataset)

        # 5. Run estimation
        results = estimator.fit_and_estimate()

        # 6. Validate MRDR results
        n_policies = len(results.estimates)
        assert len(results.estimates) == n_policies
        assert results.method == "mrdr"
        assert all(0 <= e <= 1 for e in results.estimates)

        # 7. Check MRDR-specific diagnostics
        assert results.diagnostics is not None
        assert "omega_mode" in results.metadata
        assert results.metadata["omega_mode"] == "w2"

    def test_tmle_pipeline(self, arena_sample: Any, arena_fresh_draws: Any) -> None:
        """Test TMLE: targeted maximum likelihood estimation."""
        # 1. Prepare dataset
        import random

        random.seed(42)
        np.random.seed(42)

        new_samples = []
        for i, sample in enumerate(arena_sample.samples):
            if i % 2 == 1 and sample.oracle_label is not None:
                new_samples.append(sample.model_copy(update={"oracle_label": None}))
            else:
                new_samples.append(sample)
        arena_sample.samples = new_samples

        calibrated, cal_result = calibrate_dataset(
            arena_sample,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,
            n_folds=5,
        )

        # 2. Create sampler
        sampler = PrecomputedSampler(calibrated)

        # 3. Create TMLE estimator and add fresh draws
        estimator = TMLEEstimator(
            sampler, reward_calibrator=cal_result.calibrator, n_folds=5
        )

        # 4. Add fresh draws
        for policy, fresh_dataset in arena_fresh_draws.items():
            if policy in sampler.target_policies:
                estimator.add_fresh_draws(policy, fresh_dataset)

        # 5. Run estimation
        results = estimator.fit_and_estimate()

        # 6. Validate TMLE results
        n_policies = len(results.estimates)
        assert len(results.estimates) == n_policies
        assert results.method == "tmle"
        assert all(0 <= e <= 1 for e in results.estimates)

        # 7. TMLE should have targeting step info
        assert results.diagnostics is not None
        # TMLE uses the same DR diagnostics
        summary = results.diagnostics.summary()
        assert "Weight ESS" in summary

    def test_stacked_dr_pipeline(
        self, arena_sample: Any, arena_fresh_draws: Any
    ) -> None:
        """Test StackedDR: optimal combination of DR estimators."""
        # 1. Prepare dataset
        import random

        random.seed(42)
        np.random.seed(42)

        new_samples = []
        for i, sample in enumerate(arena_sample.samples):
            if i % 2 == 1 and sample.oracle_label is not None:
                new_samples.append(sample.model_copy(update={"oracle_label": None}))
            else:
                new_samples.append(sample)
        arena_sample.samples = new_samples

        calibrated, cal_result = calibrate_dataset(
            arena_sample,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,
            n_folds=5,
        )

        # 2. Create sampler
        sampler = PrecomputedSampler(calibrated)

        # 3. Create Stacked DR estimator and add fresh draws
        estimator = StackedDREstimator(
            sampler,
            reward_calibrator=cal_result.calibrator,
            n_folds=5,
            parallel=False,  # For testing, avoid parallelism
        )

        # 4. Add fresh draws to stacked estimator
        for policy, fresh_dataset in arena_fresh_draws.items():
            if policy in sampler.target_policies:
                estimator.add_fresh_draws(policy, fresh_dataset)

        # 5. Run estimation
        results = estimator.fit_and_estimate()

        # 6. Validate stacked results
        n_policies = len(sampler.target_policies)
        assert len(results.estimates) == n_policies
        # Method name includes the component estimators
        assert results.method.startswith("StackedDR(")
        assert all(0 <= e <= 1 for e in results.estimates)

        # 7. Check stacking weights (one set per policy)
        assert "stacking_weights" in results.metadata
        weights = results.metadata["stacking_weights"]
        assert len(weights) == n_policies  # One weight vector per policy

        # Check each policy's weights
        valid_estimators = results.metadata.get("valid_estimators", [])
        # Get the actual valid components that succeeded (have non-None results)
        actual_valid = [
            name
            for name in valid_estimators
            if name in results.metadata.get("component_results", {})
            and results.metadata["component_results"][name] is not None
        ]

        for policy, policy_weights in weights.items():
            # Weights should match the number of components that actually succeeded
            # (not all valid_estimators may have succeeded)
            assert len(policy_weights) >= 1  # At least one component
            assert len(policy_weights) <= len(
                valid_estimators
            )  # At most all components
            # Note: Optimal stacking can produce negative weights (valid for minimum variance)
            # Just check they sum to 1
            assert abs(sum(policy_weights) - 1.0) < 0.01  # Sum to 1

        # 8. Check stacking diagnostics exist per policy
        assert "stacking_diagnostics" in results.metadata
        stacking_diag = results.metadata["stacking_diagnostics"]
        assert isinstance(stacking_diag, dict)
        for policy, diag in stacking_diag.items():
            assert "condition_pre" in diag
            assert "condition_post" in diag
            assert "weights" in diag


class TestEstimatorConsistency:
    """Test that different estimators give consistent results on good data."""

    def test_estimator_agreement(
        self, arena_sample: Any, arena_fresh_draws: Any
    ) -> None:
        """Different estimators should broadly agree on arena data."""
        import random

        random.seed(42)
        np.random.seed(42)

        # Prepare dataset with high oracle coverage for better agreement
        new_samples = []
        for i, sample in enumerate(arena_sample.samples):
            if i % 4 != 0 and sample.oracle_label is not None:  # Keep 75%
                new_samples.append(sample.model_copy(update={"oracle_label": None}))
            else:
                new_samples.append(sample)
        arena_sample.samples = new_samples

        calibrated, cal_result = calibrate_dataset(
            arena_sample,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,
            n_folds=5,
        )

        sampler = PrecomputedSampler(calibrated)

        # Run all estimators
        ips = CalibratedIPS(sampler)
        dr = DRCPOEstimator(sampler, reward_calibrator=cal_result.calibrator, n_folds=5)
        tmle = TMLEEstimator(
            sampler, reward_calibrator=cal_result.calibrator, n_folds=5
        )

        # Add fresh draws to DR estimators
        for policy, fresh_dataset in arena_fresh_draws.items():
            if policy in sampler.target_policies:
                dr.add_fresh_draws(policy, fresh_dataset)
                tmle.add_fresh_draws(policy, fresh_dataset)

        results_ips = ips.fit_and_estimate()
        results_dr = dr.fit_and_estimate()
        results_tmle = tmle.fit_and_estimate()

        # Check estimates are reasonably close
        for i in range(len(results_ips.estimates)):
            estimates = [
                results_ips.estimates[i],
                results_dr.estimates[i],
                results_tmle.estimates[i],
            ]

            # All should be in [0, 1]
            assert all(0 <= e <= 1 for e in estimates)

            # Range should be reasonable (not wildly different)
            estimate_range = max(estimates) - min(estimates)
            max_se = max(
                results_ips.standard_errors[i],
                results_dr.standard_errors[i],
                results_tmle.standard_errors[i],
            )

            # Estimates should generally agree, but with poor overlap they can differ
            # Allow up to 5 SEs of difference (or 0.4 absolute) for policies with bad overlap
            tolerance = max(5 * max_se, 0.4)
            assert (
                estimate_range < tolerance
            ), f"Policy {i}: range {estimate_range:.3f} > tolerance {tolerance:.3f}"

    # Note: Removed test_dr_improves_over_ips as it had an unreliable assumption.
    # With only 100 samples, good overlap (high ESS), and minimal fresh draws (1 per prompt),
    # DR won't necessarily improve over IPS. DR's advantages emerge with:
    # - Poor overlap (low ESS) where the outcome model helps
    # - Many fresh draws for better direct estimation
    # - Larger sample sizes
    # The test was failing intermittently due to these factors.


@pytest.mark.slow
class TestEstimatorStress:
    """Stress tests for estimators with edge cases."""

    def test_low_oracle_coverage(self, arena_sample: Any) -> None:
        """Test with very limited oracle labels (10%)."""
        import random

        random.seed(42)
        np.random.seed(42)

        # Keep only 10% of oracle labels
        oracle_indices = [
            i for i, s in enumerate(arena_sample.samples) if s.oracle_label is not None
        ]
        keep_n = max(2, len(oracle_indices) // 10)
        keep_indices = set(random.sample(oracle_indices, keep_n))

        new_samples = []
        for i, sample in enumerate(arena_sample.samples):
            if i not in keep_indices and sample.oracle_label is not None:
                new_samples.append(sample.model_copy(update={"oracle_label": None}))
            else:
                new_samples.append(sample)
        arena_sample.samples = new_samples

        calibrated, cal_result = calibrate_dataset(
            arena_sample, judge_field="judge_score", oracle_field="oracle_label"
        )

        sampler = PrecomputedSampler(calibrated)
        estimator = CalibratedIPS(sampler)
        results = estimator.fit_and_estimate()

        # Should still work but with higher uncertainty
        n_policies = len(results.estimates)
        assert len(results.estimates) == n_policies
        assert all(se > 0 for se in results.standard_errors)
        # Standard errors should be relatively high due to limited oracle data
        assert all(se > 0.01 for se in results.standard_errors)
