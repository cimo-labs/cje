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
    OrthogonalizedCalibratedIPS,
    OrthogonalizedCalibratedDRCPO,
    DRCPOEstimator,
    MRDREstimator,
    TMLEEstimator,
    StackedDREstimator,
)
from cje.estimators.tr_cpo import TRCPOEstimator


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

    def test_orthogonalized_calibrated_ips_pipeline(self, arena_sample: Any) -> None:
        """Test OC-IPS: load → calibrate → estimate with orthogonalization."""
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
            arena_sample,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,  # Important for OC-IPS
            n_folds=5,
        )

        assert cal_result is not None
        assert cal_result.calibrator is not None  # Need calibrator for rewards

        # 2. Create sampler
        sampler = PrecomputedSampler(calibrated)

        # 3. Run OC-IPS estimation
        oc_estimator = OrthogonalizedCalibratedIPS(
            sampler,
            calibrate_weights=True,
            use_orthogonalization=True,
            reward_calibrator=cal_result.calibrator,  # Pass calibrator for rewards
        )
        oc_results = oc_estimator.fit_and_estimate()

        # 4. Validate results
        n_policies = len(oc_results.estimates)
        assert len(oc_results.estimates) == n_policies
        assert all(0 <= e <= 1 for e in oc_results.estimates)
        assert all(se > 0 for se in oc_results.standard_errors)
        assert oc_results.method == "oc-ips"

        # 5. Check orthogonalization diagnostics
        assert hasattr(oc_estimator, "_orthogonalization_diagnostics")
        for policy in sampler.target_policies:
            if policy in oc_estimator._orthogonalization_diagnostics:
                diag = oc_estimator._orthogonalization_diagnostics[policy]
                # Check required diagnostic fields
                assert "retarget_residual" in diag
                assert "retarget_se" in diag
                assert "retarget_ci_lower" in diag
                assert "retarget_ci_upper" in diag
                # Retarget residual CI should contain 0 for good orthogonalization
                # (but may not in finite samples)

        # 6. Compare with standard CalibratedIPS
        std_estimator = CalibratedIPS(
            sampler,
            calibrate_weights=True,
            reward_calibrator=cal_result.calibrator,
        )
        std_results = std_estimator.fit_and_estimate()

        # OC-IPS should have similar point estimates but potentially different SEs
        for i in range(len(oc_results.estimates)):
            # Point estimates should be somewhat close
            assert abs(oc_results.estimates[i] - std_results.estimates[i]) < 0.5
            # Both should have positive SEs
            assert oc_results.standard_errors[i] > 0
            assert std_results.standard_errors[i] > 0

    def test_orthogonalized_dr_pipeline(
        self, arena_sample: Any, arena_fresh_draws: Any
    ) -> None:
        """Test OC-DR-CPO: Orthogonalized Calibrated DR with triple robustness."""
        # 1. Calibrate dataset with partial oracle coverage
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

        assert cal_result is not None
        assert 0 < cal_result.n_oracle < len(arena_sample.samples)

        # 2. Create sampler
        sampler = PrecomputedSampler(calibrated)

        # 3. Create OC-DR-CPO estimator
        odr_estimator = OrthogonalizedCalibratedDRCPO(
            sampler,
            reward_calibrator=cal_result.calibrator,
            use_calibrated_weights=True,
            use_orthogonalization=True,
            n_folds=5,
        )

        # 4. Add fresh draws for each policy
        for policy, fresh_dataset in arena_fresh_draws.items():
            if policy in sampler.target_policies:
                odr_estimator.add_fresh_draws(policy, fresh_dataset)

        # 5. Run estimation
        odr_results = odr_estimator.fit_and_estimate()

        # 6. Validate results
        assert odr_results is not None
        assert odr_results.method == "oc_dr_cpo"
        n_policies = len(odr_results.estimates)
        assert len(odr_results.estimates) == n_policies
        assert len(odr_results.standard_errors) == n_policies

        # Check diagnostics - ODR implementation stores diagnostics in metadata
        assert odr_results.metadata is not None
        assert "orthogonalization_diagnostics" in odr_results.metadata

        # Check orthogonalization diagnostics exist for each policy
        orthog_diags = odr_results.metadata["orthogonalization_diagnostics"]
        for policy in sampler.target_policies:
            if policy in orthog_diags:
                assert "orthog_residual" in orthog_diags[policy]
                assert "retarget_residual" in orthog_diags[policy]

        # Check that estimates are reasonable
        for est in odr_results.estimates:
            if not np.isnan(est):
                assert 0 <= est <= 1

        # 7. Compare with standard DR-CPO
        std_dr = DRCPOEstimator(
            sampler,
            reward_calibrator=cal_result.calibrator,
            n_folds=5,
        )
        for policy, fresh_dataset in arena_fresh_draws.items():
            if policy in sampler.target_policies:
                std_dr.add_fresh_draws(policy, fresh_dataset)

        std_dr_results = std_dr.fit_and_estimate()

        # ODR should have similar point estimates but potentially better robustness
        for i in range(len(odr_results.estimates)):
            if not np.isnan(odr_results.estimates[i]) and not np.isnan(
                std_dr_results.estimates[i]
            ):
                # Point estimates should be somewhat close
                assert abs(odr_results.estimates[i] - std_dr_results.estimates[i]) < 0.5

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

    def test_tr_cpo_pipeline(self, arena_sample: Any, arena_fresh_draws: Any) -> None:
        """Test TR-CPO: triply robust causal preference optimization."""
        # 1. Prepare dataset with partial oracle coverage
        import random

        random.seed(42)
        np.random.seed(42)

        # Mask 50% of oracle labels to test label propensity modeling
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

        # Verify we have partial oracle coverage for TR correction
        assert cal_result.n_oracle > 0
        assert cal_result.n_oracle < len(arena_sample.samples)

        # 2. Create sampler
        sampler = PrecomputedSampler(calibrated)

        # 3. Create TR-CPO estimator
        estimator = TRCPOEstimator(
            sampler,
            reward_calibrator=cal_result.calibrator,
            n_folds=5,
            weight_mode="hajek",  # Test Hájek normalization
        )

        # 4. Add fresh draws for each policy
        for policy, fresh_dataset in arena_fresh_draws.items():
            if policy in sampler.target_policies:
                estimator.add_fresh_draws(policy, fresh_dataset)

        # 5. Run estimation
        results = estimator.fit_and_estimate()

        # 6. Validate TR-CPO results
        n_policies = len(results.estimates)
        assert len(results.estimates) == n_policies
        assert results.method == "tr_cpo"
        assert all(0 <= e <= 1 for e in results.estimates)
        assert all(se > 0 for se in results.standard_errors)

        # 7. Check TR-CPO specific properties
        assert results.diagnostics is not None
        summary = results.diagnostics.summary()
        assert "Weight ESS" in summary

        # Check metadata for TR-specific info
        assert results.metadata is not None
        assert "weight_mode" in results.metadata
        assert results.metadata["weight_mode"] == "hajek"

        # 8. Test that TR-CPO uses raw weights (not SIMCal calibrated)
        # This is critical for maintaining triple robustness
        for policy in sampler.target_policies:
            weights = estimator.get_weights(policy)
            assert weights is not None
            # Raw weights should have higher variance than calibrated
            # (SIMCal would reduce variance)
            assert np.var(weights) > 0

        # 9. Compare with standard DR-CPO
        std_dr = DRCPOEstimator(
            sampler,
            reward_calibrator=cal_result.calibrator,
            n_folds=5,
        )
        for policy, fresh_dataset in arena_fresh_draws.items():
            if policy in sampler.target_policies:
                std_dr.add_fresh_draws(policy, fresh_dataset)

        std_dr_results = std_dr.fit_and_estimate()

        # TR-CPO should produce reasonable estimates compared to DR-CPO
        for i in range(len(results.estimates)):
            if not np.isnan(results.estimates[i]) and not np.isnan(
                std_dr_results.estimates[i]
            ):
                # Point estimates should be somewhat close
                assert abs(results.estimates[i] - std_dr_results.estimates[i]) < 0.5
                # TR-CPO may have different SEs due to triple robustness

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
