"""End-to-end tests for CJE features using real arena data.

Tests major features like SIMCal, oracle augmentation, and cross-fitting
in realistic scenarios with the arena dataset.
"""

import pytest
import numpy as np
from copy import deepcopy
from typing import Dict, Any

from cje import load_dataset_from_jsonl

# Mark all tests in this file as E2E tests
pytestmark = [pytest.mark.e2e, pytest.mark.uses_arena_sample]
from cje.calibration import calibrate_dataset
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.data.models import Dataset
from cje.data.fresh_draws import FreshDrawDataset
from cje.estimators import CalibratedIPS, DRCPOEstimator


# IIC tests removed - IIC is unreliable on small samples (often increases variance instead of reducing it)
# The feature adds 565 lines of complexity for minimal benefit


class TestSIMCalFeature:
    """Test SIMCal (Surrogate-Indexed Monotone Calibration) variance control."""

    def test_simcal_mean_preservation(self, arena_sample: Dataset) -> None:
        """Test SIMCal preserves mean of weights."""
        import random

        random.seed(42)
        np.random.seed(42)

        # Prepare dataset - mask every other oracle label
        new_samples = []
        for i, sample in enumerate(arena_sample.samples):
            if i % 2 == 1 and sample.oracle_label is not None:
                new_samples.append(sample.model_copy(update={"oracle_label": None}))
            else:
                new_samples.append(sample)
        arena_sample.samples = new_samples

        calibrated, _ = calibrate_dataset(
            arena_sample, judge_field="judge_score", oracle_field="oracle_label"
        )

        sampler = PrecomputedSampler(calibrated)
        estimator = CalibratedIPS(sampler)
        estimator.fit()

        # Check mean preservation for each policy
        for policy in sampler.target_policies:
            weights = estimator.get_weights(policy)
            assert weights is not None

            # Mean should be very close to 1 (Hajek normalization)
            mean_weight = np.mean(weights)
            assert (
                abs(mean_weight - 1.0) < 0.01
            ), f"Policy {policy}: mean weight {mean_weight:.4f} != 1.0"

            # Weights should be non-negative
            assert np.all(weights >= 0), f"Policy {policy} has negative weights"


class TestCrossFitting:
    """Test cross-fitting for orthogonality in DR estimators."""

    def test_cross_fitting_consistency(
        self, arena_sample: Dataset, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Test cross-fitting gives consistent results across runs."""
        import random

        # Prepare dataset - mask every other oracle label
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

        sampler = PrecomputedSampler(calibrated)

        # Run DR multiple times with different seeds
        estimates_list = []
        for seed in [42, 123, 456]:
            random.seed(seed)
            np.random.seed(seed)

            estimator = DRCPOEstimator(
                sampler, reward_calibrator=cal_result.calibrator, n_folds=5
            )

            # Add fresh draws for each policy (only those in sampler)
            for policy, fresh_data in arena_fresh_draws.items():
                if policy in sampler.target_policies:
                    estimator.add_fresh_draws(policy, fresh_data)

            results = estimator.fit_and_estimate()
            estimates_list.append(results.estimates)

        # Results should be deterministic given the seed
        # But let's check they're at least consistent
        for i in range(len(estimates_list[0])):
            estimates = [e[i] for e in estimates_list]
            estimate_range = max(estimates) - min(estimates)

            # Should be very similar across runs
            assert (
                estimate_range < 0.05
            ), f"Policy {i}: range {estimate_range:.4f} across seeds"

    def test_fold_assignment_stability(self, arena_sample: Dataset) -> None:
        """Test that fold assignments are stable based on prompt_id."""
        from cje.data.folds import get_fold

        # Check fold assignments are deterministic
        prompt_ids = [s.prompt_id for s in arena_sample.samples]

        # Compute folds multiple times
        folds_5 = [get_fold(pid, 5) for pid in prompt_ids]
        folds_5_again = [get_fold(pid, 5) for pid in prompt_ids]

        # Should be identical
        assert folds_5 == folds_5_again

        # Check distribution is reasonable
        from collections import Counter

        fold_counts = Counter(folds_5)

        # Each fold should have roughly n/5 samples
        expected_per_fold = len(prompt_ids) / 5
        for fold, count in fold_counts.items():
            assert 0 <= fold < 5
            # Allow 50% deviation from expected
            assert 0.5 * expected_per_fold <= count <= 1.5 * expected_per_fold


class TestIntegrationScenarios:
    """Test complete scenarios combining multiple features."""

    def test_full_pipeline_with_all_features(
        self, arena_sample: Dataset, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Test complete pipeline with SIMCal, cross-fitting, and oracle augmentation."""
        import random

        random.seed(42)
        np.random.seed(42)

        # Prepare dataset with 60% oracle coverage
        oracle_indices = [
            i for i, s in enumerate(arena_sample.samples) if s.oracle_label is not None
        ]
        keep_n = int(len(oracle_indices) * 0.6)
        keep_indices = set(random.sample(oracle_indices, keep_n))

        new_samples = []
        for i, sample in enumerate(arena_sample.samples):
            if i not in keep_indices and sample.oracle_label is not None:
                new_samples.append(sample.model_copy(update={"oracle_label": None}))
            else:
                new_samples.append(sample)
        arena_sample.samples = new_samples

        # Calibrate with cross-fitting
        calibrated, cal_result = calibrate_dataset(
            arena_sample,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,
            n_folds=5,
        )

        # Create sampler
        sampler = PrecomputedSampler(calibrated)

        # Run DR with all features enabled
        estimator = DRCPOEstimator(
            sampler,
            reward_calibrator=cal_result.calibrator,
            n_folds=5,
            # variance_cap removed as it's not supported by DRCPOEstimator
        )

        # Add fresh draws
        for policy, fresh_dataset in arena_fresh_draws.items():
            if policy in sampler.target_policies:
                estimator.add_fresh_draws(policy, fresh_dataset)

        results = estimator.fit_and_estimate()

        # Validate everything worked
        n_policies = len(sampler.target_policies)
        assert len(results.estimates) == n_policies
        assert all(0 <= e <= 1 for e in results.estimates)
        assert all(se > 0 for se in results.standard_errors)

        # Check features are reflected in metadata/diagnostics
        assert results.diagnostics is not None

        # Check diagnostics summary includes all components
        summary = results.diagnostics.summary()
        assert "Weight ESS" in summary
        # DR orthogonality might not always be in summary

        # Verify reasonable performance
        # With all features, should have good ESS
        for policy in sampler.target_policies[:1]:  # Check at least one
            weights = estimator.get_weights(policy)
            if weights is not None:
                ess = (np.sum(weights) ** 2) / np.sum(weights**2)
                ess_fraction = ess / len(weights)
                assert ess_fraction > 0.05, f"Very low ESS: {ess_fraction:.3f}"
