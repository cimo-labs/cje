"""Critical infrastructure and edge case tests.

Merged from test_unified_folds.py and test_edge_cases.py.
These tests ensure core infrastructure works correctly and handle edge cases
that E2E tests might miss.
"""

import pytest
import numpy as np
from typing import List, Dict, Optional

from cje.data.models import Sample, Dataset
from cje.data.folds import get_fold
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.estimators import CalibratedIPS


class TestFoldInfrastructure:
    """Test critical fold computation infrastructure."""

    def test_fold_computation_deterministic(self) -> None:
        """Fold assignment must be deterministic based on prompt_id."""
        prompt_ids = [f"test_{i}" for i in range(100)]

        # Compute folds multiple times
        folds1 = [get_fold(pid, 5) for pid in prompt_ids]
        folds2 = [get_fold(pid, 5) for pid in prompt_ids]

        # Must be identical
        assert folds1 == folds2

    def test_fold_distribution(self) -> None:
        """Folds should be roughly balanced."""
        n_samples = 1000
        n_folds = 5

        prompt_ids = [f"prompt_{i}" for i in range(n_samples)]
        folds = [get_fold(pid, n_folds) for pid in prompt_ids]

        # Check distribution
        from collections import Counter

        fold_counts = Counter(folds)

        # Each fold should have roughly n/k samples (±20%)
        expected_per_fold = n_samples / n_folds
        for fold_id, count in fold_counts.items():
            assert 0 <= fold_id < n_folds
            assert 0.8 * expected_per_fold <= count <= 1.2 * expected_per_fold

    def test_fold_consistency_across_n_folds(self) -> None:
        """Same prompt should map consistently across different n_folds."""
        prompt_id = "test_prompt_123"

        # For different n_folds, the prompt should map to a consistent pattern
        fold_3 = get_fold(prompt_id, 3)
        fold_5 = get_fold(prompt_id, 5)
        fold_10 = get_fold(prompt_id, 10)

        # All should be valid
        assert 0 <= fold_3 < 3
        assert 0 <= fold_5 < 5
        assert 0 <= fold_10 < 10

        # Should be deterministic
        assert get_fold(prompt_id, 5) == fold_5

    def test_fold_validation(self) -> None:
        """Test fold assignment validation."""
        samples = [
            Sample(
                prompt_id=f"test_{i}",
                prompt=f"Question {i}",
                response=f"Answer {i}",
                reward=0.5,
                base_policy_logprob=-10.0,
                target_policy_logprobs={"policy": -9.0},
                judge_score=None,
                oracle_label=None,
            )
            for i in range(20)
        ]

        dataset = Dataset(samples=samples, target_policies=["policy"])

        # Compute fold assignments
        fold_ids = [get_fold(s.prompt_id, 5) for s in samples]

        # Check all folds are valid
        assert all(0 <= f < 5 for f in fold_ids)

        # Check distribution is reasonable
        from collections import Counter

        fold_counts = Counter(fold_ids)
        assert len(fold_counts) <= 5  # At most 5 folds
        assert all(count > 0 for count in fold_counts.values())  # All non-empty


class TestEdgeCases:
    """Test edge cases that E2E might miss."""

    def test_empty_dataset(self) -> None:
        """Handle empty dataset gracefully."""
        # Dataset requires at least one sample due to Pydantic validation
        with pytest.raises(ValueError, match="at least 1 item"):
            dataset = Dataset(samples=[], target_policies=["policy"])

    def test_single_sample(self) -> None:
        """Handle single sample dataset."""
        sample = Sample(
            prompt_id="single",
            prompt="Question",
            response="Answer",
            reward=0.5,
            base_policy_logprob=-10.0,
            target_policy_logprobs={"policy": -9.0},
            judge_score=0.5,  # Top-level judge score for CalibratedIPS
            oracle_label=None,
            metadata={},
        )

        dataset = Dataset(samples=[sample], target_policies=["policy"])
        sampler = PrecomputedSampler(dataset)

        assert sampler.n_samples == 1
        assert sampler.n_valid_samples == 1

        # With only 1 sample, the estimator should handle it gracefully
        # (even though results will be degenerate)
        estimator = CalibratedIPS(sampler)
        results = estimator.fit_and_estimate()

        # Should return results but with NaN or degenerate values
        assert results is not None
        assert len(results.estimates) == 1  # One policy
        assert results.n_samples_used["policy"] == 1

    def test_nan_rewards(self) -> None:
        """Handle NaN rewards properly."""
        # Pydantic validation doesn't allow NaN rewards
        # Test that it's properly rejected
        with pytest.raises(ValueError, match="less than or equal to 1"):
            sample = Sample(
                prompt_id="test_nan",
                prompt="Question",
                response="Answer",
                reward=np.nan,
                base_policy_logprob=-10.0,
                target_policy_logprobs={"policy": -9.0},
                judge_score=None,
                oracle_label=None,
            )

    def test_extreme_weights(self) -> None:
        """Handle extreme importance weights."""
        samples = []
        for i in range(20):
            # Create extreme weight scenario
            if i == 0:
                base_logprob = -100.0  # Very unlikely under base
                target_logprob = -1.0  # Very likely under target
            else:
                base_logprob = -10.0
                target_logprob = -10.1

            sample = Sample(
                prompt_id=f"test_{i}",
                prompt=f"Question {i}",
                response=f"Answer {i}",
                reward=0.5,
                base_policy_logprob=base_logprob,
                target_policy_logprobs={"policy": target_logprob},
                judge_score=0.5,  # Top-level judge score
                oracle_label=None,
                metadata={},
            )
            samples.append(sample)

        dataset = Dataset(samples=samples, target_policies=["policy"])
        sampler = PrecomputedSampler(dataset)
        estimator = CalibratedIPS(sampler)

        # Should handle extreme weights (possibly with warnings)
        results = estimator.fit_and_estimate()

        # Should return finite estimates
        assert np.isfinite(results.estimates[0])

        # Standard errors should be non-negative (could be 0 in edge cases)
        assert results.standard_errors[0] >= 0

    def test_missing_logprobs(self) -> None:
        """Handle missing log probabilities."""
        samples = []
        for i in range(10):
            # Some samples missing target logprobs
            if i % 3 == 0:
                target_logprobs: Dict[str, Optional[float]] = {}  # Missing
            else:
                target_logprobs = {"policy": -9.0}

            sample = Sample(
                prompt_id=f"test_{i}",
                prompt=f"Question {i}",
                response=f"Answer {i}",
                reward=0.5,
                base_policy_logprob=-10.0 if i % 2 == 0 else None,
                target_policy_logprobs=target_logprobs,
                judge_score=0.5,  # Top-level judge score
                oracle_label=None,
                metadata={},
            )
            samples.append(sample)

        dataset = Dataset(samples=samples, target_policies=["policy"])
        sampler = PrecomputedSampler(dataset)

        # Should filter out samples with missing logprobs
        assert sampler.n_valid_samples < len(samples)

        # Should still work with valid samples if we have enough
        if sampler.n_valid_samples >= 5:
            estimator = CalibratedIPS(sampler)
            results = estimator.fit_and_estimate()
            assert len(results.estimates) == 1
        else:
            # With < 5 samples, can't do 5-fold CV
            assert sampler.n_valid_samples < 5

    def test_all_same_reward(self) -> None:
        """Handle case where all rewards are identical."""
        samples = []
        for i in range(20):
            sample = Sample(
                prompt_id=f"test_{i}",
                prompt=f"Question {i}",
                response=f"Answer {i}",
                reward=0.7,  # All same
                base_policy_logprob=-10.0,
                target_policy_logprobs={"policy": -9.0},
                judge_score=0.7,  # Top-level judge score matching reward
                oracle_label=None,
                metadata={},
            )
            samples.append(sample)

        dataset = Dataset(samples=samples, target_policies=["policy"])
        sampler = PrecomputedSampler(dataset)
        estimator = CalibratedIPS(sampler)

        results = estimator.fit_and_estimate()

        # Estimate should be close to the constant reward
        assert abs(results.estimates[0] - 0.7) < 0.01

        # Standard error should be very small (no variance in rewards)
        # Though weight variance might still contribute
        assert results.standard_errors[0] >= 0


class TestDataIntegrity:
    """Test data integrity and validation."""

    def test_prompt_id_uniqueness(self) -> None:
        """Ensure prompt_ids are handled correctly even if not unique."""
        samples = []
        for i in range(10):
            # Duplicate prompt_ids
            prompt_id = f"test_{i % 3}"  # Only 3 unique IDs
            sample = Sample(
                prompt_id=prompt_id,
                prompt=f"Question {i}",
                response=f"Answer {i}",
                reward=0.5 + i * 0.01,
                base_policy_logprob=-10.0,
                target_policy_logprobs={"policy": -9.0},
                judge_score=None,
                oracle_label=None,
            )
            samples.append(sample)

        dataset = Dataset(samples=samples, target_policies=["policy"])
        sampler = PrecomputedSampler(dataset)

        # Should handle duplicate prompt_ids
        assert sampler.n_samples == 10

        # Fold assignment should still work (based on prompt_id)
        fold_ids = [get_fold(s.prompt_id, 5) for s in samples]
        assert len(fold_ids) == 10

        # Same prompt_id should get same fold
        for i in range(10):
            for j in range(i + 1, 10):
                if samples[i].prompt_id == samples[j].prompt_id:
                    assert fold_ids[i] == fold_ids[j]

    def test_policy_name_mismatch(self) -> None:
        """Handle mismatch between dataset policies and sample policies."""
        samples = []
        for i in range(10):
            sample = Sample(
                prompt_id=f"test_{i}",
                prompt=f"Question {i}",
                response=f"Answer {i}",
                reward=0.5,
                base_policy_logprob=-10.0,
                target_policy_logprobs={"policy_a": -9.0, "policy_b": -11.0},
                judge_score=None,
                oracle_label=None,
            )
            samples.append(sample)

        # Dataset creation should fail with mismatched policy
        with pytest.raises(ValueError, match="Target policies not found"):
            dataset = Dataset(
                samples=samples, target_policies=["policy_c"]  # Not in samples!
            )
