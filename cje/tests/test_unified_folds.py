"""Tests for unified fold management system."""

import numpy as np
import pytest
import time
from typing import List

from cje.data import Dataset, Sample
from cje.data.folds import (
    get_fold,
    get_folds_for_dataset,
    get_folds_for_prompts,
    get_folds_with_oracle_balance,
)


def create_test_dataset(n_samples: int = 100) -> Dataset:
    """Create a test dataset with specified number of samples."""
    samples = []
    for i in range(n_samples):
        samples.append(
            Sample(
                prompt_id=f"prompt_{i}",
                prompt=f"Test prompt {i}",
                response=f"Response {i}",
                reward=np.random.random(),
                base_policy_logprob=-10.0,
                target_policy_logprobs={"policy": -10.0},
                judge_score=np.random.random(),
                oracle_label=None,
            )
        )
    return Dataset(samples=samples, target_policies=["policy"])


class TestBasicFoldAssignment:
    """Test basic fold assignment functionality."""

    def test_fold_determinism(self) -> None:
        """Same prompt_id always gets same fold."""
        fold1 = get_fold("test_123")
        fold2 = get_fold("test_123")
        assert fold1 == fold2

        # Different seeds give different folds (usually)
        fold3 = get_fold("test_123", seed=99)
        # Note: There's a small chance they could be the same
        # but for most prompt_ids they should differ

    def test_fold_range(self) -> None:
        """Folds are in correct range."""
        for n_folds in [2, 5, 10]:
            fold = get_fold("test", n_folds=n_folds)
            assert 0 <= fold < n_folds

    def test_empty_prompt_id_raises(self) -> None:
        """Empty prompt_id should raise ValueError."""
        with pytest.raises(ValueError, match="prompt_id cannot be empty"):
            get_fold("")

    def test_invalid_n_folds_raises(self) -> None:
        """n_folds < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="n_folds must be at least 2"):
            get_fold("test", n_folds=1)

    def test_fold_distribution(self) -> None:
        """Folds should be roughly evenly distributed."""
        n_samples = 1000
        n_folds = 5
        prompt_ids = [f"prompt_{i}" for i in range(n_samples)]
        folds = get_folds_for_prompts(prompt_ids, n_folds=n_folds)

        # Check all folds are used
        assert set(folds) == set(range(n_folds))

        # Check roughly even distribution (with some tolerance)
        expected_per_fold = n_samples / n_folds
        for fold in range(n_folds):
            count = np.sum(folds == fold)
            # Allow 20% deviation from expected
            assert abs(count - expected_per_fold) < 0.2 * expected_per_fold


class TestFilteringRobustness:
    """Test that folds survive filtering operations."""

    def test_fold_survives_filtering(self) -> None:
        """Folds unchanged after filtering samples."""
        # Create dataset
        dataset = create_test_dataset(100)

        # Get folds before filtering
        folds_before = get_folds_for_dataset(dataset)

        # Filter to subset (keep samples with reward > 0.5)
        filtered_samples = [
            s for s in dataset.samples if s.reward is not None and s.reward > 0.5
        ]
        filtered_dataset = Dataset(samples=filtered_samples, target_policies=["policy"])
        folds_after = get_folds_for_dataset(filtered_dataset)

        # Check consistency
        for i, sample in enumerate(filtered_dataset.samples):
            # Find original index
            orig_idx = next(
                j
                for j, s in enumerate(dataset.samples)
                if s.prompt_id == sample.prompt_id
            )
            assert folds_after[i] == folds_before[orig_idx]

    def test_fresh_draws_inherit_folds(self) -> None:
        """Fresh draws with same prompt_id get same fold."""
        prompt_id = "shared_prompt_123"

        # Logged data fold
        logged_fold = get_fold(prompt_id)

        # Fresh draw with same prompt_id
        fresh_fold = get_fold(prompt_id)

        assert logged_fold == fresh_fold

    def test_different_orderings_same_folds(self) -> None:
        """Reordering samples doesn't change their folds."""
        dataset = create_test_dataset(50)

        # Get original folds
        original_folds = get_folds_for_dataset(dataset)

        # Create reordered dataset
        reordered_samples = dataset.samples[::-1]  # Reverse order
        reordered_dataset = Dataset(
            samples=reordered_samples, target_policies=["policy"]
        )
        reordered_folds = get_folds_for_dataset(reordered_dataset)

        # Map back to original order and compare
        for i, sample in enumerate(dataset.samples):
            reordered_idx = next(
                j
                for j, s in enumerate(reordered_dataset.samples)
                if s.prompt_id == sample.prompt_id
            )
            assert original_folds[i] == reordered_folds[reordered_idx]


class TestOracleBalance:
    """Test oracle sample balancing functionality."""

    def test_oracle_balance_perfect_division(self) -> None:
        """Oracle samples are perfectly balanced when divisible."""
        n = 100
        prompt_ids = [f"p_{i}" for i in range(n)]

        # 20 oracle samples, 5 folds -> 4 per fold
        oracle_mask = np.zeros(n, dtype=bool)
        oracle_mask[:20] = True

        folds = get_folds_with_oracle_balance(prompt_ids, oracle_mask, n_folds=5)

        # Check oracle distribution
        oracle_folds = folds[oracle_mask]
        for fold in range(5):
            count = np.sum(oracle_folds == fold)
            assert count == 4  # 20 oracle / 5 folds = 4 per fold

    def test_oracle_balance_imperfect_division(self) -> None:
        """Oracle samples are balanced as evenly as possible."""
        n = 100
        prompt_ids = [f"p_{i}" for i in range(n)]

        # 22 oracle samples, 5 folds -> 4,4,4,5,5
        oracle_mask = np.zeros(n, dtype=bool)
        oracle_mask[:22] = True

        folds = get_folds_with_oracle_balance(prompt_ids, oracle_mask, n_folds=5)

        # Check oracle distribution
        oracle_folds = folds[oracle_mask]
        counts = [np.sum(oracle_folds == fold) for fold in range(5)]

        # Should have either 4 or 5 samples per fold
        assert all(c in [4, 5] for c in counts)
        assert sum(counts) == 22

    def test_oracle_balance_preserves_unlabeled(self) -> None:
        """Unlabeled samples still use hash-based assignment."""
        n = 100
        prompt_ids = [f"p_{i}" for i in range(n)]

        # 20 oracle, 80 unlabeled
        oracle_mask = np.zeros(n, dtype=bool)
        oracle_mask[:20] = True

        folds = get_folds_with_oracle_balance(prompt_ids, oracle_mask, n_folds=5)

        # Check unlabeled samples match standard assignment
        unlabeled_mask = ~oracle_mask
        unlabeled_ids = [prompt_ids[i] for i in range(n) if unlabeled_mask[i]]
        expected_unlabeled_folds = get_folds_for_prompts(unlabeled_ids)

        actual_unlabeled_folds = folds[unlabeled_mask]
        np.testing.assert_array_equal(actual_unlabeled_folds, expected_unlabeled_folds)

    def test_oracle_balance_empty_inputs(self) -> None:
        """Handle empty inputs gracefully."""
        # Empty prompt_ids
        folds = get_folds_with_oracle_balance([], np.array([], dtype=bool))
        assert len(folds) == 0

    def test_oracle_balance_mismatched_lengths(self) -> None:
        """Raise error on mismatched lengths."""
        prompt_ids = ["p1", "p2", "p3"]
        oracle_mask = np.array([True, False])  # Wrong length

        with pytest.raises(ValueError, match="must match"):
            get_folds_with_oracle_balance(prompt_ids, oracle_mask)


class TestConsistency:
    """Test consistency across components."""

    def test_all_components_same_folds(self) -> None:
        """Verify all estimators would get identical folds."""
        dataset = create_test_dataset(50)

        # Simulate what each component would compute
        judge_folds = get_folds_for_dataset(dataset)
        dr_folds = get_folds_for_dataset(dataset)
        stacked_folds = get_folds_for_dataset(dataset)

        # All should be identical
        np.testing.assert_array_equal(judge_folds, dr_folds)
        np.testing.assert_array_equal(dr_folds, stacked_folds)

    def test_consistent_with_different_n_folds(self) -> None:
        """Same prompt gets consistent fold index scaled to n_folds."""
        prompt_id = "test_prompt"

        # Get fold for different n_folds
        fold_5 = get_fold(prompt_id, n_folds=5)
        fold_10 = get_fold(prompt_id, n_folds=10)

        # Both should be valid
        assert 0 <= fold_5 < 5
        assert 0 <= fold_10 < 10

        # They should generally be different (unless by chance)
        # but both are deterministic
        assert get_fold(prompt_id, n_folds=5) == fold_5
        assert get_fold(prompt_id, n_folds=10) == fold_10


class TestPerformance:
    """Test performance characteristics."""

    def test_performance_large_dataset(self) -> None:
        """Fold assignment should be fast even for large datasets."""
        n = 10000  # Reduced from 100k for faster tests
        prompt_ids = [f"prompt_{i}" for i in range(n)]

        start = time.time()
        folds = get_folds_for_prompts(prompt_ids)
        elapsed = time.time() - start

        assert len(folds) == n
        assert elapsed < 1.0  # Should take < 1 second

    def test_caching_not_needed(self) -> None:
        """Repeated calls should be fast without caching."""
        prompt_ids = [f"prompt_{i}" for i in range(1000)]

        # Time first call
        start1 = time.time()
        folds1 = get_folds_for_prompts(prompt_ids)
        time1 = time.time() - start1

        # Time second call
        start2 = time.time()
        folds2 = get_folds_for_prompts(prompt_ids)
        time2 = time.time() - start2

        # Should be similar times (no caching needed)
        assert abs(time1 - time2) < 0.1
        np.testing.assert_array_equal(folds1, folds2)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset(self) -> None:
        """Handle empty dataset gracefully."""
        # Dataset requires at least one sample and policy due to validation
        # Test with empty list of prompt_ids directly instead
        folds = get_folds_for_prompts([])
        assert len(folds) == 0

    def test_single_sample(self) -> None:
        """Handle single sample dataset."""
        dataset = create_test_dataset(1)
        folds = get_folds_for_dataset(dataset)
        assert len(folds) == 1
        assert 0 <= folds[0] < 5

    def test_unicode_prompt_ids(self) -> None:
        """Handle unicode in prompt_ids."""
        prompt_ids = ["测试_1", "тест_2", "テスト_3", "🎯_4"]
        folds = get_folds_for_prompts(prompt_ids)
        assert len(folds) == 4
        assert all(0 <= f < 5 for f in folds)

    def test_very_long_prompt_id(self) -> None:
        """Handle very long prompt_ids."""
        long_id = "x" * 10000
        fold = get_fold(long_id)
        assert 0 <= fold < 5
