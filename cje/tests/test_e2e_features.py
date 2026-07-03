"""End-to-end tests for CJE features using real arena data.

Tests cross-cutting features (e.g. deterministic fold assignment) in
realistic scenarios with the arena dataset.
"""

import pytest


# Mark all tests in this file as E2E tests
pytestmark = [pytest.mark.e2e, pytest.mark.uses_arena_sample]
from cje.data.models import Dataset


class TestCrossFitting:
    """Test cross-fitting fold plumbing."""

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
