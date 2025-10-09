"""Test Monte Carlo variance computation utilities."""

import pytest
import numpy as np
from typing import List

from cje.data.models import Sample, Dataset
from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.data.fresh_draw_utils import compute_fresh_draw_prompt_stats


class TestMCVariance:
    """Test Monte Carlo variance computation utilities."""

    def test_fresh_draw_stats(self) -> None:
        """Test per-prompt fresh draw statistics computation."""

        # Create synthetic fresh draws with known statistics
        samples = []

        # Prompt 1: 5 draws with scores [0.1, 0.2, 0.3, 0.4, 0.5]
        # Mean = 0.3, Var = 0.025 (with ddof=1)
        for i, score in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
            sample = FreshDrawSample(
                prompt_id="prompt_1",
                target_policy="test_policy",
                draw_idx=i,
                response=f"response_{i}",
                judge_score=score,
                oracle_label=None,
                fold_id=0,
            )
            samples.append(sample)

        # Prompt 2: 3 draws with scores [0.6, 0.7, 0.8]
        # Mean = 0.7, Var = 0.01 (with ddof=1)
        for i, score in enumerate([0.6, 0.7, 0.8]):
            sample = FreshDrawSample(
                prompt_id="prompt_2",
                target_policy="test_policy",
                draw_idx=i,
                response=f"response_{i}",
                judge_score=score,
                oracle_label=None,
                fold_id=0,
            )
            samples.append(sample)

        # Create fresh draw dataset
        fresh_dataset = FreshDrawDataset(
            samples=samples, target_policy="test_policy", draws_per_prompt=4  # Average
        )

        # Compute statistics
        stats = compute_fresh_draw_prompt_stats(fresh_dataset)

        # Check prompt 1
        assert "prompt_1" in stats
        assert abs(stats["prompt_1"]["mean"] - 0.3) < 1e-10
        assert abs(stats["prompt_1"]["var"] - 0.025) < 1e-10
        assert stats["prompt_1"]["n"] == 5

        # Check prompt 2
        assert "prompt_2" in stats
        assert abs(stats["prompt_2"]["mean"] - 0.7) < 1e-10
        assert abs(stats["prompt_2"]["var"] - 0.01) < 1e-10
        assert stats["prompt_2"]["n"] == 3

    def test_variable_draws_per_prompt(self) -> None:
        """Test handling of variable number of draws per prompt."""

        samples = []

        # Create prompts with variable number of draws
        for prompt_idx in range(3):
            n_draws = prompt_idx + 1  # 1, 2, 3 draws
            for draw_idx in range(n_draws):
                sample = FreshDrawSample(
                    prompt_id=f"prompt_{prompt_idx}",
                    target_policy="test_policy",
                    draw_idx=draw_idx,
                    response=f"response_{prompt_idx}_{draw_idx}",
                    judge_score=0.5 + prompt_idx * 0.1,
                    oracle_label=None,
                    fold_id=0,
                )
                samples.append(sample)

        # Create dataset with average draws_per_prompt
        fresh_dataset = FreshDrawDataset(
            samples=samples, target_policy="test_policy", draws_per_prompt=2  # Average
        )

        # Compute statistics
        stats = compute_fresh_draw_prompt_stats(fresh_dataset)

        # Check that each prompt has correct number of draws
        assert stats["prompt_0"]["n"] == 1
        assert stats["prompt_1"]["n"] == 2
        assert stats["prompt_2"]["n"] == 3

        # Check means are correct
        assert abs(stats["prompt_0"]["mean"] - 0.5) < 1e-10
        assert abs(stats["prompt_1"]["mean"] - 0.6) < 1e-10
        assert abs(stats["prompt_2"]["mean"] - 0.7) < 1e-10
