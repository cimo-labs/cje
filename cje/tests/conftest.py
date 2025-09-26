"""Shared test fixtures and utilities for CJE test suite.

This file is automatically loaded by pytest and provides common fixtures
and utilities used across multiple test files.

Key fixtures:
- arena_sample: Real 100-sample arena dataset
- arena_sample_small: First 20 samples for fast tests
- arena_fresh_draws: Real fresh draws from arena
"""

import pytest
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

from cje.data.models import Sample, Dataset, EstimationResult
from cje.data.fresh_draws import (
    FreshDrawSample,
    FreshDrawDataset,
    load_fresh_draws_from_jsonl,
)
from cje.data.precomputed_sampler import PrecomputedSampler
from cje import load_dataset_from_jsonl


# ============================================================================
# Arena Sample Fixtures (Real Data)
# ============================================================================


@pytest.fixture(scope="session")
def arena_dataset() -> Dataset:
    """Load real arena sample dataset once per session (100 samples).

    This is real data from Arena with judge scores and oracle labels.
    Use this for integration tests and realistic scenarios.
    Session-scoped for performance.
    """
    data_path = Path(__file__).parent / "data" / "arena_sample" / "dataset.jsonl"
    if not data_path.exists():
        pytest.skip(f"Arena sample not found at {data_path}")
    return load_dataset_from_jsonl(str(data_path))


@pytest.fixture
def arena_sample() -> Dataset:
    """Load real arena sample dataset (100 samples).

    Function-scoped version for tests that modify the dataset.
    """
    data_path = Path(__file__).parent / "data" / "arena_sample" / "dataset.jsonl"
    if not data_path.exists():
        pytest.skip(f"Arena sample not found at {data_path}")
    return load_dataset_from_jsonl(str(data_path))


@pytest.fixture
def arena_sample_small(arena_dataset: Dataset) -> Dataset:
    """First 20 samples of arena dataset for fast tests.

    Smaller subset for unit tests that need real data but fast execution.
    """
    from copy import deepcopy

    small_dataset: Dataset = deepcopy(arena_dataset)
    small_dataset.samples = small_dataset.samples[:20]
    return small_dataset


@pytest.fixture
def arena_calibrated(arena_sample: Dataset) -> Dataset:
    """Pre-calibrated arena data with 50% oracle coverage.

    Ready for use with estimators that need calibrated rewards.
    """
    from cje.calibration import calibrate_dataset
    from copy import deepcopy
    import random

    # Create a copy to avoid modifying the original
    dataset = deepcopy(arena_sample)

    # Mask 50% of oracle labels to simulate partial coverage
    samples_with_oracle = [
        i
        for i, s in enumerate(dataset.samples)
        if "oracle_label" in s.metadata and s.metadata["oracle_label"] is not None
    ]

    if len(samples_with_oracle) > 2:
        random.seed(42)
        # Keep only 50% of oracle labels
        n_keep = max(2, len(samples_with_oracle) // 2)
        keep_indices = set(random.sample(samples_with_oracle, n_keep))

        for i in range(len(dataset.samples)):
            if i not in keep_indices and "oracle_label" in dataset.samples[i].metadata:
                # Remove oracle label for this sample
                dataset.samples[i].metadata["oracle_label"] = None

    calibrated_dataset, _ = calibrate_dataset(
        dataset, judge_field="judge_score", oracle_field="oracle_label"
    )
    return calibrated_dataset


@pytest.fixture
def arena_sampler(arena_calibrated: Dataset) -> PrecomputedSampler:
    """Ready-to-use sampler with calibrated arena data.

    For tests that need a fully configured sampler.
    """
    from cje.data.precomputed_sampler import PrecomputedSampler

    return PrecomputedSampler(arena_calibrated)


@pytest.fixture
def arena_fresh_draws() -> Dict[str, FreshDrawDataset]:
    """Load real fresh draws from arena sample.

    Returns dict mapping policy names to FreshDrawDataset objects.
    Policies: clone, premium, parallel_universe_prompt, unhelpful

    Note: The response files aren't in fresh draw format, so we convert them.
    """
    import json

    responses_dir = Path(__file__).parent / "data" / "arena_sample" / "responses"
    dataset_path = Path(__file__).parent / "data" / "arena_sample" / "dataset.jsonl"

    if not responses_dir.exists():
        pytest.skip(f"Fresh draws not found at {responses_dir}")

    # First, get the set of prompt_ids that exist in the dataset
    valid_prompt_ids = set()
    with open(dataset_path) as f:
        for line in f:
            data = json.loads(line)
            valid_prompt_ids.add(data["prompt_id"])

    fresh_draws = {}
    for policy_file in responses_dir.glob("*_responses.jsonl"):
        policy_name = policy_file.stem.replace("_responses", "")

        # Convert response format to fresh draw format
        samples = []
        with open(policy_file) as f:
            for line in f:
                data = json.loads(line)

                # Only include samples with prompt_ids that exist in the dataset
                if data["prompt_id"] not in valid_prompt_ids:
                    continue

                # Convert to FreshDrawSample format with all available fields
                sample = FreshDrawSample(
                    prompt_id=data["prompt_id"],
                    target_policy=policy_name,  # Use policy_name, not data["policy"]
                    judge_score=data["metadata"]["judge_score"],
                    draw_idx=0,  # Single draw per prompt
                    response=data.get(
                        "response", ""
                    ),  # Include response for completeness
                    fold_id=None,  # Will be assigned by sampler if needed
                )
                samples.append(sample)

        if samples:
            fresh_draws[policy_name] = FreshDrawDataset(
                target_policy=policy_name, draws_per_prompt=1, samples=samples
            )

    return fresh_draws


# ============================================================================
# Synthetic Test Datasets (Legacy - prefer arena fixtures)
# ============================================================================


@pytest.fixture
def basic_dataset() -> Dataset:
    """Create a basic dataset with judge scores and rewards.

    Returns:
        Dataset with 20 samples, all required fields populated.
    """
    samples = []
    for i in range(20):
        sample = Sample(
            prompt_id=f"test_{i}",
            prompt=f"Test question {i}",
            response=f"Test response {i}",
            reward=0.5 + 0.3 * np.sin(i / 3),  # Varies between 0.2 and 0.8
            base_policy_logprob=-10.0 - i * 0.1,
            target_policy_logprobs={
                "policy_a": -9.0 - i * 0.1,
                "policy_b": -11.0 - i * 0.1,
            },
            metadata={
                "judge_score": 0.5 + 0.3 * np.sin(i / 3),
            },
        )
        samples.append(sample)

    return Dataset(
        samples=samples,
        target_policies=["policy_a", "policy_b"],
    )


@pytest.fixture
def dataset_with_oracle() -> Dataset:
    """Create dataset with 50% oracle label coverage for calibration.

    Returns:
        Dataset with 20 samples, 10 having oracle labels.
    """
    samples = []
    for i in range(20):
        judge_score = 0.5 + 0.3 * np.sin(i / 3)
        oracle_label = judge_score + 0.05 * np.random.normal() if i < 10 else None

        sample = Sample(
            prompt_id=f"test_{i}",
            prompt=f"Question {i}",
            response=f"Answer {i}",
            reward=None,  # Will be set by calibration
            base_policy_logprob=-10.0 - i * 0.1,
            target_policy_logprobs={
                "improved": -9.0 - i * 0.1,
                "worse": -11.0 - i * 0.1,
            },
            metadata={
                "judge_score": float(np.clip(judge_score, 0, 1)),
                "oracle_label": (
                    float(np.clip(oracle_label, 0, 1))
                    if oracle_label is not None
                    else None
                ),
            },
        )
        samples.append(sample)

    return Dataset(
        samples=samples,
        target_policies=["improved", "worse"],
    )


@pytest.fixture
def dataset_for_dr() -> Dataset:
    """Create dataset suitable for DR estimation.

    Returns:
        Dataset with 25 samples. Folds are computed from prompt_id.
    """
    samples = []
    for i in range(25):
        sample = Sample(
            prompt_id=f"dr_test_{i}",
            prompt=f"DR question {i}",
            response=f"DR answer {i}",
            reward=0.6 + 0.2 * np.random.normal(),
            base_policy_logprob=-12.0 + np.random.normal(),
            target_policy_logprobs={
                "target": -10.0 + np.random.normal(),
            },
            metadata={
                "judge_score": 0.6 + 0.2 * np.random.normal(),
                # Note: cv_fold no longer stored - computed from prompt_id
            },
        )
        samples.append(sample)

    return Dataset(
        samples=samples,
        target_policies=["target"],
        metadata={"cross_fitted": True, "n_folds": 5},
    )


@pytest.fixture
def synthetic_fresh_draws() -> Dict[str, FreshDrawDataset]:
    """Create synthetic fresh draws for DR estimation.

    Returns:
        Dict mapping policy names to FreshDrawDataset objects.
    """
    fresh_draws = {}

    for policy in ["improved", "worse"]:
        samples = []
        for prompt_id in range(20):
            for draw_idx in range(3):  # 3 draws per prompt
                # Make "improved" policy have higher scores
                base_score = 0.7 if policy == "improved" else 0.3
                score = base_score + 0.1 * np.random.normal()

                sample = FreshDrawSample(
                    prompt_id=f"test_{prompt_id}",
                    target_policy=policy,
                    judge_score=float(np.clip(score, 0, 1)),
                    draw_idx=draw_idx,
                    fold_id=prompt_id % 5,
                    response=f"Response for {policy} draw {draw_idx}",
                )
                samples.append(sample)

        fresh_draws[policy] = FreshDrawDataset(
            samples=samples,
            target_policy=policy,
            draws_per_prompt=3,
        )

    return fresh_draws


# ============================================================================
# Test Data Factories
# ============================================================================


def create_test_samples(
    n_samples: int = 10,
    with_oracle: bool = False,
    oracle_coverage: float = 0.5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Create test sample data as raw dictionaries.

    Args:
        n_samples: Number of samples to create
        with_oracle: Whether to include oracle labels
        oracle_coverage: Fraction of samples with oracle labels
        seed: Random seed for reproducibility

    Returns:
        List of sample dictionaries suitable for PrecomputedSampler.
    """
    np.random.seed(seed)
    samples = []

    n_oracle = int(n_samples * oracle_coverage) if with_oracle else 0

    for i in range(n_samples):
        judge_score = 0.5 + 0.3 * np.sin(i / 3)

        metadata: Dict[str, Any] = {
            "judge_score": float(judge_score),
        }

        if with_oracle and i < n_oracle:
            oracle = judge_score + 0.05 * np.random.normal()
            metadata["oracle_label"] = float(np.clip(oracle, 0, 1))

        sample = {
            "prompt": f"Question {i}",
            "response": f"Answer {i}",
            "reward": float(judge_score),  # Use judge score as reward by default
            "base_policy_logprob": -10.0 - i * 0.1,
            "target_policy_logprobs": {
                "policy_a": -9.0 - i * 0.1,
                "policy_b": -11.0 - i * 0.1,
            },
            "metadata": metadata,
        }

        samples.append(sample)

    return samples


# ============================================================================
# Assertion Helpers
# ============================================================================


def assert_valid_estimation_result(
    result: EstimationResult,
    n_policies: int,
    check_diagnostics: bool = False,
) -> None:
    """Standard assertions for estimation results.

    Args:
        result: EstimationResult to validate
        n_policies: Expected number of policies
        check_diagnostics: Whether to check for diagnostics
    """
    # Check basic structure
    assert result is not None
    assert len(result.estimates) == n_policies
    assert len(result.standard_errors) == n_policies

    # Check values are reasonable
    assert not np.any(np.isnan(result.estimates)), "Estimates contain NaN"
    assert not np.any(np.isnan(result.standard_errors)), "Standard errors contain NaN"
    assert np.all(result.estimates >= 0), "Estimates should be non-negative"
    assert np.all(
        result.estimates <= 1
    ), "Estimates should be <= 1 for rewards in [0,1]"
    assert np.all(result.standard_errors >= 0), "Standard errors should be non-negative"

    # Check method is specified
    assert result.method is not None

    # Check diagnostics if requested
    if check_diagnostics:
        assert result.diagnostics is not None
        assert result.diagnostics.summary() is not None


def assert_weights_calibrated(
    weights: np.ndarray,
    target_mean: float = 1.0,
    tolerance: float = 0.01,
) -> None:
    """Assert that importance weights are properly calibrated.

    Args:
        weights: Array of importance weights
        target_mean: Expected mean (usually 1.0 for Hajek weights)
        tolerance: Tolerance for mean comparison
    """
    assert weights is not None
    assert len(weights) > 0
    assert not np.any(np.isnan(weights)), "Weights contain NaN"
    assert np.all(weights >= 0), "Weights should be non-negative"

    # Check mean is close to target
    mean_weight = np.mean(weights)
    assert (
        abs(mean_weight - target_mean) < tolerance
    ), f"Weight mean {mean_weight:.3f} not close to target {target_mean}"

    # Check not all weights are identical (unless n=1)
    if len(weights) > 1:
        assert not np.allclose(
            weights, weights[0]
        ), "All weights are identical, suggesting no calibration"


def assert_dataset_valid(dataset: Dataset) -> None:
    """Assert that a dataset is valid for CJE analysis.

    Args:
        dataset: Dataset to validate
    """
    assert dataset is not None
    assert len(dataset.samples) > 0, "Dataset has no samples"
    assert dataset.target_policies is not None
    assert len(dataset.target_policies) > 0, "Dataset has no target policies"

    # Check all samples have required fields
    for sample in dataset.samples:
        assert sample.prompt is not None
        assert sample.response is not None
        assert sample.base_policy_logprob is not None
        assert sample.target_policy_logprobs is not None

        # Check target policies match
        for policy in dataset.target_policies:
            assert (
                policy in sample.target_policy_logprobs
            ), f"Policy {policy} missing from sample target_policy_logprobs"


# ============================================================================
# Test Markers and Configuration
# ============================================================================


def pytest_configure(config: Any) -> None:
    """Register custom markers for test categorization."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "fast: marks tests as fast (< 0.1s, use synthetic data)"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, isolated)"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests using arena data"
    )
    config.addinivalue_line(
        "markers", "requires_api: marks tests that require API credentials"
    )
    config.addinivalue_line(
        "markers", "requires_fresh_draws: marks tests that need fresh draw files"
    )
    config.addinivalue_line(
        "markers", "uses_arena_sample: marks tests using real arena data"
    )
    config.addinivalue_line(
        "markers", "deprecated: marks tests superseded by E2E tests"
    )
