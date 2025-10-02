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
    """Load real arena sample dataset once per session (1000 samples).

    This is real data from Arena with judge scores and oracle labels.
    Use this for integration tests and realistic scenarios.
    Session-scoped for performance.

    Data location: examples/arena_sample/logged_data.jsonl
    """
    # Point to examples directory (shared with tutorials)
    data_path = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "arena_sample"
        / "logged_data.jsonl"
    )
    if not data_path.exists():
        pytest.skip(f"Arena sample not found at {data_path}")
    return load_dataset_from_jsonl(str(data_path))


@pytest.fixture
def arena_sample() -> Dataset:
    """Load real arena sample dataset (1000 samples).

    Function-scoped version for tests that modify the dataset.

    Data location: examples/arena_sample/logged_data.jsonl
    """
    # Point to examples directory (shared with tutorials)
    data_path = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "arena_sample"
        / "logged_data.jsonl"
    )
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
        i for i, s in enumerate(dataset.samples) if s.oracle_label is not None
    ]

    if len(samples_with_oracle) > 2:
        random.seed(42)
        # Keep only 50% of oracle labels
        n_keep = max(2, len(samples_with_oracle) // 2)
        keep_indices = set(random.sample(samples_with_oracle, n_keep))

        # Create new samples with masked oracle labels
        new_samples = []
        for i, sample in enumerate(dataset.samples):
            if i not in keep_indices and sample.oracle_label is not None:
                # Remove oracle label for this sample
                new_samples.append(sample.model_copy(update={"oracle_label": None}))
            else:
                new_samples.append(sample)
        dataset.samples = new_samples

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
    """Load real fresh draws from arena sample using the official loader.

    Returns dict mapping policy names to FreshDrawDataset objects.
    Policies: clone, parallel_universe_prompt

    This uses load_fresh_draws_auto() to test the actual production code path
    that users will rely on.

    Data location: examples/arena_sample/fresh_draws/
    """
    from cje.data.fresh_draws import load_fresh_draws_auto

    # Point to examples directory (shared with tutorials)
    responses_dir = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "arena_sample"
        / "fresh_draws"
    )

    if not responses_dir.exists():
        pytest.skip(f"Fresh draws not found at {responses_dir}")

    # Use the official loader for each policy - this is what users will do
    fresh_draws = {}
    for policy_file in responses_dir.glob("*_responses.jsonl"):
        policy_name = policy_file.stem.replace("_responses", "")

        try:
            fresh_dataset = load_fresh_draws_auto(
                data_dir=responses_dir, policy=policy_name, verbose=False
            )
            fresh_draws[policy_name] = fresh_dataset
        except FileNotFoundError:
            # Policy file exists but wasn't found by auto-loader (shouldn't happen)
            pytest.skip(f"Could not load fresh draws for {policy_name}")

    return fresh_draws


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


# ============================================================================
# Thread Cleanup (prevents pytest hanging after tests complete)
# ============================================================================


# Thread cleanup not needed - hypothesis plugin (the real culprit) has been removed
