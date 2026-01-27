"""Tests for auto-normalization of arbitrary label scales.

This module tests the auto-normalization feature that allows users to provide
data in any bounded scale (0-100, Likert 1-5, etc.) without manual preprocessing.

Test categories:
- TestScaleInfo: Unit tests for ScaleInfo utility class
- TestDetectRange: Unit tests for range detection function
- TestFreshDrawsFromDictNormalization: Integration tests for data loading
- TestAnalyzeDatasetNormalization: E2E tests for full analysis workflow
"""

import numpy as np
import pytest

from cje.data.normalization import ScaleInfo, detect_range, detect_and_normalize
from cje.data.fresh_draws import fresh_draws_from_dict, NormalizationInfo
from cje import analyze_dataset


class TestScaleInfo:
    """Tests for the ScaleInfo class."""

    def test_normalize_simple(self) -> None:
        """Test basic normalization from [0, 100] to [0, 1]."""
        scale = ScaleInfo(min_val=0, max_val=100)
        assert scale.normalize(0) == 0.0
        assert scale.normalize(50) == 0.5
        assert scale.normalize(100) == 1.0
        assert scale.normalize(75) == 0.75

    def test_inverse_simple(self) -> None:
        """Test inverse transform from [0, 1] to [0, 100]."""
        scale = ScaleInfo(min_val=0, max_val=100)
        assert scale.inverse(0) == 0.0
        assert scale.inverse(0.5) == 50.0
        assert scale.inverse(1.0) == 100.0
        assert scale.inverse(0.75) == 75.0

    def test_roundtrip(self) -> None:
        """Test that normalize followed by inverse returns original value."""
        scale = ScaleInfo(min_val=20, max_val=80)
        for val in [20, 35, 50, 65, 80]:
            normalized = scale.normalize(val)
            recovered = scale.inverse(normalized)
            assert abs(recovered - val) < 1e-10

    def test_degenerate_case(self) -> None:
        """Test handling of all-same values (degenerate case)."""
        scale = ScaleInfo(min_val=50, max_val=50)
        assert scale.normalize(50) == 0.5  # Returns midpoint

    def test_likert_scale(self) -> None:
        """Test normalization of Likert 1-5 scale."""
        scale = ScaleInfo(min_val=1, max_val=5)
        assert scale.normalize(1) == 0.0
        assert scale.normalize(3) == 0.5
        assert scale.normalize(5) == 1.0

    def test_is_identity(self) -> None:
        """Test detection of identity (no-op) scale."""
        assert ScaleInfo(min_val=0, max_val=1).is_identity()
        assert not ScaleInfo(min_val=0, max_val=100).is_identity()
        assert not ScaleInfo(min_val=0.1, max_val=0.9).is_identity()

    def test_array_operations(self) -> None:
        """Test array normalize and inverse."""
        scale = ScaleInfo(min_val=0, max_val=100)
        arr = np.array([0, 25, 50, 75, 100])
        normalized = scale.normalize_array(arr)
        np.testing.assert_array_almost_equal(normalized, [0, 0.25, 0.5, 0.75, 1.0])

        inversed = scale.inverse_array(normalized)
        np.testing.assert_array_almost_equal(inversed, arr)


class TestDetectRange:
    """Tests for the detect_range function."""

    def test_detect_0_100_range(self) -> None:
        """Test detection of 0-100 scale."""
        values = np.array([0, 25, 50, 75, 100])
        scale = detect_range(values, "test")
        assert scale.min_val == 0.0
        assert scale.max_val == 100.0

    def test_detect_partial_range(self) -> None:
        """Test detection when values don't span full theoretical range."""
        values = np.array([30, 50, 70])
        scale = detect_range(values, "test")
        assert scale.min_val == 30.0
        assert scale.max_val == 70.0

    def test_handles_nan(self) -> None:
        """Test that NaN values are ignored in range detection."""
        values = np.array([0, np.nan, 50, np.nan, 100])
        scale = detect_range(values, "test")
        assert scale.min_val == 0.0
        assert scale.max_val == 100.0

    def test_handles_large_values(self) -> None:
        """Test that large values are handled by scaling to detected max."""
        values = np.array([0, 5000, 10000])
        scale = detect_range(values, "test")
        assert scale.min_val == 0.0
        assert scale.max_val == 10000.0
        # Normalization should work
        assert scale.normalize(5000) == 0.5

    def test_handles_negative_values(self) -> None:
        """Test that negative values are handled correctly."""
        values = np.array([-5000, 0, 5000])
        scale = detect_range(values, "test")
        assert scale.min_val == -5000.0
        assert scale.max_val == 5000.0
        assert scale.normalize(0) == 0.5

    def test_empty_values_error(self) -> None:
        """Test that empty values raise an error."""
        values = np.array([])
        with pytest.raises(ValueError, match="No valid values"):
            detect_range(values, "test")

    def test_all_nan_error(self) -> None:
        """Test that all-NaN values raise an error."""
        values = np.array([np.nan, np.nan])
        with pytest.raises(ValueError, match="No valid values"):
            detect_range(values, "test")


class TestFreshDrawsFromDictNormalization:
    """Tests for fresh_draws_from_dict with auto-normalization."""

    def test_normalizes_0_100_data(self) -> None:
        """Test that 0-100 scale data is normalized to [0,1]."""
        data = {
            "policy_a": [
                {"prompt_id": "1", "judge_score": 80, "oracle_label": 90},
                {"prompt_id": "2", "judge_score": 60, "oracle_label": 70},
            ]
        }
        datasets, norm_info = fresh_draws_from_dict(data)

        # Check normalization was applied
        assert norm_info is not None
        assert norm_info.judge_score_scale.min_val == 60.0
        assert norm_info.judge_score_scale.max_val == 80.0
        assert norm_info.oracle_label_scale.min_val == 70.0
        assert norm_info.oracle_label_scale.max_val == 90.0

        # Check samples are normalized
        samples = datasets["policy_a"].samples
        # 80 normalized in range [60, 80] = 1.0
        assert samples[0].judge_score == 1.0
        # 60 normalized in range [60, 80] = 0.0
        assert samples[1].judge_score == 0.0

    def test_identity_for_0_1_data(self) -> None:
        """Test that [0,1] data is not modified."""
        data = {
            "policy_a": [
                {"prompt_id": "1", "judge_score": 0.0, "oracle_label": 0.0},
                {"prompt_id": "2", "judge_score": 0.5, "oracle_label": 0.6},
                {"prompt_id": "3", "judge_score": 1.0, "oracle_label": 1.0},
            ]
        }
        datasets, norm_info = fresh_draws_from_dict(data)

        # No normalization needed for [0,1] data
        assert norm_info is None

        # Check samples are unchanged
        samples = datasets["policy_a"].samples
        assert samples[0].judge_score == 0.0
        assert samples[1].judge_score == 0.5
        assert samples[2].judge_score == 1.0

    def test_auto_normalize_false(self) -> None:
        """Test that auto_normalize=False skips normalization."""
        data = {
            "policy_a": [
                {"prompt_id": "1", "judge_score": 0.8, "oracle_label": 0.9},
            ]
        }
        datasets, norm_info = fresh_draws_from_dict(data, auto_normalize=False)
        assert norm_info is None

    def test_normalization_info_to_dict(self) -> None:
        """Test that NormalizationInfo.to_dict() works correctly."""
        data = {
            "policy_a": [
                {"prompt_id": "1", "judge_score": 0, "oracle_label": 0},
                {"prompt_id": "2", "judge_score": 100, "oracle_label": 100},
            ]
        }
        _, norm_info = fresh_draws_from_dict(data)

        d = norm_info.to_dict()
        assert d["judge_score"]["original_range"] == (0, 100)
        assert d["oracle_label"]["original_range"] == (0, 100)
        assert d["results_scale"] == "oracle_original"


@pytest.mark.integration
@pytest.mark.uses_arena_sample
class TestAnalyzeDatasetNormalization:
    """E2E tests for analyze_dataset with auto-normalization using real arena data.

    These tests verify that users can provide data in various scales
    (0-100, Likert 1-5, etc.) and get results back in the original scale.

    Strategy: Load real arena fresh draws (in [0,1]) and scale them to different
    ranges to test normalization with realistic data distributions.
    """

    @staticmethod
    def _load_arena_fresh_draws_as_dict(scale: float = 1.0) -> dict:
        """Load arena fresh draws and convert to dict format with optional scaling.

        Args:
            scale: Multiplier for judge_score and oracle_label (e.g., 100 for 0-100)

        Returns:
            Dict suitable for analyze_dataset(fresh_draws_data=...)
        """
        from pathlib import Path
        import json

        responses_dir = (
            Path(__file__).parent.parent.parent
            / "examples"
            / "arena_sample"
            / "fresh_draws"
        )

        if not responses_dir.exists():
            pytest.skip(f"Fresh draws not found at {responses_dir}")

        data = {}
        for policy_file in responses_dir.glob("*_responses.jsonl"):
            policy_name = policy_file.stem.replace("_responses", "")
            records = []

            with open(policy_file) as f:
                for line in f:
                    record = json.loads(line)
                    # Scale the values
                    scaled_record = {
                        "prompt_id": record["prompt_id"],
                        "judge_score": record["judge_score"] * scale,
                    }
                    if record.get("oracle_label") is not None:
                        scaled_record["oracle_label"] = record["oracle_label"] * scale
                    records.append(scaled_record)

            data[policy_name] = records

        return data

    def test_0_100_arena_data_returns_0_100_results(self) -> None:
        """Test that arena data scaled to 0-100 produces results in 0-100 scale."""
        # Load real arena data scaled to 0-100
        data = self._load_arena_fresh_draws_as_dict(scale=100.0)

        results = analyze_dataset(fresh_draws_data=data)

        # Results should be in 0-100 scale
        for estimate in results.estimates:
            assert 0 <= estimate <= 100, f"Expected estimate in [0,100], got {estimate}"

        # Check normalization metadata is present
        assert "normalization" in results.metadata
        norm_meta = results.metadata["normalization"]
        assert "oracle_label" in norm_meta
        assert norm_meta["results_scale"] == "oracle_original"

    def test_0_1_arena_data_unchanged(self) -> None:
        """Test that arena data in [0,1] produces results in [0,1] scale."""
        # Load real arena data without scaling (already in [0,1])
        data = self._load_arena_fresh_draws_as_dict(scale=1.0)

        results = analyze_dataset(fresh_draws_data=data)

        # Results should be in [0,1] range
        for estimate in results.estimates:
            assert 0 <= estimate <= 1, f"Expected estimate in [0,1], got {estimate}"

        # Normalization metadata should not be present (identity transform)
        assert "normalization" not in results.metadata

    def test_likert_arena_data(self) -> None:
        """Test that arena data scaled to Likert 1-5 produces results in 1-5 scale."""
        # Load real arena data and scale to 1-5 range
        # Original is [0,1], so: 1 + value * 4 gives [1,5]
        from pathlib import Path
        import json

        responses_dir = (
            Path(__file__).parent.parent.parent
            / "examples"
            / "arena_sample"
            / "fresh_draws"
        )

        if not responses_dir.exists():
            pytest.skip(f"Fresh draws not found at {responses_dir}")

        data = {}
        for policy_file in responses_dir.glob("*_responses.jsonl"):
            policy_name = policy_file.stem.replace("_responses", "")
            records = []

            with open(policy_file) as f:
                for line in f:
                    record = json.loads(line)
                    # Scale [0,1] to [1,5]: 1 + value * 4
                    scaled_record = {
                        "prompt_id": record["prompt_id"],
                        "judge_score": 1 + record["judge_score"] * 4,
                    }
                    if record.get("oracle_label") is not None:
                        scaled_record["oracle_label"] = 1 + record["oracle_label"] * 4
                    records.append(scaled_record)

            data[policy_name] = records

        results = analyze_dataset(fresh_draws_data=data)

        # Results should be in Likert scale range
        for estimate in results.estimates:
            assert 1 <= estimate <= 5, f"Expected estimate in [1,5], got {estimate}"

        # Check normalization metadata
        assert "normalization" in results.metadata

    def test_large_scale_arena_data(self) -> None:
        """Test that arena data scaled to large range (0-10000) works correctly."""
        # Load real arena data scaled to 0-10000
        data = self._load_arena_fresh_draws_as_dict(scale=10000.0)

        results = analyze_dataset(fresh_draws_data=data)

        # Results should be in the scaled range
        for estimate in results.estimates:
            assert (
                0 <= estimate <= 10000
            ), f"Expected estimate in [0,10000], got {estimate}"

        # Check normalization metadata
        assert "normalization" in results.metadata

    def test_multiple_policies_same_scale(self) -> None:
        """Test that multiple policies share the same normalization with real data."""
        # Load real arena data scaled to 0-100
        data = self._load_arena_fresh_draws_as_dict(scale=100.0)

        # Should have multiple policies from arena
        assert len(data) >= 2, "Need at least 2 policies for this test"

        results = analyze_dataset(fresh_draws_data=data)

        # All estimates should be in the 0-100 scale
        for estimate in results.estimates:
            assert 0 <= estimate <= 100, f"Expected estimate in [0,100], got {estimate}"
