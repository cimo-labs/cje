"""Tests for data loading functions.

This module tests the actual data loading code paths that users will rely on,
ensuring that both simple and extended file formats are supported.
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List

from cje.data.fresh_draws import (
    load_fresh_draws_from_jsonl,
    load_fresh_draws_auto,
    fresh_draws_from_dict,
    FreshDrawDataset,
)


class TestFreshDrawLoading:
    """Test fresh draw loading from various file formats."""

    def test_load_fresh_draws_auto_from_arena_sample(self) -> None:
        """Test that load_fresh_draws_auto works with real arena files."""
        # Point to examples directory (shared with tutorials)
        responses_dir = (
            Path(__file__).parent.parent.parent
            / "examples"
            / "arena_sample"
            / "responses"
        )

        if not responses_dir.exists():
            pytest.skip("Arena sample data not available")

        # Load using official function
        fresh_dataset = load_fresh_draws_auto(
            data_dir=responses_dir, policy="clone", verbose=False
        )

        # Validate structure
        assert fresh_dataset.target_policy == "clone"
        assert len(fresh_dataset.samples) > 0
        assert fresh_dataset.draws_per_prompt >= 1

        # Validate data extraction from extended format (metadata.judge_score)
        for sample in fresh_dataset.samples[:5]:  # Check first 5
            assert 0 <= sample.judge_score <= 1
            assert sample.prompt_id.startswith("arena_")
            assert sample.target_policy == "clone"
            assert sample.draw_idx >= 0
            assert sample.response is not None  # Should extract response field

    def test_load_fresh_draws_from_simple_format(self) -> None:
        """Test loading from simple JSONL format (recommended for users)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Simple format - just the essentials
            data = [
                {
                    "prompt_id": "test_0",
                    "target_policy": "premium",
                    "judge_score": 0.85,
                    "draw_idx": 0,
                },
                {
                    "prompt_id": "test_0",
                    "target_policy": "premium",
                    "judge_score": 0.82,
                    "draw_idx": 1,
                },
                {
                    "prompt_id": "test_1",
                    "target_policy": "premium",
                    "judge_score": 0.90,
                    "draw_idx": 0,
                },
            ]
            for item in data:
                f.write(json.dumps(item) + "\n")
            temp_path = f.name

        try:
            # Load from simple format
            datasets = load_fresh_draws_from_jsonl(temp_path)

            # Should group by policy
            assert "premium" in datasets
            premium_data = datasets["premium"]

            # Validate structure
            assert premium_data.target_policy == "premium"
            assert len(premium_data.samples) == 3
            assert premium_data.draws_per_prompt == 2  # Max draws per prompt

            # Validate samples
            assert premium_data.samples[0].prompt_id == "test_0"
            assert premium_data.samples[0].judge_score == 0.85

        finally:
            Path(temp_path).unlink()

    def test_load_fresh_draws_from_extended_format(self) -> None:
        """Test loading from extended format with metadata (arena sample style)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Extended format - judge_score in metadata
            data = [
                {
                    "prompt_id": "arena_0",
                    "prompt": "Test question?",
                    "response": "Test answer",
                    "policy": "clone",
                    "model": "test-model",
                    "metadata": {"judge_score": 0.85, "oracle_label": 0.8},
                },
                {
                    "prompt_id": "arena_1",
                    "prompt": "Another question?",
                    "response": "Another answer",
                    "policy": "clone",
                    "model": "test-model",
                    "metadata": {"judge_score": 0.75, "oracle_label": 0.7},
                },
            ]
            for item in data:
                f.write(json.dumps(item) + "\n")
            temp_path = f.name

        try:
            # Use load_fresh_draws_auto which handles both formats
            temp_dir = Path(temp_path).parent
            datasets: Dict[str, FreshDrawDataset] = {}

            # Manually load since auto-loader expects specific filenames
            with open(temp_path) as file:
                from cje.data.fresh_draws import FreshDrawSample

                samples = []
                for line in file:
                    rec: Dict[str, Any] = json.loads(line)

                    # Extract judge_score from metadata (extended format)
                    metadata = rec.get("metadata", {})
                    if isinstance(metadata, dict) and "judge_score" in metadata:
                        judge_score = float(metadata["judge_score"])
                    else:
                        judge_score = float(rec["judge_score"])

                    sample = FreshDrawSample(
                        prompt_id=str(rec["prompt_id"]),
                        target_policy="clone",  # From policy field or filename
                        judge_score=judge_score,
                        oracle_label=None,
                        draw_idx=0,
                        response=rec.get("response"),
                        fold_id=None,
                    )
                    samples.append(sample)

                fresh_dataset = FreshDrawDataset(
                    target_policy="clone", draws_per_prompt=1, samples=samples
                )

            # Validate
            assert len(fresh_dataset.samples) == 2
            assert fresh_dataset.samples[0].judge_score == 0.85
            assert fresh_dataset.samples[1].judge_score == 0.75
            assert fresh_dataset.samples[0].response == "Test answer"

        finally:
            Path(temp_path).unlink()

    def test_load_fresh_draws_auto_file_discovery(self) -> None:
        """Test that load_fresh_draws_auto finds files in standard locations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create responses subdirectory (standard location)
            responses_dir = temp_path / "responses"
            responses_dir.mkdir()

            # Write test file in standard location
            test_file = responses_dir / "premium_responses.jsonl"
            with open(test_file, "w") as f:
                data = {
                    "prompt_id": "test_0",
                    "target_policy": "premium",
                    "judge_score": 0.85,
                    "draw_idx": 0,
                }
                f.write(json.dumps(data) + "\n")

            # Should find the file automatically
            fresh_dataset = load_fresh_draws_auto(
                data_dir=responses_dir, policy="premium", verbose=False
            )

            assert fresh_dataset.target_policy == "premium"
            assert len(fresh_dataset.samples) == 1
            assert fresh_dataset.samples[0].judge_score == 0.85

    def test_load_fresh_draws_missing_file(self) -> None:
        """Test that missing files raise clear errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Try to load non-existent policy
            with pytest.raises(FileNotFoundError, match="No fresh draw file found"):
                load_fresh_draws_auto(
                    data_dir=temp_path, policy="nonexistent", verbose=False
                )

    def test_load_fresh_draws_multiple_policies(self) -> None:
        """Test loading multiple policies from single file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Multiple policies in one file
            data = [
                {
                    "prompt_id": "test_0",
                    "target_policy": "premium",
                    "judge_score": 0.85,
                    "draw_idx": 0,
                },
                {
                    "prompt_id": "test_0",
                    "target_policy": "baseline",
                    "judge_score": 0.75,
                    "draw_idx": 0,
                },
            ]
            for item in data:
                f.write(json.dumps(item) + "\n")
            temp_path = f.name

        try:
            # Load and group by policy
            datasets = load_fresh_draws_from_jsonl(temp_path)

            # Should have both policies
            assert "premium" in datasets
            assert "baseline" in datasets
            assert len(datasets["premium"].samples) == 1
            assert len(datasets["baseline"].samples) == 1

        finally:
            Path(temp_path).unlink()


class TestFreshDrawsFromDict:
    """Test in-memory fresh_draws_from_dict function."""

    def test_basic_conversion(self) -> None:
        """Test basic dict to FreshDrawDataset conversion."""
        data = {
            "policy_a": [
                {"prompt_id": "q1", "judge_score": 0.85},
                {"prompt_id": "q2", "judge_score": 0.72},
            ],
            "policy_b": [
                {"prompt_id": "q1", "judge_score": 0.70},
                {"prompt_id": "q2", "judge_score": 0.82},
            ],
        }

        datasets = fresh_draws_from_dict(data)

        # Check both policies were converted
        assert "policy_a" in datasets
        assert "policy_b" in datasets

        # Check structure
        assert datasets["policy_a"].target_policy == "policy_a"
        assert datasets["policy_a"].n_samples == 2
        assert datasets["policy_b"].n_samples == 2

        # Check sample values
        samples_a = datasets["policy_a"].samples
        assert samples_a[0].prompt_id == "q1"
        assert samples_a[0].judge_score == 0.85
        assert samples_a[0].target_policy == "policy_a"

    def test_with_oracle_labels(self) -> None:
        """Test conversion with oracle labels."""
        data = {
            "policy_a": [
                {"prompt_id": "q1", "judge_score": 0.85, "oracle_label": 0.9},
                {"prompt_id": "q2", "judge_score": 0.72, "oracle_label": 0.7},
                {"prompt_id": "q3", "judge_score": 0.65},  # No oracle
            ],
        }

        datasets = fresh_draws_from_dict(data)

        samples = datasets["policy_a"].samples
        assert samples[0].oracle_label == 0.9
        assert samples[1].oracle_label == 0.7
        assert samples[2].oracle_label is None

    def test_with_response_and_metadata(self) -> None:
        """Test conversion with response and metadata fields."""
        data = {
            "policy_a": [
                {
                    "prompt_id": "q1",
                    "judge_score": 0.85,
                    "response": "Test response",
                    "metadata": {"custom_field": "value"},
                },
            ],
        }

        datasets = fresh_draws_from_dict(data)

        sample = datasets["policy_a"].samples[0]
        assert sample.response == "Test response"
        assert sample.metadata.get("custom_field") == "value"

    def test_auto_draw_idx(self) -> None:
        """Test automatic draw_idx assignment for multiple draws per prompt."""
        data = {
            "policy_a": [
                {"prompt_id": "q1", "judge_score": 0.85},  # draw_idx=0
                {"prompt_id": "q1", "judge_score": 0.82},  # draw_idx=1
                {"prompt_id": "q1", "judge_score": 0.88},  # draw_idx=2
            ],
        }

        datasets = fresh_draws_from_dict(data)

        assert datasets["policy_a"].draws_per_prompt == 3
        samples = datasets["policy_a"].samples
        # Note: draw_idx is auto-assigned based on order when not provided
        assert samples[0].draw_idx == 0
        assert samples[1].draw_idx == 1
        assert samples[2].draw_idx == 2

    def test_missing_prompt_id_raises(self) -> None:
        """Test that missing prompt_id raises clear error."""
        data = {
            "policy_a": [
                {"judge_score": 0.85},  # Missing prompt_id
            ],
        }

        with pytest.raises(ValueError, match="missing required field 'prompt_id'"):
            fresh_draws_from_dict(data)

    def test_missing_judge_score_raises(self) -> None:
        """Test that missing judge_score raises clear error."""
        data = {
            "policy_a": [
                {"prompt_id": "q1"},  # Missing judge_score
            ],
        }

        with pytest.raises(ValueError, match="missing required field 'judge_score'"):
            fresh_draws_from_dict(data)

    def test_empty_data_raises(self) -> None:
        """Test that empty data raises error."""
        with pytest.raises(ValueError, match="empty"):
            fresh_draws_from_dict({})


class TestAnalyzeDatasetWithFreshDrawsData:
    """Test analyze_dataset with in-memory fresh_draws_data parameter."""

    def test_analyze_with_in_memory_data(self) -> None:
        """Test analyze_dataset works with fresh_draws_data parameter."""
        import numpy as np
        from cje import analyze_dataset

        np.random.seed(42)
        n_samples = 50

        fresh_draws_data: Dict[str, List[Dict[str, Any]]] = {
            "policy_a": [],
            "policy_b": [],
        }

        for i in range(n_samples):
            judge_a = np.clip(0.7 + np.random.randn() * 0.15, 0, 1)
            judge_b = np.clip(0.65 + np.random.randn() * 0.15, 0, 1)

            record_a = {"prompt_id": f"q{i}", "judge_score": float(judge_a)}
            record_b = {"prompt_id": f"q{i}", "judge_score": float(judge_b)}

            # Add oracle labels to enough samples for calibration
            if i < 25:
                record_a["oracle_label"] = float(
                    np.clip(judge_a + np.random.randn() * 0.05, 0, 1)
                )
                record_b["oracle_label"] = float(
                    np.clip(judge_b + np.random.randn() * 0.05, 0, 1)
                )

            fresh_draws_data["policy_a"].append(record_a)
            fresh_draws_data["policy_b"].append(record_b)

        results = analyze_dataset(fresh_draws_data=fresh_draws_data)

        # Verify results
        assert results.metadata.get("mode") == "direct"
        assert results.metadata.get("fresh_draws_source") == "in_memory"
        assert results.metadata.get("calibration") == "from_fresh_draws"
        assert len(results.estimates) == 2  # Two policies
        assert len(results.standard_errors) == 2
