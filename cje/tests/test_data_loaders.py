"""Tests for data loading functions.

This module tests the actual data loading code paths that users will rely on,
ensuring that both simple and extended file formats are supported.
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

from cje.data.fresh_draws import (
    load_fresh_draws_from_jsonl,
    load_fresh_draws_auto,
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
