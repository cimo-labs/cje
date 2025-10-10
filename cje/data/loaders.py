"""Data loading utilities following SOLID principles.

This module separates data loading concerns from the Dataset model,
following the Single Responsibility Principle.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

from .models import Dataset, Sample
from .fresh_draws import FreshDrawSample, FreshDrawDataset

logger = logging.getLogger(__name__)


class DataSource(Protocol):
    """Protocol for data sources."""

    def load(self) -> List[Dict[str, Any]]:
        """Load raw data as list of dictionaries."""
        ...


class JsonlDataSource:
    """Load data from JSONL files."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(self.file_path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data


class InMemoryDataSource:
    """Load data from in-memory list."""

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def load(self) -> List[Dict[str, Any]]:
        """Return the in-memory data."""
        return self.data


class DatasetLoader:
    """Loads and converts raw data into typed Dataset objects.

    Follows Single Responsibility Principle - only handles data loading and conversion.
    """

    def __init__(
        self,
        base_policy_field: str = "base_policy_logprob",
        target_policy_logprobs_field: str = "target_policy_logprobs",
        prompt_field: str = "prompt",
        response_field: str = "response",
        reward_field: str = "reward",
    ):
        self.base_policy_field = base_policy_field
        self.target_policy_logprobs_field = target_policy_logprobs_field
        self.prompt_field = prompt_field
        self.response_field = response_field
        self.reward_field = reward_field

    def load_from_source(
        self, source: DataSource, target_policies: Optional[List[str]] = None
    ) -> Dataset:
        """Load Dataset from a data source.

        Args:
            source: Data source to load from
            target_policies: List of target policy names. If None, auto-detected.

        Returns:
            Dataset instance
        """
        data = source.load()
        return self._convert_raw_data(data, target_policies)

    def _convert_raw_data(
        self, data: List[Dict[str, Any]], target_policies: Optional[List[str]] = None
    ) -> Dataset:
        """Convert raw data to Dataset."""
        # Auto-detect target policies if needed
        if target_policies is None:
            target_policies = self._detect_target_policies(data)

        # Convert raw data to samples
        samples = []
        for idx, record in enumerate(data):
            try:
                sample = self._convert_record_to_sample(record, idx)
                samples.append(sample)
            except (KeyError, ValueError) as e:
                # Skip invalid records
                print(f"Skipping invalid record: {e}")
                continue

        if not samples:
            raise ValueError("No valid samples could be created from data")

        return Dataset(
            samples=samples,
            target_policies=target_policies,
            metadata={
                "source": "loader",
                "base_policy_field": self.base_policy_field,
                "target_policy_logprobs_field": self.target_policy_logprobs_field,
            },
        )

    def _detect_target_policies(self, data: List[Dict[str, Any]]) -> List[str]:
        """Auto-detect target policies from data."""
        policies = set()
        for record in data:
            if self.target_policy_logprobs_field in record:
                policies.update(record[self.target_policy_logprobs_field].keys())
        return sorted(list(policies))

    def _convert_record_to_sample(self, record: Dict[str, Any], idx: int = 0) -> Sample:
        """Convert a single record to a Sample.

        Args:
            record: Raw data record
            idx: Index in dataset (used as fallback if prompt is also missing)
        """
        # Get prompt_id - check top-level first, then metadata, then auto-generate
        prompt_id = record.get("prompt_id") or record.get("metadata", {}).get(
            "prompt_id"
        )
        if prompt_id is None:
            # Auto-generate from prompt hash for consistency across datasets
            # This ensures fresh draws will map to the same prompt_id
            prompt = record.get(self.prompt_field, "")
            if prompt:
                import hashlib

                # Use first 12 chars of SHA256 for readable but unique ID
                prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
                prompt_id = f"prompt_{prompt_hash}"
            else:
                # Fallback to index if no prompt either
                prompt_id = f"sample_{idx:06d}"
                logger.warning(
                    f"Record {idx} missing both 'prompt_id' and 'prompt'. "
                    f"Using index-based ID '{prompt_id}'. This is fragile - "
                    f"consider adding explicit prompt_id or prompt text for stability."
                )

        # Extract reward if present (handle nested format)
        reward = None
        if self.reward_field in record:
            reward = record[self.reward_field]
            if isinstance(reward, dict):
                reward = reward.get("mean", reward.get("value"))
            if reward is not None:
                reward = float(reward)

        # Get base log prob
        base_logprob = record.get(self.base_policy_field)

        # Get target log probs
        target_logprobs = record.get(self.target_policy_logprobs_field, {})

        # Extract judge_score and oracle_label - prioritize top-level, fallback to metadata
        judge_score = None
        oracle_label = None
        metadata_dict = record.get("metadata", {})

        # Judge score: check top-level first, then metadata
        if "judge_score" in record and record["judge_score"] is not None:
            judge_score = float(record["judge_score"])
        elif (
            "judge_score" in metadata_dict and metadata_dict["judge_score"] is not None
        ):
            judge_score = float(metadata_dict["judge_score"])

        # Oracle label: check top-level first, then metadata
        if "oracle_label" in record and record["oracle_label"] is not None:
            oracle_label = float(record["oracle_label"])
        elif (
            "oracle_label" in metadata_dict
            and metadata_dict["oracle_label"] is not None
        ):
            oracle_label = float(metadata_dict["oracle_label"])

        # Collect all other fields into metadata (excluding judge_score/oracle_label now)
        metadata = {}
        core_fields = {
            "prompt_id",
            self.prompt_field,
            self.response_field,
            self.reward_field,
            self.base_policy_field,
            self.target_policy_logprobs_field,
            "judge_score",  # Now a core field
            "oracle_label",  # Now a core field
            "metadata",
        }

        # Add non-core fields from top level
        for key, value in record.items():
            if key not in core_fields:
                metadata[key] = value

        # Add metadata dict fields (excluding judge_score/oracle_label which are now top-level)
        for key, value in metadata_dict.items():
            if key not in {"judge_score", "oracle_label"}:
                metadata[key] = value

        # Create Sample object with judge_score and oracle_label as top-level fields
        return Sample(
            prompt_id=prompt_id,
            prompt=record[self.prompt_field],
            response=record[self.response_field],
            reward=reward,
            base_policy_logprob=base_logprob,
            target_policy_logprobs=target_logprobs,
            judge_score=judge_score,
            oracle_label=oracle_label,
            metadata=metadata,
        )


class FreshDrawLoader:
    """Loader for fresh draw samples used in DR estimation."""

    @staticmethod
    def load_from_jsonl(path: str) -> Dict[str, FreshDrawDataset]:
        """Load fresh draws from JSONL file, grouped by policy.

        Expected JSONL format:
        {"prompt_id": "0", "target_policy": "premium", "judge_score": 0.85, "draw_idx": 0}
        {"prompt_id": "0", "target_policy": "premium", "judge_score": 0.82, "draw_idx": 1}
        {"prompt_id": "1", "target_policy": "premium", "judge_score": 0.90, "draw_idx": 0}

        Args:
            path: Path to JSONL file containing fresh draws

        Returns:
            Dict mapping policy names to FreshDrawDataset objects
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Fresh draws file not found: {path_obj}")

        # Group samples by policy
        samples_by_policy: Dict[str, List[FreshDrawSample]] = defaultdict(list)

        with open(path_obj, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)

                    # Create FreshDrawSample
                    sample = FreshDrawSample(
                        prompt_id=data["prompt_id"],
                        target_policy=data["target_policy"],
                        judge_score=data["judge_score"],
                        oracle_label=data.get("oracle_label"),  # Optional
                        response=data.get("response"),  # Optional
                        draw_idx=data.get(
                            "draw_idx", 0
                        ),  # Default to 0 if not provided
                        fold_id=data.get("fold_id"),  # Optional
                    )

                    samples_by_policy[sample.target_policy].append(sample)

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid line {line_num}: {e}")

        # Create FreshDrawDataset for each policy
        datasets = {}
        for policy, samples in samples_by_policy.items():
            # Determine draws_per_prompt
            prompt_counts: Dict[str, int] = defaultdict(int)
            for sample in samples:
                prompt_counts[sample.prompt_id] += 1

            # Check consistency
            draws_counts = list(prompt_counts.values())
            if draws_counts and len(set(draws_counts)) > 1:
                logger.warning(
                    f"Inconsistent draws per prompt for {policy}: {set(draws_counts)}"
                )

            draws_per_prompt = max(draws_counts) if draws_counts else 1

            datasets[policy] = FreshDrawDataset(
                samples=samples,
                target_policy=policy,
                draws_per_prompt=draws_per_prompt,
            )

        return datasets
