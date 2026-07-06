"""Data loading utilities following SOLID principles.

This module separates data loading concerns from the Dataset model,
following the Single Responsibility Principle.
"""

import json
import logging
import math
from typing import List, Dict, Any, Optional
from collections import defaultdict
from pathlib import Path

from .ingest import read_aliased_field, resolve_prompt_id
from .models import Dataset, Sample
from .fresh_draws import FreshDrawSample, FreshDrawDataset

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads and converts raw data into typed Dataset objects.

    Follows Single Responsibility Principle - only handles data loading and conversion.
    """

    def __init__(
        self,
        prompt_field: str = "prompt",
        response_field: str = "response",
        reward_field: str = "reward",
    ):
        self.prompt_field = prompt_field
        self.response_field = response_field
        self.reward_field = reward_field

    def load_from_jsonl(
        self, file_path: str, target_policies: Optional[List[str]] = None
    ) -> Dataset:
        """Load Dataset from a JSONL file.

        Args:
            file_path: Path to JSONL file
            target_policies: Optional list of target policy names

        Returns:
            Dataset instance
        """
        data = []
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return self._convert_raw_data(data, target_policies)

    def _check_score_ranges(self, data: List[Dict[str, Any]]) -> None:
        """Hard-fail on judge/oracle values outside [0, 1].

        Out-of-range records used to be silently FILTERED with only a
        warning, so a 0-100-scale calibration file died later with an
        unhelpful "No valid samples" error. A wrong scale is a dataset-level
        problem, not a per-record one: fail loudly with the observed range
        and the fix.
        """
        for field in ("judge_score", "oracle_label"):
            values: List[float] = []
            for record in data:
                raw = read_aliased_field(record, field)
                if raw is None:
                    continue
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    # Non-numeric values keep their per-record handling
                    # (filtered with a counted warning below).
                    continue
                if not math.isnan(value):
                    values.append(value)
            n_out_of_range = sum(1 for v in values if v < 0.0 or v > 1.0)
            if n_out_of_range:
                raise ValueError(
                    f"Calibration data {field} values outside [0, 1] "
                    f"(observed range {min(values)}-{max(values)}, "
                    f"{n_out_of_range} rows). Rescale to [0, 1], or pass your "
                    f"data in-memory via fresh_draws_data, which "
                    f"auto-normalizes any bounded scale."
                )

    def _convert_raw_data(
        self, data: List[Dict[str, Any]], target_policies: Optional[List[str]] = None
    ) -> Dataset:
        """Convert raw data to Dataset."""
        # Wrong-scale judge/oracle values are a dataset-level error, not a
        # per-record one: fail before the per-record filter loop.
        self._check_score_ranges(data)

        # Convert raw data to samples. Invalid records are FILTERED with a
        # counted warning; if every record is invalid we FAIL loudly.
        samples = []
        n_skipped = 0
        for idx, record in enumerate(data):
            try:
                sample = self._convert_record_to_sample(record, idx)
                samples.append(sample)
            except (KeyError, ValueError) as e:
                n_skipped += 1
                logger.warning(f"Skipping invalid record {idx}: {e}")
                continue

        if n_skipped:
            logger.warning(
                f"Skipped {n_skipped}/{len(data)} invalid records while loading dataset"
            )

        if not samples:
            raise ValueError(
                f"No valid samples could be created from data: all "
                f"{len(data)} records were skipped due to validation errors. "
                f"See the warnings above for per-record reasons."
            )

        return Dataset(
            samples=samples,
            target_policies=target_policies or [],
            metadata={"source": "loader"},
        )

    def _convert_record_to_sample(self, record: Dict[str, Any], idx: int = 0) -> Sample:
        """Convert a single record to a Sample.

        Args:
            record: Raw data record
            idx: Index in dataset (used as fallback if prompt is also missing)
        """
        # Get prompt_id - check top-level first, then metadata, then
        # auto-generate (prompt hash, index fallback). Explicit None checks
        # and string coercion (so integer ids like 0 survive validation and
        # join correctly across datasets) live in resolve_prompt_id.
        prompt_id = resolve_prompt_id(record, idx, prompt_field=self.prompt_field)

        # Extract reward if present (handle nested format)
        reward = None
        if self.reward_field in record:
            reward = record[self.reward_field]
            if isinstance(reward, dict):
                reward = reward.get("mean", reward.get("value"))
            if reward is not None:
                reward = float(reward)

        # Extract judge_score and oracle_label - prioritize top-level, fallback to metadata
        judge_score = read_aliased_field(record, "judge_score")
        if judge_score is not None:
            judge_score = float(judge_score)

        oracle_label = read_aliased_field(record, "oracle_label")
        if oracle_label is not None:
            oracle_label = float(oracle_label)

        metadata_dict = record.get("metadata", {})

        # Collect all other fields into metadata (excluding judge_score/oracle_label now)
        metadata = {}
        core_fields = {
            "prompt_id",
            self.prompt_field,
            self.response_field,
            self.reward_field,
            # 0.3.x-era OPE logprob fields: present-and-ignored in Direct
            # mode (kept out of metadata rather than swept into it).
            "base_policy_logprob",
            "target_policy_logprobs",
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

        # Create Sample object with judge_score and oracle_label as top-level
        # fields. prompt/response default to "" so minimal judge+oracle
        # calibration files ({prompt_id, judge_score, oracle_label}) load —
        # calibration needs neither text field.
        return Sample(
            prompt_id=prompt_id,
            prompt=record.get(self.prompt_field, ""),
            response=record.get(self.response_field, ""),
            reward=reward,
            judge_score=judge_score,
            oracle_label=oracle_label,
            metadata=metadata,
        )


class FreshDrawLoader:
    """Loader for fresh draw samples used in Direct-mode estimation."""

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

        # Group samples by policy. Invalid lines raise with file/line
        # context (mirroring load_fresh_draws_auto's loud behavior) instead
        # of being silently skipped; blank/whitespace-only lines (e.g. a
        # trailing newline) are skipped, not errors.
        samples_by_policy: Dict[str, List[FreshDrawSample]] = defaultdict(list)

        with open(path_obj, "r") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        raise ValueError(
                            f"expected a JSON object, got {type(data).__name__}"
                        )

                    # Create FreshDrawSample (str() so integer ids like 0
                    # survive validation and join with logged data)
                    sample = FreshDrawSample(
                        prompt_id=str(data["prompt_id"]),
                        target_policy=data["target_policy"],
                        judge_score=data["judge_score"],
                        oracle_label=data.get("oracle_label"),  # Optional
                        response=data.get("response"),  # Optional
                        draw_idx=data.get(
                            "draw_idx", 0
                        ),  # Default to 0 if not provided
                    )
                except KeyError as e:
                    raise ValueError(
                        f"Invalid fresh draw record at {path_obj}:{line_num}: "
                        f"missing required field {e}"
                    ) from e
                except ValueError as e:
                    # Covers json.JSONDecodeError and pydantic ValidationError
                    # (both subclass ValueError). Add file/line context.
                    raise ValueError(
                        f"Invalid fresh draw record at {path_obj}:{line_num}: {e}"
                    ) from e

                samples_by_policy[sample.target_policy].append(sample)

        # Create FreshDrawDataset for each policy
        datasets = {}
        for policy, samples in samples_by_policy.items():
            datasets[policy] = FreshDrawDataset(
                samples=samples,
                target_policy=policy,
            )

        return datasets
