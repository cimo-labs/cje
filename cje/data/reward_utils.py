"""Utilities for judge calibration and data preparation.

This module provides utility functions for working with calibrated rewards.
Most data loading and calibration functionality has been moved to the Dataset class.
"""

import json
import logging
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from ..calibration.judge import JudgeCalibrator

logger = logging.getLogger(__name__)


def save_jsonl(data: List[Dict], file_path: str) -> None:
    """Save data to JSONL file."""
    with open(file_path, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")


def add_rewards_to_existing_data(
    data_path: str,
    calibrator: JudgeCalibrator,
    judge_score_field: str = "judge_score",
    output_path: Optional[str] = None,
    output_reward_field: str = "reward",
) -> str:
    """Add calibrated rewards to existing data using pre-fitted calibrator.

    Useful when you've already calibrated on one dataset and want to
    apply the same calibration to new data.

    Args:
        data_path: Path to JSONL file with judge scores
        calibrator: Pre-fitted JudgeCalibrator
        judge_score_field: Field containing judge scores
        output_path: Where to save (defaults to data_path with .rewards suffix)
        output_reward_field: Field name for calibrated rewards

    Returns:
        Path to output file
    """
    # Load raw data to preserve all fields
    raw_data = []
    with open(data_path, "r") as f:
        for line in f:
            raw_data.append(json.loads(line))

    # Also load through dataset for validation
    from cje import load_dataset_from_jsonl

    dataset = load_dataset_from_jsonl(data_path)

    # Helper function to derive prompt_id consistently with DatasetLoader
    def derive_prompt_id(record: Dict, idx: int, prompt_field: str = "prompt") -> str:
        """Derive prompt_id using same logic as DatasetLoader."""
        # Check top-level first
        prompt_id = record.get("prompt_id")
        if prompt_id is not None:
            return str(prompt_id)

        # Check metadata
        prompt_id = record.get("metadata", {}).get("prompt_id")
        if prompt_id is not None:
            return str(prompt_id)

        # Generate from prompt hash
        prompt = record.get(prompt_field, "")
        if prompt:
            import hashlib

            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
            return f"prompt_{prompt_hash}"

        # Fallback to index
        return f"sample_{idx:06d}"

    # Align dataset samples with raw record indices. The loader preserves
    # record order (skipping invalid records), so dataset.samples is an
    # ordered subsequence of raw_data. Keying everything on record index —
    # not prompt_id — keeps duplicate prompt_ids as distinct responses, each
    # of which must receive f(its own judge score).
    aligned_raw_idx: List[int] = []
    raw_pos = 0
    n_raw = len(raw_data)
    for sample in dataset.samples:
        while (
            raw_pos < n_raw
            and derive_prompt_id(raw_data[raw_pos], raw_pos) != sample.prompt_id
        ):
            raw_pos += 1
        if raw_pos >= n_raw:
            raise ValueError(
                f"Could not align dataset sample with prompt_id "
                f"'{sample.prompt_id}' to a raw record in {data_path}; "
                f"loaded samples are expected to be an ordered subsequence "
                f"of the raw records."
            )
        aligned_raw_idx.append(raw_pos)
        raw_pos += 1

    # Collect judge scores per sample (positional, not keyed by prompt_id)
    judge_scores = []
    for sample, raw_idx in zip(dataset.samples, aligned_raw_idx):
        # Get judge score - standard field is top-level, custom fields in metadata
        score = None
        if judge_score_field == "judge_score":
            score = sample.judge_score  # Top-level field (can be None)
        else:
            # Custom field name - check metadata
            score = sample.metadata.get(judge_score_field)

        # If still None, try the aligned raw record
        if score is None:
            raw_record = raw_data[raw_idx]
            if judge_score_field in raw_record:
                score = raw_record[judge_score_field]

        if score is None:
            raise ValueError(
                f"Judge score field '{judge_score_field}' not found for prompt_id {sample.prompt_id}"
            )

        if isinstance(score, dict):
            score = score.get("mean", score.get("value"))

        judge_scores.append(float(score))

    judge_scores_array = np.array(judge_scores)

    # Apply calibration
    calibrated_rewards = calibrator.predict(judge_scores_array)

    # Map raw record index -> calibrated reward
    reward_by_raw_idx = {
        raw_idx: float(reward)
        for raw_idx, reward in zip(aligned_raw_idx, calibrated_rewards)
    }

    # Add rewards to raw data, matching by record index
    data = []
    skipped_count = 0
    for i, raw_record in enumerate(raw_data):
        # Start from the original raw record to preserve all fields
        record = dict(raw_record)  # Make a copy

        # Inject the calibrated reward if we have it
        if i in reward_by_raw_idx:
            record[output_reward_field] = reward_by_raw_idx[i]
        else:
            # This record was filtered by the loader, no reward to assign
            skipped_count += 1
            # Optionally set to None to be explicit
            # record[output_reward_field] = None

        data.append(record)

    if skipped_count > 0:
        logger.warning(
            f"Skipped adding rewards to {skipped_count}/{len(raw_data)} records "
            f"(filtered by DatasetLoader)"
        )

    # Save
    if output_path is None:
        path = Path(data_path)
        output_path = str(path.parent / f"{path.stem}.rewards{path.suffix}")

    save_jsonl(data, output_path)
    return output_path
