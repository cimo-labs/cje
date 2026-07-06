"""Data models and utilities for fresh draws (Direct-mode evaluation data)."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any, Tuple, TypeVar
import numpy as np
from pydantic import BaseModel, Field, field_validator

from .ingest import (
    POLICY_FILE_PATTERNS,
    read_aliased_field,
    resolve_policy_file,
    resolve_prompt_id,
)
from .normalization import ScaleInfo, detect_range

logger = logging.getLogger(__name__)


class FreshDrawSample(BaseModel):
    """A single fresh draw sample for Direct-mode evaluation.

    Represents a fresh response sampled from a target policy,
    evaluated by the judge.
    """

    prompt_id: str = Field(..., description="ID to align with calibration data")
    target_policy: str = Field(..., description="Policy that generated this response")
    judge_score: float = Field(..., ge=0, le=1, description="Judge evaluation score")
    oracle_label: Optional[float] = Field(
        None, ge=0, le=1, description="Ground truth oracle label (for calibration)"
    )
    response: Optional[str] = Field(None, description="Generated response (optional)")
    draw_idx: int = Field(
        ..., ge=0, description="Draw index for this prompt (0, 1, 2...)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., computed covariates)",
    )


class FreshDrawDataset(BaseModel):
    """Collection of fresh draws for a target policy.

    Contains pre-generated fresh samples from a target policy,
    evaluated by a judge, for use in Direct-mode estimation.
    """

    target_policy: str = Field(..., description="Target policy name")
    samples: List[FreshDrawSample] = Field(..., min_length=1)

    @field_validator("samples")
    def validate_samples(
        cls, v: List[FreshDrawSample], info: Any
    ) -> List[FreshDrawSample]:
        """Ensure samples are consistent."""
        if "target_policy" in info.data:
            policy = info.data["target_policy"]
            for sample in v:
                if sample.target_policy != policy:
                    raise ValueError(
                        f"Sample has policy '{sample.target_policy}' "
                        f"but dataset is for '{policy}'"
                    )
        return v

    @property
    def n_samples(self) -> int:
        """Total number of fresh draw samples."""
        return len(self.samples)

    @property
    def draws_per_prompt(self) -> int:
        """Maximum number of draws for any single prompt."""
        counts: Dict[str, int] = {}
        for sample in self.samples:
            counts[sample.prompt_id] = counts.get(sample.prompt_id, 0) + 1
        return max(counts.values()) if counts else 1

    def get_prompt_ids(self) -> List[str]:
        """Get unique prompt IDs in dataset."""
        return sorted(set(s.prompt_id for s in self.samples))


# ============================================================================
# Utility functions for fresh draws
# ============================================================================


def load_fresh_draws_from_jsonl(path: str) -> Dict[str, FreshDrawDataset]:
    """Load fresh draws from JSONL file, grouped by policy.

    This function delegates to FreshDrawLoader in the loaders module
    for consistency with other data loading operations.

    Expected JSONL format:
    {"prompt_id": "0", "target_policy": "premium", "judge_score": 0.85, "draw_idx": 0}
    {"prompt_id": "0", "target_policy": "premium", "judge_score": 0.82, "draw_idx": 1}
    {"prompt_id": "1", "target_policy": "premium", "judge_score": 0.90, "draw_idx": 0}

    Args:
        path: Path to JSONL file containing fresh draws

    Returns:
        Dict mapping policy names to FreshDrawDataset objects
    """
    from .loaders import FreshDrawLoader

    return FreshDrawLoader.load_from_jsonl(path)


_T = TypeVar("_T")


def _parse_fresh_draw_record(
    data: Dict[str, Any], idx: int, policy: str
) -> Dict[str, Any]:
    """Parse one raw fresh-draw record into canonical dict form.

    Preserves the review-hardened per-record semantics of the original
    load_fresh_draws_auto loop:

    - prompt_id: top level -> metadata -> prompt-hash -> index fallback,
      with explicit None checks so falsy-but-valid ids (0, "") survive;
    - judge_score / oracle_label: top level first, then metadata; missing
      judge_score is never fabricated - fail clearly;
    - float coercion here so type errors surface with file/line context.
    """
    prompt_id = resolve_prompt_id(data, idx, policy=policy)

    # Check for judge_score properly - don't use 'or' for numeric fields
    judge_score = read_aliased_field(data, "judge_score")
    if judge_score is None:
        # Never fabricate missing data - fail clearly
        raise ValueError(
            f"Missing judge_score for prompt_id={prompt_id}. "
            f"Fresh draws require judge scores."
        )

    record: Dict[str, Any] = {
        "prompt_id": prompt_id,
        "judge_score": float(judge_score),
        "response": data.get("response", ""),
    }

    # Extract oracle_label if present (for calibration)
    oracle_label = read_aliased_field(data, "oracle_label")
    if oracle_label is not None:
        record["oracle_label"] = float(oracle_label)

    if "draw_idx" in data:
        record["draw_idx"] = data["draw_idx"]

    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        record["metadata"] = metadata

    return record


def _load_policy_file_records(
    file_path: Path,
    policy: str,
    parse: Callable[[Dict[str, Any], int], _T],
) -> List[_T]:
    """Read a per-policy fresh-draws file, applying ``parse`` per record.

    Parse errors propagate loudly (with file/line context) instead of being
    swallowed; blank/whitespace-only lines (e.g. a trailing newline) are
    skipped, matching the logged-data loader — only real records should hit
    the loud parse-error path.
    """
    results: List[_T] = []
    with open(file_path, "r") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                item = parse(data, idx)
            except ValueError as e:
                # Covers json.JSONDecodeError and pydantic ValidationError
                # (both subclass ValueError). Add file/line context.
                raise ValueError(
                    f"Invalid fresh draw record at {file_path}:{idx + 1} "
                    f"for policy '{policy}': {e}"
                ) from e
            results.append(item)

    if not results:
        raise ValueError(
            f"Fresh draw file {file_path} exists but contains no records "
            f"for policy '{policy}'."
        )
    return results


def load_fresh_draws_auto(
    data_dir: Path,
    policy: str,
    verbose: bool = False,
) -> FreshDrawDataset:
    """
    Load fresh draws for a single policy from a fresh-draws directory.

    The file is located via the canonical POLICY_FILE_PATTERNS (first
    existing match wins):
    1. {data_dir}/{policy}_responses.jsonl
    2. {data_dir}/{policy}.jsonl
    3. {data_dir}/responses/{policy}.jsonl
    4. {data_dir}/fresh_draws/{policy}.jsonl

    Values must already be in [0, 1]: samples are constructed with hard
    [0, 1] bounds. The analyze_dataset directory flow instead goes through
    fresh_draws_data_from_dir + fresh_draws_from_dict, which jointly
    auto-normalizes any bounded scale.

    Args:
        data_dir: Directory to search for fresh draw files
        policy: Target policy name
        verbose: Whether to log detailed information

    Returns:
        FreshDrawDataset for the specified policy

    Raises:
        FileNotFoundError: If no fresh draw file found
    """
    # Convert to Path if string
    data_dir = Path(data_dir)

    file_path = resolve_policy_file(data_dir, policy)
    if file_path is None:
        # No file found - raise error with helpful message
        searched_paths = "\n  ".join(
            str(data_dir / pattern.format(policy=policy))
            for pattern in POLICY_FILE_PATTERNS
        )
        raise FileNotFoundError(
            f"No fresh draw file found for policy '{policy}'. Searched:\n  {searched_paths}\n"
            f"Direct mode requires judge-scored fresh draws for every target policy."
        )

    if verbose:
        logger.info(f"Loading fresh draws from {file_path}")

    def _parse_sample(data: Dict[str, Any], idx: int) -> FreshDrawSample:
        record = _parse_fresh_draw_record(data, idx, policy)
        return FreshDrawSample(
            prompt_id=record["prompt_id"],
            target_policy=policy,
            response=record["response"],
            judge_score=record["judge_score"],
            oracle_label=record.get("oracle_label"),
            draw_idx=record.get("draw_idx", 0),
        )

    fresh_samples = _load_policy_file_records(file_path, policy, _parse_sample)

    # Create dataset
    fresh_dataset = FreshDrawDataset(
        target_policy=policy,
        samples=fresh_samples,
    )

    if verbose:
        logger.info(
            f"Loaded {len(fresh_samples)} fresh draws for {policy} "
            f"({len(fresh_dataset.get_prompt_ids())} unique prompts)"
        )

    return fresh_dataset


def fresh_draws_data_from_dir(
    data_dir: Path,
    verbose: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Read a fresh-draws directory into raw records grouped by policy.

    Discovers policies via POLICY_FILE_PATTERNS, reads each policy's file
    with the same loud per-record parsing as load_fresh_draws_auto, and
    returns ``{policy: [record dicts]}`` ready for fresh_draws_from_dict —
    so directory input gets exactly the same joint scale detection and
    auto-normalization as in-memory ``fresh_draws_data``.

    Args:
        data_dir: Directory containing per-policy fresh draw files
        verbose: Whether to log detailed information

    Returns:
        Dict mapping policy names to lists of canonical record dicts

    Raises:
        ValueError: If the directory is missing or contains no policy files
    """
    data_dir = Path(data_dir)
    data: Dict[str, List[Dict[str, Any]]] = {}

    for policy in discover_policies_from_fresh_draws(data_dir):
        file_path = resolve_policy_file(data_dir, policy)
        if file_path is None:
            # Discovery and loading share POLICY_FILE_PATTERNS, so this can
            # only happen if the file disappears between the two calls.
            raise FileNotFoundError(
                f"Fresh draw file for discovered policy '{policy}' vanished "
                f"from {data_dir} while loading."
            )
        if verbose:
            logger.info(f"Loading fresh draws from {file_path}")

        def _parse_record(
            record: Dict[str, Any], idx: int, _policy: str = policy
        ) -> Dict[str, Any]:
            return _parse_fresh_draw_record(record, idx, _policy)

        data[policy] = _load_policy_file_records(file_path, policy, _parse_record)

    return data


@dataclass
class NormalizationInfo:
    """Information about normalization applied to fresh draws data.

    Stores the original scale ranges so results can be inverse-transformed
    back to the user's original scale.
    """

    judge_score_scale: ScaleInfo
    oracle_label_scale: Optional[ScaleInfo]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization in metadata."""
        result: Dict[str, Any] = {
            "judge_score": {
                "original_range": (
                    self.judge_score_scale.min_val,
                    self.judge_score_scale.max_val,
                ),
                "is_identity": self.judge_score_scale.is_identity(),
            },
        }
        if self.oracle_label_scale:
            result["oracle_label"] = {
                "original_range": (
                    self.oracle_label_scale.min_val,
                    self.oracle_label_scale.max_val,
                ),
                "is_identity": self.oracle_label_scale.is_identity(),
            }
        result["results_scale"] = "oracle_original"
        return result


def fresh_draws_from_dict(
    data: Dict[str, List[Dict[str, Any]]],
    verbose: bool = False,
    auto_normalize: bool = True,
) -> Tuple[Dict[str, FreshDrawDataset], Optional[NormalizationInfo]]:
    """Convert in-memory dict to FreshDrawDataset objects with auto-normalization.

    This allows users to provide fresh draws data directly without writing to disk.
    Values are automatically normalized to [0,1] internally, and scale information
    is returned for inverse-transforming results back to the original scale.

    Expected format:
        {
            "policy_a": [
                {"prompt_id": "1", "judge_score": 85, "oracle_label": 90},  # 0-100 scale
                {"prompt_id": "2", "judge_score": 72},
                ...
            ],
            "policy_b": [...]
        }

    Each record must have at minimum: prompt_id, judge_score
    Optional fields: oracle_label, response, draw_idx, metadata

    Args:
        data: Dict mapping policy names to lists of record dicts
        verbose: Whether to log progress
        auto_normalize: Whether to auto-detect scale and normalize to [0,1].
            If False, values must already be in [0,1].

    Returns:
        Tuple of (datasets_dict, normalization_info).
        normalization_info is None if auto_normalize=False or data is already [0,1].

    Example:
        >>> data = {
        ...     "policy_a": [
        ...         {"prompt_id": "q1", "judge_score": 85, "oracle_label": 90},
        ...         {"prompt_id": "q2", "judge_score": 72},
        ...     ]
        ... }
        >>> datasets, norm_info = fresh_draws_from_dict(data)
        >>> datasets["policy_a"].n_samples
        2
        >>> norm_info.judge_score_scale.max_val
        85.0
    """
    if not data:
        raise ValueError("fresh_draws_data is empty")

    # Collect all judge_scores and oracle_labels across all policies for range detection
    all_judge_scores: List[float] = []
    all_oracle_labels: List[float] = []

    for policy, records in data.items():
        for record in records:
            judge_score = record.get("judge_score")
            if judge_score is not None:
                all_judge_scores.append(float(judge_score))
            oracle_label = record.get("oracle_label")
            if oracle_label is not None:
                all_oracle_labels.append(float(oracle_label))

    if not all_judge_scores:
        raise ValueError("No judge_score values found in data")

    # Detect ranges and create scale info
    norm_info: Optional[NormalizationInfo] = None
    judge_scale: Optional[ScaleInfo] = None
    oracle_scale: Optional[ScaleInfo] = None

    if auto_normalize:
        # Check if values are already in [0, 1] range
        # If ALL values are within [0, 1], assume data is already normalized
        judge_arr = np.array(all_judge_scores)
        judge_in_unit_interval = np.all((judge_arr >= 0) & (judge_arr <= 1))

        oracle_in_unit_interval = True
        if all_oracle_labels:
            oracle_arr = np.array(all_oracle_labels)
            oracle_in_unit_interval = bool(
                np.all((oracle_arr >= 0) & (oracle_arr <= 1))
            )

        # Only normalize if values are OUTSIDE [0, 1]
        needs_normalization = not judge_in_unit_interval or not oracle_in_unit_interval

        if needs_normalization:
            # Detect judge score range
            judge_scale = detect_range(judge_arr, field_name="judge_score")

            # Detect oracle label range if any oracle labels exist
            if all_oracle_labels:
                oracle_scale = detect_range(
                    np.array(all_oracle_labels), field_name="oracle_label"
                )

            norm_info = NormalizationInfo(
                judge_score_scale=judge_scale,
                oracle_label_scale=oracle_scale,
            )
            if verbose:
                logger.info(
                    f"Auto-normalizing: judge_score [{judge_scale.min_val}, {judge_scale.max_val}] -> [0, 1]"
                )
                if oracle_scale:
                    logger.info(
                        f"Auto-normalizing: oracle_label [{oracle_scale.min_val}, {oracle_scale.max_val}] -> [0, 1]"
                    )

    result: Dict[str, FreshDrawDataset] = {}

    for policy, records in data.items():
        if not records:
            logger.warning(f"Policy '{policy}' has no records, skipping")
            continue

        samples: List[FreshDrawSample] = []
        prompt_draw_counts: Dict[str, int] = {}

        for idx, record in enumerate(records):
            # Validate required fields
            prompt_id = record.get("prompt_id")
            if prompt_id is None:
                raise ValueError(
                    f"Record {idx} for policy '{policy}' missing required field 'prompt_id'"
                )

            judge_score = record.get("judge_score")
            if judge_score is None:
                raise ValueError(
                    f"Record {idx} for policy '{policy}' (prompt_id={prompt_id}) "
                    f"missing required field 'judge_score'"
                )

            # Normalize values if needed
            normalized_judge = float(judge_score)
            normalized_oracle: Optional[float] = None

            if norm_info and judge_scale is not None:
                normalized_judge = judge_scale.normalize(float(judge_score))
                if record.get("oracle_label") is not None and oracle_scale:
                    normalized_oracle = oracle_scale.normalize(
                        float(record["oracle_label"])
                    )
            elif record.get("oracle_label") is not None:
                normalized_oracle = float(record["oracle_label"])

            # Track draw_idx per prompt
            if prompt_id not in prompt_draw_counts:
                prompt_draw_counts[prompt_id] = 0
            draw_idx = record.get("draw_idx", prompt_draw_counts[prompt_id])
            prompt_draw_counts[prompt_id] += 1

            sample = FreshDrawSample(
                prompt_id=str(prompt_id),
                target_policy=policy,
                judge_score=normalized_judge,
                oracle_label=normalized_oracle,
                response=record.get("response"),
                draw_idx=draw_idx,
                metadata=record.get("metadata", {}),
            )
            samples.append(sample)

        dataset = FreshDrawDataset(
            target_policy=policy,
            samples=samples,
        )
        result[policy] = dataset

        if verbose:
            n_oracle = sum(1 for s in samples if s.oracle_label is not None)
            logger.info(
                f"Created FreshDrawDataset for '{policy}': "
                f"{len(samples)} samples, {len(prompt_draw_counts)} prompts, "
                f"{n_oracle} with oracle labels"
            )

    if not result:
        raise ValueError("No valid fresh draws data found in any policy")

    return result, norm_info


# File stems that never denote a policy (auxiliary files living next to
# per-policy fresh draws).
_NON_POLICY_STEMS = frozenset(["dataset", "data", "logs"])


def discover_policies_from_fresh_draws(fresh_draws_dir: Path) -> List[str]:
    """Discover target policies from fresh draws directory.

    Recognizes exactly the canonical POLICY_FILE_PATTERNS, checked in
    order — the first location that yields any policies wins:
    1. {policy}_responses.jsonl
    2. {policy}.jsonl
    3. responses/{policy}.jsonl
    4. fresh_draws/{policy}.jsonl

    Every discovered policy is guaranteed to load, because loading
    (resolve_policy_file) resolves through the same pattern list.

    Args:
        fresh_draws_dir: Directory containing fresh draw files

    Returns:
        List of discovered policy names

    Raises:
        ValueError: If no fresh draw files found
    """
    fresh_draws_path = Path(fresh_draws_dir)
    if not fresh_draws_path.exists():
        raise ValueError(f"Fresh draws directory not found: {fresh_draws_dir}")

    policies: List[str] = []

    # Pattern 1: {policy}_responses.jsonl
    for path in fresh_draws_path.glob("*_responses.jsonl"):
        policies.append(path.stem[: -len("_responses")])

    # Patterns 2-4: bare {policy}.jsonl at the top level, then in the
    # responses/ and fresh_draws/ subdirectories. Each is a fallback: it is
    # only consulted when no earlier pattern matched, so auxiliary .jsonl
    # files next to *_responses.jsonl files are never mistaken for policies.
    for glob_pattern in ("*.jsonl", "responses/*.jsonl", "fresh_draws/*.jsonl"):
        if policies:
            break
        for path in fresh_draws_path.glob(glob_pattern):
            # Skip files that don't look like policy files
            if path.stem not in _NON_POLICY_STEMS:
                policies.append(path.stem)

    if not policies:
        patterns = ", ".join(f"'{p}'" for p in POLICY_FILE_PATTERNS)
        raise ValueError(
            f"No fresh draw files found in {fresh_draws_dir}. "
            f"Expected per-policy files matching one of: {patterns} "
            f"(e.g. 'policy_a_responses.jsonl' or 'policy_a.jsonl')"
        )

    logger.info(f"Discovered {len(policies)} policies from fresh draws: {policies}")
    return sorted(policies)


def compute_response_covariates(
    fresh_draws: FreshDrawDataset,
    covariate_names: Optional[List[str]] = None,
) -> FreshDrawDataset:
    """Compute covariates for fresh draws based on response text.

    This function computes response-level covariates (like response_length)
    and stores them in each sample's metadata field. This is needed for
    DR estimators and Direct Method to use covariates properly.

    Args:
        fresh_draws: FreshDrawDataset to augment with covariates
        covariate_names: List of covariate names to compute. Currently supported:
            - "response_length": word count (len(response.split())) - matches calibration

    Returns:
        New FreshDrawDataset with covariates computed and stored in metadata

    Example:
        >>> fresh_draws = load_fresh_draws_auto(data_dir, "policy_a")
        >>> fresh_draws_with_covs = compute_response_covariates(
        ...     fresh_draws, covariate_names=["response_length"]
        ... )
        >>> # Now fresh_draws_with_covs.samples[i].metadata["response_length"] exists
    """
    if covariate_names is None:
        covariate_names = []

    if not covariate_names:
        logger.debug("No covariate names specified, returning unchanged")
        return fresh_draws

    # Compute covariates for each sample
    updated_samples = []
    for sample in fresh_draws.samples:
        # Create a copy of the sample's metadata
        new_metadata = dict(sample.metadata) if sample.metadata else {}

        for cov_name in covariate_names:
            if cov_name == "response_length":
                # Compute response_length matching the formula in calibration/dataset.py
                # Uses word count (len(response.split())) to match calibration exactly
                if sample.response is not None:
                    # CRITICAL: Must match calibration/dataset.py AUTO_COMPUTABLE_COVARIATES
                    # which uses: lambda sample: float(len(sample.response.split()))
                    word_count = len(sample.response.split())
                    new_metadata["response_length"] = float(word_count)
                else:
                    raise ValueError(
                        f"Cannot compute response_length for sample {sample.prompt_id} "
                        f"draw {sample.draw_idx}: response is None"
                    )
            else:
                raise ValueError(
                    f"Unsupported covariate: {cov_name}. "
                    f"Currently supported: ['response_length']"
                )

        # Create new sample with updated metadata
        updated_sample = sample.model_copy(update={"metadata": new_metadata})
        updated_samples.append(updated_sample)

    # Create new dataset with updated samples
    updated_dataset = FreshDrawDataset(
        target_policy=fresh_draws.target_policy,
        samples=updated_samples,
    )

    logger.info(
        f"Computed {len(covariate_names)} covariates for {len(updated_samples)} "
        f"fresh draw samples (policy={fresh_draws.target_policy})"
    )

    return updated_dataset
