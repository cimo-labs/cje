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
    canonicalize_record,
    deduplicate_canonical_records,
    require_prompt_identity,
    resolve_policy_file,
)
from .normalization import (
    ScaleDeclaration,
    ScaleInfo,
    coerce_scale,
    detect_range,
    unit_scale,
    validate_values_on_scale,
)

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
    row_id: Optional[str] = Field(
        default=None, description="Stable source-local row identity"
    )
    observation_id: Optional[str] = Field(
        default=None, description="Optional cross-source response identity"
    )
    source_id: Optional[str] = Field(default=None, description="Logical input source")
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
    data: Dict[str, Any],
    idx: int,
    policy: str,
    *,
    source_id: str,
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
) -> Dict[str, Any]:
    """Parse one raw fresh-draw record into canonical dict form.

    Preserves the review-hardened per-record semantics of the original
    load_fresh_draws_auto loop:

    - prompt_id: top level -> metadata -> prompt-hash, with explicit None
      checks so falsy-but-valid ids (0, "") survive; records with neither
      prompt_id nor prompt text fail loudly (no fabricated index ids);
    - judge_score / oracle_label: top level first, then metadata; missing
      judge_score is never fabricated - fail clearly;
    - float coercion here so type errors surface with file/line context.
    """
    if not isinstance(data, dict):
        raise ValueError(f"expected a mapping, got {type(data).__name__}")
    require_prompt_identity(data)
    annotated = dict(data)
    annotated.setdefault("_cje_source_id", source_id)
    annotated.setdefault("_cje_row_id", f"{source_id}:line:{idx + 1}")
    return canonicalize_record(
        annotated,
        idx,
        source_id=source_id,
        policy=policy,
        judge_field=judge_field,
        oracle_field=oracle_field,
    )


def _load_policy_file_records(
    file_path: Path,
    policy: str,
    parse: Callable[[Dict[str, Any], int], _T],
    on_invalid: str = "error",
    drop_stats: Optional[Dict[str, int]] = None,
) -> List[_T]:
    """Read a per-policy fresh-draws file, applying ``parse`` per record.

    Parse errors propagate loudly (with file/line context) instead of being
    swallowed; blank/whitespace-only lines (e.g. a trailing newline) are
    skipped, matching the logged-data loader — only real records should hit
    the loud parse-error path. When ``drop_stats`` is supplied, dropped
    records are counted under the policy name.
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
                if on_invalid == "error":
                    raise ValueError(
                        f"Invalid fresh draw record at {file_path}:{idx + 1} "
                        f"for policy '{policy}': {e}"
                    ) from e
                logger.warning(
                    "Dropping invalid fresh draw record at %s:%d for policy " "%r: %s",
                    file_path,
                    idx + 1,
                    policy,
                    e,
                )
                if drop_stats is not None:
                    drop_stats[policy] = drop_stats.get(policy, 0) + 1
                continue
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
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
) -> FreshDrawDataset:
    """
    Load fresh draws for a single policy from a fresh-draws directory.

    The file is located via the canonical POLICY_FILE_PATTERNS. More than one
    matching file for the policy is rejected:
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

    def _parse_record(data: Dict[str, Any], idx: int) -> Dict[str, Any]:
        record = _parse_fresh_draw_record(
            data,
            idx,
            policy,
            source_id=str(file_path.resolve()),
            judge_field=judge_field,
            oracle_field=oracle_field,
        )
        validate_values_on_scale(
            np.asarray([record["judge_score"]], dtype=float),
            unit_scale(),
            field_name=judge_field,
        )
        if record.get("oracle_label") is not None:
            validate_values_on_scale(
                np.asarray([record["oracle_label"]], dtype=float),
                unit_scale(),
                field_name=oracle_field,
            )
        return record

    fresh_records = _load_policy_file_records(file_path, policy, _parse_record)
    datasets, _ = fresh_draws_from_dict(
        {policy: fresh_records},
        auto_normalize=False,
        judge_field=judge_field,
        oracle_field=oracle_field,
        on_invalid="error",
    )
    fresh_dataset = datasets[policy]

    if verbose:
        logger.info(
            f"Loaded {len(fresh_dataset.samples)} fresh draws for {policy} "
            f"({len(fresh_dataset.get_prompt_ids())} unique prompts)"
        )

    return fresh_dataset


def fresh_draws_data_from_dir(
    data_dir: Path,
    verbose: bool = False,
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
    on_invalid: str = "error",
    exclude_paths: Optional[List[Path]] = None,
    drop_stats: Optional[Dict[str, int]] = None,
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
    if on_invalid not in {"error", "drop"}:
        raise ValueError("on_invalid must be 'error' or 'drop'")
    data: Dict[str, List[Dict[str, Any]]] = {}
    excluded = {Path(path).resolve() for path in (exclude_paths or [])}

    for policy in discover_policies_from_fresh_draws(
        data_dir, exclude_paths=list(excluded)
    ):
        file_path = resolve_policy_file(data_dir, policy, exclude_paths=list(excluded))
        if file_path is None:
            # Discovery and loading share POLICY_FILE_PATTERNS, so this can
            # only happen if the file disappears between the two calls.
            raise FileNotFoundError(
                f"Fresh draw file for discovered policy '{policy}' vanished "
                f"from {data_dir} while loading."
            )
        if file_path.resolve() in excluded:
            continue
        if verbose:
            logger.info(f"Loading fresh draws from {file_path}")

        def _parse_record(
            record: Dict[str, Any], idx: int, _policy: str = policy
        ) -> Dict[str, Any]:
            return _parse_fresh_draw_record(
                record,
                idx,
                _policy,
                source_id=str(file_path.resolve()),
                judge_field=judge_field,
                oracle_field=oracle_field,
            )

        data[policy] = _load_policy_file_records(
            file_path,
            policy,
            _parse_record,
            on_invalid=on_invalid,
            drop_stats=drop_stats,
        )

    return data


@dataclass
class NormalizationInfo:
    """Information about normalization applied to fresh draws data.

    Stores the original scale ranges so results can be inverse-transformed
    back to the user's original scale.
    """

    judge_score_scale: ScaleInfo
    oracle_label_scale: Optional[ScaleInfo]
    judge_scale_origin: str = "observed"
    oracle_scale_origin: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization in metadata."""
        result: Dict[str, Any] = {
            "judge_score": {
                "original_range": (
                    self.judge_score_scale.min_val,
                    self.judge_score_scale.max_val,
                ),
                "is_identity": self.judge_score_scale.is_identity(),
                "origin": self.judge_scale_origin,
            },
        }
        if self.oracle_label_scale:
            result["oracle_label"] = {
                "original_range": (
                    self.oracle_label_scale.min_val,
                    self.oracle_label_scale.max_val,
                ),
                "is_identity": self.oracle_label_scale.is_identity(),
                "origin": self.oracle_scale_origin or "observed",
            }
        result["results_scale"] = "oracle_original"
        return result


def fresh_draws_from_dict(
    data: Dict[str, List[Dict[str, Any]]],
    verbose: bool = False,
    auto_normalize: bool = True,
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
    judge_scale: ScaleDeclaration = None,
    oracle_scale: ScaleDeclaration = None,
    on_invalid: str = "error",
    drop_stats: Optional[Dict[str, int]] = None,
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
        on_invalid: "error" (default) raises on invalid records; "drop"
            removes them with a counted warning.
        drop_stats: Optional dict that accumulates per-policy counts of
            dropped records when ``on_invalid="drop"``.

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
    if on_invalid not in {"error", "drop"}:
        raise ValueError("on_invalid must be 'error' or 'drop'")

    declared_judge_scale = coerce_scale(judge_scale, field_name="fresh_judge_scale")
    declared_oracle_scale = coerce_scale(oracle_scale, field_name="fresh_oracle_scale")

    canonical_data: Dict[str, List[Dict[str, Any]]] = {}
    n_dropped = 0

    def _count_drop(policy_name: str) -> None:
        if drop_stats is not None:
            drop_stats[policy_name] = drop_stats.get(policy_name, 0) + 1

    for policy, records in data.items():
        canonical_records: List[Dict[str, Any]] = []
        for idx, raw_record in enumerate(records):
            try:
                if not isinstance(raw_record, dict):
                    raise ValueError(
                        f"expected a mapping, got {type(raw_record).__name__}"
                    )
                require_prompt_identity(raw_record)
                canonical_records.append(
                    canonicalize_record(
                        raw_record,
                        idx,
                        source_id=f"in_memory:{policy}",
                        policy=str(policy),
                        judge_field=judge_field,
                        oracle_field=oracle_field,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                if on_invalid == "error":
                    raise ValueError(
                        f"Invalid fresh draw record {idx} for policy "
                        f"'{policy}': {exc}"
                    ) from exc
                n_dropped += 1
                _count_drop(str(policy))
                logger.warning(
                    "Dropping invalid fresh draw record %s for policy %r: %s",
                    idx,
                    policy,
                    exc,
                )
        if not canonical_records:
            raise ValueError(
                f"Policy '{policy}' has no valid fresh-draw records after "
                f"applying on_invalid='{on_invalid}'. Refusing to remove a "
                "requested evaluation policy silently."
            )
        canonical_data[str(policy)] = canonical_records

    if not canonical_data:
        raise ValueError("No valid fresh draws data found in any policy")

    flattened = [record for records in canonical_data.values() for record in records]
    flattened, n_duplicate_rows = deduplicate_canonical_records(
        flattened, context="fresh-draw"
    )
    canonical_data = {}
    for record in flattened:
        policy = str(record["target_policy"])
        canonical_data.setdefault(policy, []).append(record)
    if n_duplicate_rows:
        logger.warning(
            "Deduplicated %d repeated fresh-draw source rows", n_duplicate_rows
        )

    # Collect all judge_scores and oracle_labels across all policies for range detection
    all_judge_scores: List[float] = []
    all_oracle_labels: List[float] = []

    for policy, records in canonical_data.items():
        for record in records:
            judge_score = record["judge_score"]
            if judge_score is not None:
                all_judge_scores.append(float(judge_score))
            oracle_label = record.get("oracle_label")
            if oracle_label is not None:
                all_oracle_labels.append(float(oracle_label))

    if not all_judge_scores:
        raise ValueError("No judge_score values found in data")

    # Resolve source scales. Explicit declarations are semantic; observed
    # min/max inference remains only as a compatibility behavior.
    norm_info: Optional[NormalizationInfo] = None
    resolved_judge_scale: Optional[ScaleInfo] = declared_judge_scale
    resolved_oracle_scale: Optional[ScaleInfo] = declared_oracle_scale
    judge_origin = "declared" if declared_judge_scale else "unit_default"
    oracle_origin = "declared" if declared_oracle_scale else "unit_default"

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
        if resolved_judge_scale is None:
            resolved_judge_scale = (
                unit_scale()
                if judge_in_unit_interval
                else detect_range(judge_arr, field_name=judge_field)
            )
            judge_origin = "unit_default" if judge_in_unit_interval else "observed"
        validate_values_on_scale(
            judge_arr, resolved_judge_scale, field_name=judge_field
        )

        if all_oracle_labels:
            oracle_arr = np.array(all_oracle_labels)
            if resolved_oracle_scale is None:
                resolved_oracle_scale = (
                    unit_scale()
                    if oracle_in_unit_interval
                    else detect_range(oracle_arr, field_name=oracle_field)
                )
                oracle_origin = (
                    "unit_default" if oracle_in_unit_interval else "observed"
                )
            validate_values_on_scale(
                oracle_arr, resolved_oracle_scale, field_name=oracle_field
            )

        if (
            not resolved_judge_scale.is_identity()
            or (
                resolved_oracle_scale is not None
                and not resolved_oracle_scale.is_identity()
            )
            or declared_judge_scale is not None
            or declared_oracle_scale is not None
        ):
            norm_info = NormalizationInfo(
                judge_score_scale=resolved_judge_scale,
                oracle_label_scale=resolved_oracle_scale,
                judge_scale_origin=judge_origin,
                oracle_scale_origin=(oracle_origin if all_oracle_labels else None),
            )
            if verbose:
                logger.info(
                    "Normalizing %s scale [%s, %s] -> [0, 1] (%s)",
                    judge_field,
                    resolved_judge_scale.min_val,
                    resolved_judge_scale.max_val,
                    judge_origin,
                )
                if resolved_oracle_scale is not None:
                    logger.info(
                        "Normalizing %s scale [%s, %s] -> [0, 1] (%s)",
                        oracle_field,
                        resolved_oracle_scale.min_val,
                        resolved_oracle_scale.max_val,
                        oracle_origin,
                    )
    else:
        resolved_judge_scale = declared_judge_scale or unit_scale()
        resolved_oracle_scale = declared_oracle_scale or (
            unit_scale() if all_oracle_labels else None
        )
        validate_values_on_scale(
            np.asarray(all_judge_scores), unit_scale(), field_name=judge_field
        )
        if all_oracle_labels:
            validate_values_on_scale(
                np.asarray(all_oracle_labels), unit_scale(), field_name=oracle_field
            )

    result: Dict[str, FreshDrawDataset] = {}

    for policy, records in canonical_data.items():
        if not records:
            logger.warning(f"Policy '{policy}' has no records, skipping")
            continue

        samples: List[FreshDrawSample] = []
        prompt_draw_counts: Dict[str, int] = {}
        # prompt_id -> draw_idx -> description of the row that claimed it,
        # so duplicate explicit draw indices can name BOTH conflicting rows.
        used_draw_indices: Dict[str, Dict[int, str]] = {}

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

            if resolved_judge_scale is not None:
                normalized_judge = resolved_judge_scale.normalize(float(judge_score))
                if (
                    record.get("oracle_label") is not None
                    and resolved_oracle_scale is not None
                ):
                    normalized_oracle = resolved_oracle_scale.normalize(
                        float(record["oracle_label"])
                    )
            elif record.get("oracle_label") is not None:
                normalized_oracle = float(record["oracle_label"])

            # Track draw_idx per prompt: missing draw_idx auto-assigns the
            # first free sequential index (the documented 0.5.x behavior).
            if prompt_id not in prompt_draw_counts:
                prompt_draw_counts[prompt_id] = 0
                used_draw_indices[prompt_id] = {}
            draw_idx = record.get("draw_idx")
            if draw_idx is None:
                draw_idx = 0
                while draw_idx in used_draw_indices[prompt_id]:
                    draw_idx += 1
            try:
                sample = FreshDrawSample(
                    prompt_id=str(prompt_id),
                    target_policy=policy,
                    judge_score=normalized_judge,
                    oracle_label=normalized_oracle,
                    response=record.get("response"),
                    draw_idx=draw_idx,
                    metadata=record.get("metadata", {}),
                    row_id=record.get("row_id"),
                    observation_id=record.get("observation_id"),
                    source_id=record.get("source_id"),
                )
            except ValueError as exc:
                if on_invalid == "error":
                    raise ValueError(
                        f"Invalid fresh draw record {idx} for policy "
                        f"'{policy}': {exc}"
                    ) from exc
                n_dropped += 1
                _count_drop(policy)
                logger.warning(
                    "Dropping invalid fresh draw record %s for policy %r: %s",
                    idx,
                    policy,
                    exc,
                )
                continue
            # An explicit duplicate draw_idx is a conflicting row identity:
            # it always fails loudly (never dropped, regardless of
            # on_invalid) with both conflicting rows identified.
            if sample.draw_idx in used_draw_indices[prompt_id]:
                raise ValueError(
                    f"Duplicate draw_idx={sample.draw_idx} for "
                    f"prompt_id={prompt_id!r} in policy '{policy}': record "
                    f"{idx} (row_id={record.get('row_id')!r}) conflicts with "
                    f"{used_draw_indices[prompt_id][sample.draw_idx]}. Give "
                    "each draw for a prompt a distinct draw_idx, or omit "
                    "draw_idx to auto-assign sequential indices."
                )
            used_draw_indices[prompt_id][
                sample.draw_idx
            ] = f"record {idx} (row_id={record.get('row_id')!r})"
            prompt_draw_counts[prompt_id] += 1
            samples.append(sample)

        if not samples:
            raise ValueError(
                f"Policy '{policy}' has no valid fresh-draw records after "
                f"applying on_invalid='{on_invalid}'. Refusing to remove a "
                "requested evaluation policy silently."
            )

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
    if n_dropped:
        logger.warning("Dropped %d invalid fresh-draw records", n_dropped)
    return result, norm_info


# File stems that never denote a policy (auxiliary files living next to
# per-policy fresh draws).
_NON_POLICY_STEMS = frozenset(
    [
        "dataset",
        "data",
        "logs",
        "calibration",
        "calibration_data",
        "human_labels",
        "oracle_labels",
    ]
)


def discover_policies_from_fresh_draws(
    fresh_draws_dir: Path,
    *,
    exclude_paths: Optional[List[Path]] = None,
) -> List[str]:
    """Discover target policies from fresh draws directory.

    Recognizes the union of the canonical POLICY_FILE_PATTERNS:
    1. {policy}_responses.jsonl
    2. {policy}.jsonl
    3. responses/{policy}.jsonl
    4. fresh_draws/{policy}.jsonl

    Every discovered policy is guaranteed to load, because loading resolves
    through the same pattern list. Multiple files resolving to one policy are
    rejected instead of selected by precedence. The discovered policy -> file
    mapping is logged as a warning (union discovery can turn stray auxiliary
    .jsonl files into phantom policies), and files skipped for having a
    reserved auxiliary stem are warned about individually — a policy is never
    dropped silently.

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
    excluded = {Path(path).resolve() for path in (exclude_paths or [])}

    files_by_policy: Dict[str, List[Path]] = {}
    reserved_stem_skips: List[Tuple[str, Path]] = []

    def _record(policy: str, path: Path) -> None:
        if not policy:
            return
        resolved = path.resolve()
        if resolved in excluded:
            return
        if policy in _NON_POLICY_STEMS:
            # Never drop a would-be policy silently: name the skipped file
            # and the reserved stem so a real policy with a reserved name
            # is discoverable by renaming, not by debugging.
            if (policy, resolved) not in reserved_stem_skips:
                reserved_stem_skips.append((policy, resolved))
            return
        matches = files_by_policy.setdefault(policy, [])
        if resolved not in matches:
            matches.append(resolved)

    for path in fresh_draws_path.glob("*_responses.jsonl"):
        _record(path.stem[: -len("_responses")], path)
    for path in fresh_draws_path.glob("*.jsonl"):
        # Files already interpreted by the more-specific pattern must not
        # become a second synthetic policy named ``foo_responses``.
        if not path.stem.endswith("_responses"):
            _record(path.stem, path)
    for subdir in ("responses", "fresh_draws"):
        for path in (fresh_draws_path / subdir).glob("*.jsonl"):
            _record(path.stem, path)

    ambiguous = {
        policy: paths for policy, paths in files_by_policy.items() if len(paths) > 1
    }
    if ambiguous:
        details = "; ".join(
            f"{policy}: {', '.join(str(path) for path in paths)}"
            for policy, paths in sorted(ambiguous.items())
        )
        raise ValueError(
            "Multiple fresh-draw files resolve to the same policy. "
            f"Keep exactly one file per policy. {details}"
        )

    for stem, path in reserved_stem_skips:
        logger.warning(
            "Skipping %s during policy discovery: stem %r is reserved for "
            "auxiliary files (%s) and is never treated as a policy. Rename "
            "the file if it holds a real evaluation policy.",
            path,
            stem,
            ", ".join(sorted(_NON_POLICY_STEMS)),
        )

    policies = sorted(files_by_policy)

    if not policies:
        patterns = ", ".join(f"'{p}'" for p in POLICY_FILE_PATTERNS)
        raise ValueError(
            f"No fresh draw files found in {fresh_draws_dir}. "
            f"Expected per-policy files matching one of: {patterns} "
            f"(e.g. 'policy_a_responses.jsonl' or 'policy_a.jsonl')"
        )

    # Discovery is a union over every filename pattern, so any stray .jsonl
    # becomes a policy: list each policy WITH its source file loudly enough
    # that a phantom policy (or a missing one) is visible before estimation.
    logger.warning(
        "Discovered %d policies from fresh draws (every matching .jsonl "
        "becomes a policy — verify this list): %s",
        len(policies),
        "; ".join(f"{policy} <- {files_by_policy[policy][0]}" for policy in policies),
    )
    return policies


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
                if cov_name not in new_metadata:
                    raise ValueError(
                        f"Covariate '{cov_name}' is missing for sample "
                        f"{sample.prompt_id} draw {sample.draw_idx}. Custom "
                        "covariates must be supplied in metadata (or as an "
                        "unknown top-level field, which ingestion promotes)."
                    )
                try:
                    value = float(new_metadata[cov_name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Covariate '{cov_name}' must be numeric for sample "
                        f"{sample.prompt_id}, got {new_metadata[cov_name]!r}."
                    ) from exc
                if not np.isfinite(value):
                    raise ValueError(
                        f"Covariate '{cov_name}' must be finite for sample "
                        f"{sample.prompt_id}."
                    )
                new_metadata[cov_name] = value

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
