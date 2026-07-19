"""Data loading utilities following SOLID principles.

This module separates data loading concerns from the Dataset model,
following the Single Responsibility Principle.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

from .ingest import (
    canonicalize_record,
    deduplicate_canonical_records,
    require_prompt_identity,
)
from .models import Dataset, Sample
from .fresh_draws import FreshDrawDataset, fresh_draws_from_dict
from .normalization import (
    ScaleDeclaration,
    ScaleInfo,
    coerce_scale,
    unit_scale,
    validate_values_on_scale,
)

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads and converts raw data into typed Dataset objects.

    Follows Single Responsibility Principle - only handles data loading and conversion.

    Invalid records (corrupt JSON lines included) raise by default with
    file/line context. Pass ``on_invalid="drop"`` explicitly to filter them
    instead; drops are warned about with counts and recorded in the
    resulting Dataset's ``metadata["n_invalid_dropped"]``. ``strict`` is
    retained for API compatibility but no longer relaxes the default.
    """

    def __init__(
        self,
        prompt_field: str = "prompt",
        response_field: str = "response",
        reward_field: str = "reward",
        judge_field: str = "judge_score",
        oracle_field: str = "oracle_label",
        judge_scale: ScaleDeclaration = None,
        oracle_scale: ScaleDeclaration = None,
        strict: bool = False,
        on_invalid: Optional[str] = None,
    ):
        self.prompt_field = prompt_field
        self.response_field = response_field
        self.reward_field = reward_field
        self.judge_field = judge_field
        self.oracle_field = oracle_field
        self.judge_scale = (
            coerce_scale(judge_scale, field_name="calibration_judge_scale")
            or unit_scale()
        )
        self.oracle_scale = (
            coerce_scale(oracle_scale, field_name="calibration_oracle_scale")
            or unit_scale()
        )
        # Loud by default: only an explicit on_invalid="drop" filters
        # invalid records (strict is a compatibility no-op — the default
        # already errors).
        self.on_invalid = on_invalid or "error"
        if self.on_invalid not in {"error", "drop"}:
            raise ValueError("on_invalid must be 'error' or 'drop'")
        self.normalization_info: Dict[str, Any] = {
            "judge_score": self.judge_scale.to_dict(),
            "oracle_label": self.oracle_scale.to_dict(),
            "judge_field": judge_field,
            "oracle_field": oracle_field,
        }
        self._source_id = "in_memory:dataset"
        # Corrupt/non-object JSONL lines dropped before record conversion
        # (explicit on_invalid="drop" only); counted into n_invalid_dropped.
        self._n_source_dropped = 0

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
        self._source_id = str(Path(file_path).resolve())
        self._n_source_dropped = 0
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        if self.on_invalid == "error":
                            raise ValueError(
                                f"Invalid JSON at {file_path}:{line_num}: {exc}"
                            ) from exc
                        logger.warning(
                            "Dropping invalid JSON record at %s:%d: %s",
                            file_path,
                            line_num,
                            exc,
                        )
                        self._n_source_dropped += 1
                        continue
                    if not isinstance(record, dict):
                        if self.on_invalid == "error":
                            raise ValueError(
                                f"Expected JSON object at {file_path}:{line_num}, "
                                f"got {type(record).__name__}"
                            )
                        self._n_source_dropped += 1
                        continue
                    record["_cje_source_id"] = self._source_id
                    record["_cje_row_id"] = f"{self._source_id}:line:{line_num}"
                    data.append(record)
        return self._convert_raw_data(data, target_policies)

    def _check_score_ranges(self, data: List[Dict[str, Any]]) -> None:
        """Hard-fail on judge/oracle values outside [0, 1].

        Out-of-range records used to be silently FILTERED with only a
        warning, so a 0-100-scale calibration file died later with an
        unhelpful "No valid samples" error. A wrong scale is a dataset-level
        problem, not a per-record one: fail loudly with the observed range
        and the fix.
        """
        judge_values = np.asarray([r["judge_score"] for r in data], dtype=float)
        try:
            validate_values_on_scale(
                judge_values, self.judge_scale, field_name=self.judge_field
            )
        except ValueError as exc:
            if self.judge_scale.is_identity():
                bad = (judge_values < 0) | (judge_values > 1)
                raise ValueError(
                    f"Calibration data {self.judge_field} values outside [0, 1] "
                    f"(observed range {judge_values.min()}-{judge_values.max()}, "
                    f"{int(np.sum(bad))} rows). Rescale to [0, 1], or declare "
                    "calibration_judge_scale=(minimum, maximum). "
                    "fresh_draws_data continues to support observed-range "
                    "normalization for compatibility."
                ) from exc
            raise
        oracle_values = np.asarray(
            [r["oracle_label"] for r in data if r.get("oracle_label") is not None],
            dtype=float,
        )
        try:
            validate_values_on_scale(
                oracle_values, self.oracle_scale, field_name=self.oracle_field
            )
        except ValueError as exc:
            if self.oracle_scale.is_identity():
                bad = (oracle_values < 0) | (oracle_values > 1)
                raise ValueError(
                    f"Calibration data {self.oracle_field} values outside [0, 1] "
                    f"(observed range {oracle_values.min()}-{oracle_values.max()}, "
                    f"{int(np.sum(bad))} rows). Rescale to [0, 1], or declare "
                    "calibration_oracle_scale=(minimum, maximum). "
                    "fresh_draws_data continues to support observed-range "
                    "normalization for compatibility."
                ) from exc
            raise

    def _convert_raw_data(
        self, data: List[Dict[str, Any]], target_policies: Optional[List[str]] = None
    ) -> Dataset:
        """Convert raw data to Dataset."""
        canonical_records: List[Dict[str, Any]] = []
        n_skipped = 0
        for idx, record in enumerate(data):
            try:
                canonical_records.append(
                    canonicalize_record(
                        record,
                        idx,
                        source_id=self._source_id,
                        judge_field=self.judge_field,
                        oracle_field=self.oracle_field,
                        prompt_field=self.prompt_field,
                        response_field=self.response_field,
                    )
                )
            except (KeyError, TypeError, ValueError) as e:
                if self.on_invalid == "error":
                    raise ValueError(f"Invalid calibration record {idx}: {e}") from e
                n_skipped += 1
                logger.warning(f"Skipping invalid record {idx}: {e}")
                continue

        if not canonical_records:
            raise ValueError(
                f"No valid samples could be created from data: all "
                f"{len(data)} records were skipped due to validation errors. "
                f"See the warnings above for per-record reasons."
            )

        canonical_records, n_duplicates = deduplicate_canonical_records(
            canonical_records, context="calibration"
        )

        self._check_score_ranges(canonical_records)
        for record in canonical_records:
            record["judge_score"] = self.judge_scale.normalize(
                float(record["judge_score"])
            )
            if record.get("oracle_label") is not None:
                record["oracle_label"] = self.oracle_scale.normalize(
                    float(record["oracle_label"])
                )

        samples: List[Sample] = []
        for idx, record in enumerate(canonical_records):
            try:
                samples.append(
                    self._convert_record_to_sample(record, idx, canonical=True)
                )
            except (KeyError, TypeError, ValueError) as exc:
                if self.on_invalid == "error":
                    raise ValueError(
                        f"Invalid calibration record {idx}: {exc}"
                    ) from exc
                n_skipped += 1
                logger.warning("Skipping invalid record %d: %s", idx, exc)

        if not samples:
            raise ValueError(
                f"No valid samples could be created from data: all "
                f"{len(data)} records were skipped due to validation errors. "
                f"See the warnings above for per-record reasons."
            )

        n_dropped_total = n_skipped + self._n_source_dropped
        if n_dropped_total:
            logger.warning(
                f"Skipped {n_dropped_total}/{len(data) + self._n_source_dropped} "
                f"invalid records while loading dataset"
            )

        return Dataset(
            samples=samples,
            target_policies=target_policies or [],
            metadata={
                "source": "loader",
                "source_id": self._source_id,
                "normalization": self.normalization_info,
                "n_invalid_dropped": n_dropped_total,
                "n_duplicate_rows": n_duplicates,
            },
        )

    def _convert_record_to_sample(
        self, record: Dict[str, Any], idx: int = 0, canonical: bool = False
    ) -> Sample:
        """Convert a single record to a Sample.

        Args:
            record: Raw data record
            idx: Index in dataset (used as fallback if prompt is also missing)
        """
        if not canonical:
            record = canonicalize_record(
                record,
                idx,
                source_id=self._source_id,
                judge_field=self.judge_field,
                oracle_field=self.oracle_field,
                prompt_field=self.prompt_field,
                response_field=self.response_field,
            )
        prompt_id = record["prompt_id"]

        metadata = dict(record.get("metadata", {}))

        # Extract reward if present (handle nested format and custom aliases)
        reward = None
        if self.reward_field in record or self.reward_field in metadata:
            reward = record.get(self.reward_field, metadata.get(self.reward_field))
            if isinstance(reward, dict):
                reward = reward.get("mean", reward.get("value"))
            if reward is not None:
                reward = float(reward)

        # Extract judge_score and oracle_label - prioritize top-level, fallback to metadata
        judge_score = float(record["judge_score"])
        oracle_label = record.get("oracle_label")

        # Create Sample object with judge_score and oracle_label as top-level
        # fields. prompt/response default to "" so minimal judge+oracle
        # calibration files ({prompt_id, judge_score, oracle_label}) load —
        # calibration needs neither text field.
        return Sample(
            prompt_id=prompt_id,
            prompt=record.get("prompt", ""),
            response=record.get("response") or "",
            reward=reward,
            judge_score=judge_score,
            oracle_label=oracle_label,
            metadata=metadata,
            row_id=record.get("row_id"),
            observation_id=record.get("observation_id"),
            source_id=record.get("source_id"),
        )


class FreshDrawLoader:
    """Loader for fresh draw samples used in Direct-mode estimation."""

    @staticmethod
    def load_from_jsonl(
        path: str,
        judge_field: str = "judge_score",
        oracle_field: str = "oracle_label",
        on_invalid: str = "error",
    ) -> Dict[str, FreshDrawDataset]:
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
        if on_invalid not in {"error", "drop"}:
            raise ValueError("on_invalid must be 'error' or 'drop'")

        # Group samples by policy. Invalid lines raise with file/line
        # context (mirroring load_fresh_draws_auto's loud behavior) instead
        # of being silently skipped; blank/whitespace-only lines (e.g. a
        # trailing newline) are skipped, not errors.
        records_by_policy: Dict[str, List[Dict[str, Any]]] = {}

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

                    raw_policy = data.get("target_policy")
                    if raw_policy is None:
                        raise ValueError("missing required field 'target_policy'")
                    policy_name = str(raw_policy)
                    records_by_policy.setdefault(policy_name, [])
                    # Same prompt-identity contract as every other fresh-draw
                    # path: prompt_id, or prompt text to hash it from.
                    require_prompt_identity(data)
                    data["_cje_source_id"] = str(path_obj.resolve())
                    data["_cje_row_id"] = f"{path_obj.resolve()}:line:{line_num}"
                    canonical = canonicalize_record(
                        data,
                        line_num - 1,
                        source_id=str(path_obj.resolve()),
                        policy=policy_name,
                        judge_field=judge_field,
                        oracle_field=oracle_field,
                    )
                    validate_values_on_scale(
                        np.asarray([canonical["judge_score"]], dtype=float),
                        unit_scale(),
                        field_name=judge_field,
                    )
                    if canonical.get("oracle_label") is not None:
                        validate_values_on_scale(
                            np.asarray([canonical["oracle_label"]], dtype=float),
                            unit_scale(),
                            field_name=oracle_field,
                        )
                except KeyError as e:
                    raise ValueError(
                        f"Invalid fresh draw record at {path_obj}:{line_num}: "
                        f"missing required field {e}"
                    ) from e
                except ValueError as e:
                    # Covers json.JSONDecodeError and pydantic ValidationError
                    # (both subclass ValueError). Add file/line context.
                    if on_invalid == "error":
                        raise ValueError(
                            f"Invalid fresh draw record at {path_obj}:{line_num}: {e}"
                        ) from e
                    logger.warning(
                        "Dropping invalid fresh draw record at %s:%d: %s",
                        path_obj,
                        line_num,
                        e,
                    )
                    continue

                records_by_policy[policy_name].append(canonical)

        if not records_by_policy:
            raise ValueError(f"No valid fresh-draw records found in {path_obj}")
        empty_policies = sorted(
            policy for policy, records in records_by_policy.items() if not records
        )
        if empty_policies:
            raise ValueError(
                "No valid fresh-draw records remain for policy/policies: "
                + ", ".join(empty_policies)
            )
        datasets, _ = fresh_draws_from_dict(
            records_by_policy,
            auto_normalize=False,
            judge_field=judge_field,
            oracle_field=oracle_field,
            on_invalid=on_invalid,
        )
        return datasets
