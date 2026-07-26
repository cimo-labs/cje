"""Data validation utilities for Direct-mode CJE.

Validates RAW parsed JSONL records (lists of dicts, exactly as
``json.loads`` produces them) before any Dataset/FreshDrawDataset
conversion.

Operating on raw records is deliberate: the historical ``cje validate``
bug (0.2.x through 0.3.0) came from round-tripping records through the
Dataset model before validating. That round-trip replaces missing
top-level ``prompt_id`` with generated hashes, relocates judge/oracle
fields between metadata and the top level, and historically required
now-removed OPE logprob fields — so validation reported "Missing required
field: prompt_id" and "No evaluation field found" on perfectly valid data.
"""

import logging
import math
from typing import Any, Dict, List, Tuple

import numpy as np

from .ingest import read_aliased_field
from .normalization import (
    ScaleDeclaration,
    coerce_scale,
    unit_scale,
    validate_values_on_scale,
)

logger = logging.getLogger(__name__)

#: Entries in the returned issues list that start with this prefix are
#: informational notes and do not affect validity.
NOTE_PREFIX = "Note:"

# Logged-data logprob fields. Direct mode (0.4.0) does not consume them;
# their presence earns an informational note, never an issue.
_LOGPROB_FIELDS = (
    "base_policy_logprob",
    "target_policy_logprobs",
    "logprob",
    "logprobs",
    "total_logprob",
    "token_logprobs",
)

# Cap for the logprob-presence scan (record shapes are uniform in
# practice, so the leading records are representative).
_LOGPROB_SCAN_LIMIT = 100


def read_record_field(record: Dict[str, Any], field: str) -> Any:
    """Read a field from the top level first, then from metadata.

    Delegates to the shared ingest helper — the loaders use the same
    lookup, so validation accepts exactly the shapes the loaders accept.
    Falsy-but-valid values (0, "") are returned as-is; only absent/None
    counts as missing.
    """
    return read_aliased_field(record, field)


def _is_numeric(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_direct_data(
    records: List[Dict[str, Any]],
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
    judge_scale: ScaleDeclaration = None,
    oracle_scale: ScaleDeclaration = None,
) -> Tuple[bool, List[str]]:
    """Validate raw Direct-mode records (fresh draws or calibration data).

    Checks, on the raw parsed JSONL dicts:

    - ``prompt_id`` present (top level or metadata) on every record;
    - a finite numeric judge score (top level or metadata) on every record;
    - declared judge/oracle scales contain every finite observed value;
      calibration-source files without declarations use the analysis
      pipeline's [0, 1] defaults, while fresh draws retain observed-range
      compatibility;
    - oracle-label support over ALL records: fewer than 4 independent
      prompt clusters earns an informational note because analysis can still
      return the explicit UNCALIBRATED/raw-judge tier; otherwise fewer than
      50 labels earns a small-sample note (50-100+ recommended);
    - if ``target_policy`` is present, every record must have it, and the
      window/oracle checks run per policy;
    - NO logprob checks: logprob fields are ignored by Direct mode, and
      their presence earns only an informational note.

    Args:
        records: Raw parsed JSONL records (list of dicts).
        judge_field: Field containing judge scores.
        oracle_field: Field containing oracle labels.
        judge_scale: Optional declared minimum/maximum judge-score scale.
        oracle_scale: Optional declared minimum/maximum oracle-label scale.

    Returns:
        Tuple of (is_valid, issues). Entries starting with ``"Note:"``
        are informational and do not affect ``is_valid``.

    Example:
        >>> records = [json.loads(line) for line in open("draws.jsonl")]
        >>> is_valid, issues = validate_direct_data(records)
        >>> if not is_valid:
        ...     for issue in issues:
        ...         print(f"⚠️  {issue}")
    """
    issues: List[str] = []
    notes: List[str] = []
    field_conflict = object()
    reported_conflicts = set()

    def _safe_field(
        record: Dict[str, Any], field: str, index: int, label: str = ""
    ) -> Any:
        try:
            return read_record_field(record, field)
        except ValueError as exc:
            key = (id(record), field)
            if key not in reported_conflicts:
                reported_conflicts.add(key)
                issues.append(
                    f"Conflicting field '{field}' at record {index}{label}: {exc}"
                )
            return field_conflict

    if not records:
        return False, ["Data is empty"]

    try:
        declared_judge_scale = coerce_scale(judge_scale, field_name="judge_scale")
        judge_scale_valid = True
    except ValueError as exc:
        issues.append(str(exc))
        declared_judge_scale = None
        judge_scale_valid = False
    try:
        declared_oracle_scale = coerce_scale(oracle_scale, field_name="oracle_scale")
        oracle_scale_valid = True
    except ValueError as exc:
        issues.append(str(exc))
        declared_oracle_scale = None
        oracle_scale_valid = False

    n = len(records)

    # --- target_policy consistency (refuse to guess for mixed files)
    n_with_policy = sum(
        1 for record in records if record.get("target_policy") is not None
    )
    if 0 < n_with_policy < n:
        issues.append(
            f"{n - n_with_policy}/{n} records are missing 'target_policy' "
            f"while other records have it. Give every record a "
            f"target_policy, or none (per-policy files)."
        )

    # Group per policy when target_policy is present; otherwise validate
    # all records as one group.
    groups: Dict[str, List[Dict[str, Any]]] = {}
    if n_with_policy:
        for record in records:
            policy = record.get("target_policy")
            key = str(policy) if policy is not None else "<missing target_policy>"
            groups.setdefault(key, []).append(record)
    else:
        groups[""] = list(records)

    judge_values: List[float] = []

    # --- full scan per group: prompt_id + numeric judge scores
    for policy, group in sorted(groups.items()):
        label = f" for policy '{policy}'" if policy else ""
        window = group

        n_missing_prompt_id = 0
        for i, record in enumerate(window):
            prompt_value = _safe_field(record, "prompt_id", i, label)
            if prompt_value is None:
                n_missing_prompt_id += 1
        if n_missing_prompt_id:
            issues.append(
                f"Missing required field 'prompt_id' in "
                f"{n_missing_prompt_id}/{len(window)} records{label} "
                f"(checked top level and "
                f"metadata). Direct mode aligns draws across policies by "
                f"prompt_id."
            )

        n_missing_judge = 0
        invalid_judge: List[Tuple[int, str]] = []
        for i, record in enumerate(window):
            value = _safe_field(record, judge_field, i, label)
            if value is field_conflict:
                continue
            if value is None:
                n_missing_judge += 1
            elif not _is_numeric(value):
                invalid_judge.append((i, type(value).__name__))
            else:
                judge_values.append(float(value))
        if n_missing_judge:
            issues.append(
                f"Judge field '{judge_field}' is missing in "
                f"{n_missing_judge}/{len(window)} records{label} "
                f"(checked top level and "
                f"metadata). Every record needs a numeric judge score."
            )
        if invalid_judge:
            issues.append(
                f"Judge field '{judge_field}' has non-numeric values{label}. "
                f"Examples (index, type): {invalid_judge[:3]}. "
                f"Values must be numeric (int or float)."
            )

    # --- oracle labels: full scan; count ladders on the pooled total
    # (calibration pools oracle labels across all policies' draws)
    oracle_counts: Dict[str, int] = {policy: 0 for policy in groups}
    oracle_prompt_clusters = set()
    oracle_values: List[float] = []
    invalid_oracle: List[Tuple[int, str]] = []
    for policy, group in groups.items():
        for i, record in enumerate(group):
            label = f" for policy '{policy}'" if policy else ""
            value = _safe_field(record, oracle_field, i, label)
            if value is field_conflict:
                continue
            if value is None:
                continue
            if _is_numeric(value):
                oracle_counts[policy] += 1
                oracle_values.append(float(value))
                prompt_id = _safe_field(record, "prompt_id", i, label)
                if prompt_id is not None and prompt_id is not field_conflict:
                    oracle_prompt_clusters.add(str(prompt_id))
            else:
                invalid_oracle.append((i, type(value).__name__))

    if invalid_oracle:
        issues.append(
            f"Oracle field '{oracle_field}' has non-numeric values. "
            f"Examples (index, type): {invalid_oracle[:3]}. "
            f"Values must be numeric (int or float)."
        )

    # A calibration file (no target_policy) follows DatasetLoader's explicit
    # unit-scale default. Fresh draws preserve the high-level API's legacy
    # observed-range inference unless the caller declares a scale.
    is_calibration_source = n_with_policy == 0
    resolved_judge_scale = declared_judge_scale or (
        unit_scale() if is_calibration_source else None
    )
    resolved_oracle_scale = declared_oracle_scale or (
        unit_scale() if is_calibration_source else None
    )
    for (
        field,
        values,
        scale,
        declaration,
        declaration_name,
        validation_flag,
        analysis_argument,
        declaration_valid,
    ) in (
        (
            judge_field,
            judge_values,
            resolved_judge_scale,
            declared_judge_scale,
            "judge_scale",
            "--judge-scale",
            "calibration_judge_scale",
            judge_scale_valid,
        ),
        (
            oracle_field,
            oracle_values,
            resolved_oracle_scale,
            declared_oracle_scale,
            "oracle_scale",
            "--oracle-scale",
            "calibration_oracle_scale",
            oracle_scale_valid,
        ),
    ):
        if scale is None or not declaration_valid:
            continue
        try:
            validate_values_on_scale(
                np.asarray(values, dtype=float), scale, field_name=field
            )
        except ValueError as exc:
            if declaration is None and is_calibration_source:
                issues.append(
                    f"{exc} Calibration sources default to [0, 1]. "
                    f"Declare {declaration_name}=(minimum, maximum) when "
                    f"validating (CLI: {validation_flag} MIN MAX), and pass "
                    f"{analysis_argument}=(minimum, maximum) to analysis."
                )
            else:
                issues.append(str(exc))

    total_oracle = sum(oracle_counts.values())
    if total_oracle == 0:
        notes.append(
            f"{NOTE_PREFIX} No oracle labels found in field '{oracle_field}'. "
            f"Analysis can still return explicitly UNCALIBRATED raw judge-score "
            f"means; add labels from at least 4 independent prompt clusters or "
            f"use --calibration-data for calibrated oracle-scale claims."
        )
    elif len(oracle_prompt_clusters) < 4:
        notes.append(
            f"{NOTE_PREFIX} Too few independent oracle prompt clusters "
            f"({len(oracle_prompt_clusters)} from {total_oracle} labeled rows). "
            f"Analysis will use the explicitly UNCALIBRATED raw-judge tier "
            f"unless every evaluation row has an oracle label; cross-fitted "
            f"calibration needs at least 4 independent clusters."
        )
    elif total_oracle < 50:
        notes.append(
            f"{NOTE_PREFIX} Found {total_oracle} oracle samples across "
            f"{len(oracle_prompt_clusters)} independent prompt clusters. "
            f"Consider adding more (50-100 labels recommended) for better "
            f"calibration."
        )

    # --- logprob fields: informational only, never an issue
    if any(
        field in record
        for record in records[:_LOGPROB_SCAN_LIMIT]
        for field in _LOGPROB_FIELDS
    ):
        notes.append(f"{NOTE_PREFIX} logprob fields present and ignored (Direct mode)")

    is_valid = len(issues) == 0
    return is_valid, issues + notes
