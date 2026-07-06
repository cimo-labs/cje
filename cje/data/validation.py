"""Data validation utilities for Direct-mode CJE.

Validates RAW parsed JSONL records (lists of dicts, exactly as
``json.loads`` produces them) before any Dataset/FreshDrawDataset
conversion.

Operating on raw records is deliberate: the historical ``cje validate``
bug (0.2.x through 0.3.0) came from round-tripping records through the
Dataset model before validating. That round-trip replaces missing
top-level ``prompt_id`` with generated hashes, relocates judge/oracle
fields between metadata and the top level, and requires OPE logprob
fields — so validation reported "Missing required field: prompt_id" and
"No evaluation field found" on perfectly valid data.
"""

from typing import Any, Dict, List, Tuple
import logging

from .ingest import read_aliased_field

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


def _window_size(n: int) -> int:
    """Window-scan size: min(100, max(10, n // 10))."""
    return min(100, max(10, n // 10))


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float))


def validate_direct_data(
    records: List[Dict[str, Any]],
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
) -> Tuple[bool, List[str]]:
    """Validate raw Direct-mode records (fresh draws or calibration data).

    Checks, on the raw parsed JSONL dicts:

    - ``prompt_id`` present (top level or metadata) over a leading window
      of min(100, max(10, n // 10)) records;
    - a numeric judge score (top level or metadata) over the same window;
    - oracle-label counts over ALL records: 0 or fewer than 10 valid
      labels is an issue, 10-49 earns an informational note (50-100+
      recommended);
    - if ``target_policy`` is present, every record must have it, and the
      window/oracle checks run per policy;
    - NO logprob checks: logprob fields are ignored by Direct mode, and
      their presence earns only an informational note.

    Args:
        records: Raw parsed JSONL records (list of dicts).
        judge_field: Field containing judge scores.
        oracle_field: Field containing oracle labels.

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

    if not records:
        return False, ["Data is empty"]

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

    # --- window scan per group: prompt_id + numeric judge scores
    for policy, group in sorted(groups.items()):
        label = f" for policy '{policy}'" if policy else ""
        window = group[: _window_size(len(group))]

        n_missing_prompt_id = sum(
            1 for record in window if read_record_field(record, "prompt_id") is None
        )
        if n_missing_prompt_id:
            issues.append(
                f"Missing required field 'prompt_id' in "
                f"{n_missing_prompt_id}/{len(window)} of the first "
                f"{len(window)} records{label} (checked top level and "
                f"metadata). Direct mode aligns draws across policies by "
                f"prompt_id."
            )

        n_missing_judge = 0
        invalid_judge: List[Tuple[int, str]] = []
        for i, record in enumerate(window):
            value = read_record_field(record, judge_field)
            if value is None:
                n_missing_judge += 1
            elif not _is_numeric(value):
                invalid_judge.append((i, type(value).__name__))
        if n_missing_judge:
            issues.append(
                f"Judge field '{judge_field}' is missing in "
                f"{n_missing_judge}/{len(window)} of the first "
                f"{len(window)} records{label} (checked top level and "
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
    invalid_oracle: List[Tuple[int, str]] = []
    for policy, group in groups.items():
        for i, record in enumerate(group):
            value = read_record_field(record, oracle_field)
            if value is None:
                continue
            if _is_numeric(value):
                oracle_counts[policy] += 1
            else:
                invalid_oracle.append((i, type(value).__name__))

    if invalid_oracle:
        issues.append(
            f"Oracle field '{oracle_field}' has non-numeric values. "
            f"Examples (index, type): {invalid_oracle[:3]}. "
            f"Values must be numeric (int or float)."
        )

    total_oracle = sum(oracle_counts.values())
    if total_oracle == 0:
        issues.append(
            f"No oracle labels found in field '{oracle_field}'. "
            f"Calibration needs at least 10 labeled samples (50-100 "
            f"recommended): add oracle labels to a slice of these records, "
            f"or calibrate from a separate file via calibration_data_path "
            f"(--calibration-data)."
        )
    elif total_oracle < 10:
        issues.append(
            f"Too few oracle samples ({total_oracle}). "
            f"Absolute minimum is 10 samples. "
            f"Strongly recommend 50-100+ for robust calibration."
        )
    elif total_oracle < 50:
        notes.append(
            f"{NOTE_PREFIX} Found {total_oracle} oracle samples. Consider "
            f"adding more (50-100 recommended) for better calibration."
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
