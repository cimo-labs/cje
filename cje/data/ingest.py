"""Shared ingestion helpers for Direct-mode data loading.

Single source of truth for the record- and file-level conventions that were
previously duplicated (and had drifted apart) across loaders.py,
fresh_draws.py, validation.py, and the CLI:

- ``POLICY_FILE_PATTERNS`` / ``resolve_policy_file``: the canonical
  per-policy fresh-draws filename patterns. Discovery
  (``discover_policies_from_fresh_draws``), directory loading, and the CLI
  all resolve through this list, so anything discovered is guaranteed to
  load and vice versa.
- ``read_aliased_field`` / ``resolve_prompt_id``: record-level field
  aliasing (top level first, then metadata) and the review-hardened
  prompt_id fallback semantics (explicit None checks so falsy-but-valid
  ids like 0 or "" are never hash-replaced).
- ``read_jsonl_records`` / ``fresh_draws_data_from_file``: raw JSONL
  reading and the single-file multi-policy fresh-draws format (records
  grouped by their ``target_policy`` field).
"""

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


#: Canonical per-policy fresh-draws filename patterns, in resolution order.
#: ``{policy}`` is substituted with the policy name; patterns may include a
#: subdirectory. 0.4.x also accepted the never-documented
#: ``{policy}_fresh.jsonl`` and ``responses/{policy}_responses.jsonl``;
#: both were dropped when discovery and loading were unified on this list.
POLICY_FILE_PATTERNS: Tuple[str, ...] = (
    "{policy}_responses.jsonl",
    "{policy}.jsonl",
    "responses/{policy}.jsonl",
    "fresh_draws/{policy}.jsonl",
)

#: Pseudo-policy key used in ``drop_stats`` for dropped rows that cannot be
#: attributed to a policy (malformed JSON lines, records missing
#: ``target_policy``).
UNATTRIBUTED_POLICY_KEY = "(unattributed)"


def resolve_policy_file(
    dir_path: Path,
    policy: str,
    *,
    exclude_paths: Optional[List[Path]] = None,
) -> Optional[Path]:
    """Locate the fresh-draws file for a policy in a fresh-draws directory.

    Returns the unique existing match, ``None`` if no pattern matches, and
    raises when multiple documented patterns resolve to the same policy.
    """
    dir_path = Path(dir_path)
    excluded = {Path(path).resolve() for path in (exclude_paths or [])}
    matches = []
    for pattern in POLICY_FILE_PATTERNS:
        candidate = dir_path / pattern.format(policy=policy)
        if (
            candidate.exists()
            and candidate.resolve() not in excluded
            and candidate not in matches
        ):
            matches.append(candidate)
    if len(matches) > 1:
        formatted = ", ".join(str(path) for path in matches)
        raise ValueError(
            f"Multiple fresh-draw files resolve to policy '{policy}': "
            f"{formatted}. Keep exactly one file per policy."
        )
    return matches[0] if matches else None


def read_aliased_field(record: Dict[str, Any], field: str) -> Any:
    """Read a field from the top level first, then from metadata.

    This is the lookup order shared by every loader (and mirrored by
    validation), so all entry points accept exactly the same record shapes.
    Falsy-but-valid values (0, "") are returned as-is; only absent/None
    counts as missing.
    """
    if field.startswith("metadata."):
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            return None
        return metadata.get(field.split(".", 1)[1])

    value = record.get(field)
    metadata = record.get("metadata")
    metadata_value = metadata.get(field) if isinstance(metadata, dict) else None
    if value is not None and metadata_value is not None and value != metadata_value:
        raise ValueError(
            f"Conflicting values for field '{field}' at the top level and in "
            f"metadata: {value!r} != {metadata_value!r}."
        )
    return value if value is not None else metadata_value


def require_prompt_identity(
    record: Dict[str, Any], prompt_field: str = "prompt"
) -> None:
    """Ensure a fresh-draw record carries a usable prompt identity.

    prompt_id is optional — it is auto-generated from the ``prompt`` text's
    hash when absent (the documented fallback, shared by every fresh-draw
    loader). A record with NEITHER prompt_id NOR prompt text cannot be
    aligned across policies, so it fails loudly instead of receiving a
    fabricated index-based id.
    """
    if read_aliased_field(record, "prompt_id") is not None:
        return
    if read_aliased_field(record, prompt_field):
        return
    raise ValueError(
        "missing required field 'prompt_id' (and no 'prompt' text to "
        "derive it from)."
    )


def canonicalize_record(
    record: Dict[str, Any],
    index: int,
    *,
    source_id: str,
    policy: Optional[str] = None,
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
    prompt_field: str = "prompt",
    response_field: str = "response",
) -> Dict[str, Any]:
    """Promote one raw record into the canonical Direct-mode field layout.

    Every loader calls this function. Custom score fields, nested metadata,
    covariates, source-local row identity, and filename/record policy checks
    therefore have identical semantics across file, directory, and in-memory
    inputs.
    """
    if not isinstance(record, dict):
        raise ValueError(f"expected a mapping, got {type(record).__name__}")

    judge_raw = read_aliased_field(record, judge_field)
    if judge_raw is None:
        raise ValueError(f"Missing required judge field '{judge_field}'.")
    if isinstance(judge_raw, bool):
        raise ValueError(f"Judge field '{judge_field}' must be numeric, not bool.")
    try:
        judge_score = float(judge_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Judge field '{judge_field}' must be numeric, got {judge_raw!r}."
        ) from exc
    if not math.isfinite(judge_score):
        raise ValueError(f"Judge field '{judge_field}' must be finite.")

    prompt_raw = read_aliased_field(record, prompt_field)
    response_raw = read_aliased_field(record, response_field)
    oracle_raw = read_aliased_field(record, oracle_field)
    oracle_label: Optional[float] = None
    if oracle_raw is not None:
        if isinstance(oracle_raw, bool):
            raise ValueError(
                f"Oracle field '{oracle_field}' must be numeric, not bool."
            )
        try:
            oracle_label = float(oracle_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Oracle field '{oracle_field}' must be numeric, got {oracle_raw!r}."
            ) from exc
        if not math.isfinite(oracle_label):
            raise ValueError(f"Oracle field '{oracle_field}' must be finite.")

    prompt_id = resolve_prompt_id(
        record, index, prompt_field=prompt_field, policy=policy
    )
    record_policy = read_aliased_field(record, "target_policy")
    if (
        policy is not None
        and record_policy is not None
        and str(record_policy) != policy
    ):
        raise ValueError(
            f"Record target_policy {record_policy!r} conflicts with source policy "
            f"{policy!r}."
        )
    target_policy = policy if policy is not None else record_policy

    nested = record.get("metadata")
    metadata: Dict[str, Any] = dict(nested) if isinstance(nested, dict) else {}
    core_fields = {
        "metadata",
        "prompt_id",
        prompt_field,
        response_field,
        judge_field,
        oracle_field,
        "judge_score",
        "oracle_label",
        "target_policy",
        "draw_idx",
        "row_id",
        "observation_id",
        "source_id",
        "_cje_row_id",
        "_cje_source_id",
        "_cje_line_num",
    }
    for key, value in record.items():
        if key in core_fields:
            continue
        if key in metadata and metadata[key] != value:
            raise ValueError(
                f"Conflicting values for metadata field '{key}': top-level "
                f"{value!r} != nested {metadata[key]!r}."
            )
        metadata[key] = value

    # Canonical output must be safe to canonicalize again (directory input is
    # validated once while reading and once when policies are assembled).
    # Remove nested copies of promoted fields so type normalization such as
    # ``"0.5" -> 0.5`` cannot look like a top-level/metadata conflict later.
    judge_metadata_key = (
        judge_field.split(".", 1)[1]
        if judge_field.startswith("metadata.")
        else judge_field
    )
    oracle_metadata_key = (
        oracle_field.split(".", 1)[1]
        if oracle_field.startswith("metadata.")
        else oracle_field
    )
    for promoted_key in {
        "prompt_id",
        prompt_field,
        response_field,
        "judge_score",
        "oracle_label",
        judge_metadata_key,
        oracle_metadata_key,
        "target_policy",
        "draw_idx",
        "row_id",
        "observation_id",
        "source_id",
    }:
        metadata.pop(promoted_key, None)

    # Keep custom aliases available in metadata for compatibility with the
    # lower-level calibration API while also promoting canonical fields.
    if judge_field != "judge_score":
        metadata[judge_metadata_key] = judge_score
    if oracle_field != "oracle_label" and oracle_raw is not None:
        metadata[oracle_metadata_key] = oracle_label
    if response_raw is None:
        metadata["_cje_response_missing"] = True

    # Metadata must survive the JSON-safe identity signature built by
    # deduplicate_canonical_records. Checking here keeps the failure inside
    # the per-record on_invalid contract with the offending key named,
    # instead of a bare serialization TypeError after loading.
    from .models import _json_safe

    for key, value in metadata.items():
        try:
            _json_safe(value)
        except TypeError as exc:
            raise ValueError(
                f"Metadata field {key!r} has a non-JSON-encodable value of "
                f"type {type(value).__name__}; metadata values must be "
                f"JSON-encodable."
            ) from exc

    explicit_source = read_aliased_field(record, "source_id")
    annotated_source = record.get("_cje_source_id")
    canonical_source_id = str(
        explicit_source
        if explicit_source is not None
        else annotated_source if annotated_source is not None else source_id
    )
    explicit_row = read_aliased_field(record, "row_id")
    annotated_row = record.get("_cje_row_id")
    row_id = str(
        explicit_row
        if explicit_row is not None
        else (
            annotated_row
            if annotated_row is not None
            else f"{canonical_source_id}:row:{index}"
        )
    )
    observation_id_raw = read_aliased_field(record, "observation_id")

    canonical: Dict[str, Any] = {
        "prompt_id": prompt_id,
        "prompt": prompt_raw or "",
        "judge_score": judge_score,
        "response": response_raw,
        "metadata": metadata,
        "row_id": row_id,
        "source_id": canonical_source_id,
    }
    if oracle_label is not None:
        canonical["oracle_label"] = oracle_label
    if "reward" in record:
        canonical["reward"] = record["reward"]
    if target_policy is not None:
        canonical["target_policy"] = str(target_policy)
    if "draw_idx" in record:
        canonical["draw_idx"] = record["draw_idx"]
    if observation_id_raw is not None:
        canonical["observation_id"] = str(observation_id_raw)
    return canonical


def deduplicate_canonical_records(
    records: List[Dict[str, Any]], *, context: str
) -> Tuple[List[Dict[str, Any]], int]:
    """Deduplicate only exact records sharing explicit/canonical row identity.

    Equal values with different row IDs are distinct observations. A repeated
    ``(source_id, row_id)`` is accepted only when every canonical field,
    including metadata/covariates, is identical; conflicting reuse is an error.
    """
    from .models import _json_safe

    unique: List[Dict[str, Any]] = []
    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    n_duplicates = 0
    for record in records:
        source_id = str(record["source_id"])
        row_id = str(record["row_id"])
        identity = (source_id, row_id)
        try:
            signature = _json_safe(
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"source_id", "row_id"}
                }
            )
        except TypeError as exc:
            # Backstop: canonicalize_record already rejects non-encodable
            # metadata per record; anything that slips through still gets
            # row context instead of a bare serialization TypeError.
            raise ValueError(
                f"Cannot build an identity signature for {context} record "
                f"row_id={row_id!r} in source {source_id!r}: {exc}"
            ) from exc
        previous = seen.get(identity)
        if previous is None:
            seen[identity] = signature
            unique.append(record)
            continue
        if previous != signature:
            raise ValueError(
                f"Conflicting {context} records share row_id {row_id!r} in "
                f"source {source_id!r}. Row identity must be unique."
            )
        n_duplicates += 1
    return unique, n_duplicates


def resolve_prompt_id(
    record: Dict[str, Any],
    index: int,
    prompt_field: str = "prompt",
    policy: Optional[str] = None,
) -> str:
    """Resolve a record's prompt_id: top level -> metadata -> auto-generate.

    Explicit None checks throughout: falsy but valid ids like 0 or "" must
    not be hash-replaced. Found ids are coerced to string so integer ids
    like 0 survive validation and join correctly across datasets.

    Auto-generation uses the prompt hash for consistency across datasets
    (fresh draws map to the same prompt_id as logged/calibration data),
    falling back to an index-based id when the prompt is missing too. The
    index fallback is caller-specific: fresh-draw loading (``policy``
    given) uses ``fresh_{policy}_{index:06d}``, dataset loading uses
    ``sample_{index:06d}`` — each with its original warning.

    Args:
        record: Raw parsed JSONL record.
        index: Record index (used as the last-resort fallback id).
        prompt_field: Field containing the prompt text.
        policy: Target policy name when loading per-policy fresh draws.

    Returns:
        The resolved prompt_id as a string.
    """
    prompt_id = read_aliased_field(record, "prompt_id")
    if prompt_id is not None:
        return str(prompt_id)

    prompt = read_aliased_field(record, prompt_field)
    if prompt:
        if not isinstance(prompt, str):
            raise ValueError(
                f"Prompt field '{prompt_field}' must be text, got "
                f"{type(prompt).__name__}."
            )
        # Use first 12 chars of SHA256 for readable but unique ID
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        return f"prompt_{prompt_hash}"

    if policy is not None:
        generated = f"fresh_{policy}_{index:06d}"
        logger.warning(
            f"Fresh draw record {index} for policy '{policy}' missing both "
            f"'prompt_id' and 'prompt'. Using index-based ID '{generated}'. "
            f"This will NOT align with logged data for DR mode. "
            f"Add explicit prompt_id or prompt text for stability."
        )
        return generated

    generated = f"sample_{index:06d}"
    logger.warning(
        f"Record {index} missing both 'prompt_id' and 'prompt'. "
        f"Using index-based ID '{generated}'. This is fragile - "
        f"consider adding explicit prompt_id or prompt text for stability."
    )
    return generated


def read_jsonl_records(
    path: Path,
    *,
    on_invalid: str = "error",
    drop_stats: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Read JSONL objects, either raising or dropping malformed records.

    When ``drop_stats`` is supplied, dropped lines are counted under
    ``UNATTRIBUTED_POLICY_KEY`` (a malformed line has no policy to charge).
    """
    if on_invalid not in {"error", "drop"}:
        raise ValueError("on_invalid must be 'error' or 'drop'")

    def _count_drop() -> None:
        if drop_stats is not None:
            drop_stats[UNATTRIBUTED_POLICY_KEY] = (
                drop_stats.get(UNATTRIBUTED_POLICY_KEY, 0) + 1
            )

    records = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                if on_invalid == "error":
                    raise ValueError(f"Invalid JSON at {path}:{line_num}: {e}") from e
                logger.warning(
                    "Dropping invalid JSON record at %s:%d: %s", path, line_num, e
                )
                _count_drop()
                continue
            if not isinstance(record, dict):
                message = (
                    f"Expected a JSON object at {path}:{line_num}, "
                    f"got {type(record).__name__}"
                )
                if on_invalid == "error":
                    raise ValueError(message)
                logger.warning("Dropping invalid record: %s", message)
                _count_drop()
                continue
            record["_cje_line_num"] = line_num
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


# Logged-data logprob fields (same list as validation.py and the CLI). In
# fresh draws they are ignored (Direct mode needs none); a file of records
# WITHOUT target_policy but WITH these fields is an 0.3.x logged dataset
# and gets the migration error.
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


def fresh_draws_data_from_file(
    path: Path,
    *,
    on_invalid: str = "error",
    drop_stats: Optional[Dict[str, int]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Group a single fresh-draws JSONL file by target_policy.

    A file whose records have logged-data logprob fields but no
    target_policy is an 0.3.x logged dataset: raise the migration error.
    When ``drop_stats`` is supplied, dropped rows are counted per policy
    (``UNATTRIBUTED_POLICY_KEY`` when the row's policy is unknowable).
    """
    from ..interface.analysis import LOGGED_DATA_PATH_REMOVED_MESSAGE

    records = read_jsonl_records(path, on_invalid=on_invalid, drop_stats=drop_stats)

    if not any("target_policy" in record for record in records):
        if any(
            field in record
            for record in records[:_LOGPROB_SCAN_LIMIT]
            for field in _LOGPROB_FIELDS
        ):
            # The old `cje analyze logged_data.jsonl` invocation.
            raise ValueError(LOGGED_DATA_PATH_REMOVED_MESSAGE)
        raise ValueError(
            f"{path} does not look like fresh draws: no record has a "
            f"'target_policy' field. Fresh-draws records need at least "
            f"prompt_id, judge_score, and target_policy — or pass a "
            f"directory of per-policy response files instead."
        )

    fresh_draws_data: Dict[str, List[Dict[str, Any]]] = {}
    source_id = str(path.resolve())
    for idx, record in enumerate(records):
        line_num = int(record.get("_cje_line_num", idx + 1))
        policy = record.get("target_policy")
        if policy is None:
            message = (
                f"Record at {path}:{line_num} is missing 'target_policy' "
                f"(other records have it — refusing to guess)."
            )
            if on_invalid == "error":
                raise ValueError(message)
            logger.warning("Dropping invalid fresh-draw record: %s", message)
            if drop_stats is not None:
                drop_stats[UNATTRIBUTED_POLICY_KEY] = (
                    drop_stats.get(UNATTRIBUTED_POLICY_KEY, 0) + 1
                )
            continue
        annotated = dict(record)
        annotated["_cje_source_id"] = source_id
        annotated["_cje_row_id"] = f"{source_id}:line:{line_num}"
        fresh_draws_data.setdefault(str(policy), []).append(annotated)
    if not fresh_draws_data:
        raise ValueError(f"No valid fresh-draw records found in {path}")
    return fresh_draws_data
