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


def resolve_policy_file(dir_path: Path, policy: str) -> Optional[Path]:
    """Locate the fresh-draws file for a policy in a fresh-draws directory.

    Returns the first existing match in ``POLICY_FILE_PATTERNS`` order, or
    None if no pattern matches.
    """
    dir_path = Path(dir_path)
    for pattern in POLICY_FILE_PATTERNS:
        candidate = dir_path / pattern.format(policy=policy)
        if candidate.exists():
            return candidate
    return None


def read_aliased_field(record: Dict[str, Any], field: str) -> Any:
    """Read a field from the top level first, then from metadata.

    This is the lookup order shared by every loader (and mirrored by
    validation), so all entry points accept exactly the same record shapes.
    Falsy-but-valid values (0, "") are returned as-is; only absent/None
    counts as missing.
    """
    value = record.get(field)
    if value is not None:
        return value
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(field)
    return None


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

    prompt = record.get(prompt_field, "")
    if prompt:
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


def read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dict records (blank lines skipped)."""
    records = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {e}") from e
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}:{line_num}, "
                    f"got {type(record).__name__}"
                )
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


def fresh_draws_data_from_file(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Group a single fresh-draws JSONL file by target_policy.

    A file whose records have logged-data logprob fields but no
    target_policy is an 0.3.x logged dataset: raise the migration error.
    """
    from ..interface.analysis import LOGGED_DATA_PATH_REMOVED_MESSAGE

    records = read_jsonl_records(path)

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
    for idx, record in enumerate(records):
        policy = record.get("target_policy")
        if not policy:
            raise ValueError(
                f"Record {idx} in {path} is missing 'target_policy' "
                f"(other records have it — refusing to guess)."
            )
        fresh_draws_data.setdefault(str(policy), []).append(record)
    return fresh_draws_data
