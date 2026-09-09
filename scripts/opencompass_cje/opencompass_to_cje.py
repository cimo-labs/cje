#!/usr/bin/env python3
"""Convert OpenCompass evaluation output into CJE's `fresh_draws_data` format.

CJE expects:
  fresh_draws_data = {
    "policy_name": [
      {"prompt_id": "...", "judge_score": 0.83, "oracle_label": 0.9},
      ...
    ],
    ...
  }

OpenCompass can emit per-sample judge outputs (e.g. via `--dump-eval-details`).
In the OpenCompass docs, the example output is a JSON dict with a top-level
`details` list, where each record includes `origin_prompt` and `prediction`.

Unfortunately, the JSON schema can vary across evaluators and datasets, so this
script is intentionally *best-effort* and supports key overrides.

Supported (heuristics):
- Input is either:
    - a dict containing a list under `details` (or common variants), OR
    - a dict containing `details` as a dict keyed by string indices
      (``{"type": "GEN", "0": {...}, "1": {...}}``), which is the shape
      produced by OpenCompass's ``format_details()`` fallback, OR
    - a list of dict records.
- For each record, we try to extract:
    - prompt-like text (problem/question/prompt/origin_prompt/...)
    - judge decision (prediction/result/judgement/choice/...)

Judge score mapping (default):
- numeric -> float(value)
- strings:
    - yes/true/correct/pass/A -> 1.0
    - no/false/incorrect/fail/B -> 0.0
  Explicit verdicts such as "Answer: A" are supported; ambiguous prose is dropped.

Oracle labels (optional):
- Provide a CSV or JSONL file with at least:
    policy_name,prompt_id,response_id,oracle_label
  (extra columns ignored)

Outputs:
- `--out` writes CJE fresh_draws_data JSON.
- `--label-template` writes a CSV to help humans label an oracle slice.

NOTE: These bridge scripts live in the repo under `scripts/` and are not shipped
as part of the PyPI wheel.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cje_bridges._labels import (
    OracleLabels,
    ResponseIds,
    finite_number,
    first_identifier,
    load_oracle_labels,
)


_PROMPT_CANDIDATES = [
    "problem",
    "question",
    "prompt",
    "origin_prompt",
    "dialogue",
    "instruction",
    "input",
    "query",
]

_PRED_CANDIDATES = [
    "prediction",
    "predictions",
    "pred",
    "judge",
    "judge_prediction",
    "result",
    "choice",
    "judgement",
    "judgment",
    "score",
    "rating",
    "final_score",
]

_DETAILS_LIST_CANDIDATES = ["details", "detail", "records", "data"]
_NESTED_CONTAINER_CANDIDATES = ["result", "results", "eval", "evaluation"]

_AB_VERDICT_RE = re.compile(
    r"(?:(?:final\s+)?(?:answer|choice|decision|verdict|winner)\s*[:=]\s*)?"
    r"([AB]|\([AB]\)|\[[AB]\])[.!]?",
    re.IGNORECASE,
)


def _stable_json(obj: Any) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_inputs(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted([p for p in path.rglob("*.json") if p.is_file()])
    return [path]


def _pick_first(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _to_text(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return _stable_json(v)
    return str(v)


def _dict_keyed_to_list(d: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Convert ``{"type": "GEN", "0": {...}, "1": {...}}`` to ``[{...}, {...}]``.

    OpenCompass's ``format_details()`` in ``openicl_eval.py`` returns this
    shape: a dict with a ``type`` key (``"GEN"`` or ``"PPL"``) and per-sample
    entries keyed by string indices (``"0"``, ``"1"``, ...).

    Returns None if the dict doesn't match this pattern.
    """
    numeric_items = {k: v for k, v in d.items() if k.isdigit() and isinstance(v, dict)}
    if not numeric_items:
        return None
    return [numeric_items[k] for k in sorted(numeric_items, key=int)]


def _extract_details(payload: Any) -> List[Dict[str, Any]]:
    """Return per-sample detail dicts.

    Supported shapes:
    - {"details": [ {...}, ... ]}
    - {"details": {"type": "GEN", "0": {...}, ...}}  (format_details() output)
    - {"result": {"details": [...]}} (and common container variants)
    - {"records": [ ... ]}, {"data": [ ... ]}, etc.
    - [ {...}, {...} ]
    """

    # List payload is already the details list.
    if isinstance(payload, list) and all(isinstance(x, dict) for x in payload):
        return list(payload)

    if not isinstance(payload, dict):
        raise ValueError(
            "Unrecognized OpenCompass JSON structure (expected dict or list)"
        )

    for k in _DETAILS_LIST_CANDIDATES:
        v = payload.get(k)
        if isinstance(v, list) and all(isinstance(x, dict) for x in v):
            return list(v)
        # Handle format_details() dict-keyed-by-string-indices shape.
        if isinstance(v, dict):
            extracted = _dict_keyed_to_list(v)
            if extracted is not None:
                return extracted

    for container_key in _NESTED_CONTAINER_CANDIDATES:
        container = payload.get(container_key)
        if not isinstance(container, dict):
            continue
        for k in _DETAILS_LIST_CANDIDATES:
            v = container.get(k)
            if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                return list(v)
            if isinstance(v, dict):
                extracted = _dict_keyed_to_list(v)
                if extracted is not None:
                    return extracted

    raise ValueError(
        "Unrecognized OpenCompass JSON structure. Expected a dict containing a list under one of "
        f"{_DETAILS_LIST_CANDIDATES}, or a list of dicts."
    )


def _normalize_pred(v: Any) -> Optional[float]:
    if v is None:
        return None

    # NOTE: bool is a subclass of int; exclude explicitly.
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return finite_number(v, context="OpenCompass judge score")

    s = str(v).strip()
    if not s:
        return None

    s_lower = s.lower()

    # Common tokens
    if s_lower in {"a", "yes", "y", "true", "correct", "pass", "1"}:
        return 1.0
    if s_lower in {"b", "no", "n", "false", "incorrect", "fail", "0"}:
        return 0.0

    # Match the complete verdict, never an article or an option in explanation text.
    m = _AB_VERDICT_RE.fullmatch(s)
    if m:
        return 1.0 if m.group(1).strip("()[]").upper() == "A" else 0.0

    # Try parse numeric strings
    try:
        score = float(s_lower)
    except ValueError:
        return None
    return finite_number(score, context="OpenCompass judge score")


def _make_prompt_id(prompt_text: str, index: int, mode: str) -> str:
    if mode == "raw":
        return str(index)

    if mode == "hash":
        key = _stable_json({"prompt": prompt_text})
        return f"oc::{_hash_str(key)}"

    raise ValueError(f"Unknown prompt_id mode: {mode}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert OpenCompass per-sample eval JSON into CJE fresh_draws_data JSON",
    )
    ap.add_argument(
        "input",
        help="OpenCompass per-sample eval JSON file OR a directory containing *.json outputs",
    )
    ap.add_argument(
        "--out", default="cje_fresh_draws_data.json", help="Output JSON for CJE"
    )
    ap.add_argument(
        "--policy-name",
        default=None,
        help="Policy/model name. Default: inferred from the JSON filename stem.",
    )
    ap.add_argument(
        "--prompt-id-mode",
        choices=["hash", "raw"],
        default="hash",
        help="How to create prompt_id (default: hash prompt text; raw = use row index)",
    )
    ap.add_argument(
        "--prompt-field",
        default=None,
        help=f"Override prompt field key (default tries: {', '.join(_PROMPT_CANDIDATES)})",
    )
    ap.add_argument(
        "--prediction-field",
        default=None,
        help=f"Override prediction field key (default tries: {', '.join(_PRED_CANDIDATES)})",
    )
    ap.add_argument(
        "--oracle-labels",
        default=None,
        help="Optional oracle labels file (.csv or .jsonl) with policy_name,prompt_id,response_id,oracle_label",
    )
    ap.add_argument(
        "--label-template",
        default="oracle_label_template.csv",
        help="Write a CSV template for manual oracle labeling",
    )
    ap.add_argument(
        "--no-label-template",
        action="store_true",
        help="Do not write oracle label template CSV",
    )

    args = ap.parse_args()

    in_path = Path(args.input)
    inputs = _iter_inputs(in_path)
    if not inputs:
        print(f"No JSON files found under {in_path}", file=sys.stderr)
        return 2

    if not args.policy_name:
        by_stem: Dict[str, List[Path]] = {}
        for path in inputs:
            by_stem.setdefault(path.stem, []).append(path)
        for policy, paths in by_stem.items():
            if len(paths) > 1:
                raise ValueError(
                    f"Multiple input files would merge into policy {policy!r}: "
                    + ", ".join(str(path) for path in paths)
                    + ". Convert each model separately with --policy-name; "
                    "use one explicit --policy-name only for files from one model."
                )

    oracle_labels = (
        load_oracle_labels(Path(args.oracle_labels), ("policy_name", "policy"))
        if args.oracle_labels
        else OracleLabels()
    )

    fresh_draws: Dict[str, List[Dict[str, Any]]] = {}
    template_rows: List[Dict[str, Any]] = []
    response_ids = ResponseIds()
    source_prompt_counts: Counter[tuple[str, str]] = Counter()

    missing_prompt = 0
    missing_pred = 0
    total = 0

    for path in inputs:
        payload = _load_json(path)
        details = _extract_details(payload)

        policy_name = args.policy_name or path.stem
        out_list = fresh_draws.setdefault(policy_name, [])

        for i, d in enumerate(details):
            total += 1

            if not isinstance(d, dict):
                continue

            prompt_key = args.prompt_field
            pred_key = args.prediction_field

            prompt_val = (
                d.get(prompt_key) if prompt_key else _pick_first(d, _PROMPT_CANDIDATES)
            )
            pred_val = d.get(pred_key) if pred_key else _pick_first(d, _PRED_CANDIDATES)

            prompt_text = _to_text(prompt_val)
            if not prompt_text:
                missing_prompt += 1
                continue

            prompt_id = _make_prompt_id(prompt_text, i, args.prompt_id_mode)
            source_prompt_counts[(policy_name, prompt_id)] += 1
            judge_score = _normalize_pred(pred_val)
            if judge_score is None:
                missing_pred += 1
                continue

            response_id = response_ids.create(
                policy_name,
                prompt_id,
                {
                    "file": (
                        str(path.relative_to(in_path))
                        if in_path.is_dir()
                        else path.name
                    ),
                    "record": d,
                },
                native_id=first_identifier(d.get("response_id"), d.get("record_id")),
            )

            row: Dict[str, Any] = {
                "prompt_id": prompt_id,
                "response_id": response_id,
                "judge_score": judge_score,
            }

            out_list.append(row)

            template_rows.append(
                {
                    "policy_name": policy_name,
                    "prompt_id": prompt_id,
                    "response_id": response_id,
                    "judge_score": judge_score,
                    "prompt": (
                        (prompt_text[:300] + "...")
                        if len(prompt_text) > 300
                        else prompt_text
                    ),
                    "oracle_label": row.get("oracle_label", ""),
                }
            )

    oracle_labels.apply(fresh_draws, source_prompt_counts=source_prompt_counts)
    by_response = {
        (policy, sample["response_id"]): sample.get("oracle_label", "")
        for policy, samples in fresh_draws.items()
        for sample in samples
    }
    for row in template_rows:
        row["oracle_label"] = by_response[(row["policy_name"], row["response_id"])]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(fresh_draws, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    if not args.no_label_template:
        lt_path = Path(args.label_template)
        lt_path.parent.mkdir(parents=True, exist_ok=True)
        with lt_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "policy_name",
                    "prompt_id",
                    "response_id",
                    "judge_score",
                    "prompt",
                    "oracle_label",
                ],
            )
            writer.writeheader()
            writer.writerows(template_rows)

    print(f"Wrote CJE fresh_draws_data: {out_path}")
    if not args.no_label_template:
        print(f"Wrote oracle label template: {args.label_template}")

    kept = sum(len(v) for v in fresh_draws.values())
    print(
        f"Parsed {total} rows. Kept {kept}. Dropped missing_prompt={missing_prompt}, missing_pred={missing_pred}."
    )

    if total > 0 and kept == 0:
        print(
            "WARNING: All rows were dropped. The input schema may not match "
            "any known OpenCompass format. Try --prompt-field and "
            "--prediction-field overrides, or inspect the input JSON to "
            "identify the correct field names.",
            file=sys.stderr,
        )
        return 1

    if total > 0 and kept / total < 0.5:
        print(
            f"WARNING: {total - kept}/{total} rows dropped ({100 * (total - kept) / total:.0f}%). "
            "Consider using --prompt-field / --prediction-field overrides.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
