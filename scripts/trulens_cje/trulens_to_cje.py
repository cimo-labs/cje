#!/usr/bin/env python3
"""Convert TruLens records+feedback into CJE's `fresh_draws_data` format.

CJE expects:
  fresh_draws_data = {
    "policy_name": [
      {"prompt_id": "...", "judge_score": 0.83, "oracle_label": 0.9},
      ...
    ],
    ...
  }

TruLens (v1) can produce a DataFrame with records and their feedback results via:
  from trulens.core.session import TruSession
  df, feedback_cols = session.get_records_and_feedback(...)

This script bridges those two worlds so teams using TruLens feedback functions
(LLM-as-judge proxies) can calibrate + audit them with CJE using a small oracle slice.

Design notes:
- "policy" should be something like app version / app id. We default to `app_version`
  if present, else `app_id`, else user must supply `--policy-col`.
- `prompt_id` must be stable across policies. TruLens record_id is not stable across
  versions, so we default to hashing the record input text ("input"/"main_input").
  You can override with `--prompt-id-col`.

Oracle labels:
- Optional file (.csv or .jsonl) with at least:
    policy_id,prompt_id,response_id,oracle_label
  Extra columns are ignored.

Outputs:
- `--out` writes CJE fresh_draws_data JSON.
- `--label-template` writes a CSV template to label an oracle slice.

"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cje_bridges._labels import (
    OracleLabels,
    ResponseIds,
    finite_number,
    first_identifier,
    load_oracle_labels,
)


def _stable_json(obj: Any) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _pick_first_existing(cols: Iterable[str], candidates: List[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--database-url",
        default=None,
        help="SQLAlchemy-compatible URL for TruLens DB (e.g. sqlite:///default.sqlite). If omitted, TruSession default is used.",
    )
    ap.add_argument("--app-name", default=None)
    ap.add_argument("--app-version", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--limit", type=int, default=None)

    ap.add_argument(
        "--policy-col",
        default=None,
        help="Column name in records df that identifies the policy/app variant (default: auto: app_version, then app_id).",
    )
    ap.add_argument(
        "--prompt-id-col",
        default=None,
        help="Optional column to use directly as prompt_id (must be stable across policies). If omitted, hash of input is used.",
    )
    ap.add_argument(
        "--input-col",
        default=None,
        help="Column containing the record input text (default: auto: input, then main_input).",
    )
    ap.add_argument(
        "--output-col",
        default=None,
        help="Column containing the record output text (default: auto: output, then main_output).",
    )

    ap.add_argument(
        "--judge-col",
        required=True,
        help="Column name for the feedback score to treat as judge_score (must be numeric, typically 0-1).",
    )

    ap.add_argument("--out", default="cje_fresh_draws_data.json")
    ap.add_argument(
        "--oracle-labels",
        default=None,
        help="Optional oracle labels file (.csv or .jsonl) with policy_id,prompt_id,response_id,oracle_label",
    )
    ap.add_argument("--label-template", default="oracle_label_template.csv")
    ap.add_argument("--no-label-template", action="store_true")
    ap.add_argument(
        "--run-cje",
        action="store_true",
        help="If cje-eval is installed and oracle labels are present, run analyze_dataset and print a short summary.",
    )

    args = ap.parse_args()

    try:
        from trulens.core.session import TruSession  # type: ignore
    except Exception as e:
        print(
            "Failed to import trulens. Install it first (e.g., `pip install trulens`).\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        return 2

    session_kwargs: Dict[str, Any] = {}
    if args.database_url:
        # TruSession accepts **kwargs used to initialize DefaultDBConnector; database_url is one of them.
        session_kwargs["database_url"] = args.database_url

    session = TruSession(**session_kwargs)
    df, feedback_cols = session.get_records_and_feedback(
        app_name=args.app_name,
        app_version=args.app_version,
        run_name=args.run_name,
        limit=args.limit,
    )

    if args.judge_col not in df.columns:
        print(
            f"judge-col {args.judge_col!r} not found. Available feedback cols include: {feedback_cols}",
            file=sys.stderr,
        )
        return 2

    policy_col = args.policy_col or _pick_first_existing(
        df.columns, ["app_version", "app_id", "app_name"]
    )
    if policy_col is None:
        print(
            "Could not infer policy column. Pass --policy-col explicitly.",
            file=sys.stderr,
        )
        return 2

    input_col = args.input_col or _pick_first_existing(
        df.columns, ["input", "main_input"]
    )
    output_col = args.output_col or _pick_first_existing(
        df.columns, ["output", "main_output"]
    )

    oracle_labels = (
        load_oracle_labels(
            Path(args.oracle_labels), ("policy_id", "policy", "app_version", "app_id")
        )
        if args.oracle_labels
        else OracleLabels()
    )

    fresh_draws: Dict[str, List[Dict[str, Any]]] = {}
    template_rows: List[Dict[str, Any]] = []
    response_ids = ResponseIds()
    source_prompt_counts: Counter[tuple[str, str]] = Counter()

    missing = 0
    for _, row in df.iterrows():
        policy_id = row.get(policy_col)
        if policy_id is None:
            continue
        policy_id = str(policy_id)

        # Determine prompt_id
        if args.prompt_id_col:
            pid = row.get(args.prompt_id_col)
            if pid is None:
                continue
            prompt_id = str(pid)
        else:
            if input_col is None:
                print(
                    "No input column found to hash into prompt_id. Provide --prompt-id-col or --input-col.",
                    file=sys.stderr,
                )
                return 2
            inp = row.get(input_col)
            if inp is None:
                continue

            # IMPORTANT: prompt_id should be stable across runs/policies.
            # `str(dict)` / `str(list)` can be non-deterministic, so for structured inputs
            # we hash a stable JSON serialization instead.
            if isinstance(inp, (dict, list)):
                inp_s = _stable_json(inp)
            else:
                inp_s = str(inp)

            prompt_id = f"input::{_hash_str(inp_s)}"

        source_prompt_counts[(policy_id, prompt_id)] += 1
        judge_score = row.get(args.judge_col)
        # NOTE: bool is a subclass of int in Python; exclude it explicitly.
        if not isinstance(judge_score, (int, float)) or isinstance(judge_score, bool):
            missing += 1
            continue
        judge_score = finite_number(judge_score, context="TruLens judge score")
        record_id = first_identifier(row.get("record_id"), row.get("recordId"))
        output = row.get(output_col) if output_col else None
        response_id = response_ids.create(
            policy_id, prompt_id, output, native_id=record_id
        )

        sample: Dict[str, Any] = {
            "prompt_id": prompt_id,
            "response_id": response_id,
            "judge_score": float(judge_score),
        }
        fresh_draws.setdefault(policy_id, []).append(sample)

        template_rows.append(
            {
                "policy_id": policy_id,
                "prompt_id": prompt_id,
                "response_id": response_id,
                "record_id": str(record_id) if record_id is not None else "",
                "input": (
                    _stable_json(row.get(input_col))
                    if input_col and isinstance(row.get(input_col), (dict, list))
                    else (
                        str(row.get(input_col))
                        if input_col and row.get(input_col) is not None
                        else ""
                    )
                ),
                "output": (
                    _stable_json(row.get(output_col))
                    if output_col and isinstance(row.get(output_col), (dict, list))
                    else str(output) if output is not None else ""
                ),
                "judge_score": float(judge_score),
                "oracle_label": "",
            }
        )

    oracle_labels.apply(fresh_draws, source_prompt_counts=source_prompt_counts)
    by_response = {
        (policy, sample["response_id"]): sample.get("oracle_label", "")
        for policy, samples in fresh_draws.items()
        for sample in samples
    }
    for row in template_rows:
        row["oracle_label"] = by_response[(row["policy_id"], row["response_id"])]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_stable_json(fresh_draws) + "\n", encoding="utf-8")

    if not args.no_label_template:
        template_path = Path(args.label_template)
        template_path.parent.mkdir(parents=True, exist_ok=True)
        with template_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "policy_id",
                "prompt_id",
                "response_id",
                "record_id",
                "input",
                "output",
                "judge_score",
                "oracle_label",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in template_rows:
                w.writerow(r)

    print(f"Wrote CJE fresh_draws_data JSON: {out_path}")
    if not args.no_label_template:
        print(f"Wrote oracle label template CSV: {args.label_template}")
    if missing:
        print(
            f"Skipped {missing} rows without numeric judge_score in {args.judge_col!r}",
            file=sys.stderr,
        )

    if args.run_cje:
        if not oracle_labels.values:
            print(
                "--run-cje requested but no oracle labels were provided.",
                file=sys.stderr,
            )
            return 1
        try:
            from cje import analyze_dataset  # type: ignore

            results = analyze_dataset(fresh_draws_data=fresh_draws)
            print("CJE analyze_dataset succeeded.")
            est0 = getattr(results, "estimates", None)
            if est0 is not None:
                print(f"Estimates count: {len(est0)}")
        except Exception as e:
            print(f"CJE run failed: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
