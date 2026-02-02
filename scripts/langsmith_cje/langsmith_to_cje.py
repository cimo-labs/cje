#!/usr/bin/env python3
"""Convert LangSmith runs + feedback into CJE's `fresh_draws_data` format.

CJE expects:
  fresh_draws_data = {
    "policy_name": [
      {"prompt_id": "...", "judge_score": 0.83, "oracle_label": 0.9},
      ...
    ],
    ...
  }

This script is a **bridge**: it pulls per-run evaluation feedback from LangSmith
and writes a JSON file compatible with CJE.

Key idea:
- LangSmith stores run-level evaluation results as *feedback* objects.
- Many teams already run automated evals (LLM-based graders, rubrics, etc.)
  inside LangSmith; CJE lets you calibrate those judge scores to an oracle
  using a small labeled subset.

Refs:
- CJE paper: https://arxiv.org/abs/2512.11150
- LangSmith eval docs: https://docs.smith.langchain.com/evaluation

Assumptions / defaults:
- By default we treat each LangSmith *project* as a distinct policy.
  (Common workflow: one project per model / prompt variant.)
- We use `run.reference_example_id` as `prompt_id` when available.
  This tends to align runs across policies when they were generated from the
  same dataset example.

Oracle labels:
- Optional.
- If you already logged human labels into LangSmith as feedback, you can provide
  `--oracle-feedback-key` to populate `oracle_label` directly.
- Otherwise this script can emit an `oracle_label_template.csv` for manual
  labeling.

"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _as_float(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        # Avoid silently treating True/False as 1/0 unless user explicitly wants it.
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


@dataclass(frozen=True)
class _FeedbackPick:
    score: float
    created_at: datetime


def _pick_latest_numeric_feedback(feedbacks: Iterable[Any]) -> Dict[str, _FeedbackPick]:
    """Return map run_id -> latest numeric feedback (by created_at)."""

    picked: Dict[str, _FeedbackPick] = {}
    for fb in feedbacks:
        run_id = str(getattr(fb, "run_id", ""))
        if not run_id:
            continue

        score = _as_float(getattr(fb, "score", None))
        if score is None:
            score = _as_float(getattr(fb, "value", None))
        if score is None:
            continue

        created_at = getattr(fb, "created_at", None)
        if not isinstance(created_at, datetime):
            # Extremely defensive: if schema changes, fall back to "now" so we at least keep one.
            created_at = datetime.now()

        prev = picked.get(run_id)
        if prev is None or created_at > prev.created_at:
            picked[run_id] = _FeedbackPick(score=score, created_at=created_at)

    return picked


def _run_prompt_id(run: Any) -> str:
    ref = getattr(run, "reference_example_id", None)
    if ref is not None:
        return str(ref)
    return str(getattr(run, "id"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project",
        action="append",
        required=True,
        help="LangSmith project name. Repeatable. Default policy_name=project name.",
    )
    ap.add_argument(
        "--policy",
        action="append",
        default=None,
        help="Optional policy name override, parallel to --project (repeatable).",
    )
    ap.add_argument(
        "--feedback-key",
        required=True,
        help="LangSmith feedback key to treat as judge_score (e.g. 'Jaccard', 'correctness').",
    )
    ap.add_argument(
        "--oracle-feedback-key",
        default=None,
        help="Optional LangSmith feedback key to treat as oracle_label (e.g. human labels).",
    )
    ap.add_argument(
        "--run-type",
        default=None,
        help="Optional LangSmith run_type filter (e.g. 'chain', 'llm').",
    )
    ap.add_argument(
        "--include-child-runs",
        action="store_true",
        help="Include non-root runs (not recommended; default is root runs only).",
    )
    ap.add_argument(
        "--execution-order",
        type=int,
        default=1,
        help="Execution order to filter by (default: 1). Passed through to LangSmith API.",
    )
    ap.add_argument("--query", default=None, help="Optional LangSmith query string.")
    ap.add_argument("--filter", default=None, help="Optional LangSmith filter string.")
    ap.add_argument("--limit", type=int, default=None, help="Max runs per project.")

    ap.add_argument("--out", default="cje_fresh_draws_data.json", help="Output JSON for CJE")
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
    ap.add_argument(
        "--run-cje",
        action="store_true",
        help="If cje-eval is installed and oracle labels are present, run analyze_dataset and print a short summary.",
    )

    args = ap.parse_args()

    projects: List[str] = args.project
    policies: List[str]
    if args.policy is None:
        policies = projects
    else:
        policies = args.policy
        if len(policies) != len(projects):
            print(
                "If provided, --policy must be repeated the same number of times as --project.",
                file=sys.stderr,
            )
            return 2

    try:
        from langsmith import Client  # type: ignore
    except Exception as e:
        print(
            "Missing dependency: langsmith. Install with `pip install langsmith` (and set LANGSMITH_API_KEY).",
            file=sys.stderr,
        )
        print(f"Import error: {e}", file=sys.stderr)
        return 2

    client = Client()

    fresh_draws: Dict[str, List[Dict[str, Any]]] = {}
    template_rows: List[Dict[str, Any]] = []

    for project_name, policy_name in zip(projects, policies):
        # Use is_root=True unless user opts into child runs.
        is_root = None if args.include_child_runs else True

        # list_runs accepts additional query params via kwargs.
        runs = list(
            client.list_runs(
                project_name=project_name,
                run_type=args.run_type,
                is_root=is_root,
                query=args.query,
                filter=args.filter,
                error=False,
                limit=args.limit,
                execution_order=args.execution_order,
            )
        )

        run_ids = [getattr(r, "id", None) for r in runs]
        run_ids = [rid for rid in run_ids if rid is not None]

        judge_fb = list(client.list_feedback(run_ids=run_ids, feedback_key=[args.feedback_key]))
        judge_map = _pick_latest_numeric_feedback(judge_fb)

        oracle_map: Dict[str, _FeedbackPick] = {}
        if args.oracle_feedback_key:
            oracle_fb = list(
                client.list_feedback(run_ids=run_ids, feedback_key=[args.oracle_feedback_key])
            )
            oracle_map = _pick_latest_numeric_feedback(oracle_fb)

        missing_judge = 0
        for r in runs:
            rid = str(getattr(r, "id", ""))
            if rid not in judge_map:
                missing_judge += 1
                continue

            prompt_id = _run_prompt_id(r)
            judge_score = judge_map[rid].score

            sample: Dict[str, Any] = {"prompt_id": prompt_id, "judge_score": judge_score}
            if args.oracle_feedback_key and rid in oracle_map:
                sample["oracle_label"] = oracle_map[rid].score

            fresh_draws.setdefault(policy_name, []).append(sample)

            if not args.no_label_template:
                template_rows.append(
                    {
                        "policy_name": policy_name,
                        "project_name": project_name,
                        "run_id": rid,
                        "prompt_id": prompt_id,
                        "inputs_json": _stable_json(getattr(r, "inputs", {}) or {}),
                        "outputs_json": _stable_json(getattr(r, "outputs", {}) or {}),
                        "judge_score": judge_score,
                        "oracle_label": (oracle_map[rid].score if (rid in oracle_map) else ""),
                    }
                )

        if missing_judge:
            print(
                f"[{project_name}] Skipped {missing_judge}/{len(runs)} runs without numeric feedback '{args.feedback_key}'.",
                file=sys.stderr,
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_stable_json(fresh_draws) + "\n", encoding="utf-8")
    print(f"Wrote CJE fresh_draws_data JSON: {out_path}")

    if not args.no_label_template:
        template_path = Path(args.label_template)
        template_path.parent.mkdir(parents=True, exist_ok=True)
        with template_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "policy_name",
                "project_name",
                "run_id",
                "prompt_id",
                "inputs_json",
                "outputs_json",
                "judge_score",
                "oracle_label",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in template_rows:
                w.writerow(row)
        print(f"Wrote oracle label template CSV: {args.label_template}")

    if args.run_cje:
        # Only run CJE if we actually have some oracle labels.
        has_oracle = any(
            any("oracle_label" in s for s in samples) for samples in fresh_draws.values()
        )
        if not has_oracle:
            print(
                "--run-cje requested but no oracle labels were found. "
                "Provide --oracle-feedback-key or label a subset via the CSV template.",
                file=sys.stderr,
            )
            return 0
        try:
            from cje import analyze_dataset  # type: ignore

            results = analyze_dataset(fresh_draws_data=fresh_draws)
            est0 = getattr(results, "estimates", None)
            print("CJE analyze_dataset succeeded.")
            if est0 is not None:
                print(f"Estimates count: {len(est0)}")
        except Exception as e:
            print(f"CJE run failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
