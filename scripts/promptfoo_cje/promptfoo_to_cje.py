#!/usr/bin/env python3
"""Convert Promptfoo evaluation output into CJE's `fresh_draws_data` format.

CJE expects:
  fresh_draws_data = {
    "policy_name": [
      {"prompt_id": "...", "judge_score": 0.83, "oracle_label": 0.9},
      ...
    ],
    ...
  }

Promptfoo export formats vary across versions.
This script supports:
  (A) EvaluateSummaryV3 (per Promptfoo docs):
      {"version": 3, "timestamp": "...", "results": [EvaluateResult...], "prompts": [...], "stats": {...}}
  (B) Older docs-style nested results:
      {"version": 3, "timestamp": "...", "results": {"outputs": [...], ...}}

Notes:
- `judge_score` is taken from `result.score` (0-1). If missing, falls back to `gradingResult.score`.
- `prompt_id` must be stable across policies. Default is a deterministic hash of:
    prompt.label + normalized(vars)
  You can switch to `raw` mode (uses prompt.label only) if vars already uniquely identify the test.

Oracle labels:
- Optional. Provide a JSONL or CSV file with at least:
    provider_id,prompt_id,oracle_label
  (extra columns are ignored)

Outputs:
- `--out` writes CJE fresh_draws_data JSON.
- `--label-template` writes a CSV to help humans label an oracle slice.

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _detect_results(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of EvaluateResult-like dicts.

    Promptfoo has multiple export shapes.

    Supported:
    - A) EvaluateSummaryV3 at top-level: {"version": 3, "results": [ ... ]}
    - B) Older nested: {"results": {"outputs": [ ... ]}}
    - C) `promptfoo export` style wrapper: {"evalId": "...", "results": {"version": 3, "results": [ ... ]}}
    """

    # C) `promptfoo export` wrapper
    if isinstance(payload.get("results"), dict) and isinstance(payload["results"].get("results"), list):
        return payload["results"]["results"]

    # A) EvaluateSummaryV3: payload["results"] is a list
    if isinstance(payload.get("results"), list):
        return payload["results"]

    # B) Older docs: payload["results"]["outputs"]
    results = payload.get("results")
    if isinstance(results, dict) and isinstance(results.get("outputs"), list):
        # These may already be EvaluateResult-like, but could be table outputs.
        return results["outputs"]

    raise ValueError(
        "Unrecognized promptfoo JSON structure. Expected either top-level `results: []`, "
        "`results: { outputs: [] }`, or export wrapper `results: { results: [] }`."
    )


def _provider_id(r: Dict[str, Any]) -> str:
    prov = r.get("provider")
    if isinstance(prov, dict) and isinstance(prov.get("id"), str):
        return prov["id"]
    if isinstance(r.get("provider"), str):
        return r["provider"]
    # Some formats use `providerId`
    if isinstance(r.get("providerId"), str):
        return r["providerId"]
    raise ValueError(f"Cannot extract provider id from record keys={list(r.keys())}")


def _prompt_label(r: Dict[str, Any]) -> str:
    p = r.get("prompt")
    if isinstance(p, dict):
        if isinstance(p.get("label"), str):
            return p["label"]
        if isinstance(p.get("id"), str):
            return p["id"]
    if isinstance(r.get("prompt"), str):
        return r["prompt"]
    # Some formats use `promptId`
    if isinstance(r.get("promptId"), str):
        return r["promptId"]
    return "<unknown-prompt>"


def _vars(r: Dict[str, Any]) -> Dict[str, Any]:
    """Extract per-test variables.

    Promptfoo stores vars in different places depending on export shape:
    - EvaluateResult-style: r["vars"]
    - `promptfoo export` / fixtures: r["testCase"]["vars"]
    """
    tc = r.get("testCase")
    if isinstance(tc, dict) and isinstance(tc.get("vars"), dict):
        return tc["vars"]

    v = r.get("vars")
    return v if isinstance(v, dict) else {}


def _judge_score(r: Dict[str, Any]) -> Optional[float]:
    # Prefer top-level score
    score = r.get("score")
    if isinstance(score, (int, float)):
        return float(score)

    # Fall back to gradingResult.score
    gr = r.get("gradingResult")
    if isinstance(gr, dict) and isinstance(gr.get("score"), (int, float)):
        return float(gr["score"])

    return None


def _output_text(r: Dict[str, Any]) -> Optional[str]:
    # Promptfoo stores provider response under response.output
    resp = r.get("response")
    if isinstance(resp, dict):
        out = resp.get("output")
        if isinstance(out, str):
            return out
        # Sometimes structured output is an object
        if isinstance(out, (dict, list)):
            return _stable_json(out)

    # Older formats may use output/text directly
    if isinstance(r.get("output"), str):
        return r["output"]
    if isinstance(r.get("text"), str):
        return r["text"]

    return None


def _make_prompt_id(prompt_label: str, vars_obj: Dict[str, Any], mode: str) -> str:
    if mode == "raw":
        # Only safe if you know vars already unique OR you don't care about collision.
        return prompt_label

    if mode == "hash":
        key = _stable_json({"prompt": prompt_label, "vars": vars_obj})
        return f"{prompt_label}::{_hash_str(key)}"

    raise ValueError(f"Unknown prompt_id mode: {mode}")


def _load_oracle_labels(path: Path) -> Dict[Tuple[str, str], float]:
    """Map (provider_id, prompt_id) -> oracle_label."""
    labels: Dict[Tuple[str, str], float] = {}

    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            provider_id = row.get("provider_id") or row.get("provider")
            prompt_id = row.get("prompt_id")
            oracle_label = row.get("oracle_label")
            if provider_id is None or prompt_id is None or oracle_label is None:
                continue
            labels[(str(provider_id), str(prompt_id))] = float(oracle_label)
        return labels

    # CSV
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            provider_id = row.get("provider_id") or row.get("provider")
            prompt_id = row.get("prompt_id")
            oracle_label = row.get("oracle_label")
            if not provider_id or not prompt_id or oracle_label is None:
                continue
            labels[(str(provider_id), str(prompt_id))] = float(oracle_label)

    return labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json", help="Promptfoo results JSON (e.g. results.json)")
    ap.add_argument("--out", default="cje_fresh_draws_data.json", help="Output JSON for CJE")
    ap.add_argument(
        "--prompt-id-mode",
        choices=["hash", "raw"],
        default="hash",
        help="How to create prompt_id (default: hash prompt.label + vars)",
    )
    ap.add_argument(
        "--oracle-labels",
        default=None,
        help="Optional oracle labels file (.csv or .jsonl) with provider_id,prompt_id,oracle_label",
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
    ap.add_argument(
        "--run-cje",
        action="store_true",
        help="If cje-eval is installed and oracle labels are present, run analyze_dataset and print a short summary.",
    )

    args = ap.parse_args()

    results_path = Path(args.results_json)
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    records = _detect_results(payload)

    oracle_map: Dict[Tuple[str, str], float] = {}
    if args.oracle_labels:
        oracle_map = _load_oracle_labels(Path(args.oracle_labels))

    fresh_draws: Dict[str, List[Dict[str, Any]]] = {}
    template_rows: List[Dict[str, Any]] = []

    missing_score = 0
    for r in records:
        provider_id = _provider_id(r)
        prompt_label = _prompt_label(r)
        vars_obj = _vars(r)
        prompt_id = _make_prompt_id(prompt_label, vars_obj, args.prompt_id_mode)

        judge_score = _judge_score(r)
        if judge_score is None:
            missing_score += 1
            continue

        sample: Dict[str, Any] = {"prompt_id": prompt_id, "judge_score": judge_score}
        oracle = oracle_map.get((provider_id, prompt_id))
        if oracle is not None:
            sample["oracle_label"] = oracle

        fresh_draws.setdefault(provider_id, []).append(sample)

        # Labeling template row (one row per provider x prompt)
        template_rows.append(
            {
                "provider_id": provider_id,
                "prompt_id": prompt_id,
                "prompt_label": prompt_label,
                "vars_json": _stable_json(vars_obj),
                "output": (_output_text(r) or ""),
                "judge_score": judge_score,
                "oracle_label": oracle if oracle is not None else "",
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_stable_json(fresh_draws) + "\n", encoding="utf-8")

    if not args.no_label_template:
        template_path = Path(args.label_template)
        template_path.parent.mkdir(parents=True, exist_ok=True)
        with template_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "provider_id",
                "prompt_id",
                "prompt_label",
                "vars_json",
                "output",
                "judge_score",
                "oracle_label",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in template_rows:
                w.writerow(row)

    print(f"Wrote CJE fresh_draws_data JSON: {out_path}")
    if not args.no_label_template:
        print(f"Wrote oracle label template CSV: {args.label_template}")
    if missing_score:
        print(f"Skipped {missing_score} records without a numeric score", file=sys.stderr)

    if args.run_cje:
        if not args.oracle_labels:
            print("--run-cje requested but --oracle-labels not provided; skipping.", file=sys.stderr)
            return 0
        try:
            from cje import analyze_dataset  # type: ignore

            results = analyze_dataset(fresh_draws_data=fresh_draws)
            # Keep summary short and robust to API changes
            est0 = getattr(results, "estimates", None)
            print("CJE analyze_dataset succeeded.")
            if est0 is not None:
                print(f"Estimates count: {len(est0)}")
        except Exception as e:
            print(f"CJE run failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
