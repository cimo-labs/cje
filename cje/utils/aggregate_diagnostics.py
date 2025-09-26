"""
Aggregate CJE result JSON files into a single CSV for analysis.

Usage:
    python -m cje.utils.aggregate_diagnostics --input results_dir --output agg.csv

The aggregator is best-effort: it extracts core fields (policy, estimate, SE)
and selected metadata (calibration_floor, calibration_info, OUA) if present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv


def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def aggregate_json_file(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)

    rows: List[Dict[str, Any]] = []
    policies: List[str] = data.get("target_policies") or list(
        (data.get("per_policy_results") or {}).keys()
    )

    per_policy = data.get("per_policy_results", {})
    metadata = data.get("metadata", {})

    # Robust SEs (if exported elsewhere) are not part of default export;
    # we rely on per-policy robust SEs being absent (None) in most cases.

    diagnostics = data.get("diagnostics", {})
    ess_per_policy = _get(diagnostics, "ess_per_policy", default={})
    tail_indices = _get(diagnostics, "tail_indices", default={})
    hellinger_per_policy = _get(diagnostics, "hellinger_per_policy", default={})
    hellinger_overall = _float_or_none(diagnostics.get("hellinger_affinity"))

    for idx, policy in enumerate(policies):
        p = per_policy.get(policy, {})

        row: Dict[str, Any] = {
            "file": str(path),
            "timestamp": data.get("timestamp"),
            "method": data.get("method"),
            "policy": policy,
            "estimate": _float_or_none(p.get("estimate")),
            "standard_error": _float_or_none(p.get("standard_error")),
            "ci_lower": _float_or_none(p.get("ci_lower")),
            "ci_upper": _float_or_none(p.get("ci_upper")),
            "n_samples": p.get("n_samples"),
        }

        # Overlap/weights diagnostics (if available)
        row["ess_fraction"] = _float_or_none(ess_per_policy.get(policy))
        row["tail_index"] = _float_or_none(tail_indices.get(policy))
        # Prefer per-policy Hellinger when available, else overall
        hp = hellinger_per_policy.get(policy)
        row["hellinger_affinity"] = (
            _float_or_none(hp) if hp is not None else hellinger_overall
        )

        # Calibration info (global)
        cal_info = metadata.get("calibration_info", {})
        row.update(
            {
                "f_min": _float_or_none(cal_info.get("f_min")),
                "f_max": _float_or_none(cal_info.get("f_max")),
                "n_oracle": cal_info.get("n_oracle"),
                "cal_rmse": _float_or_none(cal_info.get("rmse")),
                "low_s_cov_b10": _float_or_none(
                    cal_info.get("low_s_label_coverage_bottom10")
                ),
                "low_s_cov_b20": _float_or_none(
                    cal_info.get("low_s_label_coverage_bottom20")
                ),
            }
        )

        # Calibration floor (per-policy)
        cal_floor = metadata.get("calibration_floor", {}).get(policy, {})
        row.update(
            {
                "floor_mass_logged": _float_or_none(cal_floor.get("floor_mass_logged")),
                "floor_mass_fresh": _float_or_none(cal_floor.get("floor_mass_fresh")),
            }
        )

        # OUA (var_oracle per policy)
        var_oracle = _get(metadata, "oua", "var_oracle_per_policy", default={}).get(
            policy
        )
        row["var_oracle"] = _float_or_none(var_oracle)

        rows.append(row)

    return rows


def aggregate_dir(input_path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in input_path.rglob("*.json"):
        try:
            items.extend(aggregate_json_file(p))
        except Exception:
            continue
    return items


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file"])  # placeholder
        return

    # Stable header order
    header = [
        "file",
        "timestamp",
        "method",
        "policy",
        "estimate",
        "standard_error",
        "ci_lower",
        "ci_upper",
        "n_samples",
        "f_min",
        "f_max",
        "n_oracle",
        "cal_rmse",
        "low_s_cov_b10",
        "low_s_cov_b20",
        "floor_mass_logged",
        "floor_mass_fresh",
        "var_oracle",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        # Use a distinct name to avoid conflicts with the empty-rows branch
        from csv import DictWriter

        dict_writer = DictWriter(f, fieldnames=header)
        dict_writer.writeheader()
        for r in rows:
            dict_writer.writerow({k: r.get(k) for k in header})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate CJE result JSON files into a CSV"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Directory containing result JSONs"
    )
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    args = parser.parse_args()

    rows = aggregate_dir(Path(args.input))
    write_csv(rows, Path(args.output))


if __name__ == "__main__":
    main()
