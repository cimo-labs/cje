"""
Basic analysis over aggregated diagnostics CSV.

Usage:
    python -m cje.utils.analyze_diagnostics --input agg.csv --corr out_corr.csv

Computes:
- Correlation matrix across key numeric fields
- Simple threshold-based 'should_not_ship' proxy counts
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any


NUM_FIELDS = [
    "estimate",
    "standard_error",
    "out_of_range",
    "f_min",
    "low_s_cov_b10",
    "low_s_cov_b20",
    "floor_mass_logged",
    "floor_mass_fresh",
    "var_oracle",
]


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def compute_correlation(
    rows: List[Dict[str, Any]], fields: List[str]
) -> List[List[float]]:
    import math

    # Collect columns
    cols: List[List[float]] = []
    for f in fields:
        col = [to_float(r.get(f)) for r in rows]
        cols.append(col)

    # Compute Pearson correlation
    def pearson(a: List[float], b: List[float]) -> float:
        xs = [x for x in a if not math.isnan(x)]
        ys = [
            b[i] for i, x in enumerate(a) if not math.isnan(x) and not math.isnan(b[i])
        ]
        xs = [x for i, x in enumerate(a) if not math.isnan(x) and not math.isnan(b[i])]
        if not xs or len(xs) < 3:
            return float("nan")
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
        denx = sum((x - mean_x) ** 2 for x in xs)
        deny = sum((y - mean_y) ** 2 for y in ys)
        den = (denx * deny) ** 0.5
        return num / den if den > 0 else float("nan")

    corr: List[List[float]] = []
    for i in range(len(fields)):
        row = []
        for j in range(len(fields)):
            row.append(pearson(cols[i], cols[j]))
        corr.append(row)
    return corr


def write_matrix(matrix: List[List[float]], fields: List[str], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["field"] + fields)
        for name, row in zip(fields, matrix):
            writer.writerow([name] + row)


def simple_proxy_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    import math

    from ..diagnostics.gates import OUT_OF_RANGE_REFUSE_THRESHOLD

    # Proxies for 'do not ship level claims'
    counts = {
        "refuse_level": 0,  # coverage badge REFUSE-LEVEL (out-of-range >= 5%)
        "cal_floor": 0,  # floor_mass_logged or fresh >= 0.25
    }
    for r in rows:
        status = r.get("boundary_status")
        oor = to_float(r.get("out_of_range"))
        fm_l = to_float(r.get("floor_mass_logged"))
        fm_f = to_float(r.get("floor_mass_fresh"))
        if status == "REFUSE-LEVEL" or (
            not math.isnan(oor) and oor >= OUT_OF_RANGE_REFUSE_THRESHOLD
        ):
            counts["refuse_level"] += 1
        if (fm_l and fm_l >= 0.25) or (fm_f and fm_f >= 0.25):
            counts["cal_floor"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze aggregated diagnostics CSV")
    parser.add_argument("--input", "-i", required=True, help="Aggregated CSV")
    parser.add_argument(
        "--corr", "-c", required=False, help="Output path for correlation CSV"
    )
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    if args.corr:
        matrix = compute_correlation(rows, NUM_FIELDS)
        write_matrix(matrix, NUM_FIELDS, Path(args.corr))

    counts = simple_proxy_counts(rows)
    print("Proxy 'do not ship' counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
