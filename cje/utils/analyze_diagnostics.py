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
    "ess_fraction",
    "hellinger_affinity",
    "tail_index",
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
    # Proxies for 'do not ship'
    counts = {
        "low_overlap": 0,  # Hellinger < 0.20 OR ess_fraction < 0.05
        "heavy_tail": 0,  # tail_index < 1.0
        "cal_floor": 0,  # floor_mass_logged or fresh >= 0.25
    }
    for r in rows:
        hell = to_float(r.get("hellinger_affinity"))
        ess = to_float(r.get("ess_fraction"))
        tail = to_float(r.get("tail_index"))
        fm_l = to_float(r.get("floor_mass_logged"))
        fm_f = to_float(r.get("floor_mass_fresh"))
        if (not hell or hell < 0.20) or (not ess or ess < 0.05):
            counts["low_overlap"] += 1
        if tail and tail < 1.0:
            counts["heavy_tail"] += 1
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
