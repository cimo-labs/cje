"""Offline fixed-pool label-selection example. No model calls or response text."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2_contingency
from cje import analyze_dataset

ROOT = Path(__file__).resolve().parent
POLICIES = ("base", "premium")
# Frozen before running: asymmetric residual flag rates, not fitted to a reversal.
RATES = {
    "symmetric": {"base": 36, "premium": 36},
    "differential": {"base": 60, "premium": 24},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    rows = [json.loads(x) for x in (ROOT / "data.jsonl").read_text().splitlines()]
    data = {p: [r for r in rows if r["policy"] == p] for p in POLICIES}
    assert len(rows) == 1200
    assert all(
        set(r) == {"prompt_id", "policy", "judge_score", "oracle_label"} for r in rows
    )
    assert [r["prompt_id"] for r in data["base"]] == [
        r["prompt_id"] for r in data["premium"]
    ]
    truth = {p: float(np.mean([r["oracle_label"] for r in data[p]])) for p in POLICIES}
    truth["premium-base"] = truth["premium"] - truth["base"]
    bins, flags, valuebins = {}, {}, {}
    for p in POLICIES:
        s = np.array([r["judge_score"] for r in data[p]])
        y = np.array([r["oracle_label"] for r in data[p]])
        bins[p] = np.empty(600, int)
        bins[p][np.argsort(s, kind="stable")] = np.arange(600) // 120
        edges = np.unique(np.quantile(s, np.arange(0, 1.01, 0.2)))
        valuebins[p] = np.digitize(s, edges[1:-1], right=True)
        for family, sizes in RATES.items():
            flag = np.zeros(600, bool)
            for b in range(5):
                ix = np.flatnonzero(bins[p] == b)
                flag[ix[np.argsort((y - s)[ix], kind="stable")[: sizes[p]]]] = True
            flags[family, p] = flag
    conditions = [
        ("random", None, "representative"),
        ("flagged_symmetric", "symmetric", "representative"),
        ("targeted_unknown", "symmetric", "targeted_unknown"),
        ("known_propensity_symmetric", "symmetric", "known_propensity"),
        ("flagged_differential", "differential", "representative"),
        ("known_propensity_differential", "differential", "known_propensity"),
    ]
    runs = []
    start = time.perf_counter()
    for rep in range(args.repetitions):
        shared = {}
        for name, family, kind in conditions:
            masks, props, checks, mask_hashes = {}, {}, {}, {}
            for p in POLICIES:
                # Each repetition/condition/policy has a reproducible stream.
                # The symmetric two flagged analyses deliberately reuse a mask.
                stream = "flagged_symmetric" if name == "targeted_unknown" else name
                seed = int.from_bytes(
                    hashlib.sha256(f"20261024:{rep}:{stream}:{p}".encode()).digest()[
                        :8
                    ],
                    "big",
                )
                rng = np.random.default_rng(seed)
                mask, prob = np.zeros(600, bool), np.zeros(600)
                if family is None:
                    mask[rng.choice(600, 60, replace=False)] = True
                    prob[:] = 0.1
                else:
                    for b in range(5):
                        f = np.flatnonzero((bins[p] == b) & flags[family, p])
                        u = np.flatnonzero((bins[p] == b) & ~flags[family, p])
                        nf, nu = (9, 3) if kind == "known_propensity" else (12, 0)
                        mask[rng.choice(f, nf, replace=False)] = True
                        if nu:
                            mask[rng.choice(u, nu, replace=False)] = True
                        prob[f], prob[u] = nf / len(f), nu / len(u)
                assert mask.sum() == 60
                if kind == "known_propensity":
                    assert np.all(prob > 0)
                masks[p], props[p] = mask, prob
                table = [
                    [
                        int(mask[valuebins[p] == b].sum()),
                        int((~mask[valuebins[p] == b]).sum()),
                    ]
                    for b in np.unique(valuebins[p])
                ]
                chi = chi2_contingency(table, correction=False)
                checks[p] = {
                    "chi2": float(chi.statistic),
                    "pvalue": float(chi.pvalue),
                    "dof": int(chi.dof),
                    "counts": table,
                }
                mask_hashes[p] = hashlib.sha256(mask.tobytes()).hexdigest()
            if name == "flagged_symmetric":
                shared = mask_hashes
            if name == "targeted_unknown":
                assert mask_hashes == shared
            draws = {
                p: [
                    {**r, "oracle_label": r["oracle_label"] if masks[p][i] else None}
                    for i, r in enumerate(data[p])
                ]
                for p in POLICIES
            }
            kwargs: dict[str, Any] = (
                {"label_propensities": props} if kind == "known_propensity" else {}
            )
            result = analyze_dataset(
                fresh_draws_data=draws,
                label_design=kind,
                fresh_judge_scale=(0, 1),
                fresh_oracle_scale=(0, 1),
                **kwargs,
            )
            estimates = dict(zip(result.target_policies, result.estimates))
            cis = dict(zip(result.target_policies, result.ci()))
            pair = result.compare_policies("premium", "base")
            values = {
                p: {
                    "estimate": float(estimates[p]),
                    "lo": float(cis[p][0]),
                    "hi": float(cis[p][1]),
                }
                for p in POLICIES
            }
            values["premium-base"] = {
                "estimate": float(pair["difference"]),
                "lo": float(pair["ci_lower"]),
                "hi": float(pair["ci_upper"]),
            }
            runs.append(
                {
                    "rep": rep,
                    "condition": name,
                    "values": values,
                    "quintile": checks,
                    "label_counts": {p: int(masks[p].sum()) for p in POLICIES},
                    "mask_hashes": mask_hashes,
                    "status": str(result.diagnostics.overall_status),
                    "pair_significant": bool(pair["significant"]),
                    "pair_p_value": float(pair["p_value"]),
                    "pair_gate_flagged": pair["gate_flagged"],
                    "transport_unverified": result.metadata.get("transport_unverified"),
                }
            )
        if (rep + 1) % 10 == 0:
            print(f"{rep+1}/{args.repetitions}", flush=True)
    summary = []
    for name, _, _ in conditions:
        rr = [r for r in runs if r["condition"] == name]
        for target in (*POLICIES, "premium-base"):
            vals = [r["values"][target] for r in rr]
            errors = np.array([v["estimate"] - truth[target] for v in vals])
            summary.append(
                {
                    "condition": name,
                    "target": target,
                    "truth": truth[target],
                    "bias": float(errors.mean()),
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "coverage": float(
                        np.mean([v["lo"] <= truth[target] <= v["hi"] for v in vals])
                    ),
                    "mean_ci_half_width": float(
                        np.mean([(v["hi"] - v["lo"]) / 2 for v in vals])
                    ),
                    "opposite_sign_fraction": float(
                        np.mean([v["estimate"] * truth[target] < 0 for v in vals])
                    ),
                    "quintile_pass_fraction": (
                        float(
                            np.mean(
                                [r["quintile"][target]["pvalue"] >= 0.05 for r in rr]
                            )
                        )
                        if target in POLICIES
                        else None
                    ),
                    "distinct_label_sets": (
                        len({r["mask_hashes"][target] for r in rr})
                        if target in POLICIES
                        else None
                    ),
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "runs.json").write_text(json.dumps(runs, indent=2))
    with (args.output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "repetitions": args.repetitions,
                "elapsed_seconds": time.perf_counter() - start,
                "rates": RATES,
                "summary": summary,
                "data_sha256": hashlib.sha256(
                    (ROOT / "data.jsonl").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
