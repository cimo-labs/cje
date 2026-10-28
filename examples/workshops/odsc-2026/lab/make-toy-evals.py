"""Write lab/toy-evals.csv: CONSTRUCTED numbers for the notebook's "your data" section.

Not real data. 60 prompts answered by two policies (production, candidate). A made-up judge
scores every response on 1-5, compressed toward the middle so its levels are off. Made-up
expert ratings (1-5) exist for 30 prompts per policy, picked at random at the same rate in
every judge-score bin (3 of 6 in each tenth of that policy's scores), which is the label rule
the lab recommends. The candidate is built to be better by about +0.8 points.

Seed 0, not chosen: across seeds 0-99 of this design, the notebook's CI gate ships in 91. Rerunning writes the same file:
    python3 lab/make-toy-evals.py
"""
import csv
from pathlib import Path

import numpy as np


def make_rows(seed=0, n_prompts=60, n_bins=10):
    rng = np.random.default_rng(seed)
    difficulty = rng.normal(0, 0.6, n_prompts)
    quality = {"production": 2.7 + difficulty + rng.normal(0, 0.3, n_prompts)}
    quality["candidate"] = quality["production"] + 0.8 + rng.normal(0, 0.3, n_prompts)
    rows = []
    for policy in ("production", "candidate"):
        q = quality[policy]
        judge = np.clip(np.round(1.2 + 0.7 * q + rng.normal(0, 0.2, n_prompts), 2), 1, 5)
        expert = np.clip(np.round(q + rng.normal(0, 0.4, n_prompts)), 1, 5)
        order = np.argsort(judge, kind="stable")
        per_bin = n_prompts // n_bins
        labeled = {int(i) for b in range(n_bins)
                   for i in rng.choice(order[b * per_bin:(b + 1) * per_bin], per_bin // 2, replace=False)}
        for i in range(n_prompts):
            rows.append({"prompt_id": f"q{i + 1:02d}", "variant": policy, "judge_score": f"{judge[i]:.2f}",
                         "human_rating": f"{expert[i]:.0f}" if i in labeled else ""})
    return sorted(rows, key=lambda r: (r["prompt_id"], r["variant"] != "production"))


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "toy-evals.csv"
    rows = make_rows(seed=0)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_id", "variant", "judge_score", "human_rating"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")
