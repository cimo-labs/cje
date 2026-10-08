"""Regenerate images/forest_plot_n1000_oracle25.png for the README.

Uses only the Chatbot Arena sample shipped in examples/arena_sample:
1,000 responses per policy, GPT-5 labels standing in for human ratings on
480 base-policy responses, and 50 held-out labelled responses for each other
policy. Probe prompt IDs are excluded from fitting, as the sample's README
requires, so the held-out means are an independent reference.

Run with cje-eval[viz] installed:
    python scripts/make_readme_forest_plot.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from cje import TransportAuditConfig, analyze_dataset  # noqa: E402
from cje.visualization.estimates import plot_policy_estimates  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "arena_sample"
OUT = ROOT / "images" / "forest_plot_n1000_oracle25.png"
POLICIES = ["base", "clone", "parallel_universe_prompt", "unhelpful"]
DELTA_MAX = 0.05  # practical margin on the 0-1 label scale


def read_jsonl(path: Path) -> list:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    draws = {
        p: read_jsonl(DATA / "fresh_draws" / f"{p}_responses.jsonl") for p in POLICIES
    }
    probes = {
        p: read_jsonl(DATA / "probe_slice" / f"{p}_probe.jsonl")
        for p in POLICIES
        if p != "base"
    }
    probe_ids = {row["prompt_id"] for rows in probes.values() for row in rows}
    evaluation = {
        p: [row for row in rows if row["prompt_id"] not in probe_ids]
        for p, rows in draws.items()
    }

    results = analyze_dataset(
        fresh_draws_data=evaluation,
        transport=TransportAuditConfig(
            probes_by_policy=probes,
            delta_max_by_policy={p: DELTA_MAX for p in probes},
        ),
    )
    names = list(results.metadata["target_policies"])
    lower, upper = results.confidence_interval()
    estimates = {p: float(results.estimates[i]) for i, p in enumerate(names)}
    ses = {p: float(results.standard_errors[i]) for i, p in enumerate(names)}
    cis = {p: (float(lower[i]), float(upper[i])) for i, p in enumerate(names)}

    # Reference: mean label on held-out responses (base: its own labelled responses).
    reference = {
        p: sum(r["oracle_label"] for r in rows) / len(rows)
        for p, rows in probes.items()
    }
    base_labels = [
        r["oracle_label"]
        for r in evaluation["base"]
        if r.get("oracle_label") is not None
    ]
    reference["base"] = sum(base_labels) / len(base_labels)

    audits = results.metadata.get("transport_audits", {})
    has_own_labels = {
        p
        for p in names
        if any(r.get("oracle_label") is not None for r in evaluation[p])
    }
    labels = {
        p: (
            f"{p}  (own labels)"
            if p in has_own_labels
            else f"{p}  (audit: {audits[p]['status']})"
        )
        for p in names
    }
    fig = plot_policy_estimates(
        estimates,
        ses,
        oracle_values=reference,
        policy_labels=labels,
        title="Arena sample: calibrated estimates vs held-out label means",
        confidence_intervals=cis,
    )
    # The references are 50-response means (base: its own labels), not exact
    # truth, so relabel the legend and drop the library's RMSE box.
    ax = fig.axes[0]
    for text in ax.get_legend().get_texts():
        if text.get_text().startswith("Oracle"):
            text.set_text("Mean label, held-out responses (base: its own labels)")
    for text in list(ax.texts):
        if text.get_text().startswith("RMSE"):
            text.remove()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    for p in names:
        print(
            f"{p:26s} estimate {estimates[p]:.3f}  CI [{cis[p][0]:.3f}, {cis[p][1]:.3f}]"
            f"  held-out mean {reference[p]:.3f}  {labels[p]}"
        )
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
