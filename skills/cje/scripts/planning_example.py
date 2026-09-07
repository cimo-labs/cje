"""Offline pilot -> budget plan -> held-out comparison using bundled Arena scores.

Run from a repository checkout, after installing CJE:
    python skills/cje/scripts/planning_example.py --output-dir /tmp/cje-plan

This is a retrospective label-masking example, NOT a power/coverage validation.
It targets only the fixed frame with cached base-policy reference labels (model
labels, not humans), not all Arena traffic. No network or model calls are made.
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import platform
import shlex
import sys
from typing import Any
import warnings

import numpy as np

from cje import (
    CostModel,
    analyze_dataset,
    fit_variance_model,
    plan_evaluation,
    plan_for_mde,
)
from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample

POLICIES = ("base", "parallel_universe_prompt")
ANALYSIS_CONFIG = dict(
    fresh_judge_scale=(0, 1),
    fresh_oracle_scale=(0, 1),
    label_design="representative",
    estimator_config={"inference_method": "cluster_robust"},
)
CONFIG = dict(
    seed=42,
    pilot_n=240,
    pilot_m=120,
    n_grid=[80, 110, 150],
    oracle_fraction_grid=[0.25, 0.5, 0.75],
    n_replicates=50,
    budget=34.0,
    target_effect=0.10,
    power=0.80,
    alpha=0.05,
    m_min=30,
)


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Save a complete numeric replay and return its compact audit record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {p: data_dir / f"{p}_responses.jsonl" for p in POLICIES}
    records = {
        p: {r["prompt_id"]: r for r in map(json.loads, path.read_text().splitlines())}
        for p, path in sources.items()
    }
    frame = sorted(
        i
        for i, row in records["base"].items()
        if row.get("oracle_label") is not None and i in records[POLICIES[1]]
    )
    if len(frame) < 2 * CONFIG["pilot_n"]:
        raise ValueError("This example needs at least 480 base-labeled common prompts.")
    rng = np.random.default_rng(CONFIG["seed"])
    shuffled = rng.permutation(frame).tolist()
    pilot_ids, heldout_pool = (
        shuffled[: CONFIG["pilot_n"]],
        shuffled[CONFIG["pilot_n"] :],
    )
    pilot_labels = set(rng.choice(pilot_ids, CONFIG["pilot_m"], replace=False).tolist())

    def rows(policy: str, ids: list[str], labels: set[str]) -> list[dict[str, Any]]:
        return [
            {
                "prompt_id": i,
                "judge_score": records[policy][i]["judge_score"],
                **(
                    {"oracle_label": records[policy][i]["oracle_label"]}
                    if i in labels
                    else {}
                ),
            }
            for i in ids
        ]

    # Both policy scores cost $0.01 each per common prompt; labels cost $1 each.
    # Illustrative costs, not market quotes; exclude sunk pilot and audit costs.
    costs = CostModel(surrogate_cost=0.02, oracle_cost=1.0)
    design = {
        "config": CONFIG,
        "analysis_config": ANALYSIS_CONFIG,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "costs": asdict(costs),
        "frame_ids": frame,
        "pilot_ids": pilot_ids,
        "heldout_pool_ids": heldout_pool,
        "pilot_label_probability": CONFIG["pilot_m"] / CONFIG["pilot_n"],
        "unit": "one shared prompt; two scores plus selected base references",
        "reference_provenance": "Bundled README attributes references to GPT-5, not humans.",
        "population": "Fixed cached-base-labeled common-prompt frame, not all Arena.",
        "assumptions": [
            "SRS label masking within each declared frame",
            "Base pilot variance is a proxy for both policies; independent-policy normal planning",
            "Final comparison uses its emitted paired finite-sample inference, not planned SE",
            "Calibration transport is assumed, not established; no transport probes consumed",
        ],
        "sources": {
            p: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for p, path in sources.items()
        },
        "versions": {
            p: version(p)
            for p in ["cje-eval", "numpy", "scipy", "scikit-learn", "pydantic"]
        },
        "python": platform.python_version(),
    }
    dump(output_dir / "design.json", design)  # Persist choices before fitting.
    pilot_rows = rows("base", pilot_ids, pilot_labels)
    dump(output_dir / "pilot_input.json", pilot_rows)
    pilot = FreshDrawDataset(
        target_policy="base",
        samples=[
            FreshDrawSample(target_policy="base", draw_idx=0, **row)
            for row in pilot_rows
        ],
    )
    vm = fit_variance_model(
        pilot,
        **{
            k: CONFIG[k]
            for k in ["n_grid", "oracle_fraction_grid", "n_replicates", "seed"]
        },
    )
    dump(output_dir / "pilot_fit.json", {**asdict(vm), "fit_ok": vm.fit_ok})
    if not vm.fit_ok:
        raise ValueError(
            "Poor pilot fit: inspect run.log and collect a better pilot; no plan executed."
        )
    common = dict(
        variance_model=vm,
        cost_model=costs,
        power=CONFIG["power"],
        alpha=CONFIG["alpha"],
        m_min=CONFIG["m_min"],
    )
    plan = plan_evaluation(budget=CONFIG["budget"], **common)
    target_plan = plan_for_mde(target_mde=CONFIG["target_effect"], **common)
    planning = {
        "budget_plan": plan.to_dict(),
        "target_plan": target_plan.to_dict(),
        "projected_power_at_target_effect": plan.power_to_detect(
            CONFIG["target_effect"]
        ),
        "target_fits_budget": target_plan.total_cost <= CONFIG["budget"],
        "target_fits_available_prompts": target_plan.n_samples <= len(heldout_pool),
    }
    dump(output_dir / "planning.json", planning)
    if plan.n_samples > len(heldout_pool) or plan.m_oracle > plan.n_samples:
        raise ValueError(
            "Budget plan exceeds held-out availability; collect data, do not silently clamp."
        )
    # Demonstrate the budget-constrained plan even when it cannot meet the target.
    eval_ids = rng.choice(heldout_pool, plan.n_samples, replace=False).tolist()
    labels = set(rng.choice(eval_ids, plan.m_oracle, replace=False).tolist())
    inputs = {p: rows(p, eval_ids, labels if p == "base" else set()) for p in POLICIES}
    dump(output_dir / "analysis_input.json", inputs)
    dump(
        output_dir / "selection.json",
        {
            "eval_ids": eval_ids,
            "label_ids": sorted(labels),
            "label_probability_within_evaluation": plan.m_oracle / plan.n_samples,
            "evaluation_probability_within_heldout": plan.n_samples / len(heldout_pool),
            "pilot_evaluation_overlap": len(set(pilot_ids) & set(eval_ids)),
        },
    )
    result = analyze_dataset(fresh_draws_data=inputs, **ANALYSIS_CONFIG, verbose=True)
    comparison = result.compare_policies(
        result.target_policies.index(POLICIES[1]),
        result.target_policies.index("base"),
        alpha=CONFIG["alpha"],
    )
    dump(output_dir / "result.json", result.to_dict())
    audit = {
        "planning": planning,
        "comparison": comparison,
        "diagnostics": result.diagnostics.summary(),
        "reliability_gates": result.metadata.get("reliability_gates"),
        "transport_audits": result.metadata.get("transport_audits"),
        "limitations": result.metadata.get("limitations"),
        "power_validated": False,
        "coverage_validated": False,
        "decision": "Budget replay only; do not infer validated power, coverage, transport, or a deploy winner.",
    }
    dump(output_dir / "audit.json", audit)
    print(result.summary())
    print(json.dumps(comparison, indent=2))
    print(
        f"Target fits budget: {planning['target_fits_budget']}; projected power: "
        f"{planning['projected_power_at_target_effect']:.1%} (not measured power)"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "examples/arena_sample/fresh_draws",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data-dir",
        str(args.data_dir.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
    ]
    (args.output_dir / "rerun.sh").write_text(shlex.join(command) + "\n")
    handler = logging.FileHandler(args.output_dir / "run.log", mode="w")
    logging.getLogger().addHandler(handler)
    try:
        with (
            (args.output_dir / "console.log").open("w") as log,
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            try:
                with redirect_stdout(log), redirect_stderr(log):
                    run(args.data_dir, args.output_dir)
            finally:
                dump(
                    args.output_dir / "warnings.json",
                    [
                        {"category": w.category.__name__, "message": str(w.message)}
                        for w in caught
                    ],
                )
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    print(
        f"Saved planning, numeric replay, comparison, and diagnostics to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
