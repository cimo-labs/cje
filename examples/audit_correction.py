"""Audit a fixed calibration map, then use the sampled labels for correction.

Run with an installed cje-eval, or from this checkout with:
    poetry run python examples/audit_correction.py

All data is synthetic. The candidate's human outcome has a known +0.15 shift
that its judge scores miss. The same probability-sampled slice is used first
to diagnose the old map and then to estimate a residual correction. It is not
an independent validation sample for the corrected estimate.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from cje import TransportAuditConfig, analyze_dataset
from cje.data import EstimationResult


def run_example(
    directory: Path,
) -> tuple[EstimationResult, EstimationResult, dict[str, int]]:
    rng = np.random.default_rng(42)
    # Include both endpoints so this fixture isolates residual bias from
    # unrelated extrapolation beyond the calibration score range.
    fit_scores = np.r_[0.2, rng.uniform(0.2, 0.6, 38), 0.6]
    fit_labels = 0.2 + 0.5 * fit_scores + rng.normal(0, 0.025, 40)
    path = directory / "anchor_calibration.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "prompt_id": f"fit-{i}",
                    "response_id": f"fit-{i}",
                    "judge_score": float(score),
                    "oracle_label": float(label),
                }
            )
            + "\n"
            for i, (score, label) in enumerate(zip(fit_scores, fit_labels))
        )
    )

    scores = rng.uniform(0.2, 0.6, 200)
    human = {
        "baseline": 0.2 + 0.5 * scores + rng.normal(0, 0.025, len(scores)),
        "candidate": 0.2 + 0.5 * scores + 0.15 + rng.normal(0, 0.025, len(scores)),
    }
    # One uniform sample of independent prompt units, paired across policies.
    selected = rng.choice(len(scores), size=60, replace=False)
    draws = {
        policy: [
            {
                "prompt_id": f"eval-{i}",
                "response_id": f"{policy}:eval-{i}",
                "judge_score": float(score),
            }
            for i, score in enumerate(scores)
        ]
        for policy in human
    }
    probes = {
        policy: [
            dict(draws[policy][i], oracle_label=float(labels[i])) for i in selected
        ]
        for policy, labels in human.items()
    }
    audit = TransportAuditConfig(
        probes_by_policy=probes,
        delta_max_by_policy={policy: 0.03 for policy in draws},
        family_size=2,
    )
    options: dict[str, Any] = {
        "calibration_data_path": str(path),
        "combine_oracle_sources": False,  # Preserve the external calibration fit.
        "calibration_judge_scale": (0, 1),
        "calibration_oracle_scale": (0, 1),
        "fresh_judge_scale": (0, 1),
        "fresh_oracle_scale": (0, 1),
        "output_scale": (0, 1),
        "label_design": "representative",  # Justified by the uniform sample above.
        "estimator_config": {
            "inference_method": "cluster_robust",
            "use_augmented_estimator": True,
        },
    }
    # Audit labels alone do not enter the point estimator.
    before = analyze_dataset(fresh_draws_data=draws, transport=audit, **options)
    assert set(before.metadata["point_estimator"]["routes"]) == {"plug_in"}

    # Attach labels to their exact evaluation responses. Do not move them into
    # the calibration fit or replace unobserved labels with predictions.
    labeled_draws = deepcopy(draws)
    for policy, rows in labeled_draws.items():
        labels_by_response = {
            r["response_id"]: r["oracle_label"] for r in probes[policy]
        }
        for row in rows:
            if row["response_id"] in labels_by_response:
                row["oracle_label"] = labels_by_response[row["response_id"]]
    after = analyze_dataset(fresh_draws_data=labeled_draws, **options)
    assert set(after.metadata["point_estimator"]["routes"]) == {"augmented"}
    # The original audit remains in 'before'. We do not present those reused
    # labels as an independent audit of the corrected estimate.
    counts = {
        "calibration_labels": 40,
        "correction_labels": 2 * len(selected),
        "total_human_labels": 40 + 2 * len(selected),
    }
    return before, after, counts


def summarize(result: EstimationResult) -> dict[str, Any]:
    lower, upper = result.confidence_interval()
    point = result.metadata["point_estimator"]
    policies = {
        p: {
            "estimate": float(result.estimates[i]),
            "se": float(result.standard_errors[i]),
            "ci": [float(lower[i]), float(upper[i])],
            "route": point["routes"][i],
            "residual_correction": point["residual_corrections"][i],
        }
        for i, p in enumerate(result.target_policies)
    }
    comparison = result.compare_policies(
        result.target_policies.index("candidate"),
        result.target_policies.index("baseline"),
    )
    return {
        "policies": policies,
        "candidate_minus_baseline": {
            "estimate": float(comparison["difference"]),
            "se": float(comparison["se_difference"]),
            "ci": (
                [comparison["ci_lower"], comparison["ci_upper"]]
                if "ci_lower" in comparison
                else None
            ),
            "method": comparison["method"],
            "gate_flagged": comparison.get("gate_flagged", []),
        },
    }


def main() -> None:
    with TemporaryDirectory() as directory:
        before, after, counts = run_example(Path(directory))
    output = {
        "synthetic_known_candidate_shift": 0.15,
        "human_labels": counts,
        "audit_of_original_map": before.metadata["transport_audits"],
        "audit_only": summarize(before),
        "residual_corrected": summarize(after),
        "independent_validation_of_corrected_estimate": False,
    }
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
