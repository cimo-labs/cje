"""Regression tests: calibration_data_path must not replace the estimand.

A prior implementation built the PrecomputedSampler from the calibration (or
combined-oracle) pool whenever calibration_data_path was supplied. The
combined pool carries fabricated logprobs (-1.0 for base and every target),
so all importance weights were exp(0)=1 and every policy received the
identical estimate — the mean calibrated reward over the oracle pool — while
diagnostics reported a perfect ESS. The calibration pool must be used ONLY to
fit the calibrator; estimation runs on the logged evaluation dataset.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from cje import analyze_dataset

N_LOGGED = 200
N_CALIB = 90


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def datasets(tmp_path: Path) -> Dict[str, Path]:
    rng = np.random.default_rng(42)

    logged = []
    for i in range(N_LOGGED):
        quality = float(rng.uniform(0, 1))
        judge = float(np.clip(quality + rng.normal(0, 0.1), 0, 1))
        logged.append(
            {
                "prompt_id": f"prompt_{i}",
                "prompt": f"Question {i}",
                "response": f"Answer {i}",
                "base_policy_logprob": -10.0,
                # policy_hi upweights high-quality responses, policy_lo the reverse
                "target_policy_logprobs": {
                    "policy_hi": -10.0 + 4.0 * (quality - 0.5),
                    "policy_lo": -10.0 - 4.0 * (quality - 0.5),
                },
                "judge_score": judge,
                "oracle_label": quality if i < 40 else None,
            }
        )

    calibration = []
    for i in range(N_CALIB):
        quality = float(rng.uniform(0, 1))
        judge = float(np.clip(quality + rng.normal(0, 0.1), 0, 1))
        calibration.append(
            {
                "prompt_id": f"calib_{i}",
                "prompt": f"Calib question {i}",
                "response": f"Calib answer {i}",
                "base_policy_logprob": -10.0,
                "target_policy_logprobs": {"policy_hi": -10.0, "policy_lo": -10.0},
                "judge_score": judge,
                "oracle_label": quality,
            }
        )

    logged_path = tmp_path / "logged.jsonl"
    calib_path = tmp_path / "calibration.jsonl"
    _write_jsonl(logged_path, logged)
    _write_jsonl(calib_path, calibration)
    return {"logged": logged_path, "calibration": calib_path}


@pytest.mark.parametrize("combine", [True, False])
def test_estimation_runs_on_logged_dataset(
    datasets: Dict[str, Path], combine: bool
) -> None:
    results = analyze_dataset(
        logged_data_path=str(datasets["logged"]),
        calibration_data_path=str(datasets["calibration"]),
        combine_oracle_sources=combine,
        estimator="calibrated-ips",
    )

    # Policies with opposite logprob tilts must NOT collapse to one number
    est_by_policy = dict(zip(results.metadata["target_policies"], results.estimates))
    diff = abs(est_by_policy["policy_hi"] - est_by_policy["policy_lo"])
    assert diff > 0.01, (
        f"policy_hi and policy_lo estimates identical ({est_by_policy}); "
        "estimation likely ran on the calibration pool with dummy logprobs"
    )
    # And the tilt direction must be recovered
    assert est_by_policy["policy_hi"] > est_by_policy["policy_lo"]

    # Influence functions must cover the logged dataset, not the oracle pool
    for policy, ifs in (results.influence_functions or {}).items():
        assert len(ifs) == N_LOGGED, (
            f"{policy}: IF length {len(ifs)} != n_logged {N_LOGGED}; "
            "wrong dataset was estimated on"
        )


def test_matches_no_calibration_path_ordering(datasets: Dict[str, Path]) -> None:
    """Same logged data with/without an external calibration pool must agree
    on the policy ordering (the calibrator differs, the estimand does not)."""
    with_pool = analyze_dataset(
        logged_data_path=str(datasets["logged"]),
        calibration_data_path=str(datasets["calibration"]),
        estimator="calibrated-ips",
    )
    without_pool = analyze_dataset(
        logged_data_path=str(datasets["logged"]),
        estimator="calibrated-ips",
    )
    order_with = np.argsort(with_pool.estimates).tolist()
    order_without = np.argsort(without_pool.estimates).tolist()
    assert order_with == order_without
    # Estimates on the same estimand should be in the same ballpark
    np.testing.assert_allclose(with_pool.estimates, without_pool.estimates, atol=0.15)
