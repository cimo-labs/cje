"""
Smoke test for calibration_data_path and oracle combining functionality.

This test validates that the new calibration_data_path parameter works
end-to-end without crashing.
"""

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cje import analyze_dataset


def test_calibration_data_path_direct_mode(tmp_path: Path) -> None:
    """Test calibration_data_path works in Direct mode (fresh draws only)."""

    # Create calibration data (30 samples with oracle labels)
    calibration_data = []
    for i in range(30):
        calibration_data.append(
            {
                "prompt_id": f"calib_{i}",
                "prompt": f"Calib question {i}",
                "response": f"Calib answer {i}",
                "base_policy_logprob": -10.0,
                "target_policy_logprobs": {"policy_a": -9.5},
                "judge_score": 0.4 + i * 0.01,
                "oracle_label": 0.5 + i * 0.01,
            }
        )

    # Create fresh draws directory with responses for policy_a
    fresh_draws_dir = tmp_path / "fresh_draws"
    fresh_draws_dir.mkdir()

    # Create fresh draw responses (20 samples, 10 with oracle labels)
    fresh_draw_responses = []
    for i in range(20):
        fresh_draw_responses.append(
            {
                "prompt_id": f"eval_{i}",
                "response": f"Fresh answer {i}",
                "judge_score": 0.6 + i * 0.01,
                "oracle_label": (
                    0.65 + i * 0.01 if i < 10 else None
                ),  # 50% oracle coverage
            }
        )

    # Write fresh draws (use correct filename pattern)
    policy_a_file = fresh_draws_dir / "policy_a_responses.jsonl"
    with open(policy_a_file, "w") as f:
        for item in fresh_draw_responses:
            f.write(json.dumps(item) + "\n")

    # Write calibration data
    calib_path = tmp_path / "calibration.jsonl"
    with open(calib_path, "w") as f:
        for item in calibration_data:
            f.write(json.dumps(item) + "\n")

    # Run in Direct mode with calibration_data_path (combine=True)
    results = analyze_dataset(
        fresh_draws_dir=str(fresh_draws_dir),
        calibration_data_path=str(calib_path),
        combine_oracle_sources=True,
        estimator="direct",
        verbose=True,
    )

    # Validate results
    assert results is not None
    assert len(results.estimates) == 1  # One policy
    assert results.estimates[0] >= 0 and results.estimates[0] <= 1

    # Check oracle sources metadata
    assert "oracle_sources" in results.metadata
    oracle_sources = results.metadata["oracle_sources"]

    # Should have combined calibration data + fresh draws oracle labels
    assert oracle_sources["calibration_data"]["n_oracle"] == 30
    assert oracle_sources["fresh_draws"]["n_oracle"] == 10
    assert oracle_sources["total_oracle"] == 40  # 30 + 10

    # Check calibration source in metadata
    assert results.metadata["calibration"] == "from_calibration_data_combined"
    assert results.metadata["mode"] == "direct"

    print("✅ Direct mode with calibration_data_path test passed!")
    print(f"   Combined {oracle_sources['total_oracle']} oracle labels")
    print(f"   From calibration: {oracle_sources['calibration_data']['n_oracle']}")
    print(f"   From fresh draws: {oracle_sources['fresh_draws']['n_oracle']}")


def test_calibration_data_path_direct_mode_no_combining(tmp_path: Path) -> None:
    """Test combine_oracle_sources=False uses only calibration data (Direct mode)."""

    # Create calibration data (30 samples, all with oracle labels)
    calibration_data = [
        {
            "prompt_id": f"calib_{i}",
            "prompt": f"Calib {i}",
            "response": f"Answer {i}",
            "base_policy_logprob": -10.0,
            "target_policy_logprobs": {"policy_a": -9.5},
            "judge_score": 0.4 + i * 0.01,
            "oracle_label": 0.5 + i * 0.01,
        }
        for i in range(30)
    ]

    # Fresh draws with oracle labels that must NOT enter calibration
    fresh_draws_dir = tmp_path / "fresh_draws"
    fresh_draws_dir.mkdir()
    fresh_draw_responses = [
        {
            "prompt_id": f"eval_{i}",
            "response": f"Fresh answer {i}",
            "judge_score": 0.6 + i * 0.01,
            "oracle_label": 0.65 + i * 0.01,
        }
        for i in range(20)
    ]
    policy_a_file = fresh_draws_dir / "policy_a_responses.jsonl"
    with open(policy_a_file, "w") as f:
        for item in fresh_draw_responses:
            f.write(json.dumps(item) + "\n")

    calib_path = tmp_path / "calibration.jsonl"
    with open(calib_path, "w") as f:
        for item in calibration_data:
            f.write(json.dumps(item) + "\n")

    # Run with combine_oracle_sources=False
    results = analyze_dataset(
        fresh_draws_dir=str(fresh_draws_dir),
        calibration_data_path=str(calib_path),
        combine_oracle_sources=False,  # Don't combine - use only calibration data
        estimator="direct",
    )

    # Check only calibration data was used
    assert "oracle_sources" in results.metadata
    oracle_sources = results.metadata["oracle_sources"]

    assert oracle_sources["calibration_data"]["n_oracle"] == 30  # Only calibration
    assert oracle_sources["total_oracle"] == 30  # Not combined
    assert oracle_sources["combine_enabled"] == False
    assert results.metadata["calibration"] == "from_calibration_data_only"

    print("✅ No-combining test passed!")


def test_calibration_data_only_without_eval_oracle_returns_finite_estimates(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Regression for the NaN-bootstrap gap (pre-existing, found by WP3).

    With --calibration-data only and NO oracle labels in the fresh draws,
    the default bootstrap path built an eval table with zero oracle rows:
    the calibrator refit failed, every replicate was skipped, and NaN
    estimates came back quietly. The estimator must now fall back to
    cluster-robust SEs with the oracle jackknife, loudly.
    """
    # Calibration data: old logged data (judge + oracle), 40 samples.
    # The logprob fields are present-and-ignored, as in real 0.3.x logs;
    # load_dataset_from_jsonl still requires target_policy_logprobs to
    # detect policies (pre-existing loader constraint).
    calibration_data = [
        {
            "prompt_id": f"calib_{i}",
            "prompt": f"Calib {i}",
            "response": f"Answer {i}",
            "base_policy_logprob": -10.0,
            "target_policy_logprobs": {"policy_a": -9.5, "policy_b": -9.7},
            "judge_score": round(0.2 + i * 0.015, 4),
            "oracle_label": round(0.25 + i * 0.015, 4),
        }
        for i in range(40)
    ]
    calib_path = tmp_path / "calibration.jsonl"
    calib_path.write_text("\n".join(json.dumps(r) for r in calibration_data) + "\n")

    # Fresh draws: judge scores ONLY (the documented migration workflow)
    fresh_draws_dir = tmp_path / "fresh_draws"
    fresh_draws_dir.mkdir()
    for policy, offset in (("policy_a", 0.5), ("policy_b", 0.4)):
        records = [
            {
                "prompt_id": f"eval_{i}",
                "response": f"{policy} answer {i}",
                "judge_score": round(offset + i * 0.005, 4),
            }
            for i in range(40)
        ]
        (fresh_draws_dir / f"{policy}_responses.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )

    # Default estimator config -> inference_method="bootstrap" (the gap)
    with caplog.at_level(logging.WARNING, logger="cje.estimators.direct_method"):
        results = analyze_dataset(
            fresh_draws_dir=str(fresh_draws_dir),
            calibration_data_path=str(calib_path),
        )

    # Finite estimates and SEs — not the quiet NaNs
    assert np.all(np.isfinite(results.estimates)), results.estimates
    assert np.all(np.isfinite(results.standard_errors)), results.standard_errors
    assert results.method == "calibrated_direct"  # calibrated, not naive

    # The fallback is loud and recorded in metadata
    assert any(
        "falling back to cluster-robust" in record.message for record in caplog.records
    )
    inference = results.metadata["inference"]
    assert inference["method"] == "cluster_robust"
    assert inference["requested_method"] == "bootstrap"
    assert inference["fallback_reason"] == "no_oracle_labels_in_evaluation_data"

    # Oracle (calibration) uncertainty is still included, via the jackknife
    assert results.metadata["se_components"]["includes_oracle_uncertainty"] is True


if __name__ == "__main__":
    # Run smoke test directly
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_calibration_data_path_direct_mode(tmp_path)
        test_calibration_data_path_direct_mode_no_combining(tmp_path)
