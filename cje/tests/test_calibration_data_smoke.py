"""
Smoke test for calibration_data_path and oracle combining functionality.

This test validates that the new calibration_data_path parameter works
end-to-end without crashing.
"""

import json
import tempfile
from pathlib import Path

import pytest

from cje import analyze_dataset


def test_calibration_data_path_smoke(tmp_path: Path) -> None:
    """Smoke test: calibration_data_path loads and combines oracle sources."""

    # Create logged data (100 samples, 10 with oracle labels)
    logged_data = []
    for i in range(100):
        logged_data.append(
            {
                "prompt_id": f"prompt_{i}",
                "prompt": f"Question {i}",
                "response": f"Answer {i}",
                "base_policy_logprob": -10.0,
                "target_policy_logprobs": {"target": -9.5},
                "judge_score": 0.5 + i * 0.001,
                "oracle_label": (
                    0.6 + i * 0.001 if i < 10 else None
                ),  # 10% oracle coverage
            }
        )

    # Create calibration data (50 samples, all with oracle labels)
    calibration_data = []
    for i in range(50):
        calibration_data.append(
            {
                "prompt_id": f"calib_prompt_{i}",
                "prompt": f"Calib question {i}",
                "response": f"Calib answer {i}",
                "base_policy_logprob": -10.0,  # Not used for calibration
                "target_policy_logprobs": {"target": -9.5},  # Not used for calibration
                "judge_score": 0.4 + i * 0.01,
                "oracle_label": 0.5 + i * 0.01,  # 100% oracle coverage
            }
        )

    # Write to temp files
    logged_path = tmp_path / "logged.jsonl"
    with open(logged_path, "w") as f:
        for item in logged_data:
            f.write(json.dumps(item) + "\n")

    calib_path = tmp_path / "calibration.jsonl"
    with open(calib_path, "w") as f:
        for item in calibration_data:
            f.write(json.dumps(item) + "\n")

    # Run analysis with calibration_data_path
    results = analyze_dataset(
        logged_data_path=str(logged_path),
        calibration_data_path=str(calib_path),
        combine_oracle_sources=True,  # Should combine all oracle labels
        estimator="calibrated-ips",
        verbose=True,
    )

    # Validate results
    assert results is not None
    assert len(results.estimates) == 1  # One policy
    assert results.estimates[0] >= 0 and results.estimates[0] <= 1

    # Check oracle_sources metadata was added
    assert "oracle_sources" in results.metadata
    oracle_sources = results.metadata["oracle_sources"]

    # Verify metadata structure
    assert "calibration_data" in oracle_sources
    assert "logged_data" in oracle_sources
    assert "total_oracle" in oracle_sources

    # Verify oracle counts
    assert (
        oracle_sources["calibration_data"]["n_oracle"] == 50
    )  # All calibration samples
    assert oracle_sources["logged_data"]["n_oracle"] == 10  # 10% of logged
    assert oracle_sources["total_oracle"] == 60  # Combined total (50 + 10)

    # Verify priority order metadata
    assert oracle_sources["priority_order"] == [
        "calibration_data",
        "fresh_draws",
        "logged_data",
    ]

    print("✅ Smoke test passed!")
    print(f"   Total oracle labels combined: {oracle_sources['total_oracle']}")
    print(f"   From calibration: {oracle_sources['calibration_data']['n_oracle']}")
    print(f"   From logged: {oracle_sources['logged_data']['n_oracle']}")


def test_calibration_data_path_no_combining(tmp_path: Path) -> None:
    """Test combine_oracle_sources=False uses only calibration data."""

    # Create minimal logged data
    logged_data = [
        {
            "prompt_id": f"prompt_{i}",
            "prompt": f"Question {i}",
            "response": f"Answer {i}",
            "base_policy_logprob": -10.0,
            "target_policy_logprobs": {"target": -9.5},
            "judge_score": 0.5,
            "oracle_label": 0.6,  # Has oracle labels
        }
        for i in range(20)
    ]

    # Create calibration data
    calibration_data = [
        {
            "prompt_id": f"calib_{i}",
            "prompt": f"Calib {i}",
            "response": f"Answer {i}",
            "base_policy_logprob": -10.0,
            "target_policy_logprobs": {"target": -9.5},
            "judge_score": 0.4,
            "oracle_label": 0.5,
        }
        for i in range(30)
    ]

    logged_path = tmp_path / "logged.jsonl"
    with open(logged_path, "w") as f:
        for item in logged_data:
            f.write(json.dumps(item) + "\n")

    calib_path = tmp_path / "calibration.jsonl"
    with open(calib_path, "w") as f:
        for item in calibration_data:
            f.write(json.dumps(item) + "\n")

    # Run with combine_oracle_sources=False
    results = analyze_dataset(
        logged_data_path=str(logged_path),
        calibration_data_path=str(calib_path),
        combine_oracle_sources=False,  # Don't combine - use only calibration data
        estimator="calibrated-ips",
    )

    # Check only calibration data was used
    assert "oracle_sources" in results.metadata
    oracle_sources = results.metadata["oracle_sources"]

    assert oracle_sources["calibration_data"]["n_oracle"] == 30  # Only calibration
    assert oracle_sources["total_oracle"] == 30  # Not combined
    assert oracle_sources["combine_enabled"] == False

    print("✅ No-combining test passed!")


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


if __name__ == "__main__":
    # Run smoke test directly
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_calibration_data_path_smoke(tmp_path)
        test_calibration_data_path_no_combining(tmp_path)
        test_calibration_data_path_direct_mode(tmp_path)
