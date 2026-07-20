"""
Smoke test for calibration_data_path and oracle combining functionality.

This test validates that the new calibration_data_path parameter works
end-to-end without crashing.
"""

import json
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
    tmp_path: Path,
) -> None:
    """External calibration rows support joint bootstrap inference.

    With --calibration-data only and NO oracle labels in the fresh draws,
    exact calibration provenance lets the bootstrap resample and refit the
    calibration source independently from the evaluation prompt clusters.
    """
    # Calibration data: old logged data (judge + oracle), 40 samples.
    # The logprob fields are present-and-ignored, as in real 0.3.x logs.
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

    # Supplying only n_bootstrap retains the default bootstrap inference route.
    results = analyze_dataset(
        fresh_draws_dir=str(fresh_draws_dir),
        calibration_data_path=str(calib_path),
        estimator_config={"n_bootstrap": 50},
    )

    # Finite estimates and SEs — not the quiet NaNs
    assert np.all(np.isfinite(results.estimates)), results.estimates
    assert np.all(np.isfinite(results.standard_errors)), results.standard_errors
    assert results.method == "calibrated_direct_bootstrap"

    inference = results.metadata["inference"]
    assert inference["method"] == "cluster_bootstrap_refit"
    assert inference["bootstrap_scheme"] == "positive_exponential_cluster_weights"
    assert inference["n_bootstrap_requested"] == 50
    assert inference["n_bootstrap_valid"] == 50
    assert inference["skip_rate"] == 0.0
    assert results.metadata["calibration_provenance_explicit"] is True

    # Joint bootstrap inference includes finite calibration-sample uncertainty.
    assert results.metadata["se_components"]["includes_oracle_uncertainty"] is True


def test_minimal_calibration_file_loads_and_analyzes(tmp_path: Path) -> None:
    """A minimal judge+oracle calibration file works end-to-end.

    Regression: the documented migration ("pass your logged data — or a
    minimal judge+oracle file — as calibration_data_path") used to crash for
    files without logprob fields: no target_policy_logprobs meant
    target_policies=[] failed Dataset validation ('too_short'), and records
    without prompt/response were skipped outright — even though `cje
    validate` blessed the file as a calibration source.
    """
    from cje import load_dataset_from_jsonl

    # The minimal documented schema: prompt_id + judge_score + oracle_label.
    # No prompt, no response, no logprob fields.
    calibration_data = [
        {
            "prompt_id": f"calib_{i}",
            "judge_score": round(0.2 + i * 0.02, 4),
            "oracle_label": round(0.25 + i * 0.02, 4),
        }
        for i in range(30)
    ]
    calib_path = tmp_path / "minimal_calibration.jsonl"
    calib_path.write_text("\n".join(json.dumps(r) for r in calibration_data) + "\n")

    # Loads directly — every record kept, no policies detected
    dataset = load_dataset_from_jsonl(str(calib_path))
    assert dataset.n_samples == 30
    assert dataset.target_policies == []
    assert dataset.samples[0].judge_score == pytest.approx(0.2)
    assert dataset.samples[0].oracle_label == pytest.approx(0.25)

    # And works as the calibration source for a Direct-mode analysis
    fresh_draws_dir = tmp_path / "fresh_draws"
    fresh_draws_dir.mkdir()
    for policy, offset in (("policy_a", 0.5), ("policy_b", 0.4)):
        records = [
            {
                "prompt_id": f"eval_{i}",
                "judge_score": round(offset + i * 0.005, 4),
            }
            for i in range(40)
        ]
        (fresh_draws_dir / f"{policy}_responses.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )

    results = analyze_dataset(
        fresh_draws_dir=str(fresh_draws_dir),
        calibration_data_path=str(calib_path),
        estimator_config={"n_bootstrap": 50},
    )

    assert len(results.estimates) == 2
    assert np.all(np.isfinite(results.estimates)), results.estimates
    assert np.all(np.isfinite(results.standard_errors)), results.standard_errors
    assert results.method == "calibrated_direct_bootstrap"
    assert results.metadata["oracle_sources"]["calibration_data"]["n_oracle"] == 30


def test_fit_cv_unsorted_index_mask_keeps_sample_weight_aligned() -> None:
    """Compact sample_weight follows its labels for every oracle_mask form.

    Regression: when oracle_mask was an unsorted integer index array,
    fit_cv reordered the compact labels to ascending-index order but left a
    compact sample_weight in the caller's order, silently misaligning every
    (label, weight) pair.
    """
    from cje.calibration import JudgeCalibrator

    rng = np.random.default_rng(7)
    n_total = 40
    judge_scores = rng.uniform(0.0, 1.0, n_total)
    idx = rng.permutation(n_total)[:20]  # unsorted oracle indices
    assert np.any(np.diff(idx) < 0), "test requires an unsorted index array"
    labels = np.clip(judge_scores[idx] + rng.normal(0.0, 0.2, len(idx)), 0.0, 1.0)
    weights = rng.uniform(0.1, 10.0, len(idx))  # caller order, like the labels

    cal_unsorted = JudgeCalibrator(calibration_mode="monotone")
    res_unsorted = cal_unsorted.fit_cv(
        judge_scores, labels, oracle_mask=idx, sample_weight=weights
    )

    # Equivalent boolean-mask call: labels/weights pre-sorted to the
    # ascending-index order that boolean fancy-indexing produces.
    order = np.argsort(idx)
    bool_mask = np.zeros(n_total, dtype=bool)
    bool_mask[idx] = True
    cal_bool = JudgeCalibrator(calibration_mode="monotone")
    res_bool = cal_bool.fit_cv(
        judge_scores,
        labels[order],
        oracle_mask=bool_mask,
        sample_weight=weights[order],
    )

    # Equivalent sorted-index call (already in ascending-index order).
    cal_sorted = JudgeCalibrator(calibration_mode="monotone")
    res_sorted = cal_sorted.fit_cv(
        judge_scores,
        labels[order],
        oracle_mask=np.sort(idx),
        sample_weight=weights[order],
    )

    # Stored fit weights must be in ascending-index order (aligned with the
    # stored labels), not the caller's unsorted order.
    assert cal_unsorted._fit_sample_weight is not None
    np.testing.assert_array_equal(cal_unsorted._fit_sample_weight, weights[order])

    # All three mask forms describe the same fit and must agree exactly.
    np.testing.assert_allclose(
        res_unsorted.calibrated_scores, res_bool.calibrated_scores
    )
    np.testing.assert_allclose(res_sorted.calibrated_scores, res_bool.calibrated_scores)
    assert res_unsorted.calibration_rmse == pytest.approx(res_bool.calibration_rmse)


def test_calibrate_dataset_records_actual_fold_count_with_clustered_labels() -> None:
    """Calibration metadata records the fold count the cross-fit actually used.

    Regression: calibrate_dataset pre-resolved n_folds from the label count
    only, but fit_cv further reduces folds by unique oracle prompt clusters.
    With 50 labels across 5 prompts and 5 requested folds, the metadata
    claimed 5 folds while the cross-fit used 2.
    """
    from cje.calibration import calibrate_dataset
    from cje.data.models import Dataset, Sample

    rng = np.random.default_rng(11)
    samples = []
    for prompt_i in range(5):  # 5 unique prompts, 10 labeled draws each
        for draw_j in range(10):
            score = float(rng.uniform(0.05, 0.95))
            label = float(np.clip(score + rng.normal(0.0, 0.1), 0.0, 1.0))
            samples.append(
                Sample(
                    prompt_id=f"prompt_{prompt_i}",
                    prompt=f"Question {prompt_i}",
                    response=f"Answer {prompt_i}-{draw_j}",
                    judge_score=score,
                    oracle_label=label,
                    reward=None,
                )
            )
    dataset = Dataset(samples=samples, target_policies=[])

    calibrated, result = calibrate_dataset(
        dataset, n_folds=5, calibration_mode="monotone"
    )

    # 5 unique clusters support only 2 folds (2 clusters per fold), and the
    # cross-fit fold ids must match what the metadata reports.
    assert result.fold_ids is not None
    actual_folds = len(np.unique(result.fold_ids))
    assert actual_folds == 2
    assert calibrated.metadata["calibration_info"]["n_folds"] == actual_folds
    assert calibrated.metadata["n_folds"] == actual_folds


if __name__ == "__main__":
    # Run smoke test directly
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_calibration_data_path_direct_mode(tmp_path)
        test_calibration_data_path_direct_mode_no_combining(tmp_path)
