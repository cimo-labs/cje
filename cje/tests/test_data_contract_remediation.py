import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from cje import analyze_dataset
from cje.data.fresh_draws import (
    discover_policies_from_fresh_draws,
    fresh_draws_from_dict,
    load_fresh_draws_auto,
)
from cje.data.ingest import canonicalize_record, fresh_draws_data_from_file
from cje.data.loaders import DatasetLoader, FreshDrawLoader
from cje.data.models import (
    CIInfo,
    Dataset,
    EstimationResult,
    InferenceUnavailableError,
    Sample,
)
from cje.data.normalization import ScaleInfo
from cje.data.validation import validate_direct_data
from cje.interface.analysis import (
    _combine_oracle_sources,
    _scale_oracle_source_metadata,
)
from cje.utils.export import export_results_json


def _cluster_config() -> dict:
    return {"inference_method": "cluster_robust"}


def test_dual_evaluation_sources_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one evaluation source"):
        analyze_dataset(
            fresh_draws_dir=str(tmp_path),
            fresh_draws_data={"a": [{"prompt_id": "p", "judge_score": 0.5}]},
        )


def test_mixed_filename_patterns_are_unioned_and_duplicates_error(
    tmp_path: Path,
) -> None:
    record = json.dumps({"prompt_id": "p", "judge_score": 0.5}) + "\n"
    (tmp_path / "a_responses.jsonl").write_text(record)
    (tmp_path / "b.jsonl").write_text(record)
    assert discover_policies_from_fresh_draws(tmp_path) == ["a", "b"]

    (tmp_path / "responses").mkdir()
    (tmp_path / "responses" / "a.jsonl").write_text(record)
    with pytest.raises(ValueError, match="same policy"):
        discover_policies_from_fresh_draws(tmp_path)


def test_explicit_calibration_path_is_excluded_before_policy_ambiguity(
    tmp_path: Path,
) -> None:
    evaluation = tmp_path / "a_responses.jsonl"
    evaluation.write_text(
        "\n".join(
            json.dumps({"prompt_id": f"p{i}", "judge_score": i / 3}) for i in range(4)
        )
        + "\n"
    )
    calibration = tmp_path / "a.jsonl"
    calibration.write_text(
        "\n".join(
            json.dumps(
                {
                    "prompt_id": f"c{i}",
                    "judge_score": i / 3,
                    "oracle_label": i / 3,
                }
            )
            for i in range(4)
        )
        + "\n"
    )
    assert discover_policies_from_fresh_draws(
        tmp_path, exclude_paths=[calibration]
    ) == ["a"]
    result = analyze_dataset(
        fresh_draws_dir=str(tmp_path),
        calibration_data_path=str(calibration),
        estimator_config=_cluster_config(),
    )
    assert result.metadata["calibration_status"] == "CALIBRATED"


def test_custom_fields_and_covariate_work_end_to_end() -> None:
    records = []
    for i in range(20):
        records.append(
            {
                "prompt_id": f"p{i}",
                "score": i / 19,
                "human": i / 19,
                "domain_indicator": i % 2,
            }
        )
    result = analyze_dataset(
        fresh_draws_data={"a": records},
        judge_field="score",
        oracle_field="human",
        calibration_covariates=["domain_indicator"],
        estimator_config=_cluster_config(),
    )
    assert result.metadata["calibration_status"] == "CALIBRATED"
    assert result.metadata["data_provenance"]["covariates"] == ["domain_indicator"]
    assert result.estimates[0] == pytest.approx(0.5, abs=0.05)


def test_external_calibration_controls_output_units(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(
        "\n".join(
            json.dumps(
                {
                    "prompt_id": f"c{i}",
                    "judge_score": i / 19,
                    "oracle_label": i / 19,
                }
            )
            for i in range(20)
        )
        + "\n"
    )
    fresh = {
        "a": [
            {"prompt_id": f"p{i}", "judge_score": 20 + 60 * i / 19} for i in range(20)
        ]
    }
    result = analyze_dataset(
        fresh_draws_data=fresh,
        calibration_data_path=str(calibration),
        combine_oracle_sources=False,
        estimator_config=_cluster_config(),
    )
    assert result.estimates[0] == pytest.approx(0.5, abs=0.05)
    assert result.diagnostics is not None
    assert result.calibrator is not None
    assert result.units is not None
    assert result.diagnostics.estimates["a"] == pytest.approx(result.estimates[0])
    np.testing.assert_allclose(result.calibrator.predict([20, 80]), [0, 1])
    assert result.units.estimand == "oracle_mean"

    percent_result = analyze_dataset(
        fresh_draws_data=fresh,
        calibration_data_path=str(calibration),
        combine_oracle_sources=False,
        output_scale=(0, 100),
        estimator_config=_cluster_config(),
    )
    assert percent_result.estimates[0] == pytest.approx(50, abs=5)
    assert percent_result.diagnostics is not None
    assert percent_result.diagnostics.calibration_tolerance == pytest.approx(10.0)
    assert "±10" in percent_result.diagnostics.summary()
    assert percent_result.diagnostics is not None
    assert percent_result.calibrator is not None
    assert percent_result.diagnostics.estimates["a"] == pytest.approx(
        percent_result.estimates[0]
    )
    assert percent_result.metadata["point_estimator"]["plug_in_estimates"][
        0
    ] == pytest.approx(percent_result.estimates[0])
    np.testing.assert_allclose(
        percent_result.calibrator.predict([20, 80]), [0, 100], atol=1e-8
    )
    assert percent_result.calibrator.get_calibration_info()[
        "oof_rmse"
    ] == pytest.approx(percent_result.diagnostics.calibration_rmse)


def test_empty_external_labels_use_contributing_fresh_oracle_scale(
    tmp_path: Path,
) -> None:
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(
        "\n".join(
            json.dumps({"prompt_id": f"c{i}", "judge_score": i / 3}) for i in range(4)
        )
        + "\n"
    )
    records = [
        {
            "prompt_id": f"p{i}",
            "judge_score": 100 * i / 3,
            "oracle_label": 1 + 4 * i / 3,
        }
        for i in range(4)
    ]
    result = analyze_dataset(
        fresh_draws_data={"a": records},
        calibration_data_path=str(calibration),
        fresh_judge_scale=(0, 100),
        fresh_oracle_scale=(1, 5),
        estimator_config=_cluster_config(),
    )
    assert result.estimates[0] == pytest.approx(3.0)
    assert result.units is not None
    assert result.units.output_scale["min"] == 1.0
    assert result.units.output_scale["max"] == 5.0


def test_falsy_explicit_row_and_source_ids_are_preserved() -> None:
    record = canonicalize_record(
        {"prompt_id": "p", "judge_score": 0.5, "row_id": 0, "source_id": 0},
        0,
        source_id="fallback",
    )
    assert record["row_id"] == "0"
    assert record["source_id"] == "0"


def test_invalid_typed_fresh_draw_can_drop_or_error() -> None:
    records = [
        {"prompt_id": "p0", "judge_score": 0.4, "draw_idx": 0},
        {"prompt_id": "p1", "judge_score": 0.6, "draw_idx": -1},
    ]
    datasets, _ = fresh_draws_from_dict({"a": records}, on_invalid="drop")
    assert datasets["a"].n_samples == 1
    with pytest.raises(ValueError, match="Invalid fresh draw record 1"):
        fresh_draws_from_dict({"a": records}, on_invalid="error")


def test_draw_indices_are_unique_when_explicit_and_implicit_values_mix() -> None:
    datasets, _ = fresh_draws_from_dict(
        {
            "a": [
                {"prompt_id": "p", "judge_score": 0.4, "draw_idx": 1},
                {"prompt_id": "p", "judge_score": 0.6},
            ]
        }
    )
    assert [sample.draw_idx for sample in datasets["a"].samples] == [1, 0]


def test_lower_level_fresh_loaders_share_draw_identity_contract(
    tmp_path: Path,
) -> None:
    records = [
        {"prompt_id": "p", "judge_score": 0.4},
        {"prompt_id": "p", "judge_score": 0.6},
    ]
    (tmp_path / "a.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    per_policy = load_fresh_draws_auto(tmp_path, "a")
    assert [sample.draw_idx for sample in per_policy.samples] == [0, 1]

    combined = tmp_path / "combined.jsonl"
    combined.write_text(
        "\n".join(json.dumps({**record, "target_policy": "a"}) for record in records)
        + "\n"
    )
    loaded = FreshDrawLoader.load_from_jsonl(str(combined))
    assert [sample.draw_idx for sample in loaded["a"].samples] == [0, 1]


def test_drop_mode_never_removes_an_entire_policy_silently() -> None:
    with pytest.raises(ValueError, match="Policy 'bad' has no valid"):
        fresh_draws_from_dict(
            {
                "good": [{"prompt_id": "p", "judge_score": 0.5}],
                "bad": [{"prompt_id": "p"}],
            },
            on_invalid="drop",
        )


def test_directory_canonicalization_is_idempotent_for_nested_numeric_strings(
    tmp_path: Path,
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt_id": "p",
                "metadata": {"judge_score": "0.5"},
            }
        )
        + "\n"
    )
    result = analyze_dataset(
        fresh_draws_dir=str(tmp_path), estimator_config=_cluster_config()
    )
    assert result.estimates[0] == pytest.approx(0.5)


def test_single_file_loader_honors_drop_vs_error(tmp_path: Path) -> None:
    path = tmp_path / "draws.jsonl"
    path.write_text(
        "\n{bad json\n"
        + json.dumps({"target_policy": "a", "prompt_id": "p", "judge_score": 0.5})
        + "\n"
    )
    loaded = fresh_draws_data_from_file(path, on_invalid="drop")
    assert len(loaded["a"]) == 1
    assert loaded["a"][0]["_cje_row_id"].endswith(":line:3")
    with pytest.raises(ValueError, match="Invalid JSON"):
        fresh_draws_data_from_file(path, on_invalid="error")


def test_combining_keeps_distinct_same_value_responses() -> None:
    samples = [
        Sample(
            prompt_id="shared",
            prompt="",
            response=f"response {i}",
            reward=None,
            judge_score=0.5,
            oracle_label=1.0,
            row_id=f"row-{i}",
            source_id="external",
        )
        for i in range(10)
    ]
    combined, metadata = _combine_oracle_sources(
        Dataset(samples=samples), None, None, [], "judge_score", "oracle_label"
    )
    assert combined.n_samples == 10
    assert metadata["n_duplicates"] == 0


def test_combining_deduplicates_only_explicit_row_identity() -> None:
    sample = Sample(
        prompt_id="shared",
        prompt="",
        response="response",
        reward=None,
        judge_score=0.5,
        oracle_label=1.0,
        row_id="row-1",
        source_id="external",
    )
    combined, metadata = _combine_oracle_sources(
        Dataset(samples=[sample, sample.model_copy(deep=True)]),
        None,
        None,
        [],
        "judge_score",
        "oracle_label",
    )
    assert combined.n_samples == 1
    assert metadata["n_duplicates"] == 1


def test_fresh_ingestion_deduplicates_explicit_identity_without_value_dedupe() -> None:
    base = {
        "prompt_id": "p",
        "judge_score": 0.5,
        "source_id": "source",
        "row_id": "row",
    }
    datasets, _ = fresh_draws_from_dict({"a": [base, dict(base)]})
    assert datasets["a"].n_samples == 1
    with pytest.raises(ValueError, match="Conflicting fresh-draw records"):
        fresh_draws_from_dict({"a": [base, {**base, "judge_score": 0.7}]})


def test_same_row_identity_with_different_covariates_is_a_conflict() -> None:
    sample = Sample(
        prompt_id="shared",
        prompt="",
        response="response",
        reward=None,
        judge_score=0.5,
        oracle_label=1.0,
        row_id="row-1",
        source_id="external",
        metadata={"domain": 0.0},
    )
    changed = sample.model_copy(update={"metadata": {"domain": 1.0}})
    with pytest.raises(ValueError, match="Conflicting records share row_id"):
        _combine_oracle_sources(
            Dataset(samples=[sample, changed]),
            None,
            None,
            [],
            "judge_score",
            "oracle_label",
        )


def test_oracle_conflict_metadata_uses_public_output_units() -> None:
    metadata: Dict[str, Any] = {
        "conflicts": [{"existing_value": 0.2, "new_value": 0.8, "difference": 0.6}]
    }
    _scale_oracle_source_metadata(metadata, ScaleInfo(0, 100))
    conflict = metadata["conflicts"][0]
    assert conflict["existing_value"] == pytest.approx(20)
    assert conflict["new_value"] == pytest.approx(80)
    assert conflict["difference"] == pytest.approx(60)
    assert metadata["oracle_value_scale"]["max"] == 100


def test_validation_scans_beyond_first_hundred_rows() -> None:
    records = [
        {"prompt_id": f"p{i}", "judge_score": 0.5, "oracle_label": 0.5}
        for i in range(101)
    ]
    records[-1].pop("judge_score")
    valid, issues = validate_direct_data(records)
    assert not valid
    assert any("1/101 records" in issue for issue in issues)


def test_validation_reports_alias_conflicts_without_raising() -> None:
    valid, issues = validate_direct_data(
        [
            {
                "prompt_id": "p",
                "judge_score": 0.5,
                "oracle_label": 0.5,
                "metadata": {"judge_score": 0.7},
            }
        ]
    )
    assert not valid
    assert any("Conflicting field 'judge_score'" in issue for issue in issues)


def test_uncalibrated_result_is_typed_and_graded() -> None:
    result = analyze_dataset(
        fresh_draws_data={
            "a": [{"prompt_id": f"p{i}", "judge_score": 20 + i} for i in range(10)]
        },
        estimator_config=_cluster_config(),
    )
    assert result.metadata["calibration_status"] == "UNCALIBRATED"
    assert result.metadata["claim_tier"] == "RAW_JUDGE_MEAN"
    assert result.gates["a"].flagged
    assert result.best_policy().all_flagged
    assert result.estimates[0] > 1
    assert result.metadata["point_estimator"]["plug_in_estimates"][0] == pytest.approx(
        result.estimates[0]
    )


def test_three_labeled_clusters_return_graded_raw_tier() -> None:
    records = [
        {
            "prompt_id": f"p{i}",
            "judge_score": i / 9,
            **({"oracle_label": i / 9} if i < 3 else {}),
        }
        for i in range(10)
    ]
    result = analyze_dataset(
        fresh_draws_data={"a": records}, estimator_config=_cluster_config()
    )
    assert result.method == "naive_direct"
    assert result.metadata["calibration_status"] == "UNCALIBRATED"
    reason = result.metadata["reliability_gates"]["a"]["reasons"][0]
    assert "Only 3 independent" in reason
    assert "at least 4" in reason


def test_three_fully_labeled_clusters_use_direct_oracle_mean() -> None:
    result = analyze_dataset(
        fresh_draws_data={
            "a": [
                {"prompt_id": "p0", "judge_score": 0, "oracle_label": 5},
                {"prompt_id": "p1", "judge_score": 50, "oracle_label": 3},
                {"prompt_id": "p2", "judge_score": 100, "oracle_label": 1},
            ]
        },
        fresh_judge_scale=(0, 100),
        fresh_oracle_scale=(1, 5),
        estimator_config=_cluster_config(),
    )
    assert result.estimates[0] == pytest.approx(3.0)
    assert result.method == "direct_oracle"
    assert result.metadata["calibration_status"] == "DIRECT_ORACLE"
    assert result.metadata["claim_tier"] == "DIRECT_ORACLE_MEAN"
    assert result.metadata["point_estimator"]["routes"] == ["direct_oracle"]
    assert not result.best_policy().flagged
    assert result.units is not None
    assert result.units.estimand == "oracle_mean"
    assert result.units.output_scale["min"] == 1.0
    assert result.units.output_scale["max"] == 5.0


def test_zero_label_calibration_file_falls_back_without_crashing(
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "calibration.jsonl"
    calibration_path.write_text(
        json.dumps(
            {
                "prompt_id": "cal-0",
                "prompt": "prompt",
                "response": "response",
                "judge_score": 0.5,
            }
        )
        + "\n"
    )

    result = analyze_dataset(
        fresh_draws_data={
            "a": [{"prompt_id": f"p{i}", "judge_score": i / 9} for i in range(10)]
        },
        calibration_data_path=str(calibration_path),
        estimator_config=_cluster_config(),
    )

    assert result.method == "naive_direct"
    assert result.metadata["calibration_status"] == "UNCALIBRATED"
    assert result.metadata["claim_tier"] == "RAW_JUDGE_MEAN"
    assert result.metadata["oracle_sources"]["total_oracle"] == 0
    assert result.gates["a"].flagged


def test_insufficient_labels_still_validate_requested_covariates() -> None:
    records = [
        {
            "prompt_id": f"p{i}",
            "judge_score": i / 9,
            **({"oracle_label": i / 9} if i < 3 else {}),
        }
        for i in range(10)
    ]
    with pytest.raises(ValueError, match="Covariate 'domain' is missing"):
        analyze_dataset(
            fresh_draws_data={"a": records},
            calibration_covariates=["domain"],
            estimator_config=_cluster_config(),
        )


def test_result_validation_and_versioned_serialization() -> None:
    with pytest.raises(ValueError, match="same length"):
        EstimationResult(
            estimates=np.array([0.5]),
            standard_errors=np.array([0.1, 0.1]),
            n_samples_used={"a": 10},
            method="direct",
            influence_functions=None,
            diagnostics=None,
        )

    result = EstimationResult(
        estimates=np.array([0.5, 0.6]),
        standard_errors=np.array([0.1, 0.1]),
        n_samples_used={"a": 10, "b": 10},
        method="direct",
        influence_functions=None,
        diagnostics=None,
        metadata={
            "target_policies": ["a", "b"],
            "nested": {"array": np.array([1, 2])},
            "numpy_bool": np.bool_(True),
            "started_at": datetime(2026, 7, 18, 12, 30, 45),
            "run_date": date(2026, 7, 18),
            "wall_time": time(1, 2, 3),
        },
    )
    portable_payload = result.to_dict(detail="portable")
    json.dumps(portable_payload)
    restored = EstimationResult.from_dict(portable_payload)
    assert restored.metadata["nested"]["array"] == [1, 2]
    assert restored.metadata["numpy_bool"] is True
    assert restored.metadata["started_at"] == "2026-07-18T12:30:45"
    assert restored.metadata["run_date"] == "2026-07-18"
    assert restored.metadata["wall_time"] == "01:02:03"
    assert restored.compare_policies(0, 1)["method"] == "independent_conservative"
    assert (
        restored.compare_policies(0, 1, alpha=0.10)["method"]
        == "independent_conservative"
    )
    with pytest.raises(IndexError, match="out of range"):
        result.compare_policies(-1, 0)
    with pytest.raises(ValueError, match="alpha"):
        result.compare_policies(0, 1, alpha=1.0)
    lower_95, upper_95 = restored.confidence_interval(alpha=0.05)
    lower_90, upper_90 = restored.confidence_interval(alpha=0.10)
    assert np.all((upper_90 - lower_90) < (upper_95 - lower_95))

    summary = EstimationResult.from_dict(result.to_dict(detail="summary"))
    with pytest.raises(InferenceUnavailableError):
        summary.compare_policies(0, 1)

    reloaded = EstimationResult.from_dict(summary.to_dict(detail="full"))
    with pytest.raises(InferenceUnavailableError):
        reloaded.compare_policies(0, 1)

    with pytest.raises(ValueError, match="must be an integer"):
        EstimationResult(
            estimates=np.array([0.5]),
            standard_errors=np.array([0.1]),
            n_samples_used={"a": True},
            method="direct",
            influence_functions=None,
            diagnostics=None,
            metadata={"target_policies": ["a"]},
        )

    invalid_payload = result.to_dict()
    invalid_payload["n_samples_used"]["a"] = True
    with pytest.raises(ValueError, match="must be an integer"):
        EstimationResult.from_dict(invalid_payload)


def test_serialization_preserves_percentile_interval_alpha() -> None:
    result = EstimationResult(
        estimates=np.array([0.5]),
        standard_errors=np.array([0.1]),
        n_samples_used={"a": 10},
        method="direct",
        influence_functions=None,
        diagnostics=None,
        metadata={"target_policies": ["a"]},
        ci_info=CIInfo(
            method="percentile",
            alpha=0.10,
            lower=[0.35],
            upper=[0.65],
        ),
    )

    payload = result.to_dict()
    assert payload["confidence_intervals"]["alpha"] == pytest.approx(0.10)
    assert payload["confidence_intervals"]["lower"] == [0.35]
    assert payload["confidence_intervals"]["upper"] == [0.65]

    restored = EstimationResult.from_dict(payload)
    assert restored.ci_info is not None
    assert restored.ci_info.alpha == pytest.approx(0.10)
    lower, upper = restored.confidence_interval(alpha=0.10)
    np.testing.assert_allclose(lower, [0.35])
    np.testing.assert_allclose(upper, [0.65])


def test_export_without_metadata_keeps_policy_identity(
    tmp_path: Path,
) -> None:
    result = EstimationResult(
        estimates=np.array([0.5]),
        standard_errors=np.array([0.1]),
        n_samples_used={"a": 10},
        method="direct",
        influence_functions=None,
        diagnostics=None,
        metadata={"target_policies": ["a"]},
    )
    path = tmp_path / "result.json"
    export_results_json(result, str(path), include_metadata=False)
    restored = EstimationResult.from_json(path)
    assert restored.target_policies == ["a"]


def test_custom_reward_field_is_preserved() -> None:
    loader = DatasetLoader(reward_field="human_reward", on_invalid="error")
    dataset = loader._convert_raw_data(
        [
            {
                "prompt_id": "p",
                "judge_score": 0.5,
                "human_reward": 0.75,
            }
        ]
    )
    assert dataset.samples[0].reward == pytest.approx(0.75)
