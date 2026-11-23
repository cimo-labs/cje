"""The external calibration source has the same statistical roles in both formats."""

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cje import TransportAuditConfig, analyze_dataset
from cje.data.loaders import DatasetLoader


def _external() -> list[dict[str, Any]]:
    return [
        {
            "source_id": "historical",
            "row_id": f"historical-{i}",
            "observation_id": f"response-{i}",
            "prompt_id": f"historical-prompt-{i // 2}",
            "metadata": {"score": i % 8, "rating": 1 + i % 5, "x": i % 3},
        }
        for i in range(80)
    ]


@pytest.mark.parametrize("combine", [True, False])
def test_records_match_file_for_pooled_or_external_only_fit(
    tmp_path: Path, combine: bool
) -> None:
    rows = _external()
    original = copy.deepcopy(rows)
    path = tmp_path / "calibration.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    draws = {
        policy: [
            {
                "prompt_id": f"eval-{i}",
                "score": (i + offset) % 8,
                "rating": 1 + i % 5 if offset == 0 and i % 2 == 0 else None,
                "x": i % 3,
            }
            for i in range(60)
        ]
        for policy, offset in [("A", 0), ("B", 2)]
    }
    opts: dict[str, Any] = {
        "fresh_draws_data": draws,
        "combine_oracle_sources": combine,
        "judge_field": "score",
        "oracle_field": "rating",
        "calibration_covariates": ["x"],
        "calibration_judge_scale": (0, 7),
        "fresh_judge_scale": (0, 7),
        "calibration_oracle_scale": (1, 5),
        "fresh_oracle_scale": (1, 5),
    }
    file = analyze_dataset(calibration_data_path=str(path), **opts)
    memory = analyze_dataset(calibration_data=rows, **opts)
    np.testing.assert_array_equal(memory.estimates, file.estimates)
    np.testing.assert_array_equal(memory.standard_errors, file.standard_errors)
    np.testing.assert_array_equal(
        memory.confidence_interval(), file.confidence_interval()
    )
    assert memory.compare_policies("B", "A") == file.compare_policies("B", "A")
    for key in (
        "calibration",
        "oracle_sources",
        "data_provenance",
        "transport_audits",
        "transport_unverified",
        "calibration_label_sources",
        "normalization",
    ):
        assert memory.metadata[key] == file.metadata[key]
    assert memory.metadata["data_provenance"]["calibration_fit_roles"] == {
        "external": 80,
        "evaluation": 30 if combine else 0,
    }
    assert memory.units is not None and memory.units.output_scale["min"] == 1
    assert file.metadata["calibration_data_input"] == "jsonl"
    assert memory.metadata["calibration_data_input"] == "records"
    assert "calibration_data_path" not in memory.metadata
    assert "the calibration fit on calibration_data" in memory.summary()
    assert "calibration_data_path" not in memory.summary()
    assert rows == original


def test_memory_table_validates_and_keeps_row_identity() -> None:
    draws = {"A": [{"prompt_id": f"p{i}", "judge_score": i / 9} for i in range(10)]}
    good = [
        {"prompt_id": f"h{i}", "judge_score": i / 9, "oracle_label": i / 9}
        for i in range(10)
    ]
    with pytest.raises(
        ValueError, match="only one of calibration_data_path or calibration_data"
    ):
        analyze_dataset(
            fresh_draws_data=draws, calibration_data_path="missing", calibration_data=[]
        )
    with pytest.raises(ValueError, match="No valid samples"):
        analyze_dataset(fresh_draws_data=draws, calibration_data=[])
    with pytest.raises(TypeError, match="sequence of records or a DataFrame"):
        analyze_dataset(fresh_draws_data=draws, calibration_data=good[0])  # type: ignore[arg-type]
    invalid: list[Any] = [good[0], None, {"judge_score": "bad"}, *good[1:]]
    with pytest.raises(ValueError, match=r"calibration_data\[1\]"):
        analyze_dataset(fresh_draws_data=draws, calibration_data=invalid)
    result = analyze_dataset(
        fresh_draws_data=draws, calibration_data=invalid, on_invalid="drop"
    )
    assert result.metadata["n_invalid_calibration_dropped"] == 2
    assert result.metadata["n_invalid_dropped"] == 0
    assert (
        result.metadata["data_provenance"]["calibration_fit_rows"][1]
        == "external_calibration:records:row:3"
    )
    with pytest.raises(
        ValueError, match="Conflicting calibration records share row_id"
    ):
        DatasetLoader().load_from_records(
            [dict(good[0], row_id="id"), dict(good[1], row_id="id")]
        )


def test_external_records_preserve_probe_overlap_checks() -> None:
    rows = [
        {
            "prompt_id": f"c{i}",
            "judge_score": i / 39,
            "oracle_label": i / 39,
            "observation_id": f"stable-{i}",
        }
        for i in range(40)
    ]
    draws = {"A": [{"prompt_id": f"p{i}", "judge_score": i / 19} for i in range(20)]}
    with pytest.raises(ValueError, match="probes must be held out"):
        analyze_dataset(
            fresh_draws_data=draws,
            calibration_data=rows,
            transport=TransportAuditConfig(probes_by_policy={"A": [dict(rows[0])]}),
        )


def test_external_and_same_named_policy_provenance() -> None:
    draws = {
        "calibration_data": [
            {
                "prompt_id": f"p{i}",
                "judge_score": i / 29,
                "oracle_label": i / 29 if i % 2 == 0 else None,
            }
            for i in range(30)
        ]
    }
    external = [
        {"prompt_id": f"e{i}", "judge_score": i / 29, "oracle_label": 1 - i / 29}
        for i in range(30)
    ]
    options: dict[str, Any] = {
        "fresh_draws_data": draws,
        "estimator_config": {"use_augmented_estimator": False},
    }
    borrowed = analyze_dataset(
        calibration_data=external, combine_oracle_sources=False, **options
    )
    assert borrowed.metadata["transport_unverified"] == ["calibration_data"]
    assert (
        borrowed._calibration_source_phrase()
        == "the calibration fit on calibration_data"
    )
    pooled = analyze_dataset(calibration_data=external, **options)
    assert (
        pooled._calibration_source_phrase()
        == "the calibration fit on calibration_data and calibration_data's labels"
    )
    unused = analyze_dataset(
        calibration_data=[{"prompt_id": "e", "judge_score": 0.0}], **options
    )
    assert unused._calibration_source_phrase() == "calibration_data's calibration"
    assert unused.metadata["transport_unverified"] == []


def test_pandas_table_matches_records_and_rejects_duplicate_columns() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "prompt_id": ["c0", "c1"],
            "judge_score": [0, 1],
            "oracle_label": [np.nan, 1.0],
        },
        index=["unused-0", "unused-1"],
    )
    loader = DatasetLoader()
    table = loader.load_from_records(frame)
    assert table == loader.load_from_records(frame.to_dict(orient="records"))
    assert [s.prompt_id for s in table.samples] == ["c0", "c1"]
    assert table.samples[0].oracle_label is None
    with pytest.raises(ValueError, match="columns must be unique"):
        loader.load_from_records(frame.rename(columns={"judge_score": "oracle_label"}))


def _weighted_rows(multiplier: float = 1.0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(20):
        for label, weight in [(0, 1), (1, 3)]:
            row: dict[str, Any] = {
                "row_id": f"hist-{i}-{label}",
                "prompt_id": f"hist-{i}",
                "judge_score": (i % 5) / 4,
                "oracle_label": label,
            }
            if label:
                row["audit_weight"] = weight * multiplier
            else:
                row["metadata"] = {"audit_weight": weight * multiplier}
            rows.append(row)
    return rows


def test_weighted_external_fit_changes_only_calibrated_estimates(
    tmp_path: Path,
) -> None:
    rows = _weighted_rows()
    draws = {
        "plugin": [{"prompt_id": f"p{i}", "judge_score": i / 29} for i in range(30)],
        "labeled": [
            {"prompt_id": f"p{i}", "judge_score": i / 29, "oracle_label": 0.25}
            for i in range(30)
        ],
    }
    opts: dict[str, Any] = {"combine_oracle_sources": False, "fresh_draws_data": draws}
    default = analyze_dataset(calibration_data=rows, **opts)
    ones = analyze_dataset(
        calibration_data=[dict(r, unit_weight=1) for r in rows],
        calibration_weight_field="unit_weight",
        **opts,
    )
    np.testing.assert_array_equal(default.estimates, ones.estimates)
    np.testing.assert_array_equal(default.standard_errors, ones.standard_errors)
    assert "calibration_weighting" not in default.metadata
    weighted = analyze_dataset(
        calibration_data=rows, calibration_weight_field="audit_weight", **opts
    )
    assert weighted.metadata["target_policies"] == ["labeled", "plugin"]
    np.testing.assert_allclose(default.estimates, [0.25, 0.5])
    np.testing.assert_allclose(weighted.estimates, [0.25, 0.75])
    assert weighted.metadata["calibration_weighting"] == {
        "field": "audit_weight",
        "normalization": "mean_one_over_labeled_rows",
        "n_labeled_rows": 40,
        "n_labeled_prompts": 20,
        "kish_effective_rows": pytest.approx(32),
        "kish_effective_prompts": pytest.approx(20),
        "fit_performed": True,
    }
    # The fitted mapping is identical when weights have a different common scale.
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in _weighted_rows(1e300)))
    file = analyze_dataset(
        calibration_data_path=str(path), calibration_weight_field="audit_weight", **opts
    )
    np.testing.assert_allclose(file.estimates, weighted.estimates)
    np.testing.assert_allclose(file.standard_errors, weighted.standard_errors)


def test_external_weights_keep_labels_aligned_through_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cje.calibration.judge import JudgeCalibrator

    rows = _weighted_rows()
    rows.insert(0, {"prompt_id": "missing-label", "judge_score": 0.5})
    rows.insert(3, dict(rows[1]))  # Canonical duplicate is counted once.
    rows.insert(11, {"prompt_id": "invalid", "judge_score": "x", "audit_weight": -1})
    weights_seen = []
    original = JudgeCalibrator.fit_cv

    def record_fit(self: JudgeCalibrator, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        weights_seen.append(self._fit_sample_weight.copy())  # type: ignore[union-attr]
        return result

    monkeypatch.setattr(JudgeCalibrator, "fit_cv", record_fit)
    result = analyze_dataset(
        fresh_draws_data={
            "A": [{"prompt_id": f"p{i}", "judge_score": i / 29} for i in range(30)]
        },
        calibration_data=rows,
        calibration_weight_field="audit_weight",
        combine_oracle_sources=False,
        on_invalid="drop",
        estimator_config={"inference_method": "bootstrap", "n_bootstrap": 5},
    )
    assert result.metadata["n_invalid_calibration_dropped"] == 1
    np.testing.assert_allclose(result.estimates, [0.75])
    assert result.metadata["data_provenance"]["calibration_fit_rows"] == [
        f"hist-{i}-{j}" for i in range(20) for j in (0, 1)
    ]
    np.testing.assert_allclose(weights_seen[0], np.tile([0.5, 1.5], 20))
    assert len(weights_seen) == 6
    # Each prompt's multiplier is shared across its two rows in every refit.
    for weights in weights_seen[1:]:
        np.testing.assert_allclose(weights[1::2] / weights[::2], 3)
        assert np.ptp(weights) > 0 and not np.allclose(weights, weights_seen[0])


@pytest.mark.parametrize(
    "bad", [None, False, True, 0, -1, np.nan, np.inf, -np.inf, 10**500, "2"]
)
def test_weighted_fit_rejects_invalid_weight_even_in_drop_mode(bad: Any) -> None:
    rows = _weighted_rows()
    rows[1]["audit_weight"] = bad
    with pytest.raises(ValueError, match="finite positive number.*hist-0-1"):
        analyze_dataset(
            fresh_draws_data={"A": [{"prompt_id": "p", "judge_score": 0.5}]},
            calibration_data=rows,
            combine_oracle_sources=False,
            calibration_weight_field="audit_weight",
            on_invalid="drop",
        )


def test_weighted_fit_requires_external_only_and_every_label_weight() -> None:
    rows = _weighted_rows()
    draws = {"A": [{"prompt_id": "p", "judge_score": 0.5}]}
    with pytest.raises(ValueError, match="requires external calibration data"):
        analyze_dataset(fresh_draws_data=draws, calibration_weight_field="w")
    with pytest.raises(ValueError, match="combine_oracle_sources=False"):
        analyze_dataset(
            fresh_draws_data=draws,
            calibration_data=rows,
            calibration_weight_field="audit_weight",
        )
    del rows[1]["audit_weight"]
    with pytest.raises(ValueError, match="finite positive number.*hist-0-1"):
        analyze_dataset(
            fresh_draws_data=draws,
            calibration_data=rows,
            calibration_weight_field="audit_weight",
            combine_oracle_sources=False,
        )


def test_uniform_weights_reproduce_twostage_fit_and_every_bootstrap_interval() -> None:
    rng = np.random.default_rng(563)
    s, x = rng.uniform(size=(2, 100))
    y = rng.binomial(1, 0.15 + 0.3 * s + 0.45 * x)
    rows = [
        {
            "prompt_id": f"h{i//2}",
            "judge_score": s[i],
            "x": x[i],
            "oracle_label": y[i],
            "audit_weight": 7.0,
        }
        for i in range(len(s))
    ]
    draws = {
        "A": [
            {"prompt_id": f"p{i}", "judge_score": s[i], "x": x[99 - i]}
            for i in range(36)
        ]
    }
    opts: dict[str, Any] = {
        "fresh_draws_data": draws,
        "calibration_data": rows,
        "combine_oracle_sources": False,
        "calibration_covariates": ["x"],
        "estimator_config": {"inference_method": "bootstrap", "n_bootstrap": 8},
    }
    default = analyze_dataset(**opts)
    uniform = analyze_dataset(calibration_weight_field="audit_weight", **opts)
    assert uniform.metadata["calibration_info"]["selected_mode"] == "two_stage"
    np.testing.assert_array_equal(default.estimates, uniform.estimates)
    np.testing.assert_array_equal(default.standard_errors, uniform.standard_errors)
    assert default.metadata["bootstrap_ci"] == uniform.metadata["bootstrap_ci"]
    assert uniform.metadata["calibration_weighting"][
        "kish_effective_prompts"
    ] == pytest.approx(50)
