"""Focused regressions for the statistical-core remediation."""

import numpy as np
import pytest
from types import SimpleNamespace
from typing import Optional

from cje.array_api import calibrated_mean_ci
from cje.calibration.flexible_calibrator import FlexibleCalibrator, _fit_ecdf
from cje.calibration.judge import JudgeCalibrator
from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample
from cje.diagnostics.robust_inference import (
    CalibrationProvenance,
    DirectEvalTable,
    LabelDesign,
    build_direct_eval_table,
    cluster_bootstrap_direct_with_refit,
    compute_direct_point_estimate,
    make_calibrator_factory,
    oracle_jackknife_estimates,
    residual_predictions_for_evaluation,
    validate_calibration_provenance,
)
from cje.estimators.direct_method import CalibratedDirectEstimator


def _draws(
    policy: str,
    scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    repeats: int = 1,
) -> FreshDrawDataset:
    samples = []
    for i, score in enumerate(scores):
        for draw in range(repeats):
            samples.append(
                FreshDrawSample(
                    prompt_id=f"p{i}",
                    target_policy=policy,
                    judge_score=float(score),
                    oracle_label=(None if labels is None else float(labels[i])),
                    response=None,
                    draw_idx=draw,
                )
            )
    return FreshDrawDataset(target_policy=policy, samples=samples)


def _table(
    scores: np.ndarray,
    labels: np.ndarray,
    prompts: list[str],
    policy_indices: Optional[np.ndarray] = None,
    policy_names: Optional[list[str]] = None,
    covariates: Optional[np.ndarray] = None,
) -> DirectEvalTable:
    policy_indices = (
        np.zeros(len(scores), dtype=np.int32)
        if policy_indices is None
        else np.asarray(policy_indices, dtype=np.int32)
    )
    policy_names = policy_names or ["policy"]
    unique_prompts = list(dict.fromkeys(prompts))
    prompt_codes = {prompt: index for index, prompt in enumerate(unique_prompts)}
    return DirectEvalTable(
        prompt_ids=np.asarray([prompt_codes[prompt] for prompt in prompts]),
        prompt_id_strings=list(prompts),
        policy_indices=policy_indices,
        judge_scores=np.asarray(scores),
        oracle_labels=np.asarray(labels, dtype=float),
        oracle_mask=np.isfinite(labels),
        covariates=covariates,
        covariate_names=(
            None
            if covariates is None
            else [f"cov_{index}" for index in range(covariates.shape[1])]
        ),
        policy_names=policy_names,
    )


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        (np.asarray([-1, 1], dtype=np.int16), "negative"),
        (np.asarray([0, 0], dtype=np.uint16), "duplicate"),
        (np.asarray([0, 4], dtype=np.uint16), "out-of-range"),
    ],
)
def test_integer_oracle_indices_fail_before_numpy_wraparound(
    indices: np.ndarray, message: str
) -> None:
    calibrator = JudgeCalibrator(calibration_mode="monotone")
    with pytest.raises(ValueError, match=message):
        calibrator.fit_cv(
            np.linspace(0.1, 0.9, 4),
            np.asarray([0.0, 1.0]),
            oracle_mask=indices,
            n_folds=2,
        )


def test_cross_fit_counts_unique_prompt_clusters() -> None:
    calibrator = JudgeCalibrator(calibration_mode="monotone")
    with pytest.raises(ValueError, match="unique oracle prompt clusters"):
        calibrator.fit_cv(
            np.linspace(0.1, 0.9, 12),
            np.linspace(0.1, 0.9, 12),
            prompt_ids=["same-prompt"] * 12,
        )


def test_sparse_fold_model_keys_are_used_for_jackknife() -> None:
    class SparseCalibrator:
        def get_fold_models_for_oua(self):  # type: ignore[no-untyped-def]
            return {2: object(), 4: object()}

        def predict_oof(self, scores, folds, covariates=None):  # type: ignore[no-untyped-def]
            return np.full(len(scores), float(folds[0]) / 10.0)

    jack = oracle_jackknife_estimates(SparseCalibrator(), np.asarray([0.2, 0.8]))
    assert jack is not None
    np.testing.assert_allclose(jack, [0.2, 0.4])


def test_external_provenance_bootstrap_preserves_point_estimator() -> None:
    scores = np.linspace(0.05, 0.95, 20)
    labels = np.where(scores > 0.5, 0.8, 0.2)
    calibrator = JudgeCalibrator(calibration_mode="monotone")
    calibrator.fit_cv(
        scores,
        labels,
        prompt_ids=[f"cal-{i}" for i in range(len(scores))],
    )
    provenance = CalibrationProvenance(
        scores,
        labels,
        [f"cal-{i}" for i in range(len(scores))],
    )
    draws = _draws("policy", np.linspace(0.2, 0.8, 12))

    analytic = CalibratedDirectEstimator(
        ["policy"],
        calibrator,
        inference_method="cluster_robust",
        calibration_provenance=provenance,
    )
    bootstrap = CalibratedDirectEstimator(
        ["policy"],
        calibrator,
        inference_method="bootstrap",
        n_bootstrap=20,
        calibration_provenance=provenance,
    )
    for estimator in (analytic, bootstrap):
        estimator.add_fresh_draws("policy", draws)
    result_analytic = analytic.fit_and_estimate()
    result_bootstrap = bootstrap.fit_and_estimate()

    np.testing.assert_allclose(
        result_analytic.estimates, result_bootstrap.estimates, atol=1e-12
    )
    assert result_bootstrap.metadata["inference"]["n_attempts"] == 20
    assert result_bootstrap.metadata["inference"]["skip_rate"] == 0.0
    assert (
        result_bootstrap.metadata["inference"]["bootstrap_scheme"]
        == "positive_exponential_cluster_weights"
    )


def test_full_evaluation_coverage_routes_to_raw_oracle_mean() -> None:
    scores = np.linspace(0.1, 0.9, 10)
    labels = np.asarray([0.0, 1.0] * 5)
    calibrator = JudgeCalibrator(calibration_mode="monotone")
    calibrator.fit_cv(
        scores,
        np.full(10, 0.25),
        prompt_ids=[f"cal-{i}" for i in range(10)],
    )
    estimator = CalibratedDirectEstimator(
        ["policy"], calibrator, inference_method="cluster_robust"
    )
    estimator.add_fresh_draws("policy", _draws("policy", scores, labels))
    result = estimator.fit_and_estimate()

    assert result.estimates[0] == pytest.approx(float(np.mean(labels)))
    assert result.metadata["point_estimator"]["routes"] == ["direct_oracle"]


def test_known_propensity_and_targeted_unknown_routes() -> None:
    labels = np.asarray([1.0, np.nan, 0.0, np.nan])
    table = build_direct_eval_table(
        {
            "policy": FreshDrawDataset(
                target_policy="policy",
                samples=[
                    FreshDrawSample(
                        prompt_id=f"p{i}",
                        target_policy="policy",
                        judge_score=0.5,
                        oracle_label=None if np.isnan(label) else float(label),
                        response=None,
                        draw_idx=0,
                    )
                    for i, label in enumerate(labels)
                ],
            )
        }
    )
    calibrated = np.full(4, 0.5)
    propensity = LabelDesign("known_propensity", {"policy": np.full(4, 0.5)})
    augmented = compute_direct_point_estimate(calibrated, table, calibrated, propensity)
    targeted = compute_direct_point_estimate(
        calibrated, table, calibrated, LabelDesign("targeted_unknown")
    )

    assert augmented.estimates[0] == pytest.approx(0.5)
    assert augmented.diagnostics["routes"] == ["augmented"]
    assert targeted.estimates[0] == pytest.approx(0.5)
    assert targeted.diagnostics["routes"] == ["plug_in_targeted_unknown"]


def test_positive_weight_bootstrap_supports_two_stage() -> None:
    rng = np.random.default_rng(4)
    scores = np.linspace(0.05, 0.95, 24)
    labels = np.clip(scores + rng.normal(0, 0.04, len(scores)), 0, 1)
    draws = _draws("policy", scores, labels)
    table = build_direct_eval_table({"policy": draws})
    result = cluster_bootstrap_direct_with_refit(
        table,
        make_calibrator_factory("two_stage"),
        n_bootstrap=5,
        label_design=LabelDesign("representative"),
    )
    assert result["n_valid_replicates"] == 5
    assert result["n_attempts"] == 5


def test_weighted_ecdf_aggregates_tied_probability_mass() -> None:
    first = _fit_ecdf(np.asarray([0.0, 0.0, 1.0]), np.asarray([1.0, 9.0, 1.0]))
    reordered = _fit_ecdf(np.asarray([0.0, 0.0, 1.0]), np.asarray([9.0, 1.0, 1.0]))

    expected = 5.0 / 11.0
    assert first(np.asarray([0.0]))[0] == pytest.approx(expected)
    assert reordered(np.asarray([0.0]))[0] == pytest.approx(expected)


def test_integer_judge_scores_keep_fractional_oof_predictions() -> None:
    scores = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    labels = np.asarray([0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8])
    folds = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    calibrator = FlexibleCalibrator(mode="monotone").fit(scores, labels, folds)

    predictions = calibrator.predict(scores, folds=folds)

    assert predictions.dtype.kind == "f"
    np.testing.assert_allclose(predictions, labels)


def test_representative_augmentation_is_a_weighted_ratio_estimator() -> None:
    # Heterogeneous residuals (1.0 and 0.5 on the observed rows) make the
    # estimate itself discriminate: the weighted ratio functional gives
    # sum(w_o * r_o) / sum(w_o) = (10*1.0 + 1*0.5) / 11, while an unweighted
    # residual correction would give mean([1.0, 0.5]) = 0.75.
    table = _table(
        scores=np.zeros(4),
        labels=np.asarray([1.0, np.nan, 0.5, np.nan]),
        prompts=["p0", "p1", "p2", "p3"],
    )
    weights = np.asarray([10.0, 1.0, 1.0, 1.0])
    point = compute_direct_point_estimate(
        calibrated_full=np.zeros(4),
        eval_table=table,
        residual_predictions=np.zeros(4),
        label_design=LabelDesign("representative"),
        observation_weights=weights,
    )

    correction = 10.5 / 11.0  # weighted residual mean, not the unweighted 0.75
    propensity = 11.0 / 13.0  # weighted oracle mass fraction, not the count 2/4
    assert point.estimates[0] == pytest.approx(correction)
    assert point.diagnostics["oracle_fractions"][0] == pytest.approx(propensity)
    # Pseudo-outcomes pin the Horvitz-Thompson construction row by row: every
    # row carries the correction, and observed rows add
    # (residual - correction) / propensity with the WEIGHTED propensity — a
    # count-based propensity would shift the observed rows.
    np.testing.assert_allclose(
        point.pseudo_outcomes[0],
        [
            correction + (1.0 - correction) / propensity,
            correction,
            correction + (0.5 - correction) / propensity,
            correction,
        ],
    )
    # The weighted mean of the pseudo-outcomes reproduces the estimate.
    assert float(
        np.average(point.pseudo_outcomes[0], weights=weights)
    ) == pytest.approx(point.estimates[0])


def test_calibration_base_weights_are_copied_validated_and_refit() -> None:
    scores = np.asarray([0.0, 0.0, 0.3, 0.6, 0.9])
    labels = np.asarray([1.0, 0.0, 0.2, 0.6, 0.8])
    prompts = ["c0", "c0", "c1", "c2", "c3"]
    base_weights = np.asarray([100.0, 1.0, 1.0, 1.0, 1.0])
    provenance = CalibrationProvenance(
        scores,
        labels,
        prompts,
        sample_weights=base_weights,
    )

    scores[0] = 999.0
    labels[0] = 0.0
    base_weights[0] = 1.0
    assert provenance.judge_scores[0] == 0.0
    assert provenance.oracle_labels[0] == 1.0
    assert provenance.sample_weights is not None
    assert provenance.sample_weights[0] == 100.0

    recorded_weights: list[np.ndarray] = []

    class RecordingCalibrator:
        def fit_cv(self, fit_scores, fit_labels, **kwargs):  # type: ignore[no-untyped-def]
            weights = np.asarray(kwargs["sample_weight"], dtype=float).copy()
            recorded_weights.append(weights)
            self.mean = float(np.average(fit_labels, weights=weights))
            return SimpleNamespace(fold_ids=np.zeros(len(fit_scores), dtype=int))

        def predict(self, predict_scores, covariates=None):  # type: ignore[no-untyped-def]
            return np.full(len(predict_scores), self.mean)

    eval_table = _table(
        scores=np.linspace(0.1, 0.9, 4),
        labels=np.full(4, np.nan),
        prompts=["e0", "e1", "e2", "e3"],
    )
    result = cluster_bootstrap_direct_with_refit(
        eval_table,
        RecordingCalibrator,
        n_bootstrap=3,
        calibration_provenance=provenance,
        use_augmented_estimator=False,
        n_folds=2,
    )

    assert result["n_valid_replicates"] == 3
    assert len(recorded_weights) == 4
    for weights in recorded_weights:
        assert weights[0] / weights[1] == pytest.approx(100.0)

    fitted = JudgeCalibrator(calibration_mode="monotone")
    fitted.fit_cv(
        provenance.judge_scores,
        provenance.oracle_labels,
        prompt_ids=provenance.prompt_ids,
        n_folds=2,
        sample_weight=provenance.sample_weights,
    )
    retained = CalibrationProvenance.from_fitted_calibrator(fitted)
    validate_calibration_provenance(retained, fitted)
    retained.sample_weights = np.ones(5)
    with pytest.raises(ValueError, match="sample_weight"):
        validate_calibration_provenance(retained, fitted)


def test_provenance_rejects_prompt_and_linked_covariate_mismatches() -> None:
    with pytest.raises(ValueError, match="does not match calibration prompt_id"):
        CalibrationProvenance(
            np.asarray([0.2]),
            np.asarray([0.4]),
            ["cal-prompt"],
            row_roles=["evaluation"],
            evaluation_keys=[("policy", "other-prompt", 0)],
        )

    table = _table(
        scores=np.asarray([0.2]),
        labels=np.asarray([0.4]),
        prompts=["p0"],
        covariates=np.asarray([[1.0]]),
    )
    assert table.row_keys is not None
    provenance = CalibrationProvenance(
        np.asarray([0.2]),
        np.asarray([0.4]),
        ["p0"],
        covariates=np.asarray([[2.0]]),
        row_roles=["evaluation"],
        evaluation_keys=[table.row_keys[0]],
    )

    with pytest.raises(ValueError, match="covariates"):
        residual_predictions_for_evaluation(
            object(),
            np.asarray([0.4]),
            table,
            provenance,
            calibration_fold_ids=np.asarray([0]),
        )

    mismatched_values = CalibrationProvenance(
        np.asarray([0.3]),
        np.asarray([0.4]),
        ["p0"],
        covariates=np.asarray([[1.0]]),
        row_roles=["evaluation"],
        evaluation_keys=[table.row_keys[0]],
    )
    with pytest.raises(ValueError, match=r"values for \('policy', 'p0', 0\)"):
        residual_predictions_for_evaluation(
            object(),
            np.asarray([0.4]),
            table,
            mismatched_values,
            calibration_fold_ids=np.asarray([0]),
        )


def test_full_oracle_bootstrap_skips_provenance_and_calibration() -> None:
    table = _table(
        scores=np.asarray([1.0, 2.0, 4.0, 5.0]),
        labels=np.asarray([0.0, 1.0, 0.0, 1.0]),
        prompts=["p0", "p1", "p2", "p3"],
    )

    def fail_factory():  # type: ignore[no-untyped-def]
        raise AssertionError("full oracle coverage must not fit a calibrator")

    result = cluster_bootstrap_direct_with_refit(table, fail_factory, n_bootstrap=5)

    assert result["estimates"][0] == pytest.approx(0.5)
    assert np.isfinite(result["standard_errors"][0])
    assert result["provenance_summary"]["n_rows"] == 0


def test_bootstrap_marks_only_underclustered_policies_unavailable() -> None:
    table = _table(
        scores=np.asarray([1.0, 2.0, 3.0, 4.0]),
        labels=np.asarray([0.0, 1.0, 0.0, 1.0]),
        prompts=["shared", "shared", "p1", "p2"],
        policy_indices=np.asarray([0, 0, 1, 1]),
        policy_names=["one-cluster", "two-cluster"],
    )
    result = cluster_bootstrap_direct_with_refit(
        table, lambda: object(), n_bootstrap=10
    )

    np.testing.assert_allclose(result["estimates"], [0.5, 0.5])
    assert np.isnan(result["standard_errors"][0])
    assert np.isnan(result["ci_lower"][0])
    assert np.all(np.isnan(result["bootstrap_matrix"][:, 0]))
    assert np.isfinite(result["standard_errors"][1])
    assert result["policy_cluster_counts"] == [1, 2]
    assert result["inference_unavailable_policies"] == ["one-cluster"]


def test_implicit_provenance_keeps_augmentation_on_raw_judge_scale() -> None:
    labels = np.asarray([0.0, 0.0, 1.0, 1.0, np.nan, np.nan, np.nan, np.nan])
    table = _table(
        scores=np.arange(1, 9),
        labels=labels,
        prompts=[f"p{index}" for index in range(8)],
    )
    result = cluster_bootstrap_direct_with_refit(
        table,
        make_calibrator_factory("monotone"),
        n_bootstrap=3,
        n_folds=2,
    )

    assert result["use_augmented_estimator"] is True
    assert result["augmentation_diagnostics"]["augmentation_effective"] is True
    assert result["augmentation_diagnostics"]["routes"] == ["augmented"]


def test_array_api_raw_scale_and_full_coverage_capability_contract() -> None:
    scores = np.arange(1.0, 9.0)
    labels = np.asarray([0.0, 0.0, 1.0, 1.0, np.nan, np.nan, np.nan, np.nan])
    partial = calibrated_mean_ci(
        scores,
        labels,
        cluster_ids=[f"p{index}" for index in range(8)],
        inference="bootstrap",
        n_bootstrap=3,
        n_folds=2,
    )
    assert np.isfinite(partial.estimate)
    assert partial.calibrator is not None
    assert partial.diagnostics["calibration"]["calibrator_available"] is True

    full_scores = np.linspace(1.0, 5.0, 20)
    full_labels = np.asarray([0.0, 1.0] * 10)
    full = calibrated_mean_ci(
        full_scores,
        full_labels,
        cluster_ids=[f"p{index}" for index in range(20)],
    )
    assert full.method == "cluster_robust"
    assert full.calibrator is None
    assert full.diagnostics["estimator_route"] == "direct_oracle"
    assert full.diagnostics["calibration"]["calibrator_available"] is False
    assert "sufficient clusters" in full.diagnostics["inference_reason"]
