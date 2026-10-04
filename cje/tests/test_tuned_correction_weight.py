"""The tuned correction weight: PPI++ power tuning inside the augmented estimator.

Weight one (the 0.8.x estimator) is the default.  The opt-in tuned weight must
equal the least-squares slope of labelled outcomes on predictions, clipped to
[0, 1]; fall back to one below the minimum count of labelled prompts, when too
few labelled prompts differ from the most common outcome (identical labels
would otherwise give a weight of zero and a standard error of exactly zero),
and in the known-propensity design; reduce the estimator to the labelled mean
when the predictions are uninformative; and not raise the pseudo-outcome
variance above weight one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from cje import analyze_dataset
from cje.diagnostics.robust_inference import (
    TUNED_WEIGHT_MIN_LABELS,
    TUNED_WEIGHT_MIN_MINORITY,
    DirectEvalTable,
    LabelDesign,
    compute_direct_point_estimate,
    correction_weight_decision,
    resolve_correction_weight,
)


def _table(n: int, seed: int, noise: float, labelled: int) -> tuple:
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.2, 0.8, size=n)
    outcomes = np.clip(truth + rng.normal(0, 0.15, size=n), 0, 1)
    predictions = np.clip(truth + rng.normal(0, noise, size=n), 0, 1)
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=labelled, replace=False)] = True
    labels = np.where(mask, outcomes, np.nan)
    codes = np.arange(n)
    table = DirectEvalTable(
        prompt_ids=codes,
        prompt_id_strings=[f"p{i}" for i in range(n)],
        policy_indices=np.zeros(n, dtype=np.int32),
        judge_scores=predictions,
        oracle_labels=labels,
        oracle_mask=mask,
        covariates=None,
        covariate_names=None,
        policy_names=["policy"],
    )
    return table, predictions, outcomes, mask


def test_weight_rule_matches_least_squares_slope_and_clips() -> None:
    rng = np.random.default_rng(0)
    predictions = rng.uniform(size=200)
    outcomes = 0.4 * predictions + rng.normal(0, 0.05, size=200)
    weights = np.ones(200)
    slope = np.polyfit(predictions, outcomes, 1)[0]
    assert resolve_correction_weight(
        "tuned", outcomes, predictions, weights
    ) == pytest.approx(slope, abs=1e-9)
    assert resolve_correction_weight("one", outcomes, predictions, weights) == 1.0
    # Anti-correlated predictions clip to zero; a steep slope clips to one.
    assert resolve_correction_weight("tuned", -outcomes, predictions, weights) == 0.0
    assert (
        resolve_correction_weight("tuned", 3 * predictions, predictions, weights) == 1.0
    )
    # Degenerate inputs fall back to one.
    assert (
        resolve_correction_weight("tuned", outcomes[:1], predictions[:1], weights[:1])
        == 1.0
    )
    assert (
        resolve_correction_weight("tuned", outcomes, np.full(200, 0.3), weights) == 1.0
    )
    with pytest.raises(ValueError, match="correction_weight"):
        resolve_correction_weight("half", outcomes, predictions, weights)


def test_weight_one_is_the_plain_augmented_estimator() -> None:
    table, predictions, outcomes, mask = _table(400, seed=1, noise=0.1, labelled=80)
    point = compute_direct_point_estimate(
        predictions, table, predictions, LabelDesign(), correction_weight="one"
    )
    plug_in = predictions.mean()
    correction = (outcomes[mask] - predictions[mask]).mean()
    assert point.estimates[0] == pytest.approx(plug_in + correction)
    assert point.diagnostics["correction_weights"] == [1.0]
    assert point.diagnostics["routes"] == ["augmented"]
    assert point.diagnostics["correction_weight_reasons"] == ["rule_one"]
    # The default is weight one; the tuned weight is opt-in.
    default = compute_direct_point_estimate(
        predictions, table, predictions, LabelDesign()
    )
    assert default.diagnostics["correction_weight_rule"] == "one"
    assert default.estimates[0] == point.estimates[0]
    assert np.array_equal(default.pseudo_outcomes[0], point.pseudo_outcomes[0])


def test_tuned_weight_falls_back_to_one_below_the_minimum_labels() -> None:
    assert TUNED_WEIGHT_MIN_LABELS == 20
    rng = np.random.default_rng(21)
    predictions = rng.uniform(size=19)
    outcomes = 0.5 * predictions + rng.normal(0, 0.05, size=19)
    assert resolve_correction_weight("tuned", outcomes, predictions, np.ones(19)) == 1.0
    predictions = rng.uniform(size=20)
    outcomes = 0.5 * predictions + rng.normal(0, 0.05, size=20)
    assert resolve_correction_weight("tuned", outcomes, predictions, np.ones(20)) < 1.0
    table, predictions, outcomes, mask = _table(300, seed=22, noise=0.6, labelled=15)
    point = compute_direct_point_estimate(
        predictions, table, predictions, LabelDesign(), correction_weight="tuned"
    )
    assert point.diagnostics["correction_weights"] == [1.0]
    assert point.diagnostics["correction_weight_reasons"] == [
        "too_few_labelled_clusters"
    ]
    assert point.diagnostics["correction_weight_rule"] == "tuned"
    assert point.diagnostics["correction_weight_min_labels"] == 20


def test_array_api_exposes_the_weight(tmp_path: Path) -> None:
    from cje import calibrated_mean_ci

    rng = np.random.default_rng(31)
    scores = rng.uniform(size=600)
    truth = 0.2 + 0.6 * scores
    labels = np.full(600, np.nan)
    labelled = rng.choice(600, size=120, replace=False)
    labels[labelled] = (rng.uniform(size=120) < truth[labelled]).astype(
        float
    )  # binary labels
    tuned = calibrated_mean_ci(scores, labels, correction_weight="tuned")
    one = calibrated_mean_ci(scores, labels)
    assert tuned.diagnostics["correction_weight"]["rule"] == "tuned"
    assert tuned.diagnostics["correction_weight"]["reason"] == "tuned"
    assert 0.0 <= tuned.diagnostics["correction_weight"]["weight"] <= 1.0
    assert one.diagnostics["correction_weight"] == {
        "rule": "one",
        "weight": 1.0,
        "reason": "rule_one",
        "route": "augmented",
    }
    assert tuned.se <= one.se * (1 + 1e-6)
    with pytest.raises(ValueError, match="correction_weight"):
        calibrated_mean_ci(scores, labels, correction_weight="half")


def test_tuned_weight_reduces_to_labelled_mean_on_noise() -> None:
    rng = np.random.default_rng(1002)  # a different stream from the fixture's
    table, predictions, outcomes, mask = _table(400, seed=2, noise=0.1, labelled=200)
    noise_predictions = rng.uniform(size=400)  # unrelated to the outcome
    tuned = compute_direct_point_estimate(
        noise_predictions,
        table,
        noise_predictions,
        LabelDesign(),
        correction_weight="tuned",
    )
    weight = tuned.diagnostics["correction_weights"][0]
    assert 0.0 <= weight < 0.25  # the slope's sampling SD is about 0.05 at 200 labels
    labelled_mean = outcomes[mask].mean()
    # Estimate = w*plug_in + mean(Y - w*f) on labelled rows; at w -> 0 it is the labelled mean.
    expected = (
        weight * noise_predictions.mean()
        + (outcomes[mask] - weight * noise_predictions[mask]).mean()
    )
    assert tuned.estimates[0] == pytest.approx(expected)
    assert abs(tuned.estimates[0] - labelled_mean) < 0.05
    one = compute_direct_point_estimate(
        noise_predictions,
        table,
        noise_predictions,
        LabelDesign(),
        correction_weight="one",
    )
    assert np.var(tuned.pseudo_outcomes[0]) < np.var(one.pseudo_outcomes[0])


def test_tuned_weight_does_not_raise_pseudo_outcome_variance() -> None:
    """The tuned weight minimises the corrected estimator's variance in
    expectation; in one finite sample it may exceed weight one by sampling
    noise only when the predictions are nearly perfect (weight close to one),
    and it must be strictly lower when the predictions are noisy."""
    for seed, noise, strict in [
        (3, 0.02, False),
        (4, 0.1, False),
        (5, 0.3, True),
        (6, 0.6, True),
    ]:
        table, predictions, outcomes, mask = _table(
            500, seed=seed, noise=noise, labelled=100
        )
        one = compute_direct_point_estimate(
            predictions, table, predictions, LabelDesign(), correction_weight="one"
        )
        tuned = compute_direct_point_estimate(
            predictions, table, predictions, LabelDesign(), correction_weight="tuned"
        )
        var_one = np.var(one.pseudo_outcomes[0])
        var_tuned = np.var(tuned.pseudo_outcomes[0])
        assert var_tuned <= var_one * 1.01
        if strict:
            assert var_tuned < var_one
            assert tuned.diagnostics["correction_weights"][0] < 0.95
        assert tuned.pseudo_outcomes[0].mean() == pytest.approx(tuned.estimates[0])


def test_known_propensity_design_keeps_weight_one() -> None:
    table, predictions, outcomes, mask = _table(300, seed=7, noise=0.2, labelled=60)
    design = LabelDesign(
        kind="known_propensity", propensities={"policy": np.full(300, 0.2)}
    )
    propensities = np.full(300, 0.2)
    point = compute_direct_point_estimate(
        predictions,
        table,
        predictions,
        design,
        correction_weight="tuned",
        label_propensities=propensities,
    )
    assert point.diagnostics["routes"] == ["augmented"]
    # The Horvitz-Thompson branch is uncentred, where the least-squares slope is
    # not the variance-optimal weight, so the tuned rule keeps weight one there.
    assert point.diagnostics["correction_weights"] == [1.0]
    assert point.diagnostics["correction_weight_reasons"] == [
        "known_propensity_fixed_one"
    ]
    one = compute_direct_point_estimate(
        predictions,
        table,
        predictions,
        design,
        correction_weight="one",
        label_propensities=propensities,
    )
    assert point.estimates[0] == one.estimates[0]
    assert np.array_equal(point.pseudo_outcomes[0], one.pseudo_outcomes[0])


def test_analyze_dataset_threads_the_weight_and_narrows_the_interval(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(11)
    n_cal, n_eval = 600, 400
    cal_truth = rng.uniform(0.2, 0.8, size=n_cal)
    cal_rows = [
        {
            "prompt_id": f"c{i}",
            "judge_score": float(np.clip(t + rng.normal(0, 0.25), 0, 1)),
            "oracle_label": float(np.clip(t + rng.normal(0, 0.15), 0, 1)),
        }
        for i, t in enumerate(cal_truth)
    ]
    path = tmp_path / "calibration.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in cal_rows))
    eval_truth = rng.uniform(0.2, 0.8, size=n_eval)
    labelled = set(rng.choice(n_eval, size=80, replace=False).tolist())
    # A binary judge: informative but far from the calibrated scale, where weight one over-corrects.
    rows = [
        {
            "prompt_id": f"e{i}",
            "judge_score": float(t + rng.normal(0, 0.25) > 0.5),
            **(
                {"oracle_label": float(np.clip(t + rng.normal(0, 0.15), 0, 1))}
                if i in labelled
                else {}
            ),
        }
        for i, t in enumerate(eval_truth)
    ]
    results = {}
    for rule in ("one", "tuned"):
        results[rule] = analyze_dataset(
            fresh_draws_data={"policy": rows},
            calibration_data_path=str(path),
            combine_oracle_sources=False,
            fresh_judge_scale=(0.0, 1.0),
            fresh_oracle_scale=(0.0, 1.0),
            calibration_judge_scale=(0.0, 1.0),
            calibration_oracle_scale=(0.0, 1.0),
            estimator_config={"correction_weight": rule},
        )
    for rule, result in results.items():
        point = result.metadata["point_estimator"]
        assert point["routes"] == ["augmented"]
        assert point["correction_weight_rule"] == rule
    weights = results["tuned"].metadata["point_estimator"]["correction_weights"]
    assert len(weights) == 1 and 0.0 <= weights[0] <= 1.0
    assert results["one"].metadata["point_estimator"]["correction_weights"] == [1.0]
    assert results["tuned"].standard_errors[0] <= results["one"].standard_errors[0] * (
        1 + 1e-6
    )
    with pytest.raises(ValueError, match="correction_weight"):
        analyze_dataset(
            fresh_draws_data={"policy": rows},
            calibration_data_path=str(path),
            combine_oracle_sources=False,
            fresh_judge_scale=(0.0, 1.0),
            fresh_oracle_scale=(0.0, 1.0),
            calibration_judge_scale=(0.0, 1.0),
            calibration_oracle_scale=(0.0, 1.0),
            estimator_config={"correction_weight": "half"},
        )


def _table_with_prompts(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    mask: np.ndarray,
    prompts: Optional[np.ndarray] = None,
) -> DirectEvalTable:
    n = len(predictions)
    prompts = np.arange(n) if prompts is None else np.asarray(prompts)
    return DirectEvalTable(
        prompt_ids=prompts,
        prompt_id_strings=[f"p{i}" for i in prompts],
        policy_indices=np.zeros(n, dtype=np.int32),
        judge_scores=predictions,
        oracle_labels=np.where(mask, outcomes, np.nan),
        oracle_mask=mask,
        covariates=None,
        covariate_names=None,
        policy_names=["policy"],
    )


def test_identical_labelled_outcomes_never_give_a_zero_standard_error() -> None:
    """Identical labels gave a tuned weight of zero, constant pseudo-outcomes
    and a standard error of exactly zero; they now keep weight one."""
    rng = np.random.default_rng(41)
    n = 400
    predictions = rng.uniform(0.7, 1.0, size=n)
    outcomes = np.ones(n)
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=25, replace=False)] = True
    table = _table_with_prompts(predictions, outcomes, mask)
    point = compute_direct_point_estimate(
        predictions, table, predictions, LabelDesign(), correction_weight="tuned"
    )
    assert point.diagnostics["correction_weights"] == [1.0]
    assert point.diagnostics["correction_weight_reasons"] == ["rare_outcome"]
    assert point.diagnostics["labelled_outcomes_constant"] == [True]
    assert np.std(point.pseudo_outcomes[0]) > 0


def test_rare_outcome_guard_boundary() -> None:
    rng = np.random.default_rng(42)
    n = 60
    predictions = rng.uniform(size=n)
    weights = np.ones(n)
    for minority, expected in [
        (TUNED_WEIGHT_MIN_MINORITY - 1, "rare_outcome"),
        (TUNED_WEIGHT_MIN_MINORITY, "tuned"),
    ]:
        outcomes = np.ones(n)
        outcomes[:minority] = 0.0
        weight, reason = correction_weight_decision(
            "tuned", outcomes, predictions, weights
        )
        assert reason == expected
        if expected == "rare_outcome":
            assert weight == 1.0
    # Graded outcomes with no repeated value are never rare.
    graded = rng.uniform(size=n)
    assert correction_weight_decision("tuned", graded, predictions, weights)[1] == (
        "tuned"
    )


def test_guard_counts_labelled_prompts_not_rows() -> None:
    rng = np.random.default_rng(43)
    n = 200
    predictions = rng.uniform(size=n)
    outcomes = np.clip(predictions * 0.5 + rng.normal(0, 0.1, size=n), 0, 1)
    prompts = np.repeat(np.arange(n // 10), 10)  # ten rows per prompt
    mask = np.zeros(n, dtype=bool)
    mask[:30] = True  # 30 labelled rows from 3 prompts
    table = _table_with_prompts(predictions, outcomes, mask, prompts)
    point = compute_direct_point_estimate(
        predictions, table, predictions, LabelDesign(), correction_weight="tuned"
    )
    assert point.diagnostics["labelled_rows"] == [30]
    assert point.diagnostics["labelled_clusters"] == [3]
    assert point.diagnostics["correction_weights"] == [1.0]
    assert point.diagnostics["correction_weight_reasons"] == [
        "too_few_labelled_clusters"
    ]
    # The same rows labelled one per prompt reach the threshold.
    assert (
        correction_weight_decision(
            "tuned",
            outcomes[:30],
            predictions[:30],
            np.ones(30),
            cluster_ids=np.arange(30),
        )[1]
        == "tuned"
    )


def test_weights_are_nan_where_no_correction_applies() -> None:
    rng = np.random.default_rng(44)
    n = 100
    predictions = rng.uniform(size=n)
    outcomes = rng.uniform(size=n)
    table = DirectEvalTable(
        prompt_ids=np.concatenate([np.arange(50), np.arange(50)]),
        prompt_id_strings=[f"p{i}" for i in range(50)] * 2,
        policy_indices=np.repeat(np.array([0, 1], dtype=np.int32), 50),
        judge_scores=predictions,
        oracle_labels=np.concatenate([outcomes[:50], np.full(50, np.nan)]),
        oracle_mask=np.concatenate([np.ones(50, bool), np.zeros(50, bool)]),
        covariates=None,
        covariate_names=None,
        policy_names=["fully_labelled", "unlabelled"],
    )
    point = compute_direct_point_estimate(
        predictions, table, predictions, LabelDesign(), correction_weight="tuned"
    )
    assert point.diagnostics["routes"] == ["direct_oracle", "plug_in"]
    assert all(np.isnan(w) for w in point.diagnostics["correction_weights"])
    assert point.diagnostics["correction_weight_reasons"] == [None, None]


def test_defaults_are_weight_one() -> None:
    import inspect

    from cje import calibrated_mean_ci
    from cje.diagnostics.planning import _PLANNING_MEASUREMENT_CONFIG
    from cje.diagnostics.robust_inference import (
        cluster_bootstrap_direct_with_refit,
        direct_oracle_jackknife_estimates,
    )
    from cje.estimators.direct_method import CalibratedDirectEstimator

    for function in (
        compute_direct_point_estimate,
        direct_oracle_jackknife_estimates,
        cluster_bootstrap_direct_with_refit,
        calibrated_mean_ci,
        CalibratedDirectEstimator.__init__,
    ):
        default = inspect.signature(function).parameters["correction_weight"].default
        assert default == "one", function
    assert _PLANNING_MEASUREMENT_CONFIG["correction_weight"] == "one"


def test_bootstrap_path_reports_the_weight() -> None:
    from cje import calibrated_mean_ci

    rng = np.random.default_rng(45)
    scores = rng.uniform(size=300)
    labels = np.full(300, np.nan)
    labelled = rng.choice(300, size=60, replace=False)
    labels[labelled] = np.clip(scores[labelled] + rng.normal(0, 0.2, 60), 0, 1)
    result = calibrated_mean_ci(
        scores,
        labels,
        inference="bootstrap",
        n_bootstrap=50,
        correction_weight="tuned",
    )
    summary = result.diagnostics["correction_weight"]
    assert summary["rule"] == "tuned"
    assert summary["reason"] == "tuned"
    assert summary["route"] == "augmented"
    assert 0.0 <= summary["weight"] <= 1.0


def test_identical_labels_end_to_end_match_weight_one_and_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With every label identical the calibration itself learns a constant, so
    no weight can produce outcome variation; the tuned rule must match weight
    one exactly and the user must be warned."""
    from cje import calibrated_mean_ci

    rng = np.random.default_rng(46)
    scores = rng.uniform(0.6, 1.0, size=400)
    labels = np.full(400, np.nan)
    labels[rng.choice(400, size=40, replace=False)] = 1.0
    with caplog.at_level("WARNING"):
        tuned = calibrated_mean_ci(scores, labels, correction_weight="tuned")
    one = calibrated_mean_ci(scores, labels)
    assert tuned.diagnostics["correction_weight"]["reason"] == "rare_outcome"
    assert tuned.diagnostics["correction_weight"]["weight"] == 1.0
    assert tuned.estimate == one.estimate and tuned.se == one.se
    assert "Every labelled outcome is identical" in caplog.text


def test_rare_outcome_guard_counts_prompts_not_rows() -> None:
    """The most common outcome is the one most prompts carry, so a few prompts
    with many rows can neither hide nor fake a rare outcome."""
    assert TUNED_WEIGHT_MIN_MINORITY == 5
    rng = np.random.default_rng(47)
    # 26 one-row prompts labelled 1 and 4 ten-row prompts labelled 0: by rows
    # the mode is 0, by prompts it is 1, and only 4 prompts differ.
    outcomes = np.r_[np.ones(26), np.zeros(40)]
    clusters = np.r_[np.arange(26), np.repeat(np.arange(26, 30), 10)]
    predictions = rng.uniform(0.4, 0.6, size=len(outcomes))
    assert correction_weight_decision(
        "tuned", outcomes, predictions, np.ones(len(outcomes)), cluster_ids=clusters
    ) == (1.0, "rare_outcome")
    # Ten rows of one prompt labelled 0 against 25 prompts labelled 1: ten
    # minority rows, but one minority prompt.
    outcomes = np.r_[np.zeros(10), np.ones(25)]
    clusters = np.r_[np.zeros(10, dtype=int), np.arange(1, 26)]
    predictions = rng.uniform(size=35)
    assert correction_weight_decision(
        "tuned", outcomes, predictions, np.ones(35), cluster_ids=clusters
    ) == (1.0, "rare_outcome")
    # Five distinct minority prompts (two rows each) reach the threshold.
    outcomes = np.r_[np.zeros(10), np.ones(25)]
    clusters = np.r_[np.repeat(np.arange(5), 2), np.arange(5, 30)]
    assert (
        correction_weight_decision(
            "tuned", outcomes, predictions, np.ones(35), cluster_ids=clusters
        )[1]
        == "tuned"
    )
    # Literal boundary with one row per prompt.
    outcomes = np.ones(60)
    outcomes[:4] = 0.0
    predictions = rng.uniform(size=60)
    assert correction_weight_decision("tuned", outcomes, predictions, np.ones(60))[
        1
    ] == ("rare_outcome")
    outcomes[:5] = 0.0
    assert (
        correction_weight_decision("tuned", outcomes, predictions, np.ones(60))[1]
        == "tuned"
    )


def test_outcomes_differing_only_by_float_noise_count_as_identical() -> None:
    rng = np.random.default_rng(48)
    outcomes = np.ones(40)
    outcomes[:10] -= 1e-9
    predictions = rng.uniform(size=40)
    assert (
        correction_weight_decision("tuned", outcomes, predictions, np.ones(40))[1]
        == "rare_outcome"
    )
    n = 200
    full_predictions = rng.uniform(size=n)
    full_outcomes = np.ones(n)
    full_outcomes[:20] -= 1e-9
    mask = np.zeros(n, dtype=bool)
    mask[:40] = True
    table = _table_with_prompts(full_predictions, full_outcomes, mask)
    point = compute_direct_point_estimate(
        full_predictions,
        table,
        full_predictions,
        LabelDesign(),
        correction_weight="tuned",
    )
    assert point.diagnostics["labelled_outcomes_constant"] == [True]
    assert point.diagnostics["correction_weight_min_minority"] == 5


def test_constant_predictions_reason() -> None:
    rng = np.random.default_rng(49)
    outcomes = rng.uniform(size=200)
    assert correction_weight_decision(
        "tuned", outcomes, np.full(200, 0.3), np.ones(200)
    ) == (1.0, "constant_predictions")
    assert correction_weight_decision(
        "one", outcomes, rng.uniform(size=200), np.ones(200)
    ) == (1.0, "rule_one")


def test_known_propensity_reason_under_each_rule() -> None:
    table, predictions, outcomes, mask = _table(300, seed=8, noise=0.2, labelled=60)
    propensities = np.full(300, 0.2)
    design = LabelDesign(kind="known_propensity", propensities={"policy": propensities})
    for rule, reason in [("one", "rule_one"), ("tuned", "known_propensity_fixed_one")]:
        point = compute_direct_point_estimate(
            predictions,
            table,
            predictions,
            design,
            correction_weight=rule,
            label_propensities=propensities,
        )
        assert point.diagnostics["correction_weights"] == [1.0]
        assert point.diagnostics["correction_weight_reasons"] == [reason]
        assert point.diagnostics["labelled_rows"] == [60]
        assert point.diagnostics["labelled_clusters"] == [60]


def test_routes_without_a_correction_report_nan_and_none() -> None:
    rng = np.random.default_rng(50)
    predictions = rng.uniform(size=100)
    outcomes = rng.uniform(size=100)
    table = DirectEvalTable(
        prompt_ids=np.concatenate([np.arange(50), np.arange(50)]),
        prompt_id_strings=[f"p{i}" for i in range(50)] * 2,
        policy_indices=np.repeat(np.array([0, 1], dtype=np.int32), 50),
        judge_scores=predictions,
        oracle_labels=np.concatenate([outcomes[:50], np.full(50, np.nan)]),
        oracle_mask=np.concatenate([np.ones(50, bool), np.zeros(50, bool)]),
        covariates=None,
        covariate_names=None,
        policy_names=["fully_labelled", "unlabelled", "empty"],
    )
    point = compute_direct_point_estimate(
        predictions, table, predictions, LabelDesign(), correction_weight="tuned"
    )
    diagnostics = point.diagnostics
    assert diagnostics["routes"] == ["direct_oracle", "plug_in", "no_data"]
    assert all(np.isnan(w) for w in diagnostics["correction_weights"])
    assert diagnostics["correction_weight_reasons"] == [None, None, None]
    assert diagnostics["labelled_rows"] == [50, 0, 0]
    assert diagnostics["labelled_clusters"] == [50, 0, 0]
    assert diagnostics["labelled_outcomes_constant"] == [False, False, False]
    # Targeted labels with unknown probabilities stay on the plug-in route.
    partial, *_ = _table(200, seed=9, noise=0.2, labelled=40)
    targeted = compute_direct_point_estimate(
        partial.judge_scores,
        partial,
        partial.judge_scores,
        LabelDesign(kind="targeted_unknown"),
        correction_weight="tuned",
    )
    assert targeted.diagnostics["routes"] == ["plug_in_targeted_unknown"]
    assert np.isnan(targeted.diagnostics["correction_weights"][0])
    assert targeted.diagnostics["correction_weight_reasons"] == [None]


def test_fully_labelled_array_api_reports_no_weight() -> None:
    from cje import calibrated_mean_ci

    rng = np.random.default_rng(51)
    scores = rng.uniform(size=60)
    labels = rng.uniform(size=60)
    for inference in ("cluster_robust", "bootstrap"):
        extra = {"n_bootstrap": 30} if inference == "bootstrap" else {}
        result = calibrated_mean_ci(
            scores, labels, correction_weight="tuned", inference=inference, **extra
        )
        summary = result.diagnostics["correction_weight"]
        assert summary["rule"] == "tuned"
        assert summary["route"] == "direct_oracle"
        assert np.isnan(summary["weight"]) and summary["reason"] is None


def _binary_judge_arrays() -> tuple:
    """A binary-ish judge on binary outcomes, where the tuned weight sits well
    below one, so any path that silently used weight one would differ."""
    rng = np.random.default_rng(7)
    n = 600
    truth = rng.uniform(0.2, 0.8, n)
    outcomes = (rng.uniform(size=n) < truth).astype(float)
    scores = (truth + rng.normal(0, 0.25, n) > 0.5).astype(float) * 0.6 + rng.uniform(
        0, 0.4, n
    )
    labels = np.full(n, np.nan)
    labelled = rng.choice(n, 80, replace=False)
    labels[labelled] = outcomes[labelled]
    return scores, labels


def test_tuned_rule_reaches_the_jackknife_and_bootstrap_replicates() -> None:
    from cje import calibrated_mean_ci

    scores, labels = _binary_judge_arrays()
    one = calibrated_mean_ci(scores, labels)
    tuned = calibrated_mean_ci(scores, labels, correction_weight="tuned")
    assert tuned.diagnostics["correction_weight"]["weight"] < 0.9
    var_one = one.diagnostics["cluster_robust"]["var_oracle"]
    var_tuned = tuned.diagnostics["cluster_robust"]["var_oracle"]
    assert not np.isclose(var_one, var_tuned, rtol=1e-6)
    boot_one = calibrated_mean_ci(
        scores, labels, inference="bootstrap", n_bootstrap=40, seed=0
    )
    boot_tuned = calibrated_mean_ci(
        scores,
        labels,
        inference="bootstrap",
        n_bootstrap=40,
        seed=0,
        correction_weight="tuned",
    )
    assert not np.isclose(boot_one.se, boot_tuned.se, rtol=1e-9)


def _binary_judge_rows(tmp_path: Path, constant_labels: bool = False) -> dict:
    rng = np.random.default_rng(12)
    n_cal, n_eval = 400, 400
    cal_truth = rng.uniform(0.2, 0.8, size=n_cal)
    cal_rows = [
        {
            "prompt_id": f"c{i}",
            "judge_score": float(np.clip(t + rng.normal(0, 0.25), 0, 1)),
            "oracle_label": float(rng.uniform() < t),
        }
        for i, t in enumerate(cal_truth)
    ]
    path = tmp_path / "calibration.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in cal_rows))
    eval_truth = rng.uniform(0.2, 0.8, size=n_eval)
    labelled = set(rng.choice(n_eval, size=80, replace=False).tolist())
    rows = []
    for i, t in enumerate(eval_truth):
        row: dict = {
            "prompt_id": f"e{i}",
            "judge_score": float(t + rng.normal(0, 0.25) > 0.5),
        }
        if i in labelled:
            row["oracle_label"] = 1.0 if constant_labels else float(rng.uniform() < t)
        rows.append(row)
    return dict(
        fresh_draws_data={"policy": rows},
        calibration_data_path=str(path),
        combine_oracle_sources=False,
        fresh_judge_scale=(0.0, 1.0),
        fresh_oracle_scale=(0.0, 1.0),
        calibration_judge_scale=(0.0, 1.0),
        calibration_oracle_scale=(0.0, 1.0),
    )


def test_estimator_threads_the_tuned_rule_to_jackknife_and_bootstrap(
    tmp_path: Path,
) -> None:
    kwargs = _binary_judge_rows(tmp_path)
    jack = {}
    boot = {}
    for rule in ("one", "tuned"):
        result = analyze_dataset(**kwargs, estimator_config={"correction_weight": rule})
        jack[rule] = result.metadata["se_components"]["oracle_variance_per_policy"][
            "policy"
        ]
        result = analyze_dataset(
            **kwargs,
            estimator_config={
                "correction_weight": rule,
                "inference_method": "bootstrap",
                "n_bootstrap": 40,
                "bootstrap_seed": 0,
            },
        )
        boot[rule] = result.standard_errors[0]
    assert not np.isclose(jack["one"], jack["tuned"], rtol=1e-6)
    assert not np.isclose(boot["one"], boot["tuned"], rtol=1e-9)


def test_estimator_warns_on_identical_labelled_outcomes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    kwargs = _binary_judge_rows(tmp_path, constant_labels=True)
    with caplog.at_level("WARNING", logger="cje.estimators.direct_method"):
        result = analyze_dataset(
            **kwargs, estimator_config={"correction_weight": "tuned"}
        )
    point = result.metadata["point_estimator"]
    assert point["labelled_outcomes_constant"] == [True]
    assert point["correction_weight_reasons"] == ["rare_outcome"]
    assert "Every labelled outcome is identical for policy/policies policy" in (
        caplog.text
    )
