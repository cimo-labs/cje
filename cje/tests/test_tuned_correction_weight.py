"""The tuned correction weight: PPI++ power tuning inside the augmented estimator.

Weight one (the default) must be unchanged.  The tuned weight must equal the
least-squares slope of labelled outcomes on predictions, clipped to [0, 1];
with uninformative predictions it must reduce the estimator to the labelled
mean, and it must never raise the pseudo-outcome variance above weight one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cje import analyze_dataset
from cje.diagnostics.robust_inference import (
    DirectEvalTable,
    LabelDesign,
    compute_direct_point_estimate,
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
    default = compute_direct_point_estimate(
        predictions, table, predictions, LabelDesign()
    )
    assert default.estimates[0] == point.estimates[0]
    assert np.array_equal(default.pseudo_outcomes[0], point.pseudo_outcomes[0])


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


def test_known_propensity_design_accepts_tuned_weight() -> None:
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
    assert 0.0 <= point.diagnostics["correction_weights"][0] <= 1.0
    assert np.isfinite(point.estimates[0])


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
