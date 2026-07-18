"""Focused regressions for route-aware Direct estimation and inference."""

import numpy as np
import pytest

from cje.calibration import calibrate_dataset
from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample
from cje.data.models import (
    Dataset,
    EstimationResult,
    InferenceUnavailableError,
    Sample,
)
from cje.diagnostics.robust_inference import (
    oracle_jackknife_estimates,
    oracle_jackknife_variance,
)
from cje.estimators.direct_method import CalibratedDirectEstimator
from cje.interface.analysis import analyze_dataset


pytestmark = pytest.mark.unit


def _fresh(
    policy: str, values: np.ndarray, *, labeled: bool = True
) -> FreshDrawDataset:
    return FreshDrawDataset(
        target_policy=policy,
        samples=[
            FreshDrawSample(
                prompt_id=f"p{index}",
                target_policy=policy,
                judge_score=float(value),
                oracle_label=float(value) if labeled else None,
                response=None,
                draw_idx=0,
            )
            for index, value in enumerate(values)
        ],
    )


def _direct_result(
    a: np.ndarray,
    b: np.ndarray,
    *,
    paired: bool,
    inference_method: str = "cluster_robust",
    n_bootstrap: int = 120,
) -> EstimationResult:
    estimator = CalibratedDirectEstimator(
        target_policies=["A", "B"],
        paired_comparison=paired,
        inference_method=inference_method,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=7,
    )
    estimator.add_fresh_draws("A", _fresh("A", a))
    estimator.add_fresh_draws("B", _fresh("B", b))
    return estimator.fit_and_estimate()


def test_target_policy_order_does_not_depend_on_add_order() -> None:
    estimator = CalibratedDirectEstimator(
        target_policies=["A", "B"], inference_method="cluster_robust"
    )
    estimator.add_fresh_draws("B", _fresh("B", np.zeros(4)))
    estimator.add_fresh_draws("A", _fresh("A", np.ones(4)))

    result = estimator.fit_and_estimate()

    assert result.target_policies == ["A", "B"]
    assert result.estimates.tolist() == pytest.approx([1.0, 0.0])
    assert result.metadata["point_estimator"]["routes"] == [
        "direct_oracle",
        "direct_oracle",
    ]


def test_explicit_full_oracle_bootstrap_needs_no_calibrator() -> None:
    x = np.linspace(0.1, 0.8, 16)
    result = _direct_result(
        x,
        x + 0.1,
        paired=True,
        inference_method="bootstrap",
    )

    assert result.method == "direct_oracle_bootstrap"
    assert result.bootstrap_samples is not None
    assert result.bootstrap_samples.shape == (120, 2)
    assert np.all(np.isfinite(result.standard_errors))
    components = result.metadata["se_components"]
    assert components["includes_oracle_uncertainty"] is False
    assert "oracle_variance_per_policy" not in components
    assert components["calibration_variance_decomposition"]["available"] is False


def test_unpaired_analytic_inference_uses_independent_policy_ses() -> None:
    x = np.linspace(0.1, 0.8, 20)
    y = np.clip(x + 0.08 + 0.04 * np.sin(np.arange(len(x))), 0.0, 1.0)
    paired = _direct_result(x, y, paired=True)
    independent = _direct_result(x, y, paired=False)

    paired_entry = paired.metadata["pairwise_inference"]["0-1"]
    independent_entry = independent.metadata["pairwise_inference"]["0-1"]
    expected = float(np.hypot(*independent.standard_errors))

    assert paired_entry["basis"] == "prompt_cluster_paired"
    assert independent_entry["basis"] == "independent_requested"
    assert independent_entry["se_sampling"] == pytest.approx(expected)
    assert independent_entry["se_sampling"] > paired_entry["se_sampling"]


def test_unpaired_bootstrap_decouples_policy_prompt_weights() -> None:
    x = np.linspace(0.1, 0.8, 20)
    paired = _direct_result(x, x + 0.1, paired=True, inference_method="bootstrap")
    independent = _direct_result(x, x + 0.1, paired=False, inference_method="bootstrap")
    assert paired.bootstrap_samples is not None
    assert independent.bootstrap_samples is not None

    paired_delta_sd = np.std(
        paired.bootstrap_samples[:, 0] - paired.bootstrap_samples[:, 1], ddof=1
    )
    independent_delta_sd = np.std(
        independent.bootstrap_samples[:, 0] - independent.bootstrap_samples[:, 1],
        ddof=1,
    )

    assert paired_delta_sd == pytest.approx(0.0, abs=1e-12)
    assert independent_delta_sd > 0.01
    assert (
        independent.metadata["inference"]["prompt_weight_coupling"]
        == "independent_by_policy_prompt"
    )


def test_mixed_small_label_run_preserves_direct_policy() -> None:
    result = analyze_dataset(
        fresh_draws_data={
            "A": [
                {"prompt_id": f"p{i}", "judge_score": 0.0, "oracle_label": 1.0}
                for i in range(3)
            ],
            "B": [{"prompt_id": f"p{i}", "judge_score": 0.25} for i in range(3)],
        },
        estimator_config={"inference_method": "cluster_robust"},
    )

    assert result.method == "mixed_direct"
    assert result.estimates.tolist() == pytest.approx([1.0, 0.25])
    assert result.metadata["point_estimator"]["routes"] == [
        "direct_oracle",
        "plug_in",
    ]
    assert result.metadata["claim_tier_by_policy"] == {
        "A": "DIRECT_ORACLE_MEAN",
        "B": "RAW_JUDGE_MEAN",
    }
    assert result.metadata["calibration_status"] == "MIXED"
    assert "A" not in result.metadata.get("reliability_gates", {})
    assert result.metadata["reliability_gates"]["B"]["flagged"] is True
    assert result.units is not None
    assert result.units.estimand == "mixed"


def test_direct_oracle_route_keeps_boundary_diagnostic_without_gate() -> None:
    calibration = Dataset(
        target_policies=[],
        samples=[
            Sample(
                prompt_id=f"c{i}",
                prompt="",
                response="",
                reward=None,
                judge_score=float(score),
                oracle_label=float(score),
            )
            for i, score in enumerate(np.linspace(0.2, 0.7, 20))
        ],
    )
    _, fitted = calibrate_dataset(calibration)
    estimator = CalibratedDirectEstimator(
        target_policies=["A"],
        reward_calibrator=fitted.calibrator,
        inference_method="cluster_robust",
    )
    estimator.add_fresh_draws("A", _fresh("A", np.full(10, 0.95)))

    result = estimator.fit_and_estimate()

    card = result.metadata["boundary_cards"]["A"]
    assert card["status"] == "REFUSE-LEVEL"
    assert card["applies_to_current_estimate"] is False
    assert result.metadata["reliability_gates"]["A"]["flagged"] is False
    components = result.metadata["se_components"]
    assert components["includes_oracle_uncertainty"] is False
    assert components["oracle_jackknife_status_per_policy"]["A"] == (
        "skipped_direct_oracle"
    )
    assert result.metadata["degrees_of_freedom"]["A"]["df"] == 9
    assert result.metadata["degrees_of_freedom"]["A"]["oracle_jackknife_folds"] == 0


def test_oua_recomputes_augmented_functional_not_only_plugin() -> None:
    scores = np.linspace(0.05, 0.95, 20)
    outcomes = np.clip(
        0.15 + 0.7 * scores + 0.15 * np.sin(np.arange(len(scores))), 0.0, 1.0
    )
    labels = [
        float(outcome) if index % 2 == 0 else None
        for index, outcome in enumerate(outcomes)
    ]
    calibration = Dataset(
        target_policies=["A"],
        samples=[
            Sample(
                prompt_id=f"p{index}",
                prompt="",
                response="",
                reward=None,
                judge_score=float(score),
                oracle_label=labels[index],
            )
            for index, score in enumerate(scores)
        ],
    )
    _, fitted = calibrate_dataset(calibration)
    draws = FreshDrawDataset(
        target_policy="A",
        samples=[
            FreshDrawSample(
                prompt_id=f"p{index}",
                target_policy="A",
                judge_score=float(score),
                oracle_label=labels[index],
                response=None,
                draw_idx=0,
            )
            for index, score in enumerate(scores)
        ],
    )
    estimator = CalibratedDirectEstimator(
        target_policies=["A"],
        reward_calibrator=fitted.calibrator,
        inference_method="cluster_robust",
    )
    estimator.add_fresh_draws("A", draws)
    result = estimator.fit_and_estimate()

    actual = estimator.get_oracle_jackknife("A")
    plugin_only = oracle_jackknife_estimates(fitted.calibrator, scores)
    assert actual is not None
    assert plugin_only is not None
    assert result.metadata["point_estimator"]["routes"] == ["augmented"]
    assert not np.allclose(actual, plugin_only)
    assert result.metadata["se_components"]["oracle_variance_per_policy"][
        "A"
    ] == pytest.approx(oracle_jackknife_variance(actual))


def test_unavailable_oua_does_not_claim_variance_or_cap_df() -> None:
    class NoFoldCalibrator:
        covariate_names = None

        def predict(self, scores, covariates=None):  # type: ignore[no-untyped-def]
            return np.asarray(scores, dtype=float)

        def get_fold_models_for_oua(self):  # type: ignore[no-untyped-def]
            return {}

        def get_calibration_info(self):  # type: ignore[no-untyped-def]
            return {}

    estimator = CalibratedDirectEstimator(
        target_policies=["A"],
        reward_calibrator=NoFoldCalibrator(),
        inference_method="cluster_robust",
    )
    estimator.add_fresh_draws("A", _fresh("A", np.linspace(0.1, 0.9, 6), labeled=False))
    result = estimator.fit_and_estimate()

    components = result.metadata["se_components"]
    assert components["includes_oracle_uncertainty"] is False
    assert components["oracle_jackknife_status_per_policy"]["A"] == "unavailable"
    assert components["oracle_jackknife_counts"]["A"] == 0
    assert result.metadata["degrees_of_freedom"]["A"]["df"] == 5
    assert (
        result.metadata["degrees_of_freedom"]["A"]["oracle_jackknife_status"]
        == "unavailable"
    )


def test_underclustered_bootstrap_policy_blocks_pairwise_fallback() -> None:
    one_cluster = FreshDrawDataset(
        target_policy="A",
        samples=[
            FreshDrawSample(
                prompt_id="shared",
                target_policy="A",
                judge_score=value,
                oracle_label=value,
                response=None,
                draw_idx=index,
            )
            for index, value in enumerate((0.2, 0.8))
        ],
    )
    estimator = CalibratedDirectEstimator(
        target_policies=["A", "B"],
        inference_method="bootstrap",
        n_bootstrap=120,
    )
    estimator.add_fresh_draws("A", one_cluster)
    estimator.add_fresh_draws("B", _fresh("B", np.asarray([0.3, 0.7])))
    result = estimator.fit_and_estimate()

    assert np.isnan(result.standard_errors[0])
    assert result.metadata["inference_unavailable_policies"] == ["A"]
    with pytest.raises(InferenceUnavailableError, match="fewer than two"):
        result.compare_policies(0, 1)
