"""#66 part A: calibrated_mean_ci(fit_calibrator=True) at complete label coverage.

The direct oracle mean stays the estimate on every inference path; the fitted
calibrator is descriptive (returned for reuse, with out-of-fold diagnostics)
and never changes the estimate, its interval or its route.
"""

import inspect
import json
import warnings
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import pytest

from cje import calibrated_mean_ci, transport_audit
from cje.calibration.judge import JudgeCalibrator


def _data(
    n_prompts: int = 60, draws: int = 2, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    clusters = np.repeat(np.arange(n_prompts), draws)
    n = len(clusters)
    judge = rng.uniform(size=n)
    covariates = np.column_stack(
        [
            rng.normal(size=n_prompts)[clusters] + 0.3 * rng.normal(size=n),
            rng.binomial(1, 0.4, size=n_prompts)[clusters].astype(float),
        ]
    )
    labels = np.clip(
        0.15
        + 0.6 * judge
        + 0.05 * covariates[:, 0]
        + 0.05 * covariates[:, 1]
        + rng.normal(0, 0.1, size=n),
        0.0,
        1.0,
    )
    return judge, labels, clusters, covariates


def _canon(value: Any) -> Any:
    """Exact, NaN-aware canonical form for equality checks."""
    if isinstance(value, dict):
        return {k: _canon(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    return value


def _call(*args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return calibrated_mean_ci(*args, **kwargs)


# (inference, extra kwargs, n_prompts): "auto" resolves to bootstrap below 20
# clusters and to cluster_robust above.
_PATHS = [
    ("cluster_robust", {}, 60),
    ("bootstrap", {"n_bootstrap": 50}, 60),
    ("auto", {}, 60),
    ("auto", {"n_bootstrap": 50}, 15),
]


@pytest.mark.parametrize("inference,extra,n_prompts", _PATHS)
@pytest.mark.parametrize("with_covariates", [False, True])
def test_fit_calibrator_keeps_direct_estimate_identical(
    inference: str,
    extra: Dict[str, Any],
    n_prompts: int,
    with_covariates: bool,
) -> None:
    judge, labels, clusters, covariates = _data(n_prompts=n_prompts)
    kwargs: Dict[str, Any] = dict(
        cluster_ids=clusters,
        covariates=covariates if with_covariates else None,
        inference=inference,
        seed=3,
        **extra,
    )
    base = _call(judge, labels, **kwargs)
    fitted = _call(judge, labels, fit_calibrator=True, **kwargs)

    assert base.calibrator is None
    assert isinstance(fitted.calibrator, JudgeCalibrator)
    assert fitted.estimate == base.estimate == pytest.approx(float(np.mean(labels)))
    assert _canon(fitted.se) == _canon(base.se)
    assert _canon(fitted.ci) == _canon(base.ci)
    assert (fitted.method, fitted.n, fitted.n_oracle) == (
        base.method,
        base.n,
        base.n_oracle,
    )
    assert fitted.diagnostics["estimator_route"] == "direct_oracle"
    assert fitted.summary() == base.summary()
    # Everything but the calibration block is the default call's, and the
    # calibrator never adds a boundary card or any other gate input.
    assert set(fitted.diagnostics) == set(base.diagnostics)
    assert "boundary_card" not in fitted.diagnostics
    for key in base.diagnostics:
        if key != "calibration":
            assert _canon(fitted.diagnostics[key]) == _canon(base.diagnostics[key])
    calibration = fitted.diagnostics["calibration"]
    assert calibration["used_for_estimate"] is False
    assert calibration["calibrator_available"] is True
    if inference == "auto":
        expected = "bootstrap" if n_prompts < 20 else "cluster_robust"
        assert fitted.method == expected


def test_two_stage_with_covariates() -> None:
    judge, labels, clusters, covariates = _data()
    result = calibrated_mean_ci(
        judge, labels, cluster_ids=clusters, covariates=covariates, fit_calibrator=True
    )
    calibration = result.diagnostics["calibration"]
    assert calibration["mode"] == "two_stage"
    assert calibration["selected_mode"] == "two_stage"
    assert calibration["covariates_used"] is True
    assert calibration["n_folds_without_covariates"] == 0
    assert calibration["n_oracle"] == len(judge)
    assert calibration["oracle_coverage"] == 1.0
    assert calibration["n_folds"] == 5
    assert result.calibrator is not None
    assert result.calibrator.covariate_names == ["cov_0", "cov_1"]


def test_auto_mode_without_covariates() -> None:
    judge, labels, clusters, _ = _data()
    result = calibrated_mean_ci(
        judge, labels, cluster_ids=clusters, fit_calibrator=True
    )
    calibration = result.diagnostics["calibration"]
    assert calibration["mode"] == "auto"
    assert calibration["selected_mode"] in ("monotone", "two_stage")
    assert calibration["covariates_used"] is False


def _direct_fit(
    judge: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    covariates: Optional[np.ndarray],
    seed: int,
) -> Tuple[JudgeCalibrator, Any]:
    names = None if covariates is None else ["cov_0", "cov_1"]
    calibrator = JudgeCalibrator(
        random_seed=seed,
        calibration_mode="two_stage" if covariates is not None else "auto",
        covariate_names=names,
    )
    fit = calibrator.fit_cv(
        judge,
        labels,
        np.ones(len(judge), dtype=bool),
        n_folds=5,
        prompt_ids=[str(c) for c in clusters],
        covariates=covariates,
        quiet=True,
    )
    return calibrator, fit


@pytest.mark.parametrize("with_covariates", [False, True])
def test_calibrator_equals_direct_fit_cv(with_covariates: bool) -> None:
    judge, labels, clusters, covariates = _data()
    cov = covariates if with_covariates else None
    result = calibrated_mean_ci(
        judge,
        labels,
        cluster_ids=clusters,
        covariates=cov,
        seed=11,
        fit_calibrator=True,
    )
    direct, fit = _direct_fit(judge, labels, clusters, cov, seed=11)
    assert result.calibrator is not None
    grid = np.linspace(-0.1, 1.1, 25)
    grid_cov = None if cov is None else np.tile(cov[:1], (len(grid), 1))
    np.testing.assert_array_equal(
        result.calibrator.predict(grid, covariates=grid_cov),
        direct.predict(grid, covariates=grid_cov),
    )
    np.testing.assert_array_equal(
        result.calibrator.predict_oof(judge, fit.fold_ids, covariates=cov),
        direct.predict_oof(judge, fit.fold_ids, covariates=cov),
    )
    calibration = result.diagnostics["calibration"]
    assert calibration["rmse"] == fit.calibration_rmse
    assert calibration["oof_rmse"] == fit.oof_rmse
    assert calibration["coverage_at_01"] == fit.coverage_at_01
    assert calibration["oof_coverage_at_01"] == fit.oof_coverage_at_01
    assert calibration["selected_mode"] == direct.selected_mode


@pytest.mark.parametrize("with_covariates", [False, True])
def test_oof_metrics_match_definitions(with_covariates: bool) -> None:
    judge, labels, clusters, covariates = _data()
    cov = covariates if with_covariates else None
    result = calibrated_mean_ci(
        judge, labels, cluster_ids=clusters, covariates=cov, fit_calibrator=True
    )
    _, fit = _direct_fit(judge, labels, clusters, cov, seed=42)
    assert result.calibrator is not None
    oof = result.calibrator.predict_oof(judge, fit.fold_ids, covariates=cov)
    sse = float(np.sum((labels - oof) ** 2))
    sst = float(np.sum((labels - labels.mean()) ** 2))
    calibration = result.diagnostics["calibration"]
    assert calibration["oof_r2"] == pytest.approx(1.0 - sse / sst, rel=1e-12)
    assert calibration["oof_correlation"] == pytest.approx(
        float(np.corrcoef(oof, labels)[0, 1]), rel=1e-12
    )
    assert calibration["oof_rmse"] == pytest.approx(
        float(np.sqrt(np.mean((labels - oof) ** 2))), rel=1e-12
    )
    # An informative judge: the out-of-fold fit explains much of the label.
    assert 0.3 < calibration["oof_r2"] < 1.0
    assert calibration["oof_r2"] <= calibration["oof_correlation"] ** 2 + 1e-12
    json.dumps(calibration, allow_nan=False)


def test_oof_metrics_none_for_constant_labels() -> None:
    judge, _, clusters, _ = _data()
    constant = np.full(len(judge), 0.4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = calibrated_mean_ci(
            judge, constant, cluster_ids=clusters, fit_calibrator=True
        )
    calibration = result.diagnostics["calibration"]
    assert calibration["oof_r2"] is None
    assert calibration["oof_correlation"] is None
    assert result.estimate == pytest.approx(0.4)
    json.dumps(calibration, allow_nan=False)


def test_too_few_clusters_raises_naming_fit_calibrator() -> None:
    judge = np.asarray([0.1, 0.4, 0.7, 0.2, 0.5, 0.9])
    labels = np.asarray([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    clusters = ["a", "b", "c", "a", "b", "c"]
    default = calibrated_mean_ci(judge, labels, cluster_ids=clusters)
    assert default.calibrator is None
    with pytest.raises(ValueError, match="fit_calibrator=True could not fit"):
        calibrated_mean_ci(judge, labels, cluster_ids=clusters, fit_calibrator=True)


def test_covariate_warning_only_without_fit_calibrator() -> None:
    judge, labels, clusters, covariates = _data()
    with pytest.warns(UserWarning, match="every row is labelled"):
        calibrated_mean_ci(judge, labels, cluster_ids=clusters, covariates=covariates)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibrated_mean_ci(
            judge,
            labels,
            cluster_ids=clusters,
            covariates=covariates,
            fit_calibrator=True,
        )
    assert not any("every row is labelled" in str(w.message) for w in caught)


def test_fit_calibrator_type() -> None:
    judge, labels, clusters, _ = _data()
    plain = calibrated_mean_ci(judge, labels, cluster_ids=clusters, fit_calibrator=True)
    numpy_true = calibrated_mean_ci(
        judge, labels, cluster_ids=clusters, fit_calibrator=np.True_
    )
    assert _canon(plain.diagnostics) == _canon(numpy_true.diagnostics)
    numpy_false = calibrated_mean_ci(
        judge, labels, cluster_ids=clusters, fit_calibrator=np.False_
    )
    assert numpy_false.calibrator is None
    for bad in ("yes", 1, 0, None, 1.0):
        with pytest.raises(TypeError, match="fit_calibrator must be True or False"):
            calibrated_mean_ci(
                judge, labels, cluster_ids=clusters, fit_calibrator=cast(Any, bad)
            )


@pytest.mark.parametrize(
    "inference,extra",
    [("cluster_robust", {}), ("bootstrap", {"n_bootstrap": 30})],
)
@pytest.mark.parametrize("with_covariates", [False, True])
def test_fit_calibrator_noop_at_partial_coverage(
    inference: str, extra: Dict[str, Any], with_covariates: bool
) -> None:
    judge, labels, clusters, covariates = _data()
    rng = np.random.default_rng(5)
    labelled_prompts = rng.choice(60, size=30, replace=False)
    partial = np.where(np.isin(clusters, labelled_prompts), labels, np.nan)
    kwargs: Dict[str, Any] = dict(
        cluster_ids=clusters,
        covariates=covariates if with_covariates else None,
        inference=inference,
        **extra,
    )
    base = _call(judge, partial, **kwargs)
    fitted = _call(judge, partial, fit_calibrator=True, **kwargs)
    assert _canon(
        [fitted.estimate, fitted.se, fitted.ci, fitted.method, fitted.diagnostics]
    ) == _canon([base.estimate, base.se, base.ci, base.method, base.diagnostics])
    grid = np.linspace(0.0, 1.0, 11)
    grid_cov = None if not with_covariates else np.tile(covariates[:1], (11, 1))
    np.testing.assert_array_equal(
        fitted.calibrator.predict(grid, covariates=grid_cov),
        base.calibrator.predict(grid, covariates=grid_cov),
    )


def test_composes_with_transport_audit() -> None:
    rng = np.random.default_rng(21)
    judge = rng.uniform(size=400)
    labels = np.clip(0.1 + 0.6 * judge + rng.normal(0, 0.08, size=400), 0.0, 1.0)
    pilot = calibrated_mean_ci(judge, labels, fit_calibrator=True)
    assert pilot.diagnostics["estimator_route"] == "direct_oracle"

    probe = rng.uniform(size=400)
    probe_labels = np.clip(0.1 + 0.6 * probe + rng.normal(0, 0.08, size=400), 0.0, 1.0)
    same = transport_audit(probe, probe_labels, pilot.calibrator, delta_max=0.05)
    shifted = transport_audit(
        probe, probe_labels + 0.15, pilot.calibrator, delta_max=0.05
    )
    assert same.status == "PASS"
    assert shifted.status == "FAIL"


def test_public_signature_adds_keyword_only_defaults() -> None:
    parameters = inspect.signature(calibrated_mean_ci).parameters
    for name, default in (("fit_calibrator", False), ("oracle_scale", None)):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is default
    names = list(parameters)
    assert names.index("correction_weight") < names.index("oracle_scale")
    assert names.index("oracle_scale") < names.index("fit_calibrator")
