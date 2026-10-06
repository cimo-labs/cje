"""#66 part A: calibrated_mean_ci(oracle_scale=(lo, hi)).

Labels on a declared bounded scale are mapped to [0, 1] with the scale's own
normalisation and every output comes back in the declared units. The reference
for each check is the unit call on exactly that internal normalisation, so the
mapped results must match it exactly, leaf by leaf.
"""

import warnings
from typing import Any, Dict, Iterator, List, Set, Tuple

import numpy as np
import pytest

from cje import calibrated_mean_ci, transport_audit
from cje.array_api import (
    _ORACLE_SCALE_LEVEL_PATHS,
    _ORACLE_SCALE_SPREAD_PATHS,
    _ORACLE_SCALE_UNCHANGED_PATHS,
    _ORACLE_SCALE_VARIANCE_PATHS,
    _diagnostics_to_oracle_units,
    _is_numeric_leaf,
)
from cje.calibration.judge import JudgeCalibrator
from cje.data.normalization import ScaledCalibrator, ScaleInfo

LO, HI = -2.0, 23.0  # a non-zero minimum separates levels from spreads
SCALE = ScaleInfo(LO, HI)
SPAN = HI - LO


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
    unit = np.clip(
        0.15
        + 0.6 * judge
        + 0.05 * covariates[:, 0]
        + 0.05 * covariates[:, 1]
        + rng.normal(0, 0.1, size=n),
        0.0,
        1.0,
    )
    return judge, unit, clusters, covariates


def _partial(labels: np.ndarray, clusters: np.ndarray, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    prompts = np.unique(clusters)
    labelled = rng.choice(prompts, size=len(prompts) // 2, replace=False)
    return np.where(np.isin(clusters, labelled), labels, np.nan)


def _call(*args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return calibrated_mean_ci(*args, **kwargs)


def _at(node: Any, path: Tuple[str, ...], index: List[int]) -> Any:
    """Leaf of ``node`` at ``path``, using ``index`` for list positions."""
    positions = iter(index)
    for key in path:
        node = node[next(positions)] if key == "*" else node[key]
    return node


def _indexed_leaves(
    node: Any, path: Tuple[str, ...] = (), index: Tuple[int, ...] = ()
) -> Iterator[Tuple[Tuple, Tuple[int, ...], Any]]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _indexed_leaves(value, path + (str(key),), index)
    elif isinstance(node, (list, tuple)):
        for position, value in enumerate(node):
            yield from _indexed_leaves(value, path + ("*",), index + (position,))
    else:
        yield path, index, node


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float) and np.isnan(expected):
        return isinstance(actual, float) and np.isnan(actual)
    return bool(actual == expected) and type(actual) is type(expected)


def _expected(path: Tuple[str, ...], unit_value: Any) -> Any:
    if path in _ORACLE_SCALE_LEVEL_PATHS:
        return float(SCALE.inverse(float(unit_value)))
    if path in _ORACLE_SCALE_SPREAD_PATHS:
        return float(unit_value) * SPAN
    if path in _ORACLE_SCALE_VARIANCE_PATHS:
        return float(unit_value) * SPAN**2
    assert path in _ORACLE_SCALE_UNCHANGED_PATHS, f"unclassified leaf {path}"
    return unit_value


def _assert_scaled_matches_unit(scaled: Any, unit: Any) -> Set[Tuple[str, ...]]:
    """Check every scaled output against the mapped unit call; return paths."""
    assert scaled.estimate == SCALE.inverse(unit.estimate) or (
        np.isnan(scaled.estimate) and np.isnan(unit.estimate)
    )
    assert _same(scaled.se, float(unit.se) * SPAN)
    for scaled_end, unit_end in zip(scaled.ci, unit.ci):
        assert _same(scaled_end, float(SCALE.inverse(unit_end)))
    assert (scaled.method, scaled.n, scaled.n_oracle) == (
        unit.method,
        unit.n,
        unit.n_oracle,
    )
    assert scaled.diagnostics["oracle_scale"] == SCALE.to_dict()

    seen: Set[Tuple[str, ...]] = set()
    scaled_paths = set()
    for path, index, value in _indexed_leaves(scaled.diagnostics):
        if path[:1] == ("oracle_scale",) or path == (
            "calibration",
            "coverage_tolerance",
        ):
            if path == ("calibration", "coverage_tolerance"):
                assert value == pytest.approx(0.1 * SPAN, rel=1e-15)
            if _is_numeric_leaf(value):
                seen.add(path)
            continue
        scaled_paths.add((path, index))
        unit_value = _at(unit.diagnostics, path, list(index))
        if not _is_numeric_leaf(value):
            assert _same(value, unit_value) or value == unit_value, path
            continue
        seen.add(path)
        expected = _expected(path, unit_value)
        assert _same(value, expected), (path, value, expected)
    unit_paths = {(p, i) for p, i, _ in _indexed_leaves(unit.diagnostics)}
    assert scaled_paths == unit_paths
    if "coverage_at_01" in scaled.diagnostics.get("calibration", {}):
        assert "coverage_tolerance" in scaled.diagnostics["calibration"]
    return seen


def _pair(labels_unit: np.ndarray, judge: np.ndarray, **kwargs: Any) -> Tuple[Any, Any]:
    """The scaled call on caller-unit labels and the unit call it must match."""
    caller = SCALE.inverse_array(labels_unit)
    internal = SCALE.normalize_array(caller)  # the exact internal normalisation
    scaled = _call(judge, caller, oracle_scale=(LO, HI), **kwargs)
    unit = _call(judge, internal, **kwargs)
    return scaled, unit


@pytest.mark.parametrize("coverage", ["partial", "complete"])
@pytest.mark.parametrize(
    "inference,extra",
    [("cluster_robust", {}), ("bootstrap", {"n_bootstrap": 30}), ("auto", {})],
)
@pytest.mark.parametrize("weight", ["one", "tuned"])
@pytest.mark.parametrize("with_covariates", [False, True])
def test_equivariance(
    coverage: str,
    inference: str,
    extra: Dict[str, Any],
    weight: str,
    with_covariates: bool,
) -> None:
    judge, unit_labels, clusters, covariates = _data()
    labels = unit_labels if coverage == "complete" else _partial(unit_labels, clusters)
    kwargs: Dict[str, Any] = dict(
        cluster_ids=clusters,
        covariates=covariates if with_covariates else None,
        inference=inference,
        correction_weight=weight,
        seed=4,
        **extra,
    )
    if inference == "auto":
        kwargs["n_bootstrap"] = 30  # auto resolves to bootstrap at partial coverage
    if coverage == "complete":
        kwargs["fit_calibrator"] = True
    scaled, unit = _pair(labels, judge, **kwargs)
    _assert_scaled_matches_unit(scaled, unit)
    assert isinstance(scaled.calibrator, ScaledCalibrator)
    assert isinstance(unit.calibrator, JudgeCalibrator)


def _classification_scenarios() -> Iterator[Tuple[str, np.ndarray, np.ndarray, Dict]]:
    judge, unit_labels, clusters, covariates = _data()
    partial = _partial(unit_labels, clusters)
    for inference, extra in (
        ("cluster_robust", {}),
        ("bootstrap", {"n_bootstrap": 20}),
    ):
        for weight in ("one", "tuned"):
            for cov in (None, covariates):
                base = dict(
                    cluster_ids=clusters,
                    covariates=cov,
                    inference=inference,
                    correction_weight=weight,
                    **extra,
                )
                yield "partial", partial, judge, base
                yield "complete", unit_labels, judge, dict(base)
                yield "complete-fit", unit_labels, judge, {
                    **base,
                    "fit_calibrator": True,
                }
    # Labels only on mid-range judge scores: the boundary card reports a
    # partial-identification width.
    mid = np.where((judge > 0.25) & (judge < 0.75), unit_labels, np.nan)
    yield "out-of-range", mid, judge, dict(cluster_ids=clusters)
    # One independent cluster: inference unavailable on both paths.
    two = np.asarray([0.0, 1.0])
    for inference in ("cluster_robust", "bootstrap"):
        yield "one-cluster", two, np.asarray([0.2, 0.8]), dict(
            cluster_ids=["a", "a"], inference=inference, n_bootstrap=5
        )


def test_every_numeric_diagnostic_is_classified() -> None:
    """Walk every numeric leaf on every path; the registry has no dead entries."""
    seen: Set[Tuple[str, ...]] = set()
    for _, labels, judge, kwargs in _classification_scenarios():
        scaled, unit = _pair(labels, judge, **kwargs)
        seen |= _assert_scaled_matches_unit(scaled, unit)
    registries = [
        _ORACLE_SCALE_LEVEL_PATHS,
        _ORACLE_SCALE_SPREAD_PATHS,
        _ORACLE_SCALE_VARIANCE_PATHS,
        _ORACLE_SCALE_UNCHANGED_PATHS,
    ]
    for i, first in enumerate(registries):
        for second in registries[i + 1 :]:
            assert not first & second
    assert seen == set().union(*registries)


def test_unclassified_numeric_leaf_raises() -> None:
    with pytest.raises(RuntimeError, match="no oracle_scale unit"):
        _diagnostics_to_oracle_units({"calibration": {"new_metric": 0.5}}, SCALE)
    # Non-numeric leaves pass through unchanged.
    mapped = _diagnostics_to_oracle_units(
        {"calibration": {"mode": "auto", "used": True, "missing": None}}, SCALE
    )
    assert mapped == {"calibration": {"mode": "auto", "used": True, "missing": None}}


def test_labels_at_both_bounds_map_exactly() -> None:
    judge, _, clusters, _ = _data()
    caller = np.where(np.arange(len(judge)) % 2 == 0, LO, HI)
    result = calibrated_mean_ci(
        judge, caller, cluster_ids=clusters, oracle_scale=(LO, HI)
    )
    assert result.estimate == pytest.approx(float(np.mean(caller)), rel=1e-12)


def test_inf_label_raises_not_clipped() -> None:
    judge, unit_labels, clusters, _ = _data()
    caller = SCALE.inverse_array(unit_labels)
    caller[3] = np.inf
    with pytest.raises(ValueError, match="finite"):
        calibrated_mean_ci(judge, caller, cluster_ids=clusters, oracle_scale=(LO, HI))
    partial = _partial(caller, clusters)
    partial[np.flatnonzero(np.isfinite(partial))[0]] = -np.inf
    with pytest.raises(ValueError, match="finite"):
        calibrated_mean_ci(judge, partial, cluster_ids=clusters, oracle_scale=(LO, HI))


@pytest.mark.parametrize("bad", [HI + 0.5, LO - 0.5, HI + 1e-12])
def test_values_outside_declared_scale_raise(bad: float) -> None:
    judge, unit_labels, clusters, _ = _data()
    caller = SCALE.inverse_array(unit_labels)
    caller[7] = bad
    with pytest.raises(
        ValueError, match=r"outside declared scale \[-2.0, 23.0\]"
    ) as info:
        calibrated_mean_ci(judge, caller, cluster_ids=clusters, oracle_scale=(LO, HI))
    assert "observed range" in str(info.value)
    assert "oracle_labels" in str(info.value)


@pytest.mark.parametrize(
    "declaration",
    [(1.0, 1.0), (2.0, 1.0), (0.0, np.inf), (np.nan, 1.0), (0.0, 1.0, 2.0), "ab", 5],
)
def test_invalid_declarations(declaration: Any) -> None:
    judge, unit_labels, clusters, _ = _data()
    with pytest.raises(ValueError, match="oracle_scale"):
        calibrated_mean_ci(
            judge, unit_labels, cluster_ids=clusters, oracle_scale=declaration
        )


def test_unmasked_garbage_ignored_with_oracle_mask() -> None:
    judge, unit_labels, clusters, _ = _data()
    caller = _partial(SCALE.inverse_array(unit_labels), clusters)
    mask = np.isfinite(caller)
    garbage = caller.copy()
    garbage[~mask] = np.where(np.arange(int(np.sum(~mask))) % 2 == 0, 1e6, np.inf)
    reference = calibrated_mean_ci(
        judge, caller, cluster_ids=clusters, oracle_scale=(LO, HI)
    )
    masked = calibrated_mean_ci(
        judge,
        garbage,
        oracle_mask=mask,
        cluster_ids=clusters,
        oracle_scale=(LO, HI),
    )
    assert (masked.estimate, masked.se, masked.ci) == (
        reference.estimate,
        reference.se,
        reference.ci,
    )


@pytest.mark.parametrize("coverage", ["partial", "complete"])
def test_calibrator_predicts_caller_units(coverage: str) -> None:
    judge, unit_labels, clusters, covariates = _data()
    caller = SCALE.inverse_array(unit_labels)
    if coverage == "partial":
        caller = _partial(caller, clusters)
    result = calibrated_mean_ci(
        judge,
        caller,
        cluster_ids=clusters,
        covariates=covariates,
        oracle_scale=(LO, HI),
        fit_calibrator=True,
    )
    calibrator = result.calibrator
    assert isinstance(calibrator, ScaledCalibrator)
    raw = calibrator.raw_calibrator
    assert isinstance(raw, JudgeCalibrator)
    assert calibrator.judge_scale is None

    grid = np.linspace(-0.5, 1.5, 21)  # judge scores are never rescaled
    grid_cov = np.tile(covariates[:1], (len(grid), 1))
    predictions = calibrator.predict(grid, covariates=grid_cov)
    np.testing.assert_array_equal(
        predictions, SCALE.inverse_array(raw.predict(grid, covariates=grid_cov))
    )
    assert np.all((predictions >= LO) & (predictions <= HI))

    folds = np.arange(len(judge)) % raw.n_folds
    np.testing.assert_array_equal(
        calibrator.predict_oof(judge, folds, covariates=covariates),
        SCALE.inverse_array(raw.predict_oof(judge, folds, covariates=covariates)),
    )
    assert calibrator.oracle_s_range == raw.oracle_s_range
    assert raw.oracle_reward_range is not None
    low, high = raw.oracle_reward_range
    assert calibrator.oracle_reward_range == (
        float(SCALE.inverse(low)),
        float(SCALE.inverse(high)),
    )
    info = calibrator.get_calibration_info()
    assert info["rmse"] == pytest.approx(raw.get_calibration_info()["rmse"] * SPAN)
    assert info["rmse"] == pytest.approx(result.diagnostics["calibration"]["rmse"])
    assert info["judge_input_scale"] is None
    assert info["oracle_output_scale"] == SCALE.to_dict()
    assert info["coverage_tolerance"] == pytest.approx(0.1 * SPAN)


def test_none_scale_is_the_default_call() -> None:
    judge, unit_labels, clusters, covariates = _data()
    for labels in (unit_labels, _partial(unit_labels, clusters)):
        omitted = _call(judge, labels, cluster_ids=clusters, covariates=covariates)
        explicit = _call(
            judge,
            labels,
            cluster_ids=clusters,
            covariates=covariates,
            oracle_scale=None,
        )
        assert (omitted.estimate, omitted.se, omitted.ci) == (
            explicit.estimate,
            explicit.se,
            explicit.ci,
        )
        assert "oracle_scale" not in explicit.diagnostics
        assert explicit.calibrator is None or isinstance(
            explicit.calibrator, JudgeCalibrator
        )


def test_identity_scale_matches_unscaled() -> None:
    judge, unit_labels, clusters, _ = _data()
    labels = _partial(unit_labels, clusters)
    plain = calibrated_mean_ci(judge, labels, cluster_ids=clusters)
    identity = calibrated_mean_ci(
        judge, labels, cluster_ids=clusters, oracle_scale=(0, 1)
    )
    assert (identity.estimate, identity.se, identity.ci) == (
        plain.estimate,
        plain.se,
        plain.ci,
    )
    assert identity.diagnostics["oracle_scale"]["is_identity"] is True


def test_transport_audit_in_caller_units() -> None:
    judge, unit_labels, clusters, _ = _data(n_prompts=200, draws=1)
    labels = _partial(unit_labels, clusters)
    scaled, unit = _pair(labels, judge, cluster_ids=clusters)
    rng = np.random.default_rng(9)
    probe = rng.uniform(size=300)
    probe_unit = np.clip(0.15 + 0.6 * probe + rng.normal(0, 0.1, size=300), 0, 1)
    probe_caller = SCALE.inverse_array(probe_unit)
    for shift in (0.0, 0.12):
        unit_audit = transport_audit(
            probe, probe_unit + shift, unit.calibrator, delta_max=0.04
        )
        scaled_audit = transport_audit(
            probe,
            probe_caller + shift * SPAN,
            scaled.calibrator,
            delta_max=0.04 * SPAN,
        )
        assert scaled_audit.status == unit_audit.status
        assert scaled_audit.delta_hat == pytest.approx(
            unit_audit.delta_hat * SPAN, rel=1e-9, abs=1e-12
        )
        for scaled_end, unit_end in zip(scaled_audit.delta_ci, unit_audit.delta_ci):
            assert scaled_end == pytest.approx(unit_end * SPAN, rel=1e-9, abs=1e-12)


class TestScaledCalibratorFacade:
    """`ScaledCalibrator(judge_scale=None)` and its `predict_oof`."""

    @staticmethod
    def _fitted() -> Tuple[JudgeCalibrator, np.ndarray, np.ndarray]:
        judge, unit_labels, clusters, _ = _data()
        raw = JudgeCalibrator(random_seed=1)
        fit = raw.fit_cv(
            judge,
            unit_labels,
            np.ones(len(judge), dtype=bool),
            prompt_ids=[str(c) for c in clusters],
            quiet=True,
        )
        assert fit.fold_ids is not None
        return raw, judge, fit.fold_ids

    def test_judge_scale_none_passes_scores_through(self) -> None:
        raw, _, _ = self._fitted()
        facade = ScaledCalibrator(raw, judge_scale=None, output_scale=SCALE)
        scores = np.asarray([-3.0, 0.5, 7.0])  # no judge range to validate
        np.testing.assert_array_equal(
            facade.predict(scores), SCALE.inverse_array(raw.predict(scores))
        )
        assert facade.oracle_s_range == raw.oracle_s_range

    def test_predict_oof_maps_judge_and_output_scales(self) -> None:
        raw, judge, folds = self._fitted()
        judge_scale = ScaleInfo(0.0, 10.0)
        facade = ScaledCalibrator(raw, judge_scale=judge_scale, output_scale=SCALE)
        public = judge_scale.inverse_array(judge)
        expected = SCALE.inverse_array(
            raw.predict_oof(judge_scale.normalize_array(public), folds)
        )
        np.testing.assert_array_equal(facade.predict_oof(public, folds), expected)
        with pytest.raises(ValueError, match="judge_scores"):
            facade.predict_oof(public + 20.0, folds)

    def test_default_facade_unchanged(self) -> None:
        raw, judge, _ = self._fitted()
        judge_scale = ScaleInfo(0.0, 10.0)
        facade = ScaledCalibrator(raw, judge_scale=judge_scale, output_scale=SCALE)
        public = judge_scale.inverse_array(judge)
        np.testing.assert_array_equal(
            facade.predict(public),
            SCALE.inverse_array(raw.predict(judge_scale.normalize_array(public))),
        )
        assert facade.get_calibration_info()["judge_input_scale"] == (
            judge_scale.to_dict()
        )
