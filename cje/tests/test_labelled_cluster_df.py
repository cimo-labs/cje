"""Labelled-cluster degrees of freedom for the augmented interval (issue #60).

Under representative labels the residual correction of the augmented
estimator is a mean over a policy's labelled prompt clusters that fits ``q``
parameters on them (the mean residual, plus the slope for the tuned weight).
The analytic interval therefore splits the CRV1 variance of the pseudo-outcome
into labelled and unlabelled clusters, scales the labelled part by
``n_L / (n_L - q)``, adds the oracle-jackknife variance, and takes
``min(n_L - q, Welch(n_L - q, K - 1), Welch(G - 1, K - 1))`` degrees of
freedom; the last term is the unadjusted interval's df, so the interval never
narrows. Pairs inherit the rule. Other routes, other label designs, the
bootstrap and every point estimate are unchanged.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy import stats

from cje import analyze_dataset, calibrated_mean_ci
from cje.calibration.judge import JudgeCalibrator
from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample
from cje.data.models import EstimationResult, InferenceUnavailableError
from cje.diagnostics.robust_inference import (
    CORRECTION_WEIGHT_REASONS,
    CalibrationProvenance,
    LabelDesign,
    cluster_robust_se,
    correction_fitted_parameters,
    labelled_cluster_crv1,
    labelled_cluster_df,
    labelled_cluster_variance,
    oracle_jackknife_variance,
)
from cje.estimators.direct_method import CalibratedDirectEstimator

Rows = List[Tuple[str, int, float, Optional[float]]]


# ---------------------------------------------------------------------------
# Helpers written out independently of the library
# ---------------------------------------------------------------------------


def _welch(var_sampling: float, df_sampling: float, var_oracle: float, k: int) -> float:
    """Welch--Satterthwaite df of a sampling and an oracle component."""
    if var_oracle == 0.0:
        return float(df_sampling)
    total = var_sampling + var_oracle
    return 1.0 / (
        (var_sampling / total) ** 2 / df_sampling + (var_oracle / total) ** 2 / (k - 1)
    )


def _expected_df(
    var_sampling: float,
    df_labelled: float,
    var_unadjusted: float,
    df_unadjusted: float,
    var_oracle: float,
    k: int,
) -> float:
    """min(n_L - q, Welch(n_L - q, K - 1), Welch(G - 1, K - 1))."""
    return min(
        float(df_labelled),
        _welch(var_sampling, df_labelled, var_oracle, k),
        _welch(var_unadjusted, df_unadjusted, var_oracle, k),
    )


def _half_width(variance: float, df: float) -> float:
    return float(stats.t.ppf(0.975, df) * np.sqrt(variance))


def _influence(result: EstimationResult, policy: str) -> np.ndarray:
    assert result.influence_functions is not None
    return np.asarray(result.influence_functions[policy], dtype=float)


def _split_by_hand(
    values: ArrayLike, prompts: Sequence[str], labelled: ArrayLike
) -> Tuple[float, float, int]:
    """CRV1 variance of a mean, split by labelled cluster, one cluster at a time."""
    values = np.asarray(values, dtype=float)
    labelled = np.asarray(labelled, dtype=bool)
    centred = values - float(np.mean(values))
    clusters = sorted(set(prompts))
    totals = np.array(
        [sum(c for c, p in zip(centred, prompts) if p == g) for g in clusters]
    )
    totals = totals - totals.mean()
    is_labelled = np.array(
        [any(flag for flag, p in zip(labelled, prompts) if p == g) for g in clusters]
    )
    scale = len(clusters) / (len(clusters) - 1) / len(centred) ** 2
    return (
        float(scale * np.sum(totals[is_labelled] ** 2)),
        float(scale * np.sum(totals[~is_labelled] ** 2)),
        int(is_labelled.sum()),
    )


def _pair_split_by_hand(
    result: EstimationResult,
    prompts: Dict[str, Sequence[str]],
    labelled_prompts: Iterable[str],
) -> Tuple[float, float]:
    """Paired CRV1 variance of a difference, split by labelled prompt."""
    first, second = result.metadata["target_policies"]
    contributions: List[Dict[str, float]] = []
    for policy in (first, second):
        values = _influence(result, policy)
        sums: Dict[str, float] = {}
        for value, prompt in zip(values, prompts[policy]):
            sums[prompt] = sums.get(prompt, 0.0) + float(value) / len(values)
        contributions.append(sums)
    union = sorted(set(contributions[0]) | set(contributions[1]))
    differences = np.array(
        [contributions[0].get(p, 0.0) - contributions[1].get(p, 0.0) for p in union]
    )
    differences = differences - differences.mean()
    labelled = set(labelled_prompts)
    is_labelled = np.array([p in labelled for p in union])
    scale = len(union) / (len(union) - 1)
    return (
        float(scale * np.sum(differences[is_labelled] ** 2)),
        float(scale * np.sum(differences[~is_labelled] ** 2)),
    )


def _external_calibrator(seed: int, n: int = 300) -> JudgeCalibrator:
    """A calibrator fitted on its own sample, independent of every evaluation."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.15, 0.85, size=n)
    scores = np.clip(truth + rng.normal(0, 0.2, size=n), 0, 1)
    labels = np.clip(truth + rng.normal(0, 0.15, size=n), 0, 1)
    calibrator = JudgeCalibrator(random_seed=42, calibration_mode="monotone")
    calibrator.fit_cv(
        scores, labels, n_folds=5, prompt_ids=[f"c{i}" for i in range(n)], quiet=True
    )
    return calibrator


def _population(
    seed: int, n_prompts: int, draws: int, policies: Dict[str, Dict[str, Any]]
) -> Dict[str, Rows]:
    """Shared prompt effects; per policy a shift, labelled prompts (every draw
    labelled) and partially labelled prompts (draw 0 only)."""
    rng = np.random.default_rng(seed)
    effects = rng.uniform(0.15, 0.85, size=n_prompts)
    data: Dict[str, Rows] = {}
    for name, spec in policies.items():
        labelled = set(spec.get("labelled", ()))
        partial = set(spec.get("partial", ()))
        rows: Rows = []
        for prompt in range(n_prompts):
            for draw in range(draws):
                latent = effects[prompt] + spec.get("shift", 0.0) + rng.normal(0, 0.1)
                score = float(np.clip(latent + rng.normal(0, 0.2), 0, 1))
                label = float(np.clip(latent + rng.normal(0, 0.15), 0, 1))
                observed = prompt in labelled or (prompt in partial and draw == 0)
                rows.append(
                    (f"q{prompt:03d}", draw, score, label if observed else None)
                )
        data[name] = rows
    return data


def _estimator(
    data: Dict[str, Rows],
    calibrator: JudgeCalibrator,
    provenance: Optional[CalibrationProvenance] = None,
    **kwargs: Any,
) -> CalibratedDirectEstimator:
    estimator = CalibratedDirectEstimator(
        target_policies=list(data),
        reward_calibrator=calibrator,
        calibration_provenance=(
            provenance or CalibrationProvenance.from_fitted_calibrator(calibrator)
        ),
        **kwargs,
    )
    for name, rows in data.items():
        estimator.add_fresh_draws(
            name,
            FreshDrawDataset(
                target_policy=name,
                samples=[
                    FreshDrawSample(
                        prompt_id=prompt,
                        target_policy=name,
                        judge_score=score,
                        oracle_label=label,
                        response=None,
                        draw_idx=draw,
                    )
                    for prompt, draw, score, label in rows
                ],
            ),
        )
    return estimator


def _estimate(
    data: Dict[str, Rows], calibrator: JudgeCalibrator, **kwargs: Any
) -> EstimationResult:
    return _estimator(data, calibrator, **kwargs).fit_and_estimate()


def _prompts(rows: Rows) -> List[str]:
    return [row[0] for row in rows]


def _labelled(rows: Rows) -> List[bool]:
    return [row[3] is not None for row in rows]


def _readme_draws(label_scale: float = 1.0) -> Dict[str, List[Dict[str, Any]]]:
    """The README quickstart: gpt-5.6 has 10 labelled prompts, fable-5 none."""
    scores = {
        "gpt-5.6": [0.62, 0.68, 0.72, 0.76, 0.79, 0.83, 0.85, 0.88, 0.91, 0.95]
        + [0.64, 0.69, 0.73, 0.77, 0.80, 0.84, 0.87, 0.89, 0.92, 0.94],
        "fable-5": [0.70, 0.74, 0.75, 0.78, 0.81, 0.83, 0.86, 0.90, 0.93, 0.94]
        + [0.72, 0.76, 0.79, 0.80, 0.84, 0.85, 0.88, 0.89, 0.91, 0.95],
    }
    labels: List[Optional[float]] = [
        0.55,
        0.60,
        0.70,
        0.74,
        0.75,
        0.80,
        0.90,
        0.92,
        0.88,
        0.97,
    ] + [None] * 10
    return {
        "gpt-5.6": [
            {"prompt_id": f"q{i:02d}", "judge_score": s}
            | ({"oracle_label": y * label_scale} if y is not None else {})
            for i, (s, y) in enumerate(zip(scores["gpt-5.6"], labels))
        ],
        "fable-5": [
            {"prompt_id": f"q{i:02d}", "judge_score": s}
            for i, s in enumerate(scores["fable-5"])
        ],
    }


README_LABELLED = [f"q{i:02d}" for i in range(10)]
README_PROMPTS = [f"q{i:02d}" for i in range(20)]


# ---------------------------------------------------------------------------
# Library helpers
# ---------------------------------------------------------------------------


def test_split_sums_to_the_crv1_variance_with_multi_row_clusters() -> None:
    rng = np.random.default_rng(1)
    prompts = np.repeat(np.arange(40), 3)
    values = rng.normal(size=120)
    labelled = np.zeros(120, dtype=bool)
    labelled[:15] = True  # five fully labelled prompts
    labelled[[18, 22, 27]] = True  # one draw of three more prompts
    split = labelled_cluster_crv1(values, prompts, labelled)
    crv1 = cluster_robust_se(values, prompts, np.mean, lambda x: x - np.mean(x))
    assert split["v_labelled"] + split["v_unlabelled"] == pytest.approx(
        crv1["se"] ** 2, rel=1e-12
    )
    assert split["n_labelled_clusters"] == 8
    assert split["n_clusters"] == 40
    v_lab, v_unl, n_labelled = _split_by_hand(
        values, [str(p) for p in prompts], labelled
    )
    assert split["v_labelled"] == pytest.approx(v_lab, rel=1e-12)
    assert split["v_unlabelled"] == pytest.approx(v_unl, rel=1e-12)
    assert n_labelled == 8
    # String cluster labels give the same split.
    by_name = labelled_cluster_crv1(
        values, np.asarray([f"p{p}" for p in prompts]), labelled
    )
    assert by_name["v_labelled"] == pytest.approx(split["v_labelled"], rel=1e-12)

    with pytest.raises(ValueError, match="at least two"):
        labelled_cluster_crv1(values[:3], np.zeros(3), labelled[:3])
    with pytest.raises(ValueError, match="aligned"):
        labelled_cluster_crv1(values, prompts[:-1], labelled)
    with pytest.raises(ValueError, match="aligned"):
        labelled_cluster_crv1(values, prompts, labelled[:-1])


def test_variance_inflates_the_labelled_part_and_sets_the_df() -> None:
    variance, df = labelled_cluster_variance(0.004, 0.001, 10, 1)
    assert df == 9
    assert variance == pytest.approx(0.004 * 10 / 9 + 0.001)
    variance, df = labelled_cluster_variance(0.004, 0.001, 25, 2)
    assert df == 23
    assert variance == pytest.approx(0.004 * 25 / 23 + 0.001)
    variance, df = labelled_cluster_variance(0.004, 0.001, 10, 1, inflate=False)
    assert (variance, df) == (pytest.approx(0.005), 9)
    for n_labelled, fitted in [(1, 1), (2, 2), (0, 1)]:
        variance, df = labelled_cluster_variance(0.004, 0.001, n_labelled, fitted)
        assert np.isnan(variance) and df is None
    variance, df = labelled_cluster_variance(0.004, 0.001, 2, 1)
    assert df == 1 and variance == pytest.approx(0.004 * 2 + 0.001)


def test_fitted_parameters_follow_the_realised_reason() -> None:
    expected = {
        "rule_one": 1,
        "tuned": 2,
        "too_few_labelled_clusters": 1,
        "rare_outcome": 1,
        "constant_predictions": 1,
        "known_propensity_fixed_one": 1,
    }
    assert set(expected) == set(CORRECTION_WEIGHT_REASONS)
    for reason, q in expected.items():
        assert correction_fitted_parameters(reason) == q
    assert correction_fitted_parameters(None) == 1


def test_oracle_cap_never_raises_the_labelled_df() -> None:
    unadjusted = {"se_unadjusted": 0.0095, "df_unadjusted": 199}
    assert labelled_cluster_df(0.03, 9, 0.0, 0, **unadjusted) == (9.0, False)
    # Comparable sampling and oracle variances: the labelled Welch df binds.
    df, capped = labelled_cluster_df(0.01, 29, 0.0001, 5, **unadjusted)
    labelled_welch = _welch(0.0001, 29, 0.0001, 5)
    assert labelled_welch < _welch(0.0095**2, 199, 0.0001, 5)
    assert capped and df == pytest.approx(labelled_welch, rel=1e-12)
    # A small oracle share can give a Welch df above 9; the cap keeps 9.
    df, capped = labelled_cluster_df(0.03, 9, 1e-6, 5, **unadjusted)
    assert (df, capped) == (9.0, False)


def test_a_dominant_oracle_term_cannot_narrow_the_interval() -> None:
    # G = 200, n_L = 5, q = 1, 90% of the CRV1 variance (1.0) in labelled
    # clusters, oracle variance 5 over K = 2 folds. Inflating the sampling
    # variance raises the labelled Welch df (1.527) above the unadjusted one
    # (1.440); without the second cap the interval was 0.937x the unadjusted.
    var_sampling, df_labelled = labelled_cluster_variance(0.9, 0.1, 5, 1)
    assert (var_sampling, df_labelled) == (pytest.approx(1.225), 4)
    labelled_welch = _welch(var_sampling, 4, 5.0, 2)
    unadjusted_welch = _welch(1.0, 199, 5.0, 2)
    assert labelled_welch == pytest.approx(1.5271, abs=1e-4)
    assert unadjusted_welch == pytest.approx(1.4397, abs=1e-4)
    assert _half_width(var_sampling + 5.0, labelled_welch) < 0.94 * _half_width(
        6.0, unadjusted_welch
    )
    df, capped = labelled_cluster_df(
        np.sqrt(var_sampling), 4, 5.0, 2, se_unadjusted=1.0, df_unadjusted=199
    )
    assert capped and df == pytest.approx(unadjusted_welch, rel=1e-12)
    assert _half_width(var_sampling + 5.0, df) > _half_width(6.0, unadjusted_welch)


def test_the_interval_never_narrows_across_a_parameter_grid() -> None:
    checked = 0
    for k in (2, 3, 5, 10):
        for n_labelled in (2, 3, 5, 10, 20, 60):
            for q in (1, 2):
                if n_labelled - q < 1:
                    continue
                for share in (0.1, 0.5, 0.9, 1.0):
                    for oracle_ratio in (0.0, 0.01, 0.3, 1.0, 5.0, 50.0):
                        var_sampling, df_labelled = labelled_cluster_variance(
                            share, 1.0 - share, n_labelled, q
                        )
                        assert df_labelled is not None
                        var_oracle = oracle_ratio
                        df, capped = labelled_cluster_df(
                            np.sqrt(var_sampling),
                            df_labelled,
                            var_oracle,
                            k,
                            se_unadjusted=1.0,
                            df_unadjusted=199,
                        )
                        assert df <= df_labelled
                        assert capped is (df < df_labelled)
                        unadjusted_df = _welch(1.0, 199, var_oracle, k)
                        assert df <= unadjusted_df * (1 + 1e-12)
                        assert _half_width(
                            var_sampling + var_oracle, df
                        ) >= _half_width(1.0 + var_oracle, unadjusted_df) * (1 - 1e-12)
                        checked += 1
    assert checked == 4 * 11 * 4 * 6


# ---------------------------------------------------------------------------
# Per-policy interval
# ---------------------------------------------------------------------------


def test_estimator_matches_the_hand_built_interval_on_clustered_draws() -> None:
    data = _population(
        101,
        n_prompts=150,
        draws=3,
        policies={"policy": {"labelled": range(12), "partial": range(12, 16)}},
    )
    result = _estimate(data, _external_calibrator(102))
    meta = result.metadata
    point = meta["point_estimator"]
    assert point["routes"] == ["augmented"]
    rows = data["policy"]
    v_lab, v_unl, n_labelled = _split_by_hand(
        _influence(result, "policy"), _prompts(rows), _labelled(rows)
    )
    assert n_labelled == 16 == point["labelled_clusters"][0]
    components = meta["se_components"]
    v_oracle = components["oracle_variance_per_policy"]["policy"]
    folds = components["oracle_jackknife_counts"]["policy"]
    assert folds == 5 and v_oracle > 0

    var_sampling = v_lab * 16 / 15 + v_unl
    assert result.diagnostics is not None
    assert result.diagnostics.standard_errors["policy"] == pytest.approx(
        np.sqrt(var_sampling), rel=1e-12
    )
    assert result.standard_errors[0] == pytest.approx(
        np.sqrt(var_sampling + v_oracle), rel=1e-12
    )
    expected_df = _expected_df(var_sampling, 15, v_lab + v_unl, 149, v_oracle, folds)
    info = meta["degrees_of_freedom"]["policy"]
    assert info["df"] == pytest.approx(expected_df, rel=1e-12)
    assert info["df_method"] == "labelled_clusters"
    assert info["t_critical"] == pytest.approx(stats.t.ppf(0.975, expected_df))
    assert info["se_method"] == "cluster_robust"
    assert info["n_clusters"] == 150
    assert info["n_labelled_clusters"] == 16
    assert info["fitted_parameters"] == 1
    assert info["labelled_variance_inflation"] == pytest.approx(16 / 15)
    assert info["labelled_variance_share"] == pytest.approx(
        v_lab / (v_lab + v_unl), rel=1e-10
    )
    assert info["labels_coupled"] is False
    assert info["labels_coupled_fraction"] == 0.0
    assert info["oracle_df_cap_applied"] is bool(expected_df < 15)
    assert result.ci_info is not None
    assert result.ci_info.df_per_policy == meta["degrees_of_freedom"]

    lower, upper = result.confidence_interval()
    half_width = stats.t.ppf(0.975, expected_df) * result.standard_errors[0]
    assert lower[0] == pytest.approx(result.estimates[0] - half_width, rel=1e-12)
    assert upper[0] == pytest.approx(result.estimates[0] + half_width, rel=1e-12)
    # The influence functions keep the plain CRV1 SE.
    codes = np.unique(_prompts(rows), return_inverse=True)[1]
    crv1 = cluster_robust_se(_influence(result, "policy"), codes, np.mean, lambda x: x)
    assert crv1["se"] ** 2 == pytest.approx(v_lab + v_unl, rel=1e-12)


def test_tuned_weight_counts_the_slope_and_a_guard_fallback_is_weight_one() -> None:
    calibrator = _external_calibrator(112)
    data = _population(111, 300, 1, {"policy": {"labelled": range(25)}})
    tuned = _estimate(data, calibrator, correction_weight="tuned")
    assert tuned.metadata["point_estimator"]["correction_weight_reasons"] == ["tuned"]
    info = tuned.metadata["degrees_of_freedom"]["policy"]
    assert info["fitted_parameters"] == 2
    assert info["df"] == 23.0
    assert info["oracle_df_cap_applied"] is False
    assert info["labelled_variance_inflation"] == pytest.approx(25 / 23)

    # Below 20 labelled prompts the tuned rule is weight one, and so is its
    # interval, exactly.
    data = _population(113, 300, 1, {"policy": {"labelled": range(15)}})
    tuned = _estimate(data, calibrator, correction_weight="tuned")
    one = _estimate(data, calibrator, correction_weight="one")
    assert tuned.metadata["point_estimator"]["correction_weight_reasons"] == [
        "too_few_labelled_clusters"
    ]
    assert tuned.estimates[0] == one.estimates[0]
    assert tuned.standard_errors[0] == one.standard_errors[0]
    assert tuned.metadata["degrees_of_freedom"] == one.metadata["degrees_of_freedom"]
    assert one.metadata["degrees_of_freedom"]["policy"]["fitted_parameters"] == 1


def _assert_cluster_interval(
    result: EstimationResult, policy: str, prompts: Sequence[str]
) -> None:
    """The unchanged interval: CRV1 over all clusters, Welch(G - 1, K - 1)."""
    index = result.metadata["target_policies"].index(policy)
    codes = np.unique(prompts, return_inverse=True)[1]
    crv1 = cluster_robust_se(_influence(result, policy), codes, np.mean, lambda x: x)
    components = result.metadata["se_components"]
    v_oracle = components["oracle_variance_per_policy"][policy]
    folds = components["oracle_jackknife_counts"][policy]
    assert result.standard_errors[index] == pytest.approx(
        np.sqrt(crv1["se"] ** 2 + v_oracle), rel=1e-12
    )
    info = result.metadata["degrees_of_freedom"][policy]
    assert info["df"] == pytest.approx(
        _welch(crv1["se"] ** 2, crv1["df"], v_oracle, folds), rel=1e-12
    )
    assert info["df_method"] == ("welch_satterthwaite" if v_oracle > 0 else "cluster")
    assert "n_labelled_clusters" not in info


def test_other_routes_keep_the_cluster_interval() -> None:
    data = _population(
        121,
        120,
        1,
        {
            "full": {"labelled": range(120)},
            "none": {},
            "partial": {"labelled": range(30)},
        },
    )
    result = _estimate(data, _external_calibrator(122))
    assert result.metadata["point_estimator"]["routes"] == [
        "direct_oracle",
        "plug_in",
        "augmented",
    ]
    for policy in ("full", "none"):
        _assert_cluster_interval(result, policy, _prompts(data[policy]))
    labelled = result.metadata["degrees_of_freedom"]["partial"]
    assert labelled["df_method"] == "labelled_clusters"
    assert labelled["n_labelled_clusters"] == 30
    entry = result.metadata["pairwise_inference"]["0-1"]  # direct_oracle vs plug-in
    assert entry["df_method"] in ("welch_satterthwaite", "cluster")
    assert "n_labelled_clusters" not in entry
    # Against a direct-oracle policy only the augmented policy's labels count.
    entry = result.metadata["pairwise_inference"]["0-2"]
    assert entry["df_method"] == "labelled_clusters"
    assert entry["n_labelled_clusters"] == 30
    assert entry["labelled_variance_inflation"] == pytest.approx(30 / 29)


def test_every_prompt_labelled_and_no_jackknife() -> None:
    # Draw 0 of every prompt labelled, draws 1-2 not: every cluster is
    # labelled, so nothing is left unscaled and the df is G - 1.
    data = _population(125, 40, 3, {"policy": {"partial": range(40)}})
    result = _estimate(data, _external_calibrator(126), oua_jackknife=False)
    assert result.metadata["point_estimator"]["routes"] == ["augmented"]
    info = result.metadata["degrees_of_freedom"]["policy"]
    assert info["n_labelled_clusters"] == 40
    assert info["labelled_variance_share"] == pytest.approx(1.0)
    # Without the oracle jackknife the df is the labelled df, uncapped.
    assert info["df"] == 39.0 and info["oracle_df_cap_applied"] is False
    assert info["oracle_jackknife_status"] == "not_requested"
    rows = data["policy"]
    v_lab, v_unl, _ = _split_by_hand(
        _influence(result, "policy"), _prompts(rows), _labelled(rows)
    )
    assert v_unl == pytest.approx(0.0, abs=1e-18)
    assert result.standard_errors[0] == pytest.approx(
        np.sqrt(v_lab * 40 / 39), rel=1e-12
    )


@pytest.mark.parametrize("kind", ["known_propensity", "targeted_unknown"])
def test_other_label_designs_keep_the_cluster_interval(kind: str) -> None:
    data = _population(131, 200, 1, {"policy": {"labelled": range(20)}})
    design = (
        LabelDesign(kind="known_propensity", propensities={"policy": np.full(200, 0.1)})
        if kind == "known_propensity"
        else LabelDesign(kind="targeted_unknown")
    )
    result = _estimate(data, _external_calibrator(132), label_design=design)
    expected_route = (
        "augmented" if kind == "known_propensity" else "plug_in_targeted_unknown"
    )
    assert result.metadata["point_estimator"]["routes"] == [expected_route]
    _assert_cluster_interval(result, "policy", _prompts(data["policy"]))


def test_repeated_estimates_reset_the_labelled_state() -> None:
    data = _population(141, 100, 1, {"a": {"labelled": range(12)}, "b": {}})
    estimator = _estimator(data, _external_calibrator(142))
    first = estimator.fit_and_estimate()
    second = estimator.estimate()
    assert set(estimator._labelled_inference) == {"a"}
    assert np.array_equal(first.standard_errors, second.standard_errors)
    assert first.metadata["degrees_of_freedom"] == second.metadata["degrees_of_freedom"]
    estimator.inference_method = "bootstrap"
    estimator.n_bootstrap = 20
    estimator.estimate()
    assert estimator._labelled_inference == {}


# ---------------------------------------------------------------------------
# Too few labelled prompts
# ---------------------------------------------------------------------------


def test_one_labelled_prompt_leaves_the_policy_without_an_interval(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = _population(
        151, 80, 1, {"a": {"labelled": [5]}, "b": {}, "c": {"labelled": range(30)}}
    )
    with caplog.at_level(logging.WARNING, logger="cje.estimators.direct_method"):
        result = _estimate(data, _external_calibrator(152))
    meta = result.metadata
    point = meta["point_estimator"]
    assert point["routes"] == ["augmented", "plug_in", "augmented"]
    assert np.isnan(result.standard_errors[0])
    assert np.isfinite(result.estimates[0])
    assert result.estimates[0] == pytest.approx(
        point["plug_in_estimates"][0] + point["residual_corrections"][0]
    )
    assert "no degrees of freedom for an interval" in caplog.text
    assert meta["se_methods"]["a"] == "unavailable_too_few_labelled_clusters"
    assert meta["inference_unavailable_policies"] == ["a"]
    assert meta["inference_unavailable_reasons"] == {"a": "too_few_labelled_clusters"}
    info = meta["degrees_of_freedom"]["a"]
    assert info["available"] is False and info["df"] is None
    assert info["reason"] == "too_few_labelled_clusters"
    assert info["n_labelled_clusters"] == 1 and info["fitted_parameters"] == 1
    assert set(meta["pairwise_inference"]) == {"1-2"}

    with pytest.raises(
        InferenceUnavailableError, match="1 labelled prompt; at least 2"
    ):
        result.compare_policies(0, 1)
    with pytest.raises(InferenceUnavailableError, match="labelled prompt"):
        result.compare_policies(2, 0)
    with pytest.raises(InferenceUnavailableError, match="labelled prompt"):
        result.compare_all_policies()
    assert result.compare_policies(1, 2)["method"] == "paired_if_oua"

    assert "[nan, nan]" in result.summary()
    portable = result.to_dict(detail="portable")
    assert portable["pairwise_inference_state"]["capability"] == "stored_alpha_only"
    assert set(portable["pairwise_inference_state"]["comparisons"]) == {"1-2"}
    assert portable["per_policy_results"]["a"]["standard_error"] is None
    assert result.to_dict() == portable
    restored = EstimationResult.from_dict(result.to_dict(detail="full"))
    with pytest.raises(InferenceUnavailableError, match="labelled prompt"):
        restored.compare_policies(0, 2)


def test_two_labelled_prompts_give_one_degree_of_freedom(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = _population(161, 100, 1, {"policy": {"labelled": [3, 7]}})
    with caplog.at_level(logging.WARNING, logger="cje.estimators.direct_method"):
        result = _estimate(data, _external_calibrator(162))
    info = result.metadata["degrees_of_freedom"]["policy"]
    assert info["df"] == 1.0
    assert info["n_labelled_clusters"] == 2
    assert info["labelled_variance_inflation"] == 2.0
    assert "n_L-q = 1" in caplog.text and "very wide" in caplog.text
    lower, upper = result.confidence_interval()
    assert upper[0] - lower[0] == pytest.approx(
        2 * stats.t.ppf(0.975, 1) * result.standard_errors[0]
    )


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------


def test_readme_pair_takes_the_labelled_policys_df() -> None:
    result = analyze_dataset(fresh_draws_data=_readme_draws())
    policies = result.metadata["target_policies"]
    entry = result.metadata["pairwise_inference"]["0-1"]
    assert entry["basis"] == "prompt_cluster_paired"
    assert entry["df_method"] == "labelled_clusters"
    assert entry["df"] == 9.0
    assert entry["oracle_df_cap_applied"] is False
    assert entry["n_labelled_clusters"] == 10
    assert entry["labelled_variance_inflation"] == pytest.approx(10 / 9)
    v_lab, v_unl = _pair_split_by_hand(
        result, {p: README_PROMPTS for p in policies}, README_LABELLED
    )
    assert entry["se_sampling"] ** 2 == pytest.approx(10 / 9 * v_lab + v_unl, rel=1e-10)
    assert entry["labelled_variance_share"] == pytest.approx(
        v_lab / (v_lab + v_unl), rel=1e-10
    )
    assert entry["se"] ** 2 == pytest.approx(
        entry["se_sampling"] ** 2 + entry["var_oua_diff"], rel=1e-12
    )
    comparison = result.compare_policies(0, 1)
    assert comparison["df"] == 9.0
    assert comparison["se_difference"] == pytest.approx(entry["se"])
    gpt = result.metadata["degrees_of_freedom"]["gpt-5.6"]
    assert gpt["df"] == 9.0 and gpt["labels_coupled"] is True
    assert gpt["labels_coupled_fraction"] == 1.0
    assert result.metadata["degrees_of_freedom"]["fable-5"]["df_method"] == (
        "welch_satterthwaite"
    )


def test_readme_intervals_widen_and_never_narrow() -> None:
    """Against the 0.9.0 interval rebuilt from the same result."""
    result = analyze_dataset(fresh_draws_data=_readme_draws())
    policies = result.metadata["target_policies"]
    components = result.metadata["se_components"]
    lower, upper = result.confidence_interval()
    codes = np.unique(README_PROMPTS, return_inverse=True)[1]
    for index, policy in enumerate(policies):
        crv1 = cluster_robust_se(
            _influence(result, policy), codes, np.mean, lambda x: x
        )
        v_oracle = components["oracle_variance_per_policy"][policy]
        folds = components["oracle_jackknife_counts"][policy]
        old_df = _welch(crv1["se"] ** 2, 19, v_oracle, folds)
        old_width = 2 * stats.t.ppf(0.975, old_df) * np.sqrt(crv1["se"] ** 2 + v_oracle)
        new_width = upper[index] - lower[index]
        if policy == "gpt-5.6":
            assert new_width > old_width * 1.05
        else:
            assert new_width == pytest.approx(old_width, rel=1e-12)

    entry = result.metadata["pairwise_inference"]["0-1"]
    v_lab, v_unl = _pair_split_by_hand(
        result, {p: README_PROMPTS for p in policies}, README_LABELLED
    )
    old_se = np.sqrt(v_lab + v_unl + entry["var_oua_diff"])
    old_df = _welch(v_lab + v_unl, 19, entry["var_oua_diff"], entry["oua_folds"])
    comparison = result.compare_policies(0, 1)
    new_width = comparison["ci_upper"] - comparison["ci_lower"]
    assert new_width > 2 * stats.t.ppf(0.975, old_df) * old_se * 1.05


@pytest.mark.parametrize(
    "labels_a, labels_b, weight, inflation, df, n_union",
    [
        (range(12), range(12), "one", 12 / 11, 11, 12),  # shared
        (range(10), range(10, 26), "one", 10 / 9, 9, 26),  # disjoint, unequal
        (range(25), range(10), "tuned", 10 / 9, 9, 25),  # tuned 25 vs guard-one 10
    ],
)
def test_pairs_take_the_largest_inflation_and_the_smallest_df(
    labels_a: Iterable[int],
    labels_b: Iterable[int],
    weight: str,
    inflation: float,
    df: int,
    n_union: int,
) -> None:
    data = _population(
        171,
        200,
        1,
        {
            "a": {"labelled": labels_a, "shift": 0.03},
            "b": {"labelled": labels_b, "shift": -0.03},
        },
    )
    result = _estimate(data, _external_calibrator(172), correction_weight=weight)
    if weight == "tuned":
        assert result.metadata["point_estimator"]["correction_weight_reasons"] == [
            "tuned",
            "too_few_labelled_clusters",
        ]
    entry = result.metadata["pairwise_inference"]["0-1"]
    assert entry["df_method"] == "labelled_clusters"
    assert entry["labelled_variance_inflation"] == pytest.approx(inflation)
    assert entry["n_labelled_clusters"] == n_union
    labelled = {
        row[0] for policy in ("a", "b") for row in data[policy] if row[3] is not None
    }
    v_lab, v_unl = _pair_split_by_hand(
        result, {p: _prompts(data[p]) for p in ("a", "b")}, labelled
    )
    assert entry["se_sampling"] ** 2 == pytest.approx(
        inflation * v_lab + v_unl, rel=1e-10
    )
    expected_df = _expected_df(
        entry["se_sampling"] ** 2, df, v_lab + v_unl, 199, entry["var_oua_diff"], 5
    )
    assert entry["df"] == pytest.approx(expected_df, rel=1e-12)
    assert entry["oracle_df_cap_applied"] is bool(expected_df < df)


def test_unpaired_pairs_combine_the_adjusted_policy_ses() -> None:
    data = _population(
        181, 150, 1, {"a": {"labelled": range(12)}, "b": {"labelled": range(20, 40)}}
    )
    result = _estimate(data, _external_calibrator(182), paired_comparison=False)
    entry = result.metadata["pairwise_inference"]["0-1"]
    assert entry["basis"] == "independent_requested"
    assert result.diagnostics is not None
    sampling = result.diagnostics.standard_errors
    assert entry["se_sampling"] == pytest.approx(
        np.hypot(sampling["a"], sampling["b"]), rel=1e-12
    )
    codes = np.unique(_prompts(data["a"]), return_inverse=True)[1]
    unadjusted = sum(
        cluster_robust_se(_influence(result, p), codes, np.mean, lambda x: x)["se"] ** 2
        for p in ("a", "b")
    )
    expected_df = _expected_df(
        entry["se_sampling"] ** 2, 11, unadjusted, 149, entry["var_oua_diff"], 5
    )
    assert entry["df"] == pytest.approx(expected_df, rel=1e-12)
    assert entry["df_method"] == "labelled_clusters"
    assert entry["n_labelled_clusters"] == 32
    assert entry["labelled_variance_inflation"] == pytest.approx(12 / 11)
    assert 0 < entry["labelled_variance_share"] < 1


@pytest.mark.parametrize("oracle_ratio", [1.0, 500.0])
def test_oracle_caps_bind_when_the_jackknife_is_large(
    monkeypatch: pytest.MonkeyPatch, oracle_ratio: float
) -> None:
    """Comparable oracle variance: the labelled Welch df binds. Dominant
    oracle variance: the unadjusted Welch df binds (inflating the sampling
    variance would otherwise raise the df and narrow the interval)."""
    data = _population(
        191, 120, 1, {"a": {"labelled": range(30)}, "b": {"labelled": range(30)}}
    )
    estimator = _estimator(data, _external_calibrator(192))
    plain = estimator.fit_and_estimate()
    assert plain.diagnostics is not None
    var_a = plain.diagnostics.standard_errors["a"] ** 2
    # The jackknife variance of c * [-2, -1, 0, 1, 2] is 8 c^2.
    spread = np.sqrt(oracle_ratio * var_a / 8) * np.array([-2.0, -1, 0, 1, 2])
    jackknife = {"a": 0.5 + spread, "b": np.zeros(5)}
    monkeypatch.setattr(
        estimator, "get_oracle_jackknife", lambda policy: jackknife[policy].copy()
    )
    result = estimator.estimate()
    assert result.diagnostics is not None
    sampling = result.diagnostics.standard_errors
    v_oracle = oracle_jackknife_variance(jackknife["a"])
    assert v_oracle == pytest.approx(oracle_ratio * var_a, rel=1e-9)
    rows = data["a"]
    v_lab, v_unl, _ = _split_by_hand(
        _influence(result, "a"), _prompts(rows), _labelled(rows)
    )
    labelled_welch = _welch(sampling["a"] ** 2, 29, v_oracle, 5)
    unadjusted_welch = _welch(v_lab + v_unl, 119, v_oracle, 5)
    if oracle_ratio == 1.0:
        assert labelled_welch < unadjusted_welch
    else:
        assert unadjusted_welch < labelled_welch
    info = result.metadata["degrees_of_freedom"]
    assert info["a"]["df"] == pytest.approx(
        min(labelled_welch, unadjusted_welch), rel=1e-12
    )
    assert info["a"]["df"] < 29
    assert info["a"]["oracle_df_cap_applied"] is True
    assert _half_width(sampling["a"] ** 2 + v_oracle, info["a"]["df"]) > (
        _half_width(v_lab + v_unl + v_oracle, unadjusted_welch)
    )
    assert info["b"]["df"] == 29.0  # zero oracle variance: the labelled df
    assert info["b"]["oracle_df_cap_applied"] is False

    entry = result.metadata["pairwise_inference"]["0-1"]
    assert entry["var_oua_diff"] == pytest.approx(v_oracle, rel=1e-12)
    v_lab_pair, v_unl_pair = _pair_split_by_hand(
        result,
        {p: _prompts(data[p]) for p in ("a", "b")},
        {row[0] for row in rows if row[3] is not None},
    )
    pair_labelled_welch = _welch(entry["se_sampling"] ** 2, 29, v_oracle, 5)
    pair_unadjusted_welch = _welch(v_lab_pair + v_unl_pair, 119, v_oracle, 5)
    if oracle_ratio == 1.0:
        assert pair_labelled_welch < pair_unadjusted_welch
    else:
        assert pair_unadjusted_welch < pair_labelled_welch
    assert entry["df"] == pytest.approx(
        min(pair_labelled_welch, pair_unadjusted_welch), rel=1e-12
    )
    assert entry["oracle_df_cap_applied"] is True
    assert _half_width(entry["se"] ** 2, entry["df"]) > _half_width(
        v_lab_pair + v_unl_pair + v_oracle, pair_unadjusted_welch
    )


# ---------------------------------------------------------------------------
# Array API
# ---------------------------------------------------------------------------


def _array_sample(seed: int, n: int = 400, n_labelled: int = 30) -> Tuple[Any, Any]:
    rng = np.random.default_rng(seed)
    scores = rng.uniform(size=n)
    labels = np.full(n, np.nan)
    labelled = rng.choice(n, size=n_labelled, replace=False)
    labels[labelled] = np.clip(scores[labelled] + rng.normal(0, 0.15, n_labelled), 0, 1)
    return scores, labels


def test_array_api_reports_the_adjusted_and_raw_cluster_se() -> None:
    scores, labels = _array_sample(201)
    result = calibrated_mean_ci(scores, labels)
    diag = result.diagnostics["cluster_robust"]
    assert diag["df_method"] == "labelled_clusters"
    assert diag["n_labelled_clusters"] == 30
    assert diag["fitted_parameters"] == 1
    assert diag["labels_coupled"] is True
    assert diag["labels_coupled_fraction"] == 1.0
    assert {
        "se_cluster_unadjusted",
        "n_labelled_clusters",
        "fitted_parameters",
        "labelled_variance_inflation",
        "labelled_variance_share",
        "labels_coupled",
        "labels_coupled_fraction",
        "oracle_df_cap_applied",
    } <= set(diag)
    inflation = diag["labelled_variance_inflation"]
    share = diag["labelled_variance_share"]
    assert inflation == pytest.approx(30 / 29)
    assert diag["se_cluster"] ** 2 == pytest.approx(
        diag["se_cluster_unadjusted"] ** 2 * (inflation * share + 1 - share),
        rel=1e-12,
    )
    assert diag["se_cluster"] > diag["se_cluster_unadjusted"]
    assert result.se**2 == pytest.approx(
        diag["se_cluster"] ** 2 + diag["var_oracle"], rel=1e-12
    )
    expected_df = _expected_df(
        diag["se_cluster"] ** 2,
        29,
        diag["se_cluster_unadjusted"] ** 2,
        399,
        diag["var_oracle"],
        diag["oracle_jackknife_folds"],
    )
    assert diag["df"] == pytest.approx(expected_df, rel=1e-12)
    assert diag["oracle_df_cap_applied"] is bool(expected_df < 29)
    t_crit = stats.t.ppf(0.975, diag["df"])
    assert result.ci[0] == pytest.approx(result.estimate - t_crit * result.se)
    assert result.ci[1] == pytest.approx(result.estimate + t_crit * result.se)

    tuned = calibrated_mean_ci(scores, labels, correction_weight="tuned")
    assert tuned.diagnostics["correction_weight"]["reason"] == "tuned"
    assert tuned.diagnostics["cluster_robust"]["fitted_parameters"] == 2
    assert tuned.diagnostics["cluster_robust"]["labelled_variance_inflation"] == (
        pytest.approx(30 / 28)
    )


@pytest.mark.parametrize(
    "oracle_ratio, offsets",
    [(1.0, [-2.0, -1.0, 0.0, 1.0, 2.0]), (20.0, [-1.0, 1.0])],
)
def test_array_api_oracle_caps_bind_when_the_jackknife_is_large(
    monkeypatch: pytest.MonkeyPatch, oracle_ratio: float, offsets: List[float]
) -> None:
    """Comparable oracle variance (K = 5): the labelled Welch df binds.
    Dominant oracle variance (K = 2): the unadjusted Welch df binds."""
    import cje.array_api as array_api

    scores, labels = _array_sample(201)
    plain = calibrated_mean_ci(scores, labels).diagnostics["cluster_robust"]
    shape = np.asarray(offsets)
    scale = np.sqrt(
        oracle_ratio * plain["se_cluster"] ** 2 / oracle_jackknife_variance(shape)
    )
    jackknife = (0.5 + scale * shape)[:, None]
    monkeypatch.setattr(
        array_api,
        "direct_oracle_jackknife_estimates",
        lambda *args, **kwargs: jackknife.copy(),
    )
    result = calibrated_mean_ci(scores, labels)
    diag = result.diagnostics["cluster_robust"]
    k = len(offsets)
    assert diag["oracle_jackknife_folds"] == k
    v_oracle = diag["var_oracle"]
    assert v_oracle == pytest.approx(oracle_ratio * diag["se_cluster"] ** 2)
    labelled_welch = _welch(diag["se_cluster"] ** 2, 29, v_oracle, k)
    unadjusted_welch = _welch(diag["se_cluster_unadjusted"] ** 2, 399, v_oracle, k)
    if oracle_ratio == 1.0:
        assert labelled_welch < unadjusted_welch
    else:
        assert unadjusted_welch < labelled_welch
    assert diag["df"] == pytest.approx(min(labelled_welch, unadjusted_welch), rel=1e-12)
    assert diag["df"] < 29
    assert diag["oracle_df_cap_applied"] is True
    t_crit = stats.t.ppf(0.975, diag["df"])
    assert result.ci[1] - result.estimate == pytest.approx(t_crit * result.se)
    assert result.ci[1] - result.estimate > _half_width(
        diag["se_cluster_unadjusted"] ** 2 + v_oracle, unadjusted_welch
    )


def test_array_api_reports_an_unavailable_interval_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cje.array_api as array_api

    monkeypatch.setattr(
        array_api,
        "labelled_cluster_variance",
        lambda *args, **kwargs: (float("nan"), None),
    )
    scores, labels = _array_sample(211)
    result = calibrated_mean_ci(scores, labels)
    assert np.isfinite(result.estimate)
    assert np.isnan(result.se)
    assert np.isnan(result.ci[0]) and np.isnan(result.ci[1])
    assert result.diagnostics["inference_available"] is False
    diag = result.diagnostics["cluster_robust"]
    assert diag["unavailable_reason"] == "too_few_labelled_clusters"
    assert np.isnan(diag["se_cluster"])
    assert diag["se_cluster_unadjusted"] > 0
    assert "nan" in result.summary()


# ---------------------------------------------------------------------------
# labels_coupled, scale, serialization, bootstrap
# ---------------------------------------------------------------------------


def _write_calibration(path: Path, seed: int, n: int = 300) -> None:
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.15, 0.85, size=n)
    path.write_text(
        "".join(
            json.dumps(
                {
                    "prompt_id": f"c{i}",
                    "judge_score": float(np.clip(t + rng.normal(0, 0.2), 0, 1)),
                    "oracle_label": float(np.clip(t + rng.normal(0, 0.15), 0, 1)),
                }
            )
            + "\n"
            for i, t in enumerate(truth)
        )
    )


def test_labels_coupled_says_whether_the_labels_fitted_the_calibrator(
    tmp_path: Path,
) -> None:
    # Labels only in the fresh draws: they also fitted the calibrator.
    coupled = analyze_dataset(fresh_draws_data=_readme_draws())
    info = coupled.metadata["degrees_of_freedom"]["gpt-5.6"]
    assert info["labels_coupled"] is True and info["labels_coupled_fraction"] == 1.0

    # External calibration only: the fresh-draw labels did not.
    path = tmp_path / "calibration.jsonl"
    _write_calibration(path, 221)
    data = _population(222, 200, 1, {"policy": {"labelled": range(15)}})
    records = {
        "policy": [
            {"prompt_id": prompt, "draw_idx": draw, "judge_score": score}
            | ({"oracle_label": label} if label is not None else {})
            for prompt, draw, score, label in data["policy"]
        ]
    }
    scales: Dict[str, Any] = {
        "fresh_judge_scale": (0.0, 1.0),
        "fresh_oracle_scale": (0.0, 1.0),
        "calibration_judge_scale": (0.0, 1.0),
        "calibration_oracle_scale": (0.0, 1.0),
    }
    external = analyze_dataset(
        fresh_draws_data=records,
        calibration_data_path=str(path),
        combine_oracle_sources=False,
        **scales,
    )
    info = external.metadata["degrees_of_freedom"]["policy"]
    assert external.metadata["point_estimator"]["routes"] == ["augmented"]
    assert info["labels_coupled"] is False and info["labels_coupled_fraction"] == 0.0
    # Pooled with the external rows, they did.
    pooled = analyze_dataset(
        fresh_draws_data=records, calibration_data_path=str(path), **scales
    )
    assert pooled.metadata["degrees_of_freedom"]["policy"]["labels_coupled"] is True

    # Mixed: half of the labelled rows entered the calibration fit.
    rng = np.random.default_rng(223)
    truth = rng.uniform(0.15, 0.85, size=200)
    ext_scores = np.clip(truth + rng.normal(0, 0.2, size=200), 0, 1)
    ext_labels = np.clip(truth + rng.normal(0, 0.15, size=200), 0, 1)
    data = _population(224, 200, 1, {"policy": {"labelled": range(20)}})
    linked = [row for row in data["policy"] if row[3] is not None][:10]
    calibrator = JudgeCalibrator(random_seed=42, calibration_mode="monotone")
    calibrator.fit_cv(
        np.r_[ext_scores, [row[2] for row in linked]],
        np.r_[ext_labels, [row[3] for row in linked]],
        n_folds=5,
        prompt_ids=[f"c{i}" for i in range(200)] + [row[0] for row in linked],
        quiet=True,
    )
    roles: List[Literal["external", "evaluation"]] = ["external"] * 200
    roles += ["evaluation"] * 10
    provenance = CalibrationProvenance(
        judge_scores=np.r_[ext_scores, [row[2] for row in linked]],
        oracle_labels=np.r_[ext_labels, [row[3] for row in linked]],
        prompt_ids=[f"c{i}" for i in range(200)] + [row[0] for row in linked],
        row_roles=roles,
        evaluation_keys=[None] * 200 + [("policy", row[0], row[1]) for row in linked],
    )
    mixed = _estimate(data, calibrator, provenance=provenance)
    info = mixed.metadata["degrees_of_freedom"]["policy"]
    assert info["labels_coupled"] is False
    assert info["labels_coupled_fraction"] == 0.5


def test_new_keys_are_scale_free_and_survive_a_round_trip() -> None:
    unit = analyze_dataset(
        fresh_draws_data=_readme_draws(),
        fresh_judge_scale=(0.0, 1.0),
        fresh_oracle_scale=(0.0, 1.0),
    )
    tenfold = analyze_dataset(
        fresh_draws_data=_readme_draws(label_scale=10.0),
        fresh_judge_scale=(0.0, 1.0),
        fresh_oracle_scale=(0.0, 10.0),
    )
    assert np.allclose(tenfold.standard_errors, 10 * unit.standard_errors, rtol=1e-9)
    scale_free = (
        "df",
        "df_method",
        "n_labelled_clusters",
        "fitted_parameters",
        "labelled_variance_inflation",
        "labelled_variance_share",
        "labels_coupled",
        "labels_coupled_fraction",
        "oracle_df_cap_applied",
    )
    unit_info = unit.metadata["degrees_of_freedom"]["gpt-5.6"]
    tenfold_info = tenfold.metadata["degrees_of_freedom"]["gpt-5.6"]
    for key in scale_free:
        assert tenfold_info[key] == pytest.approx(unit_info[key], rel=1e-9), key
    unit_pair = unit.metadata["pairwise_inference"]["0-1"]
    tenfold_pair = tenfold.metadata["pairwise_inference"]["0-1"]
    for key in (
        "df",
        "n_labelled_clusters",
        "labelled_variance_inflation",
        "labelled_variance_share",
        "oracle_df_cap_applied",
    ):
        assert tenfold_pair[key] == pytest.approx(unit_pair[key], rel=1e-9), key
    # A pair carries these df fields only; fitted_parameters and
    # labels_coupled(_fraction) stay per policy.
    assert set(unit_pair) == {
        "policy1",
        "policy2",
        "se",
        "df",
        "df_method",
        "basis",
        "se_sampling",
        "var_oua_diff",
        "n_pairs",
        "oua_folds",
        "n_labelled_clusters",
        "labelled_variance_inflation",
        "labelled_variance_share",
        "oracle_df_cap_applied",
    }
    assert tenfold_pair["se_sampling"] == pytest.approx(
        10 * unit_pair["se_sampling"], rel=1e-9
    )

    restored = EstimationResult.from_dict(unit.to_dict(detail="full"))
    assert (
        restored.metadata["degrees_of_freedom"] == unit.metadata["degrees_of_freedom"]
    )
    assert np.allclose(restored.confidence_interval(), unit.confidence_interval())
    assert restored.compare_policies(0, 1)["ci_lower"] == pytest.approx(
        unit.compare_policies(0, 1)["ci_lower"]
    )


def test_bootstrap_intervals_are_unchanged() -> None:
    result = analyze_dataset(
        fresh_draws_data=_readme_draws(),
        estimator_config={"inference_method": "bootstrap", "n_bootstrap": 200},
    )
    assert result.ci_info is not None and result.ci_info.method == "percentile"
    assert "degrees_of_freedom" not in result.metadata
    assert "pairwise_inference" not in result.metadata
    assert result.metadata["inference_unavailable_reasons"] == {}
    lower, upper = result.confidence_interval()
    assert list(lower) == result.metadata["bootstrap_ci"]["lower"]
    assert list(upper) == result.metadata["bootstrap_ci"]["upper"]
