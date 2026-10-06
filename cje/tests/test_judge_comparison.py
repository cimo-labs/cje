"""compare_judges: several judges against one oracle on the same labelled rows (#71).

Each judge gets its own cross-fitted calibrator on folds shared by every judge;
the table reports out-of-fold quality per judge, the pairwise rows the R² and
RMSE differences and the label multiplier (1 - R²_ref) / (1 - R²_J) against a
reference, and every interval comes from a paired prompt-cluster bootstrap
that refits each judge.
"""

from __future__ import annotations

import json
import logging
import warnings
from typing import Any, Dict, Tuple

import numpy as np
import pytest

import cje
from cje import calibrated_mean_ci, compare_judges, transport_audit
from cje.calibration.judge import JudgeCalibrator
from cje.judge_comparison import _finite_n

SEED = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)))


def _issue_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The reproduction in issue #71: 4000 rows, 800 labelled, two judges."""
    rng = np.random.default_rng(1)
    n, n_lab = 4000, 800
    q = rng.normal(size=n)
    y = (rng.uniform(size=n) < _sigmoid(3 * q - 0.5)).astype(float)
    s1 = _sigmoid(1.2 * (q + rng.normal(0, 0.9, n)) + 2.0)  # lenient
    s2 = _sigmoid(1.2 * (q + rng.normal(0, 0.5, n)) - 2.0)  # strict, sharper
    lab = np.zeros(n, bool)
    lab[rng.choice(n, n_lab, replace=False)] = True
    pid = np.array([f"p{i}" for i in range(n)])
    return s1, s2, y, lab, pid


def _small(
    n: int = 600, n_lab: int = 240, seed: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous outcome, a sharp and a noisy judge, random labels."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=n)
    y = np.clip(_sigmoid(1.5 * q) + rng.normal(0, 0.08, n), 0, 1)
    sharp = q + rng.normal(0, 0.4, n)
    noisy = q + rng.normal(0, 1.2, n)
    labels = np.full(n, np.nan)
    idx = rng.choice(n, n_lab, replace=False)
    labels[idx] = y[idx]
    return sharp, noisy, labels


def _manual_oof(
    s: np.ndarray,
    labels: np.ndarray,
    prompt_ids: Any,
    *,
    weights: Any = None,
    mode: str = "auto",
) -> Tuple[JudgeCalibrator, np.ndarray, np.ndarray]:
    """Reference fit through the public JudgeCalibrator API."""
    mask = ~np.isnan(labels)
    cal = JudgeCalibrator(random_seed=SEED, calibration_mode=mode)  # type: ignore[arg-type]
    fit = cal.fit_cv(
        s,
        labels[mask],
        mask,
        n_folds=5,
        prompt_ids=prompt_ids,
        quiet=True,
        sample_weight=None if weights is None else weights[mask],
    )
    assert fit.fold_ids is not None
    oof = cal.predict_oof(s[mask], fit.fold_ids[mask])
    return cal, oof, fit.calibrated_scores


def _r2(y: np.ndarray, oof: np.ndarray, w: Any = None) -> float:
    r = y - oof
    rc = r - np.average(r, weights=w)
    yc = y - np.average(y, weights=w)
    return float(1 - np.average(rc**2, weights=w) / np.average(yc**2, weights=w))


def cast_float(value: Any) -> float:
    assert value is not None
    return float(value)


@pytest.fixture(autouse=True)
def _quiet_logs() -> Any:
    logging.disable(logging.INFO)
    yield
    logging.disable(logging.NOTSET)


# ---------------------------------------------------------------------------
# Point statistics
# ---------------------------------------------------------------------------


def test_issue71_reproduction_matches_manual() -> None:
    s1, s2, y, lab, pid = _issue_data()
    labels = np.where(lab, y, np.nan)
    cmp = compare_judges({"v1": s1, "v2": s2}, labels, pid, n_bootstrap=4)

    manual: Dict[str, Tuple[float, float]] = {}
    for name, s in (("v1", s1), ("v2", s2)):
        cal, oof, _ = _manual_oof(s, labels, list(pid))
        r = y[lab] - oof
        manual[name] = (_r2(y[lab], oof), float(np.sqrt(np.mean(r**2))))
        row = cmp.row(name)
        assert row.r2_within == pytest.approx(manual[name][0], abs=1e-12)
        assert row.r2_pooled == row.r2_within  # no policy_ids: one group
        assert row.oof_rmse == pytest.approx(manual[name][1], abs=1e-12)
        assert row.oof_rmse == pytest.approx(
            cal.get_calibration_info()["oof_rmse"], abs=1e-12
        )
        assert row.selected_mode == cal.selected_mode
        assert row.n_labelled_rows == 800 and row.n_labelled_clusters == 800

    # The issue's numbers.
    assert round(cmp.row("v1").r2_within, 3) == 0.322
    assert round(cmp.row("v2").r2_within, 3) == 0.454
    assert round(cmp.row("v1").oof_rmse, 3) == 0.409
    assert round(cmp.row("v2").oof_rmse, 3) == 0.367
    pair = cmp.pair("v2")
    assert pair.reference == "v1"
    assert pair.r2_within_diff == pytest.approx(
        manual["v2"][0] - manual["v1"][0], abs=1e-12
    )
    assert pair.label_multiplier == pytest.approx(
        (1 - manual["v1"][0]) / (1 - manual["v2"][0]), abs=1e-12
    )
    assert round(pair.label_multiplier, 2) == 1.24
    assert pair.variance_ratio_at_n is None and pair.label_multiplier_at_n is None


def test_issue71_finite_n_matches_closed_form() -> None:
    s1, s2, y, lab, pid = _issue_data()
    labels = np.where(lab, y, np.nan)
    cmp = compare_judges(
        {"v1": s1, "v2": s2}, labels, pid, n_unlabeled=3200, n_bootstrap=4
    )
    v1, v2 = cmp.row("v1"), cmp.row("v2")
    # Var(f) is over every row; Var(Y - f) is the centred OOF residual variance.
    _, oof1, f1 = _manual_oof(s1, labels, list(pid))
    assert v1.var_f == pytest.approx(np.var(f1), abs=1e-12)
    assert v1.var_residual == pytest.approx(np.var(y[lab] - oof1), abs=1e-12)
    n, big_n = 800.0, 4000.0
    v_ref = v1.var_f / big_n + v1.var_residual / n
    v_j = v2.var_f / big_n + v2.var_residual / n
    pair = cmp.pair("v2")
    assert pair.variance_ratio_at_n == pytest.approx(v_ref / v_j, rel=1e-12)
    # The issue's finite-N prediction at N = 4000.
    assert round(cast_float(pair.variance_ratio_at_n), 2) == 1.16
    m_ref = v1.var_residual / (v_j - v1.var_f / big_n)
    assert pair.label_multiplier_at_n == pytest.approx(m_ref / n, rel=1e-12)
    assert pair.label_multiplier_at_n_capped is False
    assert cmp.diagnostics["planned"] == {
        "n_unlabeled": 3200,
        "labelled_rows_per_policy": 800.0,
        "rows_per_policy": 4000.0,
    }


def test_cluster_ids_none_matches_calibrated_mean_ci() -> None:
    sharp, _, labels = _small()
    cmp = compare_judges({"a": sharp}, labels, None, n_bootstrap=2)
    res = calibrated_mean_ci(sharp, labels)
    theirs = res.calibrator
    assert theirs is not None
    ours = cmp.calibrators["a"]
    assert ours.selected_mode == theirs.selected_mode
    assert np.array_equal(ours.predict(sharp), theirs.predict(sharp))
    assert ours.n_folds == theirs.n_folds == cmp.diagnostics["n_folds"]
    assert cmp.row("a").oof_rmse == pytest.approx(
        res.diagnostics["calibration"]["oof_rmse"], abs=1e-12
    )
    assert cmp.diagnostics["cluster_ids_given"] is False
    with pytest.raises(TypeError):
        compare_judges({"a": sharp}, labels)  # type: ignore[call-arg]


def test_affine_rescaling_and_judge_scales_change_nothing() -> None:
    sharp, noisy, labels = _small()
    base = compare_judges({"a": sharp, "b": noisy}, labels, None, n_bootstrap=3)
    rescaled = compare_judges(
        {"a": 10 * sharp + 1, "b": 100 * noisy - 37},
        labels,
        None,
        judge_scales={"a": (-100.0, 100.0)},
        n_bootstrap=3,
    )
    for name in ("a", "b"):
        x, z = base.row(name), rescaled.row(name)
        assert z.selected_mode == x.selected_mode
        for field in ("oof_rmse", "r2_within", "r2_pooled", "var_f", "var_residual"):
            assert getattr(z, field) == pytest.approx(getattr(x, field), abs=1e-10)
        np.testing.assert_allclose(z.r2_within_ci, x.r2_within_ci, atol=1e-10)
    assert rescaled.pair("b").label_multiplier == pytest.approx(
        base.pair("b").label_multiplier, abs=1e-10
    )
    assert rescaled.diagnostics["judge_scales"] == {"a": [-100.0, 100.0]}
    # A binary 0/1 judge next to a 1-10 judge: each keeps the row it gets alone.
    rng = np.random.default_rng(5)
    binary = (sharp + rng.normal(0, 0.3, len(sharp)) > 0).astype(float)
    ten = np.clip(np.round(5.5 + 2 * noisy), 1, 10)
    both = compare_judges(
        {"binary": binary, "ten": ten},
        labels,
        None,
        judge_scales={"binary": (0, 1), "ten": (1, 10)},
        n_bootstrap=3,
    )
    assert both.row("binary") == compare_judges(
        {"binary": binary}, labels, None, n_bootstrap=3
    ).row("binary")
    with pytest.raises(ValueError, match=r"'ten' has \d+ score\(s\) outside"):
        compare_judges(
            {"ten": ten}, labels, None, judge_scales={"ten": (0, 1)}, n_bootstrap=2
        )
    with pytest.raises(ValueError, match="not one of the judges"):
        compare_judges(
            {"ten": ten}, labels, None, judge_scales={"x": (0, 1)}, n_bootstrap=2
        )
    with pytest.raises(ValueError, match="lo < hi"):
        compare_judges(
            {"ten": ten}, labels, None, judge_scales={"ten": (10, 1)}, n_bootstrap=2
        )


def test_policy_tracking_judge_within_vs_pooled() -> None:
    rng = np.random.default_rng(11)
    n_per = 600
    policy = np.repeat(["base", "new"], n_per)
    q = rng.normal(size=2 * n_per)
    y = np.clip(
        0.3 + 0.3 * (policy == "new") + 0.08 * q + rng.normal(0, 0.06, 2 * n_per), 0, 1
    )
    tracker = (policy == "new") + 0.3 * q + rng.normal(0, 1.0, 2 * n_per)
    within = q + rng.normal(0, 0.5, 2 * n_per)
    labels = np.full(2 * n_per, np.nan)
    idx = rng.choice(2 * n_per, 600, replace=False)
    labels[idx] = y[idx]
    cmp = compare_judges(
        {"tracker": tracker, "within": within},
        labels,
        None,
        policy_ids=policy,
        n_bootstrap=3,
    )
    t, w = cmp.row("tracker"), cmp.row("within")
    assert t.r2_pooled > t.r2_within + 0.05
    assert w.r2_within > t.r2_within
    assert cmp.diagnostics["policies"] == ["base", "new"]

    # Hand-computed within-policy centring.
    mask = ~np.isnan(labels)
    _, oof, f = _manual_oof(tracker, labels, None)
    r, yl, pl = y[mask] - oof, y[mask], policy[mask]
    ss_r = sum(np.sum((r[pl == p] - r[pl == p].mean()) ** 2) for p in ("base", "new"))
    ss_y = sum(np.sum((yl[pl == p] - yl[pl == p].mean()) ** 2) for p in ("base", "new"))
    assert t.r2_within == pytest.approx(1 - ss_r / ss_y, abs=1e-12)
    assert t.var_residual == pytest.approx(ss_r / mask.sum(), abs=1e-12)
    ss_f = sum(
        np.sum((f[policy == p] - f[policy == p].mean()) ** 2) for p in ("base", "new")
    )
    assert t.var_f == pytest.approx(ss_f / len(f), abs=1e-12)
    assert t.r2_pooled == pytest.approx(_r2(yl, oof), abs=1e-12)

    # Without policy_ids, within and pooled coincide exactly.
    pooled = compare_judges({"tracker": tracker}, labels, None, n_bootstrap=3)
    assert pooled.row("tracker").r2_within == pooled.row("tracker").r2_pooled


def test_policy_without_labels_is_excluded() -> None:
    sharp, noisy, labels = _small(n=600, n_lab=200)
    policy = np.repeat(["a", "b", "c"], 200)
    labels = labels.copy()
    labels[policy == "c"] = np.nan
    with pytest.warns(
        UserWarning, match=r"excluded from every statistic: \['c'\]"
    ) as rec:
        cmp = compare_judges(
            {"s": sharp, "n": noisy}, labels, None, policy_ids=policy, n_bootstrap=3
        )
    assert sum("excluded from every statistic" in str(w.message) for w in rec) == 1
    assert cmp.diagnostics["policies"] == ["a", "b"]
    assert cmp.diagnostics["policies_without_labels"] == ["c"]
    keep = policy != "c"
    ref = compare_judges(
        {"s": sharp[keep], "n": noisy[keep]},
        labels[keep],
        None,
        policy_ids=policy[keep],
        n_bootstrap=3,
    )
    for name in ("s", "n"):
        for field in ("oof_rmse", "r2_within", "r2_pooled", "var_f", "var_residual"):
            assert getattr(cmp.row(name), field) == pytest.approx(
                getattr(ref.row(name), field), abs=1e-12
            )
    assert cmp.pair("n").label_multiplier == pytest.approx(
        ref.pair("n").label_multiplier, abs=1e-12
    )


def test_finite_n_helper_limits_and_cap() -> None:
    a = np.asarray
    # Plentiful unlabelled rows: the finite-N multiplier tends to the ratio of
    # residual variances, i.e. (1 - R²_ref) / (1 - R²_J).
    ratio, mult, capped = _finite_n(a(0.05), a(0.20), a(0.10), a(0.10), 100.0, 1e12)
    assert float(mult) == pytest.approx(2.0, rel=1e-8)
    assert float(ratio) == pytest.approx(2.0, rel=1e-8)
    assert not bool(capped)
    # Identical components: the reference needs exactly as many labels.
    ratio, mult, capped = _finite_n(a(0.1), a(0.2), a(0.1), a(0.2), 100.0, 400.0)
    assert float(ratio) == 1.0
    assert float(mult) == pytest.approx(1.0, rel=1e-12)
    # A reference whose Var(f) term alone exceeds J's variance cannot match J
    # even with every row labelled: capped at N / n.
    ratio, mult, capped = _finite_n(a(0.9), a(0.5), a(0.01), a(0.01), 100.0, 400.0)
    assert bool(capped) and float(mult) == 4.0
    assert float(ratio) > 1


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_matches_manual_paired_refit() -> None:
    rng = np.random.default_rng(8)
    n_prompts, draws = 150, 2
    prompt = np.repeat(np.arange(n_prompts), draws)
    q = rng.normal(size=n_prompts)[prompt] + rng.normal(0, 0.3, n_prompts * draws)
    y = np.clip(_sigmoid(q) + rng.normal(0, 0.1, len(q)), 0, 1)
    s1 = q + rng.normal(0, 0.5, len(q))
    s2 = q + rng.normal(0, 1.0, len(q))
    labelled_prompts = rng.choice(n_prompts, 70, replace=False)
    labels = np.where(np.isin(prompt, labelled_prompts), y, np.nan)
    ids = [f"q{p}" for p in prompt]
    n_boot = 5
    cmp = compare_judges(
        {"s1": s1, "s2": s2}, labels, ids, n_bootstrap=n_boot, alpha=0.2
    )
    assert cmp.row("s1").n_labelled_clusters == 70
    assert cmp.row("s1").n_labelled_rows == 140

    mask = ~np.isnan(labels)
    modes = {name: cmp.calibrators[name].selected_mode for name in ("s1", "s2")}
    assert cmp.diagnostics["bootstrap"]["refit_modes"] == modes
    boot = np.random.default_rng(SEED)
    r2s: Dict[str, list] = {"s1": [], "s2": []}
    rmses: Dict[str, list] = {"s1": [], "s2": []}
    for _ in range(n_boot):
        # Clusters in first-appearance order: q0, q1, ... (prompt order here).
        w = boot.exponential(size=n_prompts)[prompt]
        for name, s in (("s1", s1), ("s2", s2)):
            _, oof, _ = _manual_oof(s, labels, ids, weights=w, mode=str(modes[name]))
            r2s[name].append(_r2(y[mask], oof, w[mask]))
            rmses[name].append(
                float(np.sqrt(np.average((y[mask] - oof) ** 2, weights=w[mask])))
            )
    for name in ("s1", "s2"):
        np.testing.assert_allclose(
            cmp.row(name).r2_within_ci, np.percentile(r2s[name], [10, 90]), atol=1e-10
        )
        np.testing.assert_allclose(
            cmp.row(name).oof_rmse_ci, np.percentile(rmses[name], [10, 90]), atol=1e-10
        )
    diff = np.asarray(r2s["s2"]) - np.asarray(r2s["s1"])
    np.testing.assert_allclose(
        cmp.pair("s2").r2_within_diff_ci, np.percentile(diff, [10, 90]), atol=1e-10
    )
    mult = (1 - np.asarray(r2s["s1"])) / (1 - np.asarray(r2s["s2"]))
    np.testing.assert_allclose(
        cmp.pair("s2").label_multiplier_ci, np.percentile(mult, [10, 90]), atol=1e-10
    )


def test_judge_row_independent_of_other_judges_and_deterministic() -> None:
    sharp, noisy, labels = _small()
    alone = compare_judges({"a": sharp}, labels, None, n_bootstrap=6)
    with_b = compare_judges({"a": sharp, "b": noisy}, labels, None, n_bootstrap=6)
    assert alone.row("a") == with_b.row("a")
    again = compare_judges({"a": sharp, "b": noisy}, labels, None, n_bootstrap=6)
    assert again.to_dict() == with_b.to_dict()
    other_seed = compare_judges(
        {"a": sharp, "b": noisy}, labels, None, n_bootstrap=6, seed=7
    )
    assert other_seed.diagnostics["seed"] == 7
    assert other_seed.row("a").r2_within_ci != with_b.row("a").r2_within_ci


def test_identical_judges_pair_exactly() -> None:
    sharp, _, labels = _small()
    cmp = compare_judges({"a": sharp, "b": sharp.copy()}, labels, None, n_bootstrap=5)
    pair = cmp.pair("b")
    assert pair.r2_within_diff == 0.0 and pair.r2_within_diff_ci == (0.0, 0.0)
    assert pair.oof_rmse_diff == 0.0 and pair.oof_rmse_diff_ci == (0.0, 0.0)
    assert pair.label_multiplier == 1.0 and pair.label_multiplier_ci == (1.0, 1.0)


def test_refit_failure_names_replicate_and_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sharp, noisy, labels = _small()
    original = JudgeCalibrator.fit_cv

    def failing(self: JudgeCalibrator, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("sample_weight") is not None:
            raise ValueError("boom")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(JudgeCalibrator, "fit_cv", failing)
    with pytest.raises(RuntimeError, match=r"replicate 0 failed for judge 'a'.*boom"):
        compare_judges({"a": sharp, "b": noisy}, labels, None, n_bootstrap=3)


def test_judges_on_different_folds_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    sharp, noisy, labels = _small()
    original = JudgeCalibrator.fit_cv
    calls = {"n": 0}

    def shifted(self: JudgeCalibrator, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 2:
            assert result.fold_ids is not None
            result.fold_ids = (result.fold_ids + 1) % self.n_folds
        return result

    monkeypatch.setattr(JudgeCalibrator, "fit_cv", shifted)
    with pytest.raises(RuntimeError, match="different cross-fitting folds"):
        compare_judges({"a": sharp, "b": noisy}, labels, None, n_bootstrap=2)


# ---------------------------------------------------------------------------
# Options, inputs and outputs
# ---------------------------------------------------------------------------


def test_covariates_force_two_stage() -> None:
    sharp, noisy, labels = _small()
    rng = np.random.default_rng(2)
    x = rng.normal(size=(len(sharp), 2))
    cmp = compare_judges(
        {"a": sharp, "b": noisy}, labels, None, covariates=x, n_bootstrap=3
    )
    for name in ("a", "b"):
        assert cmp.row(name).selected_mode == "two_stage"
        assert cmp.row(name).covariates_used is True
    assert cmp.diagnostics["bootstrap"]["refit_modes"] == {
        "a": "two_stage",
        "b": "two_stage",
    }
    with pytest.raises(ValueError, match="covariates must be"):
        compare_judges({"a": sharp}, labels, None, covariates=x[:-1], n_bootstrap=2)
    bad = x.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        compare_judges({"a": sharp}, labels, None, covariates=bad, n_bootstrap=2)


def test_input_validation() -> None:
    sharp, noisy, labels = _small()
    ok = {"a": sharp, "b": noisy}

    def call(**overrides: Any) -> Any:
        kwargs: Dict[str, Any] = {
            "scores_by_judge": ok,
            "oracle_labels": labels,
            "cluster_ids": None,
            "n_bootstrap": 2,
        }
        kwargs.update(overrides)
        return compare_judges(**kwargs)

    nan_scores = noisy.copy()
    nan_scores[:3] = np.nan
    with pytest.raises(ValueError, match=r"Judge 'b' has 3 non-finite score"):
        call(scores_by_judge={"a": sharp, "b": nan_scores})
    with pytest.raises(ValueError, match="every judge must score the same rows"):
        call(scores_by_judge={"a": sharp, "b": noisy[:-1]})
    with pytest.raises(ValueError, match="non-empty mapping"):
        call(scores_by_judge={})
    with pytest.raises(ValueError, match="non-empty strings"):
        call(scores_by_judge={1: sharp})
    with pytest.raises(ValueError, match="1-D"):
        call(scores_by_judge={"a": sharp.reshape(-1, 2)})
    with pytest.raises(ValueError, match="oracle_labels must have shape"):
        call(oracle_labels=labels[:-1])
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        call(oracle_labels=np.where(np.isnan(labels), np.nan, labels * 5))
    inf_labels = labels.copy()
    inf_labels[np.flatnonzero(~np.isnan(labels))[0]] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        call(oracle_labels=inf_labels)
    with pytest.raises(ValueError, match="No labelled rows"):
        call(oracle_labels=np.full(len(labels), np.nan))
    with pytest.raises(ValueError, match="vary within no policy"):
        call(oracle_labels=np.where(np.isnan(labels), np.nan, 0.5))
    with pytest.raises(ValueError, match="cluster_ids length"):
        call(cluster_ids=np.arange(len(labels) - 1))
    with pytest.raises(ValueError, match="policy_ids length"):
        call(policy_ids=np.zeros(len(labels) - 1))
    with pytest.raises(ValueError, match="reference 'z'"):
        call(reference="z")
    with pytest.raises(ValueError, match="n_unlabeled must be non-negative"):
        call(n_unlabeled=-1)
    with pytest.raises(TypeError, match="n_unlabeled must be an integer"):
        call(n_unlabeled=1.5)
    with pytest.raises(TypeError, match="n_unlabeled must be an integer"):
        call(n_unlabeled=True)
    with pytest.raises(ValueError, match="n_bootstrap must be at least 2"):
        call(n_bootstrap=1)
    with pytest.raises(TypeError, match="n_bootstrap must be an integer"):
        call(n_bootstrap=2.0)
    with pytest.raises(ValueError, match="n_folds must be at least 2"):
        call(n_folds=1)
    for alpha in (0.0, 1.0, True):
        with pytest.raises(ValueError, match="alpha must be in"):
            call(alpha=alpha)


def test_warn_below_twenty_labelled_clusters() -> None:
    sharp, noisy, _ = _small()
    rng = np.random.default_rng(4)
    q = sharp
    y = np.clip(_sigmoid(q) + rng.normal(0, 0.1, len(q)), 0, 1)
    labels = np.full(len(q), np.nan)
    idx = rng.choice(len(q), 12, replace=False)
    labels[idx] = y[idx]
    with pytest.warns(UserWarning, match="Only 12 labelled prompt clusters") as rec:
        compare_judges({"a": sharp, "b": noisy}, labels, None, n_bootstrap=2)
    assert sum("labelled prompt clusters" in str(w.message) for w in rec) == 1
    _, _, labels = _small()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        compare_judges({"a": sharp, "b": noisy}, labels, None, n_bootstrap=2)


def test_single_judge_and_reference_choice() -> None:
    sharp, noisy, labels = _small()
    single = compare_judges({"only": sharp}, labels, None, n_bootstrap=3)
    assert [q.judge for q in single.table] == ["only"] and single.pairwise == []
    assert "only" in single.summary()
    flipped = compare_judges(
        {"a": sharp, "b": noisy}, labels, None, reference="b", n_bootstrap=3
    )
    assert flipped.reference == "b"
    assert [p.judge for p in flipped.pairwise] == ["a"]
    with pytest.raises(KeyError):
        flipped.pair("b")
    with pytest.raises(KeyError):
        flipped.row("zzz")


def test_to_dict_json_safe_and_summary_wording() -> None:
    sharp, noisy, labels = _small()
    cmp = compare_judges(
        {"sharp": sharp, "noisy": noisy},
        labels,
        None,
        reference="noisy",
        n_unlabeled=500,
        n_bootstrap=4,
    )
    payload = cmp.to_dict()
    text = json.dumps(payload, allow_nan=False)
    assert "calibrators" not in payload
    assert isinstance(payload["table"][0]["r2_within_ci"], list)
    assert json.loads(text)["pairwise"][0]["judge"] == "sharp"
    summary = cmp.summary()
    assert "noisy (reference) needs" in summary
    assert "labels per label of sharp" in summary
    assert "plentiful unlabelled rows" in summary
    assert "at 500 unlabelled rows per policy" in summary
    assert cje.compare_judges is compare_judges
    assert {"compare_judges", "JudgeComparison", "JudgeQuality", "JudgePair"} <= set(
        cje.__all__
    )


def test_calibrators_usable_for_transport_audit() -> None:
    sharp, noisy, labels = _small(n=800, n_lab=300, seed=9)
    cmp = compare_judges({"sharp": sharp, "noisy": noisy}, labels, None, n_bootstrap=2)
    rng = np.random.default_rng(10)
    q = rng.normal(size=200)
    probe_scores = q + rng.normal(0, 0.4, 200)
    probe_labels = np.clip(_sigmoid(1.5 * q) + rng.normal(0, 0.08, 200), 0, 1)
    audit = transport_audit(
        probe_scores, probe_labels, cmp.calibrators["sharp"], delta_max=0.05
    )
    assert audit.status in ("PASS", "FAIL", "INCONCLUSIVE")


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_end_to_end_clustered_policies_with_scales() -> None:
    """Two policies on shared prompts, three draws per prompt, whole prompts
    labelled, a 1-10 judge next to a [0, 1] judge, planned finite N."""
    rng = np.random.default_rng(21)
    n_prompts, draws = 200, 3
    rows = [
        (name, f"prompt{p}", offset)
        for name, offset in (("base", 0.0), ("new", 0.4))
        for p in range(n_prompts)
        for _ in range(draws)
    ]
    policy = np.asarray([r[0] for r in rows])
    prompt = np.asarray([r[1] for r in rows])
    shift = np.asarray([r[2] for r in rows], dtype=float)
    prompt_effect = {f"prompt{p}": rng.normal(0, 0.7) for p in range(n_prompts)}
    q = (
        shift
        + np.asarray([prompt_effect[p] for p in prompt])
        + rng.normal(0, 0.7, len(rows))
    )
    y = (rng.uniform(size=len(rows)) < _sigmoid(2 * q)).astype(float)
    sharp = _sigmoid(q + rng.normal(0, 0.35, len(rows)))  # [0, 1]
    likert = np.clip(np.round(5.5 + 2.0 * (q + rng.normal(0, 1.3, len(rows)))), 1, 10)
    labelled_prompts = {f"prompt{p}" for p in rng.choice(n_prompts, 60, replace=False)}
    labels = np.where(np.isin(prompt, sorted(labelled_prompts)), y, np.nan)

    cmp = compare_judges(
        {"likert": likert, "sharp": sharp},
        labels,
        prompt,
        policy_ids=policy,
        judge_scales={"likert": (1, 10), "sharp": (0, 1)},
        n_unlabeled=2000,
        n_bootstrap=40,
    )
    assert cmp.reference == "likert"
    likert_row, sharp_row = cmp.row("likert"), cmp.row("sharp")
    # 60 labelled prompts x 3 draws x 2 policies; prompts are shared by policies.
    assert sharp_row.n_labelled_rows == 360
    assert sharp_row.n_labelled_clusters == 60
    assert cmp.diagnostics["n_clusters"] == n_prompts
    assert cmp.diagnostics["policies"] == ["base", "new"]
    assert cmp.diagnostics["planned"]["labelled_rows_per_policy"] == 180.0
    mask = ~np.isnan(labels)
    yl, pl = y[mask], policy[mask]
    var_y_within = sum(
        np.sum((yl[pl == p] - yl[pl == p].mean()) ** 2) for p in ("base", "new")
    ) / int(mask.sum())
    for row in (likert_row, sharp_row):
        for field in ("oof_rmse", "r2_within", "r2_pooled"):
            lo, hi = getattr(row, f"{field}_ci")
            assert lo <= hi
        assert row.var_residual == pytest.approx(
            (1 - row.r2_within) * var_y_within, rel=1e-10
        )
    pair = cmp.pair("sharp")
    assert sharp_row.r2_within > likert_row.r2_within
    assert pair.r2_within_diff_ci[0] > 0
    assert pair.oof_rmse_diff_ci[1] < 0
    assert pair.label_multiplier_ci[0] > 1
    assert pair.label_multiplier == pytest.approx(
        (1 - likert_row.r2_within) / (1 - sharp_row.r2_within), rel=1e-12
    )
    assert pair.label_multiplier_at_n is not None
    assert 1 < pair.label_multiplier_at_n < pair.label_multiplier
    assert pair.variance_ratio_at_n_ci is not None
    assert pair.variance_ratio_at_n_ci[0] <= pair.variance_ratio_at_n_ci[1]
    json.dumps(cmp.to_dict(), allow_nan=False)
