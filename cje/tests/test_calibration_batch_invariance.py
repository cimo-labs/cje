"""Two-stage ranks must not depend on how the same prediction rows are batched."""

from typing import Optional

import numpy as np
import pytest
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from cje.calibration.flexible_calibrator import _predict_spline_ridge
from cje.calibration.judge import JudgeCalibrator


@pytest.mark.parametrize("n_features", [1, 3])
def test_smooth_predictions_preserve_fit_and_are_identical_across_batches(
    n_features: int,
) -> None:
    rng = np.random.default_rng(11)
    X = rng.uniform(size=(40, n_features))
    labels = rng.uniform(size=40)
    model = make_pipeline(
        SplineTransformer(n_knots=8, include_bias=False), RidgeCV()
    ).fit(X, labels)
    coefficients = model[-1].coef_.copy()
    queries = np.tile(X, (7, 1))
    expected = _predict_spline_ridge(model, queries)
    # This is the same fitted linear map, with only the reduction order fixed.
    np.testing.assert_allclose(expected, model.predict(queries), rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(model[-1].coef_, coefficients)
    for batch_size in [1, 7, 40, 171]:
        batched = np.concatenate(
            [
                _predict_spline_ridge(model, queries[i : i + batch_size])
                for i in range(0, len(queries), batch_size)
            ]
        )
        np.testing.assert_array_equal(batched, expected)
    order = rng.permutation(len(queries))
    np.testing.assert_array_equal(
        _predict_spline_ridge(model, np.asfortranarray(queries[order])), expected[order]
    )


@pytest.mark.parametrize("with_covariates", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_public_full_and_oof_predictions_are_batch_invariant(
    with_covariates: bool, weighted: bool
) -> None:
    rng = np.random.default_rng(7)
    scores = np.r_[rng.uniform(0.05, 0.95, 35), np.ones(5)]
    labels = np.clip(
        0.4 + 0.3 * np.sin(9 * scores) + rng.normal(0, 0.1, len(scores)), 0, 1
    )
    covariates: Optional[np.ndarray] = (
        rng.uniform(size=(40, 2)) if with_covariates else None
    )
    cal = JudgeCalibrator(
        calibration_mode="two_stage",
        covariate_names=["a", "b"] if with_covariates else None,
    )
    cal.fit_cv(
        scores,
        labels,
        n_folds=5,
        covariates=covariates,
        sample_weight=rng.uniform(0.5, 3, 40) if weighted else None,
    )
    # Include training knots, repeated scores, unseen values and boundary scores.
    query = np.r_[scores, scores, rng.uniform(0, 1, 13)]
    cov = (
        np.vstack([covariates, covariates, rng.uniform(size=(13, 2))])
        if covariates is not None
        else None
    )
    folds = np.arange(len(query)) % 5
    full = cal.predict(query, covariates=cov)
    oof = cal.predict_oof(query, folds, covariates=cov)
    for size in [1, 3, 17, 40]:
        pieces = [slice(i, i + size) for i in range(0, len(query), size)]
        split = np.concatenate(
            [
                cal.predict(query[p], covariates=cov[p] if cov is not None else None)
                for p in pieces
            ]
        )
        split_oof = np.concatenate(
            [
                cal.predict_oof(
                    query[p], folds[p], covariates=cov[p] if cov is not None else None
                )
                for p in pieces
            ]
        )
        np.testing.assert_array_equal(split, full)
        np.testing.assert_array_equal(split_oof, oof)
    order = rng.permutation(len(query))
    np.testing.assert_array_equal(
        cal.predict(query[order], covariates=cov[order] if cov is not None else None),
        full[order],
    )
    np.testing.assert_array_equal(
        cal.predict_oof(
            query[order],
            folds[order],
            covariates=cov[order] if cov is not None else None,
        ),
        oof[order],
    )


def test_two_stage_fallbacks_keep_the_same_batch_contract() -> None:
    rng = np.random.default_rng(7)
    scores = np.r_[rng.uniform(size=45), np.ones(5)]
    cal = JudgeCalibrator(calibration_mode="two_stage")
    cal.fit_cv(scores, rng.uniform(size=50), n_folds=5)
    flex = cal._flexible_calibrator
    assert flex is not None
    query = np.tile(scores, 3)
    # The private fallback for an absent fold uses the full pipeline.
    expected = flex.predict(query, folds=np.full(len(query), 99))
    singleton = np.concatenate(
        [flex.predict(np.asarray([q]), folds=np.asarray([99])) for q in query]
    )
    np.testing.assert_array_equal(expected, singleton)
    # When the full pipeline is absent, prediction uses the fold ensemble.
    flex._full_g_model = None
    expected = flex.predict(query)
    singleton = np.concatenate([flex.predict(np.asarray([q])) for q in query])
    np.testing.assert_array_equal(expected, singleton)
