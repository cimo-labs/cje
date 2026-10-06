"""JudgeCalibrator.fit_cv input validation and quiet logging.

Labels outside [0, 1] used to be clipped by the bounded isotonic fits, so a
calibrator trained on a 0-500 amount predicted 1.0 everywhere without an error.
"""

import logging
from typing import Any

import numpy as np
import pytest

from cje.calibration import JudgeCalibrator
from cje.calibration.flexible_calibrator import FlexibleCalibrator

N = 400


def _scores(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 1, N)


@pytest.mark.parametrize("mode", ["monotone", "two_stage", "auto"])
def test_fit_cv_rejects_labels_in_original_units(mode: Any) -> None:
    rng = np.random.default_rng(1)
    S = _scores()
    Y = np.clip(500 * S + rng.normal(0, 40, N), 0, 500)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        JudgeCalibrator(calibration_mode=mode).fit_cv(S, Y, n_folds=5, quiet=True)


def test_fit_cv_rejects_labels_centred_on_zero() -> None:
    rng = np.random.default_rng(2)
    S = _scores()
    Z = S - 0.5 + rng.normal(0, 0.1, N)
    with pytest.raises(ValueError, match="analyze_dataset"):
        JudgeCalibrator(calibration_mode="monotone").fit_cv(S, Z, quiet=True)


def test_fit_cv_rejects_non_finite_labels_clearly() -> None:
    S = _scores()
    Y = S.copy()
    Y[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        JudgeCalibrator(calibration_mode="monotone").fit_cv(S, Y, quiet=True)


def test_fit_cv_checks_only_the_labelled_rows() -> None:
    # Partial labelling with a boolean mask: only the labelled values matter.
    S = _scores()
    mask = np.zeros(N, dtype=bool)
    mask[:200] = True
    Y = np.clip(S[:200] + np.random.default_rng(3).normal(0, 0.05, 200), 0, 1)
    result = JudgeCalibrator(calibration_mode="monotone").fit_cv(
        S, Y, oracle_mask=mask, quiet=True
    )
    assert np.all((result.calibrated_scores >= 0) & (result.calibrated_scores <= 1))


def test_flexible_calibrator_fit_rejects_out_of_range_labels() -> None:
    S = _scores()
    folds = np.arange(N) % 5
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        FlexibleCalibrator(mode="monotone").fit(S, S * 5, folds)


@pytest.mark.parametrize("covariate_names", [None, ["length", "platform"]])
def test_fit_cv_quiet_auto_with_covariates_logs_nothing_at_info(
    caplog: pytest.LogCaptureFixture, covariate_names: list[str] | None
) -> None:
    rng = np.random.default_rng(4)
    S = _scores()
    B = np.clip(np.round(S + rng.normal(0, 0.3, N)), 0, 1)
    X = np.column_stack([rng.integers(1, 30, N), rng.integers(0, 2, N)]).astype(float)

    with caplog.at_level(logging.DEBUG, logger="cje"):
        for _ in range(3):
            idx = rng.integers(0, N, N)
            JudgeCalibrator(
                calibration_mode="auto", covariate_names=covariate_names
            ).fit_cv(S[idx], B[idx], covariates=X[idx], quiet=True)

    loud = [r for r in caplog.records if r.levelno >= logging.INFO]
    assert loud == [], [r.getMessage() for r in loud]


def test_fit_cv_not_quiet_still_reports_auto_mode_forcing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rng = np.random.default_rng(5)
    S = _scores()
    B = np.clip(np.round(S + rng.normal(0, 0.3, N)), 0, 1)
    X = rng.integers(1, 30, (N, 1)).astype(float)

    with caplog.at_level(logging.INFO, logger="cje"):
        JudgeCalibrator(calibration_mode="auto", covariate_names=["length"]).fit_cv(
            S, B, covariates=X
        )

    assert any("forcing two_stage" in r.getMessage() for r in caplog.records)
