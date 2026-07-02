"""Regression tests: OUA jackknife predictions must route through predict_oof.

Two-stage calibrators are a pipeline g(S[, X]) -> ECDF rank -> isotonic. The
raw per-fold isotonic models exposed by get_fold_models_for_oua() therefore
expect the RANK INDEX in [0, 1], not the judge score. A prior implementation
fed them raw judge scores in every no-covariate jackknife path, producing
garbage leave-one-fold-out replicates whenever two_stage was active (it is
auto-selectable by default). All jackknife paths now route through
reward_calibrator.predict_oof(scores, fold_ids[, covariates]), which applies
the mode-appropriate transform.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from cje.calibration.judge import JudgeCalibrator
from cje.estimators.calibrated_ips import CalibratedIPS


def _fit_two_stage_calibrator(
    n: int = 800, n_oracle: int = 200, seed: int = 13
) -> tuple:
    rng = np.random.default_rng(seed)
    # Skewed scores so the ECDF rank transform differs materially from identity
    judge_scores = rng.beta(5, 2, size=n)
    truth = np.clip(judge_scores**2 + rng.normal(0, 0.05, size=n), 0, 1)
    oracle_mask = np.zeros(n, dtype=bool)
    oracle_mask[rng.choice(n, size=n_oracle, replace=False)] = True
    # fit_cv expects the labeled-subset array plus a full-length mask
    oracle_labels = truth[oracle_mask]

    cal = JudgeCalibrator(calibration_mode="two_stage")
    cal.fit_cv(judge_scores, oracle_labels, oracle_mask, n_folds=5)
    return cal, judge_scores


def test_raw_fold_models_expect_rank_index_not_scores() -> None:
    """Documents the trap: raw two-stage fold models are NOT score->reward maps."""
    cal, judge_scores = _fit_two_stage_calibrator()
    fold_models = cal.get_fold_models_for_oua()
    assert fold_models, "two_stage calibrator should expose fold models"

    diffs = []
    for fold_id, fold_model in fold_models.items():
        raw = np.clip(fold_model.predict(judge_scores), 0.0, 1.0)
        routed = np.clip(
            cal.predict_oof(
                judge_scores, np.full(len(judge_scores), fold_id, dtype=int)
            ),
            0.0,
            1.0,
        )
        diffs.append(float(np.mean(np.abs(raw - routed))))
    # If these coincided, feeding raw scores would have been harmless.
    assert max(diffs) > 0.01, (
        "Expected raw fold-model predictions on judge scores to differ from "
        f"predict_oof routing for two_stage mode; got max mean|diff|={max(diffs):.4f}"
    )


class _SpyCalibrator:
    """Calibrator double that records how jackknife predictions are requested."""

    covariate_names: List[str] = []

    def __init__(self, n_folds: int = 3):
        self._n_folds = n_folds
        self.predict_oof_calls: List[np.ndarray] = []

    def get_fold_models_for_oua(self) -> Dict[int, Any]:
        # Raw fold models whose direct use would poison the jackknife: they
        # return a sentinel so any fold_model.predict() path is detectable.
        class _Poison:
            def predict(self, x: np.ndarray) -> np.ndarray:
                return np.full(len(x), -1000.0)

        return {k: _Poison() for k in range(self._n_folds)}

    def predict_oof(
        self,
        judge_scores: np.ndarray,
        fold_ids: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        assert covariates is None
        assert len(set(fold_ids.tolist())) == 1, "constant fold id per replicate"
        self.predict_oof_calls.append(fold_ids.copy())
        k = int(fold_ids[0])
        # Distinct, known value per fold
        return np.full(len(judge_scores), 0.1 * (k + 1))


class _StubSampler:
    def __init__(self, n: int):
        self._n = n

    def get_data_for_policy(self, policy: str) -> List[Dict[str, Any]]:
        return [{"judge_score": 0.5} for _ in range(self._n)]


class _StubIPS(CalibratedIPS):
    """CalibratedIPS with the machinery bypassed except get_oracle_jackknife."""

    def __init__(self, n: int, calibrator: Any):
        # Deliberately skip CalibratedIPS.__init__; set only what
        # get_oracle_jackknife touches.
        self.reward_calibrator = calibrator
        self.sampler = _StubSampler(n)  # type: ignore[assignment]
        self.oua_jackknife = True
        self._weights = np.ones(n)

    def get_weights(self, policy: str) -> np.ndarray:
        return self._weights


def test_calibrated_ips_jackknife_routes_through_predict_oof() -> None:
    n = 40
    spy = _SpyCalibrator(n_folds=3)
    est = _StubIPS(n, spy)

    jack = est.get_oracle_jackknife("policy_a")

    assert jack is not None
    assert len(spy.predict_oof_calls) == 3, "one predict_oof call per fold"
    # Hajek weights of 1 => replicate k equals the constant OOF prediction.
    np.testing.assert_allclose(np.sort(jack), [0.1, 0.2, 0.3], rtol=1e-12)
    # Nothing may come from the raw fold models (sentinel would clip to 0.0).
    assert not np.any(jack == 0.0)
