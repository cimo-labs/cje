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

import numpy as np

from cje.calibration.judge import JudgeCalibrator


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
