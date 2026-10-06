"""Calibrator input checks and covariate reporting (0.9.1).

- #68: `JudgeCalibrator.fit_cv`, `FlexibleCalibrator.fit` and
  `calibrated_mean_ci` share one oracle-label check (see also
  test_fit_cv_label_validation.py), so all three refuse non-finite labels and
  labels outside [0, 1] (they used to be clipped silently) with the same
  message; routine covariate and mode-selection messages respect
  ``quiet=True``.
- #67: when covariates are given but too few labelled rows let the full model
  or a cross-fitting fold use them, CJE warns, reports ``selected_mode``
  "monotone" for a full-model fallback, and records ``covariates_used`` and
  ``n_folds_without_covariates``. Estimates are unchanged: only the labels,
  diagnostics and warnings are new.
- #66: `calibrated_mean_ci` warns when covariates are passed at complete
  label coverage, where they are ignored.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from cje import analyze_dataset, calibrated_mean_ci
from cje.calibration import JudgeCalibrator, calibrate_dataset
from cje.calibration.flexible_calibrator import TWO_STAGE_MIN_ROWS, FlexibleCalibrator
from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample
from cje.data.models import Dataset, Sample
from cje.estimators.direct_method import CalibratedDirectEstimator

FALLBACK_MESSAGE = "Covariates were supplied but"


def _scores_and_covariates(n: int = 400, seed: int = 0) -> Tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0, 1, n)
    covariates = np.column_stack(
        [rng.integers(1, 30, n), rng.integers(0, 2, n)]
    ).astype(float)
    outcome = np.clip(np.round(scores + rng.normal(0, 0.3, n)), 0, 1)
    return scores, covariates, outcome


def _fallback_rows(n_labelled: int, seed: int = 3) -> Tuple[np.ndarray, ...]:
    """The #67 reproduction: one binary covariate that drives the outcome."""
    rng = np.random.default_rng(seed)
    n = 3000
    x = rng.integers(0, 2, n).astype(float)
    scores = rng.binomial(1, 0.5, n).astype(float)
    outcome = rng.binomial(1, 0.1 + 0.3 * scores + 0.4 * x).astype(float)
    labels = np.full(n, np.nan)
    labelled = rng.choice(n, n_labelled, replace=False)
    labels[labelled] = outcome[labelled]
    return scores, labels, x[:, None]


def _fallback_warnings(caught: List[warnings.WarningMessage]) -> List[str]:
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, UserWarning) and FALLBACK_MESSAGE in str(w.message)
    ]


# ---------------------------------------------------------------------------
# #68: oracle labels outside [0, 1] or non-finite
# ---------------------------------------------------------------------------


class TestOracleLabelRange:
    """#68: labels in original units used to be clipped into a calibrator that
    predicts 1.0 almost everywhere (or floored at 0) without any error."""

    @pytest.mark.parametrize(
        "mode,with_covariates",
        [("monotone", False), ("two_stage", True), ("auto", False), ("auto", True)],
    )
    def test_labels_on_0_to_500_raise(self, mode: str, with_covariates: bool) -> None:
        """#68: a capped amount on 0-500 raises instead of collapsing to 1.0."""
        scores, covariates, _ = _scores_and_covariates()
        rng = np.random.default_rng(1)
        amounts = np.clip(500 * scores + rng.normal(0, 40, len(scores)), 0, 500)
        calibrator = JudgeCalibrator(
            calibration_mode=mode,  # type: ignore[arg-type]
            covariate_names=["length", "platform"] if with_covariates else None,
        )
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]") as excinfo:
            calibrator.fit_cv(
                scores,
                amounts,
                covariates=covariates if with_covariates else None,
                quiet=True,
            )
        message = str(excinfo.value)
        assert "(y - lo) / (hi - lo)" in message
        assert "analyze_dataset" in message
        assert "got values in [0, 500]" in message

    def test_labels_centred_on_zero_raise(self) -> None:
        """#68: labels centred on 0 used to be floored at 0 without a message."""
        scores, _, _ = _scores_and_covariates()
        rng = np.random.default_rng(2)
        centred = scores - 0.5 + rng.normal(0, 0.1, len(scores))
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
            JudgeCalibrator(calibration_mode="monotone").fit_cv(
                scores, centred, quiet=True
            )

    def test_likert_labels_raise_with_partial_mask(self) -> None:
        """#68: the check applies to the resolved labels of a boolean mask."""
        scores, _, _ = _scores_and_covariates()
        mask = np.zeros(len(scores), dtype=bool)
        mask[::4] = True
        likert = np.round(1 + 4 * scores[mask])
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
            JudgeCalibrator().fit_cv(scores, likert, mask, quiet=True)

    def test_flexible_calibrator_fit_raises(self) -> None:
        """#68: FlexibleCalibrator.fit carries the same check."""
        scores, _, _ = _scores_and_covariates(n=100)
        folds = np.arange(100) % 5
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
            FlexibleCalibrator(mode="monotone").fit(scores, 5 * scores, folds)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_labels_raise_clear_message(self, bad: float) -> None:
        """#68: NaN used to fail only with sklearn's 'Input y contains NaN'."""
        scores, _, outcome = _scores_and_covariates(n=100)
        labels = outcome.copy()
        labels[[3, 7]] = bad
        with pytest.raises(ValueError, match="must be finite"):
            JudgeCalibrator(calibration_mode="monotone").fit_cv(
                scores, labels, quiet=True
            )
        with pytest.raises(ValueError, match="must be finite"):
            FlexibleCalibrator(mode="monotone").fit(scores, labels, np.arange(100) % 5)

    def test_message_matches_calibrated_mean_ci(self) -> None:
        """#68: fit_cv and calibrated_mean_ci reject the same labels identically."""
        scores, _, _ = _scores_and_covariates(n=200)
        mask = np.zeros(200, dtype=bool)
        mask[:60] = True
        labels = np.full(200, np.nan)
        labels[mask] = 100 * scores[mask]
        with pytest.raises(ValueError) as from_array_api:
            calibrated_mean_ci(scores, labels)
        with pytest.raises(ValueError) as from_fit_cv:
            JudgeCalibrator().fit_cv(scores, labels[mask], mask, quiet=True)
        assert str(from_array_api.value) == str(from_fit_cv.value)

    def test_valid_labels_unchanged(self) -> None:
        """#68: boundary values 0 and 1 and integer labels still fit, and an
        integer copy of the labels fits exactly like the float labels."""
        scores, _, outcome = _scores_and_covariates(n=200)
        assert set(np.unique(outcome)) == {0.0, 1.0}
        as_float = JudgeCalibrator(calibration_mode="monotone").fit_cv(
            scores, outcome, quiet=True
        )
        as_int = JudgeCalibrator(calibration_mode="monotone").fit_cv(
            scores, outcome.astype(int), quiet=True
        )
        np.testing.assert_array_equal(
            as_float.calibrated_scores, as_int.calibrated_scores
        )
        assert as_float.oof_rmse == as_int.oof_rmse

    def test_internal_callers_with_normalised_labels_still_work(self) -> None:
        """#68: analyze_dataset normalises 0-100 labels before calibrating, so
        the new check never fires on its path."""
        rng = np.random.default_rng(4)
        records = []
        for i in range(120):
            score = float(rng.uniform(0, 100))
            record: Dict[str, Any] = {"prompt_id": f"p{i}", "judge_score": score}
            if i % 3 == 0:
                record["oracle_label"] = float(
                    np.clip(score + rng.normal(0, 10), 0, 100)
                )
            records.append(record)
        results = analyze_dataset(
            fresh_draws_data={"policy_a": records},
            fresh_judge_scale=(0, 100),
            fresh_oracle_scale=(0, 100),
        )
        assert results.metadata["calibration_status"] == "CALIBRATED"
        assert 0.0 <= results.estimates[0] <= 100.0


# ---------------------------------------------------------------------------
# #68: quiet=True silences routine covariate and mode messages
# ---------------------------------------------------------------------------


class TestQuietLogging:
    """#68: inside a user-written bootstrap loop, every fit_cv call logged a
    WARNING about missing covariate names and an INFO about auto mode."""

    @pytest.mark.parametrize("names", [None, ["length", "platform"]])
    def test_auto_mode_with_covariates_is_silent_when_quiet(
        self, names: Optional[List[str]], caplog: pytest.LogCaptureFixture
    ) -> None:
        """#68: nothing at INFO or above with quiet=True, with or without names."""
        scores, covariates, outcome = _scores_and_covariates()
        rng = np.random.default_rng(5)
        with caplog.at_level(logging.DEBUG, logger="cje"):
            for _ in range(3):
                idx = rng.integers(0, len(scores), len(scores))
                JudgeCalibrator(calibration_mode="auto", covariate_names=names).fit_cv(
                    scores[idx], outcome[idx], covariates=covariates[idx], quiet=True
                )
        loud = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
        assert loud == []
        # The routine messages are still available at DEBUG.
        debug = " ".join(r.getMessage() for r in caplog.records)
        assert "forcing two_stage" in debug
        if names is None:
            assert "no covariate_names specified" in debug

    def test_auto_mode_without_covariates_is_silent_when_quiet(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """#68: the mode-selection summary follows quiet too."""
        scores, _, outcome = _scores_and_covariates()
        with caplog.at_level(logging.DEBUG, logger="cje"):
            JudgeCalibrator(calibration_mode="auto").fit_cv(scores, outcome, quiet=True)
        assert [r for r in caplog.records if r.levelno >= logging.INFO] == []
        assert any(
            "Calibration mode selection" in r.getMessage() for r in caplog.records
        )

    def test_routine_messages_still_log_at_info_without_quiet(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """#68: quiet=False keeps the auto-mode message at INFO."""
        scores, covariates, outcome = _scores_and_covariates()
        with caplog.at_level(logging.INFO, logger="cje"):
            JudgeCalibrator(
                calibration_mode="auto", covariate_names=["length", "platform"]
            ).fit_cv(scores, outcome, covariates=covariates)
        info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any("forcing two_stage" in message for message in info)


# ---------------------------------------------------------------------------
# #67: covariates dropped below 20 labelled rows
# ---------------------------------------------------------------------------


class TestCovariateFallbackReporting:
    """#67: covariates were dropped silently while diagnostics said two_stage."""

    def test_threshold_is_twenty_labelled_rows(self) -> None:
        """#67: the documented threshold is the one the fit uses."""
        assert TWO_STAGE_MIN_ROWS == 20

    def test_full_model_fallback_at_18_labels(self) -> None:
        """#67: at 18 labels neither the full model nor any fold can use the
        covariates; selected_mode reports monotone and a UserWarning fires."""
        scores, labels, x = _fallback_rows(18)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calibrated_mean_ci(scores, labels, covariates=x)
        messages = _fallback_warnings(caught)
        assert len(messages) == 1
        assert "only 18 labelled rows" in messages[0]
        assert "selected_mode 'monotone'" in messages[0]

        calibration = result.diagnostics["calibration"]
        assert calibration["mode"] == "two_stage"
        assert calibration["selected_mode"] == "monotone"
        assert calibration["covariates_used"] is False
        assert result.calibrator is not None
        n_folds = result.calibrator.n_folds
        assert calibration["n_folds_without_covariates"] == n_folds == 5

        # Callers that always pass the fitted covariates keep working, and the
        # rewards do not vary with the covariate within a judge level.
        rewards = result.calibrator.predict(scores, covariates=x)
        for level in (0.0, 1.0):
            assert len(np.unique(rewards[scores == level].round(6))) == 1
        assert result.diagnostics["cluster_robust"]["oracle_jackknife_folds"] == 5

    def test_fold_fallback_at_22_labels(self) -> None:
        """#67: at 22 labels the full model uses the covariates but every fold
        complement has fewer than 20 rows; the warning and count say so."""
        scores, labels, x = _fallback_rows(22)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calibrated_mean_ci(scores, labels, covariates=x)
        messages = _fallback_warnings(caught)
        assert len(messages) == 1
        assert "5 of 5 cross-fitting folds" in messages[0]
        assert "about 25 labelled rows" in messages[0]

        calibration = result.diagnostics["calibration"]
        assert calibration["selected_mode"] == "two_stage"
        assert calibration["covariates_used"] is True
        assert calibration["n_folds_without_covariates"] == 5

    def test_no_fallback_at_200_labels(self) -> None:
        """#67: with enough labels nothing is dropped and nothing warns."""
        scores, labels, x = _fallback_rows(200)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calibrated_mean_ci(scores, labels, covariates=x)
        assert _fallback_warnings(caught) == []
        calibration = result.diagnostics["calibration"]
        assert calibration["selected_mode"] == "two_stage"
        assert calibration["covariates_used"] is True
        assert calibration["n_folds_without_covariates"] == 0

    def test_without_covariates_fields_report_nothing_dropped(self) -> None:
        """#67: the fields are always present; no covariates means none used."""
        scores, labels, _ = _fallback_rows(200)
        calibration = calibrated_mean_ci(scores, labels).diagnostics["calibration"]
        assert calibration["covariates_used"] is False
        assert calibration["n_folds_without_covariates"] == 0

    def test_fallback_fit_is_the_judge_only_fit(self) -> None:
        """#67: the relabelled fallback keeps the models it always fitted, so
        estimates are unchanged: it predicts exactly like the same two-stage
        calibrator fitted without covariates."""
        scores, covariates, outcome = _scores_and_covariates(n=60, seed=8)
        labelled = np.zeros(60, dtype=bool)
        labelled[:18] = True
        with_cov = JudgeCalibrator(
            calibration_mode="two_stage", covariate_names=["length", "platform"]
        )
        with pytest.warns(UserWarning, match=FALLBACK_MESSAGE):
            fit_with = with_cov.fit_cv(
                scores, outcome[labelled], labelled, covariates=covariates
            )
        without_cov = JudgeCalibrator(calibration_mode="two_stage")
        fit_without = without_cov.fit_cv(scores, outcome[labelled], labelled)

        assert with_cov.selected_mode == "monotone"
        assert without_cov.selected_mode == "two_stage"
        np.testing.assert_array_equal(
            fit_with.calibrated_scores, fit_without.calibrated_scores
        )
        assert fit_with.oof_rmse == fit_without.oof_rmse
        folds = np.asarray(fit_with.fold_ids)[labelled]
        np.testing.assert_array_equal(
            with_cov.predict_oof(scores[labelled], folds, covariates[labelled]),
            without_cov.predict_oof(scores[labelled], folds),
        )
        assert sorted(with_cov.get_fold_models_for_oua()) == [0, 1, 2, 3, 4]

    def test_monotone_fit_without_covariates_still_refuses_them(self) -> None:
        """#67: only a fit that dropped its covariates accepts them at predict."""
        scores, _, outcome = _scores_and_covariates(n=100)
        calibrator = JudgeCalibrator(calibration_mode="monotone")
        calibrator.fit_cv(scores, outcome, quiet=True)
        with pytest.raises(ValueError, match="monotone mode"):
            calibrator.predict(scores, covariates=np.ones((100, 1)))

    def test_bootstrap_refits_after_full_model_fallback(self) -> None:
        """#67: a monotone refit cannot take covariates, so the bootstrap keeps
        refitting two-stage (each replicate falls back the same way)."""
        scores, labels, x = _fallback_rows(18)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calibrated_mean_ci(
                scores, labels, covariates=x, inference="bootstrap", n_bootstrap=20
            )
        # One warning from the full fit; the quiet replicate refits log at DEBUG.
        assert len(_fallback_warnings(caught)) == 1
        assert result.diagnostics["calibration"]["selected_mode"] == "monotone"
        assert result.diagnostics["bootstrap"]["refit_mode"] == "two_stage"
        assert np.isfinite(result.se)

    def test_direct_estimator_bootstrap_refits_after_full_model_fallback(
        self,
    ) -> None:
        """#67: without calibration_provenance the estimator's bootstrap reads
        the calibrator's selected_mode; after a full-model fallback that is
        "monotone", which cannot refit with covariates, so the bootstrap must
        refit with the requested mode (each replicate falls back the same way)."""
        rng = np.random.default_rng(3)
        n = 300
        x = rng.integers(0, 2, n).astype(float)
        s = rng.uniform(size=n)
        y = np.clip(0.2 + 0.5 * s + 0.2 * x + rng.normal(0, 0.1, n), 0, 1)
        labelled = np.zeros(n, dtype=bool)
        labelled[rng.choice(n, 18, replace=False)] = True
        prompt_ids = [f"p{i}" for i in range(n)]

        calibrator = JudgeCalibrator(
            calibration_mode="two_stage", covariate_names=["platform"]
        )
        with pytest.warns(UserWarning, match=FALLBACK_MESSAGE):
            calibrator.fit_cv(
                s,
                y[labelled],
                labelled,
                covariates=x[:, None],
                prompt_ids=prompt_ids,
            )
        assert calibrator.selected_mode == "monotone"

        draws = FreshDrawDataset(
            target_policy="a",
            samples=[
                FreshDrawSample(
                    prompt_id=prompt_ids[i],
                    target_policy="a",
                    response="",
                    judge_score=float(s[i]),
                    oracle_label=float(y[i]) if labelled[i] else None,
                    draw_idx=0,
                    metadata={"platform": float(x[i])},
                )
                for i in range(n)
            ],
        )
        estimator = CalibratedDirectEstimator(
            target_policies=["a"],
            reward_calibrator=calibrator,
            inference_method="bootstrap",
            n_bootstrap=20,
        )
        estimator.add_fresh_draws("a", draws)
        estimator.fit()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = estimator.estimate()
        # The bootstrap's quiet refits must not repeat the fallback warning.
        assert len(_fallback_warnings(caught)) == 0

        assert result.metadata["calibration_provenance_explicit"] is False
        assert result.metadata["inference"]["bootstrap_refit_mode"] == "two_stage"
        assert np.isfinite(result.estimates[0])
        assert np.isfinite(result.standard_errors[0])
        assert result.standard_errors[0] > 0

    def test_auto_fallback_does_not_claim_monotone_relationship(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """#67: an auto-mode fallback is monotone because there were too few
        labelled rows, not because a monotone relationship was checked, so it
        must not log "Monotone relationship confirmed"."""
        scores, labels, x = _fallback_rows(18)
        labelled = ~np.isnan(labels)
        calibrator = JudgeCalibrator(calibration_mode="auto")
        with caplog.at_level(logging.INFO, logger="cje"):
            with pytest.warns(UserWarning, match=FALLBACK_MESSAGE):
                calibrator.fit_cv(scores, labels[labelled], labelled, covariates=x)
        messages = [r.getMessage() for r in caplog.records]
        assert calibrator.selected_mode == "monotone"
        # The auto-mode summary is still logged at INFO, so the capture works.
        assert any("Auto-calibration selected: monotone" in m for m in messages)
        assert not any("Monotone relationship confirmed" in m for m in messages)

    @pytest.mark.parametrize("mode", ["auto", "two_stage"])
    def test_constant_fit_after_fallback_names_the_label_count(
        self, mode: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """#67: once a fallback reports "monotone", the constant-fit warning
        can fire; the caller already asked for auto or two-stage, so the
        remedy is more labels or the judge orientation, not auto mode."""
        rng = np.random.default_rng(2)
        n = 100
        s = rng.uniform(size=n)
        x = rng.normal(size=(n, 1))
        y = np.clip(1 - s + rng.normal(0, 0.05, n), 0, 1)
        labelled = np.zeros(n, dtype=bool)
        labelled[:16] = True
        calibrator = JudgeCalibrator(
            calibration_mode=mode, covariate_names=["x"]  # type: ignore[arg-type]
        )
        with caplog.at_level(logging.WARNING, logger="cje"):
            with pytest.warns(UserWarning, match=FALLBACK_MESSAGE):
                calibrator.fit_cv(s, y[labelled], labelled, 4, covariates=x)
        assert calibrator.selected_mode == "monotone"
        constant = [
            r.getMessage()
            for r in caplog.records
            if "collapsed to a constant" in r.getMessage()
        ]
        assert len(constant) == 1
        assert "calibration_mode='auto'" not in constant[0]
        assert "fewer than 20 labelled rows" in constant[0]
        assert "orientation" in constant[0]

    @staticmethod
    def _fresh_draws(n_labelled: int) -> Dict[str, List[Dict[str, Any]]]:
        rng = np.random.default_rng(3)
        n = 300
        x = rng.integers(0, 2, n).astype(float)
        s = rng.uniform(size=n)
        y = np.clip(0.2 + 0.5 * s + 0.2 * x + rng.normal(0, 0.1, n), 0, 1)
        labelled = set(rng.choice(n, n_labelled, replace=False).tolist())
        records = []
        for i in range(n):
            record: Dict[str, Any] = {
                "prompt_id": f"p{i}",
                "judge_score": float(s[i]),
                "platform": float(x[i]),
            }
            if i in labelled:
                record["oracle_label"] = float(y[i])
            records.append(record)
        return {"policy_a": records}

    @pytest.mark.parametrize(
        "n_labelled,selected,used,n_without",
        [
            (18, "monotone", False, 5),
            (22, "two_stage", True, 5),
            (200, "two_stage", True, 0),
        ],
    )
    @pytest.mark.parametrize("inference", ["cluster_robust", "bootstrap"])
    def test_analyze_dataset_calibration_metadata(
        self,
        n_labelled: int,
        selected: str,
        used: bool,
        n_without: int,
        inference: str,
    ) -> None:
        """#67: analyze_dataset reports the same fields on both inference
        paths. Its bootstrap refits with the requested calibration_mode from
        the provenance, so the estimator's fallback branch is covered by
        test_direct_estimator_bootstrap_refits_after_full_model_fallback."""
        config: Dict[str, Any] = {"inference_method": inference}
        if inference == "bootstrap":
            config["n_bootstrap"] = 20
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = analyze_dataset(
                fresh_draws_data=self._fresh_draws(n_labelled),
                calibration_covariates=["platform"],
                estimator_config=config,
            )
        info = results.metadata["calibration_info"]
        assert info["selected_mode"] == selected
        assert info["covariates"] == ["platform"]
        assert info["covariates_used"] is used
        assert info["n_folds"] == 5
        assert info["n_folds_without_covariates"] == n_without
        assert bool(_fallback_warnings(caught)) is (n_without > 0)
        assert np.isfinite(results.standard_errors[0])

    def test_calibrate_dataset_metadata(self) -> None:
        """#67: calibrate_dataset's calibration_info carries the fields too."""
        rng = np.random.default_rng(6)
        samples = [
            Sample(
                prompt_id=f"p{i}",
                prompt="",
                response="",
                reward=None,
                judge_score=float(rng.uniform()),
                oracle_label=float(rng.uniform()) if i < 18 else None,
                metadata={"platform": float(i % 2)},
            )
            for i in range(60)
        ]
        dataset = Dataset(samples=samples, target_policies=["policy_a"])
        with pytest.warns(UserWarning, match=FALLBACK_MESSAGE):
            calibrated, _ = calibrate_dataset(dataset, covariate_names=["platform"])
        info = calibrated.metadata["calibration_info"]
        assert info["selected_mode"] == "monotone"
        assert info["covariates_used"] is False
        assert info["n_folds_without_covariates"] == 5


# ---------------------------------------------------------------------------
# #66: covariates ignored at complete label coverage
# ---------------------------------------------------------------------------


class TestCompleteCoverageCovariates:
    """#66: at complete coverage the covariates were validated and then
    dropped without a warning."""

    def test_covariates_at_complete_coverage_warn(self) -> None:
        """#66: the direct-mean route warns that the covariates are ignored."""
        scores, covariates, outcome = _scores_and_covariates(n=120)
        with pytest.warns(UserWarning, match="every row is labelled"):
            result = calibrated_mean_ci(scores, outcome, covariates=covariates)
        assert result.calibrator is None
        assert result.estimate == pytest.approx(float(np.mean(outcome)))
        assert result.diagnostics["calibration"]["covariates_used"] is False
        assert result.diagnostics["calibration"]["n_folds_without_covariates"] == 0

    def test_no_warning_without_covariates(self) -> None:
        """#66: complete coverage without covariates stays silent."""
        scores, _, outcome = _scores_and_covariates(n=120)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calibrated_mean_ci(scores, outcome)
        assert not any("every row is labelled" in str(w.message) for w in caught)


def test_bootstrap_with_fold_fallback_warns_once() -> None:
    """#67: at 22 labels the full model uses the covariates but every fold falls
    back; a bootstrap call warns once (from the full fit), not once per
    replicate, even though scikit-learn resets the warnings registry."""
    scores, labels, x = _fallback_rows(22)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        result = calibrated_mean_ci(
            scores, labels, covariates=x, inference="bootstrap", n_bootstrap=30
        )
    assert len(_fallback_warnings(caught)) == 1
    assert result.diagnostics["calibration"]["n_folds_without_covariates"] > 0


def test_quiet_fit_records_fallback_without_warning() -> None:
    """#67/#68: fit_cv(quiet=True) records a dropped-covariate fallback in its
    attributes and logs it at DEBUG instead of warning."""
    from cje.calibration import JudgeCalibrator

    scores, labels, x = _fallback_rows(18)
    mask = ~np.isnan(labels)
    calibrator = JudgeCalibrator(
        calibration_mode="two_stage", covariate_names=["platform"]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibrator.fit_cv(scores, labels[mask], mask, covariates=x, quiet=True)
    assert len(_fallback_warnings(caught)) == 0
    assert calibrator.covariates_used is False
