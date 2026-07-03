"""Regression tests for loud-by-default behavior.

Each test pins a former silent failure mode: data that used to be silently
fabricated, downgraded, relabeled, or dropped must now either FAIL with a
clear error or FILTER with a counted warning.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cje import analyze_dataset
from cje.data.models import Dataset, Sample


def _make_sample(
    prompt_id: str,
    judge_score: float = 0.5,
    base_lp: float = -10.0,
    target_lp: float = -11.0,
) -> Sample:
    return Sample(
        prompt_id=prompt_id,
        prompt=f"prompt {prompt_id}",
        response=f"response {prompt_id}",
        base_policy_logprob=base_lp,
        target_policy_logprobs={"pi_target": target_lp},
        judge_score=judge_score,
        oracle_label=None,
        reward=judge_score,
    )


def _make_logged_dataset(n: int = 20) -> Dataset:
    samples = [
        _make_sample(f"p{i}", judge_score=0.1 + 0.8 * i / max(n - 1, 1))
        for i in range(n)
    ]
    return Dataset(samples=samples, target_policies=["pi_target"])


class TestFreshDrawsDataWithLoggedData:
    """logged_data_path was removed in 0.4.0 and must fail loudly, never be
    silently ignored (as OPE modes once silently ignored fresh_draws_data)."""

    def test_fresh_draws_data_with_logged_data_path_raises(self) -> None:
        fresh_draws_data = {
            "pi_target": [{"prompt_id": "p0", "judge_score": 0.7}],
        }
        with pytest.raises(
            ValueError,
            match="'logged_data_path' is no longer accepted",
        ):
            analyze_dataset(
                logged_data_path="logged.jsonl",
                fresh_draws_data=fresh_draws_data,
            )


class TestNonexistentFreshDrawsDir:
    """A typo'd fresh_draws_dir must fail loudly, not silently degrade."""

    def test_nonexistent_fresh_draws_dir_raises(self, tmp_path: Path) -> None:
        missing_dir = str(tmp_path / "no_such_responses_dir")
        with pytest.raises(ValueError, match="no_such_responses_dir"):
            analyze_dataset(fresh_draws_dir=missing_dir)


class TestZeroOracleDirectMode:
    """Zero-oracle Direct runs used to return raw judge means silently labeled
    method='calibrated_direct'."""

    @pytest.fixture
    def judge_only_fresh_draws(self) -> dict:
        rng = np.random.default_rng(7)
        data = {}
        for policy in ["policy_a", "policy_b"]:
            records = []
            for i in range(30):
                records.append(
                    {
                        "prompt_id": f"p{i}",
                        "judge_score": float(np.clip(rng.uniform(0.1, 0.9), 0, 1)),
                    }
                )
            data[policy] = records
        return data

    def test_zero_oracle_labeled_naive_direct_with_warning(
        self, judge_only_fresh_draws: dict, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            results = analyze_dataset(fresh_draws_data=judge_only_fresh_draws)

        assert results.method == "naive_direct"
        warning_text = " ".join(
            record.message
            for record in caplog.records
            if record.levelno >= logging.WARNING
        )
        assert "No oracle labels found" in warning_text
        assert "UNCALIBRATED judge-score means" in warning_text
        assert "CIs do not account for judge bias" in warning_text

    def test_with_oracle_labels_still_calibrated_direct(self) -> None:
        rng = np.random.default_rng(11)
        data = {}
        for policy in ["policy_a", "policy_b"]:
            records = []
            for i in range(40):
                score = float(np.clip(rng.uniform(0.05, 0.95), 0, 1))
                records.append(
                    {
                        "prompt_id": f"p{i}",
                        "judge_score": score,
                        "oracle_label": float(
                            np.clip(score + rng.normal(0, 0.05), 0, 1)
                        ),
                    }
                )
            data[policy] = records

        results = analyze_dataset(fresh_draws_data=data)
        # Bootstrap inference may be auto-selected, so match the prefix.
        assert results.method.startswith("calibrated_direct")


class TestNonFiniteLogprobs:
    """NaN/inf logprobs used to pass Sample validation ('v > 0' is False for
    NaN and -inf), silently corrupting importance weights downstream."""

    @pytest.mark.parametrize("bad_value", [float("nan"), float("-inf"), float("inf")])
    def test_non_finite_base_logprob_rejected(self, bad_value: float) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="finite"):
            _make_sample("p0", base_lp=bad_value)

    @pytest.mark.parametrize("bad_value", [float("nan"), float("-inf"), float("inf")])
    def test_non_finite_target_logprob_rejected(self, bad_value: float) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="finite"):
            _make_sample("p0", target_lp=bad_value)

    def test_none_logprobs_still_accepted(self) -> None:
        sample = Sample(
            prompt_id="p0",
            prompt="prompt",
            response="response",
            base_policy_logprob=None,
            target_policy_logprobs={"pi_target": None},
            judge_score=0.5,
            oracle_label=None,
            reward=None,
        )
        assert sample.base_policy_logprob is None
        assert sample.target_policy_logprobs["pi_target"] is None


class TestFreshDrawLoadErrors:
    """load_fresh_draws_auto used to swallow per-record parse errors and raise
    a misleading FileNotFoundError even when the file existed."""

    def test_invalid_judge_score_raises_value_error_with_context(
        self, tmp_path: Path
    ) -> None:
        from cje.data.fresh_draws import load_fresh_draws_auto

        file_path = tmp_path / "pi_responses.jsonl"
        file_path.write_text('{"prompt_id": "p1", "judge_score": 8.5, "draw_idx": 0}\n')

        with pytest.raises(ValueError) as excinfo:
            load_fresh_draws_auto(tmp_path, "pi")

        message = str(excinfo.value)
        assert str(file_path) in message
        assert "8.5" in message
        assert not isinstance(excinfo.value, FileNotFoundError)

    def test_genuinely_missing_file_still_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        from cje.data.fresh_draws import load_fresh_draws_auto

        with pytest.raises(FileNotFoundError, match="No fresh draw file found"):
            load_fresh_draws_auto(tmp_path, "pi")

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        """A trailing newline (or interior blank line) must not crash the
        loud parse-error path; the logged-data loader already skips blanks."""
        from cje.data.fresh_draws import load_fresh_draws_auto

        file_path = tmp_path / "pi_responses.jsonl"
        file_path.write_text(
            '{"prompt_id": "p1", "judge_score": 0.5, "draw_idx": 0}\n'
            "\n"
            '{"prompt_id": "p2", "judge_score": 0.7, "draw_idx": 0}\n'
            "\n"
        )

        fresh = load_fresh_draws_auto(tmp_path, "pi")
        assert len(fresh.samples) == 2
        assert sorted(s.prompt_id for s in fresh.samples) == ["p1", "p2"]


class TestPromptIdHandling:
    """Falsy-but-valid prompt_ids (0, \"\") used to be hash-replaced, and
    integer ids were dropped via a blanket except with only a print()."""

    @staticmethod
    def _record(prompt_id: object, judge_score: float = 0.5) -> dict:
        return {
            "prompt_id": prompt_id,
            "prompt": f"prompt {prompt_id}",
            "response": f"response {prompt_id}",
            "base_policy_logprob": -10.0,
            "target_policy_logprobs": {"pi_target": -11.0},
            "judge_score": judge_score,
        }

    def test_integer_prompt_ids_load_as_strings(self) -> None:
        from cje.data.loaders import DatasetLoader, InMemoryDataSource

        records = [self._record(0), self._record(1)]
        dataset = DatasetLoader().load_from_source(InMemoryDataSource(records))

        assert [s.prompt_id for s in dataset.samples] == ["0", "1"]

    def test_fresh_draw_loader_coerces_integer_prompt_ids(self, tmp_path: Path) -> None:
        import json

        from cje.data.loaders import FreshDrawLoader

        file_path = tmp_path / "fresh.jsonl"
        lines = [
            json.dumps(
                {
                    "prompt_id": pid,
                    "target_policy": "pi",
                    "judge_score": 0.5,
                    "draw_idx": 0,
                }
            )
            for pid in [0, 1]
        ]
        file_path.write_text("\n".join(lines) + "\n")

        datasets = FreshDrawLoader.load_from_jsonl(str(file_path))
        assert sorted(s.prompt_id for s in datasets["pi"].samples) == ["0", "1"]

    def test_all_records_invalid_raises(self, caplog: pytest.LogCaptureFixture) -> None:
        from cje.data.loaders import DatasetLoader, InMemoryDataSource

        # Out-of-range judge scores -> every record fails validation
        records = [
            {
                "prompt_id": f"p{i}",
                "prompt": "prompt",
                "response": "response",
                "judge_score": 5.0,  # must be in [0, 1]
            }
            for i in range(3)
        ]
        with caplog.at_level(logging.WARNING):
            with pytest.raises(ValueError, match="all 3 records were skipped"):
                DatasetLoader().load_from_source(InMemoryDataSource(records))

    def test_skipped_records_are_counted_in_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from cje.data.loaders import DatasetLoader, InMemoryDataSource

        records = [self._record("p0"), {"prompt_id": "p1", "judge_score": 5.0}]
        with caplog.at_level(logging.WARNING):
            dataset = DatasetLoader().load_from_source(InMemoryDataSource(records))

        assert len(dataset.samples) == 1
        assert any("Skipped 1/2" in r.message for r in caplog.records)

    def test_minimal_records_load_without_text_fields(self) -> None:
        """Minimal judge+oracle records (no prompt/response/logprobs) load.

        This is the documented calibration_data_path schema; it used to be
        rejected wholesale (missing 'prompt' KeyError skipped every record,
        and target_policies=[] failed Dataset validation)."""
        from cje.data.loaders import DatasetLoader, InMemoryDataSource

        records = [
            {"prompt_id": f"p{i}", "judge_score": 0.5, "oracle_label": 0.5}
            for i in range(3)
        ]
        dataset = DatasetLoader().load_from_source(InMemoryDataSource(records))

        assert len(dataset.samples) == 3
        assert dataset.target_policies == []
        assert all(s.prompt == "" and s.response == "" for s in dataset.samples)


class TestValidateDirectDataScan:
    """validate_direct_data must scan a window, not just data[0] (the
    0.3.0 regression was a field checked only on the first record)."""

    @staticmethod
    def _record(i: int, with_judge: bool = True) -> dict:
        record = {
            "prompt_id": f"p{i}",
            "oracle_label": 0.5,
        }
        if with_judge:
            record["judge_score"] = 0.5
        return record

    def test_judge_only_on_first_record_is_invalid(self) -> None:
        from cje.data.validation import validate_direct_data

        data = [self._record(i, with_judge=(i == 0)) for i in range(50)]
        is_valid, issues = validate_direct_data(data)

        assert not is_valid
        assert any(
            "judge_score" in issue and "missing" in issue.lower() for issue in issues
        )

    def test_judge_on_all_records_is_valid(self) -> None:
        from cje.data.validation import validate_direct_data

        data = [self._record(i, with_judge=True) for i in range(50)]
        is_valid, issues = validate_direct_data(data)

        assert is_valid, issues


class TestDuplicatePromptIdRewards:
    """add_rewards_to_existing_data used to key the judge-score collection and
    the reward join on prompt_id, giving every duplicate the last reward."""

    def test_duplicates_each_get_reward_of_own_score(self, tmp_path: Path) -> None:
        import json

        from cje.calibration.judge import JudgeCalibrator
        from cje.data.reward_utils import add_rewards_to_existing_data

        # Identity-ish calibrator on [0, 1]
        scores = np.linspace(0.0, 1.0, 20)
        calibrator = JudgeCalibrator(calibration_mode="monotone")
        calibrator.fit_transform(scores, scores)

        def record(prompt_id: str, judge_score: float) -> dict:
            return {
                "prompt_id": prompt_id,
                "prompt": f"prompt {prompt_id}",
                "response": f"response {judge_score}",
                "base_policy_logprob": -10.0,
                "target_policy_logprobs": {"pi_target": -11.0},
                "judge_score": judge_score,
            }

        records = [record("dup", 0.2), record("dup", 0.8), record("other", 0.5)]
        data_path = tmp_path / "data.jsonl"
        data_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        output_path = add_rewards_to_existing_data(str(data_path), calibrator)

        with open(output_path) as f:
            output = [json.loads(line) for line in f]

        expected_low = float(calibrator.predict(np.array([0.2]))[0])
        expected_high = float(calibrator.predict(np.array([0.8]))[0])
        assert output[0]["reward"] == pytest.approx(expected_low)
        assert output[1]["reward"] == pytest.approx(expected_high)
        assert output[0]["reward"] != output[1]["reward"]


class TestIndexMaskAlignment:
    """Integer (index) oracle_mask with unsorted indices used to silently
    misalign oracle labels: boolean fancy-indexing returns scores in ascending
    index order while labels stayed in caller order."""

    @staticmethod
    def _make_data() -> tuple:
        rng = np.random.default_rng(3)
        n = 200
        judge = rng.uniform(0.0, 1.0, n)
        oracle = np.clip(judge + rng.normal(0.0, 0.05, n), 0.0, 1.0)
        return judge, oracle

    def test_shuffled_index_mask_matches_sorted_fit_transform(self) -> None:
        from cje.calibration.judge import JudgeCalibrator

        judge, oracle = self._make_data()
        idx_sorted = np.arange(100)
        idx_shuffled = np.random.default_rng(5).permutation(idx_sorted)

        result_sorted = JudgeCalibrator(calibration_mode="monotone").fit_transform(
            judge, oracle[idx_sorted], oracle_mask=idx_sorted
        )
        result_shuffled = JudgeCalibrator(calibration_mode="monotone").fit_transform(
            judge, oracle[idx_shuffled], oracle_mask=idx_shuffled
        )

        np.testing.assert_allclose(
            result_shuffled.calibrated_scores, result_sorted.calibrated_scores
        )
        assert result_shuffled.calibration_rmse == pytest.approx(
            result_sorted.calibration_rmse
        )

    def test_shuffled_index_mask_matches_sorted_fit_cv(self) -> None:
        from cje.calibration.judge import JudgeCalibrator

        judge, oracle = self._make_data()
        idx_sorted = np.arange(100)
        idx_shuffled = np.random.default_rng(5).permutation(idx_sorted)

        result_sorted = JudgeCalibrator(calibration_mode="monotone").fit_cv(
            judge, oracle[idx_sorted], oracle_mask=idx_sorted, n_folds=5
        )
        result_shuffled = JudgeCalibrator(calibration_mode="monotone").fit_cv(
            judge, oracle[idx_shuffled], oracle_mask=idx_shuffled, n_folds=5
        )

        np.testing.assert_allclose(
            result_shuffled.calibrated_scores, result_sorted.calibrated_scores
        )
        assert result_shuffled.oof_rmse == pytest.approx(result_sorted.oof_rmse)

    def test_duplicate_indices_rejected(self) -> None:
        from cje.calibration.judge import JudgeCalibrator

        judge, oracle = self._make_data()
        idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8])
        with pytest.raises(ValueError, match="duplicate indices"):
            JudgeCalibrator(calibration_mode="monotone").fit_transform(
                judge, oracle[idx], oracle_mask=idx
            )


class TestNoFabricatedCalibratorFallbacks:
    """predict_oof used to zero-fill folds without a model, and
    FlexibleCalibrator fabricated constant/mean-of-scores rewards when no
    models were fitted."""

    def test_predict_oof_unknown_fold_raises(self) -> None:
        from cje.calibration.judge import JudgeCalibrator

        judge = np.linspace(0.0, 1.0, 50)
        calibrator = JudgeCalibrator(calibration_mode="monotone")
        calibrator.fit_cv(judge, judge, n_folds=5)

        with pytest.raises(ValueError, match=r"fold ids \[7\].*no fitted"):
            calibrator.predict_oof(np.array([0.5]), np.array([7]))

    def test_predict_oof_known_folds_still_works(self) -> None:
        from cje.calibration.judge import JudgeCalibrator

        judge = np.linspace(0.0, 1.0, 50)
        calibrator = JudgeCalibrator(calibration_mode="monotone")
        calibrator.fit_cv(judge, judge, n_folds=5)

        predictions = calibrator.predict_oof(np.array([0.2, 0.8]), np.array([0, 4]))
        assert predictions.shape == (2,)
        assert np.all((predictions >= 0) & (predictions <= 1))

    def test_unfitted_flexible_calibrator_raises_full_model_path(self) -> None:
        from cje.calibration.flexible_calibrator import FlexibleCalibrator

        calibrator = FlexibleCalibrator(mode="two_stage")
        with pytest.raises(RuntimeError, match="no fitted"):
            calibrator.predict(np.array([0.5]), folds=None)

    def test_unfitted_flexible_calibrator_raises_oof_path(self) -> None:
        from cje.calibration.flexible_calibrator import FlexibleCalibrator

        calibrator = FlexibleCalibrator(mode="two_stage")
        with pytest.raises(RuntimeError, match="no fitted"):
            calibrator.predict(np.array([0.5]), folds=np.array([0]))


class TestAntiCorrelatedJudgeWarning:
    """An anti-correlated judge in monotone mode used to silently collapse
    all calibrated rewards to the oracle mean (std=0, no warning)."""

    def test_anti_correlated_judge_warns_fit_transform(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        judge = np.linspace(0.0, 1.0, 50)
        oracle = 1.0 - judge

        from cje.calibration.judge import JudgeCalibrator

        with caplog.at_level(logging.WARNING):
            JudgeCalibrator(calibration_mode="monotone").fit_transform(judge, oracle)

        assert any(
            "collapsed to a constant" in record.message and "inverted" in record.message
            for record in caplog.records
        )

    def test_anti_correlated_judge_warns_fit_cv(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        judge = np.linspace(0.0, 1.0, 50)
        oracle = 1.0 - judge

        from cje.calibration.judge import JudgeCalibrator

        with caplog.at_level(logging.WARNING):
            JudgeCalibrator(calibration_mode="monotone").fit_cv(
                judge, oracle, n_folds=5
            )

        assert any(
            "collapsed to a constant" in record.message for record in caplog.records
        )

    def test_correlated_judge_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        judge = np.linspace(0.0, 1.0, 50)

        from cje.calibration.judge import JudgeCalibrator

        with caplog.at_level(logging.WARNING):
            JudgeCalibrator(calibration_mode="monotone").fit_transform(judge, judge)

        assert not any(
            "collapsed to a constant" in record.message for record in caplog.records
        )


class TestCombineOracleSourcesKeepsAllPairs:
    """_combine_oracle_sources used to dedupe oracle pairs by prompt_id alone:
    K policies' fresh-draw labels for one prompt collapsed to the last policy's
    pair while n_from_fresh still counted every overwrite."""

    @staticmethod
    def _fresh_draw(
        policy: str, prompt_id: str, judge: float, oracle: float, draw_idx: int = 0
    ) -> Any:
        from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample

        return FreshDrawDataset(
            target_policy=policy,
            draws_per_prompt=1,
            samples=[
                FreshDrawSample(
                    prompt_id=prompt_id,
                    target_policy=policy,
                    judge_score=judge,
                    oracle_label=oracle,
                    draw_idx=draw_idx,
                    response=None,
                    fold_id=None,
                )
            ],
        )

    def test_two_policies_same_prompt_contribute_two_pairs(self) -> None:
        from cje.interface.service import AnalysisService

        fresh_draws = {
            "policy_a": self._fresh_draw("policy_a", "p1", judge=0.4, oracle=0.45),
            "policy_b": self._fresh_draw("policy_b", "p1", judge=0.8, oracle=0.85),
        }

        combined, metadata = AnalysisService()._combine_oracle_sources(
            None,
            None,
            fresh_draws,
            ["policy_a", "policy_b"],
            "judge_score",
            "oracle_label",
        )

        assert combined.n_samples == 2
        assert metadata["fresh_draws"]["n_oracle"] == 2
        assert metadata["total_oracle"] == 2
        judge_scores = sorted(
            s.judge_score for s in combined.samples if s.judge_score is not None
        )
        assert judge_scores == [0.4, 0.8]

    def test_identical_scores_across_policies_do_not_collapse(self) -> None:
        """Respin M1: with binary/rubric scores, different policies' draws for
        one prompt collide on (judge, oracle); the pair key must identify the
        response (policy + draw), not just the "fresh_draws" family."""
        from cje.interface.service import AnalysisService

        fresh_draws = {
            "policy_a": self._fresh_draw("policy_a", "p1", judge=1.0, oracle=1.0),
            "policy_b": self._fresh_draw("policy_b", "p1", judge=1.0, oracle=1.0),
        }

        combined, metadata = AnalysisService()._combine_oracle_sources(
            None,
            None,
            fresh_draws,
            ["policy_a", "policy_b"],
            "judge_score",
            "oracle_label",
        )

        assert combined.n_samples == 2
        assert metadata["fresh_draws"]["n_oracle"] == 2
        # No cross-policy false "conflicts" either: fresh draws are one family
        assert metadata["n_conflicts"] == 0
        sources = sorted(s.metadata["source"] for s in combined.samples)
        assert sources == [
            "fresh_draws:policy_a:draw0",
            "fresh_draws:policy_b:draw0",
        ]

    def test_true_duplicates_are_deduped(self) -> None:
        from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample
        from cje.interface.service import AnalysisService

        # Distinct draws (draw_idx 0 and 1) are distinct responses: identical
        # (judge, oracle) values keep BOTH pairs. Only a re-ingested copy of
        # the same draw (same policy + prompt + draw_idx + values) dedupes.
        fd = FreshDrawDataset(
            target_policy="policy_a",
            draws_per_prompt=2,
            samples=[
                FreshDrawSample(
                    prompt_id="p1",
                    target_policy="policy_a",
                    judge_score=0.4,
                    oracle_label=0.45,
                    draw_idx=i,
                    response=None,
                    fold_id=None,
                )
                for i in (0, 1, 1)  # draw 1 ingested twice -> true duplicate
            ],
        )

        combined, metadata = AnalysisService()._combine_oracle_sources(
            None,
            None,
            {"policy_a": fd},
            ["policy_a"],
            "judge_score",
            "oracle_label",
        )

        assert combined.n_samples == 2
        assert metadata["fresh_draws"]["n_oracle"] == 2

    def test_cross_source_conflict_warned_but_both_pairs_kept(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from cje.interface.service import AnalysisService

        def make_dataset(judge: float, oracle: float) -> Dataset:
            sample = Sample(
                prompt_id="p1",
                prompt="prompt",
                response="response",
                base_policy_logprob=-1.0,
                target_policy_logprobs={"policy_a": -1.0},
                judge_score=judge,
                oracle_label=oracle,
                reward=None,
            )
            return Dataset(samples=[sample], target_policies=["policy_a"])

        calibration_dataset = make_dataset(judge=0.9, oracle=0.9)
        logged_dataset = make_dataset(judge=0.2, oracle=0.2)

        with caplog.at_level(logging.WARNING):
            combined, metadata = AnalysisService()._combine_oracle_sources(
                calibration_dataset,
                logged_dataset,
                None,
                ["policy_a"],
                "judge_score",
                "oracle_label",
            )

        assert combined.n_samples == 2  # both pairs kept, none overwritten
        assert metadata["n_conflicts"] == 1
        assert any(
            "conflicting oracle labels" in record.message for record in caplog.records
        )
