"""Regression tests for loud-by-default behavior.

Each test pins a former silent failure mode: data that used to be silently
fabricated, downgraded, relabeled, or dropped must now either FAIL with a
clear error or FILTER with a counted warning.
"""

import logging
from pathlib import Path

import numpy as np
import pytest

from cje import analyze_dataset
from cje.data.models import Dataset, Sample
from cje.interface.mode_detection import detect_analysis_mode


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
    """fresh_draws_data used to be silently ignored when logged_data_path was set."""

    def test_fresh_draws_data_with_logged_data_path_raises(self) -> None:
        fresh_draws_data = {
            "pi_target": [{"prompt_id": "p0", "judge_score": 0.7}],
        }
        with pytest.raises(
            ValueError,
            match=(
                "fresh_draws_data together with logged_data_path is not yet "
                "supported; write draws to disk and pass fresh_draws_dir"
            ),
        ):
            analyze_dataset(
                logged_data_path="logged.jsonl",
                fresh_draws_data=fresh_draws_data,
            )


class TestNonexistentFreshDrawsDir:
    """A typo'd fresh_draws_dir used to silently downgrade auto mode DR -> IPS."""

    def test_nonexistent_fresh_draws_dir_raises(self, tmp_path: Path) -> None:
        dataset = _make_logged_dataset()
        missing_dir = str(tmp_path / "no_such_responses_dir")
        with pytest.raises(FileNotFoundError, match="no_such_responses_dir"):
            detect_analysis_mode(dataset, missing_dir)

    def test_none_fresh_draws_dir_still_selects_ips(self) -> None:
        dataset = _make_logged_dataset()
        mode, _, coverage = detect_analysis_mode(dataset, None)
        assert mode == "ips"
        assert coverage == 1.0

    def test_existing_fresh_draws_dir_selects_dr(self, tmp_path: Path) -> None:
        dataset = _make_logged_dataset()
        mode, _, _ = detect_analysis_mode(dataset, str(tmp_path))
        assert mode == "dr"


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

        # Missing required 'response' field -> every record is skipped
        records = [
            {
                "prompt_id": f"p{i}",
                "prompt": "prompt",
                "base_policy_logprob": -10.0,
                "target_policy_logprobs": {"pi_target": -11.0},
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

        records = [self._record("p0"), {"prompt_id": "p1", "prompt": "x"}]
        with caplog.at_level(logging.WARNING):
            dataset = DatasetLoader().load_from_source(InMemoryDataSource(records))

        assert len(dataset.samples) == 1
        assert any("Skipped 1/2" in r.message for r in caplog.records)


class TestValidateRewardScan:
    """validate_cje_data used to check the reward field only on data[0]."""

    @staticmethod
    def _record(with_reward: bool) -> dict:
        record = {
            "prompt_id": "p",
            "prompt": "prompt",
            "response": "response",
            "base_policy_logprob": -10.0,
            "target_policy_logprobs": {"pi_target": -11.0},
        }
        if with_reward:
            record["reward"] = 0.5
        return record

    def test_reward_only_on_first_record_is_invalid(self) -> None:
        from cje.data.validation import validate_cje_data

        data = [self._record(with_reward=(i == 0)) for i in range(50)]
        is_valid, issues = validate_cje_data(data, reward_field="reward")

        assert not is_valid
        assert any(
            "reward" in issue.lower() and "missing" in issue.lower() for issue in issues
        )

    def test_reward_on_all_records_is_valid(self) -> None:
        from cje.data.validation import validate_cje_data

        data = [self._record(with_reward=True) for i in range(50)]
        is_valid, issues = validate_cje_data(data, reward_field="reward")

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
