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
