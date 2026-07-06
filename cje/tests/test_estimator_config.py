"""estimator_config validation in analyze_dataset (0.5.0, D3).

Pre-0.5.0, a user-supplied oua_jackknife collided with the pipeline's own
default (TypeError: multiple values), unknown keys surfaced as an opaque
TypeError from CalibratedDirectEstimator.__init__, and a user
reward_calibrator silently fought the pipeline-managed one. Now: the user
value overrides the default, unknown keys raise a ValueError listing the
valid keys, and reward_calibrator is rejected outright.
"""

from typing import Any, Dict, List

import pytest

from cje import analyze_dataset

pytestmark = pytest.mark.unit


def _labeled_draws(n: int = 40) -> Dict[str, List[Dict[str, Any]]]:
    """Small deterministic fresh draws; every other row carries an oracle label."""
    records = []
    for i in range(n):
        s = 0.2 + 0.6 * i / (n - 1)
        record: Dict[str, Any] = {"prompt_id": f"p{i:03d}", "judge_score": s}
        if i % 2 == 0:
            record["oracle_label"] = 0.25 + 0.5 * s
        records.append(record)
    return {"pi": records}


class TestOuaJackknifeOverride:
    def test_user_false_overrides_calibrated_default(self) -> None:
        """{"oua_jackknife": False} must be honored (default would be True
        because a calibrator is fit from the oracle labels)."""
        result = analyze_dataset(
            fresh_draws_data=_labeled_draws(),
            estimator_config={
                "oua_jackknife": False,
                "inference_method": "cluster_robust",
            },
        )
        se_components = result.metadata["se_components"]
        assert se_components["includes_oracle_uncertainty"] is False
        assert "oracle_variance_per_policy" not in se_components

    def test_default_keeps_oracle_uncertainty(self) -> None:
        result = analyze_dataset(
            fresh_draws_data=_labeled_draws(),
            estimator_config={"inference_method": "cluster_robust"},
        )
        se_components = result.metadata["se_components"]
        assert se_components["includes_oracle_uncertainty"] is True
        assert "oracle_variance_per_policy" in se_components


class TestUnknownKeyRejected:
    def test_unknown_key_raises_with_valid_keys_listed(self) -> None:
        with pytest.raises(ValueError, match="n_folds") as excinfo:
            analyze_dataset(
                fresh_draws_data=_labeled_draws(),
                estimator_config={"n_folds": 10},
            )
        message = str(excinfo.value)
        assert "Valid keys" in message
        for key in (
            "oua_jackknife",
            "inference_method",
            "n_bootstrap",
            "bootstrap_seed",
            "use_augmented_estimator",
            "paired_comparison",
        ):
            assert key in message


class TestRewardCalibratorRejected:
    def test_reward_calibrator_raises(self) -> None:
        with pytest.raises(
            ValueError, match="reward_calibrator is managed by analyze_dataset"
        ):
            analyze_dataset(
                fresh_draws_data=_labeled_draws(),
                estimator_config={"reward_calibrator": object()},
            )
