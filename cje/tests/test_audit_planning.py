"""Reference-model power, real audit semantics, and complete label accounting."""

import json
from typing import Any

import numpy as np
import pytest
from scipy import stats

from cje import AuditScenario, CostModel, plan_transport_audits
from cje.diagnostics.audit_planning import _pass_probability
from cje.diagnostics.transport import audit_transportability


def plan(scenarios: dict[str, AuditScenario], **kwargs: Any) -> Any:
    return plan_transport_audits(
        scenarios, calibration_labels=40, cost_model=CostModel(oracle_cost=2), **kwargs
    )


@pytest.mark.parametrize(
    "n,bias,alpha", [(30, 0.0, 0.05), (70, 0.03, 0.025), (150, -0.02, 0.01)]
)
def test_power_matches_independent_simulated_t_audits(
    n: int, bias: float, alpha: float
) -> None:
    scenario = AuditScenario(0.12, 0.05, assumed_bias=bias)
    residuals = np.random.default_rng(2190 + n).normal(bias, 0.12, size=(40_000, n))
    means = residuals.mean(axis=1)
    half = (
        stats.t.ppf(1 - alpha / 2, n - 1) * residuals.std(axis=1, ddof=1) / np.sqrt(n)
    )
    observed = np.mean((means - half >= -0.05) & (means + half <= 0.05))
    assert _pass_probability(n, scenario, alpha) == pytest.approx(observed, abs=0.008)


def test_required_size_is_minimal_and_family_power_is_conservative() -> None:
    scenarios = {"a": AuditScenario(0.1, 0.05), "b": AuditScenario(0.15, 0.05, 0.01)}
    result = plan(scenarios)
    assert result.per_audit_power == pytest.approx(0.9)
    for entry in result.policies.values():
        n = entry.required_clusters
        assert n is not None and n > 20
        assert _pass_probability(n, entry.scenario, 0.025) >= 0.9
        assert _pass_probability(n - 1, entry.scenario, 0.025) < 0.9
    assert sum(1 - p.modeled_power for p in result.policies.values()) <= 0.2


def test_calibration_and_all_underlying_ratings_are_charged() -> None:
    result = plan(
        {
            "a": AuditScenario(0, 0.05, labels_per_cluster=3),
            "b": AuditScenario(0, 0.05, labels_per_cluster=2.5),
        },
        budget=299,
    )
    assert result.required_audit_labels == 110
    assert result.required_total_labels == 150
    assert result.human_label_cost == 300
    assert result.within_budget is False
    assert "40 calibration" in result.summary()
    data = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert data["empirical_transport_verified"] is False
    assert data["finite_population_correction"] is False


def test_availability_does_not_change_sample_size_or_apply_fpc() -> None:
    small = plan({"a": AuditScenario(0.1, 0.05, available_clusters=30)})
    large = plan({"a": AuditScenario(0.1, 0.05, available_clusters=10_000)})
    assert (
        small.policies["a"].required_clusters == large.policies["a"].required_clusters
    )
    assert small.policies["a"].within_available is False
    assert large.policies["a"].within_available is True
    assert small.policies["a"].power_at_available < 0.8
    assert (
        plan({"a": AuditScenario(0, 0.05, available_clusters=19)})
        .policies["a"]
        .power_at_available
        == 0
    )


@pytest.mark.parametrize("bias", [0.05, -0.06])
def test_bias_at_or_outside_margin_cannot_meet_power(bias: float) -> None:
    result = plan(
        {"a": AuditScenario(0.1, 0.05, bias, available_clusters=100)}, budget=1000
    )
    entry = result.policies["a"]
    assert entry.required_clusters is None
    assert entry.reason == "assumed_mean_outside_margin"
    assert result.required_total_labels is None and result.human_label_cost is None
    assert result.within_budget is None and entry.within_available is False


def test_search_limit_is_distinguished_from_impossible_mean() -> None:
    result = plan({"a": AuditScenario(0.5, 0.01)}, max_clusters=30)
    assert result.policies["a"].reason == "search_limit_reached"
    assert result.required_total_labels is None


def test_zero_variance_boundary_matches_real_audit() -> None:
    class ZeroCalibrator:
        def predict(self, scores: np.ndarray) -> np.ndarray:
            return np.zeros_like(scores)

    result = plan({"a": AuditScenario(0, 0.125, 0.125)})
    assert result.policies["a"].required_clusters == 20
    for n, expected in [(19, "INCONCLUSIVE"), (20, "PASS")]:
        audit = audit_transportability(
            ZeroCalibrator(),
            [
                {"judge_score": 0.5, "oracle_label": 0.125, "prompt_id": str(i)}
                for i in range(n)
            ],
            delta_max=0.125,
        )
        assert audit.status == expected


def test_real_audit_interval_matches_reference_statistic() -> None:
    class ZeroCalibrator:
        def predict(self, scores: np.ndarray) -> np.ndarray:
            return np.zeros_like(scores)

    residuals = np.random.default_rng(12).normal(0.01, 0.1, 50)
    audit = audit_transportability(
        ZeroCalibrator(),
        [
            {"judge_score": 0.5, "oracle_label": y, "prompt_id": str(i)}
            for i, y in enumerate(residuals)
        ],
        delta_max=0.05,
        family_size=3,
    )
    half = stats.t.ppf(1 - 0.05 / 3 / 2, 49) * residuals.std(ddof=1) / np.sqrt(50)
    assert audit.delta_ci == pytest.approx(
        (residuals.mean() - half, residuals.mean() + half)
    )


def test_planning_is_scale_invariant_and_responds_to_margin_and_family() -> None:
    base = plan({"a": AuditScenario(0.1, 0.05)})
    for scale in [1e-200, 1e200]:
        scaled = plan({"a": AuditScenario(0.1 * scale, 0.05 * scale)})
        assert (
            scaled.policies["a"].required_clusters
            == base.policies["a"].required_clusters
        )
    narrow = plan({"a": AuditScenario(0.1, 0.04)})
    family = plan({"a": AuditScenario(0.1, 0.05)}, family_size=17)
    assert narrow.policies["a"].required_clusters > base.policies["a"].required_clusters
    assert family.policies["a"].required_clusters > base.policies["a"].required_clusters


@pytest.mark.parametrize(
    "kwargs",
    [
        {"residual_sd": True, "delta_max": 0.05},
        {"residual_sd": -1, "delta_max": 0.05},
        {"residual_sd": float("nan"), "delta_max": 0.05},
        {"residual_sd": 0.1, "delta_max": 0},
        {"residual_sd": 0.1, "delta_max": 0.05, "assumed_bias": float("inf")},
        {"residual_sd": 0.1, "delta_max": 0.05, "available_clusters": 1.5},
        {"residual_sd": 0.1, "delta_max": 0.05, "labels_per_cluster": 0.5},
    ],
)
def test_invalid_scenario_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        AuditScenario(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0},
        {"power": 0.5},
        {"power": 1},
        {"family_size": 0},
        {"family_size": 1e2},
        {"min_clusters": 1},
        {"max_clusters": 19},
        {"budget": -1},
        {"alpha": 1e-30},
        {"family_size": 10**1000},
    ],
)
def test_invalid_plan_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        plan({"a": AuditScenario(0.1, 0.05)}, **kwargs)


def test_missing_policies_and_invalid_cost_rejected() -> None:
    with pytest.raises(ValueError):
        plan({})
    with pytest.raises(ValueError):
        plan({" ": AuditScenario(0.1, 0.05)})
    with pytest.raises(TypeError):
        plan({"a": {}})  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        plan_transport_audits(
            {"a": AuditScenario(0.1, 0.05)},
            calibration_labels=True,
            cost_model=CostModel(),
        )
    with pytest.raises(ValueError):
        plan_transport_audits(
            {"a": AuditScenario(0.1, 0.05)},
            calibration_labels=40,
            cost_model=CostModel(oracle_cost=float("nan")),
        )
