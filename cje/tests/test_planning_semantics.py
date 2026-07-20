"""Regression tests for claim-calibrated planning diagnostics."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy import stats

from cje.diagnostics.planning import (
    CostModel,
    EvaluationPlan,
    FittedVarianceModel,
    _check_labeling_ignorability,
    _check_labeling_score_balance,
)
import cje.diagnostics.simulation_planning as simulation_module


def _plan() -> EvaluationPlan:
    return EvaluationPlan(
        n_samples=400,
        m_oracle=50,
        total_cost=120.0,
        mde=0.1,
        se_level=0.02,
        power=0.8,
        alpha=0.05,
        sigma2_eval=0.08,
        sigma2_cal=0.02,
        cost_model=CostModel(surrogate_cost=0.1, oracle_cost=1.0),
    )


def test_power_to_detect_is_exact_two_sided_normal_power() -> None:
    plan = _plan()
    se = plan.se_comparison
    effect = 0.04
    critical = stats.norm.ppf(1 - plan.alpha / 2)
    noncentrality = effect / se
    expected = stats.norm.sf(critical - noncentrality) + stats.norm.cdf(
        -critical - noncentrality
    )

    assert plan.power_to_detect(effect) == pytest.approx(expected)
    assert plan.power_to_detect(-effect) == pytest.approx(expected)
    assert plan.power_to_detect(0.0) == pytest.approx(plan.alpha)


def test_variance_fractions_use_the_selected_plan_denominators() -> None:
    plan = _plan()
    eval_component = 0.08 / 400
    cal_component = 0.02 / 50
    total = eval_component + cal_component

    assert plan.eval_variance_component == pytest.approx(eval_component)
    assert plan.cal_variance_component == pytest.approx(cal_component)
    assert plan.eval_variance_fraction == pytest.approx(eval_component / total)
    assert plan.cal_variance_fraction == pytest.approx(cal_component / total)
    assert plan.to_dict()["eval_variance_fraction"] == pytest.approx(
        eval_component / total
    )


def _score_balance_dataset(shift: bool = False) -> Any:
    labeled = np.linspace(0.0, 0.2 if shift else 1.0, 30)
    unlabeled = np.linspace(0.8, 1.0, 30) if shift else labeled.copy()
    samples = [
        SimpleNamespace(judge_score=float(score), oracle_label=0.0) for score in labeled
    ] + [
        SimpleNamespace(judge_score=float(score), oracle_label=None)
        for score in unlabeled
    ]
    return SimpleNamespace(samples=samples)


def test_ks_balance_check_never_grades_ignorability() -> None:
    balanced = _check_labeling_score_balance(_score_balance_dataset())
    shifted = _check_labeling_score_balance(_score_balance_dataset(shift=True))

    assert balanced["score_balance_status"] == "BALANCE_NOT_REJECTED"
    assert balanced["is_ignorable"] is None
    assert "does not establish" in balanced["recommendation"]
    assert shifted["score_balance_status"] == "SHIFT_DETECTED"
    assert shifted["score_shift_detected"] is True
    assert shifted["is_ignorable"] is None
    # The compatibility alias returns the same claim-calibrated schema.
    assert _check_labeling_ignorability(_score_balance_dataset()) == balanced


def test_simulation_result_uses_plan_shares_and_records_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FittedVarianceModel(
        sigma2_eval=0.08,
        sigma2_cal=0.02,
        r_squared=0.9,
        n_measurements=12,
    )
    monkeypatch.setattr(
        simulation_module,
        "simulate_variance_model",
        lambda **kwargs: model,
    )
    cost = CostModel(surrogate_cost=0.1, oracle_cost=1.0)

    result = simulation_module.simulate_planning(
        r2=0.7,
        budget=500.0,
        cost_model=cost,
        n_total=750,
        oracle_fraction=0.3,
        n_replicates=17,
        seed=123,
        verbose=False,
    )

    assert result.eval_variance_fraction == result.plan.eval_variance_fraction
    assert result.cal_variance_fraction == result.plan.cal_variance_fraction
    assert result.scenario_fingerprint["dgp"] == "binary_oracle_monotone_judge_v1"
    assert result.scenario_fingerprint["seed"] == 123
    assert result.scenario_fingerprint["n_total"] == 750
    assert result.scenario_fingerprint["variance_measurement"] == {
        "inference_method": "cluster_robust"
    }
