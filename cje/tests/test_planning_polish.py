"""Planning polish (0.5.0): fit_ok, poor-fit warnings, m_min passthrough.

The NNLS variance fitter can return arbitrarily poor fits (r_squared can go
negative); pre-0.5.0 the only guard was a verbose-mode print. Planning now
warns via logger when the model's fit_ok is False, and simulate_planning
forwards m_min to plan_evaluation.
"""

import logging
from unittest.mock import patch

import pytest

from cje.diagnostics.planning import (
    CostModel,
    FittedVarianceModel,
    plan_evaluation,
    plan_for_mde,
)
from cje.diagnostics.simulation_planning import simulate_planning

pytestmark = pytest.mark.unit


def _model(r_squared: float) -> FittedVarianceModel:
    return FittedVarianceModel(
        sigma2_eval=0.05,
        sigma2_cal=0.02,
        r_squared=r_squared,
        n_measurements=9,
    )


_COST = CostModel(surrogate_cost=0.01, oracle_cost=0.16)


class TestFitOk:
    def test_true_at_and_above_threshold(self) -> None:
        assert _model(0.5).fit_ok is True
        assert _model(0.95).fit_ok is True

    def test_false_below_threshold(self) -> None:
        assert _model(0.49).fit_ok is False
        assert _model(-0.3).fit_ok is False  # NNLS can go negative


class TestPoorFitWarning:
    def test_plan_evaluation_warns_on_poor_fit(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            plan = plan_evaluation(
                budget=5000, variance_model=_model(0.2), cost_model=_COST
            )
        assert plan.n_samples > 0
        assert any("poorly fitted variance model" in r.message for r in caplog.records)

    def test_plan_evaluation_silent_on_good_fit(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            plan_evaluation(budget=5000, variance_model=_model(0.9), cost_model=_COST)
        assert not any(
            "poorly fitted variance model" in r.message for r in caplog.records
        )

    def test_plan_for_mde_warns_once_despite_iteration(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            plan = plan_for_mde(
                target_mde=0.05, variance_model=_model(-0.3), cost_model=_COST
            )
        assert plan.mde <= 0.05 * 1.02
        warnings = [
            r for r in caplog.records if "poorly fitted variance model" in r.message
        ]
        assert len(warnings) == 1


class TestSimulatePlanningMMin:
    def test_m_min_is_forwarded_to_plan_evaluation(self) -> None:
        # Patch the simulation so the test is fast and deterministic; the
        # point is the passthrough, not the simulation itself.
        with patch(
            "cje.diagnostics.simulation_planning.simulate_variance_model",
            return_value=_model(0.9),
        ):
            low = simulate_planning(
                r2=0.9, budget=5000, cost_model=_COST, m_min=30, verbose=False
            )
            high = simulate_planning(
                r2=0.9, budget=5000, cost_model=_COST, m_min=500, verbose=False
            )
        assert low.plan.m_oracle >= 30
        assert high.plan.m_oracle >= 500

    def test_m_min_default_matches_plan_evaluation(self) -> None:
        import inspect

        sim_default = inspect.signature(simulate_planning).parameters["m_min"].default
        plan_default = inspect.signature(plan_evaluation).parameters["m_min"].default
        assert sim_default == plan_default == 30
