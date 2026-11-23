import itertools
import math
from typing import Any

import pytest

from cje.diagnostics.stratified_planning import plan_binary_reviews


def test_rare_failures_change_the_objective_and_budget() -> None:
    common: Any = dict(judge_fail_share=0.01, fail_sd=math.sqrt(0.24), pass_sd=0.14)
    slope = plan_binary_reviews(objective="slope", budget=50, **common)
    mean = plan_binary_reviews(objective="mean", budget=50, **common)
    assert (slope.fail_reviews, mean.fail_reviews) == (39, 2)
    plan = plan_binary_reviews(objective="slope", budget=500, target_se=0.05, **common)
    assert (plan.fail_reviews, plan.pass_reviews, plan.meets_target) == (124, 35, True)
    assert plan.modeled_se == pytest.approx(0.0499548182958135)
    short = plan_binary_reviews(objective="slope", budget=158, target_se=0.05, **common)
    assert short.total_reviews == 158 and short.meets_target is False


def test_against_every_possible_quota_for_declared_pool() -> None:
    for a, b, cap0, cap1 in itertools.product(
        (0, 0.3, 0.5), (0, 0.1, 0.5), (2, 5, 15), (2, 7, 12)
    ):
        feasible = [
            (n, m, math.sqrt(a * a / n + b * b / m))
            for n in range(2, cap0 + 1)
            for m in range(2, cap1 + 1)
            if n + m <= 20
        ]
        goal = 0.15
        met = [x for x in feasible if x[2] <= goal]
        plan = plan_binary_reviews(
            objective="slope",
            budget=20,
            judge_fail_share=0.9,
            fail_sd=a,
            pass_sd=b,
            available_fail=cap0,
            available_pass=cap1,
            target_se=goal,
        )
        total = (
            min(x[0] + x[1] for x in met) if met else max(x[0] + x[1] for x in feasible)
        )
        assert plan.total_reviews == total
        assert plan.modeled_se == pytest.approx(
            min(x[2] for x in feasible if x[0] + x[1] == total)
        )
        assert plan.meets_target == bool(met)


@pytest.mark.parametrize(
    "wrong",
    [
        {"objective": "unknown"},
        {"judge_fail_share": 0},
        {"judge_fail_share": 1},
        {"judge_fail_share": float("nan")},
        {"budget": 3},
        {"budget": True},
        {"available_fail": 1},
        {"available_pass": 0},
        {"fail_sd": 0.51},
        {"pass_sd": -1},
        {"target_se": 0},
        {"target_se": float("inf")},
    ],
)
def test_invalid_design(wrong: dict[str, Any]) -> None:
    kwargs: Any = dict(objective="slope", judge_fail_share=0.05, budget=50)
    kwargs.update(wrong)
    with pytest.raises(ValueError):
        plan_binary_reviews(**kwargs)
