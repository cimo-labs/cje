"""Plan random reviews within the fail and pass strata of a binary judge."""

from dataclasses import dataclass
import math
from typing import Literal, Optional, Tuple

from .audit_planning import _finite, _integer


@dataclass(frozen=True)
class BinaryReviewPlan:
    """Quotas for independent reviews at equal per-review cost, in label-scale units."""

    objective: str
    fail_reviews: int
    pass_reviews: int
    modeled_se: float
    target_se: Optional[float]
    meets_target: Optional[bool]

    @property
    def total_reviews(self) -> int:
        return self.fail_reviews + self.pass_reviews


def plan_binary_reviews(
    *,
    objective: Literal["mean", "slope"],
    judge_fail_share: float,
    budget: int,
    fail_sd: float = 0.5,
    pass_sd: float = 0.5,
    target_se: Optional[float] = None,
    available_fail: Optional[int] = None,
    available_pass: Optional[int] = None,
) -> BinaryReviewPlan:
    """Allocate an upper budget across a binary judge's two observed strata.

    The outcome is a [0,1] label. The slope is mean(label | pass) minus
    mean(label | fail). The mean uses a known target traffic share for each
    stratum. SDs describe independent units within those strata. Defaults use
    the range-based upper bound 0.5. A zero SD is a deterministic-outcome
    assumption, which a small constant pilot cannot establish.

    With a target SE, return the smallest feasible total whose modeled SE meets
    it, or the best allocation within the budget with meets_target=False.
    Otherwise use the available budget. Require two independent reviews per
    stratum so each stratum can supply a sample variance. Availability limits
    recruitment. Variance targets future independent prompts, with no finite
    population correction or allowance for estimated traffic shares. All
    numbers are planning assumptions, before outcomes are observed.
    """
    if objective not in ("mean", "slope"):
        raise ValueError("objective must be 'mean' or 'slope'")
    share = _finite(judge_fail_share, "judge_fail_share")
    if not 0 < share < 1:
        raise ValueError("judge_fail_share must lie strictly between zero and one")
    budget = _integer(budget, "budget", 4)
    fail_cap = (
        budget
        if available_fail is None
        else _integer(available_fail, "available_fail", 2)
    )
    pass_cap = (
        budget
        if available_pass is None
        else _integer(available_pass, "available_pass", 2)
    )
    fail_sd = _finite(fail_sd, "fail_sd", 0)
    pass_sd = _finite(pass_sd, "pass_sd", 0)
    if max(fail_sd, pass_sd) > 0.5:
        raise ValueError(
            "For outcomes in [0,1], each population SD must be at most 0.5"
        )
    if target_se is not None:
        target_se = _finite(target_se, "target_se", 0)
        if target_se == 0:
            raise ValueError("target_se must be positive")

    a, b = (
        (share * fail_sd, (1 - share) * pass_sd)
        if objective == "mean"
        else (fail_sd, pass_sd)
    )

    def allocate(total: int) -> Tuple[float, int]:
        lo, hi = max(2, total - pass_cap), min(fail_cap, total - 2)
        # For n fails, V(n)=a²/n+b²/(total-n). The constrained integer
        # optimum is one of the two neighbors of its continuous minimum.
        fraction = a / (a + b) if a + b else 0.5
        left = min(hi, max(lo, math.floor(total * fraction)))
        return min(
            (math.hypot(a / math.sqrt(n), b / math.sqrt(total - n)), n)
            for n in (left, min(hi, left + 1))
        )

    total = min(budget, fail_cap + pass_cap)
    if target_se is not None and allocate(total)[0] <= target_se:
        lower = 4
        while lower < total:
            mid = (lower + total) // 2
            if allocate(mid)[0] <= target_se:
                total = mid
            else:
                lower = mid + 1
    se, n_fail = allocate(total)
    return BinaryReviewPlan(
        objective,
        n_fail,
        total - n_fail,
        se,
        target_se,
        None if target_se is None else se <= target_se,
    )
