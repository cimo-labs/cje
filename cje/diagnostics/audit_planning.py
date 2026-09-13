"""Human-label budgets for held-out residual transport audits.

This is a Gaussian reference calculation for independent, equally weighted
cluster residuals. It plans the probability that the actual t interval lies
inside a declared margin, including randomness in the sample variance. It does
not grade an observed audit or infer that a pilot's residual model will transport.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any, Mapping

from scipy import integrate, stats

from .planning import CostModel


def _finite(value: float, name: str, minimum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return float(value)


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


@dataclass(frozen=True)
class AuditScenario:
    """Assumptions for one policy's audit, in its reported oracle/output units.

    ``residual_sd`` is the SD of independent cluster-level residuals, not a
    standard error or the SD of correlated rows within clusters. For unequal
    cluster sizes/weights, first justify an appropriate independent-unit model.
    ``available_clusters`` excludes clusters used to fit the calibrator.
    ``labels_per_cluster`` is the known or anticipated mean count of underlying
    human ratings needed for one such unit, including repeated raters.
    """

    residual_sd: float
    delta_max: float
    assumed_bias: float = 0.0
    available_clusters: int | None = None
    labels_per_cluster: float = 1.0

    def __post_init__(self) -> None:
        _finite(self.residual_sd, "residual_sd", 0)
        if _finite(self.delta_max, "delta_max") <= 0:
            raise ValueError("delta_max must be positive")
        _finite(self.assumed_bias, "assumed_bias")
        _finite(self.labels_per_cluster, "labels_per_cluster", 1)
        if self.available_clusters is not None:
            _integer(self.available_clusters, "available_clusters", 0)


@dataclass(frozen=True)
class PolicyAuditPlan:
    """A sample-size calculation, never an observed PASS/FAIL audit result."""

    scenario: AuditScenario
    required_clusters: int | None
    modeled_power: float | None
    power_at_available: float | None
    within_available: bool | None
    required_labels: float | None
    reason: str


@dataclass(frozen=True)
class AuditBudgetPlan:
    """Family audit budget including calibration labels already acquired."""

    policies: dict[str, PolicyAuditPlan]
    calibration_labels: int
    required_audit_labels: float | None
    required_total_labels: float | None
    human_label_cost: float | None
    budget: float | None
    within_budget: bool | None
    alpha: float
    family_size: int
    family_power: float
    per_audit_power: float
    min_clusters: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            planning_model="independent_equal_weight_gaussian_cluster_residuals",
            power_target="all_planned_audits_pass; conservative union bound",
            interval="two_sided_t; Bonferroni alpha/family_size",
            finite_population_correction=False,
            empirical_transport_verified=False,
            cost_scope="human ratings only; includes calibration and audits",
        )
        return result

    def summary(self) -> str:
        rows = [
            f"Transport audit plan: {self.family_power:.0%} modeled family power",
            "Gaussian cluster-residual assumptions; this is not an observed audit.",
        ]
        for policy, plan in self.policies.items():
            if plan.required_clusters is None:
                rows.append(f"  {policy}: no sample size returned ({plan.reason})")
            else:
                available = plan.scenario.available_clusters
                suffix = (
                    "availability not declared"
                    if available is None
                    else f"{available} available"
                )
                if plan.within_available is False:
                    suffix += " (insufficient)"
                rows.append(
                    f"  {policy}: {plan.required_clusters:,} independent audit units; {suffix}"
                )
        if self.required_total_labels is not None:
            rows.append(
                f"  Human ratings: {self.calibration_labels:,} calibration + "
                f"{self.required_audit_labels:,.1f} audit = {self.required_total_labels:,.1f} total"
            )
            rows.append(f"  Human-label cost: {self.human_label_cost:,.2f}")
        if self.within_budget is False:
            rows.append("  Declared human-label budget is insufficient.")
        return "\n".join(rows)


def _pass_probability(n: int, scenario: AuditScenario, alpha: float) -> float:
    """Exact normal-model t-interval containment, by one-dimensional quadrature."""
    sd, bias, margin = scenario.residual_sd, scenario.assumed_bias, scenario.delta_max
    if n < 2:
        return 0.0
    if sd == 0:
        return float(abs(bias) <= margin)
    scale = max(sd, abs(bias), margin)
    sd, bias, margin = sd / scale, bias / scale, margin / scale
    se = sd / math.sqrt(n)
    if se == 0:
        raise ValueError("Residual SD is too small relative to the other assumptions")
    critical = float(stats.t.ppf(1 - alpha / 2, n - 1))
    # Integrate over Z=(mean-bias)/SE. The omitted standard-normal tail beyond
    # +/-12 has mass <4e-33. Splitting at the absolute-value kink aids quadrature.
    lower = max(-12.0, (-margin - bias) / se)
    upper = min(12.0, (margin - bias) / se)
    if lower >= upper:
        return 0.0

    def integrand(z: float) -> float:
        slack = max(0.0, margin - abs(bias + se * z))
        ratio = slack / (critical * se)
        if ratio > 1e150:
            return float(stats.norm.pdf(z))
        max_chi_squared = (n - 1) * ratio**2
        return float(stats.norm.pdf(z) * stats.chi2.cdf(max_chi_squared, n - 1))

    kink = -bias / se
    points = [kink] if lower < kink < upper else None
    value, _ = integrate.quad(integrand, lower, upper, points=points, epsabs=1e-9)
    return min(1.0, max(0.0, float(value)))


def plan_transport_audits(
    scenarios: Mapping[str, AuditScenario],
    *,
    calibration_labels: int,
    cost_model: CostModel,
    budget: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    family_size: int | None = None,
    min_clusters: int = 20,
    max_clusters: int = 100_000,
) -> AuditBudgetPlan:
    """Plan held-out audit units and their complete human-label cost.

    ``power`` is the target probability that ALL supplied audits pass, under
    the declared Gaussian assumptions. Each of K scenarios is planned at
    1-(1-power)/K, giving a conservative union bound without assuming the audits
    are independent across policies. ``family_size`` controls the audit's
    Bonferroni confidence intervals and must cover at least these K scenarios.
    Extra family members receive no power or cost plan from this call.

    The calculation matches unweighted ``transport_audit`` t intervals, with
    random sample SD, and uses no finite-population correction. Availability
    restricts feasibility, not the assumed population or the variance formula.
    Labels used for calibration must not reappear in the audit sample. Include
    prior calibration and pilot ratings in ``calibration_labels`` if their cost
    belongs to this evaluation. Cost units follow ``cost_model.oracle_cost``;
    surrogate/model execution costs are not included here.

    A residual mean beyond the margin cannot be repaired by increasing n.
    Finite residual variance and bias assumptions require external justification;
    normality, estimated pilot variance, drift and clustering can invalidate the
    modeled power. Zero SD is an explicit deterministic assumption, not something
    to conclude from observing a small constant pilot.
    """
    if not scenarios or any(not isinstance(k, str) or not k.strip() for k in scenarios):
        raise ValueError("scenarios must contain nonempty policy names")
    if any(not isinstance(s, AuditScenario) for s in scenarios.values()):
        raise TypeError("Each scenario must be an AuditScenario")
    calibration_labels = _integer(calibration_labels, "calibration_labels", 0)
    min_clusters = _integer(min_clusters, "min_clusters", 2)
    max_clusters = _integer(max_clusters, "max_clusters", min_clusters)
    alpha = _finite(alpha, "alpha")
    power = _finite(power, "power")
    if not 0 < alpha < 1 or not 0.5 < power < 1:
        raise ValueError("Require 0 < alpha < 1 and 0.5 < power < 1")
    family_size = _integer(
        len(scenarios) if family_size is None else family_size,
        "family_size",
        len(scenarios),
    )
    oracle_cost = _finite(cost_model.oracle_cost, "oracle_cost")
    if oracle_cost <= 0:
        raise ValueError("oracle_cost must be positive")
    if budget is not None:
        budget = _finite(budget, "budget", 0)
    per_audit_power = 1 - (1 - power) / len(scenarios)
    try:
        per_alpha = alpha / family_size
    except OverflowError as error:
        raise ValueError("family_size exceeds numerical precision") from error
    if 1 - per_alpha / 2 == 1 or per_audit_power == 1:
        raise ValueError("Family confidence or power exceeds numerical precision")
    policies: dict[str, PolicyAuditPlan] = {}
    for policy, scenario in scenarios.items():
        available = scenario.available_clusters
        available_power = (
            None
            if available is None
            else (
                _pass_probability(available, scenario, per_alpha)
                if available >= min_clusters
                else 0.0
            )
        )
        required: int | None = None
        reason = "planned"
        if abs(scenario.assumed_bias) > scenario.delta_max or (
            abs(scenario.assumed_bias) == scenario.delta_max
            and scenario.residual_sd > 0
        ):
            reason = "assumed_mean_outside_margin"
        else:
            low, high = min_clusters - 1, min_clusters
            while (
                _pass_probability(high, scenario, per_alpha) < per_audit_power
                and high < max_clusters
            ):
                low, high = high, min(2 * high, max_clusters)
            if _pass_probability(high, scenario, per_alpha) < per_audit_power:
                reason = "search_limit_reached"
            else:
                while high - low > 1:
                    middle = (low + high) // 2
                    if (
                        _pass_probability(middle, scenario, per_alpha)
                        >= per_audit_power
                    ):
                        high = middle
                    else:
                        low = middle
                required = high
        policies[policy] = PolicyAuditPlan(
            scenario=scenario,
            required_clusters=required,
            modeled_power=(
                None
                if required is None
                else _pass_probability(required, scenario, per_alpha)
            ),
            power_at_available=available_power,
            within_available=(
                None
                if available is None
                else required is not None and required <= available
            ),
            required_labels=(
                None if required is None else required * scenario.labels_per_cluster
            ),
            reason=reason,
        )
    audit_labels = (
        None
        if any(p.required_labels is None for p in policies.values())
        else sum(
            float(p.required_labels)
            for p in policies.values()
            if p.required_labels is not None
        )
    )
    total = None if audit_labels is None else calibration_labels + audit_labels
    cost = None if total is None else oracle_cost * total
    return AuditBudgetPlan(
        policies=policies,
        calibration_labels=calibration_labels,
        required_audit_labels=audit_labels,
        required_total_labels=total,
        human_label_cost=cost,
        budget=budget,
        within_budget=None if budget is None or cost is None else cost <= budget,
        alpha=alpha,
        family_size=family_size,
        family_power=power,
        per_audit_power=per_audit_power,
        min_clusters=min_clusters,
    )
