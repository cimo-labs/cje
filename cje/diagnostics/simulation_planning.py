"""Simulation-based variance model and planning interface.

Allows users to get a variance model without real data by specifying
judge quality (isotonic R²). This enables planning before data collection.

Two approaches to get a FittedVarianceModel:
1. **Pilot-based**: fit_variance_model(fresh_draws) - from real data
2. **Simulation-based**: simulate_variance_model(r2) - from judge quality estimate

Both return FittedVarianceModel, which plugs into plan_evaluation() or plan_for_mde().

Usage:
    from cje import simulate_variance_model, plan_evaluation, CostModel

    # Get variance model via simulation
    variance_model = simulate_variance_model(r2=0.7)

    # Use with standard planning functions
    cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
    plan = plan_evaluation(budget=5000, variance_model=variance_model, cost_model=cost)

    # If you only know correlation, convert first
    from cje import correlation_to_r2
    r2 = correlation_to_r2(0.8)  # Pearson r → isotonic R²
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import logging
import numpy as np

from .planning import (
    FittedVarianceModel,
    EvaluationPlan,
    CostModel,
    plan_evaluation,
    _PLANNING_MEASUREMENT_CONFIG,
    _build_measurement_grid,
    _fit_variance_model_from_measurements,
)

logger = logging.getLogger(__name__)


@dataclass
class SimulationPlanningResult:
    """Result of simulation-based planning.

    Contains the planning result plus diagnostics that help users understand
    *why* the recommendation was made.

    Attributes:
        plan: The actual evaluation plan with allocation and MDE.
        variance_model: Fitted variance model used for planning.
        r2: Isotonic R² (judge quality) used for simulation.
        eval_variance_fraction: Fraction of planned variance from σ²_eval/n.
        cal_variance_fraction: Fraction of planned variance from σ²_cal/m.
        scenario_fingerprint: Inputs that define the simulated planning scenario.
    """

    plan: EvaluationPlan
    variance_model: FittedVarianceModel
    r2: float
    eval_variance_fraction: float
    cal_variance_fraction: float
    scenario_fingerprint: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary of planning result.

        Returns:
            Multi-line string with key planning metrics.
        """
        return (
            f"Simulation Planning Result (R² = {self.r2:.2f})\n"
            f"  Planned variance decomposition: {self.eval_variance_fraction:.0%} eval, "
            f"{self.cal_variance_fraction:.0%} calibration\n"
            f"{self.plan.summary()}"
        )

    def explain(self) -> str:
        """Educational explanation of the planning result.

        Helps users understand how judge quality affects the recommendation.

        Returns:
            Multi-line string with educational content about the result.
        """
        # Judge quality tier
        if self.r2 >= 0.85:
            quality_tier = "excellent"
            quality_desc = (
                "Your judge is highly predictive of oracle values. "
                "Calibration uncertainty is low, so you can use mostly surrogate "
                "scores with modest oracle investment for calibration."
            )
        elif self.r2 >= 0.65:
            quality_tier = "good"
            quality_desc = (
                "Your judge explains most oracle variance. "
                "A moderate oracle investment balances calibration accuracy "
                "with evaluation coverage."
            )
        elif self.r2 >= 0.40:
            quality_tier = "moderate"
            quality_desc = (
                "Your judge captures some but not most oracle variance. "
                "More oracle labels are needed to reduce calibration uncertainty."
            )
        else:
            quality_tier = "low"
            quality_desc = (
                "Your judge has limited predictive power. "
                "Consider collecting more oracle labels or improving the judge. "
                "Heavy oracle investment is recommended."
            )

        # Variance interpretation
        if self.cal_variance_fraction > 0.6:
            variance_insight = (
                "Calibration variance dominates → investing in more oracle labels "
                "will have higher impact than more surrogate samples."
            )
        elif self.eval_variance_fraction > 0.7:
            variance_insight = (
                "Evaluation variance dominates → more surrogate samples will have "
                "higher impact than more oracle labels."
            )
        else:
            variance_insight = (
                "Variance is balanced between evaluation and calibration → "
                "the optimal allocation balances both investments."
            )

        return (
            f"With R² = {self.r2:.2f}, your judge quality is {quality_tier}.\n"
            f"{quality_desc}\n\n"
            f"Variance share at the planned allocation: "
            f"{self.eval_variance_fraction:.0%} eval, "
            f"{self.cal_variance_fraction:.0%} calibration\n"
            f"→ {variance_insight}\n\n"
            f"Optimal allocation: m/n = {self.plan.oracle_fraction:.0%} oracle fraction\n"
            f"Achievable MDE: {self.plan.mde:.1%} ({self.plan.power:.0%} power)"
        )


def correlation_to_r2(correlation: float, relationship: str = "linear") -> float:
    """Convert Pearson correlation to approximate isotonic R².

    For linear relationships, isotonic R² ≈ r² (correlation squared).
    For monotone nonlinear relationships (e.g., Y = sqrt(S)), isotonic R²
    will generally be higher than r².

    Args:
        correlation: Pearson correlation coefficient (-1 to 1).
        relationship: Type of relationship ("linear" or "monotone").
            - "linear": Returns r² (correlation squared)
            - "monotone": Returns a midpoint-to-1 heuristic that assumes real
              monotone nonlinearity. WARNING: this heuristic INFLATES R² if the
              relationship is actually linear — only use "monotone" when you
              have evidence of nonlinearity; otherwise use the lower "linear"
              estimate.

    Returns:
        Approximate isotonic R² (0 to 1).

    Example:
        >>> correlation_to_r2(0.7)  # Linear relationship
        0.49
        >>> correlation_to_r2(0.7, "monotone")  # Nonlinear monotone
        0.745  # Midpoint between r²=0.49 and 1.0
    """
    if not -1 <= correlation <= 1:
        raise ValueError(f"Correlation must be in [-1, 1], got {correlation}")

    r2_linear = correlation**2

    if relationship == "linear":
        return r2_linear
    elif relationship == "monotone":
        # For monotone nonlinear relationships, isotonic R² is typically higher
        # than r². Use the midpoint between r² and 1 as a rough heuristic; note
        # this overstates R² when the relationship is actually linear.
        return min(1.0, r2_linear + 0.5 * (1 - r2_linear))
    else:
        raise ValueError(
            f"relationship must be 'linear' or 'monotone', got {relationship}"
        )


def _calibrated_mixing_weight(
    r2: float,
    seed: int,
    n_probe: int = 20000,
    tol: float = 0.02,
) -> float:
    """Find the internal mixing weight whose BINARIZED labels realize the
    requested isotonic R² against the judge.

    The naive mixing judge = √r2·oracle + √(1−r2)·noise targets the CONTINUOUS
    oracle, but the generator binarizes labels at 0.5, which attenuates the
    realized isotonic R² 15-25% below the request. This bisects on an internal
    r2_eff ∈ (r2, min(1, 2·r2)] such that the realized isotonic R² of the
    binarized labels against the judge (measured on a large seeded probe
    sample) hits the requested r2 within ±tol. Deterministic given seed.
    """
    if r2 <= 0.0 or r2 >= 1.0:
        return float(r2)

    from sklearn.isotonic import IsotonicRegression

    probe_rng = np.random.default_rng(seed)
    oracle = probe_rng.uniform(0, 1, n_probe)
    noise = probe_rng.uniform(0, 1, n_probe)
    labels = (oracle > 0.5).astype(float)
    ss_tot = float(np.sum((labels - labels.mean()) ** 2))

    def realized_isotonic_r2(weight: float) -> float:
        judge = np.sqrt(weight) * oracle + np.sqrt(1 - weight) * noise
        iso = IsotonicRegression(out_of_bounds="clip")
        pred = iso.fit(judge, labels).predict(judge)
        ss_res = float(np.sum((labels - pred) ** 2))
        return 1 - ss_res / ss_tot

    lo, hi = r2, min(1.0, 2.0 * r2)
    if realized_isotonic_r2(hi) < r2:
        # Attenuation larger than the 2x interval allows; widen to the maximum
        # (realized R² at weight 1.0 is exactly 1: labels are monotone in judge).
        hi = 1.0

    for _ in range(10):
        mid = 0.5 * (lo + hi)
        realized = realized_isotonic_r2(mid)
        if abs(realized - r2) <= tol:
            return mid
        if realized < r2:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def _generate_synthetic_data(
    n_total: int,
    oracle_fraction: float,
    r2: float,
    seed: int,
) -> List[dict]:
    """Generate synthetic data with specified judge quality.

    Creates synthetic judge scores and binary oracle labels where the isotonic
    R² between judge and the BINARIZED labels is approximately the specified
    value.

    Math note: the judge mixes the continuous oracle with independent uniform
    noise, judge = √w·oracle + √(1−w)·noise, with oracle, noise ~ U(0, 1). This
    gives Var(judge|oracle) = (1 − w)/12 (the uniform-variance factor), and the
    R² of the judge against the CONTINUOUS oracle equals w. Labels are
    binarized at 0.5 (mimicking binary KPIs), which attenuates the realized
    isotonic R² below w — so the internal weight w is calibrated numerically
    (see _calibrated_mixing_weight) to deliver the requested r2 on the
    binarized labels.

    Args:
        n_total: Total number of samples to generate.
        oracle_fraction: Fraction of samples to have oracle labels.
        r2: Target isotonic R² of the binarized labels given the judge.
        seed: Random seed for reproducibility.

    Returns:
        List of sample dicts with prompt_id, judge_score, and optionally oracle_label.
    """
    # Calibrate the mixing weight so the binarized labels realize the
    # requested isotonic R² (deterministic given seed).
    r2_eff = _calibrated_mixing_weight(r2, seed)

    rng = np.random.default_rng(seed)

    # Generate oracle labels (ground truth) as uniform [0, 1]
    oracle_values = rng.uniform(0, 1, n_total)

    # Generate judge scores from the calibrated mixture
    noise = rng.uniform(0, 1, n_total)
    judge_scores = np.sqrt(r2_eff) * oracle_values + np.sqrt(1 - r2_eff) * noise

    # Normalize judge scores to [0, 1]
    judge_scores = (judge_scores - judge_scores.min()) / (
        judge_scores.max() - judge_scores.min() + 1e-10
    )

    # Randomly select which samples get oracle labels
    n_oracle = int(n_total * oracle_fraction)
    oracle_indices = set(rng.choice(n_total, size=n_oracle, replace=False))

    # Build sample records
    records = []
    for i in range(n_total):
        record = {
            "prompt_id": f"prompt_{i}",
            "judge_score": float(judge_scores[i]),
        }
        if i in oracle_indices:
            # Convert continuous oracle to binary (like real data)
            record["oracle_label"] = 1 if oracle_values[i] > 0.5 else 0
        records.append(record)

    return records


# =============================================================================
# Public API: Core Primitive
# =============================================================================


def simulate_variance_model(
    r2: float,
    n_total: int = 1000,
    oracle_fraction: float = 0.4,
    n_replicates: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> FittedVarianceModel:
    """Get a variance model from judge quality (R²) without real data.

    Runs a simulation to estimate variance components.

    For production planning decisions, prefer fit_variance_model() with real pilot
    data when available.

    Args:
        r2: Judge quality as isotonic R² (0 to 1). This is the fraction of oracle
            variance explained by the judge after isotonic calibration.
            - 0.9+: Excellent judge (minimal calibration uncertainty)
            - 0.7-0.9: Good judge (moderate calibration uncertainty)
            - 0.5-0.7: Moderate judge (significant calibration uncertainty)
            - <0.5: Weak judge (high calibration uncertainty)
        n_total: Simulated dataset size (default 1000).
        oracle_fraction: Fraction with oracle labels (default 0.4).
        n_replicates: Subsample replicates per measurement (default 50).
        seed: Random seed for reproducibility.
        verbose: Print progress and diagnostics.

    Returns:
        FittedVarianceModel that can be used with plan_evaluation() or plan_for_mde().

    Raises:
        ValueError: If r2 is not in [0, 1].

    Example:
        >>> from cje import simulate_variance_model, plan_evaluation, CostModel
        >>> # Get variance model
        >>> variance_model = simulate_variance_model(r2=0.7)
        >>> # Use with standard planning
        >>> cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
        >>> plan = plan_evaluation(budget=5000, variance_model=variance_model, cost_model=cost)
        >>> print(f"MDE: {plan.mde:.1%}")
    """
    from ..interface.analysis import analyze_dataset

    if not 0 <= r2 <= 1:
        raise ValueError(f"r2 must be in [0, 1], got {r2}")

    rng = np.random.default_rng(seed)

    if verbose:
        print(f"Running simulation (R²={r2:.2f}, n={n_total})...")

    # Generate base synthetic data
    base_data = _generate_synthetic_data(n_total, oracle_fraction, r2, seed)

    # Measurement grid: derived exactly the way fit_variance_model does
    # (post-0.4.0) — m levels come from the grid's own scale, not the oracle
    # pool, so the orthogonal grid survives large oracle fractions.
    n_grid = sorted(
        set(
            n
            for n in [
                max(50, int(n_total * 0.2)),
                max(100, int(n_total * 0.4)),
                max(150, int(n_total * 0.6)),
            ]
            if n <= n_total * 0.8
        )
    )
    if len(n_grid) < 2:
        raise ValueError(
            f"Need at least 2 valid n values, but only {len(n_grid)} fit in the "
            f"simulated dataset (n_total={n_total}). Increase n_total."
        )
    n_oracle_available = int(n_total * oracle_fraction)
    measurement_points = _build_measurement_grid(
        n_grid, [0.15, 0.25, 0.40], n_oracle_available
    )

    oracle_records = [r for r in base_data if "oracle_label" in r]
    non_oracle_records = [r for r in base_data if "oracle_label" not in r]

    measurements: List[Tuple[int, int, float]] = []

    for n, m in measurement_points:
        if verbose:
            print(f"  Measuring n={n}, m={m}...", end="", flush=True)

        se_values = []
        for rep in range(n_replicates):
            # Sample exactly m labeled records (keeping their labels)
            oracle_idx = sorted(
                int(i) for i in rng.choice(len(oracle_records), size=m, replace=False)
            )
            oracle_idx_set = set(oracle_idx)
            sampled_oracle = [oracle_records[i] for i in oracle_idx]

            # The remaining pool supplies the n - m judge-only records; any
            # oracle labels are stripped so analyze_dataset sees exactly
            # (n, m). Labeling is random by construction, so stripping
            # preserves the randomized labeling mechanism. (Previously the
            # unlabeled pool alone
            # supplied these records, so e.g. at oracle_fraction=1.0 the
            # recorded (n, m) was never actually delivered.)
            remaining = [
                oracle_records[i]
                for i in range(len(oracle_records))
                if i not in oracle_idx_set
            ] + non_oracle_records
            extra_idx = rng.choice(len(remaining), size=n - m, replace=False)
            sampled_unlabeled = [
                {
                    "prompt_id": remaining[i]["prompt_id"],
                    "judge_score": remaining[i]["judge_score"],
                }
                for i in extra_idx
            ]

            subsample = sampled_oracle + sampled_unlabeled

            if len(subsample) < 20:
                continue

            try:
                result = analyze_dataset(
                    fresh_draws_data={"synthetic": subsample},
                    verbose=False,
                    estimator_config=_PLANNING_MEASUREMENT_CONFIG,
                )
                if (
                    result.standard_errors is not None
                    and len(result.standard_errors) > 0
                    and result.standard_errors[0] > 0
                ):
                    se_values.append(float(result.standard_errors[0]))
            except Exception as e:
                logger.debug(f"analyze_dataset failed: {e}")
                continue

        if se_values:
            median_se = float(np.median(se_values))
            measurements.append((n, m, median_se**2))
            if verbose:
                print(f" SE={median_se:.4f}")
        elif verbose:
            print(" FAILED")

    if len(measurements) < 3:
        raise ValueError(
            f"Only got {len(measurements)} valid measurements, need at least 3. "
            f"Try increasing n_total or oracle_fraction."
        )

    # Fit via the same NNLS fitter used for pilot data
    model = _fit_variance_model_from_measurements(measurements)

    if verbose:
        print(
            f"Fitted model: σ²_eval={model.sigma2_eval:.4f}, "
            f"σ²_cal={model.sigma2_cal:.4f}, R²={model.r_squared:.2f}"
        )

    return model


# =============================================================================
# Public API: Convenience Wrappers
# =============================================================================


def simulate_planning(
    r2: float,
    budget: float,
    cost_model: CostModel,
    n_total: int = 1000,
    oracle_fraction: float = 0.4,
    n_replicates: int = 50,
    m_min: int = 30,
    power: float = 0.8,
    alpha: float = 0.05,
    seed: int = 42,
    verbose: bool = True,
) -> SimulationPlanningResult:
    """Plan evaluation based on simulated judge quality (convenience wrapper).

    Combines simulate_variance_model() + plan_evaluation() into one call,
    returning additional diagnostics useful for exploration.

    For composable workflows, use simulate_variance_model() directly:
        variance_model = simulate_variance_model(r2=0.7)
        plan = plan_evaluation(budget=5000, variance_model=variance_model, cost_model=cost)

    Args:
        r2: Judge quality as isotonic R² (0 to 1).
        budget: Total budget in cost units (e.g., dollars).
        cost_model: Cost parameters (surrogate_cost, oracle_cost).
        n_total: Simulated dataset size (default 1000).
        oracle_fraction: Fraction with oracle labels in simulation (default 0.4).
        n_replicates: Subsample replicates per measurement (default 50).
        m_min: Minimum oracle labels in the plan (forwarded to
            plan_evaluation; default 30, matching plan_evaluation).
        power: Statistical power for MDE calculation (default 0.8).
        alpha: Significance level for MDE calculation (default 0.05).
        seed: Random seed for reproducibility.
        verbose: Print progress and diagnostics (default True).

    Returns:
        SimulationPlanningResult with plan, variance model, and diagnostics.

    Raises:
        ValueError: If r2 is not in [0, 1].

    Example:
        >>> from cje import simulate_planning, CostModel
        >>> cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
        >>> result = simulate_planning(r2=0.7, budget=5000, cost_model=cost)
        >>> print(result.explain())  # Educational output
        With R² = 0.70, your judge quality is good...
    """
    # Get variance model using the core primitive
    variance_model = simulate_variance_model(
        r2=r2,
        n_total=n_total,
        oracle_fraction=oracle_fraction,
        n_replicates=n_replicates,
        seed=seed,
        verbose=verbose,
    )

    # Plan evaluation
    plan = plan_evaluation(
        budget=budget,
        variance_model=variance_model,
        cost_model=cost_model,
        m_min=m_min,
        power=power,
        alpha=alpha,
    )

    # Attribute variance at the selected allocation. Raw fitted coefficients
    # have different denominators and are not themselves variance shares.
    eval_fraction = plan.eval_variance_fraction
    cal_fraction = plan.cal_variance_fraction

    scenario_fingerprint: Dict[str, Any] = {
        "schema_version": 1,
        "dgp": "binary_oracle_monotone_judge_v1",
        "r2": float(r2),
        "n_total": int(n_total),
        "oracle_fraction": float(oracle_fraction),
        "n_replicates": int(n_replicates),
        "seed": int(seed),
        "budget": float(budget),
        "m_min": int(m_min),
        "power": float(power),
        "alpha": float(alpha),
        "cost_model": {
            "surrogate_cost": float(cost_model.surrogate_cost),
            "oracle_cost": float(cost_model.oracle_cost),
        },
        "variance_measurement": dict(_PLANNING_MEASUREMENT_CONFIG),
    }

    return SimulationPlanningResult(
        plan=plan,
        variance_model=variance_model,
        r2=r2,
        eval_variance_fraction=eval_fraction,
        cal_variance_fraction=cal_fraction,
        scenario_fingerprint=scenario_fingerprint,
    )
