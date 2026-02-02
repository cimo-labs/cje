"""Simulation-based variance model and planning interface.

Allows users to get a variance model without real data by specifying
judge quality (isotonic R²). This enables planning before data collection.

Two approaches to get a FittedVarianceModel:
1. **Pilot-based**: fit_variance_model(fresh_draws) - from real data
2. **Simulation-based**: simulate_variance_model(r2) - from judge quality estimate

Both return FittedVarianceModel, which plugs into plan_evaluation() or plan_for_mde().

Usage:
    from cje import simulate_variance_model, plan_evaluation, CostModel

    # Get variance model via simulation (takes 2-4 minutes)
    variance_model = simulate_variance_model(r2=0.7)

    # Use with standard planning functions
    cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
    plan = plan_evaluation(budget=5000, variance_model=variance_model, cost_model=cost)

    # If you only know correlation, convert first
    from cje import correlation_to_r2
    r2 = correlation_to_r2(0.8)  # Pearson r → isotonic R²
"""

from dataclasses import dataclass
from typing import List
import logging
import numpy as np

from .planning import (
    FittedVarianceModel,
    EvaluationPlan,
    CostModel,
    plan_evaluation,
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
        eval_variance_fraction: Fraction of total variance from evaluation (σ²_eval share).
        cal_variance_fraction: Fraction of total variance from calibration (σ²_cal share).
    """

    plan: EvaluationPlan
    variance_model: FittedVarianceModel
    r2: float
    eval_variance_fraction: float
    cal_variance_fraction: float

    def summary(self) -> str:
        """Human-readable summary of planning result.

        Returns:
            Multi-line string with key planning metrics.
        """
        return (
            f"Simulation Planning Result (R² = {self.r2:.2f})\n"
            f"  Variance decomposition: {self.eval_variance_fraction:.0%} eval, "
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
            f"Variance share: {self.eval_variance_fraction:.0%} eval, "
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
            - "monotone": Returns conservative estimate assuming some nonlinearity

    Returns:
        Approximate isotonic R² (0 to 1).

    Example:
        >>> correlation_to_r2(0.7)  # Linear relationship
        0.49
        >>> correlation_to_r2(0.7, "monotone")  # Nonlinear monotone
        0.735  # Higher because isotonic captures nonlinear monotone patterns
    """
    if not -1 <= correlation <= 1:
        raise ValueError(f"Correlation must be in [-1, 1], got {correlation}")

    r2_linear = correlation**2

    if relationship == "linear":
        return r2_linear
    elif relationship == "monotone":
        # For monotone nonlinear relationships, isotonic R² is typically higher.
        # Use a conservative estimate: midpoint between r² and 1.
        # This reflects that isotonic regression captures any monotone pattern.
        return min(1.0, r2_linear + 0.5 * (1 - r2_linear))
    else:
        raise ValueError(
            f"relationship must be 'linear' or 'monotone', got {relationship}"
        )


def _generate_synthetic_data(
    n_total: int,
    oracle_fraction: float,
    r2: float,
    seed: int,
) -> List[dict]:
    """Generate synthetic data with specified judge quality.

    Creates synthetic judge scores and oracle labels where the isotonic R²
    between judge and oracle is approximately the specified value.

    Args:
        n_total: Total number of samples to generate.
        oracle_fraction: Fraction of samples to have oracle labels.
        r2: Target isotonic R² (judge quality).
        seed: Random seed for reproducibility.

    Returns:
        List of sample dicts with prompt_id, judge_score, and optionally oracle_label.
    """
    rng = np.random.default_rng(seed)

    # Generate oracle labels (ground truth) as uniform [0, 1]
    oracle_values = rng.uniform(0, 1, n_total)

    # Generate judge scores with specified R² relationship
    # Higher R² means judge is more predictive of oracle
    # Use: judge = oracle * sqrt(r2) + noise * sqrt(1 - r2) (linear case)
    # This gives Var(judge|oracle) = 1 - r2, so R² ≈ r2
    noise = rng.uniform(0, 1, n_total)
    judge_scores = np.sqrt(r2) * oracle_values + np.sqrt(1 - r2) * noise

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
    n_replicates: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> FittedVarianceModel:
    """Get a variance model from judge quality (R²) without real data.

    Runs a simulation to estimate variance components. Takes 2-4 minutes.

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
        n_replicates: Bootstrap replicates per measurement (default 5).
        seed: Random seed for reproducibility.
        verbose: Print progress and diagnostics.

    Returns:
        FittedVarianceModel that can be used with plan_evaluation() or plan_for_mde().

    Raises:
        ValueError: If r2 is not in [0, 1].

    Example:
        >>> from cje import simulate_variance_model, plan_evaluation, CostModel
        >>> # Get variance model (takes 2-4 minutes)
        >>> variance_model = simulate_variance_model(r2=0.7)
        >>> # Use with standard planning
        >>> cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
        >>> plan = plan_evaluation(budget=5000, variance_model=variance_model, cost_model=cost)
        >>> print(f"MDE: {plan.mde:.1%}")
    """
    from ..interface.analysis import analyze_dataset
    from scipy.optimize import nnls

    if not 0 <= r2 <= 1:
        raise ValueError(f"r2 must be in [0, 1], got {r2}")

    rng = np.random.default_rng(seed)

    if verbose:
        print(f"Running simulation (R²={r2:.2f}, n={n_total})...")

    # Generate base synthetic data
    base_data = _generate_synthetic_data(n_total, oracle_fraction, r2, seed)

    # Measurement grid: vary n and m independently
    n_grid = [
        max(50, int(n_total * 0.2)),
        max(100, int(n_total * 0.4)),
        max(150, int(n_total * 0.6)),
    ]
    n_oracle_available = int(n_total * oracle_fraction)
    m_values = [
        max(10, int(n_oracle_available * 0.3)),
        max(20, int(n_oracle_available * 0.5)),
        max(30, int(n_oracle_available * 0.7)),
    ]

    measurements: List[tuple] = []

    for n in n_grid:
        for m in m_values:
            if m >= n or m > n_oracle_available:
                continue

            if verbose:
                print(f"  Measuring n={n}, m={m}...", end="", flush=True)

            se_values = []
            for rep in range(n_replicates):
                # Subsample data
                oracle_records = [r for r in base_data if "oracle_label" in r]
                non_oracle_records = [r for r in base_data if "oracle_label" not in r]

                # Sample using indices to satisfy mypy
                n_oracle_sample = min(m, len(oracle_records))
                oracle_indices = rng.choice(
                    len(oracle_records), size=n_oracle_sample, replace=False
                )
                sampled_oracle = [oracle_records[i] for i in oracle_indices]

                n_non_oracle = min(n - len(sampled_oracle), len(non_oracle_records))
                if n_non_oracle > 0:
                    non_oracle_indices = rng.choice(
                        len(non_oracle_records), size=n_non_oracle, replace=False
                    )
                    sampled_non_oracle = [
                        non_oracle_records[i] for i in non_oracle_indices
                    ]
                else:
                    sampled_non_oracle = []

                subsample = sampled_oracle + sampled_non_oracle

                if len(subsample) < 20:
                    continue

                try:
                    result = analyze_dataset(
                        fresh_draws_data={"synthetic": subsample},
                        verbose=False,
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

    # Fit model via NNLS
    X = np.array([[1 / n, 1 / m] for n, m, _ in measurements])
    y = np.array([var for _, _, var in measurements])

    coeffs, _ = nnls(X, y)
    sigma2_eval, sigma2_cal = coeffs

    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if verbose:
        print(
            f"Fitted model: σ²_eval={sigma2_eval:.4f}, σ²_cal={sigma2_cal:.4f}, R²={r_squared:.2f}"
        )

    return FittedVarianceModel(
        sigma2_eval=float(sigma2_eval),
        sigma2_cal=float(sigma2_cal),
        r_squared=float(r_squared),
        n_measurements=len(measurements),
    )


# =============================================================================
# Public API: Convenience Wrappers
# =============================================================================


def simulate_planning(
    r2: float,
    budget: float,
    cost_model: CostModel,
    n_total: int = 1000,
    oracle_fraction: float = 0.4,
    n_replicates: int = 5,
    power: float = 0.8,
    alpha: float = 0.05,
    seed: int = 42,
    verbose: bool = True,
) -> SimulationPlanningResult:
    """Plan evaluation based on simulated judge quality (convenience wrapper).

    Combines simulate_variance_model() + plan_evaluation() into one call,
    returning additional diagnostics useful for exploration. Takes 2-4 minutes.

    For composable workflows, use simulate_variance_model() directly:
        variance_model = simulate_variance_model(r2=0.7)
        plan = plan_evaluation(budget=5000, variance_model=variance_model, cost_model=cost)

    Args:
        r2: Judge quality as isotonic R² (0 to 1).
        budget: Total budget in cost units (e.g., dollars).
        cost_model: Cost parameters (surrogate_cost, oracle_cost).
        n_total: Simulated dataset size (default 1000).
        oracle_fraction: Fraction with oracle labels in simulation (default 0.4).
        n_replicates: Bootstrap replicates per measurement (default 5).
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
        power=power,
        alpha=alpha,
    )

    # Compute variance fractions
    total_var = variance_model.sigma2_eval + variance_model.sigma2_cal
    if total_var > 0:
        eval_fraction = variance_model.sigma2_eval / total_var
        cal_fraction = variance_model.sigma2_cal / total_var
    else:
        eval_fraction = 0.5
        cal_fraction = 0.5

    return SimulationPlanningResult(
        plan=plan,
        variance_model=variance_model,
        r2=r2,
        eval_variance_fraction=eval_fraction,
        cal_variance_fraction=cal_fraction,
    )


def simulate_planning_sweep(
    r2_values: List[float],
    budget: float,
    cost_model: CostModel,
    n_total: int = 1000,
    oracle_fraction: float = 0.4,
    n_replicates: int = 5,
    power: float = 0.8,
    alpha: float = 0.05,
    seed: int = 42,
    verbose: bool = False,
) -> List[SimulationPlanningResult]:
    """Run planning across multiple R² values for sensitivity analysis.

    Note: Each R² value runs a simulation (~2-4 min), so sweeping across
    N values takes N × 2-4 minutes. Consider using fewer R² values or
    running overnight for large sweeps.

    Args:
        r2_values: List of R² values to evaluate.
        budget: Total budget in cost units.
        cost_model: Cost parameters.
        n_total: Simulated dataset size per R² value (default 1000).
        oracle_fraction: Fraction with oracle labels (default 0.4).
        n_replicates: Bootstrap replicates per measurement (default 5).
        power: Statistical power for MDE calculation.
        alpha: Significance level for MDE calculation.
        seed: Base random seed (incremented for each R² value).
        verbose: Print progress per simulation (default False for sweeps).

    Returns:
        List of SimulationPlanningResult, one per R² value.

    Example:
        >>> from cje import simulate_planning_sweep, CostModel
        >>> cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
        >>> # Note: This takes ~6-12 minutes for 3 R² values
        >>> results = simulate_planning_sweep([0.5, 0.7, 0.9], budget=5000, cost_model=cost)
        >>> for r in results:
        ...     print(f"R²={r.r2}: MDE={r.plan.mde:.1%}, oracle={r.plan.oracle_fraction:.0%}")
    """
    if len(r2_values) > 0:
        print(
            f"Running {len(r2_values)} simulations (~{len(r2_values) * 2}-{len(r2_values) * 4} min total)..."
        )

    results = []
    for i, r2 in enumerate(r2_values):
        if not verbose:
            print(
                f"  Simulating R²={r2:.2f} ({i+1}/{len(r2_values)})...",
                end="",
                flush=True,
            )
        result = simulate_planning(
            r2=r2,
            budget=budget,
            cost_model=cost_model,
            n_total=n_total,
            oracle_fraction=oracle_fraction,
            n_replicates=n_replicates,
            power=power,
            alpha=alpha,
            seed=seed + i * 1000,  # Different seed per R² value
            verbose=verbose,
        )
        if not verbose:
            print(f" MDE={result.plan.mde:.1%}")
        results.append(result)
    return results
