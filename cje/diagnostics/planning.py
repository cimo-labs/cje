"""Budget optimization for oracle/surrogate allocation.

Implements the Square Root Allocation Law from CJE paper Appendix F.

The key insight is that total variance decomposes into two independent components:
    V_total(n, m) = σ²_eval/n + σ²_cal/m

where:
    - n = number of evaluation samples (scored by surrogate)
    - m = number of oracle labels (for calibration)
    - σ²_eval = intrinsic evaluation variance
    - σ²_cal = intrinsic calibration variance

Given a budget B = c_S·n + c_Y·m, the optimal allocation follows the Square Root Law:
    m*/n* = √(c_S/c_Y) · √(σ²_cal/σ²_eval)

Usage:
    from cje.diagnostics import (
        estimate_variance_components,
        compute_optimal_allocation,
        diagnose_allocation_efficiency,
        CostModel,
    )

    # After a pilot run
    result = analyze_dataset(...)

    # Extract variance components from pilot
    sigma2_eval, sigma2_cal = estimate_variance_components(result)

    # Plan optimal allocation for production
    cost_model = CostModel(oracle_cost=16.0)  # 16× cost ratio
    allocation = compute_optimal_allocation(
        budget=1000.0,
        cost_model=cost_model,
        sigma2_eval=sigma2_eval,
        sigma2_cal=sigma2_cal,
    )
    print(allocation.summary())

    # Diagnose current allocation efficiency
    diag = diagnose_allocation_efficiency(result, cost_model)
    print(diag["recommendation"])
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from scipy import stats

from ..data.models import EstimationResult


@dataclass
class CostModel:
    """Cost parameters for budget optimization.

    Attributes:
        surrogate_cost: Cost per surrogate (judge) score (c_S). Default 1.0.
        oracle_cost: Cost per oracle label (c_Y). Default 16.0 (paper's Arena benchmark).

    Example:
        # GPT-4o-mini judge vs GPT-5 oracle (approximately 16× cost ratio)
        cost_model = CostModel(surrogate_cost=1.0, oracle_cost=16.0)

        # Custom costs in dollars
        cost_model = CostModel(surrogate_cost=0.001, oracle_cost=0.50)
    """

    surrogate_cost: float = 1.0
    oracle_cost: float = 16.0

    @property
    def cost_ratio(self) -> float:
        """c_S / c_Y ratio."""
        return self.surrogate_cost / self.oracle_cost


@dataclass
class BudgetAllocation:
    """Result of optimal budget allocation via Square Root Law.

    Attributes:
        n_samples: Optimal number of evaluation samples to collect.
        m_oracle: Optimal number of oracle labels to collect.
        oracle_fraction: Ratio m/n (oracle fraction).
        total_cost: Total cost at optimal allocation.
        expected_se: Expected standard error at optimal allocation.
        calibration_uncertainty_share: ω = Var_cal/Var_total (for diagnostics).
        budget: Input budget (for reproducibility).
        sigma2_eval: Input evaluation variance (for reproducibility).
        sigma2_cal: Input calibration variance (for reproducibility).
        cost_model: Input cost model (for reproducibility).
    """

    n_samples: int
    m_oracle: int
    oracle_fraction: float
    total_cost: float
    expected_se: float
    calibration_uncertainty_share: float

    # Input parameters (for reproducibility)
    budget: float
    sigma2_eval: float
    sigma2_cal: float
    cost_model: CostModel

    def summary(self) -> str:
        """Human-readable allocation summary."""
        return (
            f"Optimal allocation: n={self.n_samples:,}, m={self.m_oracle:,} "
            f"({self.oracle_fraction:.1%} oracle)\n"
            f"Expected SE: {self.expected_se:.4f} | "
            f"Calibration share: {self.calibration_uncertainty_share:.1%}\n"
            f"Total cost: ${self.total_cost:.2f}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "n_samples": self.n_samples,
            "m_oracle": self.m_oracle,
            "oracle_fraction": self.oracle_fraction,
            "total_cost": self.total_cost,
            "expected_se": self.expected_se,
            "calibration_uncertainty_share": self.calibration_uncertainty_share,
            "budget": self.budget,
            "sigma2_eval": self.sigma2_eval,
            "sigma2_cal": self.sigma2_cal,
            "cost_model": {
                "surrogate_cost": self.cost_model.surrogate_cost,
                "oracle_cost": self.cost_model.oracle_cost,
            },
        }


def estimate_variance_components(
    result: EstimationResult,
    policy: Optional[str] = None,
) -> Tuple[float, float]:
    """Extract intrinsic variance components from a pilot run.

    Returns (σ²_eval, σ²_cal) scaled to be sample-size-independent.
    These can be used with compute_optimal_allocation() to plan future experiments.

    The intrinsic variances satisfy:
        Var_total = σ²_eval/n + σ²_cal/m

    So to get sample-size-independent values, we scale:
        σ²_eval = (Var_total - Var_cal) × n
        σ²_cal = Var_cal × m

    Args:
        result: EstimationResult from a pilot run. Must have metadata with
            'n_samples', 'oracle_fraction', and 'se_components'.
        policy: Specific policy to extract variance for. If None, averages
            across all policies.

    Returns:
        (sigma2_eval, sigma2_cal): Intrinsic variance components that can be
            used with compute_optimal_allocation().

    Raises:
        ValueError: If result lacks necessary metadata for extraction.

    Example:
        result = analyze_dataset(logged_data_path="pilot.jsonl", ...)
        sigma2_eval, sigma2_cal = estimate_variance_components(result)
        # Now use these with compute_optimal_allocation()
    """
    # Get sample sizes from metadata
    n = result.metadata.get("n_samples")
    if n is None:
        # Fallback: estimate from result shape
        n = len(result.estimates) * 100

    oracle_fraction = result.metadata.get("oracle_fraction", 0.1)
    m = max(int(n * oracle_fraction), 1)

    # Get variance components from se_components
    se_components = result.metadata.get("se_components", {})
    oracle_var_map = se_components.get("oracle_variance_per_policy", {})

    # Total variance (average across policies if not specified)
    target_policies = result.metadata.get("target_policies", [])

    if policy and policy in target_policies:
        idx = target_policies.index(policy)
        var_total = float(result.standard_errors[idx] ** 2)
        var_cal = float(oracle_var_map.get(policy, 0.0))
    else:
        var_total = float(np.mean(result.standard_errors**2))
        if oracle_var_map:
            var_cal = float(np.mean(list(oracle_var_map.values())))
        else:
            var_cal = 0.0

    # Evaluation variance = total - calibration
    var_eval = max(var_total - var_cal, 1e-10)

    # Scale to intrinsic (sample-size-independent) variances
    sigma2_eval = var_eval * n
    sigma2_cal = var_cal * m

    return sigma2_eval, sigma2_cal


def compute_optimal_allocation(
    budget: float,
    cost_model: CostModel,
    sigma2_eval: float,
    sigma2_cal: float,
) -> BudgetAllocation:
    """Compute optimal (n*, m*) via Square Root Allocation Law.

    Minimizes total variance subject to budget constraint:
        V_total(n, m) = σ²_eval/n + σ²_cal/m
        subject to: c_S·n + c_Y·m = B, m ≤ n

    From CJE paper Appendix F (Proposition 5):
        n* = B·√(σ²_eval/c_S) / (√(c_S·σ²_eval) + √(c_Y·σ²_cal))
        m* = B·√(σ²_cal/c_Y) / (√(c_S·σ²_eval) + √(c_Y·σ²_cal))

    Args:
        budget: Total budget in cost units (e.g., dollars).
        cost_model: Cost parameters (surrogate_cost, oracle_cost).
        sigma2_eval: Intrinsic evaluation variance (from estimate_variance_components).
        sigma2_cal: Intrinsic calibration variance (from estimate_variance_components).

    Returns:
        BudgetAllocation with optimal n*, m* and diagnostic information.

    Example:
        # Plan for $1000 budget with 16× oracle cost ratio
        allocation = compute_optimal_allocation(
            budget=1000.0,
            cost_model=CostModel(oracle_cost=16.0),
            sigma2_eval=1.0,
            sigma2_cal=0.061,
        )
        print(allocation.summary())
        # Optimal allocation: n=..., m=... (X.X% oracle)
    """
    c_S = cost_model.surrogate_cost
    c_Y = cost_model.oracle_cost

    # Handle edge cases
    if sigma2_cal <= 0:
        # No calibration variance → spend all on evaluation
        n_star = int(budget / c_S)
        m_star = 1
    elif sigma2_eval <= 0:
        # No evaluation variance → spend all on calibration
        m_star = int(budget / c_Y)
        n_star = m_star
    else:
        # Square Root Law (paper Proposition 5)
        denom = np.sqrt(c_S * sigma2_eval) + np.sqrt(c_Y * sigma2_cal)

        n_star_float = budget * np.sqrt(sigma2_eval / c_S) / denom
        m_star_float = budget * np.sqrt(sigma2_cal / c_Y) / denom

        # Enforce m ≤ n constraint
        if m_star_float > n_star_float:
            # Binding constraint: m = n, re-solve
            n_star_float = m_star_float = budget / (c_S + c_Y)

        n_star = int(max(n_star_float, 1))
        m_star = int(max(min(m_star_float, n_star), 1))

    # Compute expected variance at optimal allocation
    var_eval = sigma2_eval / n_star if n_star > 0 else np.inf
    var_cal = sigma2_cal / m_star if m_star > 0 else np.inf
    var_total = var_eval + var_cal
    expected_se = float(np.sqrt(var_total))

    # Calibration uncertainty share
    omega = var_cal / var_total if var_total > 0 else 0.0

    return BudgetAllocation(
        n_samples=n_star,
        m_oracle=m_star,
        oracle_fraction=m_star / n_star if n_star > 0 else 0.0,
        total_cost=c_S * n_star + c_Y * m_star,
        expected_se=expected_se,
        calibration_uncertainty_share=float(omega),
        budget=budget,
        sigma2_eval=sigma2_eval,
        sigma2_cal=sigma2_cal,
        cost_model=cost_model,
    )


def diagnose_allocation_efficiency(
    result: EstimationResult,
    cost_model: CostModel,
) -> Dict[str, Any]:
    """Diagnose whether current allocation is efficient.

    Uses the Spend-Balance Rule from CJE paper Appendix F:
        At optimum: ω = Spend_oracle / Spend_total
        where ω = Var_cal / Var_total (calibration uncertainty share)

    Deviations diagnose inefficiency:
        - ω > spend_fraction → UNDER_LABELED (invest in more oracle labels)
        - ω < spend_fraction → OVER_LABELED (evaluate more prompts)

    Args:
        result: EstimationResult from an evaluation run.
        cost_model: Cost parameters for the analysis.

    Returns:
        Dictionary with:
            - status: "UNDER_LABELED", "OVER_LABELED", or "BALANCED"
            - calibration_uncertainty_share: ω = Var_cal/Var_total
            - oracle_spend_fraction: Spend_oracle/Spend_total
            - imbalance: ω - spend_fraction (positive = under-labeled)
            - recommendation: Human-readable suggestion
            - current_n, current_m, current_oracle_fraction: Current allocation

    Example:
        diag = diagnose_allocation_efficiency(result, CostModel())
        if diag["status"] == "UNDER_LABELED":
            print(f"Collect more oracle labels: {diag['recommendation']}")
    """
    # Get variance decomposition
    se_components = result.metadata.get("se_components", {})
    oracle_var_map = se_components.get("oracle_variance_per_policy", {})

    var_total = float(np.mean(result.standard_errors**2))
    if oracle_var_map:
        var_cal = float(np.mean(list(oracle_var_map.values())))
    else:
        var_cal = 0.0

    omega = var_cal / var_total if var_total > 0 else 0.0

    # Get actual spend from metadata
    n = result.metadata.get("n_samples", 1000)
    oracle_fraction = result.metadata.get("oracle_fraction", 0.1)
    m = int(n * oracle_fraction)

    spend_oracle = cost_model.oracle_cost * m
    spend_total = cost_model.surrogate_cost * n + spend_oracle
    spend_fraction = spend_oracle / spend_total if spend_total > 0 else 0.0

    imbalance = omega - spend_fraction

    # Diagnose with 10% tolerance
    if imbalance > 0.1:
        status = "UNDER_LABELED"
        recommendation = (
            "Calibration uncertainty dominates. "
            f"Collect more oracle labels (current: {oracle_fraction:.1%}, "
            f"target: ~{omega:.1%})."
        )
    elif imbalance < -0.1:
        status = "OVER_LABELED"
        recommendation = (
            "Sampling noise dominates. " "Evaluate more prompts with the surrogate."
        )
    else:
        status = "BALANCED"
        recommendation = "Allocation is near-optimal."

    return {
        "status": status,
        "calibration_uncertainty_share": omega,
        "oracle_spend_fraction": spend_fraction,
        "imbalance": imbalance,
        "recommendation": recommendation,
        "current_n": n,
        "current_m": m,
        "current_oracle_fraction": oracle_fraction,
    }


def compute_mde_contours(
    n_range: List[int],
    oracle_fractions: List[float],
    sigma2_eval: float,
    sigma2_cal: float,
    power: float = 0.8,
    alpha: float = 0.05,
) -> np.ndarray:
    """Compute MDE (minimum detectable effect) grid for sample size planning.

    MDE is the smallest effect size detectable at given power and significance:
        MDE = (z_{1-α/2} + z_{power}) · √2 · SE

    This produces a grid like Figure 8 in the CJE paper, showing how MDE
    varies with sample size and oracle fraction.

    Args:
        n_range: List of sample sizes to evaluate.
        oracle_fractions: List of oracle fractions to evaluate.
        sigma2_eval: Intrinsic evaluation variance (from estimate_variance_components).
        sigma2_cal: Intrinsic calibration variance (from estimate_variance_components).
        power: Statistical power (default 0.8 = 80%).
        alpha: Significance level (default 0.05 for two-sided 95% CI).

    Returns:
        mde_grid: Array of shape (len(n_range), len(oracle_fractions)) containing
            MDE values. Lower is better (smaller effects detectable).

    Example:
        # Create MDE contour data
        mde_grid = compute_mde_contours(
            n_range=[500, 1000, 2000, 5000],
            oracle_fractions=[0.05, 0.10, 0.25, 0.50],
            sigma2_eval=1.0,
            sigma2_cal=0.061,
        )

        # Find minimum MDE
        print(f"Best MDE: {mde_grid.min():.4f}")

        # Plot with matplotlib
        import matplotlib.pyplot as plt
        plt.contourf(oracle_fractions, n_range, mde_grid)
        plt.colorbar(label="MDE")
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    c = z_alpha + z_power

    mde_grid = np.zeros((len(n_range), len(oracle_fractions)))

    for i, n in enumerate(n_range):
        for j, frac in enumerate(oracle_fractions):
            m = max(int(n * frac), 1)
            var_total = sigma2_eval / n + sigma2_cal / m
            se = np.sqrt(var_total)
            mde_grid[i, j] = c * np.sqrt(2) * se

    return mde_grid
