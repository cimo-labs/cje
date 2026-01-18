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
    from cje.data.fresh_draws import load_fresh_draws_auto, discover_policies_from_fresh_draws
    from cje.diagnostics import (
        fit_variance_model_from_pilot,
        compute_optimal_allocation,
        CostModel,
    )

    # Load pilot data
    policies = discover_policies_from_fresh_draws(pilot_dir)
    fresh_draws_dict = {p: load_fresh_draws_auto(pilot_dir, p) for p in policies}

    # Fit variance model from empirical measurements
    variance_model = fit_variance_model_from_pilot(fresh_draws_dict, verbose=True)
    print(f"R² = {variance_model.r_squared:.3f}")

    # Plan optimal allocation for production
    cost_model = CostModel(oracle_cost=16.0)  # 16× cost ratio
    allocation = compute_optimal_allocation(
        budget=1000.0,
        cost_model=cost_model,
        variance_model=variance_model,
    )
    print(allocation.summary())
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import logging
import numpy as np
from scipy import stats

from ..data.models import EstimationResult
from ..data.fresh_draws import FreshDrawDataset, FreshDrawSample

logger = logging.getLogger(__name__)


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


@dataclass
class FittedVarianceModel:
    """Result of fitting Var = sigma2_eval/n + sigma2_cal/m to empirical data.

    This model is fitted by measuring variance at multiple (n, m) allocations
    and using linear regression to estimate the intrinsic variance components.
    Unlike the dual-bootstrap decomposition, this approach directly measures
    variance and can extrapolate to allocations larger than the pilot.

    Attributes:
        sigma2_eval: Intrinsic evaluation variance (sampling variance component).
        sigma2_cal: Intrinsic calibration variance (oracle labeling component).
        r_squared: R² of the fit, indicating how well the 1/n + 1/m form fits.
        n_measurements: Number of (n, m, var) measurements used for fitting.
    """

    sigma2_eval: float
    sigma2_cal: float
    r_squared: float
    n_measurements: int

    def predict_variance(self, n: int, m: int) -> float:
        """Predict variance at allocation (n, m)."""
        return self.sigma2_eval / n + self.sigma2_cal / m

    def predict_se(self, n: int, m: int) -> float:
        """Predict standard error at allocation (n, m)."""
        return float(np.sqrt(self.predict_variance(n, m)))

    def summary(self) -> str:
        """Human-readable summary of fitted model."""
        return (
            f"FittedVarianceModel (R²={self.r_squared:.3f}, n={self.n_measurements} measurements)\n"
            f"  σ²_eval = {self.sigma2_eval:.6f}\n"
            f"  σ²_cal  = {self.sigma2_cal:.6f}"
        )


@dataclass
class EvaluationPlan:
    """Result of evaluation planning with MDE-centric outputs.

    This is the primary output of plan_evaluation() and plan_for_mde().
    It provides both the optimal allocation and the achievable MDE for
    pairwise policy comparisons.

    Attributes:
        n_samples: Number of evaluation samples (prompts) to collect.
        m_oracle: Number of oracle labels to collect.
        total_cost: Total cost of this allocation.
        mde: Minimum detectable effect at specified power (for pairwise A vs B).
            Uses conservative √2 factor assuming independent samples.
        se_level: Standard error for single policy estimate.
        power: Statistical power used for MDE calculation.
        alpha: Significance level used for MDE calculation.
        sigma2_eval: Evaluation variance component from fitted model.
        sigma2_cal: Calibration variance component from fitted model.
        cost_model: Cost model used for optimization.
    """

    n_samples: int
    m_oracle: int
    total_cost: float
    mde: float
    se_level: float
    power: float
    alpha: float
    sigma2_eval: float
    sigma2_cal: float
    cost_model: CostModel

    @property
    def oracle_fraction(self) -> float:
        """Fraction of samples with oracle labels (m/n)."""
        return self.m_oracle / self.n_samples if self.n_samples > 0 else 0.0

    @property
    def se_comparison(self) -> float:
        """SE for pairwise comparison (conservative √2 × SE_level)."""
        return float(np.sqrt(2)) * self.se_level

    def mde_at_power(self, power: float) -> float:
        """Compute MDE at a different power level.

        Args:
            power: Desired statistical power (e.g., 0.9 for 90%).

        Returns:
            MDE for pairwise comparison at specified power.
        """
        z_alpha = float(stats.norm.ppf(1 - self.alpha / 2))
        z_power = float(stats.norm.ppf(power))
        return (z_alpha + z_power) * self.se_comparison

    def power_to_detect(self, effect_size: float) -> float:
        """Compute power to detect a specific effect size.

        Args:
            effect_size: Effect size to detect (e.g., 0.02 for 2%).

        Returns:
            Statistical power (probability of detecting the effect).
        """
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z = effect_size / self.se_comparison - z_alpha
        return float(stats.norm.cdf(z))

    def summary(self) -> str:
        """Human-readable summary of the evaluation plan."""
        return (
            f"Evaluation Plan\n"
            f"  Allocation: n={self.n_samples:,}, m={self.m_oracle:,} "
            f"({self.oracle_fraction:.1%} oracle)\n"
            f"  Cost: ${self.total_cost:,.0f}\n"
            f"  Single-policy SE: {self.se_level:.4f}\n"
            f"  MDE ({self.power:.0%} power): {self.mde:.1%}\n"
            f"  → Can detect {self.mde:.1%} difference between policies"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "n_samples": self.n_samples,
            "m_oracle": self.m_oracle,
            "oracle_fraction": self.oracle_fraction,
            "total_cost": self.total_cost,
            "mde": self.mde,
            "se_level": self.se_level,
            "se_comparison": self.se_comparison,
            "power": self.power,
            "alpha": self.alpha,
            "sigma2_eval": self.sigma2_eval,
            "sigma2_cal": self.sigma2_cal,
            "cost_model": {
                "surrogate_cost": self.cost_model.surrogate_cost,
                "oracle_cost": self.cost_model.oracle_cost,
            },
        }


# =============================================================================
# Main Public API
# =============================================================================


def fit_variance_model(
    fresh_draws_dict: Dict[str, "FreshDrawDataset"],
    n_grid: Optional[List[int]] = None,
    oracle_fraction_grid: Optional[List[float]] = None,
    n_replicates: int = 150,
    seed: int = 42,
    verbose: bool = True,
) -> FittedVarianceModel:
    """Fit Var(θ̂) = σ²_eval/n + σ²_cal/m from pilot data.

    This is the main entry point for budget planning. It measures variance
    empirically by computing Var(θ̂) across outer replicates at multiple
    (n, m) allocations, then fits the variance model.

    The key insight: We measure the actual variance of point estimates,
    NOT bootstrap SE². This gives R² > 0.9 when labeling is ignorable.

    Args:
        fresh_draws_dict: Pilot data (policy_name -> FreshDrawDataset).
            Use a single policy (typically "base") to ensure ignorable labeling.
        n_grid: Sample sizes to measure (default: auto from pilot size).
        oracle_fraction_grid: Oracle fractions to measure (default: [0.15, 0.25, 0.40]).
        n_replicates: Replicates per grid point for stable Var estimates.
        seed: Random seed for reproducibility.
        verbose: Print progress and diagnostics.

    Returns:
        FittedVarianceModel that can predict variance at any (n, m).

    Example:
        from cje.diagnostics import fit_variance_model, plan_evaluation

        # Fit variance model from pilot data
        model = fit_variance_model({"base": pilot_data}, verbose=True)
        print(f"R² = {model.r_squared:.3f}")  # Should be > 0.85

        # Use for planning
        plan = plan_evaluation(budget=5000, variance_model=model)
        print(plan.summary())

    Notes:
        - Use a single policy with random labeling for best fit (ignorable labeling)
        - The model extrapolates well to allocations larger than the pilot
        - Check model.r_squared - values > 0.85 indicate the 1/n + 1/m form is valid
    """
    # Get total available prompts
    all_prompt_ids: set[str] = set()
    for fd in fresh_draws_dict.values():
        all_prompt_ids.update(s.prompt_id for s in fd.samples)
    n_total = len(all_prompt_ids)

    if verbose:
        print(f"Fitting variance model from pilot (n={n_total} prompts)")

    # Check labeling ignorability
    ignorability = check_labeling_ignorability(fresh_draws_dict)
    if verbose:
        if ignorability.get("is_ignorable"):
            print(f"  ✓ Labeling is ignorable (KS p={ignorability['ks_pvalue']:.3f})")
        elif ignorability.get("is_ignorable") is False:
            print(
                f"  ⚠ Labeling may NOT be ignorable (KS p={ignorability['ks_pvalue']:.3f})"
            )
            print(f"    {ignorability['recommendation']}")

    # Auto-select grid if not provided
    if n_grid is None:
        # Use 3-4 points spanning 10%-60% of pilot size
        n_grid = [
            max(50, int(n_total * 0.15)),
            max(100, int(n_total * 0.30)),
            max(150, int(n_total * 0.45)),
            max(200, int(n_total * 0.60)),
        ]
        # Remove duplicates and filter to valid sizes
        n_grid = sorted(set(n for n in n_grid if n <= n_total * 0.8))

    if oracle_fraction_grid is None:
        oracle_fraction_grid = [0.15, 0.25, 0.40]

    if len(n_grid) < 2:
        raise ValueError(
            f"Need at least 2 valid n values, but only {len(n_grid)} fit in pilot "
            f"(n_total={n_total}). Collect more pilot data."
        )

    if verbose:
        print(f"  Grid: n={n_grid}, oracle_frac={oracle_fraction_grid}")
        print(
            f"  Measuring {len(n_grid) * len(oracle_fraction_grid)} allocations "
            f"({n_replicates} replicates each)..."
        )

    # Measure variance at each grid point using DIRECT method (Var of θ̂, not SE²)
    measurements: List[Tuple[int, int, float]] = []
    for i, n in enumerate(n_grid):
        for j, frac in enumerate(oracle_fraction_grid):
            m = max(int(n * frac), 1)
            if verbose:
                print(f"    n={n}, m={m} ({frac:.0%})...", end="", flush=True)

            result = measure_variance_direct(
                fresh_draws_dict=fresh_draws_dict,
                n_prompts=n,
                oracle_fraction=frac,
                n_replicates=n_replicates,
                seed=seed + i * 100 + j,
                verbose=False,
            )

            if not np.isnan(result["variance"]) and result["variance"] > 0:
                measurements.append((n, m, result["variance"]))
                if verbose:
                    print(f" SE={result['se']:.4f}")
            else:
                if verbose:
                    print(" FAILED")

    if len(measurements) < 3:
        raise ValueError(
            f"Only got {len(measurements)} valid measurements, need at least 3. "
            "Try increasing n_replicates or using larger oracle fractions."
        )

    # Fit the model
    model = _fit_variance_model_from_measurements(measurements)

    if verbose:
        print(f"\nFitted model (R²={model.r_squared:.3f}):")
        print(f"  σ²_eval = {model.sigma2_eval:.6f}")
        print(f"  σ²_cal  = {model.sigma2_cal:.6f}")

        if model.r_squared < 0.85:
            print("\n  ⚠ Low R² - check if labeling is truly ignorable")
            print("    (Use a single policy with random oracle sampling)")

    return model


def plan_evaluation(
    budget: float,
    variance_model: FittedVarianceModel,
    cost_model: Optional[CostModel] = None,
    m_min: int = 30,
    power: float = 0.8,
    alpha: float = 0.05,
) -> EvaluationPlan:
    """Plan optimal allocation for a given budget.

    Uses the Square Root Allocation Law to minimize variance, then computes
    the achievable MDE for pairwise policy comparisons.

    Args:
        budget: Total budget in cost units (e.g., dollars).
        variance_model: FittedVarianceModel from fit_variance_model().
        cost_model: Cost parameters. Default: CostModel(oracle_cost=16.0).
        m_min: Minimum oracle labels (default 30, needed for calibration).
        power: Statistical power for MDE (default 0.8 = 80%).
        alpha: Significance level (default 0.05 = 95% CI).

    Returns:
        EvaluationPlan with allocation and achievable MDE.

    Example:
        from cje.diagnostics import fit_variance_model, plan_evaluation

        model = fit_variance_model({"base": pilot_data})

        # "I have $5000, what can I detect?"
        plan = plan_evaluation(budget=5000, variance_model=model)
        print(plan.summary())
        # Evaluation Plan
        #   Allocation: n=4,200, m=80 (1.9% oracle)
        #   Cost: $5,000
        #   Single-policy SE: 0.0086
        #   MDE (80% power): 2.4%
        #   → Can detect 2.4% difference between policies

    Notes:
        - MDE uses conservative √2 factor assuming independent policy samples
        - Actual MDE may be better due to correlation (same calibrator, same prompts)
        - Use plan.power_to_detect(effect) to check power for specific effects
    """
    if cost_model is None:
        cost_model = CostModel()

    c_S = cost_model.surrogate_cost
    c_Y = cost_model.oracle_cost
    sigma2_eval = variance_model.sigma2_eval
    sigma2_cal = variance_model.sigma2_cal

    # Square Root Law (CJE paper Appendix F, Proposition 5)
    # n* = B·√(σ²_eval/c_S) / (√(c_S·σ²_eval) + √(c_Y·σ²_cal))
    # m* = B·√(σ²_cal/c_Y) / (√(c_S·σ²_eval) + √(c_Y·σ²_cal))
    denom = np.sqrt(c_S * sigma2_eval) + np.sqrt(c_Y * sigma2_cal)

    n_star_float = budget * np.sqrt(sigma2_eval / c_S) / denom
    m_star_float = budget * np.sqrt(sigma2_cal / c_Y) / denom

    # Enforce m ≤ n constraint
    if m_star_float > n_star_float:
        n_star_float = m_star_float = budget / (c_S + c_Y)

    # Round to integers
    n_star = int(max(n_star_float, 1))
    m_star = int(max(min(m_star_float, n_star), 1))

    # Enforce minimum oracle constraint
    if m_star < m_min:
        m_star = m_min
        remaining_budget = budget - c_Y * m_star
        if remaining_budget > 0:
            n_star = int(remaining_budget / c_S)
        else:
            n_star = m_star

    # Compute variance and SE at this allocation
    var_total = sigma2_eval / n_star + sigma2_cal / m_star
    se_level = float(np.sqrt(var_total))

    # MDE for pairwise comparison: MDE = (z_α/2 + z_β) × √2 × SE
    # Conservative √2 factor assumes independent samples
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    se_comparison = np.sqrt(2) * se_level
    mde = (z_alpha + z_power) * se_comparison

    total_cost = c_S * n_star + c_Y * m_star

    return EvaluationPlan(
        n_samples=n_star,
        m_oracle=m_star,
        total_cost=total_cost,
        mde=mde,
        se_level=se_level,
        power=power,
        alpha=alpha,
        sigma2_eval=sigma2_eval,
        sigma2_cal=sigma2_cal,
        cost_model=cost_model,
    )


def plan_for_mde(
    target_mde: float,
    variance_model: FittedVarianceModel,
    cost_model: Optional[CostModel] = None,
    m_min: int = 30,
    power: float = 0.8,
    alpha: float = 0.05,
) -> EvaluationPlan:
    """Find minimum budget to achieve target MDE.

    Solves the inverse problem: given a target MDE for pairwise comparisons,
    what budget is needed?

    Args:
        target_mde: Target minimum detectable effect (e.g., 0.01 for 1%).
        variance_model: FittedVarianceModel from fit_variance_model().
        cost_model: Cost parameters. Default: CostModel(oracle_cost=16.0).
        m_min: Minimum oracle labels (default 30, needed for calibration).
        power: Statistical power (default 0.8 = 80%).
        alpha: Significance level (default 0.05).

    Returns:
        EvaluationPlan with required budget and allocation.

    Example:
        from cje.diagnostics import fit_variance_model, plan_for_mde

        model = fit_variance_model({"base": pilot_data})

        # "I need to detect 1% differences"
        plan = plan_for_mde(target_mde=0.01, variance_model=model)
        print(f"Required budget: ${plan.total_cost:,.0f}")
        print(plan.summary())

    Notes:
        - Uses conservative √2 factor for pairwise comparisons
        - Returns optimal allocation at the minimum required budget
        - Enforces m_min constraint (may result in slightly lower MDE)
    """
    if cost_model is None:
        cost_model = CostModel()

    # Work backwards from MDE to required SE
    # MDE = (z_α/2 + z_β) × √2 × SE_level
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    critical_value = z_alpha + z_power

    # SE_comparison = MDE / critical_value
    # SE_level = SE_comparison / √2
    se_comparison_needed = target_mde / critical_value
    se_level_needed = se_comparison_needed / np.sqrt(2)
    var_needed = se_level_needed**2

    # Var = σ²_eval/n + σ²_cal/m
    # At optimal allocation from Square Root Law:
    # Var_min = (√(c_S·σ²_eval) + √(c_Y·σ²_cal))² / B
    # So B = (√(c_S·σ²_eval) + √(c_Y·σ²_cal))² / Var_needed
    c_S = cost_model.surrogate_cost
    c_Y = cost_model.oracle_cost
    sigma2_eval = variance_model.sigma2_eval
    sigma2_cal = variance_model.sigma2_cal

    numerator = (np.sqrt(c_S * sigma2_eval) + np.sqrt(c_Y * sigma2_cal)) ** 2
    budget_needed = numerator / var_needed

    # Now use plan_evaluation to get the actual allocation
    # This handles m_min constraints etc.
    plan = plan_evaluation(
        budget=budget_needed,
        variance_model=variance_model,
        cost_model=cost_model,
        m_min=m_min,
        power=power,
        alpha=alpha,
    )

    return plan


def _fit_variance_model_from_measurements(
    measurements: List[Tuple[int, int, float]],
) -> FittedVarianceModel:
    """Fit Var = sigma2_eval/n + sigma2_cal/m to empirical measurements.

    Uses non-negative least squares to ensure variance components are positive.

    Args:
        measurements: List of (n, m, measured_variance) tuples

    Returns:
        FittedVarianceModel with empirically-calibrated variance components

    Raises:
        ValueError: If fewer than 3 measurements provided

    Example:
        measurements = [
            (100, 25, 0.0055),
            (200, 50, 0.0028),
            (400, 100, 0.0014),
        ]
        model = fit_variance_model(measurements)
        print(f"Predicted SE at n=1000, m=100: {model.predict_se(1000, 100):.4f}")
    """
    from scipy.optimize import nnls

    if len(measurements) < 3:
        raise ValueError(
            f"Need at least 3 measurements to fit model, got {len(measurements)}"
        )

    # Filter out invalid measurements
    valid = [(n, m, var) for n, m, var in measurements if var > 0 and not np.isnan(var)]
    if len(valid) < 3:
        raise ValueError(f"Need at least 3 valid measurements, got {len(valid)}")

    # Build design matrix: [1/n, 1/m] for each measurement
    X = np.array([[1 / n, 1 / m] for n, m, _ in valid])
    y = np.array([var for _, _, var in valid])

    # Non-negative least squares (variances must be positive)
    coeffs, _ = nnls(X, y)
    sigma2_eval, sigma2_cal = coeffs

    # R² to validate functional form
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return FittedVarianceModel(
        sigma2_eval=float(sigma2_eval),
        sigma2_cal=float(sigma2_cal),
        r_squared=float(r_squared),
        n_measurements=len(valid),
    )


def fit_variance_model_from_pilot(
    fresh_draws_dict: Dict[str, FreshDrawDataset],
    n_grid: Optional[List[int]] = None,
    oracle_fraction_grid: Optional[List[float]] = None,
    n_replicates: int = 15,
    n_bootstrap: int = 150,
    seed: int = 42,
    verbose: bool = True,
) -> FittedVarianceModel:
    """Measure variance at grid of allocations and fit the variance model.

    This is the main entry point for budget planning. It:
    1. Measures variance empirically at multiple (n, m) points within pilot range
    2. Fits Var = sigma2_eval/n + sigma2_cal/m to those measurements
    3. Returns a model that can extrapolate to larger production allocations

    Args:
        fresh_draws_dict: Pilot data (policy_name -> FreshDrawDataset)
        n_grid: Sample sizes to measure (default: auto-selected from pilot size)
        oracle_fraction_grid: Oracle fractions to measure (default: [0.20, 0.35, 0.50])
        n_replicates: Replicates per grid point for stable estimates
        n_bootstrap: Bootstrap iterations per replicate
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        FittedVarianceModel that can predict variance at any (n, m)

    Example:
        from cje.data.fresh_draws import load_fresh_draws_auto, discover_policies_from_fresh_draws

        # Load pilot data
        policies = discover_policies_from_fresh_draws(pilot_dir)
        fresh_draws_dict = {p: load_fresh_draws_auto(pilot_dir, p) for p in policies}

        # Fit variance model
        model = fit_variance_model_from_pilot(fresh_draws_dict, verbose=True)
        print(model.summary())

        # Predict variance at production allocation
        print(f"SE at n=2000, m=200: {model.predict_se(2000, 200):.4f}")
    """
    # Get total available prompts
    all_prompt_ids_set: set[str] = set()
    for fd in fresh_draws_dict.values():
        all_prompt_ids_set.update(s.prompt_id for s in fd.samples)
    n_total = len(all_prompt_ids_set)

    if verbose:
        print(f"Fitting variance model from pilot (n={n_total} prompts)")

    # Auto-select grid if not provided
    if n_grid is None:
        # Use 3 points spanning 10%-50% of pilot size
        n_grid = [
            max(50, int(n_total * 0.15)),
            max(100, int(n_total * 0.30)),
            max(150, int(n_total * 0.50)),
        ]
        # Remove duplicates and sort
        n_grid = sorted(set(n_grid))

    if oracle_fraction_grid is None:
        oracle_fraction_grid = [0.20, 0.35, 0.50]

    # Filter grid to valid values
    n_grid = [n for n in n_grid if n <= n_total]
    if len(n_grid) < 2:
        raise ValueError(
            f"Need at least 2 valid n values, but only {len(n_grid)} fit in pilot "
            f"(n_total={n_total}). Collect more pilot data."
        )

    if verbose:
        print(f"  Grid: n={n_grid}, oracle_frac={oracle_fraction_grid}")
        print(f"  Measuring {len(n_grid) * len(oracle_fraction_grid)} allocations...")

    # Measure variance at each grid point
    measurements = []
    for i, n in enumerate(n_grid):
        for j, frac in enumerate(oracle_fraction_grid):
            m = max(int(n * frac), 1)
            if verbose:
                print(f"    n={n}, m={m} ({frac:.0%})...", end="", flush=True)

            result = measure_variance_at_allocation(
                fresh_draws_dict=fresh_draws_dict,
                n_prompts=n,
                oracle_fraction=frac,
                n_replicates=n_replicates,
                n_bootstrap=n_bootstrap,
                seed=seed + i * 100 + j,
                verbose=False,
            )

            if not np.isnan(result["variance"]) and result["variance"] > 0:
                measurements.append((n, m, result["variance"]))
                if verbose:
                    print(f" SE={result['mean_se']:.4f}")
            else:
                if verbose:
                    print(" FAILED")

    if len(measurements) < 3:
        raise ValueError(
            f"Only got {len(measurements)} valid measurements, need at least 3. "
            "Try increasing n_replicates or using larger oracle fractions."
        )

    # Fit the model
    model = _fit_variance_model_from_measurements(measurements)

    if verbose:
        print(f"\nFitted model (R²={model.r_squared:.3f}):")
        print(f"  σ²_eval = {model.sigma2_eval:.6f}")
        print(f"  σ²_cal  = {model.sigma2_cal:.6f}")

    return model


def compute_optimal_allocation(
    budget: float,
    cost_model: CostModel,
    variance_model: Optional[FittedVarianceModel] = None,
    sigma2_eval: Optional[float] = None,
    sigma2_cal: Optional[float] = None,
    m_min: int = 30,
    m_audit_per_slice: int = 0,
    n_slices: int = 1,
    warn_on_mismatch: bool = True,
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
        variance_model: FittedVarianceModel from fit_variance_model_from_pilot().
            Preferred method - uses empirically-calibrated variance components.
        sigma2_eval: Intrinsic evaluation variance (deprecated, use variance_model).
        sigma2_cal: Intrinsic calibration variance (deprecated, use variance_model).
        m_min: Minimum oracle labels required to fit calibrator (default 30).
        m_audit_per_slice: Additional oracle labels per audit slice.
        n_slices: Number of audit slices (e.g., policies to audit).
        warn_on_mismatch: Warn if computed oracle_fraction differs >50% from
            theoretical optimal (suggests unit mismatch or estimation error).

    Returns:
        BudgetAllocation with optimal n*, m* and diagnostic information.

    Example:
        # Recommended: Use empirically-fitted model
        model = fit_variance_model_from_pilot(fresh_draws_dict)
        allocation = compute_optimal_allocation(
            budget=5000.0,
            cost_model=CostModel(oracle_cost=16.0),
            variance_model=model,
        )
        print(allocation.summary())
    """
    # Extract sigma2 values from model or use provided values
    if variance_model is not None:
        sigma2_eval = variance_model.sigma2_eval
        sigma2_cal = variance_model.sigma2_cal
    elif sigma2_eval is None or sigma2_cal is None:
        raise ValueError(
            "Must provide either variance_model or both sigma2_eval and sigma2_cal"
        )
    c_S = cost_model.surrogate_cost
    c_Y = cost_model.oracle_cost

    # Reserve budget for audit slices
    audit_cost = m_audit_per_slice * n_slices * c_Y
    effective_budget = budget - audit_cost
    if effective_budget < 0:
        raise ValueError(
            f"Audit reserve ({audit_cost}) exceeds budget ({budget}). "
            "Reduce m_audit_per_slice or n_slices."
        )

    # Handle edge cases
    if sigma2_cal <= 0:
        # No calibration variance → spend all on evaluation
        n_star = int(effective_budget / c_S)
        m_star = max(m_min, 1)
    elif sigma2_eval <= 0:
        # No evaluation variance → spend all on calibration
        m_star = int(effective_budget / c_Y)
        n_star = m_star
    else:
        # Square Root Law (paper Proposition 5)
        denom = np.sqrt(c_S * sigma2_eval) + np.sqrt(c_Y * sigma2_cal)

        n_star_float = effective_budget * np.sqrt(sigma2_eval / c_S) / denom
        m_star_float = effective_budget * np.sqrt(sigma2_cal / c_Y) / denom

        # Theoretical optimal oracle fraction for sanity check
        theoretical_frac = np.sqrt(c_S / c_Y) * np.sqrt(sigma2_cal / sigma2_eval)

        # Enforce m ≤ n constraint
        if m_star_float > n_star_float:
            # Binding constraint: m = n, re-solve
            n_star_float = m_star_float = effective_budget / (c_S + c_Y)

        # Round down initially
        n_star = int(max(n_star_float, 1))
        m_star = int(max(min(m_star_float, n_star), 1))

        # Enforce minimum oracle constraint
        if m_star < m_min:
            m_star = m_min
            # Recompute n given m_min constraint
            remaining_budget = effective_budget - c_Y * m_star
            if remaining_budget > 0:
                n_star = int(remaining_budget / c_S)
            else:
                n_star = m_star

        # Greedy leftover-spending: use remaining budget optimally
        leftover = effective_budget - (c_S * n_star + c_Y * m_star)
        while leftover >= min(c_S, c_Y):
            # Marginal variance reduction per cost
            if n_star > 0 and m_star > 0:
                marginal_n = sigma2_eval / (n_star * (n_star + 1)) / c_S
                marginal_m = (
                    sigma2_cal / (m_star * (m_star + 1)) / c_Y if m_star < n_star else 0
                )
            else:
                break

            if marginal_n > marginal_m and leftover >= c_S:
                n_star += 1
                leftover -= c_S
            elif marginal_m > 0 and leftover >= c_Y:
                m_star += 1
                leftover -= c_Y
            else:
                break

        # Warn on oracle fraction mismatch
        actual_frac = m_star / n_star if n_star > 0 else 0.0
        if warn_on_mismatch and theoretical_frac > 0:
            ratio = actual_frac / theoretical_frac
            if ratio < 0.5 or ratio > 2.0:
                logger.warning(
                    f"Oracle fraction mismatch: computed {actual_frac:.1%}, "
                    f"theoretical {theoretical_frac:.1%}. "
                    "Check for unit mismatch in costs or variance estimates."
                )

    # Compute expected variance at optimal allocation
    var_eval = sigma2_eval / n_star if n_star > 0 else np.inf
    var_cal = sigma2_cal / m_star if m_star > 0 else np.inf
    var_total = var_eval + var_cal
    expected_se = float(np.sqrt(var_total))

    # Calibration uncertainty share
    omega = var_cal / var_total if var_total > 0 else 0.0

    # Add audit labels back to total m
    m_total = m_star + m_audit_per_slice * n_slices

    return BudgetAllocation(
        n_samples=n_star,
        m_oracle=m_total,
        oracle_fraction=m_total / n_star if n_star > 0 else 0.0,
        total_cost=c_S * n_star + c_Y * m_total,
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


# =============================================================================
# Validation Experiment: Direct Variance Measurement
# =============================================================================


@dataclass
class VarianceModelComparison:
    """Results of comparing variance model fits.

    Compares the simple model (a/n + b/m) vs the floor model (a/n + b/m + c).
    """

    # Simple model: Var = sigma2_eval/n + sigma2_cal/m
    simple_sigma2_eval: float
    simple_sigma2_cal: float
    simple_r_squared: float

    # Floor model: Var = sigma2_eval/n + sigma2_cal/m + floor
    floor_sigma2_eval: float
    floor_sigma2_cal: float
    floor_term: float
    floor_r_squared: float

    # Raw measurements
    measurements: List[Tuple[int, int, float]]  # (n, m, variance)
    n_measurements: int

    # Diagnostics
    measurement_method: str  # "point_estimate_variance" or "bootstrap_se_squared"

    def summary(self) -> str:
        """Human-readable comparison summary."""
        return (
            f"Variance Model Comparison ({self.measurement_method})\n"
            f"  n_measurements = {self.n_measurements}\n"
            f"\n"
            f"  Simple model (a/n + b/m):\n"
            f"    σ²_eval = {self.simple_sigma2_eval:.6f}\n"
            f"    σ²_cal  = {self.simple_sigma2_cal:.6f}\n"
            f"    R²      = {self.simple_r_squared:.4f}\n"
            f"\n"
            f"  Floor model (a/n + b/m + c):\n"
            f"    σ²_eval = {self.floor_sigma2_eval:.6f}\n"
            f"    σ²_cal  = {self.floor_sigma2_cal:.6f}\n"
            f"    floor   = {self.floor_term:.6f}\n"
            f"    R²      = {self.floor_r_squared:.4f}\n"
            f"\n"
            f"  R² improvement from floor: {self.floor_r_squared - self.simple_r_squared:+.4f}"
        )


def fit_variance_model_with_floor(
    measurements: List[Tuple[int, int, float]],
) -> Tuple[float, float, float, float]:
    """Fit Var = a/n + b/m + c with non-negative constraints.

    Args:
        measurements: List of (n, m, measured_variance) tuples

    Returns:
        Tuple of (sigma2_eval, sigma2_cal, floor, r_squared)
    """
    from scipy.optimize import nnls

    valid = [(n, m, var) for n, m, var in measurements if var > 0 and not np.isnan(var)]
    if len(valid) < 4:
        raise ValueError(
            f"Need at least 4 measurements for floor model, got {len(valid)}"
        )

    # Build design matrix: [1/n, 1/m, 1] for each measurement
    X = np.array([[1 / n, 1 / m, 1.0] for n, m, _ in valid])
    y = np.array([var for _, _, var in valid])

    # Non-negative least squares
    coeffs, _ = nnls(X, y)
    sigma2_eval, sigma2_cal, floor = coeffs

    # R²
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(sigma2_eval), float(sigma2_cal), float(floor), float(r_squared)


def _compute_point_estimate_single_replicate(
    subsampled_dict: Dict[str, "FreshDrawDataset"],
    seed: int,
) -> Optional[np.ndarray]:
    """Compute point estimate θ̂ for a single subsample (NO bootstrap).

    This is the key difference from the old approach: we compute the actual
    point estimate, not a bootstrap SE estimate.

    Returns:
        Array of point estimates per policy, or None if calibration failed.
    """
    from .robust_inference import build_direct_eval_table
    from ..calibration.judge import JudgeCalibrator

    # Build eval table
    eval_table = build_direct_eval_table(subsampled_dict)

    # Check we have enough oracle labels
    n_oracle = int(np.sum(eval_table.oracle_mask))
    if n_oracle < 10:
        return None

    # Fit calibrator using fit_transform
    try:
        calibrator = JudgeCalibrator(
            random_seed=seed,
            calibration_mode="monotone",
        )

        # Use fit_transform which handles oracle_mask properly
        result = calibrator.fit_transform(
            judge_scores=eval_table.judge_scores,
            oracle_labels=eval_table.oracle_labels[eval_table.oracle_mask],
            oracle_mask=eval_table.oracle_mask,
        )

        # Get calibrated values from result
        calibrated = result.calibrated_scores

        # Compute θ̂_aug = mean(f̂(S)) + mean(Y - f̂(S)) for oracle samples
        # This is the augmented estimator without bootstrap
        n_policies = eval_table.n_policies
        estimates = np.zeros(n_policies)

        for p in range(n_policies):
            p_mask = eval_table.policy_indices == p
            n_p = np.sum(p_mask)

            if n_p == 0:
                estimates[p] = np.nan
                continue

            # Plug-in: mean of calibrated predictions
            plug_in = np.mean(calibrated[p_mask])

            # Residual correction using oracle samples for this policy
            p_oracle_mask = p_mask & eval_table.oracle_mask
            n_oracle_p = np.sum(p_oracle_mask)

            if n_oracle_p > 0:
                residuals = (
                    eval_table.oracle_labels[p_oracle_mask] - calibrated[p_oracle_mask]
                )
                residual_correction = np.mean(residuals)
            else:
                residual_correction = 0.0

            estimates[p] = plug_in + residual_correction

        return estimates

    except Exception as e:
        logger.debug(f"Point estimate failed: {e}")
        return None


def measure_variance_direct(
    fresh_draws_dict: Dict[str, "FreshDrawDataset"],
    n_prompts: int,
    oracle_fraction: float,
    n_replicates: int = 200,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Measure variance by computing Var(θ̂) across outer replicates (NO bootstrap).

    This is the analyst-recommended approach: instead of running bootstrap and
    averaging SE estimates, we:
    1. Draw many outer subsamples at each (n, m)
    2. Compute the point estimate θ̂_r for each replicate
    3. Estimate Var(n, m) = Var_r(θ̂_r) directly

    Args:
        fresh_draws_dict: Full dataset (policy_name -> FreshDrawDataset)
        n_prompts: Number of unique prompts to sample
        oracle_fraction: Fraction of sampled prompts with oracle labels
        n_replicates: Number of outer replicates (default 200)
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        Dictionary with:
        - variance: Var_r(θ̂_r) - the true variance of point estimates
        - se: sqrt(variance)
        - n_actual: Number of prompts
        - m_actual: Number of oracle labels
        - n_valid_replicates: How many replicates succeeded
        - estimates: Array of all point estimates (for diagnostics)
    """
    rng = np.random.default_rng(seed)
    all_estimates: List[np.ndarray] = []

    for rep in range(n_replicates):
        # Subsample data
        try:
            subsampled = _subsample_fresh_draws(
                fresh_draws_dict,
                n_prompts=n_prompts,
                oracle_fraction=oracle_fraction,
                rng=rng,
            )
        except ValueError:
            continue

        if not subsampled:
            continue

        # Compute point estimate (no bootstrap!)
        estimates = _compute_point_estimate_single_replicate(
            subsampled, seed=seed + rep
        )

        if estimates is not None and not np.all(np.isnan(estimates)):
            all_estimates.append(estimates)

    if len(all_estimates) < 10:
        return {
            "variance": np.nan,
            "se": np.nan,
            "n_actual": n_prompts,
            "m_actual": int(n_prompts * oracle_fraction),
            "n_valid_replicates": len(all_estimates),
            "estimates": np.array([]),
        }

    # Stack estimates: (n_replicates, n_policies)
    estimates_matrix = np.array(all_estimates)

    # Compute variance across replicates (average across policies)
    variance_per_policy = np.nanvar(estimates_matrix, axis=0, ddof=1)
    mean_variance = float(np.nanmean(variance_per_policy))

    if verbose:
        print(
            f"    n={n_prompts}, m={int(n_prompts * oracle_fraction)}: "
            f"SE={np.sqrt(mean_variance):.4f} (from {len(all_estimates)} replicates)"
        )

    return {
        "variance": mean_variance,
        "se": float(np.sqrt(mean_variance)),
        "n_actual": n_prompts,
        "m_actual": int(n_prompts * oracle_fraction),
        "n_valid_replicates": len(all_estimates),
        "estimates": estimates_matrix,
    }


def check_labeling_ignorability(
    fresh_draws_dict: Dict[str, "FreshDrawDataset"],
) -> Dict[str, Any]:
    """Check if oracle labeling is ignorable (uniform/random).

    Compares the distribution of judge scores between labeled and unlabeled
    samples. If labeling is ignorable (MAR), these should be similar.

    Returns:
        Dictionary with:
        - ks_statistic: KS test statistic (0 = identical distributions)
        - ks_pvalue: p-value (low = labeling is NOT ignorable)
        - labeled_mean: Mean score of labeled samples
        - unlabeled_mean: Mean score of unlabeled samples
        - labeled_std: Std of labeled samples
        - unlabeled_std: Std of unlabeled samples
        - is_ignorable: Boolean (True if p > 0.05)
        - recommendation: Human-readable guidance
    """
    # Collect all samples
    labeled_scores = []
    unlabeled_scores = []

    for fd in fresh_draws_dict.values():
        for sample in fd.samples:
            if sample.oracle_label is not None:
                labeled_scores.append(sample.judge_score)
            else:
                unlabeled_scores.append(sample.judge_score)

    labeled_scores_arr = np.array(labeled_scores)
    unlabeled_scores_arr = np.array(unlabeled_scores)

    if len(labeled_scores_arr) < 10 or len(unlabeled_scores_arr) < 10:
        return {
            "ks_statistic": np.nan,
            "ks_pvalue": np.nan,
            "labeled_mean": np.nan,
            "unlabeled_mean": np.nan,
            "is_ignorable": None,
            "recommendation": "Insufficient data to test ignorability",
        }

    # KS test for distribution equality
    ks_stat, ks_pval = stats.ks_2samp(labeled_scores_arr, unlabeled_scores_arr)

    # Summary statistics
    labeled_mean = float(np.mean(labeled_scores_arr))
    unlabeled_mean = float(np.mean(unlabeled_scores_arr))
    labeled_std = float(np.std(labeled_scores_arr))
    unlabeled_std = float(np.std(unlabeled_scores_arr))
    mean_diff = labeled_mean - unlabeled_mean

    # Decision
    is_ignorable = ks_pval > 0.05

    if is_ignorable:
        recommendation = (
            "Labeling appears ignorable (p={:.3f}). "
            "The simple e(Z)=m/n assumption is likely valid."
        ).format(ks_pval)
    else:
        recommendation = (
            "Labeling is NOT ignorable (p={:.3f}, mean diff={:.3f}). "
            "Consider modeling propensity e(Z) = P(L=1|Z) explicitly. "
            "Labeled samples have {} judge scores than unlabeled."
        ).format(ks_pval, mean_diff, "higher" if mean_diff > 0 else "lower")

    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
        "labeled_mean": labeled_mean,
        "unlabeled_mean": unlabeled_mean,
        "labeled_std": labeled_std,
        "unlabeled_std": unlabeled_std,
        "mean_difference": mean_diff,
        "n_labeled": len(labeled_scores),
        "n_unlabeled": len(unlabeled_scores),
        "is_ignorable": is_ignorable,
        "recommendation": recommendation,
    }


def run_variance_validation_experiment(
    fresh_draws_dict: Dict[str, "FreshDrawDataset"],
    n_grid: Optional[List[int]] = None,
    oracle_fraction_grid: Optional[List[float]] = None,
    n_replicates: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> VarianceModelComparison:
    """Run the analyst-recommended validation experiment.

    This experiment tests whether measuring Var(θ̂) directly (instead of
    bootstrap SE²) fixes the poor model fit.

    Steps:
    1. For each (n, oracle_frac) in grid:
       - Draw R outer replicates (clustered by prompt)
       - Compute θ̂_r for each replicate (NO bootstrap)
       - Compute Var(n, m) = Var_r(θ̂_r)
    2. Fit simple model: Var = a/n + b/m
    3. Fit floor model: Var = a/n + b/m + c
    4. Compare R² values
    5. Check if labeling is ignorable

    Args:
        fresh_draws_dict: Pilot data
        n_grid: Sample sizes to test (default: auto from pilot size)
        oracle_fraction_grid: Oracle fractions (default: [0.1, 0.2, 0.3])
        n_replicates: Outer replicates per grid point (default: 200)
        seed: Random seed
        verbose: Print progress

    Returns:
        VarianceModelComparison with fit results
    """
    # Get total available prompts
    all_prompt_ids_set: set[str] = set()
    for fd in fresh_draws_dict.values():
        all_prompt_ids_set.update(s.prompt_id for s in fd.samples)
    n_total = len(all_prompt_ids_set)

    if verbose:
        print(f"Running variance validation experiment (n_total={n_total})")
        print("=" * 60)

    # Check labeling ignorability first
    if verbose:
        print("\n1. Checking labeling ignorability...")

    ignorability = check_labeling_ignorability(fresh_draws_dict)
    if verbose:
        print(
            f"   KS test: statistic={ignorability['ks_statistic']:.3f}, "
            f"p={ignorability['ks_pvalue']:.3f}"
        )
        print(
            f"   Labeled mean={ignorability['labeled_mean']:.3f}, "
            f"unlabeled mean={ignorability['unlabeled_mean']:.3f}"
        )
        print(f"   {ignorability['recommendation']}")

    # Auto-select grid if not provided
    if n_grid is None:
        # Use 4 points: 100, 200, 400, 800 (or as far as pilot allows)
        n_grid = [n for n in [100, 200, 400, 800] if n <= n_total * 0.8]
        if len(n_grid) < 2:
            n_grid = [max(50, int(n_total * 0.3)), max(100, int(n_total * 0.6))]

    if oracle_fraction_grid is None:
        oracle_fraction_grid = [0.1, 0.2, 0.3]

    if verbose:
        print("\n2. Measuring variance at grid points...")
        print(f"   n_grid = {n_grid}")
        print(f"   oracle_fraction_grid = {oracle_fraction_grid}")
        print(f"   n_replicates = {n_replicates}")

    # Measure variance at each grid point using DIRECT method (no bootstrap)
    measurements: List[Tuple[int, int, float]] = []

    for i, n in enumerate(n_grid):
        for j, frac in enumerate(oracle_fraction_grid):
            m = max(int(n * frac), 1)

            if verbose:
                print(
                    f"   [{i * len(oracle_fraction_grid) + j + 1}/"
                    f"{len(n_grid) * len(oracle_fraction_grid)}] "
                    f"n={n}, m={m} ({frac:.0%})...",
                    end="",
                    flush=True,
                )

            result = measure_variance_direct(
                fresh_draws_dict=fresh_draws_dict,
                n_prompts=n,
                oracle_fraction=frac,
                n_replicates=n_replicates,
                seed=seed + i * 100 + j,
                verbose=False,
            )

            if not np.isnan(result["variance"]) and result["variance"] > 0:
                measurements.append((n, m, result["variance"]))
                if verbose:
                    print(
                        f" SE={result['se']:.4f} "
                        f"({result['n_valid_replicates']} valid replicates)"
                    )
            else:
                if verbose:
                    print(" FAILED")

    if len(measurements) < 4:
        raise ValueError(
            f"Only got {len(measurements)} valid measurements, need at least 4. "
            "Try increasing n_replicates or using larger oracle fractions."
        )

    # Fit both models
    if verbose:
        print(f"\n3. Fitting variance models to {len(measurements)} measurements...")

    # Simple model: a/n + b/m
    simple_model = _fit_variance_model_from_measurements(measurements)

    # Floor model: a/n + b/m + c
    floor_sigma2_eval, floor_sigma2_cal, floor_term, floor_r2 = (
        fit_variance_model_with_floor(measurements)
    )

    comparison = VarianceModelComparison(
        simple_sigma2_eval=simple_model.sigma2_eval,
        simple_sigma2_cal=simple_model.sigma2_cal,
        simple_r_squared=simple_model.r_squared,
        floor_sigma2_eval=floor_sigma2_eval,
        floor_sigma2_cal=floor_sigma2_cal,
        floor_term=floor_term,
        floor_r_squared=floor_r2,
        measurements=measurements,
        n_measurements=len(measurements),
        measurement_method="point_estimate_variance",
    )

    if verbose:
        print("\n" + comparison.summary())
        print("\n4. Interpretation:")

        r2_improvement = floor_r2 - simple_model.r_squared
        if simple_model.r_squared > 0.85:
            print("   ✓ Simple model fits well (R² > 0.85)!")
            print("   → The 1/n + 1/m form is valid for your data.")
            print(
                "   → Previous poor fit was likely due to measuring bootstrap SE² "
                "instead of Var(θ̂)."
            )
        elif r2_improvement > 0.3:
            print("   ⚠ Simple model fits poorly, but floor model fits well.")
            print(
                f"   → Floor term c = {floor_term:.6f} suggests bias/misspecification."
            )
            print(
                "   → For planning, use floor model but treat floor as MSE, not variance."
            )
        else:
            print("   ✗ Neither model fits well.")
            print("   → Check if labeling is ignorable (model propensity e(Z)).")
            print("   → Consider larger pilot or different calibration approach.")

    return comparison


# =============================================================================
# Empirical Ablation-Based Variance Measurement (Original - kept for comparison)
# =============================================================================


def _subsample_fresh_draws(
    fresh_draws_dict: Dict[str, FreshDrawDataset],
    n_prompts: int,
    oracle_fraction: float,
    rng: np.random.Generator,
) -> Dict[str, FreshDrawDataset]:
    """Subsample fresh draws to (n_prompts, oracle_fraction) allocation.

    Args:
        fresh_draws_dict: Full dataset (policy_name -> FreshDrawDataset)
        n_prompts: Number of unique prompts to sample
        oracle_fraction: Fraction of sampled prompts to have oracle labels
        rng: Random number generator

    Returns:
        Subsampled fresh_draws_dict with masked oracle labels
    """
    # Collect all unique prompt IDs across all policies
    all_prompt_ids_set: set[str] = set()
    for fd in fresh_draws_dict.values():
        all_prompt_ids_set.update(s.prompt_id for s in fd.samples)
    all_prompt_ids_list = sorted(all_prompt_ids_set)

    if n_prompts > len(all_prompt_ids_list):
        raise ValueError(
            f"Requested {n_prompts} prompts but only {len(all_prompt_ids_list)} available"
        )

    # Sample prompts
    sampled_prompt_ids = set(
        rng.choice(all_prompt_ids_list, size=n_prompts, replace=False)
    )

    # Determine which prompts keep oracle labels
    n_oracle = max(int(n_prompts * oracle_fraction), 1)
    oracle_prompt_ids = set(
        rng.choice(list(sampled_prompt_ids), size=n_oracle, replace=False)
    )

    # Build subsampled datasets
    subsampled_dict = {}
    for policy_name, fd in fresh_draws_dict.items():
        subsampled_samples = []
        for sample in fd.samples:
            if sample.prompt_id not in sampled_prompt_ids:
                continue

            # Deep copy to avoid modifying original
            new_sample = FreshDrawSample(
                prompt_id=sample.prompt_id,
                target_policy=sample.target_policy,
                judge_score=sample.judge_score,
                oracle_label=(
                    sample.oracle_label
                    if sample.prompt_id in oracle_prompt_ids
                    else None
                ),
                response=sample.response,
                draw_idx=sample.draw_idx,
                fold_id=sample.fold_id,
                metadata=sample.metadata.copy(),
            )
            subsampled_samples.append(new_sample)

        if subsampled_samples:
            subsampled_dict[policy_name] = FreshDrawDataset(
                target_policy=policy_name,
                draws_per_prompt=fd.draws_per_prompt,
                samples=subsampled_samples,
            )

    return subsampled_dict


def measure_variance_at_allocation(
    fresh_draws_dict: Dict[str, FreshDrawDataset],
    n_prompts: int,
    oracle_fraction: float,
    n_replicates: int = 30,
    n_bootstrap: int = 200,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, float]:
    """Subsample data to (n_prompts, oracle_fraction) and measure SE directly.

    This is the empirical approach: actually run estimation at a given allocation
    and measure the standard error, rather than inferring it from variance
    decomposition formulas.

    Args:
        fresh_draws_dict: Full dataset (policy_name -> FreshDrawDataset)
        n_prompts: Number of unique prompts to sample
        oracle_fraction: Fraction of sampled prompts with oracle labels
        n_replicates: Number of subsampling replicates for stable SE estimate
        n_bootstrap: Bootstrap iterations per replicate (fewer for speed)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with:
            - mean_se: Average SE across replicates
            - std_se: Standard deviation of SE across replicates
            - n_actual: Actual number of prompts sampled
            - m_actual: Actual number of oracle labels
            - variance: mean_se^2
    """
    from .robust_inference import (
        build_direct_eval_table,
        cluster_bootstrap_direct_with_refit,
        make_calibrator_factory,
    )

    rng = np.random.default_rng(seed)
    ses = []

    for rep in range(n_replicates):
        # Subsample data
        try:
            subsampled = _subsample_fresh_draws(
                fresh_draws_dict,
                n_prompts=n_prompts,
                oracle_fraction=oracle_fraction,
                rng=rng,
            )
        except ValueError as e:
            if verbose:
                logger.warning(f"Replicate {rep}: {e}")
            continue

        if not subsampled:
            continue

        # Build eval table
        eval_table = build_direct_eval_table(subsampled)

        # Check we have enough oracle labels
        n_oracle = int(np.sum(eval_table.oracle_mask))
        if n_oracle < 5:
            if verbose:
                logger.warning(
                    f"Replicate {rep}: Only {n_oracle} oracle labels, skipping"
                )
            continue

        # Create calibrator factory (use monotone mode for simplicity)
        calibrator_factory = make_calibrator_factory(
            mode="monotone",
            seed=seed + rep,
        )

        # Run bootstrap
        try:
            result = cluster_bootstrap_direct_with_refit(
                eval_table=eval_table,
                calibrator_factory=calibrator_factory,
                n_bootstrap=n_bootstrap,
                min_oracle_per_replicate=max(3, int(n_oracle * 0.1)),
                seed=seed + rep,
                use_augmented_estimator=True,
            )
            # Average SE across policies
            se = float(np.nanmean(result["standard_errors"]))
            if not np.isnan(se) and se > 0:
                ses.append(se)
        except Exception as e:
            if verbose:
                logger.warning(f"Replicate {rep}: Bootstrap failed: {e}")
            continue

    if not ses:
        return {
            "mean_se": np.nan,
            "std_se": np.nan,
            "n_actual": n_prompts,
            "m_actual": int(n_prompts * oracle_fraction),
            "variance": np.nan,
        }

    mean_se = float(np.mean(ses))
    std_se = float(np.std(ses, ddof=1)) if len(ses) > 1 else 0.0

    return {
        "mean_se": mean_se,
        "std_se": std_se,
        "n_actual": n_prompts,
        "m_actual": int(n_prompts * oracle_fraction),
        "variance": mean_se**2,
    }


def measure_variance_grid(
    fresh_draws_dict: Dict[str, FreshDrawDataset],
    n_grid: List[int],
    oracle_fraction_grid: List[float],
    n_replicates: int = 20,
    n_bootstrap: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> np.ndarray:
    """Measure SE at a grid of (n, oracle_fraction) allocations.

    Args:
        fresh_draws_dict: Full dataset
        n_grid: List of prompt counts to test
        oracle_fraction_grid: List of oracle fractions to test
        n_replicates: Replicates per allocation for stable SE
        n_bootstrap: Bootstrap iterations per replicate
        seed: Random seed
        verbose: Print progress

    Returns:
        variance_grid: Shape (len(n_grid), len(oracle_fraction_grid))
            where variance_grid[i, j] = measured Var(estimate) at allocation (i, j)
    """
    variance_grid = np.zeros((len(n_grid), len(oracle_fraction_grid)))

    total_cells = len(n_grid) * len(oracle_fraction_grid)
    cell_idx = 0

    for i, n in enumerate(n_grid):
        for j, frac in enumerate(oracle_fraction_grid):
            cell_idx += 1
            if verbose:
                print(
                    f"  [{cell_idx}/{total_cells}] Measuring n={n}, oracle_frac={frac:.0%}...",
                    end="",
                    flush=True,
                )

            result = measure_variance_at_allocation(
                fresh_draws_dict=fresh_draws_dict,
                n_prompts=n,
                oracle_fraction=frac,
                n_replicates=n_replicates,
                n_bootstrap=n_bootstrap,
                seed=seed + cell_idx * 1000,
                verbose=False,
            )
            variance_grid[i, j] = result["variance"]

            if verbose:
                if np.isnan(result["variance"]):
                    print(" FAILED")
                else:
                    print(f" SE={result['mean_se']:.4f}")

    return variance_grid
