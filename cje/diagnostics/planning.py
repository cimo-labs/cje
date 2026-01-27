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
    from cje.data.fresh_draws import load_fresh_draws_auto
    from cje.diagnostics import fit_variance_model, plan_evaluation, CostModel

    # Load pilot data
    pilot_data = load_fresh_draws_auto(pilot_dir, "base")

    # Fit variance model from empirical measurements
    variance_model = fit_variance_model(pilot_data, verbose=True)
    print(f"R² = {variance_model.r_squared:.3f}")

    # Plan optimal allocation for production
    cost_model = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
    plan = plan_evaluation(budget=5000, variance_model=variance_model, cost_model=cost_model)
    print(plan.summary())
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional, TYPE_CHECKING
import logging
import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from ..data.fresh_draws import FreshDrawDataset, FreshDrawSample

logger = logging.getLogger(__name__)


@dataclass
class CostModel:
    """Cost parameters for budget optimization.

    The optimal oracle/surrogate allocation depends critically on relative costs.
    You must specify costs that reflect your actual setup.

    **Recommendation: Use actual dollar costs** so that budget values clearly represent
    real dollars. For example, with surrogate_cost=0.01 and oracle_cost=0.16, a budget
    of 5000 clearly means $5,000.

    Attributes:
        surrogate_cost: Cost per surrogate (judge) score in dollars (c_S).
        oracle_cost: Cost per oracle label in dollars (c_Y).

    Example:
        # GPT-4o-mini surrogate ($0.01) vs GPT-4o oracle ($0.16)
        cost_model = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
    """

    surrogate_cost: float = 1.0
    oracle_cost: float = 16.0

    def __post_init__(self) -> None:
        if self.surrogate_cost <= 0:
            raise ValueError(
                f"surrogate_cost must be positive, got {self.surrogate_cost}"
            )
        if self.oracle_cost <= 0:
            raise ValueError(f"oracle_cost must be positive, got {self.oracle_cost}")

    @property
    def cost_ratio(self) -> float:
        """c_S / c_Y ratio."""
        return self.surrogate_cost / self.oracle_cost


@dataclass
class FittedVarianceModel:
    """Result of fitting Var = sigma2_eval/n + sigma2_cal/m to empirical data.

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

    Attributes:
        n_samples: Number of evaluation samples (prompts) to collect.
        m_oracle: Number of oracle labels to collect.
        total_cost: Total cost of this allocation.
        mde: Minimum detectable effect at specified power (for pairwise A vs B).
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
        """Compute MDE at a different power level."""
        z_alpha = float(stats.norm.ppf(1 - self.alpha / 2))
        z_power = float(stats.norm.ppf(power))
        return (z_alpha + z_power) * self.se_comparison

    def power_to_detect(self, effect_size: float) -> float:
        """Compute power to detect a specific effect size."""
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
    fresh_draws: "FreshDrawDataset",
    n_grid: Optional[List[int]] = None,
    oracle_fraction_grid: Optional[List[float]] = None,
    n_replicates: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> FittedVarianceModel:
    """Fit Var(θ̂) = σ²_eval/n + σ²_cal/m from pilot data.

    Uses the actual CJE pipeline (analyze_dataset) to measure variance at
    different (n, m) allocations, then fits NNLS to decompose into σ²_eval
    and σ²_cal components.

    Args:
        fresh_draws: Pilot data from the base policy where calibration will be
            learned (FreshDrawDataset with oracle labels).
        n_grid: Sample sizes to measure (default: auto from pilot size).
        oracle_fraction_grid: Oracle fractions to measure (default: [0.15, 0.25, 0.40]).
        n_replicates: Replicates per grid point for stable SE estimates (default 5).
        seed: Random seed for reproducibility.
        verbose: Print progress and diagnostics.

    Returns:
        FittedVarianceModel that can predict variance at any (n, m).

    Example:
        model = fit_variance_model(pilot_data, verbose=True)
        print(f"R² = {model.r_squared:.3f}")
    """
    # Get total available prompts
    all_prompt_ids: set[str] = set()
    all_prompt_ids.update(s.prompt_id for s in fresh_draws.samples)
    n_total = len(all_prompt_ids)

    if verbose:
        print(f"Fitting variance model from pilot (n={n_total} prompts)")

    # Check labeling ignorability
    ignorability = _check_labeling_ignorability(fresh_draws)
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
        n_grid = [
            max(50, int(n_total * 0.15)),
            max(100, int(n_total * 0.30)),
            max(150, int(n_total * 0.45)),
            max(200, int(n_total * 0.60)),
        ]
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

    # Build measurement grid with INDEPENDENT variation in n and m
    # Key insight: to separate σ²_eval/n from σ²_cal/m, we need measurements where
    # n varies while m is held constant (and vice versa).
    #
    # Design principle: Create a 2D grid with explicit variation in both dimensions
    # rather than using oracle fractions (which create collinearity).
    n_oracle_available = sum(
        1 for s in fresh_draws.samples if s.oracle_label is not None
    )

    # Define m values: small, medium, large (relative to available oracle)
    m_values = sorted(
        set(
            [
                max(15, int(n_oracle_available * 0.15)),
                max(25, int(n_oracle_available * 0.30)),
                max(40, int(n_oracle_available * 0.50)),
            ]
        )
    )

    # Build orthogonal grid: vary n at each fixed m, and vary m at each fixed n
    measurement_points: List[Tuple[int, int]] = []

    for n in n_grid:
        for m in m_values:
            # Constraint: m < n (can't have more oracle than samples)
            # Also need m < n_oracle_available (can't use more oracle than exists)
            if m < n and m <= n_oracle_available:
                measurement_points.append((n, m))

    # Sort and dedupe
    measurement_points = sorted(set(measurement_points))

    # Verify we have enough variation - need at least 2 different n and 2 different m
    unique_n = set(n for n, m in measurement_points)
    unique_m = set(m for n, m in measurement_points)
    if len(unique_n) < 2 or len(unique_m) < 2:
        raise ValueError(
            f"Grid has insufficient variation (n={unique_n}, m={unique_m}). "
            f"Need at least 2 unique values in each dimension. "
            f"Collect more pilot data (recommend 200+ prompts with 100+ oracle labels)."
        )

    if verbose:
        print(
            f"  Measuring {len(measurement_points)} allocations "
            f"({n_replicates} replicates each)..."
        )

    # Measure variance at each grid point
    measurements: List[Tuple[int, int, float]] = []
    for i, (n, m) in enumerate(measurement_points):
        if verbose:
            print(f"    n={n}, m={m} ({m/n:.0%})...", end="", flush=True)

        result = _measure_variance_direct(
            fresh_draws=fresh_draws,
            n_prompts=n,
            m_oracle=m,
            n_replicates=n_replicates,
            seed=seed + i * 100,
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

        # Warn if one component dominates completely (potential identification issue)
        total_sigma2 = model.sigma2_eval + model.sigma2_cal
        if total_sigma2 > 0:
            eval_share = model.sigma2_eval / total_sigma2
            if eval_share < 0.01:
                print("\n  ⚠ σ²_eval ≈ 0: All variance attributed to calibration.")
                print("    This may indicate:")
                print("    - Judge scores are highly predictive of oracle (good!)")
                print(
                    "    - OR grid design couldn't separate components (collect more pilot data)"
                )
            elif eval_share > 0.99:
                print("\n  ⚠ σ²_cal ≈ 0: All variance attributed to evaluation.")
                print("    This may indicate:")
                print("    - Calibration is very stable (good!)")
                print("    - OR need more oracle label variation in grid")

    return model


def plan_evaluation(
    budget: float,
    variance_model: FittedVarianceModel,
    cost_model: CostModel,
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
        cost_model: Cost parameters (required).
        m_min: Minimum oracle labels (default 30).
        power: Statistical power for MDE (default 0.8).
        alpha: Significance level (default 0.05).

    Returns:
        EvaluationPlan with allocation and achievable MDE.
    """
    c_S = cost_model.surrogate_cost
    c_Y = cost_model.oracle_cost
    sigma2_eval = variance_model.sigma2_eval
    sigma2_cal = variance_model.sigma2_cal

    # Square Root Law (CJE paper Appendix F, Proposition 7)
    denom = np.sqrt(c_S * sigma2_eval) + np.sqrt(c_Y * sigma2_cal)

    if denom < 1e-10:
        # Edge case: allocate based on which component has variance
        if sigma2_cal > 0:
            n_star = m_min
            m_star = max(int(budget / c_Y), m_min)
        else:
            n_star = max(int(budget / c_S), 1)
            m_star = m_min
    else:
        n_star_float = (
            budget * np.sqrt(sigma2_eval / c_S) / denom if sigma2_eval > 0 else m_min
        )
        m_star_float = (
            budget * np.sqrt(sigma2_cal / c_Y) / denom if sigma2_cal > 0 else m_min
        )

        # Enforce m ≤ n constraint
        if m_star_float > n_star_float:
            n_star_float = m_star_float = budget / (c_S + c_Y)

        n_star = int(max(n_star_float, 1))
        m_star = int(max(min(m_star_float, n_star), 1))

    # Enforce minimum oracle constraint
    if m_star < m_min:
        m_star = m_min
        remaining = budget - c_Y * m_star
        n_star = max(int(remaining / c_S), m_star) if remaining > 0 else m_star

    if n_star < m_star:
        n_star = m_star

    # Compute variance and SE
    var_total = sigma2_eval / n_star + sigma2_cal / m_star
    se_level = float(np.sqrt(var_total))

    # MDE for pairwise comparison
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
    cost_model: CostModel,
    m_min: int = 30,
    power: float = 0.8,
    alpha: float = 0.05,
) -> EvaluationPlan:
    """Find minimum budget to achieve target MDE."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    critical_value = z_alpha + z_power

    se_comparison_needed = target_mde / critical_value
    se_level_needed = se_comparison_needed / np.sqrt(2)
    var_needed = se_level_needed**2

    c_S = cost_model.surrogate_cost
    c_Y = cost_model.oracle_cost
    sigma2_eval = variance_model.sigma2_eval
    sigma2_cal = variance_model.sigma2_cal

    numerator = (np.sqrt(c_S * sigma2_eval) + np.sqrt(c_Y * sigma2_cal)) ** 2
    budget_needed = numerator / var_needed

    return plan_evaluation(
        budget=budget_needed,
        variance_model=variance_model,
        cost_model=cost_model,
        m_min=m_min,
        power=power,
        alpha=alpha,
    )


# =============================================================================
# Variance Measurement (using actual CJE pipeline)
# =============================================================================


def _measure_variance_direct(
    fresh_draws: "FreshDrawDataset",
    n_prompts: int,
    m_oracle: int,
    n_replicates: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Measure variance using the actual CJE pipeline (analyze_dataset).

    Runs analyze_dataset on subsampled data and uses the reported SE² as the
    variance estimate. This ensures planning uses the SAME estimator as actual
    evaluation (Direct mode with bootstrap inference).

    Args:
        fresh_draws: Full pilot dataset
        n_prompts: Number of prompts to subsample (n)
        m_oracle: Number of oracle-labeled prompts to include (m)
        n_replicates: Number of subsamples to average (default 5 for stability)
        seed: Random seed
    """
    from ..interface.analysis import analyze_dataset

    # Separate prompts with and without oracle labels
    oracle_prompt_ids = sorted(
        set(s.prompt_id for s in fresh_draws.samples if s.oracle_label is not None)
    )
    non_oracle_prompt_ids = sorted(
        set(s.prompt_id for s in fresh_draws.samples if s.oracle_label is None)
    )

    n_oracle_available = len(oracle_prompt_ids)
    n_non_oracle_available = len(non_oracle_prompt_ids)

    # Validate we have enough data
    if m_oracle > n_oracle_available:
        m_oracle = n_oracle_available
    n_non_oracle_needed = n_prompts - m_oracle
    if n_non_oracle_needed > n_non_oracle_available:
        n_non_oracle_needed = n_non_oracle_available
        n_prompts = m_oracle + n_non_oracle_needed

    rng = np.random.default_rng(seed)
    se_values: List[float] = []

    for rep in range(n_replicates):
        # Sample exactly m prompts with oracle and (n-m) without
        sampled_oracle = set(
            rng.choice(oracle_prompt_ids, size=m_oracle, replace=False)
        )
        sampled_non_oracle = (
            set(
                rng.choice(
                    non_oracle_prompt_ids, size=n_non_oracle_needed, replace=False
                )
            )
            if n_non_oracle_needed > 0
            else set()
        )

        # Build records - oracle prompts keep their labels, others don't
        subsampled_records = []
        for s in fresh_draws.samples:
            if s.prompt_id in sampled_oracle:
                subsampled_records.append(
                    {
                        "prompt_id": s.prompt_id,
                        "judge_score": s.judge_score,
                        "oracle_label": s.oracle_label,
                    }
                )
            elif s.prompt_id in sampled_non_oracle:
                subsampled_records.append(
                    {
                        "prompt_id": s.prompt_id,
                        "judge_score": s.judge_score,
                    }
                )

        if len(subsampled_records) < 20:
            continue

        try:
            policy_name = fresh_draws.target_policy
            result = analyze_dataset(
                fresh_draws_data={policy_name: subsampled_records},
                verbose=False,
            )

            if (
                result.standard_errors is not None
                and len(result.standard_errors) > 0
                and result.standard_errors[0] > 0
            ):
                se_values.append(float(result.standard_errors[0]))
        except Exception as e:
            logger.debug(f"analyze_dataset failed for replicate {rep}: {e}")
            continue

    if not se_values:
        return {
            "variance": np.nan,
            "se": np.nan,
            "n_actual": n_prompts,
            "m_actual": m_oracle,
            "n_valid_replicates": 0,
        }

    median_se = float(np.median(se_values))
    return {
        "variance": median_se**2,
        "se": median_se,
        "n_actual": n_prompts,
        "m_actual": m_oracle,
        "n_valid_replicates": len(se_values),
    }


def _fit_variance_model_from_measurements(
    measurements: List[Tuple[int, int, float]],
) -> FittedVarianceModel:
    """Fit Var = sigma2_eval/n + sigma2_cal/m using NNLS."""
    from scipy.optimize import nnls

    if len(measurements) < 3:
        raise ValueError(f"Need at least 3 measurements, got {len(measurements)}")

    valid = [(n, m, var) for n, m, var in measurements if var > 0 and not np.isnan(var)]
    if len(valid) < 3:
        raise ValueError(f"Need at least 3 valid measurements, got {len(valid)}")

    X = np.array([[1 / n, 1 / m] for n, m, _ in valid])
    y = np.array([var for _, _, var in valid])

    coeffs, _ = nnls(X, y)
    sigma2_eval, sigma2_cal = coeffs

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


# =============================================================================
# Diagnostic Utilities
# =============================================================================


def _check_labeling_ignorability(
    fresh_draws: "FreshDrawDataset",
) -> Dict[str, Any]:
    """Check if oracle labeling appears ignorable (random within policy)."""
    labeled_scores = []
    unlabeled_scores = []

    for s in fresh_draws.samples:
        if s.judge_score is not None:
            if s.oracle_label is not None:
                labeled_scores.append(s.judge_score)
            else:
                unlabeled_scores.append(s.judge_score)

    if len(labeled_scores) < 10 or len(unlabeled_scores) < 10:
        return {
            "is_ignorable": None,
            "ks_statistic": None,
            "ks_pvalue": None,
            "n_labeled": len(labeled_scores),
            "n_unlabeled": len(unlabeled_scores),
            "recommendation": "Not enough samples to test ignorability",
        }

    ks_stat, ks_pvalue = stats.ks_2samp(labeled_scores, unlabeled_scores)
    is_ignorable = ks_pvalue > 0.05

    recommendation = (
        "Labeling appears ignorable (good for variance model)"
        if is_ignorable
        else "Labeling may NOT be ignorable - consider random oracle sampling"
    )

    return {
        "is_ignorable": is_ignorable,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "n_labeled": len(labeled_scores),
        "n_unlabeled": len(unlabeled_scores),
        "recommendation": recommendation,
    }
