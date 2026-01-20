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

    # Load pilot data (use single policy for ignorable labeling)
    base_data = load_fresh_draws_auto(pilot_dir, "base")

    # Fit variance model from empirical measurements
    variance_model = fit_variance_model({"base": base_data}, verbose=True)
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
    fresh_draws_dict: Dict[str, "FreshDrawDataset"],
    n_grid: Optional[List[int]] = None,
    oracle_fraction_grid: Optional[List[float]] = None,
    n_replicates: int = 150,
    seed: int = 42,
    verbose: bool = True,
) -> FittedVarianceModel:
    """Fit Var(θ̂) = σ²_eval/n + σ²_cal/m from pilot data.

    Uses cross-fitted (out-of-fold) predictions for unbiased residual estimation.

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
        model = fit_variance_model({"base": pilot_data}, verbose=True)
        print(f"R² = {model.r_squared:.3f}")
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
        print(
            f"  Measuring {len(n_grid) * len(oracle_fraction_grid)} allocations "
            f"({n_replicates} replicates each)..."
        )

    # Measure variance at each grid point
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
# Variance Measurement (with OOF cross-fitting fix)
# =============================================================================


def _compute_point_estimate_single_replicate(
    subsampled_dict: Dict[str, "FreshDrawDataset"],
    seed: int,
    n_folds: int = 5,
) -> Optional[np.ndarray]:
    """Compute point estimate θ̂_aug using OUT-OF-FOLD cross-fitting.

    This is the key fix: we use cross-fitted (out-of-fold) predictions for the
    residual correction. This breaks the isotonic regression property that would
    make residuals zero on training data.

    The estimator is:
        θ̂_aug = mean(f̂_full(S)) + mean(Y - f̂_oof(S))

    where f̂_oof uses cross-fitted predictions (calibrator trained on other folds).

    Returns:
        Array of point estimates per policy, or None if calibration failed.
    """
    from .robust_inference import build_direct_eval_table
    from ..calibration.judge import JudgeCalibrator

    eval_table = build_direct_eval_table(subsampled_dict)

    n_oracle = int(np.sum(eval_table.oracle_mask))
    if n_oracle < 10:
        return None

    oracle_indices = np.where(eval_table.oracle_mask)[0]
    n_oracle_total = len(oracle_indices)

    if n_oracle_total < n_folds:
        n_folds = max(2, n_oracle_total)

    try:
        rng = np.random.default_rng(seed)
        fold_assignments = np.zeros(len(eval_table.oracle_mask), dtype=int)
        shuffled_oracle_idx = rng.permutation(oracle_indices)
        for i, idx in enumerate(shuffled_oracle_idx):
            fold_assignments[idx] = i % n_folds

        # Compute OUT-OF-FOLD calibrated predictions
        calibrated_oof = np.full(len(eval_table.judge_scores), np.nan)

        for fold in range(n_folds):
            train_mask = eval_table.oracle_mask & (fold_assignments != fold)
            test_mask = eval_table.oracle_mask & (fold_assignments == fold)

            if np.sum(train_mask) < 5:
                continue

            calibrator = JudgeCalibrator(
                random_seed=seed + fold,
                calibration_mode="monotone",
            )

            # Fit calibrator on training fold oracle samples
            train_judge = eval_table.judge_scores[train_mask]
            train_oracle = eval_table.oracle_labels[train_mask]
            calibrator.fit_transform(
                judge_scores=train_judge,
                oracle_labels=train_oracle,
            )

            # Predict on test fold (out-of-fold predictions)
            test_judge = eval_table.judge_scores[test_mask]
            calibrated_oof[test_mask] = calibrator.predict(test_judge)

        # Full model for plug-in estimate
        full_calibrator = JudgeCalibrator(random_seed=seed, calibration_mode="monotone")
        result = full_calibrator.fit_transform(
            judge_scores=eval_table.judge_scores,
            oracle_labels=eval_table.oracle_labels[eval_table.oracle_mask],
            oracle_mask=eval_table.oracle_mask,
        )
        calibrated_full = result.calibrated_scores

        # Compute θ̂_aug per policy
        n_policies = eval_table.n_policies
        estimates = np.zeros(n_policies)

        for p in range(n_policies):
            p_mask = eval_table.policy_indices == p
            n_p = np.sum(p_mask)

            if n_p == 0:
                estimates[p] = np.nan
                continue

            plug_in = np.mean(calibrated_full[p_mask])

            p_oracle_mask = p_mask & eval_table.oracle_mask
            n_oracle_p = np.sum(p_oracle_mask)

            if n_oracle_p > 0:
                oof_preds = calibrated_oof[p_oracle_mask]
                oracle_vals = eval_table.oracle_labels[p_oracle_mask]
                valid = ~np.isnan(oof_preds)
                if np.sum(valid) > 0:
                    residuals = oracle_vals[valid] - oof_preds[valid]
                    residual_correction = np.mean(residuals)
                else:
                    residual_correction = 0.0
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
    """Measure variance by computing Var(θ̂) across outer replicates."""
    from ..data.fresh_draws import FreshDrawDataset, FreshDrawSample

    all_prompt_ids: set[str] = set()
    for fd in fresh_draws_dict.values():
        all_prompt_ids.update(s.prompt_id for s in fd.samples)

    all_prompt_ids_list = sorted(all_prompt_ids)
    n_available = len(all_prompt_ids_list)

    if n_prompts > n_available:
        n_prompts = n_available

    m_oracle = max(int(n_prompts * oracle_fraction), 1)

    rng = np.random.default_rng(seed)
    estimates_per_replicate: List[np.ndarray] = []

    for rep in range(n_replicates):
        sampled_prompts = set(
            rng.choice(all_prompt_ids_list, size=n_prompts, replace=False)
        )
        oracle_prompts = set(
            rng.choice(list(sampled_prompts), size=m_oracle, replace=False)
        )

        subsampled_dict: Dict[str, FreshDrawDataset] = {}
        for policy_name, fd in fresh_draws_dict.items():
            subsampled_samples = []
            for s in fd.samples:
                if s.prompt_id in sampled_prompts:
                    if s.prompt_id in oracle_prompts:
                        subsampled_samples.append(s)
                    else:
                        subsampled_samples.append(
                            FreshDrawSample(
                                prompt_id=s.prompt_id,
                                target_policy=s.target_policy,
                                judge_score=s.judge_score,
                                oracle_label=None,
                                response=s.response,
                                draw_idx=s.draw_idx,
                                metadata=s.metadata,
                            )
                        )

            if subsampled_samples:
                subsampled_dict[policy_name] = FreshDrawDataset(
                    target_policy=fd.target_policy,
                    draws_per_prompt=fd.draws_per_prompt,
                    samples=subsampled_samples,
                )

        estimates = _compute_point_estimate_single_replicate(
            subsampled_dict, seed=seed + rep
        )
        if estimates is not None:
            estimates_per_replicate.append(estimates)

    if len(estimates_per_replicate) < 10:
        return {
            "variance": np.nan,
            "se": np.nan,
            "n_actual": n_prompts,
            "m_actual": m_oracle,
            "n_valid_replicates": len(estimates_per_replicate),
        }

    estimates_array = np.array(estimates_per_replicate)
    var_per_policy = np.nanvar(estimates_array, axis=0, ddof=1)
    mean_variance = np.nanmean(var_per_policy)

    return {
        "variance": float(mean_variance),
        "se": float(np.sqrt(mean_variance)),
        "n_actual": n_prompts,
        "m_actual": m_oracle,
        "n_valid_replicates": len(estimates_per_replicate),
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


def check_labeling_ignorability(
    fresh_draws_dict: Dict[str, "FreshDrawDataset"],
) -> Dict[str, Any]:
    """Check if oracle labeling appears ignorable (random within policy)."""
    labeled_scores = []
    unlabeled_scores = []

    for fd in fresh_draws_dict.values():
        for s in fd.samples:
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
