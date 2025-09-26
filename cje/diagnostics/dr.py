"""
Doubly robust diagnostic computations.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from scipy import stats
import logging

from .weights import tail_weight_ratio, mass_concentration

logger = logging.getLogger(__name__)


def _p_value_from_z(z: float) -> float:
    """Convert z-score to two-sided p-value."""
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def compute_orthogonality_score(
    weights: np.ndarray,
    rewards: np.ndarray,
    outcome_predictions: np.ndarray,
    return_ci: bool = True,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute the orthogonality score for DR estimation.

    The orthogonality score is E[W * (R - q̂)], which should be zero
    under correct specification and proper cross-fitting.

    Args:
        weights: Importance weights (should be mean-one)
        rewards: Observed rewards
        outcome_predictions: Outcome model predictions q̂(X, A)
        return_ci: Whether to compute confidence interval
        alpha: Significance level for CI (default 0.05 for 95% CI)

    Returns:
        Dictionary with:
        - 'score': The orthogonality score
        - 'se': Standard error of the score
        - 'ci_lower': Lower bound of CI (if return_ci=True)
        - 'ci_upper': Upper bound of CI (if return_ci=True)
        - 'p_value': P-value for test that score = 0
        - 'passes_test': Boolean, True if CI contains 0

    References:
        Section 9.3 of the CJE paper on orthogonality diagnostics.
    """
    n = len(weights)

    # Compute the orthogonality score
    residuals = rewards - outcome_predictions
    score_components = weights * residuals
    score = np.mean(score_components)

    # Compute standard error
    se = np.std(score_components) / np.sqrt(n)

    # Test statistic
    z_stat = score / se if se > 0 else 0
    p_value = _p_value_from_z(z_stat)

    result = {
        "score": float(score),
        "se": float(se),
        "z_statistic": float(z_stat),
        "p_value": float(p_value),
    }

    if return_ci:
        # Critical value for two-sided test
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_lower = score - z_crit * se
        ci_upper = score + z_crit * se

        result["ci_lower"] = float(ci_lower)
        result["ci_upper"] = float(ci_upper)
        result["passes_test"] = ci_lower <= 0 <= ci_upper

    return result


def compute_dm_ips_decomposition(
    g_hat: np.ndarray,
    weights: np.ndarray,
    rewards: np.ndarray,
    q_hat: np.ndarray,
) -> Dict[str, Any]:
    """Compute the DM-IPS decomposition for DR estimation.

    Decomposes the DR estimate into:
    - Direct Method (DM): E[ĝ(X)]
    - IPS Augmentation: E[Ŵ * (R - q̂(X, A))]

    Args:
        g_hat: Outcome model predictions under target policy ĝ(X)
        weights: Importance weights Ŵ
        rewards: Observed rewards R
        q_hat: Outcome model predictions under logging policy q̂(X, A)

    Returns:
        Dictionary with:
        - 'dm_component': Direct method estimate
        - 'ips_augmentation': IPS correction term
        - 'total': Total DR estimate
        - 'dm_se': Standard error of DM component
        - 'ips_se': Standard error of IPS component
        - 'dm_contribution': Fraction of estimate from DM
        - 'ips_contribution': Fraction of estimate from IPS
        - 'correlation': Correlation between components
    """
    n = len(weights)

    # DM component
    dm_component = np.mean(g_hat)
    dm_se = np.std(g_hat) / np.sqrt(n)

    # IPS augmentation
    residuals = rewards - q_hat
    ips_terms = weights * residuals
    ips_augmentation = np.mean(ips_terms)
    ips_se = np.std(ips_terms) / np.sqrt(n)

    # Total DR estimate
    total = dm_component + ips_augmentation

    # Component contributions (as fractions)
    if abs(total) > 1e-10:
        dm_contribution = abs(dm_component) / (
            abs(dm_component) + abs(ips_augmentation)
        )
        ips_contribution = abs(ips_augmentation) / (
            abs(dm_component) + abs(ips_augmentation)
        )
    else:
        dm_contribution = 0.5
        ips_contribution = 0.5

    # Correlation between components
    if len(g_hat) > 1:
        corr_matrix = np.corrcoef(g_hat, ips_terms)
        correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
    else:
        correlation = 0.0

    return {
        "dm_component": float(dm_component),
        "ips_augmentation": float(ips_augmentation),
        "total": float(total),
        "dm_se": float(dm_se),
        "ips_se": float(ips_se),
        "dm_contribution": float(dm_contribution),
        "ips_contribution": float(ips_contribution),
        "correlation": float(correlation),
    }


def compute_dr_policy_diagnostics(
    dm_component: np.ndarray,
    ips_correction: np.ndarray,
    dr_estimate: float,
    fresh_rewards: Optional[np.ndarray] = None,
    outcome_predictions: Optional[np.ndarray] = None,
    influence_functions: Optional[np.ndarray] = None,
    unique_folds: Optional[List[int]] = None,
    policy: str = "unknown",
) -> Dict[str, Any]:
    """Compute comprehensive DR diagnostics for a single policy.

    Args:
        dm_component: Direct method (outcome model) component
        ips_correction: IPS correction component
        dr_estimate: Final DR estimate
        fresh_rewards: Fresh draw rewards (for outcome model R²)
        outcome_predictions: Outcome model predictions
        influence_functions: Per-sample influence functions
        unique_folds: Unique fold IDs used in cross-fitting
        policy: Policy name

    Returns:
        Dictionary with diagnostic metrics
    """
    n = len(dm_component)

    diagnostics = {
        "policy": policy,
        "n_samples": n,
        "dm_mean": float(dm_component.mean()),
        "ips_corr_mean": float(ips_correction.mean()),
        "dr_estimate": float(dr_estimate),
        "dm_std": float(dm_component.std()),
        "ips_corr_std": float(ips_correction.std()),
    }

    # Outcome model fit (if fresh rewards available)
    if fresh_rewards is not None and outcome_predictions is not None:
        mask = ~np.isnan(fresh_rewards) & ~np.isnan(outcome_predictions)
        if mask.sum() > 0:
            residuals = fresh_rewards[mask] - outcome_predictions[mask]
            diagnostics["residual_mean"] = float(residuals.mean())
            diagnostics["residual_std"] = float(residuals.std())
            diagnostics["residual_rmse"] = float(np.sqrt((residuals**2).mean()))

            # R² (out-of-fold if cross-fitted)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((fresh_rewards[mask] - fresh_rewards[mask].mean()) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-12)
            diagnostics["r2_oof"] = float(r2)
        else:
            diagnostics["r2_oof"] = np.nan
            diagnostics["residual_rmse"] = np.nan

    # Influence function diagnostics
    if influence_functions is not None:
        diagnostics["if_mean"] = float(influence_functions.mean())
        diagnostics["if_std"] = float(influence_functions.std())
        diagnostics["if_var"] = float(influence_functions.var())

        # Check for heavy tails
        diagnostics["if_tail_ratio_99_5"] = tail_weight_ratio(
            np.abs(influence_functions), 0.05, 0.99
        )
        diagnostics["if_top1_mass"] = mass_concentration(
            np.abs(influence_functions), 0.01
        )

        # Score function test (should be mean zero for TMLE)
        score_mean = influence_functions.mean()
        score_se = influence_functions.std() / np.sqrt(n)
        score_z = score_mean / score_se if score_se > 0 else 0
        diagnostics["score_mean"] = float(score_mean)
        diagnostics["score_se"] = float(score_se)
        diagnostics["score_z"] = float(score_z)
        diagnostics["score_p"] = _p_value_from_z(score_z)

    # Cross-fitting info
    if unique_folds is not None:
        diagnostics["cross_fitted"] = True
        diagnostics["unique_folds"] = len(unique_folds)
    else:
        diagnostics["cross_fitted"] = False
        diagnostics["unique_folds"] = 1

    # Coverage check (do we have enough fresh draws?)
    diagnostics["coverage_ok"] = True
    if fresh_rewards is not None:
        coverage = (~np.isnan(fresh_rewards)).mean()
        diagnostics["fresh_draw_coverage"] = float(coverage)
        if coverage < 0.8:
            diagnostics["coverage_ok"] = False
            logger.warning(f"Low fresh draw coverage for {policy}: {coverage:.1%}")

    # Component correlation (ideally low for orthogonality)
    corr = np.corrcoef(dm_component, ips_correction)[0, 1]
    diagnostics["component_correlation"] = float(corr) if not np.isnan(corr) else 0.0

    return diagnostics


def compute_dr_diagnostics_all(
    estimator: Any,
    influence_functions: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Compute DR diagnostics for all policies.

    Args:
        estimator: A DR estimator with _dm_component and _ips_correction
        influence_functions: Optional dict of influence functions by policy

    Returns:
        Dictionary with per-policy diagnostics and summary metrics
    """
    all_diagnostics = {}

    for policy in estimator.sampler.target_policies:
        # Get components
        dm = estimator._dm_component.get(policy)
        ips = estimator._ips_correction.get(policy)

        if dm is None or ips is None:
            logger.warning(f"Missing DR components for {policy}")
            continue

        # Get optional data
        fresh_rewards = None
        outcome_preds = None
        if hasattr(estimator, "_fresh_rewards"):
            fresh_rewards = estimator._fresh_rewards.get(policy)
        if hasattr(estimator, "_outcome_predictions"):
            outcome_preds = estimator._outcome_predictions.get(policy)

        # Get influence functions if available
        ifs = None
        if influence_functions and policy in influence_functions:
            ifs = influence_functions[policy]

        # Compute diagnostics
        all_diagnostics[policy] = compute_dr_policy_diagnostics(
            dm_component=dm,
            ips_correction=ips,
            dr_estimate=dm.mean() + ips.mean(),
            fresh_rewards=fresh_rewards,
            outcome_predictions=outcome_preds,
            influence_functions=ifs,
            policy=policy,
        )

    return all_diagnostics
