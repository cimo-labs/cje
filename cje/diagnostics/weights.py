"""
Weight diagnostic computations for importance sampling.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import logging

from .models import Status

logger = logging.getLogger(__name__)


# ========== Core Metrics ==========


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute ESS = (sum(w))^2 / sum(w^2).

    ESS measures how many "effective" samples we have after weighting.
    ESS = n means perfect overlap, ESS << n means poor overlap.
    """
    s = weights.sum()
    s2 = np.sum(weights**2)
    return float((s * s) / np.maximum(s2, 1e-12))


def compute_ess(weights: np.ndarray) -> float:
    """Alias for effective_sample_size() - kept for backward compatibility."""
    return effective_sample_size(weights)


def tail_weight_ratio(
    weights: np.ndarray, q_low: float = 0.05, q_high: float = 0.99
) -> float:
    """Compute ratio of high to low quantiles.

    DEPRECATED: Use hill_tail_index() instead for more theoretically grounded
    tail behavior assessment.

    Args:
        weights: Importance weights
        q_low: Lower quantile (default 0.05 to avoid instability)
        q_high: Upper quantile (default 0.99)

    Returns:
        Ratio of high/low quantiles (inf if low quantile is ~0)
    """
    lo = np.quantile(weights, q_low)
    hi = np.quantile(weights, q_high)
    if lo <= 1e-12:
        return float(np.inf)
    return float(hi / lo)


def mass_concentration(weights: np.ndarray, top_pct: float = 0.01) -> float:
    """Fraction of total weight held by top x% of samples.

    Args:
        weights: Importance weights
        top_pct: Top percentage to consider (0.01 = top 1%)

    Returns:
        Fraction of total weight in top samples
    """
    n = len(weights)
    k = max(1, int(n * top_pct))
    sorted_weights = np.sort(weights)[::-1]  # Descending
    return float(sorted_weights[:k].sum() / weights.sum())


# ========== Tail Index Estimation ==========


def hill_tail_index(
    weights: np.ndarray,
    k_fraction: float = 0.05,
    min_k: int = 10,
    max_k: Optional[int] = None,
) -> float:
    """Estimate the tail index using Hill's estimator.

    The Hill estimator quantifies the heaviness of the right tail.
    For a Pareto-type tail with P(X > x) ~ x^(-α), it estimates α.

    Interpretation:
    - α > 3: Light tails, all moments exist
    - 2 < α ≤ 3: Moderate tails, variance exists
    - 1 < α ≤ 2: Heavy tails, variance may be infinite
    - α ≤ 1: Very heavy tails, even mean may be infinite

    Args:
        weights: Importance weights (positive)
        k_fraction: Fraction of largest values to use (default 0.05 = top 5%)
        min_k: Minimum number of order statistics to use
        max_k: Maximum number of order statistics (default: 10% of n)

    Returns:
        Estimated tail index α. Lower values = heavier tails.
        Returns np.inf if estimation fails.

    References:
        Hill, B. M. (1975). A simple general approach to inference about
        the tail of a distribution. Ann. Statist. 3(5): 1163-1174.
    """
    n = len(weights)
    if n < min_k:
        logger.warning(f"Too few samples ({n}) for Hill estimator, need >= {min_k}")
        return float(np.inf)

    # Determine k (number of order statistics to use)
    k = max(min_k, int(n * k_fraction))
    if max_k is not None:
        k = min(k, max_k)
    else:
        k = min(k, int(n * 0.1))  # Default max: 10% of samples

    # Get the k largest weights
    sorted_weights = np.sort(weights)[::-1]  # Descending order

    # Check for degenerate cases
    if sorted_weights[k] <= 0 or sorted_weights[0] <= 0:
        logger.warning("Zero or negative weights in tail, cannot estimate tail index")
        return float(np.inf)

    # Hill estimator: α̂ = 1/k * Σ(log(X_i) - log(X_{k+1}))
    # where X_i are the top k order statistics
    log_ratios = np.log(sorted_weights[:k]) - np.log(sorted_weights[k])

    # Check for numerical issues
    if not np.all(np.isfinite(log_ratios)):
        logger.warning("Numerical issues in Hill estimator (inf/nan in log ratios)")
        return float(np.inf)

    # Check for zero sum (would cause division by zero)
    log_ratio_sum = np.sum(log_ratios)
    if abs(log_ratio_sum) < 1e-10:
        logger.debug("Hill estimator undefined (uniform weights in tail)")
        return float(np.inf)  # Undefined for uniform weights

    hill_estimate = k / log_ratio_sum

    # Sanity check the estimate
    if hill_estimate <= 0 or not np.isfinite(hill_estimate):
        logger.warning(f"Invalid Hill estimate: {hill_estimate}")
        return float(np.inf)

    return float(hill_estimate)


def hill_tail_index_stable(
    weights: np.ndarray,
    k_fractions: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute Hill estimator over multiple k values for stability assessment.

    Args:
        weights: Importance weights
        k_fractions: Array of fractions to try (default: [0.01, 0.02, 0.05, 0.1])

    Returns:
        Dictionary with:
        - 'estimate': Median estimate across k values
        - 'min': Minimum estimate
        - 'max': Maximum estimate
        - 'std': Standard deviation of estimates
    """
    if k_fractions is None:
        k_fractions = np.array([0.01, 0.02, 0.05, 0.1])

    estimates: List[float] = []
    for k_frac in k_fractions:
        est = hill_tail_index(weights, k_fraction=k_frac)
        if np.isfinite(est):
            estimates.append(est)

    if not estimates:
        return {
            "estimate": float(np.inf),
            "min": float(np.inf),
            "max": float(np.inf),
            "std": 0.0,
        }

    estimates_array = np.array(estimates)
    return {
        "estimate": float(np.median(estimates_array)),
        "min": float(np.min(estimates_array)),
        "max": float(np.max(estimates_array)),
        "std": float(np.std(estimates_array)),
    }


# ========== Diagnostic Computation ==========


def compute_weight_diagnostics(
    weights: np.ndarray,
    policy: str = "unknown",
    compute_hill: bool = True,
) -> Dict[str, Any]:
    """Compute weight diagnostics for a single policy.

    Returns dict with: ess_fraction, max_weight, tail_index (if computed), status
    """
    n = len(weights)
    ess = effective_sample_size(weights)
    ess_fraction = ess / n if n > 0 else 0.0

    # Hill tail index (primary tail measure)
    if compute_hill and n >= 50:  # Need reasonable sample size
        tail_index = hill_tail_index(weights)
    else:
        tail_index = None

    # Determine status based on ESS and tail index
    if ess_fraction < 0.01:
        status = Status.CRITICAL
    elif tail_index is not None and tail_index < 1:
        # Very heavy tail - infinite mean risk
        status = Status.CRITICAL
    elif ess_fraction < 0.1:
        status = Status.WARNING
    elif tail_index is not None and tail_index < 2:
        # Heavy tail - infinite variance risk
        status = Status.WARNING
    else:
        status = Status.GOOD

    result = {
        "ess_fraction": ess_fraction,
        "max_weight": float(weights.max()),
        "status": status,
    }

    if tail_index is not None:
        result["tail_index"] = tail_index

    return result
