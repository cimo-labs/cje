"""
Overlap metrics for importance sampling diagnostics.

These metrics quantify how well two policies overlap, which determines
the reliability of importance-weighted estimates. The key insight is that
some metrics (like Hellinger affinity) measure structural compatibility
that cannot be improved by calibration, while others (like ESS) can be
improved through techniques like SIMCal.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class OverlapMetrics:
    """Comprehensive overlap diagnostics between policies.

    Attributes:
        hellinger_affinity: Bhattacharyya coefficient ∈ (0,1], measures structural overlap.
            1.0 = perfect overlap, < 0.2 = catastrophic mismatch
        ess_fraction: Effective sample size as fraction of n ∈ (0,1].
            Measures statistical efficiency. 1.0 = perfect, < 0.1 = poor
        tail_index: Hill estimator of tail heaviness.
            None if n < 50, < 2 indicates infinite variance
        overlap_quality: Categorical assessment of overlap
        efficiency_loss: Fraction of data effectively wasted (1 - ess_fraction)
        can_calibrate: Whether calibration methods like SIMCal could help
        recommended_method: Suggested estimation method given the overlap
        confidence_penalty: How much wider CIs are vs uniform sampling
        auto_tuned_threshold: ESS threshold for target CI width (if computed)
    """

    # Core metrics
    hellinger_affinity: float  # ∈ (0,1], structural overlap
    ess_fraction: float  # ∈ (0,1], statistical efficiency
    tail_index: Optional[float]  # > 0, tail heaviness (None if n < 50)

    # Derived interpretations
    overlap_quality: str  # "good", "marginal", "poor", "catastrophic"
    efficiency_loss: float  # How much data we're effectively losing
    can_calibrate: bool  # Whether SIMCal can potentially help

    # Recommendations
    recommended_method: str  # "ips", "calibrated-ips", "dr", "refuse"
    confidence_penalty: float  # CI width multiplier vs uniform sampling

    # Auto-tuning info
    auto_tuned_threshold: Optional[float] = None  # ESS threshold for target CI

    # σ(S) structural floors
    aessf_sigmaS: Optional[float] = None  # A-ESSF on judge marginal σ(S)
    aessf_sigmaS_lcb: Optional[float] = None  # Lower confidence bound for A-ESSF
    bc_sigmaS: Optional[float] = None  # Bhattacharyya coefficient on σ(S)

    def summary(self) -> str:
        """Human-readable summary of overlap diagnostics."""
        return (
            f"Overlap: {self.overlap_quality} "
            f"({self.hellinger_affinity:.0%} similarity, "
            f"{self.ess_fraction:.0%} efficiency). "
            f"Recommendation: {self.recommended_method}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "hellinger_affinity": self.hellinger_affinity,
            "ess_fraction": self.ess_fraction,
            "tail_index": self.tail_index,
            "overlap_quality": self.overlap_quality,
            "efficiency_loss": self.efficiency_loss,
            "can_calibrate": self.can_calibrate,
            "recommended_method": self.recommended_method,
            "confidence_penalty": self.confidence_penalty,
            "auto_tuned_threshold": self.auto_tuned_threshold,
            "aessf_sigmaS": self.aessf_sigmaS,
            "aessf_sigmaS_lcb": self.aessf_sigmaS_lcb,
            "bc_sigmaS": self.bc_sigmaS,
        }
        return {k: v for k, v in d.items() if v is not None}  # Filter None values


def hellinger_affinity(weights: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Bhattacharyya coefficient (Hellinger affinity).

    This measures the overlap between two distributions. For importance weights
    w = p'/p, the affinity is E[√w] under the base distribution.

    Key properties:
    - Value in (0, 1] where 1 indicates perfect overlap
    - Cannot be improved by weight calibration (measures structural mismatch)
    - Related to Hellinger distance: H = √(1 - A²)

    Args:
        weights: Importance weights (will be normalized to mean 1)
        epsilon: Small constant for numerical stability

    Returns:
        Hellinger affinity (Bhattacharyya coefficient)
    """
    weights = np.asarray(weights)

    # Handle empty or invalid input
    if len(weights) == 0:
        return float(np.nan)

    # Remove any negative or nan weights (shouldn't exist but be defensive)
    valid_mask = (weights >= 0) & np.isfinite(weights)
    if not np.any(valid_mask):
        logger.warning("No valid weights found for Hellinger affinity computation")
        return float(np.nan)

    weights_valid = weights[valid_mask]

    # Normalize to mean 1 for numerical stability and interpretability
    mean_w = np.mean(weights_valid)
    if mean_w <= epsilon:
        return 0.0  # Catastrophic case - no overlap

    normalized = weights_valid / mean_w

    # Compute affinity with numerical guards
    # For mean-1 weights, this equals E[√w]
    sqrt_weights = np.sqrt(np.maximum(normalized, epsilon))
    affinity = float(np.mean(sqrt_weights))

    # Theoretical bound: affinity ∈ (0, 1] for mean-1 weights
    # In practice might slightly exceed 1 due to numerics
    return min(affinity, 1.0)


def compute_auto_tuned_threshold(
    n: int, target_ci_halfwidth: float, level: str = "critical"
) -> float:
    """
    Compute ESS threshold for desired confidence interval width.

    Based on the variance bound for IPS with bounded rewards:
    Var(V_IPS) ≤ 1/(4n·ESS_fraction)

    This gives a 95% CI half-width of approximately:
    HW ≈ 1.96/(2√(n·ESS_fraction))

    Solving for ESS_fraction given target HW:
    ESS_fraction ≥ (1.96/(2·target))² / n

    Which simplifies to:
    ESS_fraction ≥ 0.9604 / (n·target²)

    Args:
        n: Sample size
        target_ci_halfwidth: Desired CI half-width (e.g., 0.01 for ±1%)
        level: "critical" or "warning" (warning uses half the critical threshold)

    Returns:
        Minimum ESS fraction needed for target precision
    """
    if n <= 0 or target_ci_halfwidth <= 0:
        return 0.1  # Fallback to default

    # Based on variance bound for bounded rewards
    # (1.96/2)² = 0.9604
    threshold = 0.9604 / (n * target_ci_halfwidth**2)

    if level == "warning":
        threshold *= 0.5  # Warning at half the critical level

    # Cap at reasonable bounds
    return min(max(threshold, 0.001), 1.0)


def compute_overlap_metrics(
    weights: np.ndarray,
    target_ci_halfwidth: float = 0.01,
    n_samples: Optional[int] = None,
    compute_tail_index: bool = True,
    auto_tune_threshold: bool = False,
) -> OverlapMetrics:
    """
    Compute comprehensive overlap diagnostics.

    This function computes three complementary metrics:
    1. Hellinger affinity: Structural overlap (cannot be improved)
    2. ESS fraction: Statistical efficiency (can be improved by calibration)
    3. Tail index: Pathological behavior (partially improvable)

    Args:
        weights: Importance weights (will be normalized to mean 1)
        target_ci_halfwidth: Desired CI half-width for auto-tuning
        n_samples: Sample size (defaults to len(weights))
        compute_tail_index: Whether to compute Hill tail index
        auto_tune_threshold: Whether to compute auto-tuned ESS threshold

    Returns:
        OverlapMetrics with diagnostics and recommendations
    """
    weights = np.asarray(weights)
    n = n_samples or len(weights)

    if len(weights) == 0:
        # Return worst-case metrics for empty input
        return OverlapMetrics(
            hellinger_affinity=0.0,
            ess_fraction=0.0,
            tail_index=None,
            overlap_quality="catastrophic",
            efficiency_loss=1.0,
            can_calibrate=False,
            recommended_method="refuse",
            confidence_penalty=np.inf,
            auto_tuned_threshold=None,
        )

    # Normalize to mean 1 for consistent metrics
    weights = weights / np.mean(weights)

    # 1. Hellinger affinity (structural overlap)
    hellinger = hellinger_affinity(weights)

    # 2. ESS fraction (statistical efficiency)
    ess = float(np.sum(weights) ** 2 / np.sum(weights**2))
    ess_fraction = ess / n

    # 3. Tail index (pathological behavior)
    tail_index = None
    if compute_tail_index and n >= 50:
        try:
            from .weights import hill_tail_index

            tail_index = hill_tail_index(weights)
        except (ImportError, ValueError) as e:
            logger.debug(f"Could not compute tail index: {e}")

    # 4. Auto-tuned threshold (if requested)
    auto_tuned_threshold = None
    if auto_tune_threshold:
        auto_tuned_threshold = compute_auto_tuned_threshold(
            n, target_ci_halfwidth, "critical"
        )

    # Interpret overlap quality based on Hellinger affinity
    if hellinger < 0.20:
        quality = "catastrophic"
        can_calibrate = False  # Too far gone
    elif hellinger < 0.35:
        quality = "poor"
        can_calibrate = True  # Might help somewhat
    elif hellinger < 0.50:
        quality = "marginal"
        can_calibrate = True
    else:
        quality = "good"
        can_calibrate = True

    # Compute efficiency loss (how much data we're wasting)
    efficiency_loss = 1.0 - ess_fraction

    # Confidence interval penalty vs uniform sampling
    # Based on Var ≤ 1/(4n·ESS_frac) for bounded rewards
    if ess_fraction > 0.001:
        confidence_penalty = 1.0 / np.sqrt(ess_fraction)
    else:
        confidence_penalty = np.inf

    # Recommendation engine based on all metrics
    if quality == "catastrophic":
        recommended = "refuse"
    elif tail_index and tail_index < 1.5:
        # Extremely heavy tails - need bias correction
        recommended = "dr"
    elif ess_fraction < 0.10:
        # Low ESS - depends on overlap quality
        if quality == "poor":
            recommended = "refuse"
        else:
            recommended = "dr"
    elif ess_fraction < 0.30 and can_calibrate:
        # Moderate ESS with decent overlap - calibration can help
        recommended = "calibrated-ips"
    else:
        # Good enough for standard IPS
        recommended = "ips"

    return OverlapMetrics(
        hellinger_affinity=hellinger,
        ess_fraction=ess_fraction,
        tail_index=tail_index,
        overlap_quality=quality,
        efficiency_loss=efficiency_loss,
        can_calibrate=can_calibrate,
        recommended_method=recommended,
        confidence_penalty=confidence_penalty,
        auto_tuned_threshold=auto_tuned_threshold,
    )


def diagnose_overlap_problems(
    metrics: OverlapMetrics, verbose: bool = True
) -> Tuple[bool, str]:
    """
    Diagnose overlap problems and suggest solutions.

    Provides human-readable explanations of overlap issues and
    actionable recommendations for addressing them.

    Args:
        metrics: Computed overlap metrics
        verbose: Whether to print diagnosis

    Returns:
        Tuple of (should_proceed, explanation)
    """
    msgs = []

    # Explain the problem in intuitive terms
    if metrics.overlap_quality == "catastrophic":
        msgs.append(
            f"❌ Catastrophic overlap ({metrics.hellinger_affinity:.0%} similarity)\n"
            f"   The policies are fundamentally incompatible - like comparing\n"
            f"   apples to oranges. No statistical method can fix this.\n"
            f"   {metrics.efficiency_loss:.0%} of your data is effectively ignored."
        )
        should_proceed = False

    elif metrics.overlap_quality == "poor":
        msgs.append(
            f"⚠️  Poor overlap ({metrics.hellinger_affinity:.0%} similarity)\n"
            f"   Only {metrics.ess_fraction:.0%} of your data is effectively used.\n"
            f"   Confidence intervals will be {metrics.confidence_penalty:.1f}× wider."
        )
        should_proceed = True

    elif metrics.overlap_quality == "marginal":
        msgs.append(
            f"⚠️  Marginal overlap ({metrics.hellinger_affinity:.0%} similarity)\n"
            f"   {metrics.ess_fraction:.0%} effective sample size.\n"
            f"   Some variance inflation expected."
        )
        should_proceed = True

    else:
        msgs.append(
            f"✓ Good overlap ({metrics.hellinger_affinity:.0%} similarity)\n"
            f"  {metrics.ess_fraction:.0%} effective sample size"
        )
        should_proceed = True

    # Add specific warnings about tail behavior
    if metrics.tail_index is not None:
        if metrics.tail_index < 1:
            msgs.append(
                f"⚠️  Extremely heavy tails (α={metrics.tail_index:.2f})\n"
                f"   Infinite mean - estimates are unreliable!"
            )
        elif metrics.tail_index < 2:
            msgs.append(
                f"⚠️  Heavy tails detected (α={metrics.tail_index:.2f})\n"
                f"   Infinite variance - estimates may be unstable."
            )

    # Provide actionable recommendations
    msgs.append("\n📊 Recommendation:")
    if metrics.recommended_method == "refuse":
        msgs.append("   Do not proceed with importance sampling estimation.")
        msgs.append("   Solutions:")
        msgs.append("   • Use policies with better overlap (>35% similarity)")
        msgs.append("   • Collect data under a more diverse logging policy")
        msgs.append("   • Consider online A/B testing instead")

    elif metrics.recommended_method == "dr":
        msgs.append("   Use doubly-robust methods with fresh draws.")
        msgs.append("   The outcome model can compensate for poor overlap.")

    elif metrics.recommended_method == "calibrated-ips":
        msgs.append("   Use CalibratedIPS for variance reduction.")
        msgs.append("   Weight calibration can improve efficiency by 2-3×.")

    else:
        msgs.append("   Standard IPS should work adequately.")

    # Add auto-tuning info if available
    if metrics.auto_tuned_threshold is not None:
        msgs.append(
            f"\n📏 Auto-tuned ESS threshold: {metrics.auto_tuned_threshold:.1%}"
        )
        if metrics.ess_fraction >= metrics.auto_tuned_threshold:
            msgs.append("   ✓ Meets threshold for target precision")
        else:
            msgs.append("   ✗ Below threshold for target precision")

    explanation = "\n".join(msgs)

    if verbose:
        print(explanation)

    return should_proceed, explanation
