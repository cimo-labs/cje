"""
Overlap metrics for importance sampling diagnostics.

These metrics quantify how well two policies overlap, which determines
the reliability of importance-weighted estimates. The key insight is that
some metrics (like Hellinger affinity) measure structural compatibility
that cannot be improved by calibration, while others (like ESS) can be
improved through weight stabilization.
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
        can_calibrate: Whether weight stabilization could help
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
    can_calibrate: bool  # Whether weight stabilization can potentially help

    # Recommendations
    recommended_method: str  # "ips", "calibrated-ips", "dr", "refuse"
    confidence_penalty: float  # CI width multiplier vs uniform sampling

    # Auto-tuning info
    auto_tuned_threshold: Optional[float] = None  # ESS threshold for target CI

    # σ(S) structural floors (populated when judge scores are provided)
    # bc_sigmaS: binned Bhattacharyya affinity A_B between the judge-score
    #   marginals under pi_0 and pi' (the paper's gate: A_B >= 0.85).
    # aessf_sigmaS: A-ESSF = A_B^2, a structural ceiling on the ESS fraction
    #   achievable by ANY weight function measurable in sigma(S)
    #   (Cauchy-Schwarz: 1 = E[w] <= sqrt(E[w^{3/2}] E[w^{1/2}]) and
    #   Jensen give ESS/n = 1/E[w^2] <= (E[sqrt(w)])^2).
    aessf_sigmaS: Optional[float] = None  # A-ESSF on judge marginal σ(S)
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
            "bc_sigmaS": self.bc_sigmaS,
        }
        return {k: v for k, v in d.items() if v is not None}  # Filter None values


def hellinger_affinity(weights: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute the ACTION-SPACE Bhattacharyya coefficient (Hellinger affinity).

    This measures the overlap between two policies over the full action
    space. For importance weights w = p'/p, the affinity is E[√w] under the
    base distribution.

    .. note::
        This is NOT the paper's judge-space overlap gate. The paper gates on
        the binned Bhattacharyya affinity A_B computed between judge-score
        marginals (``bhattacharyya_judge_space``, gate A_B >= 0.85). By the
        data-processing inequality the judge-space A_B is an UPPER bound on
        this action-space value, so the two use different grading ladders.

    Key properties:
    - Value in (0, 1] where 1 indicates perfect overlap
    - Cannot be improved by weight calibration (measures structural mismatch)
    - Related to Hellinger distance: H = √(1 - A)

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


def bhattacharyya_judge_space(
    judge_scores_logged: np.ndarray,
    weights: Optional[np.ndarray] = None,
    judge_scores_target: Optional[np.ndarray] = None,
    n_bins: int = 20,
) -> float:
    """Binned Bhattacharyya affinity A_B in JUDGE-SCORE space.

    This is the paper's overlap gate (arXiv:2512.11150, appendix
    diagnostics): A_B = Σ_b sqrt(p_b(π₀) · p_b(π')) over binned
    judge-score densities, with **gate A_B >= 0.85** for reliable IPS and
    A_B < 0.5 indicating severe distribution mismatch. It is a DIFFERENT
    quantity from ``hellinger_affinity(weights) = E[sqrt(w)]``, which is the
    action-space affinity: by the data-processing inequality the judge-space
    A_B upper-bounds the action-space value, so the two are graded on
    different ladders (see ``cje.diagnostics.gates``).

    Two estimation modes for the π' judge-score marginal:

    1. **Fresh draws available** (preferred): pass ``judge_scores_target``,
       the judge scores of fresh draws sampled from π'.
    2. **Logs only**: pass ``weights`` (RAW importance weights π'/π₀ for the
       logged samples). The π' marginal is estimated as the weight-weighted
       histogram of the logged judge scores.

    Args:
        judge_scores_logged: Judge scores S of logged samples (under π₀)
        weights: Raw importance weights aligned with judge_scores_logged
            (used when judge_scores_target is not provided)
        judge_scores_target: Judge scores of fresh draws from π' (preferred)
        n_bins: Number of histogram bins (default 20)

    Returns:
        A_B ∈ [0, 1]; NaN when it cannot be computed.
    """
    logged = np.asarray(judge_scores_logged, dtype=float)
    valid_logged = np.isfinite(logged)

    if judge_scores_target is not None:
        target = np.asarray(judge_scores_target, dtype=float)
        target = target[np.isfinite(target)]
        logged = logged[valid_logged]
        if len(logged) == 0 or len(target) == 0:
            return float(np.nan)
        lo = min(float(np.min(logged)), float(np.min(target)))
        hi = max(float(np.max(logged)), float(np.max(target)))
        if hi <= lo:
            return 1.0  # All scores identical: distributions coincide
        edges = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(logged, bins=edges)
        q, _ = np.histogram(target, bins=edges)
        p_norm = p / p.sum()
        q_norm = q / q.sum()
    elif weights is not None:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(np.asarray(judge_scores_logged)):
            raise ValueError(
                f"weights length ({len(w)}) does not match judge scores "
                f"length ({len(np.asarray(judge_scores_logged))})"
            )
        w = w[valid_logged]
        logged = logged[valid_logged]
        keep = np.isfinite(w) & (w >= 0)
        logged, w = logged[keep], w[keep]
        if len(logged) == 0 or w.sum() <= 0:
            return float(np.nan)
        lo, hi = float(np.min(logged)), float(np.max(logged))
        if hi <= lo:
            return 1.0
        edges = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(logged, bins=edges)
        q, _ = np.histogram(logged, bins=edges, weights=w)
        p_norm = p / p.sum()
        q_norm = q / q.sum()
    else:
        raise ValueError(
            "bhattacharyya_judge_space requires either judge_scores_target "
            "(fresh draws) or weights (logs-only mode)"
        )

    affinity = float(np.sum(np.sqrt(p_norm * q_norm)))
    return min(max(affinity, 0.0), 1.0)


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
    judge_scores: Optional[np.ndarray] = None,
) -> OverlapMetrics:
    """
    Compute comprehensive overlap diagnostics.

    This function computes complementary metrics:
    1. Hellinger affinity: Action-space structural overlap (cannot be improved)
    2. ESS fraction: Statistical efficiency (can be improved by calibration)
    3. Tail index: Pathological behavior (partially improvable)
    4. When ``judge_scores`` is provided: the paper's judge-space
       Bhattacharyya affinity A_B (``bc_sigmaS``, gate A_B >= 0.85) and the
       σ(S) structural ESS ceiling (``aessf_sigmaS = A_B²``)

    Args:
        weights: Importance weights (will be normalized to mean 1)
        target_ci_halfwidth: Desired CI half-width for auto-tuning
        n_samples: Sample size (defaults to len(weights))
        compute_tail_index: Whether to compute Hill tail index
        auto_tune_threshold: Whether to compute auto-tuned ESS threshold
        judge_scores: Optional judge scores aligned with weights, used to
            populate the σ(S) fields (bc_sigmaS, aessf_sigmaS)

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

    # σ(S) structural floors when judge scores are available
    bc_sigmaS: Optional[float] = None
    aessf_sigmaS: Optional[float] = None
    if judge_scores is not None:
        try:
            bc_value = bhattacharyya_judge_space(judge_scores, weights=weights)
            if np.isfinite(bc_value):
                bc_sigmaS = float(bc_value)
                aessf_sigmaS = float(bc_value**2)
        except ValueError as e:
            logger.debug(f"Could not compute judge-space Bhattacharyya: {e}")

    # Compute efficiency loss (how much data we're wasting)
    efficiency_loss = 1.0 - ess_fraction

    # Confidence interval penalty vs uniform sampling
    # Based on Var ≤ 1/(4n·ESS_frac) for bounded rewards
    if ess_fraction > 0.001:
        confidence_penalty = 1.0 / np.sqrt(ess_fraction)
    else:
        confidence_penalty = np.inf

    # Recommendation engine based on all metrics (canonical ESS ladder)
    from .gates import ESS_GOOD_THRESHOLD, ESS_WARNING_THRESHOLD, TAIL_INDEX_WARNING

    if quality == "catastrophic":
        recommended = "refuse"
    elif (
        tail_index is not None
        and np.isfinite(tail_index)
        and tail_index < TAIL_INDEX_WARNING
    ):
        # Heavy tails (α < 2: infinite-variance risk) - need DR's bias
        # correction rather than logs-only IPS
        recommended = "dr"
    elif ess_fraction < ESS_WARNING_THRESHOLD:
        # Low ESS - depends on overlap quality
        if quality == "poor":
            recommended = "refuse"
        else:
            recommended = "dr"
    elif ess_fraction < ESS_GOOD_THRESHOLD and can_calibrate:
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
        aessf_sigmaS=aessf_sigmaS,
        bc_sigmaS=bc_sigmaS,
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
        if np.isnan(metrics.tail_index):
            msgs.append(
                "⚠️  Tail index could not be estimated (α=n/a)\n"
                "   Weights are too degenerate for the Hill estimator - "
                "treat tail risk as unknown, not light."
            )
        elif metrics.tail_index < 1:
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


# =============================================================================
# TTC and CLE Diagnostics
# =============================================================================


def _target_typical_region(
    base_logprobs: np.ndarray,
    target_logprobs: np.ndarray,
    target_typical_mass: float,
    token_counts: Optional[np.ndarray],
) -> np.ndarray:
    """Boolean mask of logged samples inside the target-typical region T.

    T = {(x,a) : surprisal(a|x) <= q_{target_typical_mass}} where the
    quantile is taken under the TARGET distribution (estimated from the
    logged sample via self-normalized importance weights) and surprisal is
    the mean per-token surprisal -log pi'(a|x) / |a| when a length
    normalizer is available, else total sequence surprisal.
    """
    from scipy.special import logsumexp

    n = len(base_logprobs)

    # Importance weights give the target distribution over logged samples
    log_weights = target_logprobs - base_logprobs
    normalized_log_weights = log_weights - logsumexp(log_weights)
    target_weights = np.exp(normalized_log_weights)

    # Surprisal under the target policy, per-token when lengths are known
    target_surprisal = -target_logprobs
    if token_counts is not None:
        token_counts = np.asarray(token_counts, dtype=float)
        if len(token_counts) != n:
            raise ValueError(
                f"token_counts length ({len(token_counts)}) does not match "
                f"logprobs length ({n})"
            )
        if not np.all(np.isfinite(token_counts)) or np.any(token_counts <= 0):
            raise ValueError("token_counts must be finite and positive")
        target_surprisal = target_surprisal / token_counts

    # Threshold = target-distribution quantile of surprisal at target_typical_mass
    sorted_idx = np.argsort(target_surprisal)
    cumulative = np.cumsum(target_weights[sorted_idx])
    idx = int(np.searchsorted(cumulative, target_typical_mass))
    threshold = target_surprisal[sorted_idx[min(idx, n - 1)]]

    # T = samples with target surprisal <= threshold (where target concentrates)
    result: np.ndarray = target_surprisal <= threshold
    return result


def compute_ttc(
    base_logprobs: np.ndarray,
    target_logprobs: np.ndarray,
    target_typical_mass: float = 0.9,
    token_counts: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Target-Typicality Coverage (TTC).

    From the CJE paper (arXiv:2512.11150, setup):
    > T = {(x,a) : mean per-token surprisal under π' <= q_0.9 of per-token
    > surprisal under π'}, so α = 0.9 by construction. TTC = β̂ = the
    > empirical fraction of LOGGED samples falling in T.
    > Gate: if TTC < 0.7, logs-only IPS will fail; prefer Direct or DR.

    TTC measures how well the logging policy covers the regions where the target
    policy concentrates. It is computed as:

    1. Define T = target-typical region containing target_typical_mass of target
       mass (via a per-token surprisal threshold when lengths are available)
    2. TTC = β = P_π₀(T) = logger mass on T = fraction of logged samples in T

    Low TTC means the logger rarely generates what the target wants,
    setting a hard precision floor that logs-only methods cannot beat.

    .. warning::
        The paper's statistic normalizes surprisal by the TOKEN count |a|.
        The Sample schema stores only sequence-level logprobs, so callers
        should pass ``token_counts`` when available (e.g. a
        ``response_token_count`` metadata field). Callers in the estimation
        pipeline fall back to response length in characters as a length
        normalizer — a documented proxy that approximates the paper's
        statistic (it removes the length confound but with character rather
        than token units). With ``token_counts=None`` the statistic degrades
        to TOTAL sequence surprisal, which confounds typicality with
        response length; the 0.7 gate was calibrated for the per-token
        construction.

    Note: TTC is distinct from Bhattacharyya affinity (hellinger_affinity).
    TTC measures coverage in action space (does logger cover target-typical regions?);
    Bhattacharyya measures shape mismatch of the weight/judge distributions.
    Both are CLE diagnostics but for different components of the bound.

    Args:
        base_logprobs: Log probabilities under logging policy, log π₀(a|x)
        target_logprobs: Log probabilities under target policy, log π'(a|x)
        target_typical_mass: Fraction of target mass to define T.
            Default 0.9 matches the paper (α = 0.9 by construction).
        token_counts: Optional per-sample response lengths (tokens preferred;
            characters acceptable as a proxy) used to compute mean per-token
            surprisal. Without it, total sequence surprisal is used (see
            warning above).

    Returns:
        TTC ∈ (0, 1] where:
        - TTC >= 0.7: Good coverage, IPS may work
        - TTC ∈ [0.3, 0.7): Marginal, consider DR
        - TTC < 0.3: Poor coverage, IPS will fail regardless of ESS
    """
    base_logprobs = np.asarray(base_logprobs)
    target_logprobs = np.asarray(target_logprobs)
    n = len(base_logprobs)

    if n == 0:
        return 0.0

    in_T = _target_typical_region(
        base_logprobs, target_logprobs, target_typical_mass, token_counts
    )

    # TTC = β = logger mass on T (fraction of samples, since samples are from logger)
    return float(np.mean(in_T))


@dataclass
class CLEDiagnostics:
    """
    Coverage-Limited Efficiency (CLE) diagnostics.

    The CLE bound explains why IPS can fail even with high ESS:

        SE(Ψ̂) ≥ (σ_T · α) / √(β·n) · √(1 + χ²_T)

    Where:
    - T = target-typical region (defined via target surprisal threshold)
    - α = P_π'(T) = target mass on T (≈ target_typical_mass by construction)
    - β = P_π₀(T) = logger mass on T = TTC (the key diagnostic!)
    - σ_T = outcome noise in T
    - χ²_T = chi-squared divergence **inside T** (shape mismatch where it matters)

    The bound has three multiplicative factors:
    1. Coverage penalty (α/√β): Explodes when logger rarely visits target-typical regions
    2. Shape mismatch √(1+χ²_T): Inflates floor even with good coverage
    3. Noise term (σ_T/√n): Standard sampling noise

    **Key insight**: TTC = β, not α!
    - TTC measures: "How well does the logger cover where the target concentrates?"
    - Low TTC means logger rarely generates what target wants → IPS fails

    **TTC vs Bhattacharyya**:
    - TTC (β): Coverage in **action space** - logger coverage of target-typical regions
    - Bhattacharyya (E[√w]): Shape mismatch of the **importance weights**
      (action space; the paper's judge-space gate A_B >= 0.85 is the separate
      ``bhattacharyya_judge_space`` metric)

    Attributes:
        ttc: Target-Typicality Coverage = β = logger mass on T
        bhattacharyya: Bhattacharyya coefficient = E[√w] over importance
            weights (ACTION-space affinity; not the paper's judge-space A_B,
            see ``bhattacharyya_judge_space``)
        alpha: Target mass on T (≈ target_typical_mass by construction;
            default 0.9, matching the paper's q_0.9 construction)
        beta: Same as TTC (alias for clarity in CLE bound)
        coverage_penalty: α/√β (the CLE coverage factor)
        chi_squared_T: Var(w) **inside T only** (shape mismatch where it matters)
        shape_mismatch: √(1 + χ²_T)
        cle_factor: Combined CLE inflation = coverage_penalty × shape_mismatch
        ess_fraction: Effective sample size as fraction of n
        n_samples: Number of samples used
    """

    ttc: float  # β = logger mass on T (the key diagnostic!)
    bhattacharyya: float  # E[√w] over importance weights (action space)
    alpha: float  # Target mass on T (≈ target_typical_mass by construction)
    beta: float  # Alias for ttc
    coverage_penalty: float
    chi_squared_T: float  # Computed inside T only
    shape_mismatch: float
    cle_factor: float
    ess_fraction: float
    n_samples: int

    def summary(self) -> str:
        """Human-readable summary."""
        if self.ttc >= 0.7:
            status = "GOOD"
            msg = "IPS may work"
        elif self.ttc >= 0.3:
            status = "MARGINAL"
            msg = "consider DR"
        else:
            status = "POOR"
            msg = "IPS will fail"

        return (
            f"CLE: {status} | TTC={self.ttc:.1%} ({msg}) | "
            f"BC={self.bhattacharyya:.1%} | "
            f"coverage_penalty={self.coverage_penalty:.2f} | "
            f"shape_mismatch={self.shape_mismatch:.1f} | "
            f"cle_factor={self.cle_factor:.1f}x"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ttc": self.ttc,
            "bhattacharyya": self.bhattacharyya,
            "alpha": self.alpha,
            "beta": self.beta,
            "coverage_penalty": self.coverage_penalty,
            "chi_squared_T": self.chi_squared_T,
            "shape_mismatch": self.shape_mismatch,
            "cle_factor": self.cle_factor,
            "ess_fraction": self.ess_fraction,
            "n_samples": self.n_samples,
        }


def compute_cle_diagnostics(
    base_logprobs: np.ndarray,
    target_logprobs: np.ndarray,
    target_typical_mass: float = 0.9,
    token_counts: Optional[np.ndarray] = None,
) -> CLEDiagnostics:
    """
    Compute Coverage-Limited Efficiency (CLE) diagnostics.

    These diagnostics explain why IPS can fail even with high ESS by decomposing
    the efficiency loss into coverage penalty and shape mismatch components.

    The CLE bound is:
        SE(Ψ̂) ≥ (σ_T · α) / √(β·n) · √(1 + χ²_T)

    Where:
    - T = target-typical region (containing target_typical_mass of target mass)
    - α = target mass on T (≈ target_typical_mass by construction;
      default 0.9, matching the paper's q_0.9 construction)
    - β = logger mass on T = TTC (the key diagnostic!)
    - χ²_T = weight variance inside T

    **Key insight**: TTC = β (logger coverage), not α!
    - Low TTC means logger rarely covers where target concentrates → IPS fails

    See ``compute_ttc`` for the per-token-surprisal construction and the
    caveats that apply when ``token_counts`` is unavailable.

    Args:
        base_logprobs: Log probabilities under logging policy, log π₀(a|x)
        target_logprobs: Log probabilities under target policy, log π'(a|x)
        target_typical_mass: Fraction of target mass to define T.
            Default 0.9 matches the paper (α = 0.9 by construction).
        token_counts: Optional per-sample response lengths used to compute
            mean per-token surprisal (see ``compute_ttc``).

    Returns:
        CLEDiagnostics with all CLE bound components

    Example:
        >>> cle = compute_cle_diagnostics(base_lp, target_lp)
        >>> print(cle.summary())
        CLE: POOR | TTC=12.5% (IPS will fail) | BC=8.3% | coverage_penalty=2.31 | ...

        >>> if cle.ttc < 0.7:
        ...     print("Use Direct or DR methods instead of IPS")
    """
    from scipy.special import logsumexp

    base_logprobs = np.asarray(base_logprobs)
    target_logprobs = np.asarray(target_logprobs)
    n = len(base_logprobs)

    if n == 0:
        return CLEDiagnostics(
            ttc=0.0,
            bhattacharyya=0.0,
            alpha=0.0,
            beta=0.0,
            coverage_penalty=np.inf,
            chi_squared_T=np.inf,
            shape_mismatch=np.inf,
            cle_factor=np.inf,
            ess_fraction=0.0,
            n_samples=0,
        )

    # Compute importance weights
    log_weights = target_logprobs - base_logprobs
    weights = np.exp(log_weights - np.max(log_weights))  # Numerical stability
    weights = weights / np.mean(weights)  # Normalize to mean 1

    # --- Bhattacharyya coefficient (surrogate space) ---
    # E[√w] under base distribution
    bhattacharyya = float(np.mean(np.sqrt(np.maximum(weights, 1e-10))))
    bhattacharyya = min(bhattacharyya, 1.0)

    # --- ESS fraction ---
    ess = (np.sum(weights) ** 2) / np.sum(weights**2)
    ess_fraction = ess / n

    # --- Define T = target-typical region ---
    # T contains target_typical_mass of target mass, thresholded on
    # (per-token, when lengths available) target surprisal
    normalized_log_weights = log_weights - logsumexp(log_weights)
    target_weights = np.exp(normalized_log_weights)
    in_T = _target_typical_region(
        base_logprobs, target_logprobs, target_typical_mass, token_counts
    )

    # --- α = target mass on T ---
    alpha = float(np.sum(target_weights[in_T]))

    # --- β = TTC = logger mass on T ---
    # Since samples are from logger, β = fraction of samples in T
    beta = float(np.mean(in_T))
    ttc = beta  # TTC = β = logger coverage of target-typical region

    # --- χ²_T = Var(w) inside T only ---
    if np.sum(in_T) > 0:
        weights_in_T = weights[in_T]
        # Normalize to mean 1 within T
        mean_w_T = np.mean(weights_in_T)
        if mean_w_T > 1e-10:
            mean_one_in_T = weights_in_T / mean_w_T
            chi_squared_T = float(np.var(mean_one_in_T))
        else:
            chi_squared_T = np.inf
    else:
        chi_squared_T = np.inf

    # --- Shape mismatch ---
    shape_mismatch = (
        np.sqrt(1 + chi_squared_T) if np.isfinite(chi_squared_T) else np.inf
    )

    # --- Coverage penalty = α / √β ---
    if beta > 1e-10:
        coverage_penalty = alpha / np.sqrt(beta)
    else:
        coverage_penalty = np.inf

    # --- Combined CLE factor ---
    if np.isfinite(coverage_penalty) and np.isfinite(shape_mismatch):
        cle_factor = coverage_penalty * shape_mismatch
    else:
        cle_factor = np.inf

    return CLEDiagnostics(
        ttc=ttc,
        bhattacharyya=bhattacharyya,
        alpha=alpha,
        beta=beta,
        coverage_penalty=coverage_penalty,
        chi_squared_T=chi_squared_T,
        shape_mismatch=shape_mismatch,
        cle_factor=cle_factor,
        ess_fraction=ess_fraction,
        n_samples=n,
    )
