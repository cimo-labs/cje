"""Transportability diagnostics for calibrator reuse across policies and time.

This module implements Diagnostic 5 from the CJE playbook: testing whether a
calibrator fitted on one policy/era can safely transport to another. The probe
protocol uses 40-60 labeled samples to detect mean shifts, regional miscalibration,
and coverage failures.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransportDiagnostics:
    """Diagnostics for calibrator transportability across policies/eras.

    Attributes:
        status: Transport test result (PASS/WARN/FAIL)
        delta_hat: Global mean residual shift (probe mean(Y - f(S)))
        delta_ci: 95% CI for delta_hat (tuple of lower, upper)
        delta_se: Standard error of delta_hat
        decile_residuals: Mean residuals by risk-index decile
        decile_counts: Sample counts per decile
        coverage: Fraction of probe with U in calibrator's training range
        ks_statistic: Kolmogorov-Smirnov test statistic for U distribution
        boundary_slopes: (left_slope, right_slope) at calibrator boundaries
        recommended_action: Next step based on failure mode
        n_probe: Number of probe samples used
        group_label: Optional label for target group (e.g., "policy:gpt-4-mini")
    """

    status: Literal["PASS", "WARN", "FAIL"]
    delta_hat: float
    delta_ci: tuple[float, float]
    delta_se: float
    decile_residuals: List[float]
    decile_counts: List[int]
    coverage: float
    ks_statistic: float
    boundary_slopes: Optional[tuple[float, float]]
    recommended_action: str
    n_probe: int
    group_label: Optional[str] = None

    def summary(self) -> str:
        """Generate concise summary."""
        lines = []
        lines.append(f"Transport Diagnostic: {self.status}")
        if self.group_label:
            lines.append(f"Group: {self.group_label}")
        lines.append(f"N={self.n_probe}")
        lines.append(
            f"Mean shift: {self.delta_hat:+.3f} (95% CI: [{self.delta_ci[0]:+.3f}, {self.delta_ci[1]:+.3f}])"
        )
        lines.append(f"Coverage: {self.coverage:.1%}")
        lines.append(f"Action: {self.recommended_action}")

        # Report worst decile residual
        if self.decile_residuals:
            max_abs_residual = max(
                abs(r) for r in self.decile_residuals if not np.isnan(r)
            )
            lines.append(f"Worst decile residual: {max_abs_residual:.3f}")

        return " | ".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for serialization."""
        return {
            "status": self.status,
            "delta_hat": float(self.delta_hat),
            "delta_ci": [float(self.delta_ci[0]), float(self.delta_ci[1])],
            "delta_se": float(self.delta_se),
            "decile_residuals": [
                float(r) if not np.isnan(r) else None for r in self.decile_residuals
            ],
            "decile_counts": [int(c) for c in self.decile_counts],
            "coverage": float(self.coverage),
            "ks_statistic": float(self.ks_statistic),
            "boundary_slopes": (
                [float(self.boundary_slopes[0]), float(self.boundary_slopes[1])]
                if self.boundary_slopes
                else None
            ),
            "recommended_action": self.recommended_action,
            "n_probe": int(self.n_probe),
            "group_label": self.group_label,
        }


def audit_transportability(
    calibrator: Any,  # JudgeCalibrator instance
    probe_samples: List[Any],  # List of Sample objects with oracle_label
    bins: int = 10,
    group_label: Optional[str] = None,
) -> TransportDiagnostics:
    """Test if calibrator transports to probe samples.

    Implements the cheap probe protocol from playbook §4 Diagnostic 5:
    1. Compute global residual mean δ̂ and 95% CI
    2. Check regional residuals by risk-index deciles
    3. Verify coverage of probe U within calibrator's training range

    Args:
        calibrator: Fitted JudgeCalibrator instance
        probe_samples: List of Sample objects with judge_score (top-level field)
            and oracle_label set. For data loaded via DatasetLoader, judge_score
            is automatically promoted to top-level.
        bins: Number of deciles for regional check (default 10)
        group_label: Optional label for target group (e.g., "policy:gpt-4-mini")

    Returns:
        TransportDiagnostics with PASS/WARN/FAIL status and recommended action

    Example:
        >>> from cje.calibration import fit_judge_calibrator
        >>> from cje.diagnostics.transport import audit_transportability
        >>> from cje.data import load_samples
        >>>
        >>> # Fit calibrator on source policy
        >>> calibrator = fit_judge_calibrator(source_samples, mode="auto")
        >>>
        >>> # Test transport to new policy with 50-sample probe
        >>> probe = load_samples("probes/gpt4_mini_probe.jsonl")
        >>> diag = audit_transportability(
        ...     calibrator,
        ...     probe,
        ...     group_label="policy:gpt-4-mini"
        ... )
        >>>
        >>> print(diag.summary())
        >>> if diag.status == "FAIL":
        ...     print(f"Action: {diag.recommended_action}")
    """
    from ..data.models import Sample

    # Extract judge scores and oracle labels from probe samples
    probe_scores = []
    probe_labels = []
    probe_covariates = []

    for sample in probe_samples:
        if not isinstance(sample, Sample):
            raise TypeError(f"Expected Sample object, got {type(sample)}")

        # Get judge score from metadata or top level
        judge_score = sample.metadata.get("judge_score") if sample.metadata else None
        if judge_score is None:
            # Try top-level judge_score field (promoted by DatasetLoader)
            judge_score = sample.judge_score
        if judge_score is None:
            raise ValueError(
                f"Sample {sample.prompt_id} missing judge_score "
                f"(checked both metadata and top-level field)"
            )

        if sample.oracle_label is None:
            raise ValueError(f"Sample {sample.prompt_id} missing oracle_label")

        probe_scores.append(judge_score)
        probe_labels.append(sample.oracle_label)

        # Extract covariates if calibrator uses them
        if calibrator.covariate_names:
            cov_values = []
            for cov_name in calibrator.covariate_names:
                cov_val = sample.metadata.get(cov_name)
                if cov_val is None:
                    raise ValueError(
                        f"Sample {sample.prompt_id} missing covariate '{cov_name}' "
                        f"required by calibrator"
                    )
                cov_values.append(cov_val)
            probe_covariates.append(cov_values)

    S = np.array(probe_scores)
    Y = np.array(probe_labels)
    n_probe = len(Y)

    # Prepare covariates if needed
    X_cov = np.array(probe_covariates) if probe_covariates else None

    if n_probe < 30:
        logger.warning(
            f"Small probe size (n={n_probe}). Recommend at least 40-60 samples."
        )

    # Step 1: Get calibrator predictions
    R_hat = calibrator.predict(S, covariates=X_cov)

    # Compute residuals
    residuals = Y - R_hat

    # Step 2: Global mean shift
    delta_hat = float(np.mean(residuals))
    delta_se = float(np.std(residuals, ddof=1) / np.sqrt(n_probe))
    delta_ci = (delta_hat - 1.96 * delta_se, delta_hat + 1.96 * delta_se)

    # Step 3: Get risk index U (uniformized)
    # For two-stage calibrators, get the transformed index; for monotone, use scores
    try:
        if (
            hasattr(calibrator, "_flexible_calibrator")
            and calibrator._flexible_calibrator is not None
        ):
            # Two-stage: get uniformized risk index
            U = calibrator._flexible_calibrator.index(S, folds=None, covariates=X_cov)
            # Normalize to [0,1] if not already
            U_min, U_max = U.min(), U.max()
            if U_max > U_min:
                U = (U - U_min) / (U_max - U_min)
        else:
            # Monotone: use score directly, normalize to [0,1]
            S_min, S_max = S.min(), S.max()
            if S_max > S_min:
                U = (S - S_min) / (S_max - S_min)
            else:
                U = np.zeros_like(S)
    except Exception as e:
        logger.warning(f"Failed to compute risk index U: {e}. Using raw scores.")
        S_min, S_max = S.min(), S.max()
        if S_max > S_min:
            U = (S - S_min) / (S_max - S_min)
        else:
            U = np.zeros_like(S)

    # Bin by U deciles
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_indices = np.digitize(U, bin_edges[1:-1])  # Returns 0 to bins-1

    decile_residuals = []
    decile_counts = []

    for b in range(bins):
        mask = bin_indices == b
        count = int(np.sum(mask))
        decile_counts.append(count)

        if count > 0:
            mean_resid = float(np.mean(residuals[mask]))
            decile_residuals.append(mean_resid)
        else:
            decile_residuals.append(np.nan)

    # Step 4: Coverage check
    # Assume calibrator was trained on U in [0, 1] range; check what fraction of probe is in range
    # For a more precise check, would need calibrator to store training U range
    # For now, use a simple heuristic: fraction of probe U in [0.05, 0.95]
    coverage = float(np.mean((U >= 0.05) & (U <= 0.95)))

    # Step 5: KS test (compare probe U to uniform[0,1])
    from scipy.stats import ks_2samp, kstest

    # Use one-sample KS test against uniform
    ks_result = kstest(U, "uniform")
    ks_statistic = float(ks_result.statistic)

    # Step 6: Boundary slopes
    # For simplicity, leave as None (would need calibrator metadata)
    boundary_slopes = None

    # Step 7: Classify status and recommend action
    status, action = _classify_transport_status(
        delta_ci=delta_ci,
        decile_residuals=decile_residuals,
        decile_counts=decile_counts,
        coverage=coverage,
        n_probe=n_probe,
    )

    logger.info(
        f"Transport audit: {status} | δ={delta_hat:+.3f} [{delta_ci[0]:+.3f}, {delta_ci[1]:+.3f}] | "
        f"coverage={coverage:.1%} | action={action}"
    )

    return TransportDiagnostics(
        status=status,
        delta_hat=delta_hat,
        delta_ci=delta_ci,
        delta_se=delta_se,
        decile_residuals=decile_residuals,
        decile_counts=decile_counts,
        coverage=coverage,
        ks_statistic=ks_statistic,
        boundary_slopes=boundary_slopes,
        recommended_action=action,
        n_probe=n_probe,
        group_label=group_label,
    )


def _classify_transport_status(
    delta_ci: tuple[float, float],
    decile_residuals: List[float],
    decile_counts: List[int],
    coverage: float,
    n_probe: int,
) -> tuple[Literal["PASS", "WARN", "FAIL"], str]:
    """Classify transport status using playbook thresholds.

    Traffic-light logic from §4 Diagnostic 5:
    - PASS: 0 ∈ CI(δ̂) AND all decile |mean_resid| ≤ 0.05 AND coverage ≥ 95%
    - WARN: |δ̂| ∈ [0.02, 0.05] OR 1-2 bins > 0.05 OR coverage ∈ [85%, 95%)
    - FAIL: 0 ∉ CI(δ̂) OR 3+ bins > 0.05 OR coverage < 85%

    Returns:
        (status, recommended_action)
    """
    # Check global shift
    zero_in_ci = delta_ci[0] <= 0 <= delta_ci[1]
    abs_delta = abs((delta_ci[0] + delta_ci[1]) / 2)  # Midpoint as proxy

    # Check regional residuals (ignore NaN from empty bins)
    valid_residuals = [
        r
        for r, count in zip(decile_residuals, decile_counts)
        if count > 0 and not np.isnan(r)
    ]

    if valid_residuals:
        max_abs_residual = max(abs(r) for r in valid_residuals)
        n_bins_exceed_005 = sum(1 for r in valid_residuals if abs(r) > 0.05)
    else:
        max_abs_residual = 0.0
        n_bins_exceed_005 = 0

    # Check for monotone regional pattern (sign changes could indicate U-shaped miscalibration)
    has_pattern = (
        _detect_regional_pattern(valid_residuals)
        if len(valid_residuals) >= 5
        else False
    )

    # Classify
    if not zero_in_ci:
        # Uniform mean shift detected
        if n_bins_exceed_005 == 0 and coverage >= 0.85:
            # Pure mean shift, no regional pattern
            return "WARN", "mean_anchor"
        else:
            # Mean shift + regional issues
            return "FAIL", "refit_two_stage"

    # CI contains zero - check regional and coverage
    if n_bins_exceed_005 >= 3 or coverage < 0.85 or has_pattern:
        # Significant regional miscalibration or poor coverage
        if has_pattern:
            return "FAIL", "refit_two_stage"
        elif coverage < 0.85:
            return "FAIL", "add_labels_boundary"
        else:
            thin_deciles = [
                i for i, c in enumerate(decile_counts) if c < max(3, n_probe / 20)
            ]
            if thin_deciles:
                thin_str = ",".join(str(i) for i in thin_deciles[:3])
                return "FAIL", f"collect_more_in_deciles_{thin_str}"
            else:
                return "FAIL", "refit_two_stage"

    elif (n_bins_exceed_005 in [1, 2]) or (coverage < 0.95) or (abs_delta > 0.02):
        # Marginal issues - warn
        if coverage < 0.95:
            return "WARN", "add_labels_boundary"
        else:
            thin_deciles = [
                i for i, c in enumerate(decile_counts) if c < max(3, n_probe / 20)
            ]
            if thin_deciles:
                thin_str = ",".join(str(i) for i in thin_deciles[:3])
                return "WARN", f"collect_more_in_deciles_{thin_str}"
            else:
                return "WARN", "monitor"

    else:
        # All checks pass
        return "PASS", "none"


def _detect_regional_pattern(residuals: List[float]) -> bool:
    """Detect non-random regional pattern (e.g., U-shaped).

    Returns True if residuals show systematic regional pattern.
    """
    if len(residuals) < 5:
        return False

    # Simple heuristic: check for U-shape (both ends high, middle low)
    # or inverted U (both ends low, middle high)
    n = len(residuals)
    left_third = residuals[: n // 3]
    middle_third = residuals[n // 3 : 2 * n // 3]
    right_third = residuals[2 * n // 3 :]

    if not left_third or not middle_third or not right_third:
        return False

    left_mean = np.mean(left_third)
    middle_mean = np.mean(middle_third)
    right_mean = np.mean(right_third)

    # U-shape: left and right similar sign and magnitude, middle opposite
    u_shaped = (
        abs(left_mean - right_mean) < 0.03  # Ends similar
        and abs(left_mean) > 0.04  # Ends non-trivial
        and abs(middle_mean - left_mean) > 0.05  # Middle diverges
    )

    # Monotone trend: consistent increase or decrease
    monotone_increasing = all(
        residuals[i] <= residuals[i + 1] + 0.02 for i in range(len(residuals) - 1)
    )
    monotone_decreasing = all(
        residuals[i] >= residuals[i + 1] - 0.02 for i in range(len(residuals) - 1)
    )

    return bool(u_shaped or monotone_increasing or monotone_decreasing)
