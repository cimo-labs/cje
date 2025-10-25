"""Transportability diagnostics using simple unbiasedness test.

Tests whether a calibrator trained on base policy transports to target policies
by checking if mean residual E[Y - f̂(S)] = 0 on target policy samples.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Literal, Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransportDiagnostics:
    """Diagnostics for calibrator transportability.

    Attributes:
        status: PASS/WARN/FAIL based on unbiasedness test (0 ∈ CI?)
        delta_hat: Mean residual (Y - f̂(S)) for target policy
        delta_ci: 95% CI for delta_hat (parametric)
        delta_se: Standard error of delta_hat
        decile_residuals: Mean residuals by decile (for visualization)
        decile_counts: Sample counts per decile
        coverage: Fraction of samples in score range
        recommended_action: Next step if WARN/FAIL
        n_probe: Number of target samples
        group_label: Optional label (e.g., "policy:gpt-4-mini")
    """

    status: Literal["PASS", "WARN", "FAIL"]
    delta_hat: float
    delta_ci: tuple[float, float]
    delta_se: float
    decile_residuals: List[float]
    decile_counts: List[int]
    coverage: float
    recommended_action: str
    n_probe: int
    group_label: Optional[str] = None

    def summary(self) -> str:
        """Generate concise summary."""
        lines = []
        lines.append(f"Transport: {self.status}")
        if self.group_label:
            lines.append(f"Group: {self.group_label}")
        lines.append(f"N={self.n_probe}")
        lines.append(
            f"δ̂: {self.delta_hat:+.3f} (CI: [{self.delta_ci[0]:+.3f}, {self.delta_ci[1]:+.3f}])"
        )

        if self.status != "PASS":
            lines.append(f"Action: {self.recommended_action}")

        return " | ".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
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
            "recommended_action": self.recommended_action,
            "n_probe": int(self.n_probe),
            "group_label": self.group_label,
        }


def audit_transportability(
    calibrator: Any,
    probe_samples: List[Any],
    bins: int = 10,
    group_label: Optional[str] = None,
) -> TransportDiagnostics:
    """Test if calibrator transports to target policy.

    Simple unbiasedness test:
    - Compute mean residual δ̂ = E[Y - f̂(S)] for target policy
    - Get 95% CI for δ̂ (parametric: δ̂ ± 1.96*SE)
    - PASS if 0 ∈ CI (unbiased), WARN/FAIL if 0 ∉ CI (biased)

    A calibrator that transports well should have mean residual ≈ 0.

    Args:
        calibrator: Fitted JudgeCalibrator
        probe_samples: Target policy samples with judge_score and oracle_label
        bins: Number of bins for visualization (default 10)
        group_label: Optional label (e.g., "policy:gpt-4-mini")

    Returns:
        TransportDiagnostics with PASS/WARN/FAIL status

    Example:
        >>> from cje.calibration import calibrate_dataset
        >>> from cje.diagnostics.transport import audit_transportability
        >>>
        >>> # Fit calibrator on base policy
        >>> calibrated, result = calibrate_dataset(base_dataset, ...)
        >>> calibrator = result.calibrator
        >>>
        >>> # Test if calibrator transports to target policy
        >>> diag = audit_transportability(
        ...     calibrator,
        ...     probe_samples=target_fresh_draws,
        ...     group_label="policy:gpt-4-mini"
        ... )
        >>> print(diag.summary())
        >>> # Output: "Transport: PASS | N=200 | δ̂: -0.012 (CI: [-0.039, +0.014])"
    """
    from ..data.models import Sample

    # Extract probe data
    probe_scores, probe_labels = _extract_scores_labels(probe_samples)
    S_probe = np.array(probe_scores)
    Y_probe = np.array(probe_labels)
    n_probe = len(Y_probe)

    # Get calibrator predictions
    R_hat_probe = calibrator.predict(S_probe)
    residuals_probe = Y_probe - R_hat_probe

    # Compute target statistics
    delta_hat = float(residuals_probe.mean())
    delta_se = float(residuals_probe.std(ddof=1) / np.sqrt(n_probe))
    delta_ci = (delta_hat - 1.96 * delta_se, delta_hat + 1.96 * delta_se)

    # Bin residuals for visualization
    bin_edges = np.quantile(S_probe, np.linspace(0, 1, bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        bin_edges = np.array([S_probe.min() - 1e-6, S_probe.max() + 1e-6])

    bin_indices = np.digitize(S_probe, bin_edges[1:-1])
    actual_bins = len(bin_edges) - 1

    decile_residuals = []
    decile_counts = []
    for b in range(actual_bins):
        mask = bin_indices == b
        count = int(mask.sum())
        decile_counts.append(count)
        if count > 0:
            decile_residuals.append(float(residuals_probe[mask].mean()))
        else:
            decile_residuals.append(np.nan)

    # Coverage: fraction of bins with >= 3 samples
    coverage = float(sum(1 for c in decile_counts if c >= 3) / max(actual_bins, 1))

    # Classify status
    status, action = _classify_status(
        delta_hat=delta_hat, delta_ci=delta_ci, coverage=coverage
    )

    logger.info(
        f"Transport audit: {status} | δ̂={delta_hat:+.3f} | "
        f"CI=[{delta_ci[0]:+.3f}, {delta_ci[1]:+.3f}] | action={action}"
    )

    return TransportDiagnostics(
        status=status,
        delta_hat=delta_hat,
        delta_ci=delta_ci,
        delta_se=delta_se,
        decile_residuals=decile_residuals,
        decile_counts=decile_counts,
        coverage=coverage,
        recommended_action=action,
        n_probe=n_probe,
        group_label=group_label,
    )


def _extract_scores_labels(samples: List[Any]) -> tuple[List[float], List[float]]:
    """Extract judge scores and oracle labels from samples."""
    from ..data.models import Sample

    scores = []
    labels = []

    for sample in samples:
        if not isinstance(sample, Sample):
            raise TypeError(f"Expected Sample, got {type(sample)}")

        # Get judge score
        judge_score = sample.judge_score
        if judge_score is None and sample.metadata:
            judge_score = sample.metadata.get("judge_score")
        if judge_score is None:
            raise ValueError(f"Sample {sample.prompt_id} missing judge_score")

        # Get oracle label
        if sample.oracle_label is None:
            raise ValueError(f"Sample {sample.prompt_id} missing oracle_label")

        scores.append(judge_score)
        labels.append(sample.oracle_label)

    return scores, labels


def _classify_status(
    delta_hat: float, delta_ci: tuple[float, float], coverage: float
) -> tuple[Literal["PASS", "WARN", "FAIL"], str]:
    """Classify transport status based on unbiasedness test.

    Test: Is 0 ∈ CI for mean residual?
      - PASS: 0 ∈ CI (calibrator is unbiased)
      - WARN: 0 slightly outside CI (marginal bias)
      - FAIL: 0 far outside CI (clear bias)

    Args:
        delta_hat: Mean residual
        delta_ci: 95% confidence interval for mean residual
        coverage: Fraction of bins with sufficient samples

    Returns:
        Tuple of (status, recommended_action)
    """
    # Simple CI test: is mean residual distinguishable from 0?
    if delta_ci[0] <= 0 <= delta_ci[1]:
        return "PASS", "none"

    # Check magnitude of bias
    if abs(delta_hat) < 0.05:
        return "WARN", "monitor"
    else:
        return "FAIL", "refit_two_stage"
