"""Transportability diagnostics using simple unbiasedness test.

Tests whether a calibrator trained on base policy transports to target policies
by checking if mean residual E[Y - f̂(S)] = 0 on target policy samples.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Literal, Optional, Any, Dict
import logging

from scipy import stats

from .gates import TRANSPORT_FAIL_DELTA_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class TransportDiagnostics:
    """Diagnostics for calibrator transportability.

    Attributes:
        status: PASS/WARN/FAIL based on unbiasedness test (0 ∈ CI?)
        delta_hat: Mean residual (Y - f̂(S)) for target policy
        delta_ci: (1 - alpha) CI for delta_hat (parametric, t-based)
        delta_se: Standard error of delta_hat
        decile_residuals: Mean residuals by decile — DISPLAY-ONLY. At the
            recommended probe sizes (40-60 rows) each bin holds 4-6 samples;
            never gate on per-decile values, only on the pooled CI.
        decile_counts: Sample counts per decile
        coverage: Fraction of samples in score range
        recommended_action: Next step if WARN/FAIL
        n_probe: Number of target samples
        group_label: Optional label (e.g., "policy:gpt-5.6-mini")
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

    def plot(self, ax: Optional[Any] = None, figsize: tuple = (10, 5)) -> Any:
        """Plot transportability diagnostics (requires the viz extra).

        Shows decile-level residuals with overall mean and CI. The plotting
        code lives in cje.visualization.transport; this method delegates.

        Args:
            ax: Optional matplotlib axes. If None, creates new figure.
            figsize: Figure size if creating new figure.

        Returns:
            matplotlib figure object
        """
        from ..visualization.transport import plot_transport_diagnostics

        return plot_transport_diagnostics(self, ax=ax, figsize=figsize)


def audit_transportability(
    calibrator: Any,
    probe_samples: List[Any],
    bins: int = 10,
    group_label: Optional[str] = None,
    alpha: float = 0.05,
) -> TransportDiagnostics:
    """Test if calibrator transports to target policy.

    Simple unbiasedness test:
    - Compute mean residual δ̂ = E[Y - f̂(S)] for target policy
    - Get the (1 - α) CI for δ̂ (parametric: δ̂ ± t_{1-α/2, n-1}·SE).
      t critical values matter here: the recommended probe slices are
      small (40-60 rows), where the z interval under-covers and inflates
      the audit's false-alarm rate.
    - PASS if 0 ∈ CI (unbiased), WARN/FAIL if 0 ∉ CI (biased)

    A calibrator that transports well should have mean residual ≈ 0.

    Notes:
        - The returned `decile_residuals` are DISPLAY-ONLY diagnostics:
          at recommended probe sizes each bin holds 4-6 rows. Gate on the
          pooled CI, never on per-decile values.
        - Auditing K policies at per-test α inflates the family-wise
          false-alarm rate (a Benjamini-Hochberg correction is a planned
          option, not implemented).

    Args:
        calibrator: Fitted JudgeCalibrator
        probe_samples: Target policy samples with judge_score and oracle_label
        bins: Number of bins for visualization (default 10)
        group_label: Optional label (e.g., "policy:gpt-5.6-mini")
        alpha: Significance level for the CI (default 0.05 → 95% CI)

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
        ...     group_label="policy:gpt-5.6-mini"
        ... )
        >>> print(diag.summary())
        >>> # Output: "Transport: PASS | N=200 | δ̂: -0.012 (CI: [-0.039, +0.014])"
    """

    # Extract probe data
    probe_scores, probe_labels = _extract_scores_labels(probe_samples)
    S_probe = np.array(probe_scores)
    Y_probe = np.array(probe_labels)
    n_probe = len(Y_probe)

    # Get calibrator predictions
    R_hat_probe = calibrator.predict(S_probe)
    residuals_probe = Y_probe - R_hat_probe

    # Compute target statistics. t critical values (df = n_probe - 1), not
    # z: probe slices are small (40-60 rows recommended), where 1.96 SEs
    # under-cover and inflate the false-alarm rate to ~6-7%.
    delta_hat = float(residuals_probe.mean())
    delta_se = float(residuals_probe.std(ddof=1) / np.sqrt(n_probe))
    df = max(n_probe - 1, 1)
    t_crit = float(stats.t.ppf(1 - alpha / 2, df))
    delta_ci = (delta_hat - t_crit * delta_se, delta_hat + t_crit * delta_se)

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
    """Extract judge scores and oracle labels from samples.

    Accepts either:
    - List[Sample]: CJE Sample objects
    - List[dict]: Dicts with 'judge_score' and 'oracle_label' keys
    """
    from ..data.models import Sample

    scores = []
    labels = []

    for i, sample in enumerate(samples):
        # Handle dict input (from fresh draws JSONLs, DataFrames, etc.)
        if isinstance(sample, dict):
            judge_score = sample.get("judge_score")
            oracle_label = sample.get("oracle_label")
            sample_id = sample.get("prompt_id", f"sample_{i}")
        # Handle Sample objects
        elif isinstance(sample, Sample):
            judge_score = sample.judge_score
            if judge_score is None and sample.metadata:
                judge_score = sample.metadata.get("judge_score")
            oracle_label = sample.oracle_label
            sample_id = sample.prompt_id
        else:
            raise TypeError(f"Expected dict or Sample, got {type(sample)}")

        # Validate
        if judge_score is None:
            raise ValueError(f"Sample {sample_id} missing judge_score")
        if oracle_label is None:
            raise ValueError(f"Sample {sample_id} missing oracle_label")

        scores.append(float(judge_score))
        labels.append(float(oracle_label))

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
        delta_ci: (1 - alpha) confidence interval for mean residual
        coverage: Fraction of bins with sufficient samples

    Returns:
        Tuple of (status, recommended_action)
    """
    # Simple CI test: is mean residual distinguishable from 0?
    if delta_ci[0] <= 0 <= delta_ci[1]:
        return "PASS", "none"

    # Check magnitude of bias (WARN/FAIL split canonical in gates.py)
    if abs(delta_hat) < TRANSPORT_FAIL_DELTA_THRESHOLD:
        return "WARN", "monitor"
    else:
        return "FAIL", "refit_two_stage"


def compute_residuals(
    calibrator: Any,
    data: List[Dict[str, Any]],
    sort_by: Optional[Literal["residual", "abs_residual"]] = "residual",
) -> List[Dict[str, Any]]:
    """Compute residuals for each sample, optionally sorted.

    Useful for inspecting which samples have the worst calibration errors.

    Args:
        calibrator: Fitted calibrator with .predict() method
        data: List of dicts with 'judge_score' and 'oracle_label' keys
        sort_by: How to sort results:
            - "residual": worst overestimates first (most negative)
            - "abs_residual": biggest errors first
            - None: preserve original order

    Returns:
        List of dicts with 'calibrated' and 'residual' fields added.
        Original dict fields are preserved.

    Example:
        >>> from cje.diagnostics import compute_residuals
        >>> samples = compute_residuals(calibrator, probe_data)
        >>> # Inspect worst overestimates (judge fooled)
        >>> for s in samples[:3]:
        ...     print(f"Residual: {s['residual']:.2f}")
        ...     print(f"Response: {s['response'][:100]}...")
    """
    results = []

    for sample in data:
        # Validate required fields
        if "judge_score" not in sample:
            raise ValueError("Sample missing 'judge_score' field")
        if "oracle_label" not in sample:
            raise ValueError("Sample missing 'oracle_label' field")

        # Compute calibrated prediction and residual
        judge_score = float(sample["judge_score"])
        oracle_label = float(sample["oracle_label"])
        calibrated = float(calibrator.predict([[judge_score]])[0])
        residual = oracle_label - calibrated

        # Copy original dict and add new fields
        enriched = dict(sample)
        enriched["calibrated"] = calibrated
        enriched["residual"] = residual

        results.append(enriched)

    # Sort if requested
    if sort_by == "residual":
        results.sort(key=lambda x: x["residual"])
    elif sort_by == "abs_residual":
        results.sort(key=lambda x: abs(x["residual"]), reverse=True)

    return results
