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

    def plot(self, ax: Optional[Any] = None, figsize: tuple = (10, 5)) -> Any:
        """Plot transportability diagnostics.

        Shows decile-level residuals with overall mean and CI.

        Args:
            ax: Optional matplotlib axes. If None, creates new figure.
            figsize: Figure size if creating new figure.

        Returns:
            matplotlib figure object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib required for plotting. Install with: pip install matplotlib"
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        n_bins = len(self.decile_residuals)
        x = np.arange(n_bins)

        # Filter out NaN values for plotting
        residuals = np.array(self.decile_residuals)
        counts = np.array(self.decile_counts)
        valid_mask = ~np.isnan(residuals)

        # Modern color palette (positive=green, negative=red)
        colors = ["#ef4444" if r < 0 else "#10b981" for r in residuals[valid_mask]]

        # Plot decile bars
        ax.bar(
            x[valid_mask],
            residuals[valid_mask],
            color=colors,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )

        # Add overall mean line with CI band
        ax.axhline(
            y=self.delta_hat,
            color="#374151",
            linewidth=2,
            linestyle="-",
        )
        ax.axhspan(
            self.delta_ci[0],
            self.delta_ci[1],
            alpha=0.15,
            color="#6b7280",
        )

        # Zero line
        ax.axhline(y=0, color="#9ca3af", linewidth=1.5, linestyle="--")

        # Labels
        title = f"Transportability: {self.status}"
        if self.group_label:
            title += f" ({self.group_label})"
        ax.set_title(title, fontsize=12, fontweight="bold", color="#111827")
        ax.set_xlabel("Score Decile", fontsize=10, color="#374151")
        ax.set_ylabel("Mean Residual (Y − Ŷ)", fontsize=10, color="#374151")
        ax.set_xticks(x)
        ax.set_xticklabels([f"D{i+1}" for i in range(len(counts))], fontsize=9)

        # Status indicator with modern colors
        status_colors = {"PASS": "#10b981", "WARN": "#f59e0b", "FAIL": "#ef4444"}
        status_color = status_colors.get(self.status, "#6b7280")
        ax.text(
            0.02,
            0.98,
            self.status,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color=status_color,
            verticalalignment="top",
        )

        # Stats text
        ax.text(
            0.98,
            0.98,
            f"δ̂={self.delta_hat:+.3f}  CI=[{self.delta_ci[0]:+.2f}, {self.delta_ci[1]:+.2f}]",
            transform=ax.transAxes,
            fontsize=9,
            color="#6b7280",
            verticalalignment="top",
            horizontalalignment="right",
        )

        # Clean up spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        return fig


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


def plot_transport_comparison(
    diagnostics: Dict[str, TransportDiagnostics],
    figsize: tuple = (10, 6),
    title: str = "Transportability Audit",
) -> Any:
    """Plot multiple transport diagnostics as a forest plot.

    Args:
        diagnostics: Dict mapping labels to TransportDiagnostics objects
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib figure object

    Example:
        >>> diag_clone = audit_transportability(calibrator, clone_probe, group_label="clone")
        >>> diag_unhelpful = audit_transportability(calibrator, unhelpful_probe, group_label="unhelpful")
        >>> fig = plot_transport_comparison({"clone": diag_clone, "unhelpful": diag_unhelpful})
        >>> plt.show()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib required for plotting. Install with: pip install matplotlib"
        )

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by mean residual (most negative at bottom)
    sorted_items = sorted(
        diagnostics.items(), key=lambda x: x[1].delta_hat, reverse=True
    )
    labels = [k for k, _ in sorted_items]
    diags = [v for _, v in sorted_items]

    y_pos = list(range(len(labels)))

    # Extract data
    means = [d.delta_hat for d in diags]
    ci_lowers = [d.delta_ci[0] for d in diags]
    ci_uppers = [d.delta_ci[1] for d in diags]
    statuses = [d.status for d in diags]

    # Modern color palette
    status_colors = {"PASS": "#10b981", "WARN": "#f59e0b", "FAIL": "#ef4444"}

    # Plot CI lines and point estimates
    for i, (y, diag) in enumerate(zip(y_pos, diags)):
        color = status_colors.get(diag.status, "#6b7280")

        # CI line
        ax.plot(
            [ci_lowers[i], ci_uppers[i]],
            [y, y],
            color=color,
            linewidth=2.5,
            solid_capstyle="round",
        )

        # Point estimate
        ax.scatter(
            [means[i]],
            [y],
            color=color,
            s=80,
            zorder=5,
            edgecolors="white",
            linewidth=1.5,
        )

    # Zero reference line
    ax.axvline(x=0, color="#9ca3af", linestyle="--", linewidth=1.5, zorder=0)

    # Y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)

    # X-axis
    ax.set_xlabel("Calibration Error (Y − Ŷ)", fontsize=11, color="#374151")

    # Status labels on right (outside axes)
    if ci_lowers and ci_uppers:
        x_max = max(abs(min(ci_lowers)), abs(max(ci_uppers))) * 1.15
    else:
        x_max = 0.1
    for i, (y, status) in enumerate(zip(y_pos, statuses)):
        color = status_colors.get(status, "#6b7280")
        ax.text(
            x_max,
            y,
            status,
            ha="left",
            va="center",
            fontsize=10,
            color=color,
            fontweight="bold",
            clip_on=False,
        )

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    # Set x limits with padding for status labels
    x_min = min(ci_lowers) - abs(min(ci_lowers)) * 0.1
    ax.set_xlim(x_min, x_max + x_max * 0.3)

    plt.tight_layout()
    return fig
