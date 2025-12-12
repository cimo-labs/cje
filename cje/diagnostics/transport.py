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

        # Color by sign (green=good, red=bad)
        colors = ["#d62728" if r < 0 else "#2ca02c" for r in residuals[valid_mask]]

        # Plot decile bars
        bars = ax.bar(
            x[valid_mask],
            residuals[valid_mask],
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add overall mean line with CI band
        ax.axhline(
            y=self.delta_hat,
            color="black",
            linewidth=2,
            linestyle="-",
            label=f"Mean δ̂={self.delta_hat:+.3f}",
        )
        ax.axhspan(
            self.delta_ci[0],
            self.delta_ci[1],
            alpha=0.2,
            color="gray",
            label=f"95% CI [{self.delta_ci[0]:+.3f}, {self.delta_ci[1]:+.3f}]",
        )

        # Zero line
        ax.axhline(y=0, color="black", linewidth=1, linestyle="--", alpha=0.5)

        # Labels
        title = f"Transportability: {self.status}"
        if self.group_label:
            title += f" ({self.group_label})"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Score Decile", fontsize=10)
        ax.set_ylabel("Mean Residual (Y - Ŷ)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"D{i+1}\n(n={c})" for i, c in enumerate(counts)], fontsize=8
        )

        # Add status indicator
        status_colors = {"PASS": "green", "WARN": "orange", "FAIL": "red"}
        status_color = status_colors.get(self.status, "gray")
        ax.text(
            0.02,
            0.98,
            self.status,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            color=status_color,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.legend(loc="upper right", fontsize=9)
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
        from matplotlib.patches import Patch
    except ImportError:
        raise ImportError(
            "matplotlib required for plotting. Install with: pip install matplotlib"
        )

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by mean residual
    sorted_items = sorted(diagnostics.items(), key=lambda x: x[1].delta_hat)
    labels = [k for k, _ in sorted_items]
    diags = [v for _, v in sorted_items]

    y_pos = np.arange(len(labels))

    # Extract data
    means = [d.delta_hat for d in diags]
    ci_lowers = [d.delta_ci[0] for d in diags]
    ci_uppers = [d.delta_ci[1] for d in diags]
    statuses = [d.status for d in diags]

    # Calculate error bars
    errors = [
        [m - ci_l for m, ci_l in zip(means, ci_lowers)],
        [ci_u - m for m, ci_u in zip(means, ci_uppers)],
    ]

    # Color by status
    status_colors = {"PASS": "#2ca02c", "WARN": "#ff7f0e", "FAIL": "#d62728"}
    colors = [status_colors.get(s, "gray") for s in statuses]

    # Plot
    ax.barh(
        y_pos,
        means,
        xerr=errors,
        color=colors,
        alpha=0.7,
        capsize=5,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(x=0, color="black", linestyle="--", linewidth=2, alpha=0.7)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{label}\n(n={diags[i].n_probe})" for i, label in enumerate(labels)]
    )
    ax.set_xlabel("Mean Residual δ̂ (Y - Ŷ)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Add status text on bars
    for i, (mean, status) in enumerate(zip(means, statuses)):
        x_offset = max(abs(ci_lowers[i]), abs(ci_uppers[i])) * 0.1
        x_pos = mean + x_offset if mean >= 0 else mean - x_offset
        ha = "left" if mean >= 0 else "right"
        ax.text(x_pos, i, status, va="center", ha=ha, fontsize=9, fontweight="bold")

    # Legend
    legend_elements = [
        Patch(facecolor="#2ca02c", alpha=0.7, label="PASS (CI includes 0)"),
        Patch(facecolor="#ff7f0e", alpha=0.7, label="WARN (small bias)"),
        Patch(facecolor="#d62728", alpha=0.7, label="FAIL (significant bias)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    return fig
