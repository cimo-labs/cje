"""
Display and formatting utilities for diagnostics.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import DirectDiagnostics


def format_diagnostic_comparison(
    diag1: "DirectDiagnostics",
    diag2: "DirectDiagnostics",
    label1: str = "Run 1",
    label2: str = "Run 2",
) -> str:
    """Compare two diagnostic objects side by side.

    Args:
        diag1: First diagnostic object
        diag2: Second diagnostic object
        label1: Label for first run
        label2: Label for second run

    Returns:
        Formatted comparison table
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"DIAGNOSTIC COMPARISON: {label1} vs {label2}")
    lines.append("=" * 80)

    # Basic info
    lines.append(f"{'Metric':<30} {label1:>20} {label2:>20} {'Δ':>10}")
    lines.append("-" * 80)

    # Sample counts
    lines.append(
        f"{'Total samples':<30} {diag1.n_samples_total:>20d} "
        f"{diag2.n_samples_total:>20d} "
        f"{diag2.n_samples_total - diag1.n_samples_total:>+10d}"
    )
    lines.append(
        f"{'Valid samples':<30} {diag1.n_samples_valid:>20d} "
        f"{diag2.n_samples_valid:>20d} "
        f"{diag2.n_samples_valid - diag1.n_samples_valid:>+10d}"
    )

    # Calibration (if available)
    if diag1.is_calibrated and diag2.is_calibrated:
        rmse_1 = diag1.calibration_rmse if diag1.calibration_rmse is not None else 0.0
        rmse_2 = diag2.calibration_rmse if diag2.calibration_rmse is not None else 0.0
        lines.append(
            f"{'Calibration RMSE':<30} {rmse_1:>20.3f} "
            f"{rmse_2:>20.3f} "
            f"{rmse_2 - rmse_1:>+10.3f}"
        )

    # Per-policy comparison
    lines.append("-" * 80)
    lines.append("Per-policy estimates:")

    common_policies = set(diag1.policies) & set(diag2.policies)
    for policy in sorted(common_policies):
        est1 = diag1.estimates.get(policy, 0.0)
        est2 = diag2.estimates.get(policy, 0.0)
        se1 = diag1.standard_errors.get(policy, 0.0)
        se2 = diag2.standard_errors.get(policy, 0.0)

        lines.append(
            f"{policy:<30} {est1:>8.3f}±{se1:.3f} "
            f"{est2:>8.3f}±{se2:.3f} "
            f"{est2 - est1:>+10.3f}"
        )

    lines.append("=" * 80)

    return "\n".join(lines)
