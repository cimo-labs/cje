"""
Display and formatting utilities for diagnostics.

Updated to work with both new diagnostic objects and legacy dictionaries.
"""

from typing import Dict, Any, Union, Optional, TYPE_CHECKING, List
import numpy as np

if TYPE_CHECKING:
    from .models import IPSDiagnostics, DRDiagnostics


def create_weight_summary_table(
    all_diagnostics: Union[Dict[str, Any], "IPSDiagnostics", "DRDiagnostics"],
) -> str:
    """Create a formatted table of weight diagnostics.

    Args:
        all_diagnostics: Either:
            - Dictionary of diagnostic values by policy
            - IPSDiagnostics or DRDiagnostics object

    Returns:
        Formatted table string
    """
    # Import here to avoid circular dependency
    from .models import IPSDiagnostics, DRDiagnostics

    lines = []
    lines.append("\nWeight Summary")
    lines.append("-" * 70)
    lines.append(f"{'Policy':<30} {'ESS':>8} {'Max Weight':>12} {'Status':<10}")
    lines.append("-" * 70)

    # Handle new diagnostic objects
    if isinstance(all_diagnostics, (IPSDiagnostics, DRDiagnostics)):
        diag = all_diagnostics
        for policy in diag.policies:
            ess = diag.ess_per_policy.get(policy, 0.0)
            max_w = diag.max_weight_per_policy.get(policy, 1.0)
            # Use per-policy status if available
            if (
                hasattr(diag, "status_per_policy")
                and diag.status_per_policy
                and policy in diag.status_per_policy
            ):
                status = diag.status_per_policy[policy].value
            else:
                # Fallback: Determine status based on ESS
                if ess > 0.5:
                    status = "GOOD"
                elif ess > 0.2:
                    status = "WARNING"
                else:
                    status = "CRITICAL"

            lines.append(f"{policy:<30} {ess:>7.1%} {max_w:>12.4f} {status:<10}")

        # Add overall summary
        lines.append("-" * 70)
        lines.append(
            f"{'Overall':<30} {diag.weight_ess:>7.1%} "
            f"{max(diag.max_weight_per_policy.values()):>12.4f} "
            f"{diag.weight_status.value:<10}"
        )

    # Handle legacy dictionary format
    elif isinstance(all_diagnostics, dict):
        for policy, diag in all_diagnostics.items():
            # Handle WeightDiagnostics objects or dict format
            if hasattr(diag, "ess_fraction"):  # WeightDiagnostics object
                ess = getattr(diag, "ess_fraction", 0.0)
                max_w = getattr(diag, "max_weight", 1.0)
                # Try both 'status' and 'consistency_flag' for compatibility
                status = getattr(
                    diag, "status", getattr(diag, "consistency_flag", "UNKNOWN")
                )
            elif isinstance(diag, dict):  # Legacy dict format
                ess = diag.get("ess_fraction", 0.0)
                max_w = diag.get("max_weight", 1.0)
                # Try both 'status' and 'consistency_flag' for compatibility
                status_val = diag.get("status", diag.get("consistency_flag", "UNKNOWN"))
                # Handle Status enum if present
                if status_val is not None and hasattr(status_val, "value"):
                    status = status_val.value.upper()
                else:
                    status = str(status_val) if status_val is not None else "UNKNOWN"
            else:
                ess = 0.0
                max_w = 1.0
                status = "UNKNOWN"

            lines.append(f"{policy:<30} {ess:>7.1%} {max_w:>12.4f} {status:<10}")

    return "\n".join(lines)


def format_dr_diagnostic_summary(
    diagnostics: Union[Dict[str, Any], "DRDiagnostics"],
) -> str:
    """Format DR diagnostics as a readable summary table.

    Args:
        diagnostics: Either:
            - Dictionary of DR diagnostics by policy (legacy)
            - DRDiagnostics object (new)

    Returns:
        Formatted summary string
    """
    # Import here to avoid circular dependency
    from .models import DRDiagnostics

    lines = []
    lines.append("=" * 100)
    lines.append("DR DIAGNOSTICS SUMMARY")
    lines.append("=" * 100)

    # Check if we have a DRDiagnostics object
    if isinstance(diagnostics, DRDiagnostics):
        # New diagnostic object
        lines.append(
            f"Estimator: {diagnostics.estimator_type} | "
            f"Method: {diagnostics.method} | "
            f"Cross-fitted: {diagnostics.dr_cross_fitted} ({diagnostics.dr_n_folds} folds)"
        )
        lines.append(
            f"Samples: {diagnostics.n_samples_valid}/{diagnostics.n_samples_total} | "
            f"Weight ESS: {diagnostics.weight_ess:.1%} | "
            f"Status: {diagnostics.overall_status.value}"
        )
        lines.append("-" * 100)

        # Header
        lines.append(
            f"{'Policy':<20} {'Estimate±SE':<20} "
            f"{'DM Mean':>10} {'IPS Corr':>10} {'IF Tail':>10}"
        )
        lines.append("-" * 100)

        # Per-policy rows
        for policy in diagnostics.policies:
            est = diagnostics.estimates.get(policy, 0.0)
            se = diagnostics.standard_errors.get(policy, 0.0)

            # Get detailed diagnostics if available
            policy_diag = diagnostics.get_policy_diagnostics(policy)
            if policy_diag:
                dm = policy_diag.get("dm_mean", 0.0)
                ips_corr = policy_diag.get("ips_corr_mean", 0.0)
                if_tail = policy_diag.get("if_tail_ratio_99_5", 0.0)
            else:
                # Try decomposition if policy diagnostics not available
                if (
                    diagnostics.dm_ips_decompositions
                    and policy in diagnostics.dm_ips_decompositions
                ):
                    decomp = diagnostics.dm_ips_decompositions[policy]
                    dm = decomp.get("dm_component", 0.0)
                    ips_corr = decomp.get("ips_augmentation", 0.0)
                else:
                    dm = 0.0
                    ips_corr = 0.0
                if_tail = 0.0

            lines.append(
                f"{policy:<20} {est:>7.3f}±{se:.3f}  "
                f"{dm:>10.3f} {ips_corr:>10.3f} {if_tail:>10.1f}"
            )

        lines.append("-" * 100)

        # Summary statistics
        min_r2, max_r2 = diagnostics.outcome_r2_range
        lines.append(
            f"Outcome R² range: [{min_r2:.3f}, {max_r2:.3f}] | "
            f"RMSE: {diagnostics.outcome_rmse_mean:.3f} | "
            f"Worst IF tail: {diagnostics.worst_if_tail_ratio:.1f}"
        )

        # Check if influence functions are stored
        if diagnostics.has_influence_functions():
            n_ifs = (
                len(diagnostics.influence_functions)
                if diagnostics.influence_functions
                else 0
            )
            lines.append(f"Influence functions stored for {n_ifs} policies")

    # Handle legacy dictionary format
    elif isinstance(diagnostics, dict):
        # Check if it's the old format with per_policy key
        if "per_policy" in diagnostics:
            diagnostics = diagnostics["per_policy"]

        # Header
        lines.append(
            f"{'Policy':<20} {'DM':>7} {'IPS':>7} {'DR±SE':<20} "
            f"{'Score(mean±se, p)':<25} {'RMSE(R,g)':>10} {'|IF| tail(p99/p5)':>17}"
        )
        lines.append("-" * 100)

        # Per-policy rows
        worst_if_tail = 0.0
        r2_values = []
        max_score_z = 0.0

        if not isinstance(diagnostics, dict):
            return "\n".join(lines)

        for policy, diag in diagnostics.items():
            if isinstance(diag, dict):
                dm = diag.get("dm_mean", 0.0)
                ips = diag.get("ips_corr_mean", 0.0)
                dr = diag.get("dr_estimate", 0.0)

                # Standard error (from influence functions if available)
                if "if_std" in diag and "n_samples" in diag:
                    se = diag["if_std"] / np.sqrt(diag["n_samples"])
                else:
                    se = 0.0

                # Score test (for TMLE)
                score_mean = diag.get("score_mean", 0.0)
                score_se = diag.get("score_se", 0.0)
                score_p = diag.get("score_p", 1.0)
                score_str = f"{score_mean:>7.3f}±{score_se:.3f} (p={score_p:.2f})"

                # Outcome model RMSE
                rmse = diag.get("residual_rmse", np.nan)

                # IF tail ratio
                if_tail = diag.get("if_tail_ratio_99_5", 0.0)

                lines.append(
                    f"{policy:<20} {dm:>7.3f} {ips:>7.3f} {dr:>7.3f}±{se:.3f}  "
                    f"{score_str:<25} {rmse:>10.3f} {if_tail:>17.1f}"
                )

                # Track worst metrics
                worst_if_tail = max(worst_if_tail, if_tail)
                if "r2_oof" in diag and not np.isnan(diag["r2_oof"]):
                    r2_values.append(diag["r2_oof"])
                if "score_z" in diag:
                    max_score_z = max(max_score_z, abs(diag["score_z"]))

        lines.append("-" * 100)

        # Summary statistics
        lines.append(f"Worst IF tail ratio (p99/p5): {worst_if_tail:.1f}")
        if r2_values:
            lines.append(f"R² OOF range: [{min(r2_values):.3f}, {max(r2_values):.3f}]")

        # TMLE-specific
        if isinstance(diagnostics, dict) and "tmle_max_score_z" in diagnostics:
            lines.append(
                f"TMLE max |score z|: {diagnostics['tmle_max_score_z']:.2f} (should be ~0)"
            )
        elif max_score_z > 0:
            lines.append(f"TMLE max |score z|: {max_score_z:.2f} (should be ~0)")

    lines.append("=" * 100)

    # Warnings (works for both formats)
    if isinstance(diagnostics, DRDiagnostics):
        worst_if_tail = diagnostics.worst_if_tail_ratio
    elif isinstance(diagnostics, dict):
        worst_if_tail = (
            max(
                diag.get("if_tail_ratio_99_5", 0.0)
                for diag in diagnostics.values()
                if isinstance(diag, dict)
            )
            if diagnostics
            else 0.0
        )
    else:
        worst_if_tail = 0.0

    if worst_if_tail > 100:
        lines.append("\n⚠️  Warning: Heavy-tailed influence functions detected")
        lines.append("   Consider using more fresh draws or checking policy overlap")

    return "\n".join(lines)


def format_diagnostic_comparison(
    diag1: Union["IPSDiagnostics", "DRDiagnostics"],
    diag2: Union["IPSDiagnostics", "DRDiagnostics"],
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

    # ESS
    lines.append(
        f"{'Weight ESS':<30} {diag1.weight_ess:>20.1%} "
        f"{diag2.weight_ess:>20.1%} "
        f"{100*(diag2.weight_ess - diag1.weight_ess):>+9.1f}%"
    )

    # Calibration (if available)
    if diag1.is_calibrated and diag2.is_calibrated:
        r2_1 = diag1.calibration_r2 if diag1.calibration_r2 is not None else 0.0
        r2_2 = diag2.calibration_r2 if diag2.calibration_r2 is not None else 0.0
        lines.append(
            f"{'Calibration R²':<30} {r2_1:>20.3f} "
            f"{r2_2:>20.3f} "
            f"{r2_2 - r2_1:>+10.3f}"
        )

    # DR-specific (if both are DR)
    from .models import DRDiagnostics

    if isinstance(diag1, DRDiagnostics) and isinstance(diag2, DRDiagnostics):
        lines.append("-" * 80)
        lines.append("DR-specific metrics:")

        # Outcome R²
        min1, max1 = diag1.outcome_r2_range
        min2, max2 = diag2.outcome_r2_range
        lines.append(
            f"{'Outcome R² (min)':<30} {min1:>20.3f} "
            f"{min2:>20.3f} {min2 - min1:>+10.3f}"
        )
        lines.append(
            f"{'Outcome R² (max)':<30} {max1:>20.3f} "
            f"{max2:>20.3f} {max2 - max1:>+10.3f}"
        )

        # IF tail ratio
        lines.append(
            f"{'Worst IF tail ratio':<30} {diag1.worst_if_tail_ratio:>20.1f} "
            f"{diag2.worst_if_tail_ratio:>20.1f} "
            f"{diag2.worst_if_tail_ratio - diag1.worst_if_tail_ratio:>+10.1f}"
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
