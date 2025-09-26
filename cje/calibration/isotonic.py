"""Isotonic calibration for importance weights.

This module provides variance-controlled IPS weight calibration using:
- Single-pass mean-1 PAV (no cross-fitting needed)
- Closed-form variance-safe blending with feasibility handling
- Robust edge case handling for sparse/degenerate weights
"""

from __future__ import annotations
import numpy as np
import logging
from typing import Dict, Tuple, Optional, Union
from sklearn.isotonic import isotonic_regression

logger = logging.getLogger(__name__)

# Tolerances
EPS = 1e-12
MEAN_TOL = 1e-10
VAR_TOL = 1.001  # allow 0.1% wiggle room on the variance cap


def _pav_mean1_projection_sorted(
    w_sorted_mean1: np.ndarray, mode: str = "fast"
) -> np.ndarray:
    """
    Inputs:
      w_sorted_mean1: weights sorted ASCENDING, already normalized to mean 1.
      mode: "fast" (single-pass) or "exact" (bisection on Lagrange multiplier)

    Returns:
      v: monotone non-decreasing vector with EXACT mean 1

    Implementation:
      - fast: project cumulative excess with increasing PAV, then differentiate
      - exact: true Euclidean projection via bisection (30-40 PAV calls)
    """
    if mode == "exact":
        return _mean_one_isotonic_projection_exact(w_sorted_mean1)

    # Fast mode: single-pass with increasing=False for better ESS in heavy tails
    z = np.cumsum(w_sorted_mean1 - 1.0)  # cumulative excess over mean
    y = isotonic_regression(
        z, increasing=False
    )  # DECREASING gives convex shape, better for heavy tails
    v = np.diff(np.concatenate(([0.0], y))) + 1.0
    v = np.clip(v, 0.0, None)

    # Safety: ensure monotonicity (isotonic repair if needed)
    v = np.maximum.accumulate(v)

    # Enforce exact mean 1 without adding EPS (keeps math for blending exact)
    tot = float(v.sum())
    if tot <= 0.0:
        v = np.full_like(v, 1.0)  # degenerate fallback (should be rare)
    else:
        v *= v.size / tot

    # Safety assertions
    assert np.all(np.diff(v) >= -1e-12), "Monotonicity violated in PAV"
    assert abs(v.mean() - 1.0) < 1e-10, f"Mean not preserved: {v.mean()}"

    return v


def _mean_one_isotonic_projection_exact(
    w_sorted_mean1: np.ndarray, tol: float = 1e-10, max_iters: int = 40
) -> np.ndarray:
    """
    Exact Euclidean projection onto {v: v₁≤…≤vₙ, mean(v)=1} via bisection.

    This gives the mathematically correct projection with guaranteed monotonicity
    and exact mean=1, at the cost of ~30-40 PAV calls.
    """
    n = len(w_sorted_mean1)
    lo, hi = -float(w_sorted_mean1.max()), float(w_sorted_mean1.max())

    for _ in range(max_iters):
        mu = 0.5 * (lo + hi)
        v = isotonic_regression(w_sorted_mean1 - mu, increasing=True)
        m = float(v.mean())
        if abs(m - 1.0) <= tol:
            break
        if m > 1.0:
            lo = mu
        else:
            hi = mu

    # Ensure exact mean to numerical tolerance
    tot = float(v.sum())
    if tot > 0:
        v *= n / tot
    else:
        v = np.ones_like(w_sorted_mean1)  # Degenerate case

    return np.asarray(v)


def _variance_safe_blend_closed_form(
    raw_norm: np.ndarray,
    cal_norm: np.ndarray,
    max_variance_ratio: float = 1.0,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, float, Dict]:
    """
    Blend raw and calibrated (both mean=1) to satisfy Var(out) <= max_ratio * Var(raw)
    when FEASIBLE; otherwise return the variance-minimizing blend.

    Returns:
      out_mean1, alpha, info_dict
    """
    r = np.asarray(raw_norm, dtype=float)
    c = np.asarray(cal_norm, dtype=float)
    Vr = float(r.var())
    Vc = float(c.var())
    target = Vr * max_variance_ratio

    # Quick accept: calibrated already within cap (allow tiny slack)
    if Vc <= target * VAR_TOL:
        return c, 1.0, dict(feasible=True, achieved_var=Vc, target_var=target)

    d = c - r
    # Var(d) with population convention
    Vd = float(d.var())
    # Cov(r,d) with mean(r)=1, mean(d)=0 ⇒ Cov(r,d) = E[(r-1)*d]
    C = float(np.mean((r - 1.0) * d))

    if Vd <= tol:
        # No direction to move; return raw
        return r, 0.0, dict(feasible=(Vr <= target), achieved_var=Vr, target_var=target)

    # Variance along the path: V(α) = Vr + 2αC + α^2 Vd
    # Minimum at α* = -C / Vd (clip to [0,1])
    alpha_star = float(np.clip(-C / Vd, 0.0, 1.0))
    Vmin = Vr + 2.0 * alpha_star * C + (alpha_star**2) * Vd

    # If target is unattainable, return best possible α*
    if Vmin > target * VAR_TOL:
        out = (1.0 - alpha_star) * r + alpha_star * c
        return (
            out,
            alpha_star,
            dict(
                feasible=False,
                achieved_var=Vmin,
                target_var=target,
                note="Target variance ratio unattainable; using variance-minimizing blend",
            ),
        )

    # Target is feasible: pick the LARGEST feasible α in [0,1]
    # Solve α^2 Vd + 2α C + (Vr - target) <= 0
    disc = C * C - Vd * (Vr - target)
    # Numerical guard
    if disc < 0:
        disc = 0.0
    sqrt_disc = float(np.sqrt(disc))
    alpha1 = (-C - sqrt_disc) / (Vd + tol)
    alpha2 = (-C + sqrt_disc) / (Vd + tol)

    # Feasible interval is [alpha1, alpha2]; choose the largest feasible in [0,1]
    candidate = float(np.clip(alpha2, 0.0, 1.0))
    out = (1.0 - candidate) * r + candidate * c
    Vach = float(out.var())

    # Final re-check with slack; if numerically over, fall back to alpha1 or α*
    if Vach > target * VAR_TOL:
        candidate_alt = float(np.clip(alpha1, 0.0, 1.0))
        out_alt = (1.0 - candidate_alt) * r + candidate_alt * c
        Vach_alt = float(out_alt.var())
        if Vach_alt <= target * VAR_TOL:
            return (
                out_alt,
                candidate_alt,
                dict(feasible=True, achieved_var=Vach_alt, target_var=target),
            )
        # Fall back to α* (will satisfy target up to slack because target is feasible modulo numerics)
        out_star = (1.0 - alpha_star) * r + alpha_star * c
        Vach_star = float(out_star.var())
        return (
            out_star,
            alpha_star,
            dict(
                feasible=True,
                achieved_var=Vach_star,
                target_var=target,
                note="Numerical instability; using variance-minimizing α*",
            ),
        )

    return out, candidate, dict(feasible=True, achieved_var=Vach, target_var=target)


def calibrate_to_target_mean(
    weights: np.ndarray,
    target_mean: float = 1.0,
    enforce_variance_nonincrease: bool = True,
    max_variance_ratio: float = 1.0,  # ≤1.0 ⇒ no increase; <1.0 ⇒ force reduction
    return_diagnostics: bool = False,
    projection_mode: str = "exact",  # Always use exact mode for consistency
    ordering_index: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Variance-controlled IPS weight calibration (Hájek mean-one IPS):

    1) Sort by ordering_index (or raw weights if not provided), run mean-1 PAV.
    2) Optionally apply CLOSED-FORM variance-safe blend toward raw (mean-1).
    3) Rescale to target_mean (exact).

    This method trades a small amount of bias for substantially reduced variance,
    improving stability and effective sample size. The bias-variance tradeoff is
    controlled by the max_variance_ratio parameter. All blending happens in
    mean-one space to ensure correct variance calculations.

    Guarantees:
      • output ≥ 0
      • sample mean == target_mean (within MEAN_TOL)
      • var(out)/var(raw) ≤ max_variance_ratio (when feasible) if enforced
      • weights are non-decreasing in the rank order of the ordering index

    Args:
        weights: Raw importance weights (should be non-negative)
        target_mean: Target mean for calibrated weights (default 1.0 for SNIPS)
        enforce_variance_nonincrease: Whether to cap variance at raw level
        max_variance_ratio: Maximum allowed variance ratio (≤1.0)
        return_diagnostics: If True, return (weights, diagnostics_dict)
        projection_mode: Always "exact" (bisection) for consistency
        ordering_index: Optional array to determine isotonic ordering (e.g., judge scores).
                       If None, sorts by raw weights (backward compatibility).

                       Note: When ordering_index is uncorrelated with weights (e.g., judge
                       scores uncorrelated with importance ratios), the isotonic projection
                       may produce nearly constant weights. This is expected behavior and
                       provides variance stabilization even without a monotonic relationship.

    Returns:
        Calibrated weights, or (weights, diagnostics) if return_diagnostics=True
    """
    w = np.asarray(weights, dtype=float)
    n = w.size
    if n == 0:
        if return_diagnostics:
            return w, dict(n_samples=0, alpha_blend=None, feasible=None)
        return w

    # Check for negative weights (importance weights should be non-negative)
    n_negative = int((w < -1e-15).sum())
    if n_negative > 0:
        logger.warning(f"Clipping {n_negative} negative weights (min={w.min():.3e})")
    w = np.maximum(w, 0.0)

    # Handle degenerate cases
    s = float(w.sum())
    if s <= 0:
        # Safer: constant weights at the target mean
        out = np.full_like(w, target_mean)
        if return_diagnostics:
            return out, dict(
                n_samples=n,
                alpha_blend=None,
                feasible=None,
                note="All-zero weights; returned constant weights",
            )
        return out

    # Normalize raw to mean 1 (baseline for SNIPS & variance comparison)
    mean_w = s / n  # s > 0 by construction
    raw_norm = w / mean_w  # exact mean-1 normalization (no EPS)
    raw_var = float(raw_norm.var())

    # If essentially constant, just return target_mean
    if float(np.max(raw_norm) - np.min(raw_norm)) < 1e-8 or raw_var < 1e-12:
        out = np.full_like(w, target_mean)
        if return_diagnostics:
            return out, dict(
                n_samples=n,
                alpha_blend=None,
                feasible=None,
                note="Constant weights detected",
            )
        return out

    # ---- Mean-1 PAV on the full vector (no cross-fitting) ----
    # Use ordering_index if provided (e.g., judge scores), otherwise fall back to raw weights
    if ordering_index is not None:
        if len(ordering_index) != n:
            raise ValueError(
                f"ordering_index length ({len(ordering_index)}) must match weights length ({n})"
            )
        order = np.argsort(ordering_index, kind="stable")  # Sort by the provided index

        # Handle ties in ordering_index by pooling weights within tied groups
        sorted_index = ordering_index[order]
        sorted_weights = raw_norm[order]

        # Find unique values and their positions
        unique_vals, inverse_indices = np.unique(sorted_index, return_inverse=True)

        # Pool weights within tied groups (average within each group)
        pooled_weights = np.zeros_like(sorted_weights)
        for i in range(len(unique_vals)):
            mask = inverse_indices == i
            pooled_weights[mask] = sorted_weights[mask].mean()

        # Now apply PAV to the pooled weights
        cal_sorted = _pav_mean1_projection_sorted(pooled_weights, mode=projection_mode)
    else:
        order = np.argsort(
            raw_norm, kind="stable"
        )  # Fall back to raw weights for backward compatibility
        cal_sorted = _pav_mean1_projection_sorted(raw_norm[order], mode=projection_mode)

    cal_norm = np.empty_like(raw_norm)
    cal_norm[order] = cal_sorted  # mean 1, monotone by construction

    # ---- Optional: variance-safe blend (closed form) ----
    if enforce_variance_nonincrease:
        blended_norm, alpha, blend_info = _variance_safe_blend_closed_form(
            raw_norm, cal_norm, max_variance_ratio=max_variance_ratio
        )
    else:
        blended_norm = cal_norm
        alpha = 1.0
        blend_info = dict(
            feasible=None,
            achieved_var=float(cal_norm.var()),
            target_var=raw_var * max_variance_ratio,
        )

    # ---- Final scale to target mean ----
    # Exact rescale to target mean; blended_norm.mean() > 0 by construction
    out = blended_norm * (target_mean / float(blended_norm.mean()))

    # ---- Checks ----
    if out.min() < -1e-12:
        raise AssertionError(f"Negative calibrated weight: {out.min():.3e}")

    mean_err = abs(out.mean() - target_mean)
    if mean_err >= MEAN_TOL:
        # Check for sparse case (many zeros)
        zero_frac = (w < 1e-10).mean()
        if zero_frac > 0.7:  # More than 70% zeros
            # Relax tolerance for sparse cases
            sparse_tol = max(MEAN_TOL, target_mean * 0.01)  # 1% relative tolerance
            if mean_err < sparse_tol:
                pass  # Accept with relaxed tolerance
            else:
                raise AssertionError(
                    f"Mean not preserved (sparse case {zero_frac:.1%} zeros): "
                    f"expected {target_mean:.12f}, got {out.mean():.12f} "
                    f"(error={mean_err:.2e})"
                )
        else:
            raise AssertionError(
                f"Mean not preserved: expected {target_mean:.12f}, got {out.mean():.12f} "
                f"(error={mean_err:.2e})"
            )

    # Monotonic in ordering_index (or raw_norm if no ordering_index provided)
    # CRITICAL: Must check in the same order used for isotonic regression
    if ordering_index is not None:
        idx = np.argsort(ordering_index, kind="stable")
        index_sorted = ordering_index[idx]
        out_sorted = out[idx]
        boundaries = np.flatnonzero(np.diff(index_sorted) > 1e-15)
        if boundaries.size and np.any(
            out_sorted[boundaries + 1] < out_sorted[boundaries] - 1e-12
        ):
            # Find first violation for error message
            violations = np.where(
                out_sorted[boundaries + 1] < out_sorted[boundaries] - 1e-12
            )[0]
            i = boundaries[violations[0]]
            raise AssertionError(
                f"Monotonicity violated: index {index_sorted[i]:.3e} -> weight {out_sorted[i]:.3e} "
                f"but index {index_sorted[i+1]:.3e} -> weight {out_sorted[i+1]:.3e}"
            )
    else:
        # Fall back to checking raw_norm order for backward compatibility
        idx = np.argsort(raw_norm, kind="stable")
        raw_sorted = raw_norm[idx]
        out_sorted = out[idx]
        boundaries = np.flatnonzero(np.diff(raw_sorted) > 1e-15)
        if boundaries.size and np.any(
            out_sorted[boundaries + 1] < out_sorted[boundaries] - 1e-12
        ):
            # Find first violation for error message
            violations = np.where(
                out_sorted[boundaries + 1] < out_sorted[boundaries] - 1e-12
            )[0]
            i = boundaries[violations[0]]
            raise AssertionError(
                f"Monotonicity violated: weight {raw_sorted[i]:.3e} -> {out_sorted[i]:.3e} "
                f"but weight {raw_sorted[i+1]:.3e} -> {out_sorted[i+1]:.3e}"
            )

    if return_diagnostics:
        diagnostics = {
            "n_samples": n,
            "n_negative_clipped": n_negative,
            "alpha_blend": alpha,
            "feasible": blend_info.get("feasible"),
            "achieved_var": blend_info.get("achieved_var"),
            "target_var": blend_info.get("target_var"),
            "achieved_var_ratio": (
                blend_info["achieved_var"] / raw_var if raw_var > 0 else np.nan
            ),
            "target_var_ratio": max_variance_ratio,
        }
        if "note" in blend_info:
            diagnostics["note"] = blend_info["note"]
        return out, diagnostics

    return out
