"""
Minimal boundary diagnostics for reward calibration extrapolation detection.

Provides a simple 3-signal triage to detect when estimates may be unreliable
due to extrapolation, WITHOUT requiring ground truth labels.

This implements the paper's coverage badge (arXiv:2512.11150, appendix
diagnostics): judge-score mass outside the oracle calibration range at or
above the 5% threshold triggers REFUSE-LEVEL — level (absolute) claims are
refused while rankings may stand.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .gates import OUT_OF_RANGE_REFUSE_THRESHOLD


@dataclass
class BoundaryCard:
    """Minimal boundary diagnostic result.

    Attributes:
        status: "OK" | "CAUTION" | "REFUSE-LEVEL"
        out_of_range: Fraction of S outside oracle S-range (primary signal)
        saturation: Fraction of R near reward bounds
        estimator_gap: Optional absolute gap between two estimators' values
            for the same policy, if the caller computes one. The 0.4.x
            pipeline never populates it (there is only one estimator).
        partial_id_width: Width of partial-ID band under monotonicity
        note: Brief explanation of status
    """

    status: str
    out_of_range: float
    saturation: float
    estimator_gap: Optional[float] = None
    partial_id_width: Optional[float] = None
    note: str = ""


def boundary_card(
    S_policy: np.ndarray,
    S_oracle: np.ndarray,
    R_policy: np.ndarray,
    R_min: float,
    R_max: float,
    est_calips: Optional[float] = None,
    est_dr: Optional[float] = None,
    band_frac: float = 0.05,
) -> BoundaryCard:
    """Minimal boundary triage without ground truth labels.

    Args:
        S_policy: Judge scores for target policy
        S_oracle: Judge scores from oracle calibration slice
        R_policy: Calibrated rewards for target policy
        R_min: Minimum reward in oracle calibration set
        R_max: Maximum reward in oracle calibration set
        est_calips: Optional estimate from a second estimator, used only for
            the cross-estimator gap signal (unused in the 0.4.x pipeline)
        est_dr: Optional estimate from another estimator (see est_calips)
        band_frac: Fraction of range to consider "near boundary" (default 5%)

    Returns:
        BoundaryCard with status and key metrics
    """
    if len(S_policy) == 0 or len(S_oracle) == 0 or len(R_policy) == 0:
        return BoundaryCard(
            status="OK", out_of_range=0.0, saturation=0.0, note="insufficient data"
        )

    # Signal 1: Out-of-range S mass (primary identification risk)
    Smin, Smax = float(np.min(S_oracle)), float(np.max(S_oracle))
    out_lower = float(np.mean(S_policy < Smin))
    out_upper = float(np.mean(S_policy > Smax))
    out_of_range = out_lower + out_upper

    # Signal 2: Reward saturation near boundaries
    margin = band_frac * (R_max - R_min) if R_max > R_min else 0.0
    sat = float(np.mean((R_policy <= R_min + margin) | (R_policy >= R_max - margin)))

    # Signal 3: Estimator gap (optional)
    gap = None
    if est_calips is not None and est_dr is not None:
        gap = abs(est_dr - est_calips)

    # Partial-ID band width under monotonicity
    pidw = None
    if out_of_range > 0:
        # Conservative bound: uncovered mass times boundary value
        pidw = out_lower * R_min + out_upper * (1.0 - R_max)

    # Single gate with clear thresholds (canonical: gates.py)
    if out_of_range >= OUT_OF_RANGE_REFUSE_THRESHOLD:
        status = "REFUSE-LEVEL"
        note = "Non-trivial judge mass outside oracle range; do not ship levels."
    elif sat >= 0.20 or (gap is not None and gap >= 0.10):
        status = "CAUTION"
        note = "Boundary effects likely; report partial-ID band and caveat."
    else:
        status = "OK"
        note = "Coverage looks fine; levels appear reliable."

    return BoundaryCard(
        status=status,
        out_of_range=float(out_of_range),
        saturation=float(sat),
        estimator_gap=gap,
        partial_id_width=pidw,
        note=note,
    )
