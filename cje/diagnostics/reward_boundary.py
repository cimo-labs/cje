"""
Scalar range-support diagnostics for reward calibration.

Provides a descriptive check for extrapolation beyond labeled judge-score
support, without requiring target-policy oracle labels. It is not a residual
transport audit and does not establish policy-ranking validity.

This implements the paper's coverage badge (arXiv:2512.11150, appendix
diagnostics): judge-score mass outside the oracle calibration range at or
above the 5% threshold triggers REFUSE-LEVEL for level (absolute) claims.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import logging

import numpy as np

from .gates import OUT_OF_RANGE_REFUSE_THRESHOLD, SATURATION_CAUTION_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class BoundaryCard:
    """Minimal boundary diagnostic result.

    Attributes:
        status: "OK" | "CAUTION" | "REFUSE-LEVEL" | "INCONCLUSIVE"
        out_of_range: Fraction of S outside oracle S-range (primary signal)
        saturation: Fraction of R near reward bounds
        partial_id_width: Width of partial-ID band under monotonicity
        note: Brief explanation of status
    """

    status: str
    out_of_range: float
    saturation: float
    partial_id_width: Optional[float] = None
    note: str = ""


def boundary_card(
    S_policy: np.ndarray,
    S_oracle: np.ndarray,
    R_policy: np.ndarray,
    R_min: float,
    R_max: float,
    band_frac: float = 0.05,
) -> BoundaryCard:
    """Check scalar score-range support without target ground-truth labels.

    Args:
        S_policy: Judge scores for target policy
        S_oracle: Judge scores from oracle calibration slice
        R_policy: Calibrated rewards for target policy
        R_min: Minimum reward in oracle calibration set
        R_max: Maximum reward in oracle calibration set
        band_frac: Fraction of range to consider "near boundary" (default 5%)

    Returns:
        BoundaryCard with status and key metrics
    """
    if len(S_policy) == 0 or len(S_oracle) == 0 or len(R_policy) == 0:
        return BoundaryCard(
            status="INCONCLUSIVE",
            out_of_range=float("nan"),
            saturation=float("nan"),
            note="Scalar range support was not checked: insufficient data.",
        )

    # Signal 1: Out-of-range S mass (primary identification risk)
    Smin, Smax = float(np.min(S_oracle)), float(np.max(S_oracle))
    out_lower = float(np.mean(S_policy < Smin))
    out_upper = float(np.mean(S_policy > Smax))
    out_of_range = out_lower + out_upper

    # Signal 2: Reward saturation near boundaries
    margin = band_frac * (R_max - R_min) if R_max > R_min else 0.0
    sat = float(np.mean((R_policy <= R_min + margin) | (R_policy >= R_max - margin)))

    # Partial-ID band width under monotonicity
    pidw = None
    if out_of_range > 0:
        # Conservative bound: uncovered mass times boundary value
        pidw = out_lower * R_min + out_upper * (1.0 - R_max)

    # Single gate with clear thresholds (canonical: gates.py)
    if out_of_range >= OUT_OF_RANGE_REFUSE_THRESHOLD:
        status = "REFUSE-LEVEL"
        note = (
            "Material target score mass lies outside labeled calibration range; "
            "do not report absolute level claims from this fit."
        )
    elif sat >= SATURATION_CAUTION_THRESHOLD:
        status = "CAUTION"
        note = (
            "No material scalar range extrapolation was detected, but calibrated "
            "predictions are concentrated near observed reward boundaries. This "
            "does not establish residual transport."
        )
    else:
        status = "OK"
        note = (
            "No material scalar range extrapolation was detected at the 5% "
            "threshold. This does not establish residual transport or covariate "
            "support."
        )

    return BoundaryCard(
        status=status,
        out_of_range=float(out_of_range),
        saturation=float(sat),
        partial_id_width=pidw,
        note=note,
    )


def boundary_card_dict(
    calibrator: Any,
    S_policy: np.ndarray,
    R_policy: np.ndarray,
    warn_label: Optional[str] = None,
    *,
    emit_warning: bool = True,
) -> Optional[Dict[str, Any]]:
    """Coverage badge as a metadata dict, from a fitted calibrator.

    Wraps `boundary_card` against the oracle S/reward ranges the reward
    calibrator stored at fit time, attaches ``oracle_s_range`` to the dict,
    and emits the REFUSE-LEVEL warning. Used by both
    `CalibratedDirectEstimator` (per policy) and `calibrated_mean_ci`.

    Args:
        calibrator: Fitted calibrator exposing ``oracle_s_range`` and
            ``oracle_reward_range`` (min/max of the oracle slice — the card
            reads only min/max of S_oracle, so the stored range stands in
            for the full slice).
        S_policy: Judge scores for the evaluated sample (non-finite values
            are dropped).
        R_policy: Calibrated rewards for the evaluated sample.
        warn_label: Policy name for the per-policy REFUSE-LEVEL warning text
            (Direct mode). None uses the single-sample wording.
        emit_warning: Whether a REFUSE-LEVEL card should emit a warning. Set
            False when the card is descriptive but does not apply to the
            reported point-estimator route.

    Returns:
        Serialized card dict with ``oracle_s_range`` attached, or None when
        the calibrator has no stored ranges or the sample is empty.
    """
    s_range = getattr(calibrator, "oracle_s_range", None)
    r_range = getattr(calibrator, "oracle_reward_range", None)
    if s_range is None or r_range is None:
        return None

    S_arr = np.asarray(S_policy, dtype=float)
    S_arr = S_arr[np.isfinite(S_arr)]
    R_arr = np.asarray(R_policy, dtype=float)
    if len(S_arr) == 0 or len(R_arr) == 0:
        return None

    card = boundary_card(
        S_policy=S_arr,
        S_oracle=np.asarray(s_range, dtype=float),
        R_policy=R_arr,
        R_min=float(r_range[0]),
        R_max=float(r_range[1]),
    )
    card_dict: Dict[str, Any] = asdict(card)
    card_dict["oracle_s_range"] = [float(s_range[0]), float(s_range[1])]

    if card.status == "REFUSE-LEVEL" and emit_warning:
        if warn_label is not None:
            logger.warning(
                f"REFUSE-LEVEL for policy '{warn_label}': "
                f"{card.out_of_range:.1%} of fresh-draw judge "
                f"scores fall outside the oracle calibration range "
                f"[{s_range[0]:.3f}, {s_range[1]:.3f}]. Do not report level "
                f"(absolute) claims for this policy from this fit. "
                f"Collect oracle labels covering the missing score range."
            )
        else:
            logger.warning(
                f"REFUSE-LEVEL coverage badge: {card.out_of_range:.1%} of judge "
                f"scores fall outside the oracle calibration range "
                f"[{s_range[0]:.3f}, {s_range[1]:.3f}]. Do not ship level "
                f"(absolute) claims from this estimate."
            )
    return card_dict
