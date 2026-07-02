"""Canonical diagnostic gate thresholds and status helpers.

Single source of truth for the CJE paper's diagnostic gates
(arXiv:2512.11150, appendix "Diagnostics for Reliable Off-Policy
Evaluation"). Every surface that grades ESS, TTC, judge-space
Bhattacharyya affinity, Hill tail indices, or calibration coverage must
import thresholds from this module so that a given number receives the
same verdict everywhere (estimator gates, diagnostics objects, display
tables, dashboards, and READMEs).

Paper gate summary:

- ESS fraction: ESS/n >= 0.30 is the ship-gate for reliable IPS.
- TTC (Target-Typicality Coverage): TTC >= 0.70 required before relying
  on logged data; below 0.30 logs-only IPS will fail outright.
- Bhattacharyya affinity in judge-score space: A_B >= 0.85 for reliable
  IPS; A_B < 0.5 indicates severe distribution mismatch.
- Hill tail index: alpha < 2 implies infinite-variance risk (WARNING);
  alpha < 1 implies infinite-mean risk (CRITICAL).
- Coverage badge: >= 5% of target judge-score mass outside the oracle
  calibration range triggers REFUSE-LEVEL (level claims refused;
  rankings may stand).
"""

import math
from typing import Optional

from .models import Status

# ---------------------------------------------------------------------------
# ESS fraction (effective sample size / n)
# ---------------------------------------------------------------------------
# The paper's ship-gate for IPS is ESS/n >= 0.30 (bounds variance inflation
# to <= 3.3x). Below 0.10 the estimate is dominated by a small subset of
# samples and is graded CRITICAL.
ESS_GOOD_THRESHOLD = 0.30
ESS_WARNING_THRESHOLD = 0.10

# ---------------------------------------------------------------------------
# TTC (Target-Typicality Coverage)
# ---------------------------------------------------------------------------
# TTC < 0.70 => logs-only IPS will fail (CLE precision floor); prefer
# Direct/DR. TTC < 0.30 => poor coverage, IPS fails regardless of ESS.
TTC_GOOD_THRESHOLD = 0.70
TTC_CRITICAL_THRESHOLD = 0.30

# ---------------------------------------------------------------------------
# Bhattacharyya affinity A_B in judge-score space (binned)
# ---------------------------------------------------------------------------
BHATTACHARYYA_GOOD_THRESHOLD = 0.85
BHATTACHARYYA_SEVERE_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Hill tail index
# ---------------------------------------------------------------------------
TAIL_INDEX_CRITICAL = 1.0  # alpha < 1: infinite-mean risk
TAIL_INDEX_WARNING = 2.0  # alpha < 2: infinite-variance risk

# ---------------------------------------------------------------------------
# Coverage badge (boundary_card)
# ---------------------------------------------------------------------------
# Fraction of target judge-score mass outside the oracle calibration range
# at or above which level claims are refused (REFUSE-LEVEL).
OUT_OF_RANGE_REFUSE_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Companion refusal-gate thresholds used by CalibratedIPS
# ---------------------------------------------------------------------------
RAW_NEAR_ZERO_CRITICAL = 0.85  # > 85% of raw weights near zero
TOP_5PCT_CONCENTRATION_WARNING = 0.30  # top 5% of samples hold > 30% of weight
WEIGHT_CV_WARNING = 2.0  # coefficient of variation of weights


_STATUS_ORDER = {Status.GOOD: 0, Status.WARNING: 1, Status.CRITICAL: 2}


def worst_status(*statuses: Optional[Status]) -> Status:
    """Combine statuses, returning the worst. None entries are ignored.

    Returns Status.GOOD when no non-None status is provided.
    """
    worst = Status.GOOD
    for status in statuses:
        if status is not None and _STATUS_ORDER[status] > _STATUS_ORDER[worst]:
            worst = status
    return worst


def ess_status(ess_fraction: float) -> Status:
    """Grade an ESS fraction on the canonical (paper) ladder.

    >= 0.30 GOOD (paper ship-gate) / >= 0.10 WARNING / else CRITICAL.
    """
    if ess_fraction >= ESS_GOOD_THRESHOLD:
        return Status.GOOD
    if ess_fraction >= ESS_WARNING_THRESHOLD:
        return Status.WARNING
    return Status.CRITICAL


def ttc_status(ttc: float) -> Status:
    """Grade Target-Typicality Coverage on the paper's ladder.

    >= 0.70 GOOD / >= 0.30 WARNING (marginal, consider DR) /
    else CRITICAL (IPS will fail regardless of ESS).
    """
    if ttc >= TTC_GOOD_THRESHOLD:
        return Status.GOOD
    if ttc >= TTC_CRITICAL_THRESHOLD:
        return Status.WARNING
    return Status.CRITICAL


def bhattacharyya_status(affinity: float) -> Status:
    """Grade a judge-space Bhattacharyya affinity A_B on the paper's ladder.

    >= 0.85 GOOD (reliable IPS gate) / >= 0.50 WARNING / else CRITICAL
    (severe distribution mismatch).
    """
    if affinity >= BHATTACHARYYA_GOOD_THRESHOLD:
        return Status.GOOD
    if affinity >= BHATTACHARYYA_SEVERE_THRESHOLD:
        return Status.WARNING
    return Status.CRITICAL


def tail_status(
    tail_index: Optional[float], ess_fraction: Optional[float] = None
) -> Optional[Status]:
    """Grade a Hill tail index, treating NaN as "estimation failed".

    Args:
        tail_index: Hill tail index. None means "not computed" (e.g. too
            few samples requested no computation); NaN means estimation
            FAILED (degenerate weights) and must not be read as a healthy
            tail; +inf means a genuinely uniform (lightest possible) tail.
        ess_fraction: Optional ESS fraction used to escalate unknown tails:
            when the tail could not be estimated AND ESS is below the
            ship-gate, the combination is graded WARNING rather than
            silently ignored.

    Returns:
        Status, or None when the tail is not computed / unknown with no
        corroborating evidence of trouble.
    """
    if tail_index is None:
        return None
    if math.isnan(tail_index):
        # Estimation failed - unknown tail. Escalate only when ESS also
        # indicates trouble; otherwise stay silent (unknown != healthy).
        if ess_fraction is not None and ess_fraction < ESS_GOOD_THRESHOLD:
            return Status.WARNING
        return None
    if tail_index < TAIL_INDEX_CRITICAL:
        return Status.CRITICAL
    if tail_index < TAIL_INDEX_WARNING:
        return Status.WARNING
    return Status.GOOD


def format_tail_index(tail_index: Optional[float]) -> str:
    """Format a tail index for display, rendering failures as 'n/a'."""
    if tail_index is None or math.isnan(tail_index):
        return "n/a"
    if math.isinf(tail_index):
        return "inf"
    return f"{tail_index:.2f}"
