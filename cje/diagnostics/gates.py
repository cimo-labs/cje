"""Canonical diagnostic gate thresholds and status helpers.

Single source of truth for the CJE paper's diagnostic gates
(arXiv:2512.11150). Every surface that grades diagnostics must import
thresholds from this module so that a given number receives the same
verdict everywhere.

Paper gate summary (surviving Direct-mode gates):

- Coverage badge: >= 5% of target judge-score mass outside the oracle
  calibration range triggers REFUSE-LEVEL (level claims refused;
  rankings may stand).
"""

from typing import Dict, Optional

from .models import Status

# ---------------------------------------------------------------------------
# Coverage badge (boundary_card)
# ---------------------------------------------------------------------------
# Fraction of target judge-score mass outside the oracle calibration range
# at or above which level claims are refused (REFUSE-LEVEL).
OUT_OF_RANGE_REFUSE_THRESHOLD = 0.05

# Fraction of calibrated rewards near the oracle reward bounds at or above
# which the card downgrades to CAUTION (boundary effects likely) when the
# out-of-range mass is below the refuse threshold.
SATURATION_CAUTION_THRESHOLD = 0.20

# Canonical boundary-card status -> per-policy Status ladder: a CAUTION card
# is a real WARNING tier between GOOD and CRITICAL (it used to leave the
# policy GOOD, making CAUTION invisible in overall_status).
BOUNDARY_CARD_STATUS_TO_STATUS: Dict[str, Status] = {
    "OK": Status.GOOD,
    "CAUTION": Status.WARNING,
    "REFUSE-LEVEL": Status.CRITICAL,
}

# ---------------------------------------------------------------------------
# Transport audit (audit_transportability)
# ---------------------------------------------------------------------------
# |delta_hat| below this splits WARN (marginal bias, monitor) from FAIL
# (clear bias, refit) once the residual CI excludes zero.
TRANSPORT_FAIL_DELTA_THRESHOLD = 0.05

# Canonical transport status -> Status ladder for consumers that grade
# TransportDiagnostics alongside other diagnostics.
TRANSPORT_STATUS_TO_STATUS: Dict[str, Status] = {
    "PASS": Status.GOOD,
    "WARN": Status.WARNING,
    "FAIL": Status.CRITICAL,
}


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
