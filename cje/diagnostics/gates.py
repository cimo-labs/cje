"""Canonical diagnostic gate thresholds and status helpers.

Single source of truth for the CJE paper's diagnostic gates
(arXiv:2512.11150). Every surface that grades diagnostics must import
thresholds from this module so that a given number receives the same
verdict everywhere.

Diagnostic summary:

- Scalar range support: >= 5% of target judge-score mass outside the labeled
  calibration range triggers REFUSE-LEVEL for absolute claims. This check does
  not establish residual transport or the validity of policy rankings.
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
    "INCONCLUSIVE": Status.WARNING,
}

# ---------------------------------------------------------------------------
# Transport audit (audit_transportability)
# ---------------------------------------------------------------------------
# Deprecated compatibility constant. Residual audits are graded against an
# explicit, user-declared delta_max; this value is not used by the classifier.
TRANSPORT_FAIL_DELTA_THRESHOLD = 0.05

# Canonical transport status -> Status ladder for consumers that grade
# TransportDiagnostics alongside other diagnostics.
TRANSPORT_STATUS_TO_STATUS: Dict[str, Status] = {
    "PASS": Status.GOOD,
    "FAIL": Status.CRITICAL,
    "INCONCLUSIVE": Status.WARNING,
    "NOT_GRADED": Status.WARNING,
    # NOT_CHECKED means no probe was supplied at all — informational, not a
    # defect: it maps to GOOD so probe-less runs do not warn in worst-status
    # rollups. The per-policy audit record still says NOT_CHECKED explicitly.
    "NOT_CHECKED": Status.GOOD,
    # Compatibility for previously serialized diagnostics only.
    "WARN": Status.WARNING,
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
