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

from typing import Optional

from .models import Status

# ---------------------------------------------------------------------------
# Coverage badge (boundary_card)
# ---------------------------------------------------------------------------
# Fraction of target judge-score mass outside the oracle calibration range
# at or above which level claims are refused (REFUSE-LEVEL).
OUT_OF_RANGE_REFUSE_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# ESS fraction (effective sample size / n)
# ---------------------------------------------------------------------------
# NOTE(WP3): the ESS ladder survives only for the weight-summary display
# table, which is pruned with the diagnostics-model redesign; it dies there.
ESS_GOOD_THRESHOLD = 0.30
ESS_WARNING_THRESHOLD = 0.10


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
