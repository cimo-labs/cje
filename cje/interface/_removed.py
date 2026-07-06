"""Migration errors for estimator names removed in 0.4.0.

Single source of truth for the CLI and the API: both route estimator names
through validate_estimator_name so a removed 0.3.x OPE name surfaces the
same migration copy everywhere (pinned verbatim by test_migration_errors.py).
"""

from typing import Tuple

# Direct-mode estimator names that survive in 0.4.x+ ("auto" resolves to
# "direct" before validation).
_VALID_ESTIMATORS: Tuple[str, ...] = (
    "calibrated-direct",
    "direct",
)

# Estimator names removed in 0.4.0 with the off-policy (IPS/DR) modes
# (includes the ghost aliases oc-dr-cpo / tr-cpo / tr-cpo-e that 0.3.x
# still recognized in its mode-inference lists).
REMOVED_ESTIMATORS: Tuple[str, ...] = (
    "calibrated-ips",
    "raw-ips",
    "dr-cpo",
    "mrdr",
    "tmle",
    "stacked-dr",
    "oc-dr-cpo",
    "tr-cpo",
    "tr-cpo-e",
)

# Exact migration copy, shared by the CLI and the API; pinned verbatim by
# test_migration_errors.py.
REMOVED_ESTIMATOR_MESSAGE = """\
estimator='{name}' was removed in cje-eval 0.4.0 (Direct-mode only).
Off-policy estimators (calibrated-ips, raw-ips, dr-cpo, mrdr, tmle, stacked-dr)
live on the frozen 0.3.x line: pip install "cje-eval==0.3.*"
(requires Python <=3.12; on 3.13 use a 3.12 env for OPE).
Use estimator='calibrated-direct' (the default) with fresh draws instead."""


def validate_estimator_name(name: str) -> str:
    """Validate an estimator name, raising the migration error for removed ones.

    Single source of truth for CLI and API. Returns the name unchanged when
    it is valid.
    """
    if name in _VALID_ESTIMATORS:
        return name
    if name in REMOVED_ESTIMATORS:
        raise ValueError(REMOVED_ESTIMATOR_MESSAGE.format(name=name))
    raise ValueError(f"Unknown estimator type: {name}")
