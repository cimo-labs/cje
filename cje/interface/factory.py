"""Estimator registry and builder utilities.

Centralizes creation of estimators (and rejection of removed ones) so the
CLI and the analysis service stay in sync.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

from ..estimators.direct_method import CalibratedDirectEstimator

logger = logging.getLogger(__name__)

# Type alias for builder functions
BuilderFn = Callable[
    [List[str], Dict[str, Any], Optional[Any], bool],
    CalibratedDirectEstimator,
]


def _build_calibrated_direct(
    target_policies: List[str],
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> CalibratedDirectEstimator:
    """Build direct method estimator for on-policy evaluation."""
    cfg = dict(config)
    if calibration_result and getattr(calibration_result, "calibrator", None):
        cfg.setdefault("reward_calibrator", calibration_result.calibrator)
        if verbose:
            logger.info(
                "Using reward_calibrator for direct-method calibration-aware inference"
            )
    return CalibratedDirectEstimator(target_policies=target_policies, **cfg)


REGISTRY: Dict[str, BuilderFn] = {
    "calibrated-direct": _build_calibrated_direct,
    "direct": _build_calibrated_direct,  # Alias
}

# Estimator names removed in 0.4.0 with the off-policy (IPS/DR) modes.
REMOVED_ESTIMATORS: Tuple[str, ...] = (
    "calibrated-ips",
    "raw-ips",
    "dr-cpo",
    "mrdr",
    "tmle",
    "stacked-dr",
)


def get_estimator_names() -> Tuple[str, ...]:
    return tuple(REGISTRY.keys())


def validate_estimator_name(name: str) -> str:
    """Validate an estimator name, raising the migration error for removed ones.

    Single source of truth for CLI and API. Returns the name unchanged when
    it is valid.
    """
    if name in REGISTRY:
        return name
    if name in REMOVED_ESTIMATORS:
        # NOTE(WP2): the exact migration copy is pinned by test_migration_errors.
        raise ValueError(
            f"Estimator '{name}' was removed in 0.4.0 — full migration message "
            f'coming in WP2. For IPS/DR estimators pin pip install "cje-eval==0.3.*".'
        )
    raise ValueError(f"Unknown estimator type: {name}")


def create_estimator(
    name: str,
    target_policies: List[str],
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> CalibratedDirectEstimator:
    validate_estimator_name(name)
    return REGISTRY[name](target_policies, config, calibration_result, verbose)
