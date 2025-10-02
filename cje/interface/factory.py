"""Estimator registry and builder utilities.

Centralizes creation of estimators to avoid drift between CLI choices,
Hydra configs, and analysis code.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union
import logging

from ..data.precomputed_sampler import PrecomputedSampler
from ..estimators.calibrated_ips import CalibratedIPS
from ..estimators.direct_method import CalibratedDirectEstimator
from ..estimators.orthogonalized_ips import OrthogonalizedCalibratedIPS
from ..estimators.dr_base import DRCPOEstimator
from ..estimators.orthogonalized_calibrated_dr import OrthogonalizedCalibratedDRCPO
from ..estimators.mrdr import MRDREstimator
from ..estimators.tmle import TMLEEstimator
from ..estimators.stacking import StackedDREstimator

logger = logging.getLogger(__name__)

# Type alias for builder functions
BuilderFn = Callable[
    [PrecomputedSampler, Dict[str, Any], Optional[Any], bool],
    Union[
        CalibratedDirectEstimator,
        CalibratedIPS,
        OrthogonalizedCalibratedIPS,
        DRCPOEstimator,
        OrthogonalizedCalibratedDRCPO,
        MRDREstimator,
        TMLEEstimator,
        StackedDREstimator,
    ],
]


def _build_calibrated_ips(
    sampler: PrecomputedSampler,
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> CalibratedIPS:
    # Pass calibrator for DR-aware direction selection if available
    cfg = dict(config)
    if calibration_result and getattr(calibration_result, "calibrator", None):
        cfg.setdefault("reward_calibrator", calibration_result.calibrator)
        if verbose:
            logger.info(
                "Using reward_calibrator for DR-aware SIMCal direction selection"
            )
    return CalibratedIPS(sampler, **cfg)


def _build_calibrated_direct(
    sampler: PrecomputedSampler,
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> CalibratedDirectEstimator:
    """Build direct method estimator for on-policy evaluation."""
    cfg = dict(config)
    if calibration_result and getattr(calibration_result, "calibrator", None):
        cfg.setdefault("reward_calibrator", calibration_result.calibrator)
        if verbose:
            logger.info("Using reward_calibrator for direct method OUA")
    # DirectEstimator takes target_policies, not sampler
    return CalibratedDirectEstimator(target_policies=sampler.target_policies, **cfg)


def _build_raw_ips(
    sampler: PrecomputedSampler,
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> CalibratedIPS:
    clip_weight = config.get("clip_weight", 100.0)
    return CalibratedIPS(sampler, calibrate_weights=False, clip_weight=clip_weight)


def _build_orthogonalized_ips(
    sampler: PrecomputedSampler,
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> OrthogonalizedCalibratedIPS:
    cfg = dict(config)
    # Remove n_folds if present (not used by IPS estimators)
    cfg.pop("n_folds", None)
    if calibration_result and getattr(calibration_result, "calibrator", None):
        cfg.setdefault("reward_calibrator", calibration_result.calibrator)
        if verbose:
            logger.info("Using reward_calibrator for OC-IPS orthogonalization")
    return OrthogonalizedCalibratedIPS(sampler, **cfg)


def _build_dr_cpo(
    sampler: PrecomputedSampler,
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> DRCPOEstimator:
    n_folds = config.get("n_folds", 5)
    if calibration_result and getattr(calibration_result, "calibrator", None):
        if verbose:
            logger.info("Using calibration models for DR outcome model")
        return DRCPOEstimator(
            sampler, n_folds=n_folds, reward_calibrator=calibration_result.calibrator
        )
    return DRCPOEstimator(sampler, n_folds=n_folds)


def _build_oc_dr_cpo(
    sampler: PrecomputedSampler,
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> OrthogonalizedCalibratedDRCPO:
    n_folds = config.get("n_folds", 5)
    if calibration_result and getattr(calibration_result, "calibrator", None):
        if verbose:
            logger.info("Using calibration models for OC-DR-CPO")
        return OrthogonalizedCalibratedDRCPO(
            sampler, n_folds=n_folds, reward_calibrator=calibration_result.calibrator
        )
    return OrthogonalizedCalibratedDRCPO(sampler, n_folds=n_folds)


def _build_mrdr(
    sampler: PrecomputedSampler,
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> MRDREstimator:
    n_folds = config.get("n_folds", 5)
    omega_mode = config.get("omega_mode", "snips")
    if calibration_result and getattr(calibration_result, "calibrator", None):
        if verbose:
            logger.info("Using calibration models for MRDR")
        return MRDREstimator(
            sampler,
            n_folds=n_folds,
            omega_mode=omega_mode,
            reward_calibrator=calibration_result.calibrator,
        )
    return MRDREstimator(sampler, n_folds=n_folds, omega_mode=omega_mode)


def _build_tmle(
    sampler: PrecomputedSampler,
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> TMLEEstimator:
    n_folds = config.get("n_folds", 5)
    link = config.get("link", "logit")
    if calibration_result and getattr(calibration_result, "calibrator", None):
        if verbose:
            logger.info("Using calibration models for TMLE")
        return TMLEEstimator(
            sampler,
            n_folds=n_folds,
            link=link,
            reward_calibrator=calibration_result.calibrator,
        )
    return TMLEEstimator(sampler, n_folds=n_folds, link=link)


def _build_tr_cpo(
    sampler: PrecomputedSampler,
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> Any:
    from ..estimators.tr_cpo import TRCPOEstimator

    n_folds = config.get("n_folds", 5)
    weight_mode = config.get("weight_mode", "hajek")
    use_efficient_tr = config.get("use_efficient_tr", True)
    if calibration_result and getattr(calibration_result, "calibrator", None):
        if verbose:
            logger.info("Using calibration models for TR-CPO")
        return TRCPOEstimator(
            sampler,
            n_folds=n_folds,
            weight_mode=weight_mode,
            use_efficient_tr=use_efficient_tr,
            reward_calibrator=calibration_result.calibrator,
        )
    return TRCPOEstimator(
        sampler,
        n_folds=n_folds,
        weight_mode=weight_mode,
        use_efficient_tr=use_efficient_tr,
    )


def _build_stacked_dr(
    sampler: PrecomputedSampler,
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> StackedDREstimator:
    # Let StackedDREstimator use its own default if not specified
    estimators = config.get("estimators", None)
    parallel = config.get("parallel", True)
    seed = config.get("seed", 42)
    n_folds = config.get("n_folds", 5)
    covariance_regularization = config.get("covariance_regularization", 1e-4)
    use_calibrated_weights = config.get("use_calibrated_weights", True)
    weight_mode = config.get("weight_mode", "hajek")
    if verbose:
        logger.info(f"Using stacked DR with estimators: {estimators}")
    # Pass calibrator when available so DR components can reuse calibration models
    kwargs: Dict[str, Any] = {
        "estimators": estimators,
        "parallel": parallel,
        "seed": seed,
        "n_folds": n_folds,
        "covariance_regularization": covariance_regularization,
        "use_calibrated_weights": use_calibrated_weights,
        "weight_mode": weight_mode,
    }
    if calibration_result and getattr(calibration_result, "calibrator", None):
        if verbose:
            logger.info("Using calibration models for stacked DR components")
        kwargs["reward_calibrator"] = calibration_result.calibrator
    return StackedDREstimator(sampler, **kwargs)


REGISTRY: Dict[str, BuilderFn] = {
    "calibrated-ips": _build_calibrated_ips,
    "calibrated-direct": _build_calibrated_direct,
    "direct": _build_calibrated_direct,  # Alias
    "orthogonalized-ips": _build_orthogonalized_ips,
    "raw-ips": _build_raw_ips,
    "dr-cpo": _build_dr_cpo,
    "oc-dr-cpo": _build_oc_dr_cpo,
    # Raw variant (use_efficient_tr=False) and efficient variant (default)
    "tr-cpo": lambda s, c, r, v: _build_tr_cpo(
        s, {**c, "use_efficient_tr": False}, r, v
    ),
    "tr-cpo-e": lambda s, c, r, v: _build_tr_cpo(
        s, {**c, "use_efficient_tr": True}, r, v
    ),
    "mrdr": _build_mrdr,
    "tmle": _build_tmle,
    "stacked-dr": _build_stacked_dr,
}


def get_estimator_names() -> Tuple[str, ...]:
    return tuple(REGISTRY.keys())


def create_estimator(
    name: str,
    sampler: PrecomputedSampler,
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> Any:
    if name not in REGISTRY:
        raise ValueError(f"Unknown estimator type: {name}")
    return REGISTRY[name](sampler, config, calibration_result, verbose)
