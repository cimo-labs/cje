"""Judge score calibration for reward calibration.

This module calibrates cheap LLM judge scores to oracle labels using
monotonic regression on a labeled subset. It supports both monotone and
two-stage calibration, with automatic mode selection when requested.

All modes share ONE cross-fitted implementation (`FlexibleCalibrator`):
`JudgeCalibrator` handles input parsing, fold assignment, and diagnostics,
then delegates model fitting and prediction.
"""

import hashlib
import numpy as np
from typing import Optional, Tuple, Dict, List, Literal, TYPE_CHECKING, Any
from dataclasses import dataclass
import logging

if TYPE_CHECKING:
    from .flexible_calibrator import FlexibleCalibrator

logger = logging.getLogger(__name__)


def resolve_n_folds(
    n_oracle: int, requested_folds: int, n_oracle_clusters: Optional[int] = None
) -> int:
    """Choose the number of calibration CV folds for the available labels.

    Cross-fitted calibration needs at least 2 oracle samples per fold. When
    fewer than ``2 * requested_folds`` oracle labels are available (but at
    least 4), reduce the fold count instead of failing outright. Below 4
    labels the requested count is returned unchanged so that ``fit_cv``'s
    actionable too-few-labels error fires.

    Args:
        n_oracle: Number of oracle-labeled samples available for calibration
        requested_folds: Desired number of CV folds

    Returns:
        Number of folds to use (possibly reduced)
    """
    effective_n = n_oracle if n_oracle_clusters is None else n_oracle_clusters
    if effective_n >= requested_folds * 2 or effective_n < 4:
        # Enough labels for the requested folds, or too few to calibrate
        # at all (let the calibrator raise its actionable error).
        return requested_folds
    reduced_folds = max(2, effective_n // 2)
    logger.warning(
        f"Only {effective_n} unique oracle-labeled clusters available; reducing "
        f"calibration folds from {requested_folds} to {reduced_folds}. Results "
        f"will be noisier — provide at least {requested_folds * 2} oracle labels "
        f"for stable calibration."
    )
    return reduced_folds


def _balanced_cluster_folds(
    prompt_ids: List[str],
    n_folds: int,
    seed: int,
    balancing_prompt_ids: Optional[List[str]] = None,
) -> np.ndarray:
    """Assign whole prompt clusters to deterministic, balanced folds.

    Hashing directly modulo ``n_folds`` can leave folds empty on small oracle
    slices.  Sorting cluster ids by a stable seeded hash and assigning them
    round-robin preserves determinism while guaranteeing fold sizes differ by
    at most one cluster.
    """
    unique_prompts = sorted(set(balancing_prompt_ids or prompt_ids))

    def hash_key(prompt_id: str) -> bytes:
        return hashlib.blake2b(f"{prompt_id}-{seed}".encode(), digest_size=16).digest()

    ordered = sorted(unique_prompts, key=lambda pid: (hash_key(pid), pid))
    prompt_to_fold = {pid: rank % n_folds for rank, pid in enumerate(ordered)}
    for prompt_id in set(prompt_ids) - set(prompt_to_fold):
        prompt_to_fold[prompt_id] = int.from_bytes(hash_key(prompt_id), "big") % n_folds
    return np.asarray([prompt_to_fold[pid] for pid in prompt_ids], dtype=int)


def _index_mask_to_bool(
    indices: np.ndarray, oracle_labels: np.ndarray, n_total: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an integer index mask to a boolean mask, keeping labels aligned.

    Boolean fancy-indexing (``judge_scores[bool_mask]``) returns scores in
    ascending index order. If the caller passed unsorted indices, the labels
    must be reordered identically or every (score, label) pair silently
    misaligns.

    Args:
        indices: Integer indices of oracle-labeled samples (caller order)
        oracle_labels: Oracle labels in the same caller order as ``indices``
        n_total: Total number of samples

    Returns:
        Tuple of (bool_mask, oracle_labels reordered to ascending-index order,
        argsort permutation applied to the labels — any other compact array
        given in caller order, e.g. sample_weight, must be reordered with it)

    Raises:
        ValueError: If indices contain duplicates or lengths mismatch
    """
    indices = np.asarray(indices)
    oracle_labels = np.asarray(oracle_labels)

    if indices.ndim != 1:
        raise ValueError("oracle_mask index array must be one-dimensional")
    if not np.issubdtype(indices.dtype, np.integer) or np.issubdtype(
        indices.dtype, np.bool_
    ):
        raise ValueError("oracle_mask indices must have an integer dtype")
    if np.any(indices < 0):
        bad = indices[indices < 0]
        raise ValueError(
            f"oracle_mask contains negative indices {bad[:5].tolist()}; "
            "indices must be in [0, n_samples)."
        )
    if np.any(indices >= n_total):
        bad = indices[indices >= n_total]
        raise ValueError(
            f"oracle_mask contains out-of-range indices {bad[:5].tolist()} "
            f"for n_samples={n_total}."
        )

    unique_indices, counts = np.unique(indices, return_counts=True)
    if np.any(counts > 1):
        duplicated = unique_indices[counts > 1]
        raise ValueError(
            f"oracle_mask contains duplicate indices "
            f"{duplicated[:5].tolist()}{'...' if duplicated.size > 5 else ''}; "
            f"each sample may be indexed at most once."
        )
    if len(oracle_labels) != len(indices):
        raise ValueError(
            f"oracle_labels length ({len(oracle_labels)}) must match the "
            f"number of oracle_mask indices ({len(indices)})."
        )

    bool_mask = np.zeros(n_total, dtype=bool)
    bool_mask[indices] = True

    order = np.argsort(indices)
    return bool_mask, oracle_labels[order], order


@dataclass
class CalibrationResult:
    """Result of judge calibration."""

    calibrated_scores: np.ndarray  # Calibrated scores for all data
    calibration_rmse: float  # RMSE on oracle subset
    coverage_at_01: float  # Fraction within ±0.1 of true label
    n_oracle: int  # Number of oracle samples used
    calibrator: Optional["JudgeCalibrator"] = None  # The fitted calibrator
    fold_ids: Optional[np.ndarray] = None  # CV fold assignment for each sample
    oof_rmse: Optional[float] = None  # Out-of-fold RMSE (if cross-fitted)
    oof_coverage_at_01: Optional[float] = None  # Out-of-fold coverage (if cross-fitted)

    def summary(self) -> str:
        """Format calibration results."""
        return (
            f"Calibration Summary:\n"
            f"  Oracle samples: {self.n_oracle}\n"
            f"  RMSE: {self.calibration_rmse:.3f}\n"
            f"  Coverage (±0.1): {self.coverage_at_01:.1%}"
        )


class JudgeCalibrator:
    """Calibrate judge scores to oracle labels for reward calibration.

    This is the core judge-score calibration implementation. It provides a
    mean-preserving, largely monotone mapping from judge scores to oracle
    labels with automatic mode selection.

    Args:
        random_seed: Random seed for reproducibility
        calibration_mode: Calibration mode - 'auto' (default), 'monotone', or 'two_stage'
    """

    def __init__(
        self,
        random_seed: int = 42,
        calibration_mode: Optional[Literal["monotone", "two_stage", "auto"]] = "auto",
        covariate_names: Optional[List[str]] = None,
    ):
        """Initialize judge calibrator.

        Args:
            random_seed: Random seed for reproducibility
            calibration_mode: Calibration method to use:
                - 'auto' (default): Automatically select based on cross-validation
                - 'monotone': Force standard isotonic regression
                - 'two_stage': Force flexible two-stage calibration
                - None: Use monotone (for backward compatibility)
            covariate_names: Optional list of covariate names to extract from Sample.metadata
                for use in two-stage calibration (e.g., ["response_length", "domain"])
        """
        self.random_seed = random_seed
        # None defaults to 'monotone' for backward compatibility
        self.calibration_mode = (
            calibration_mode if calibration_mode is not None else "monotone"
        )
        self.covariate_names = covariate_names or []
        # Store selected mode (for auto, this gets updated after selection)
        self.selected_mode: Optional[str] = (
            None if self.calibration_mode == "auto" else self.calibration_mode
        )
        # The single cross-fitted implementation for every mode
        self._flexible_calibrator: Optional["FlexibleCalibrator"] = None
        self._fold_ids: Optional[np.ndarray] = None
        self._n_folds: int = 5
        self._prompt_ids: Optional[List[str]] = (
            None  # Store prompt_ids for fold assignment
        )
        self.oracle_coverage: Optional[float] = (
            None  # Fraction of samples with oracle labels
        )
        # Training support, stored at fit time for coverage badges
        # (boundary cards): the min/max judge score in the oracle slice and
        # the min/max calibrated reward on that slice.
        self.oracle_s_range: Optional[Tuple[float, float]] = None
        self.oracle_reward_range: Optional[Tuple[float, float]] = None
        # Fit-time quality metrics, exposed via get_calibration_info() for
        # estimator diagnostics (empty until fit_cv runs).
        self._calibration_info: Dict[str, Any] = {}
        # Exact compact fit inputs are retained so lower-level callers can
        # construct an explicit calibration-provenance contract.  These are
        # statistical state, not serialized result payloads.
        self._fit_judge_scores: Optional[np.ndarray] = None
        self._fit_oracle_labels: Optional[np.ndarray] = None
        self._fit_prompt_ids: Optional[List[str]] = None
        self._fit_covariates: Optional[np.ndarray] = None
        self._fit_sample_weight: Optional[np.ndarray] = None
        self._oracle_fold_ids: Optional[np.ndarray] = None

    @property
    def n_folds(self) -> int:
        """CV fold count actually used by the last `fit_cv` (after auto-reduction)."""
        return self._n_folds

    def _store_oracle_ranges(
        self, oracle_scores: np.ndarray, oracle_calibrated: np.ndarray
    ) -> None:
        """Store the oracle-slice judge-score and reward support at fit time.

        The boundary card (paper's coverage badge) compares a target
        policy's judge scores against this range: judge mass outside it
        means the calibrator extrapolates and level claims are at risk.
        """
        if len(oracle_scores) > 0:
            self.oracle_s_range = (
                float(np.min(oracle_scores)),
                float(np.max(oracle_scores)),
            )
        if len(oracle_calibrated) > 0:
            self.oracle_reward_range = (
                float(np.min(oracle_calibrated)),
                float(np.max(oracle_calibrated)),
            )

    def _warn_if_constant_monotone_fit(
        self, oracle_scores: np.ndarray, oracle_y: np.ndarray
    ) -> None:
        """Warn loudly when monotone calibration collapses to a constant.

        Isotonic regression on a judge that is anti-correlated with the
        oracle fits a single constant (the oracle mean): every calibrated
        reward becomes identical and all policy differences vanish silently.
        """
        if self._flexible_calibrator is None:
            return
        fitted = np.asarray(
            self._flexible_calibrator.predict(np.asarray(oracle_scores), folds=None)
        )
        if len(np.unique(fitted)) == 1 and len(np.unique(np.asarray(oracle_y))) > 1:
            logger.warning(
                f"Monotone calibration collapsed to a constant "
                f"({fitted[0]:.3f}, the oracle mean) even though oracle labels "
                f"vary. All calibrated rewards will be identical, erasing "
                f"policy differences. The judge scale may be inverted "
                f"(anti-correlated with the oracle) — check the judge score "
                f"orientation or use calibration_mode='auto'."
            )

    def predict(
        self, judge_scores: np.ndarray, covariates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply calibration to new judge scores.

        Args:
            judge_scores: Judge scores to calibrate
            covariates: Optional covariate matrix (n_samples, n_covariates)

        Returns:
            Calibrated scores
        """
        if self._flexible_calibrator is None:
            raise RuntimeError("Calibrator must be fitted before prediction")

        if (
            covariates is not None
            and (self.selected_mode or self.calibration_mode) == "monotone"
        ):
            raise ValueError(
                "Covariates provided but calibrator was fitted in monotone mode without covariate support"
            )

        return np.clip(
            self._flexible_calibrator.predict(
                np.asarray(judge_scores), folds=None, covariates=covariates
            ),
            0.0,
            1.0,
        )

    def fit_cv(
        self,
        judge_scores: np.ndarray,
        oracle_labels: Optional[np.ndarray] = None,
        oracle_mask: Optional[np.ndarray] = None,
        n_folds: int = 5,
        prompt_ids: Optional[List[str]] = None,
        covariates: Optional[np.ndarray] = None,
        quiet: bool = False,
        sample_weight: Optional[np.ndarray] = None,
    ) -> CalibrationResult:
        """Fit both global and cross-fitted calibration models.

        This method:
        1. Assigns fold IDs to all samples (canonical hash-based folds)
        2. Fits per-fold models f^(-k) for cross-fitted predictions
        3. Fits a global model f_all on all oracle data (for stable rewards)

        Args:
            judge_scores: Raw judge scores for all data
            oracle_labels: True labels for oracle subset
            oracle_mask: Boolean mask indicating which samples have oracle labels
            n_folds: Number of CV folds (auto-reduced when labels are scarce;
                see `resolve_n_folds`)
            prompt_ids: Optional prompt IDs for fold assignment. When None,
                stable row-index ids are synthesized so every caller shares
                the canonical hash-fold path.
            covariates: Optional covariate matrix (n_samples, n_covariates)
            quiet: Log fit progress at DEBUG instead of INFO. Used by the
                per-replicate bootstrap refits, which would otherwise emit
                thousands of identical "CV Calibration complete" lines.
            sample_weight: Optional positive per-label fit weights. Length
                must match either judge_scores (aligned with all rows) or
                oracle_labels (compact, in the caller's label order — kept
                aligned with the labels even when oracle_mask is an unsorted
                index array).

        Returns:
            CalibrationResult with both global and CV calibration
        """
        fit_log_level = logging.DEBUG if quiet else logging.INFO
        judge_scores = np.asarray(judge_scores)
        n_total = len(judge_scores)

        # Handle different input formats.  When an index-array oracle_mask
        # reorders the compact labels, the same permutation must be applied
        # to any compact sample_weight below.
        oracle_label_order: Optional[np.ndarray] = None
        if oracle_mask is not None:
            # Explicit mask provided (can be boolean array or indices)
            oracle_mask_array = np.asarray(oracle_mask)
            if oracle_labels is None:
                raise ValueError("oracle_labels required when oracle_mask provided")
            oracle_labels = np.asarray(oracle_labels)

            # Check if mask is indices (all signed/unsigned integer widths)
            # or boolean.  Floats are not accepted as either representation.
            if np.issubdtype(oracle_mask_array.dtype, np.integer) and not np.issubdtype(
                oracle_mask_array.dtype, np.bool_
            ):
                # Convert indices to a boolean mask, reordering labels to the
                # ascending-index order produced by boolean fancy-indexing.
                oracle_mask, oracle_labels, oracle_label_order = _index_mask_to_bool(
                    oracle_mask_array, oracle_labels, n_total
                )
            elif np.issubdtype(oracle_mask_array.dtype, np.bool_):
                # Already boolean
                oracle_mask = oracle_mask_array.astype(bool)
                if oracle_mask.ndim != 1 or len(oracle_mask) != n_total:
                    raise ValueError(
                        f"Boolean oracle_mask must have shape ({n_total},), got "
                        f"{oracle_mask.shape}."
                    )
                if len(oracle_labels) != int(np.sum(oracle_mask)):
                    raise ValueError(
                        f"oracle_labels length ({len(oracle_labels)}) must match "
                        f"the number of True values in oracle_mask "
                        f"({int(np.sum(oracle_mask))})."
                    )
            else:
                raise ValueError(
                    "oracle_mask must be a boolean mask or an integer index array"
                )

            oracle_scores = judge_scores[oracle_mask]
            oracle_y = oracle_labels  # oracle_labels is already compact
        elif oracle_labels is not None and len(oracle_labels) < n_total:
            n_oracle = len(oracle_labels)
            oracle_scores = judge_scores[:n_oracle]
            oracle_y = np.asarray(oracle_labels)
            oracle_mask = np.zeros(n_total, dtype=bool)
            oracle_mask[:n_oracle] = True
        elif oracle_labels is not None:
            oracle_labels = np.asarray(oracle_labels)
            if len(oracle_labels) != n_total:
                raise ValueError(
                    f"oracle_labels length ({len(oracle_labels)}) must match "
                    f"judge_scores length ({n_total}) or be shorter for partial labeling"
                )
            oracle_scores = judge_scores
            oracle_y = oracle_labels
            oracle_mask = np.ones(n_total, dtype=bool)
        else:
            raise ValueError(
                "oracle_labels is required for calibration. "
                "Provide oracle labels for at least a subset of samples."
            )

        n_oracle = len(oracle_y)
        self.oracle_coverage = (
            n_oracle / n_total
        )  # Store for calibration-uncertainty checks

        # Resolve prompt clusters before choosing K. Cross-fitting requires
        # independent held-out clusters, not merely a sufficient row count.
        if prompt_ids is None:
            prompt_ids = [f"row_{i}" for i in range(n_total)]
        if len(prompt_ids) != n_total:
            raise ValueError(
                f"prompt_ids length ({len(prompt_ids)}) must match "
                f"judge_scores length ({n_total})"
            )
        self._prompt_ids = prompt_ids

        oracle_prompt_ids = [prompt_ids[i] for i in np.where(oracle_mask)[0]]
        n_oracle_clusters = len(set(oracle_prompt_ids))
        n_folds = resolve_n_folds(n_oracle, n_folds, n_oracle_clusters)
        self._n_folds = n_folds

        if n_oracle_clusters < 4:
            raise ValueError(
                f"Too few unique oracle prompt clusters ({n_oracle_clusters}) for "
                "cross-fitted calibration. Need at least 4 independent prompt "
                "clusters; repeated labels within one prompt do not create "
                "independent folds."
            )
        if n_oracle_clusters < n_folds * 2:
            raise ValueError(
                f"Too few unique oracle prompt clusters ({n_oracle_clusters}) for "
                f"{n_folds}-fold CV. Need at least {n_folds * 2} (2 clusters "
                "per fold)."
            )

        # Balanced assignment guarantees every requested fold exists.  Assign
        # all rows by prompt so repeated draws never cross fit against one
        # another.
        self._fold_ids = _balanced_cluster_folds(
            prompt_ids,
            n_folds,
            self.random_seed,
            balancing_prompt_ids=oracle_prompt_ids,
        )

        # Extract oracle fold IDs / covariates for the cross-fitted models
        oracle_fold_ids = self._fold_ids[oracle_mask]
        self._oracle_fold_ids = np.asarray(oracle_fold_ids, dtype=int)

        oracle_covariates = None
        if covariates is not None:
            if len(covariates) != n_total:
                raise ValueError(
                    f"Covariates length ({len(covariates)}) must match judge_scores length ({n_total})"
                )
            oracle_covariates = covariates[oracle_mask]

        oracle_sample_weight: Optional[np.ndarray] = None
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.ndim != 1:
                raise ValueError("sample_weight must be one-dimensional")
            if len(weights) == n_total:
                oracle_sample_weight = weights[oracle_mask]
            elif len(weights) == n_oracle:
                # Compact weights arrive in the caller's label order. If an
                # index-array oracle_mask reordered the labels, reorder the
                # weights identically so every (label, weight) pair stays
                # intact.
                if oracle_label_order is not None:
                    weights = weights[oracle_label_order]
                oracle_sample_weight = weights
            else:
                raise ValueError(
                    f"sample_weight length ({len(weights)}) must match either "
                    f"judge_scores ({n_total}) or oracle labels ({n_oracle})."
                )
            if not np.all(np.isfinite(oracle_sample_weight)) or np.any(
                oracle_sample_weight <= 0
            ):
                raise ValueError("sample_weight values must be finite and positive")

        self._fit_judge_scores = np.asarray(oracle_scores, dtype=float).copy()
        self._fit_oracle_labels = np.asarray(oracle_y, dtype=float).copy()
        self._fit_prompt_ids = list(oracle_prompt_ids)
        self._fit_covariates = (
            None
            if oracle_covariates is None
            else np.asarray(oracle_covariates, dtype=float).copy()
        )
        self._fit_sample_weight = (
            None
            if oracle_sample_weight is None
            else np.asarray(oracle_sample_weight, dtype=float).copy()
        )

        # Step 2: Fit the single cross-fitted implementation (per-fold models
        # plus the full model for inference) for whatever mode was requested.
        from .flexible_calibrator import FlexibleCalibrator

        logger.log(fit_log_level, f"Calibration mode: {self.calibration_mode}")
        logger.log(
            fit_log_level,
            f"Fitting FlexibleCalibrator with {n_oracle} oracle samples",
        )
        self._flexible_calibrator = FlexibleCalibrator(
            mode=self.calibration_mode,
            random_seed=self.random_seed,
            covariate_names=self.covariate_names,
        )
        self._flexible_calibrator.fit(
            oracle_scores,
            oracle_y,
            oracle_fold_ids,
            oracle_covariates,
            sample_weight=oracle_sample_weight,
        )
        self.selected_mode = self._flexible_calibrator.selected_mode

        # Log selected mode if auto was used
        if self.calibration_mode == "auto":
            logger.log(
                fit_log_level, f"Auto-calibration selected: {self.selected_mode}"
            )
            if self.selected_mode == "two_stage":
                logger.log(
                    fit_log_level,
                    "  → Non-monotone relationship detected, using flexible calibration",
                )
            else:
                logger.log(
                    fit_log_level,
                    "  → Monotone relationship confirmed, using standard calibration",
                )

        # Get calibrated scores using the full model (no folds for inference)
        # Clip to [0,1] to ensure rewards stay in valid range
        calibrated_scores = np.clip(
            self._flexible_calibrator.predict(
                judge_scores, folds=None, covariates=covariates
            ),
            0.0,
            1.0,
        )

        if self.selected_mode == "monotone":
            self._warn_if_constant_monotone_fit(oracle_scores, oracle_y)

        # Compute diagnostics with both global and OOF predictions
        oracle_calibrated = calibrated_scores[oracle_mask]
        rmse = np.sqrt(
            np.average(
                (oracle_calibrated - oracle_y) ** 2,
                weights=oracle_sample_weight,
            )
        )
        coverage_01 = np.average(
            np.abs(oracle_calibrated - oracle_y) <= 0.1,
            weights=oracle_sample_weight,
        )

        # Record the training support for coverage badges (boundary cards)
        self._store_oracle_ranges(oracle_scores, oracle_calibrated)

        # Compute OOF diagnostics for oracle points
        oracle_oof = np.clip(
            self._flexible_calibrator.predict(
                oracle_scores, oracle_fold_ids, oracle_covariates
            ),
            0.0,
            1.0,
        )

        rmse_oof = float(
            np.sqrt(
                np.average((oracle_oof - oracle_y) ** 2, weights=oracle_sample_weight)
            )
        )
        coverage_01_oof = float(
            np.average(
                np.abs(oracle_oof - oracle_y) <= 0.1,
                weights=oracle_sample_weight,
            )
        )

        # Add information about calibration mode to log message
        if self.calibration_mode == "auto":
            mode_str = f" [{self.selected_mode} via auto]"
        else:
            mode_str = f" [{self.calibration_mode}]"

        logger.log(
            fit_log_level,
            f"CV Calibration complete{mode_str}: {n_oracle} oracle samples, {n_folds} folds, "
            f"RMSE={rmse:.3f} (OOF: {rmse_oof:.3f}), "
            f"coverage@0.1={coverage_01:.1%} (OOF: {coverage_01_oof:.1%})",
        )

        self._calibration_info = {
            "rmse": float(rmse),
            "coverage_at_01": float(coverage_01),
            "n_oracle_labels": int(n_oracle),
            "oof_rmse": float(rmse_oof),
            "oof_coverage_at_01": float(coverage_01_oof),
        }

        return CalibrationResult(
            calibrated_scores=calibrated_scores,
            calibration_rmse=float(rmse),
            coverage_at_01=float(coverage_01),
            n_oracle=n_oracle,
            calibrator=self,
            fold_ids=self._fold_ids,
            oof_rmse=rmse_oof,
            oof_coverage_at_01=coverage_01_oof,
        )

    def get_calibration_info(self) -> Dict[str, Any]:
        """Fit-time calibration quality metrics for estimator diagnostics.

        CalibratedDirectEstimator._build_diagnostics guards on this method
        to populate calibration_rmse / calibration_coverage /
        n_oracle_labels — it used to be unimplemented, leaving those
        diagnostics fields always None.

        Returns:
            Dict with "rmse", "coverage_at_01" (P(|pred - oracle| <= 0.1)
            on the oracle slice), "n_oracle_labels", "oof_rmse", and
            "oof_coverage_at_01". Empty dict before fitting.
        """
        return dict(self._calibration_info)

    def get_fold_models_for_oua(self) -> Dict[int, Any]:
        """Get fold models for the oracle jackknife used in calibration-aware inference.

        This method provides a unified interface to access fold models regardless of
        the calibration mode (monotone, two_stage, auto) being used.

        Returns:
            Dictionary of fold_id -> model. NOTE: two-stage fold models expect
            the RANK INDEX, not raw judge scores — route predictions through
            `predict_oof`, which applies the mode-appropriate transform.
            Empty dict if no fold models available.
        """
        if self._flexible_calibrator is None:
            return {}
        return self._flexible_calibrator.fold_models()

    def predict_oof(
        self,
        judge_scores: np.ndarray,
        fold_ids: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Out-of-fold predictions using cross-fitted models.

        This is the single OOF entry point: it applies the mode-appropriate
        transform (isotonic for monotone; g(S) -> ECDF rank -> isotonic for
        two-stage) using the per-fold models fitted by `fit_cv`.

        Args:
            judge_scores: Judge scores to calibrate
            fold_ids: Fold assignment for each score
            covariates: Optional covariate matrix (n_samples, n_covariates)

        Returns:
            Cross-fitted calibrated scores

        Raises:
            RuntimeError: If `fit_cv` has not been called.
            ValueError: If `fold_ids` contains folds with no fitted model
                (zero-filling or silently substituting the full model would
                fabricate reward predictions).
        """
        if self._flexible_calibrator is None:
            raise RuntimeError("Must call fit_cv before predict_oof")

        if (
            covariates is not None
            and (self.selected_mode or self.calibration_mode) == "monotone"
        ):
            raise ValueError(
                "Covariates provided but calibrator was fitted in monotone mode without covariate support"
            )

        judge_scores = np.asarray(judge_scores)
        fold_ids = np.asarray(fold_ids)

        # Fold ids without a fitted model must fail loudly: falling back to
        # the full model would silently break the out-of-fold contract.
        fitted_models = self.get_fold_models_for_oua()
        fitted_folds = sorted(fitted_models.keys())
        unknown_folds = [int(f) for f in np.unique(fold_ids) if f not in fitted_models]
        if unknown_folds:
            raise ValueError(
                f"predict_oof received fold ids {unknown_folds} with no fitted "
                f"model; fitted folds are {fitted_folds}. Fold ids must match "
                f"those used in fit_cv."
            )

        return np.clip(
            self._flexible_calibrator.predict(judge_scores, fold_ids, covariates),
            0.0,
            1.0,
        )
