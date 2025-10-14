"""Judge score calibration using isotonic regression (AutoCal-R).

This module implements AutoCal-R (Automatic Calibration for Rewards), which
calibrates cheap LLM judge scores to actual business KPIs/oracle labels using
monotonic regression on a labeled subset. AutoCal-R automatically selects
between monotone and two-stage calibration based on the relationship structure.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, Literal, TYPE_CHECKING, Any
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from dataclasses import dataclass
import logging
import hashlib

if TYPE_CHECKING:
    from .flexible_calibrator import FlexibleCalibrator

logger = logging.getLogger(__name__)


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
    """Calibrate judge scores to oracle labels using isotonic regression (AutoCal-R core).

    This is the core implementation of AutoCal-R (Automatic Calibration for Rewards),
    which provides mean-preserving, largely monotone mapping from judge scores to
    oracle labels with automatic mode selection.

    Args:
        random_seed: Random seed for reproducibility
        balance_oracle_folds: Whether to balance oracle samples across folds
        calibration_mode: AutoCal-R mode - 'auto' (default), 'monotone', or 'two_stage'
    """

    def __init__(
        self,
        random_seed: int = 42,
        balance_oracle_folds: bool = True,
        calibration_mode: Optional[Literal["monotone", "two_stage", "auto"]] = "auto",
        covariate_names: Optional[List[str]] = None,
    ):
        """Initialize judge calibrator.

        Args:
            random_seed: Random seed for reproducibility
            balance_oracle_folds: Whether to balance oracle samples across folds
            calibration_mode: Calibration method to use:
                - 'auto' (default): Automatically select based on cross-validation
                - 'monotone': Force standard isotonic regression
                - 'two_stage': Force flexible two-stage calibration
                - None: Use monotone (for backward compatibility)
            covariate_names: Optional list of covariate names to extract from Sample.metadata
                for use in two-stage calibration (e.g., ["response_length", "domain"])
        """
        self.random_seed = random_seed
        self.balance_oracle_folds = (
            balance_oracle_folds  # Whether to balance oracle samples across folds
        )
        # None defaults to 'monotone' for backward compatibility
        self.calibration_mode = (
            calibration_mode if calibration_mode is not None else "monotone"
        )
        self.covariate_names = covariate_names or []
        # Store selected mode (for auto, this gets updated after selection)
        self.selected_mode: Optional[str] = (
            None if calibration_mode == "auto" else calibration_mode
        )
        self._final_calibrator: Optional[IsotonicRegression] = None
        self._flexible_calibrator: Optional["FlexibleCalibrator"] = (
            None  # Will hold FlexibleCalibrator if needed
        )
        self._fold_models: Dict[int, IsotonicRegression] = {}
        self._fold_ids: Optional[np.ndarray] = None
        self._n_folds: int = 5
        self._prompt_ids: Optional[List[str]] = (
            None  # Store prompt_ids for fold assignment
        )

    def fit_transform(
        self,
        judge_scores: np.ndarray,
        oracle_labels: Optional[np.ndarray] = None,
        oracle_mask: Optional[np.ndarray] = None,
        covariates: Optional[np.ndarray] = None,
    ) -> CalibrationResult:
        """Calibrate judge scores using oracle labels.

        Args:
            judge_scores: Raw judge scores for all data
            oracle_labels: True labels for oracle subset (if oracle_mask provided)
            oracle_mask: Boolean mask indicating which samples have oracle labels
            covariates: Optional covariate matrix (n_samples, n_covariates) for two-stage mode

        Returns:
            CalibrationResult with calibrated scores and diagnostics

        Example:
            # With explicit mask
            calibrator = JudgeCalibrator()
            result = calibrator.fit_transform(
                judge_scores=all_scores,
                oracle_labels=oracle_values,
                oracle_mask=has_oracle_label
            )

            # Or with implicit mask (oracle_labels shorter than judge_scores)
            result = calibrator.fit_transform(
                judge_scores=all_scores,
                oracle_labels=oracle_subset_labels
            )
        """
        judge_scores = np.asarray(judge_scores)
        n_total = len(judge_scores)

        # Handle different input formats
        if oracle_mask is not None:
            # Explicit mask provided (can be boolean array or indices)
            oracle_mask_array = np.asarray(oracle_mask)
            if oracle_labels is None:
                raise ValueError("oracle_labels required when oracle_mask provided")
            oracle_labels = np.asarray(oracle_labels)

            # Check if mask is indices (integers) or boolean
            if oracle_mask_array.dtype in [np.int32, np.int64, int]:
                # Convert indices to boolean mask
                bool_mask = np.zeros(n_total, dtype=bool)
                bool_mask[oracle_mask_array] = True
                oracle_mask = bool_mask
            else:
                # Already boolean
                oracle_mask = oracle_mask_array.astype(bool)

            # Extract oracle subset
            oracle_scores = judge_scores[oracle_mask]
            oracle_y = oracle_labels

        elif oracle_labels is not None and len(oracle_labels) < n_total:
            # Oracle labels provided for first n samples
            n_oracle = len(oracle_labels)
            oracle_scores = judge_scores[:n_oracle]
            oracle_y = np.asarray(oracle_labels)
            oracle_mask = np.zeros(n_total, dtype=bool)
            oracle_mask[:n_oracle] = True

        elif oracle_labels is not None:
            # All data has oracle labels (no holdout)
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
            # No oracle labels provided
            raise ValueError(
                "oracle_labels is required for calibration. "
                "Provide oracle labels for at least a subset of samples."
            )

        n_oracle = len(oracle_y)

        if n_oracle < 10:
            raise ValueError(f"Too few oracle samples ({n_oracle}). Need at least 10.")

        # Initialize calibrated scores
        calibrated_scores = np.copy(judge_scores)

        # Extract oracle covariates if provided
        oracle_covariates = None
        if covariates is not None:
            if len(covariates) != n_total:
                raise ValueError(
                    f"Covariates length ({len(covariates)}) must match judge_scores length ({n_total})"
                )
            oracle_covariates = covariates[oracle_mask]

        # Use appropriate calibration based on mode
        if self.calibration_mode != "monotone":
            # Use flexible calibrator for auto/two_stage modes
            from .flexible_calibrator import FlexibleCalibrator

            self._flexible_calibrator = FlexibleCalibrator(
                mode=self.calibration_mode,
                random_seed=self.random_seed,
                covariate_names=self.covariate_names,
            )
            # Create simple fold split for flexible calibrator
            oracle_folds = np.arange(len(oracle_y)) % 5
            self._flexible_calibrator.fit(
                oracle_scores, oracle_y, oracle_folds, oracle_covariates
            )

            # Log selected mode if auto was used
            if self.calibration_mode == "auto":
                selected = self._flexible_calibrator.selected_mode
                self.selected_mode = selected  # Store for metadata
                logger.info(f"Auto-calibration selected: {selected}")

            # Transform all scores
            calibrated_scores = np.clip(
                self._flexible_calibrator.predict(judge_scores, covariates=covariates),
                0.0,
                1.0,
            )

            # Store isotonic for compatibility (if available)
            if hasattr(self._flexible_calibrator, "iso_reg"):
                self._final_calibrator = self._flexible_calibrator.iso_reg
        else:
            # Standard monotone calibration
            self._final_calibrator = IsotonicRegression(out_of_bounds="clip")
            self._final_calibrator.fit(oracle_scores, oracle_y)

            # Apply calibration to all samples using full model
            # Clip to [0,1] to ensure rewards stay in valid range even if oracle labels exceed bounds
            calibrated_scores = np.clip(
                self._final_calibrator.predict(judge_scores), 0.0, 1.0
            )

        # Compute diagnostics on oracle subset
        oracle_calibrated = calibrated_scores[oracle_mask]
        rmse = np.sqrt(np.mean((oracle_calibrated - oracle_y) ** 2))
        coverage_01 = np.mean(np.abs(oracle_calibrated - oracle_y) <= 0.1)

        # Log summary
        logger.info(
            f"Calibration complete: {n_oracle} oracle samples, "
            f"RMSE={rmse:.3f}, coverage@0.1={coverage_01:.1%}"
        )

        return CalibrationResult(
            calibrated_scores=calibrated_scores,
            calibration_rmse=float(rmse),
            coverage_at_01=float(coverage_01),
            n_oracle=n_oracle,
            calibrator=self,
            fold_ids=self._fold_ids,
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
        if self._flexible_calibrator is not None:
            # Use flexible calibrator for predictions
            return np.clip(
                self._flexible_calibrator.predict(
                    judge_scores, folds=None, covariates=covariates
                ),
                0.0,
                1.0,
            )

        if self._final_calibrator is None:
            raise RuntimeError("Calibrator must be fitted before prediction")

        if covariates is not None:
            raise ValueError(
                "Covariates provided but calibrator was fitted in monotone mode without covariate support"
            )

        # Predict and clip to [0,1] to ensure rewards stay in valid range
        result = self._final_calibrator.predict(np.asarray(judge_scores))
        return np.clip(np.asarray(result), 0.0, 1.0)

    def fit_cv(
        self,
        judge_scores: np.ndarray,
        oracle_labels: Optional[np.ndarray] = None,
        oracle_mask: Optional[np.ndarray] = None,
        n_folds: int = 5,
        prompt_ids: Optional[List[str]] = None,
        covariates: Optional[np.ndarray] = None,
    ) -> CalibrationResult:
        """Fit both global and cross-fitted calibration models.

        This method:
        1. Fits a global model f_all on all oracle data (for stable rewards)
        2. Fits per-fold models f^(-k) for cross-fitted predictions (for DR)
        3. Assigns fold IDs to all samples (labeled by CV, unlabeled by hash)

        Args:
            judge_scores: Raw judge scores for all data
            oracle_labels: True labels for oracle subset
            oracle_mask: Boolean mask indicating which samples have oracle labels
            n_folds: Number of CV folds
            prompt_ids: Optional prompt IDs for fold assignment
            covariates: Optional covariate matrix (n_samples, n_covariates)

        Returns:
            CalibrationResult with both global and CV calibration
        """
        judge_scores = np.asarray(judge_scores)
        n_total = len(judge_scores)
        self._n_folds = n_folds

        # Handle different input formats (same as fit_transform)
        if oracle_mask is not None:
            # Explicit mask provided (can be boolean array or indices)
            oracle_mask_array = np.asarray(oracle_mask)
            if oracle_labels is None:
                raise ValueError("oracle_labels required when oracle_mask provided")
            oracle_labels = np.asarray(oracle_labels)

            # Check if mask is indices (integers) or boolean
            if oracle_mask_array.dtype in [np.int32, np.int64, int]:
                # Convert indices to boolean mask
                bool_mask = np.zeros(n_total, dtype=bool)
                bool_mask[oracle_mask_array] = True
                oracle_mask = bool_mask
            else:
                # Already boolean
                oracle_mask = oracle_mask_array.astype(bool)

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
        if n_oracle < n_folds * 2:
            raise ValueError(
                f"Too few oracle samples ({n_oracle}) for {n_folds}-fold CV. "
                f"Need at least {n_folds * 2}."
            )

        # Step 1: Assign fold IDs to all samples first (unified approach)
        if prompt_ids is not None:
            # Use the unified fold system when prompt_ids are available
            # ALWAYS use hash-based folds to ensure consistency across all components
            from ..data.folds import get_folds_for_prompts

            self._prompt_ids = prompt_ids
            # Always use hash-based folds for consistency
            self._fold_ids = get_folds_for_prompts(
                prompt_ids, n_folds, self.random_seed
            )
        else:
            # Fallback to old system for backward compatibility
            self._fold_ids = np.zeros(n_total, dtype=int)

            # Labeled samples: assign by KFold
            oracle_indices = np.where(oracle_mask)[0]
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
            for fold_id, (_, test_idx) in enumerate(kf.split(oracle_indices)):
                fold_samples = oracle_indices[test_idx]
                self._fold_ids[fold_samples] = fold_id

            # Unlabeled samples: assign deterministically by stable hash
            unlabeled_mask = ~oracle_mask
            unlabeled_indices = np.where(unlabeled_mask)[0]
            if len(unlabeled_indices) > 0:

                def _fold_for_idx(i: int, seed: int, n_folds: int) -> int:
                    """Stable hash-based fold assignment."""
                    h = hashlib.blake2b(f"{i}-{seed}".encode(), digest_size=2)
                    return int.from_bytes(h.digest(), "big") % n_folds

                for idx in unlabeled_indices:
                    self._fold_ids[idx] = _fold_for_idx(
                        int(idx), self.random_seed, n_folds
                    )

        # Extract oracle fold IDs for use in flexible calibrator
        oracle_fold_ids = self._fold_ids[oracle_mask]

        # Extract oracle covariates if provided
        oracle_covariates = None
        if covariates is not None:
            if len(covariates) != n_total:
                raise ValueError(
                    f"Covariates length ({len(covariates)}) must match judge_scores length ({n_total})"
                )
            oracle_covariates = covariates[oracle_mask]

        # Step 2: Fit global model
        if self.calibration_mode != "monotone":
            logger.info(f"Calibration mode: {self.calibration_mode}")

            # Use flexible calibration
            from .flexible_calibrator import FlexibleCalibrator

            # Fit flexible calibrator
            logger.info(f"Fitting FlexibleCalibrator with {n_oracle} oracle samples")
            self._flexible_calibrator = FlexibleCalibrator(
                mode=self.calibration_mode,
                random_seed=self.random_seed,
                covariate_names=self.covariate_names,
            )
            self._flexible_calibrator.fit(
                oracle_scores, oracle_y, oracle_fold_ids, oracle_covariates
            )

            # Log selected mode if auto was used
            if self.calibration_mode == "auto":
                selected = self._flexible_calibrator.selected_mode
                self.selected_mode = selected  # Store for metadata
                logger.info(f"Auto-calibration selected: {selected}")
                if selected == "two_stage":
                    logger.info(
                        "  → Non-monotone relationship detected, using flexible calibration"
                    )
                else:
                    logger.info(
                        "  → Monotone relationship confirmed, using standard calibration"
                    )

            # Get calibrated scores using the full model (no folds for inference)
            calibrated_scores = np.clip(
                self._flexible_calibrator.predict(
                    judge_scores, folds=None, covariates=covariates
                ),
                0.0,
                1.0,
            )
        else:
            logger.info("Calibration mode: monotone (standard isotonic regression)")
            # Use standard monotone calibration
            self._final_calibrator = IsotonicRegression(out_of_bounds="clip")
            self._final_calibrator.fit(oracle_scores, oracle_y)
            # Clip to [0,1] to ensure rewards stay in valid range even if oracle labels exceed bounds
            calibrated_scores = np.clip(
                self._final_calibrator.predict(judge_scores), 0.0, 1.0
            )

        # Step 3: Fit per-fold models
        if self._flexible_calibrator is not None:
            # Flexible calibrator already has per-fold models fitted
            self._fold_models = {}  # Will use flexible calibrator for predictions
        else:
            # Standard isotonic per-fold models
            self._fold_models = {}
            for fold_id in range(n_folds):
                # Get training indices (all oracle samples NOT in this fold)
                train_mask = oracle_mask & (self._fold_ids != fold_id)
                train_scores = judge_scores[train_mask]
                # Get corresponding oracle labels for training samples
                oracle_fold_mask = self._fold_ids[oracle_mask] != fold_id
                train_labels = oracle_y[oracle_fold_mask]

                if len(train_scores) > 0:
                    fold_model = IsotonicRegression(out_of_bounds="clip")
                    fold_model.fit(train_scores, train_labels)
                    self._fold_models[fold_id] = fold_model
                else:
                    # Fallback to global model if not enough data
                    self._fold_models[fold_id] = self._final_calibrator

        # Compute diagnostics with both global and OOF predictions
        oracle_calibrated = calibrated_scores[oracle_mask]
        rmse = np.sqrt(np.mean((oracle_calibrated - oracle_y) ** 2))
        coverage_01 = np.mean(np.abs(oracle_calibrated - oracle_y) <= 0.1)

        # Compute OOF diagnostics for oracle points
        oracle_oof = np.empty_like(oracle_y)
        if self._flexible_calibrator is not None:
            # Get OOF predictions from flexible calibrator
            oracle_fold_ids = self._fold_ids[oracle_mask]
            oracle_oof = np.clip(
                self._flexible_calibrator.predict(
                    oracle_scores, oracle_fold_ids, oracle_covariates
                ),
                0.0,
                1.0,
            )
        else:
            # Standard isotonic OOF predictions
            for fold_id, model in self._fold_models.items():
                mask = self._fold_ids[oracle_mask] == fold_id
                if np.any(mask):
                    # Clip predictions to [0,1]
                    oracle_oof[mask] = np.clip(
                        model.predict(oracle_scores[mask]), 0.0, 1.0
                    )

        rmse_oof = float(np.sqrt(np.mean((oracle_oof - oracle_y) ** 2)))
        coverage_01_oof = float(np.mean(np.abs(oracle_oof - oracle_y) <= 0.1))

        # Add information about calibration mode to log message
        mode_str = ""
        if self._flexible_calibrator is not None:
            if self.calibration_mode == "auto":
                mode_str = f" [{self._flexible_calibrator.selected_mode} via auto]"
            else:
                mode_str = f" [{self.calibration_mode}]"
        else:
            mode_str = " [monotone]"

        logger.info(
            f"CV Calibration complete{mode_str}: {n_oracle} oracle samples, {n_folds} folds, "
            f"RMSE={rmse:.3f} (OOF: {rmse_oof:.3f}), "
            f"coverage@0.1={coverage_01:.1%} (OOF: {coverage_01_oof:.1%})"
        )

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

    def predict_all(self, judge_scores: np.ndarray) -> np.ndarray:
        """Predict using global model f_all (stable for rewards).

        Args:
            judge_scores: Judge scores to calibrate

        Returns:
            Globally calibrated scores
        """
        return self.predict(judge_scores)

    def index(
        self,
        judge_scores: np.ndarray,
        folds: Optional[np.ndarray] = None,
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get the index used for isotonic regression.

        This returns the appropriate index for outcome modeling:
        - For monotone calibration: returns judge_scores unchanged
        - For two-stage calibration: returns the transformed index

        Args:
            judge_scores: Judge scores to transform
            folds: Optional fold assignments for OOF transformation
            covariates: Optional covariate matrix (n_samples, n_covariates)

        Returns:
            Index values suitable for isotonic regression
        """
        if self._flexible_calibrator is not None:
            return self._flexible_calibrator.index(judge_scores, folds, covariates)
        else:
            # Standard monotone calibration uses judge scores directly
            if covariates is not None:
                raise ValueError("Covariates not supported in monotone mode")
            return judge_scores

    def has_fold_models(self) -> bool:
        """Check if fold models are available for OUA.

        Returns:
            True if fold models exist, False otherwise
        """
        return len(self.get_fold_models_for_oua()) > 0

    def get_fold_models_for_oua(self) -> Dict[int, Any]:
        """Get fold models for OUA jackknife, handling both standard and flexible calibration.

        This method provides a unified interface to access fold models regardless of
        the calibration mode (monotone, two_stage, auto) being used.

        Returns:
            Dictionary of fold_id -> model suitable for predict() calls
            Empty dict if no fold models available
        """
        # If using FlexibleCalibrator (auto/two_stage modes)
        if self._flexible_calibrator is not None:
            # Check which mode was actually selected
            selected_mode = getattr(self._flexible_calibrator, "selected_mode", None)

            if selected_mode == "two_stage":
                # For two-stage, the isotonic models are in _iso_models
                models = getattr(self._flexible_calibrator, "_iso_models", {})
            else:
                # For monotone (or as fallback), use _monotone_models
                models = getattr(self._flexible_calibrator, "_monotone_models", {})

            # Ensure we return a dict even if None
            return models if models is not None else {}

        # Standard isotonic calibration
        return self._fold_models if self._fold_models is not None else {}

    def predict_oof(
        self,
        judge_scores: np.ndarray,
        fold_ids: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Out-of-fold predictions using cross-fitted models.

        Args:
            judge_scores: Judge scores to calibrate
            fold_ids: Fold assignment for each score
            covariates: Optional covariate matrix (n_samples, n_covariates)

        Returns:
            Cross-fitted calibrated scores
        """
        if self._flexible_calibrator is not None:
            # Use flexible calibrator for OOF predictions
            return np.clip(
                self._flexible_calibrator.predict(judge_scores, fold_ids, covariates),
                0.0,
                1.0,
            )

        if not self._fold_models:
            raise RuntimeError("Must call fit_cv before predict_oof")

        if covariates is not None:
            raise ValueError(
                "Covariates provided but calibrator was fitted in monotone mode without covariate support"
            )

        judge_scores = np.asarray(judge_scores)
        fold_ids = np.asarray(fold_ids)

        predictions = np.zeros_like(judge_scores)
        for fold_id, model in self._fold_models.items():
            fold_mask = fold_ids == fold_id
            if np.any(fold_mask):
                # Clip predictions to [0,1] to ensure valid rewards
                predictions[fold_mask] = np.clip(
                    model.predict(judge_scores[fold_mask]), 0.0, 1.0
                )

        return np.clip(predictions, 0.0, 1.0)  # Extra safety clip


def calibrate_judge_scores(
    judge_scores: np.ndarray,
    oracle_labels: np.ndarray,
    oracle_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Convenience function for judge calibration.

    Args:
        judge_scores: Raw judge scores for all data
        oracle_labels: True labels for oracle subset
        oracle_mask: Optional boolean mask for oracle samples

    Returns:
        Tuple of (calibrated_scores, diagnostics_dict)

    Example:
        # Calibrate judge scores with 25% oracle labels
        cal_scores, stats = calibrate_judge_scores(
            judge_scores=all_judge_scores,
            oracle_labels=oracle_subset_labels[:1000]  # First 1000 have labels
        )

        print(f"Calibration RMSE: {stats['rmse']:.3f}")
        print(f"Coverage: {stats['coverage']:.1%}")
    """
    calibrator = JudgeCalibrator()
    result = calibrator.fit_transform(judge_scores, oracle_labels, oracle_mask)

    diagnostics = {
        "rmse": result.calibration_rmse,
        "coverage": result.coverage_at_01,
        "n_oracle": result.n_oracle,
    }

    return result.calibrated_scores, diagnostics
