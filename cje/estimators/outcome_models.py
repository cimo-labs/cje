"""Outcome models for Doubly Robust estimation.

Outcome models predict E[R|X,A,S] and are used in the direct method
component of DR estimators. They must be cross-fitted to maintain orthogonality.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import numpy as np
import logging

if TYPE_CHECKING:
    from ..calibration.judge import JudgeCalibrator

logger = logging.getLogger(__name__)


class BaseOutcomeModel(ABC):
    """Abstract base class for cross-fitted outcome models in DR estimation.

    All outcome models must support cross-fitted prediction where each
    sample is predicted using a model trained on other folds.
    Subclasses only need to implement the single-model training and prediction.
    """

    def __init__(self, n_folds: int = 5):
        """Initialize the outcome model.

        Args:
            n_folds: Number of folds for cross-fitting
        """
        if n_folds < 2:
            raise ValueError(f"n_folds must be at least 2, got {n_folds}")

        self.n_folds = n_folds
        self.fold_models: Dict[int, Any] = {}
        self.fold_assignments: Optional[np.ndarray] = None
        self._fitted = False

    @abstractmethod
    def _fit_single_model(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: np.ndarray,
        judge_scores: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> Any:
        """Fit a single model on training data.

        Args:
            prompts: Training prompts
            responses: Training responses
            rewards: Training rewards (calibrated)
            judge_scores: Training judge scores
            covariates: Optional covariate matrix (n_samples, n_covariates)

        Returns:
            A fitted model object
        """
        pass

    @abstractmethod
    def _predict_single_model(
        self,
        model: Any,
        prompts: List[str],
        responses: List[str],
        judge_scores: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Make predictions using a fitted model.

        Args:
            model: A model returned by _fit_single_model
            prompts: Prompts to predict on
            responses: Responses to predict on
            judge_scores: Judge scores to predict on
            covariates: Optional covariate matrix (n_samples, n_covariates)

        Returns:
            Predicted rewards
        """
        pass

    def fit(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: np.ndarray,
        judge_scores: Optional[np.ndarray] = None,
        fold_ids: Optional[np.ndarray] = None,
        covariates: Optional[np.ndarray] = None,
    ) -> None:
        """Fit cross-fitted models on logged data."""
        if fold_ids is None:
            raise ValueError("fold_ids is required for cross-fitted outcome models")

        if judge_scores is None:
            raise ValueError("judge_scores is required for outcome models")

        # Validate inputs
        n = len(prompts)
        if (
            len(responses) != n
            or len(rewards) != n
            or len(judge_scores) != n
            or len(fold_ids) != n
        ):
            raise ValueError(
                f"Input length mismatch: prompts={len(prompts)}, responses={len(responses)}, "
                f"rewards={len(rewards)}, judge_scores={len(judge_scores)}, fold_ids={len(fold_ids)}"
            )

        # Validate covariates if provided
        if covariates is not None and len(covariates) != n:
            raise ValueError(
                f"Covariates length mismatch: expected {n}, got {len(covariates)}"
            )

        # Remap fold IDs to be sequential 0..K-1 for the subset
        original_fold_ids = fold_ids.astype(int)
        unique_folds = sorted(np.unique(original_fold_ids))

        # Create mapping from original to sequential
        fold_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_folds)}
        self.fold_assignments = np.vectorize(fold_id_map.__getitem__)(original_fold_ids)

        # Store reverse mapping for potential debugging
        self._fold_id_map = fold_id_map
        self._reverse_fold_map = {v: k for k, v in fold_id_map.items()}

        # Adjust n_folds to actual number of unique folds
        if len(unique_folds) != self.n_folds:
            logger.info(
                f"Adjusting n_folds from {self.n_folds} to {len(unique_folds)} based on data"
            )
            self.n_folds = len(unique_folds)

        # Train a model for each fold on the other folds (using remapped IDs)
        for fold in range(self.n_folds):
            train_mask = self.fold_assignments != fold

            if not np.any(train_mask):
                raise ValueError(f"No training data for fold {fold}")

            # Cast to numpy array for type safety
            train_mask_arr = np.asarray(train_mask)
            train_prompts = [p for i, p in enumerate(prompts) if train_mask_arr[i]]
            train_responses = [r for i, r in enumerate(responses) if train_mask_arr[i]]
            train_rewards = rewards[train_mask]
            train_scores = judge_scores[train_mask]

            # Extract fold-specific covariates if provided
            train_covariates = None
            if covariates is not None:
                train_covariates = covariates[train_mask]

            # Allow subclasses to provide fold-specific kwargs (e.g., sample weights)
            extra_kwargs = {}
            if hasattr(self, "_get_fold_fit_kwargs"):
                extra_kwargs = self._get_fold_fit_kwargs(train_mask_arr)

            model = self._fit_single_model(
                train_prompts,
                train_responses,
                train_rewards,
                train_scores,
                covariates=train_covariates,
                **extra_kwargs,
            )
            self.fold_models[fold] = model

            logger.debug(
                f"Fitted model for fold {fold} on {len(train_prompts)} samples"
            )

        self._fitted = True
        logger.info(
            f"{self.__class__.__name__} fitted with {self.n_folds} cross-fitted models"
        )

    def predict(
        self,
        prompts: List[str],
        responses: List[str],
        judge_scores: Optional[np.ndarray] = None,
        fold_ids: Optional[np.ndarray] = None,
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict using cross-fitted models."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if judge_scores is None:
            raise ValueError("judge_scores required for prediction")

        # Use provided fold_ids or fall back to stored ones
        if fold_ids is None:
            if self.fold_assignments is None:
                raise ValueError("fold_ids must be provided or set during fit()")
            if len(prompts) != len(self.fold_assignments):
                raise ValueError(
                    f"Using stored fold_assignments but length mismatch: "
                    f"prompts={len(prompts)}, fold_assignments={len(self.fold_assignments)}"
                )
            fold_ids = self.fold_assignments

        # Validate inputs
        n = len(prompts)
        if len(responses) != n or len(judge_scores) != n or len(fold_ids) != n:
            raise ValueError(
                f"Input length mismatch: prompts={len(prompts)}, responses={len(responses)}, "
                f"judge_scores={len(judge_scores)}, fold_ids={len(fold_ids)}"
            )

        # Validate covariates if provided
        if covariates is not None and len(covariates) != n:
            raise ValueError(
                f"Covariates length mismatch: expected {n}, got {len(covariates)}"
            )

        fold_ids = fold_ids.astype(int)

        # Remap fold IDs if we have a mapping (from fit())
        if hasattr(self, "_fold_id_map") and self._fold_id_map:
            # Map incoming fold IDs to compact range
            mapped_fold_ids = np.array(
                [self._fold_id_map.get(fid, None) for fid in fold_ids]
            )

            # Check for unmapped fold IDs
            if None in mapped_fold_ids:
                unmapped = set(fid for fid in fold_ids if fid not in self._fold_id_map)
                raise ValueError(
                    f"Unmapped fold IDs: {sorted(unmapped)}. "
                    f"Known mappings: {self._fold_id_map}"
                )

            fold_ids = mapped_fold_ids.astype(int)

        # Guard against unknown fold IDs
        unknown_folds = set(np.unique(fold_ids)) - set(self.fold_models.keys())
        if unknown_folds:
            raise ValueError(
                f"Unknown fold ids in predict(): {sorted(unknown_folds)}. "
                f"Available folds: {sorted(self.fold_models.keys())}"
            )

        predictions = np.zeros(n)

        # Predict each fold using its out-of-fold model
        for fold in self.fold_models:
            fold_mask = fold_ids == fold
            if not fold_mask.any():
                continue

            fold_prompts = [p for i, p in enumerate(prompts) if fold_mask[i]]
            fold_responses = [r for i, r in enumerate(responses) if fold_mask[i]]
            fold_scores = judge_scores[fold_mask]

            # Extract fold-specific covariates if provided
            fold_covariates = None
            if covariates is not None:
                fold_covariates = covariates[fold_mask]

            fold_predictions = self._predict_single_model(
                self.fold_models[fold],
                fold_prompts,
                fold_responses,
                fold_scores,
                covariates=fold_covariates,
            )

            # Validate prediction shape
            if len(fold_predictions) != fold_mask.sum():
                raise ValueError(
                    f"Model returned {len(fold_predictions)} predictions but expected {fold_mask.sum()}"
                )

            predictions[fold_mask] = fold_predictions

        return predictions


class IsotonicOutcomeModel(BaseOutcomeModel):
    """Cross-fitted isotonic outcome model for DR estimation.

    This model uses g(x,a,s) = f^(-k)(z) where z is the calibrator's index
    (either raw judge scores for monotone calibration or transformed index
    for two-stage calibration), and f^(-k) is the isotonic regression learned
    with the k-th fold held out for cross-fitting.

    The isotonic models are trained fresh during the DR fit process,
    ensuring proper cross-fitting for orthogonality.
    """

    def __init__(self, n_folds: int = 5, calibrator: Optional[Any] = None):
        """Initialize isotonic outcome model.

        Args:
            n_folds: Number of cross-fitting folds (default 5)
            calibrator: Optional calibrator with index() method for transforming scores
        """
        super().__init__(n_folds)
        self.calibrator = calibrator

    def fit(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: np.ndarray,
        judge_scores: Optional[np.ndarray] = None,
        fold_ids: Optional[np.ndarray] = None,
        covariates: Optional[np.ndarray] = None,
    ) -> None:
        """Fit cross-fitted models with proper index transformation."""
        if fold_ids is None:
            raise ValueError("fold_ids is required for cross-fitted outcome models")

        if judge_scores is None:
            raise ValueError("judge_scores is required for outcome models")

        # Pre-compute transformed indices if calibrator is available
        if self.calibrator is not None and hasattr(self.calibrator, "index"):
            # Get OOF indices for all data at once, passing covariates if available
            transformed_scores = self.calibrator.index(
                judge_scores, fold_ids, covariates=covariates
            )
        else:
            transformed_scores = judge_scores

        # Store original judge_scores for later use, replace with transformed
        self._original_judge_scores = judge_scores
        judge_scores_to_use = transformed_scores

        # Call parent fit with transformed scores (covariates not needed after transformation)
        super().fit(
            prompts, responses, rewards, judge_scores_to_use, fold_ids, covariates=None
        )

    def _fit_single_model(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: np.ndarray,
        judge_scores: np.ndarray,  # These are already transformed indices
        covariates: Optional[np.ndarray] = None,
    ) -> Any:
        """Fit an isotonic regression model on training data."""
        from sklearn.isotonic import IsotonicRegression

        # judge_scores are already transformed by fit() method
        # covariates are already incorporated in the transformation
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(judge_scores, rewards)
        return model

    def predict(
        self,
        prompts: List[str],
        responses: List[str],
        judge_scores: Optional[np.ndarray] = None,
        fold_ids: Optional[np.ndarray] = None,
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict using cross-fitted models with proper index transformation."""
        if judge_scores is None:
            raise ValueError("judge_scores required for prediction")

        # Transform judge scores if calibrator is available
        if self.calibrator is not None and hasattr(self.calibrator, "index"):
            # For prediction, use the ensemble index (folds=None), passing covariates if available
            transformed_scores = self.calibrator.index(
                judge_scores, folds=None, covariates=covariates
            )
        else:
            transformed_scores = judge_scores

        # Call parent predict with transformed scores (covariates not needed after transformation)
        return super().predict(
            prompts, responses, transformed_scores, fold_ids, covariates=None
        )

    def _predict_single_model(
        self,
        model: Any,
        prompts: List[str],
        responses: List[str],
        judge_scores: np.ndarray,  # These are already transformed indices
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict using the fitted isotonic model."""
        # judge_scores are already transformed by predict() method
        # covariates are already incorporated in the transformation
        predictions: np.ndarray = model.predict(judge_scores)
        return predictions


class LinearOutcomeModel(BaseOutcomeModel):
    """Example custom outcome model using linear regression.

    This demonstrates how users can implement their own outcome models
    by extending BaseOutcomeModel.
    """

    def __init__(self, n_folds: int = 5, alpha: float = 1.0):
        """Initialize linear outcome model.

        Args:
            n_folds: Number of cross-fitting folds
            alpha: Regularization strength for Ridge regression
        """
        super().__init__(n_folds)
        self.alpha = alpha

    def _fit_single_model(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: np.ndarray,
        judge_scores: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> Any:
        """Fit a Ridge regression model on features."""
        from sklearn.linear_model import Ridge

        features = self._extract_features(prompts, responses, judge_scores, covariates)
        # Use fit_intercept=False since we add bias column manually
        model = Ridge(alpha=self.alpha, fit_intercept=False)
        model.fit(features, rewards)
        return model

    def _predict_single_model(
        self,
        model: Any,
        prompts: List[str],
        responses: List[str],
        judge_scores: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict using the fitted Ridge model."""
        features = self._extract_features(prompts, responses, judge_scores, covariates)
        predictions = model.predict(features)
        clipped: np.ndarray = np.clip(predictions, 0, 1)
        return clipped

    def _extract_features(
        self,
        prompts: List[str],
        responses: List[str],
        judge_scores: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Extract simple features from inputs."""
        # Length features
        prompt_lengths = np.array([len(p.split()) for p in prompts]).reshape(-1, 1)
        response_lengths = np.array([len(r.split()) for r in responses]).reshape(-1, 1)

        # Judge scores
        scores = judge_scores.reshape(-1, 1)

        # Start with basic features
        feature_matrix = np.hstack([prompt_lengths, response_lengths, scores])

        # Add user-provided covariates if available
        if covariates is not None:
            if covariates.ndim == 1:
                covariates = covariates.reshape(-1, 1)
            feature_matrix = np.hstack([feature_matrix, covariates])

        # Add bias term
        bias = np.ones((len(prompts), 1))
        final_features: np.ndarray = np.hstack([feature_matrix, bias])

        return final_features


class CalibratorBackedOutcomeModel(BaseOutcomeModel):
    """Outcome model that reuses cross-fitted reward calibrators.

    Instead of refitting a model on (S, R=f_all(S)), this model reuses
    the cross-fitted calibrators f^(-k) that were already trained during
    reward calibration. This preserves orthogonality and avoids redundant
    computation.

    This is the recommended default for DR estimation when using isotonic
    calibration for rewards.
    """

    def __init__(self, reward_calibrator: "JudgeCalibrator", n_folds: int = 5):
        """Initialize with a fitted reward calibrator.

        Args:
            reward_calibrator: A fitted JudgeCalibrator with cross-fitted models
            n_folds: Number of folds (should match calibrator's n_folds)
        """
        super().__init__(n_folds)
        self.calibrator = reward_calibrator

        # Verify calibrator has cross-fitted models
        if (
            not hasattr(reward_calibrator, "has_fold_models")
            or not reward_calibrator.has_fold_models()
        ):
            raise ValueError(
                "CalibratorBackedOutcomeModel requires a calibrator fitted with "
                "fit_cv(). Use enable_cross_fit=True in calibrate_dataset()."
            )

        if reward_calibrator._n_folds != n_folds:
            logger.warning(
                f"Calibrator has {reward_calibrator._n_folds} folds but outcome model "
                f"requested {n_folds}. Using calibrator's fold count."
            )
            self.n_folds = reward_calibrator._n_folds

    def _fit_single_model(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: np.ndarray,
        judge_scores: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> Any:
        """No training needed - reuse calibrator's models."""
        # Just return a reference to the calibrator
        return self.calibrator

    def _predict_single_model(
        self,
        model: Any,
        prompts: List[str],
        responses: List[str],
        judge_scores: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict using the calibrator's cross-fitted models.

        This should never be called directly since we override fit() and predict()
        to use the calibrator's predict_oof() method directly.
        """
        # This is a fallback that shouldn't be reached
        return judge_scores

    def fit(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: np.ndarray,
        judge_scores: Optional[np.ndarray] = None,
        fold_ids: Optional[np.ndarray] = None,
        covariates: Optional[np.ndarray] = None,
    ) -> None:
        """Fit by storing fold assignments (no model training needed).

        Args:
            prompts: Training prompts
            responses: Training responses
            rewards: Training rewards (not used)
            judge_scores: Training judge scores
            fold_ids: Pre-assigned fold IDs from calibration
            covariates: Covariates (stored for later use)
        """
        n_samples = len(prompts)

        # Store covariates for later use in predictions
        self._covariates = covariates

        if fold_ids is not None:
            # Use provided fold assignments
            self.fold_assignments = np.asarray(fold_ids)
        else:
            # Try to get from calibrator
            if (
                hasattr(self.calibrator, "_fold_ids")
                and self.calibrator._fold_ids is not None
            ):
                if len(self.calibrator._fold_ids) == n_samples:
                    self.fold_assignments = self.calibrator._fold_ids
                else:
                    # Strict error to avoid accidental in-fold leakage
                    raise ValueError(
                        f"CalibratorBackedOutcomeModel requires exact fold_ids when "
                        f"calibrator's stored fold_ids don't match the data subset. "
                        f"Calibrator has {len(self.calibrator._fold_ids)} fold IDs but "
                        f"we have {n_samples} samples. Pass explicit fold_ids from the "
                        f"'cv_fold' metadata to avoid accidental in-fold predictions."
                    )
            else:
                # No fold IDs available - require explicit ones
                raise ValueError(
                    "CalibratorBackedOutcomeModel requires fold_ids to be provided. "
                    "Either pass them explicitly or ensure the calibrator was fitted "
                    "with fit_cv() and has matching data size."
                )

        self._fitted = True
        logger.info(
            f"CalibratorBackedOutcomeModel ready: {n_samples} samples, "
            f"{self.n_folds} folds (reusing calibrator models)"
        )

    def predict(
        self,
        prompts: List[str],
        responses: List[str],
        judge_scores: Optional[np.ndarray] = None,
        fold_ids: Optional[np.ndarray] = None,
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict using cross-fitted calibration models.

        Args:
            prompts: Prompts to predict on
            responses: Responses to predict on
            judge_scores: Judge scores to calibrate
            fold_ids: Fold assignments for each sample
            covariates: Covariates for prediction

        Returns:
            Cross-fitted predictions using f^(-k)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        if fold_ids is None:
            # Use stored fold assignments if they match
            if self.fold_assignments is not None and len(self.fold_assignments) == len(
                prompts
            ):
                fold_ids = self.fold_assignments
            else:
                # For new data, require explicit fold assignments
                raise ValueError(
                    "fold_ids required for CalibratorBackedOutcomeModel.predict() "
                    "when predicting on new data to avoid accidental in-fold predictions. "
                    "Provide fold assignments from the calibration phase."
                )

        if judge_scores is None:
            raise ValueError("judge_scores required for prediction")

        # Use calibrator's out-of-fold predictions, passing covariates if available
        predictions = self.calibrator.predict_oof(
            judge_scores, fold_ids, covariates=covariates
        )

        return predictions
