"""Oracle slice augmentation for honest confidence intervals.

This module implements the augmentation term that accounts for the uncertainty
in learning the judge→oracle calibration map f̂(S) from a finite oracle slice.
The augmentation restores first-order unbiasedness and provides honest CIs
that properly widen when the oracle slice is small.

Theory:
    The base estimator uses proxy outcome f̂(S) everywhere. The true target is E[W*Y].
    The gap is E[W*(Y - f̂(S))]. On the oracle slice, we can estimate this gap using:

    AUG = (L/p) * m(S) * (Y - f̂(S))

    where:
    - L ∈ {0,1} indicates oracle label presence
    - p = E[L] is the labeling probability (MCAR assumption)
    - m(S) = E[W|S] is the conditional mean of weights given score
    - Y is the true oracle label
    - f̂(S) is the calibrated proxy from judge scores

    This augmentation is unbiased for the gap and has controlled variance.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from sklearn.isotonic import IsotonicRegression
import logging

logger = logging.getLogger(__name__)


@dataclass
class OracleSliceConfig:
    """Configuration for oracle slice augmentation.

    By default, CJE uses OUA jackknife for oracle uncertainty (variance addition).
    This bias correction augmentation is optional and primarily used in TR-CPO
    under MAR or as an MCAR engineering fallback.

    Args:
        enable_augmentation: Whether to add the bias correction term (default False)
        enable_cross_fit: Whether to cross-fit m̂(S) and π̂(S)
        min_pi: Minimum value for π̂(S) to avoid division issues
        use_mar: Whether to model MAR labeling (vs assuming MCAR)
    """

    enable_augmentation: bool = (
        False  # Default to OUA jackknife only; TR-CPO enables explicitly
    )
    enable_cross_fit: bool = True
    min_pi: float = 0.01
    use_mar: bool = False  # Start with MCAR, add MAR in phase 2


class OracleSliceAugmentation:
    """Computes augmentation terms for oracle slice uncertainty.

    This handles:
    1. Estimating m̂(S) = E[W|S] via isotonic regression
    2. Computing augmentation: (L/p) * m̂(S) * (Y - f̂(S))
    3. Tracking diagnostics about the augmentation

    The augmentation corrects for the bias from using f̂(S) instead of Y,
    while the influence function properly accounts for the added variance.
    """

    def __init__(self, config: Optional[OracleSliceConfig] = None):
        """Initialize oracle slice augmentation.

        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or OracleSliceConfig()
        self._m_hat_cache: Dict[str, np.ndarray] = {}
        self._m_hat_models: Dict[str, Any] = {}
        self._diagnostics: Dict[str, Dict] = {}

    def fit_m_hat(
        self,
        weights: np.ndarray,
        judge_scores: np.ndarray,
        policy: str,
        cv_folds: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Estimate m̂(S) = E[W|S] via isotonic regression.

        Args:
            weights: The actual calibrated weights used by the estimator
            judge_scores: Judge scores S
            policy: Target policy name
            cv_folds: Optional fold assignments for cross-fitting

        Returns:
            m_hat: Estimated E[W|S], normalized to mean 1
        """
        if not self.config.enable_augmentation:
            return np.ones_like(weights)

        # Handle edge cases
        if len(weights) == 0 or len(judge_scores) == 0:
            logger.warning(f"Empty weights or scores for policy {policy}")
            return np.ones_like(weights)

        # Cross-fitted version if requested and folds available
        if self.config.enable_cross_fit and cv_folds is not None:
            m_hat = self._fit_m_hat_cross_fitted(weights, judge_scores, cv_folds)

            # Also fit global model for unlabeled rows
            iso_global = IsotonicRegression(out_of_bounds="clip")
            iso_global.fit(judge_scores, weights)
            self._m_hat_models[policy] = iso_global

            # Fill any missing values (unlabeled rows) with global fit
            unlabeled = cv_folds < 0
            # Only check isnan for float dtypes
            if np.issubdtype(cv_folds.dtype, np.floating):
                unlabeled |= np.isnan(cv_folds)
            if np.any(unlabeled):
                m_hat[unlabeled] = iso_global.predict(judge_scores[unlabeled])

        else:
            # Global fit only (simpler, used for IPS)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(judge_scores, weights)
            m_hat = iso.predict(judge_scores)
            self._m_hat_models[policy] = iso

        # Normalize to mean 1 (preserves unbiasedness)
        if m_hat.mean() > 0:
            m_hat = m_hat / m_hat.mean()
        else:
            logger.warning(f"m_hat has zero mean for policy {policy}")
            m_hat = np.ones_like(m_hat)

        # Cache for later use
        self._m_hat_cache[policy] = m_hat

        logger.debug(
            f"Fitted m̂(S) for policy {policy}: "
            f"mean={m_hat.mean():.3f}, std={m_hat.std():.3f}, "
            f"range=[{m_hat.min():.3f}, {m_hat.max():.3f}]"
        )

        return m_hat

    def _fit_m_hat_cross_fitted(
        self, weights: np.ndarray, judge_scores: np.ndarray, cv_folds: np.ndarray
    ) -> np.ndarray:
        """Fit m̂(S) using cross-fitting to avoid overfitting.

        Args:
            weights: Calibrated weights
            judge_scores: Judge scores
            cv_folds: Fold assignments (-1 for unlabeled)

        Returns:
            m_hat: Out-of-fold predictions
        """
        m_hat = np.zeros_like(weights)
        unique_folds = np.unique(cv_folds[cv_folds >= 0])

        if len(unique_folds) < 2:
            # Not enough folds for cross-fitting, use global
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(judge_scores, weights)
            return np.asarray(iso.predict(judge_scores))

        for fold in unique_folds:
            train_mask = (cv_folds >= 0) & (cv_folds != fold)
            test_mask = cv_folds == fold

            if train_mask.sum() > 0 and test_mask.sum() > 0:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(judge_scores[train_mask], weights[train_mask])
                m_hat[test_mask] = iso.predict(judge_scores[test_mask])

        return m_hat

    def compute_augmentation(
        self,
        policy: str,
        rewards: np.ndarray,  # f̂(S) - calibrated proxy outcomes
        data: List[Dict[str, Any]],
        dataset_samples: Optional[List[Any]] = None,
        oracle_field: str = "oracle_label",
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Compute augmentation vector and diagnostics.

        Args:
            policy: Target policy name
            rewards: Calibrated rewards f̂(S) for each sample
            data: List of data dictionaries with prompt_id
            dataset_samples: Original dataset samples with metadata
            oracle_field: Field name for oracle labels in metadata

        Returns:
            aug_vector: Per-sample augmentation values
            diagnostics: Dictionary with p_oracle, n_oracle, variance info
        """
        if not self.config.enable_augmentation:
            return np.zeros_like(rewards), {}

        if policy not in self._m_hat_cache:
            logger.warning(f"No m̂(S) fitted for policy {policy}, skipping augmentation")
            return np.zeros_like(rewards), {}

        m_hat = self._m_hat_cache[policy]

        # Extract oracle labels and mask
        oracle_labels, oracle_mask = self._extract_oracle_info(
            data, dataset_samples, oracle_field
        )

        # Check if we have any oracle labels
        p = oracle_mask.mean() if len(oracle_mask) > 0 else 0.0
        n_oracle = int(oracle_mask.sum())

        if p <= 0 or n_oracle == 0:
            logger.debug(f"No oracle labels for policy {policy}")
            return np.zeros_like(rewards), {
                "p_oracle": 0.0,
                "n_oracle": 0,
                "aug_mean": 0.0,
                "aug_var": 0.0,
                "slice_variance_share": 0.0,
            }

        # Compute augmentation: (L/p) * m̂(S) * (Y - f̂(S))
        # Note: For MCAR, π(S) = p (constant)
        residuals = np.where(oracle_mask, oracle_labels - rewards, 0.0)
        aug = (oracle_mask / p) * m_hat * residuals

        # Compute diagnostics
        aug_mean = float(aug.mean())
        aug_var = float(np.var(aug, ddof=1)) if len(aug) > 1 else 0.0

        # Store diagnostics
        diagnostics = {
            "p_oracle": float(p),
            "n_oracle": n_oracle,
            "aug_mean": aug_mean,
            "aug_var": aug_var,
            "slice_variance_share": 0.0,  # Will be computed by estimator
        }

        self._diagnostics[policy] = diagnostics

        logger.debug(
            f"Augmentation for {policy}: mean={aug_mean:.4f}, "
            f"var={aug_var:.4f}, p={p:.3f}, n_oracle={n_oracle}"
        )

        return aug, diagnostics

    def _extract_oracle_info(
        self,
        data: List[Dict[str, Any]],
        dataset_samples: Optional[List[Any]],
        oracle_field: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract oracle labels and mask from dataset.

        Args:
            data: List of data dictionaries with prompt_id
            dataset_samples: Original dataset samples
            oracle_field: Field name for oracle labels

        Returns:
            oracle_labels: Array of oracle values (0 where missing)
            oracle_mask: Boolean array indicating oracle presence
        """
        n = len(data)
        oracle_labels = np.zeros(n)
        oracle_mask = np.zeros(n, dtype=bool)

        if dataset_samples is None:
            # Try to extract from data directly
            for i, d in enumerate(data):
                if "metadata" in d and oracle_field in d["metadata"]:
                    val = d["metadata"][oracle_field]
                    if val is not None:
                        oracle_labels[i] = float(val)
                        oracle_mask[i] = True
        else:
            # Build mapping from prompt_id to metadata
            metadata_map = {}
            for sample in dataset_samples:
                if hasattr(sample, "prompt_id") and hasattr(sample, "metadata"):
                    pid = str(sample.prompt_id)
                    metadata_map[pid] = sample.metadata

            # Extract oracle labels for each data point
            for i, d in enumerate(data):
                prompt_id = str(d.get("prompt_id", ""))
                if prompt_id in metadata_map:
                    metadata = metadata_map[prompt_id]
                    if oracle_field in metadata and metadata[oracle_field] is not None:
                        oracle_labels[i] = float(metadata[oracle_field])
                        oracle_mask[i] = True

        return oracle_labels, oracle_mask

    def get_diagnostics(self, policy: Optional[str] = None) -> Dict[str, Any]:
        """Get augmentation diagnostics.

        Args:
            policy: Specific policy to get diagnostics for (None = all)

        Returns:
            Dictionary of diagnostics
        """
        if policy is not None:
            return self._diagnostics.get(policy, {})
        return self._diagnostics.copy()
