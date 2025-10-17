# cje/core/mrdr.py
"""
MRDR estimator: policy-specific, cross-fitted weighted isotonic outcome models.

This estimator properly inherits from DREstimator, using the base DR infrastructure
while supporting policy-specific weighted outcome models.
"""

from __future__ import annotations
from typing import Dict, Optional, Any, List, cast
import logging
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .dr_base import DREstimator
from .outcome_models import BaseOutcomeModel
from ..data.precomputed_sampler import PrecomputedSampler
from ..data.models import EstimationResult
from ..data.fresh_draws import FreshDrawDataset

logger = logging.getLogger(__name__)


class WeightedIsotonicOutcomeModel(BaseOutcomeModel):
    """Weighted isotonic outcome model for MRDR.

    Extends BaseOutcomeModel to support sample weights in isotonic regression.
    """

    def __init__(self, n_folds: int = 5, calibrator: Optional[Any] = None):
        super().__init__(n_folds)
        self.sample_weights: Optional[np.ndarray] = None
        self.calibrator = calibrator  # Note: kept as 'calibrator' since this is internal to WeightedIsotonicOutcomeModel
        self._promptid_to_fold: Dict[str, int] = {}  # Store for MRDR to access

    def set_weights(self, weights: np.ndarray) -> None:
        """Set the sample weights for training."""
        self.sample_weights = weights

    def _get_fold_fit_kwargs(self, train_mask: np.ndarray) -> dict:
        """Get fold-specific kwargs for training (subsets sample weights)."""
        if self.sample_weights is None:
            return {}
        # Subset weights to match training fold
        return {"sample_weight": self.sample_weights[train_mask]}

    def _fit_single_model(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: np.ndarray,
        judge_scores: np.ndarray,  # These will be pre-transformed by fit() if calibrator exists
        covariates: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Any:
        """Fit a weighted isotonic regression model on training data."""
        model = IsotonicRegression(out_of_bounds="clip")

        # covariates are already incorporated in transformation by fit() method

        # Use provided sample weights (already subset by _get_fold_fit_kwargs)
        if sample_weight is not None:
            model.fit(judge_scores, rewards, sample_weight=sample_weight)
        else:
            model.fit(judge_scores, rewards)
        return model

    def _predict_single_model(
        self,
        model: Any,
        prompts: List[str],
        responses: List[str],
        judge_scores: np.ndarray,  # These will be pre-transformed by predict() if calibrator exists
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict using the fitted isotonic model."""
        # covariates are already incorporated in transformation by predict() method
        predictions: np.ndarray = model.predict(judge_scores)
        return predictions

    def fit(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: np.ndarray,
        judge_scores: Optional[np.ndarray] = None,
        fold_ids: Optional[np.ndarray] = None,
        covariates: Optional[np.ndarray] = None,
        *,
        prompt_ids: Optional[List[str]] = None,
    ) -> None:
        """Fit with optional prompt_id tracking for cross-fitting."""
        # Store prompt_id to fold mapping if provided
        if fold_ids is not None and prompt_ids is not None:
            self._promptid_to_fold = {
                pid: int(fid) for pid, fid in zip(prompt_ids, fold_ids)
            }

        # Pre-compute transformed indices if calibrator is available
        if self.calibrator is not None and hasattr(self.calibrator, "index"):
            # Get OOF indices for all data at once, passing covariates
            transformed_scores = self.calibrator.index(
                judge_scores, fold_ids, covariates=covariates
            )
        else:
            transformed_scores = judge_scores

        # Call base class fit with transformed scores (covariates not needed after transformation)
        super().fit(
            prompts, responses, rewards, transformed_scores, fold_ids, covariates=None
        )

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
        # Note: covariates are included in the transformation when using two-stage calibration
        if self.calibrator is not None and hasattr(self.calibrator, "index"):
            # For prediction, use the ensemble index (folds=None)
            transformed_scores = self.calibrator.index(
                judge_scores, folds=None, covariates=covariates
            )
        else:
            transformed_scores = judge_scores

        # Call parent predict with transformed scores (covariates already incorporated)
        return super().predict(prompts, responses, transformed_scores, fold_ids)


class MRDREstimator(DREstimator):
    """MRDR estimator with policy-specific weighted isotonic outcome models.

    Implements Multiple Robust Doubly Robust (MRDR) estimation with separate
    weighted outcome models for each target policy. The weights (omega) for
    each policy's outcome model are derived from that policy's importance weights.

    The choice of omega weighting scheme significantly impacts model stability and
    performance. Based on empirical testing with real-world data, the default has been
    changed from "snips" to "w" for better stability and generalization.

    Args:
        sampler: PrecomputedSampler with calibrated rewards
        n_folds: Cross-fitting folds (default 5)
        omega_mode: Weighting scheme for MRDR regression. One of:
            - "w": |W| [default, recommended]
                * Most stable and balanced weighting
                * Avoids extreme weight concentration
                * Positive R² values consistently
                * Lower RMSE in outcome predictions
                * Top 10% of samples typically receive ~18% of weight mass
                * Use for most applications

            - "w2": W²
                * Moderate weight concentration
                * More emphasis on high-weight samples
                * Top 10% of samples typically receive ~36% of weight mass
                * Use when you want moderate concentration without extremes

            - "snips": (W - 1)²
                * Can lead to extreme weight concentration
                * Top 10% of samples can receive 80%+ of weight mass
                * Often produces negative R² values
                * Higher RMSE due to overfitting on few samples
                * Only use with low-variance, well-behaved weights
                * Original theoretical default, but empirically problematic

        min_sample_weight: Floor applied to ω to avoid degenerate 0-weight fits (default 1e-8)
        use_calibrated_weights: Use CalibratedIPS (default True)
        use_policy_specific_models: If True, fit separate weighted models per policy.
                                   If False, use single shared model (simplified version).
        **kwargs: Passed through to DREstimator

    Empirical Performance (Arena 10k, 50% oracle coverage):
        | Mode    | R² Range        | RMSE  | Top 10% Weight |
        |---------|-----------------|-------|----------------|
        | "w"     | 0.376 to 0.404  | 0.169 | 18.2%          |
        | "w2"    | 0.245 to 0.285  | 0.183 | 35.7%          |
        | "snips" | -0.355 to 0.034 | 0.224 | 84.1%          |

    Why the Default Changed:
        Originally, MRDR used "snips" based on theoretical properties with Hájek
        (mean-one) weights. However, empirical testing revealed severe issues:
        1. Weight Concentration: Small fraction of samples dominates training
        2. Negative R²: Models often worse than mean prediction
        3. Poor Generalization: Overfitting to few high-weight samples

        The "w" mode provides stable positive R² values, distributes weight more
        evenly, achieves lower prediction error, and generalizes better.

    Usage:
        # Using default (recommended)
        estimator = MRDREstimator(sampler, n_folds=5)  # Uses omega_mode="w"

        # Explicitly setting omega mode
        estimator = MRDREstimator(sampler, n_folds=5, omega_mode="w2")

    Monitoring:
        Check weight diagnostics to monitor concentration. If top 10% of samples
        receive >50% of weight mass, consider switching from "snips" to "w".
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        n_folds: int = 5,
        omega_mode: str = "w",
        min_sample_weight: float = 1e-8,
        use_calibrated_weights: bool = True,
        weight_mode: str = "hajek",
        use_policy_specific_models: bool = True,
        reward_calibrator: Optional[Any] = None,
        **kwargs: Any,
    ):
        if omega_mode not in {"snips", "w2", "w"}:
            raise ValueError(
                f"omega_mode must be one of ['snips','w2','w'], got {omega_mode}"
            )

        # Use standard isotonic as default (will be overridden if policy-specific)
        from .outcome_models import IsotonicOutcomeModel

        # Pass reward_calibrator for proper index transformation with two-stage calibration
        outcome_model = IsotonicOutcomeModel(
            n_folds=n_folds, calibrator=reward_calibrator
        )

        # Initialize DR base (which will pass calibrator to CalibratedIPS)
        super().__init__(
            sampler=sampler,
            outcome_model=outcome_model,
            n_folds=n_folds,
            use_calibrated_weights=use_calibrated_weights,
            weight_mode=weight_mode,
            reward_calibrator=reward_calibrator,
            **kwargs,
        )

        self.omega_mode = omega_mode
        self.min_sample_weight = min_sample_weight
        self.use_policy_specific_models = use_policy_specific_models

        # Store policy-specific models
        self._policy_models: Dict[str, WeightedIsotonicOutcomeModel] = {}

        if use_policy_specific_models:
            logger.info(
                f"MRDREstimator: Using policy-specific weighted models with omega_mode='{omega_mode}'"
            )
        else:
            logger.info(
                "MRDREstimator: Using simplified version with shared outcome model"
            )

    def _omega_from_weights(self, w: np.ndarray, mode: str) -> np.ndarray:
        """Compute MRDR regression weights ω from mean-one IPS weights W.

        The omega weights are computed from calibrated importance weights (W)
        which have mean 1.0. The weights are then:
        1. Transformed according to the omega mode
        2. Floored at min_sample_weight (default 1e-8) to avoid zero weights
        3. Used as sample weights in IsotonicRegression.fit()

        Args:
            w: Mean-one importance weights from CalibratedIPS
            mode: One of "w", "w2", or "snips"

        Returns:
            Omega weights for weighted isotonic regression
        """
        if mode == "snips":
            # Recommended with Hájek (mean-one) weights
            return (w - 1.0) ** 2
        if mode == "w2":
            return w**2
        if mode == "w":
            return np.asarray(np.abs(w))
        raise ValueError(f"Unknown omega_mode: {mode}")

    def fit(self) -> None:
        """Fit weight calibration and policy-specific weighted outcome models.

        For each target policy, fits a separate WeightedIsotonicOutcomeModel
        using omega weights derived from that policy's importance weights.
        """
        # First fit IPS weights using base class
        self.ips_estimator.fit()

        # Call estimate() to populate IPS diagnostics (needed for weight ESS, etc.)
        self._ips_result = self.ips_estimator.estimate()

        self._fitted = True

        if not self.use_policy_specific_models:
            # Use base implementation with shared model
            super().fit()
            return

        # Extract covariate names from reward_calibrator (same as dr_base._fit_outcome_model)
        covariate_names: List[str] = []
        if self.reward_calibrator is not None and hasattr(
            self.reward_calibrator, "covariate_names"
        ):
            covariate_names = self.reward_calibrator.covariate_names or []
        self._covariate_names = covariate_names

        # Build prompt_id -> fold map from dataset metadata (if available)
        # This ensures we reuse the same folds as reward calibration
        cv_map = {}
        if hasattr(self.sampler, "dataset") and self.sampler.dataset:
            cv_map = {
                str(s.prompt_id): int(s.metadata["cv_fold"])
                for s in self.sampler.dataset.samples
                if "cv_fold" in s.metadata and s.metadata["cv_fold"] is not None
            }
            if cv_map:
                logger.info(
                    f"Reusing calibration folds for MRDR: {len(cv_map)} samples with cv_fold metadata"
                )

        # Fit policy-specific weighted models
        for policy in self.sampler.target_policies:
            # Get IPS weights for this policy
            weights = self.get_weights(policy)
            if weights is None:
                logger.warning(f"No weights available for policy '{policy}'. Skipping.")
                continue

            # Compute omega weights for outcome model
            omega = self._omega_from_weights(weights, self.omega_mode)
            omega = np.maximum(
                omega, self.min_sample_weight
            )  # Floor to avoid zero weights

            # Get data for this policy
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                logger.warning(f"No data available for policy '{policy}'. Skipping.")
                continue

            # Extract arrays
            prompts = [d["prompt"] for d in data]
            responses = [d["response"] for d in data]
            rewards = np.array([d["reward"] for d in data], dtype=float)
            judge_scores = np.array([d.get("judge_score") for d in data], dtype=float)
            prompt_ids = [str(d.get("prompt_id")) for d in data]

            # Create and fit weighted model for this policy
            # Pass calibrator for proper index transformation with two-stage calibration
            model = WeightedIsotonicOutcomeModel(
                n_folds=self.n_folds, calibrator=self.reward_calibrator
            )
            model.set_weights(omega)

            # Extract covariates if available
            covariates_array = None
            if hasattr(self, "_covariate_names") and self._covariate_names:
                covariates_list = []
                for d in data:
                    sample_covariates = []
                    for cov_name in self._covariate_names:
                        cov_value = d.get(cov_name)
                        if cov_value is None:
                            raise ValueError(
                                f"Covariate '{cov_name}' not found for policy '{policy}'"
                            )
                        sample_covariates.append(float(cov_value))  # type: ignore[arg-type]
                    covariates_list.append(sample_covariates)
                covariates_array = np.array(covariates_list, dtype=float)

            # Use cv_map if available (from calibration), otherwise create new folds
            if cv_map:
                # Reuse folds from calibration for consistency
                fold_ids = np.array(
                    [cv_map.get(pid, 0) for pid in prompt_ids], dtype=int
                )
                model.fit(
                    prompts,
                    responses,
                    rewards,
                    judge_scores,
                    fold_ids,
                    covariates_array,
                    prompt_ids=prompt_ids,
                )
            else:
                # Create fold assignments if not provided
                from sklearn.model_selection import KFold

                kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
                fold_ids = np.zeros(len(prompts), dtype=int)
                for fold_idx, (_, test_idx) in enumerate(kf.split(prompts)):
                    fold_ids[test_idx] = fold_idx
                model.fit(
                    prompts,
                    responses,
                    rewards,
                    judge_scores,
                    fold_ids,
                    covariates_array,
                    prompt_ids=prompt_ids,
                )

            self._policy_models[policy] = model
            logger.debug(
                f"Fitted weighted outcome model for policy '{policy}' with omega_mode='{self.omega_mode}'"
            )

        # Store prompt_id to fold mapping if available
        if cv_map:
            # Use the cv_map we built from dataset metadata
            self._promptid_to_fold = cv_map
        elif self._policy_models:
            # Create from first policy model if available
            first_model = next(iter(self._policy_models.values()))
            if hasattr(first_model, "_promptid_to_fold"):
                self._promptid_to_fold = first_model._promptid_to_fold
            else:
                self._promptid_to_fold = {}
        else:
            self._promptid_to_fold = {}

        logger.info(
            f"MRDR fitted with {len(self._policy_models)} policy-specific models"
        )

    def estimate(self) -> EstimationResult:
        """Compute MRDR estimates using policy-specific weighted outcome models.

        Each policy uses its own outcome model trained with omega weights
        derived from that policy's importance weights.
        """
        if not self.use_policy_specific_models:
            # Use base implementation
            result = super().estimate()
            result.method = "mrdr"
            if result.metadata is None:
                result.metadata = {}
            result.metadata.update(
                {
                    "omega_mode": self.omega_mode,
                    "min_sample_weight": self.min_sample_weight,
                    "note": "Simplified MRDR using shared outcome model",
                }
            )
            return result

        self._validate_fitted()

        # Auto-load fresh draws if not already loaded
        self._auto_load_fresh_draws()

        estimates: List[float] = []
        standard_errors: List[float] = []
        n_samples_used: Dict[str, int] = {}

        # Store components for diagnostics
        self._dm_component: Dict[str, np.ndarray] = {}
        self._ips_correction: Dict[str, np.ndarray] = {}
        self._outcome_predictions: Dict[str, np.ndarray] = {}

        for policy in self.sampler.target_policies:
            # Check if we have a model for this policy
            if policy not in self._policy_models:
                logger.warning(f"No outcome model for policy '{policy}'. Using NaN.")
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            # Get fresh draws (required for DR)
            if policy not in self._fresh_draws:
                logger.warning(
                    f"No fresh draws for policy '{policy}'. Skipping DR estimation."
                )
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            fresh_dataset = self._fresh_draws[policy]

            # Get data and weights
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            weights = self.get_weights(policy)
            if weights is None or len(weights) != len(data):
                raise ValueError(f"Weight/data mismatch for policy '{policy}'")

            # Extract arrays
            rewards = np.array([d["reward"] for d in data], dtype=float)
            judge_scores = np.array([d.get("judge_score") for d in data], dtype=float)
            prompt_ids = [str(d.get("prompt_id")) for d in data]
            prompts = [d["prompt"] for d in data]
            responses = [d["response"] for d in data]

            # Extract covariates if using them
            logged_covariates = None
            if hasattr(self, "_covariate_names") and self._covariate_names:
                covariate_values = []
                for d in data:
                    sample_covariates = []
                    for cov_name in self._covariate_names:
                        cov_value = d.get(cov_name)
                        if cov_value is None:
                            raise ValueError(
                                f"Covariate '{cov_name}' not found or is None in data for policy '{policy}'"
                            )
                        try:
                            sample_covariates.append(float(cov_value))  # type: ignore[arg-type]
                        except (TypeError, ValueError) as e:
                            raise ValueError(
                                f"Covariate '{cov_name}' has non-numeric value: {e}"
                            )
                    covariate_values.append(sample_covariates)
                logged_covariates = np.array(covariate_values)

            # Get fold assignments
            if self._promptid_to_fold:
                fold_ids = np.array(
                    [self._promptid_to_fold.get(pid, 0) for pid in prompt_ids],
                    dtype=int,
                )
            else:
                fold_ids = np.zeros(len(prompt_ids), dtype=int)

            # Get policy-specific outcome model
            outcome_model = self._policy_models[policy]

            # Get predictions on logged data
            g_logged = outcome_model.predict(
                prompts, responses, judge_scores, fold_ids, covariates=logged_covariates
            )

            # Get predictions on fresh draws
            g_fresh_all = []
            for i, prompt_id in enumerate(prompt_ids):
                fresh_scores = fresh_dataset.get_scores_for_prompt_id(prompt_id)
                fresh_prompts = [prompts[i]] * len(fresh_scores)
                fresh_responses = [""] * len(fresh_scores)
                fresh_fold_ids = np.full(len(fresh_scores), fold_ids[i])

                # Get covariate for this prompt (if using covariates)
                fresh_covariates = None
                if logged_covariates is not None:
                    fresh_covariates = np.tile(
                        logged_covariates[i], (len(fresh_scores), 1)
                    )

                g_fresh_prompt = outcome_model.predict(
                    fresh_prompts,
                    fresh_responses,
                    fresh_scores,
                    fresh_fold_ids,
                    covariates=fresh_covariates,
                )
                g_fresh_all.append(g_fresh_prompt.mean())

            g_fresh = np.array(g_fresh_all)

            # Compute DR estimate with oracle augmentation
            dm_term = float(g_fresh.mean())
            ips_corr_base = weights * (rewards - g_logged)

            # IPS correction (no oracle augmentation)
            ips_corr_total = ips_corr_base
            ips_corr = float(np.mean(ips_corr_total))
            psi = dm_term + ips_corr

            # Store components for diagnostics
            self._dm_component[policy] = g_fresh
            self._ips_correction[policy] = ips_corr_total
            self._fresh_rewards[policy] = (
                rewards  # Store logged rewards for diagnostics
            )
            self._outcome_predictions[policy] = g_logged

            # Compute influence functions and standard error
            if_contrib = g_fresh + ips_corr_total - psi

            # IIC removed - use influence functions directly

            se = (
                float(np.std(if_contrib, ddof=1) / np.sqrt(len(if_contrib)))
                if len(if_contrib) > 1
                else 0.0
            )

            # Store influence functions (always needed for proper inference)
            self._influence_functions[policy] = if_contrib

            # Store sample indices for IF alignment in stacking (using parent's helper)
            self._store_sample_indices(policy, data)

            estimates.append(psi)
            standard_errors.append(se)
            n_samples_used[policy] = len(rewards)

            logger.info(
                f"MRDR[{policy}]: {psi:.4f} ± {se:.4f} (DM={dm_term:.4f}, IPS_corr={ips_corr:.4f})"
            )

        # Build DR diagnostics using stored components
        from ..diagnostics.dr import compute_dr_policy_diagnostics

        dr_diagnostics_per_policy: Dict[str, Dict[str, Any]] = {}
        for idx, policy in enumerate(self.sampler.target_policies):
            if policy not in self._dm_component or np.isnan(estimates[idx]):
                continue

            # Use base class helper for consistent diagnostic computation
            dr_diagnostics_per_policy[policy] = self._compute_policy_diagnostics(
                policy, estimates[idx]
            )

        # Build diagnostics
        diagnostics = self._build_dr_diagnostics(
            estimates=estimates,
            standard_errors=standard_errors,
            n_samples_used=n_samples_used,
            dr_diagnostics_per_policy=dr_diagnostics_per_policy,
            ips_diagnostics=(
                self.ips_estimator.get_diagnostics()
                if hasattr(self.ips_estimator, "get_diagnostics")
                else None
            ),
        )

        # Create result with MRDR metadata (no influence functions here - they're first-class)
        metadata = {
            "omega_mode": self.omega_mode,
            "min_sample_weight": self.min_sample_weight,
            "use_policy_specific_models": self.use_policy_specific_models,
            "n_policy_models": len(self._policy_models),
            "cross_fitted": True,
            "n_folds": self.n_folds,
        }

        # Add sample indices for IF alignment in stacking
        if hasattr(self, "_if_sample_indices"):
            metadata["if_sample_indices"] = self._if_sample_indices

        result = EstimationResult(
            estimates=np.array(estimates, dtype=float),
            standard_errors=np.array(standard_errors, dtype=float),
            n_samples_used=n_samples_used,
            method="mrdr",
            influence_functions=self._influence_functions,
            diagnostics=diagnostics,
            metadata=metadata,
        )

        # Apply OUA jackknife using base class method (inherits from DREstimator)
        self._apply_oua_jackknife(result)

        return result
