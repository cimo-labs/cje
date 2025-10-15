"""Base class for Doubly Robust (DR) estimators.

DR estimators combine a direct method (outcome model) with an IPS correction
to achieve better bias-variance tradeoffs and double robustness properties.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union, cast
import logging
import dataclasses
from pathlib import Path

from .calibrated_ips import CalibratedIPS
from .base_estimator import BaseCJEEstimator
from .outcome_models import IsotonicOutcomeModel, CalibratorBackedOutcomeModel
from ..data.models import EstimationResult
from ..diagnostics import DRDiagnostics, IPSDiagnostics
from ..data.precomputed_sampler import PrecomputedSampler
from ..data.fresh_draws import FreshDrawDataset, validate_fresh_draws
from ..diagnostics.dr import (
    compute_dr_policy_diagnostics,
    compute_orthogonality_score,
    compute_dm_ips_decomposition,
)

logger = logging.getLogger(__name__)


class DREstimator(BaseCJEEstimator):
    """Base class for Doubly Robust estimators with flexible weight method.

    Key insight: DR = Direct Method + IPS correction
    This class uses CalibratedIPS for the importance weighting component,
    which can operate in calibrated or raw mode for flexibility.

    The DR formula from the paper (equation 13):
    V_DR(π') = (1/n) Σ [g(X_i, A'_i, S'_i) + W_i * (R_i - g(X_i, A_i, S_i))]

    Where:
    - g is the outcome model (uses cross-fitted isotonic calibration)
    - A'_i are pre-generated fresh draws from the target policy
    - S'_i are pre-evaluated judge scores on fresh draws
    - W_i are the importance weights (raw or calibrated)
    - R_i are the rewards on logged data (from full calibration model)

    Args:
        sampler: PrecomputedSampler with logged data
        outcome_model: Outcome model for predictions (default: IsotonicOutcomeModel)
        n_folds: Number of cross-fitting folds (default 5)
        use_calibrated_weights: If True, use SIMCal calibration; if False, use raw weights (default True)
        weight_mode: "hajek" for mean-one normalized weights, "raw" for unnormalized (default "hajek")
        reward_calibrator: Optional reward calibrator for CalibratorBackedOutcomeModel (always use if available)
        **kwargs: Additional arguments passed to the base class (e.g., oracle_slice_config)

    Monte Carlo Variance Handling:
        When there's only M=1 fresh draw per prompt, we cannot estimate within-prompt variance directly.
        The estimator automatically applies a conservative upper bound:
        - Uses total variance across single draws as upper bound for within-prompt variance
        - Conservative because mixture variance >= average within-component variance
        - Respects [0,1] scale constraint (variance <= 0.25)
        - For mixed cases (some M>=2, some M=1), combines exact computation with upper bound

        This ensures confidence intervals properly reflect uncertainty even with limited fresh draws.

    Note: The reward_calibrator (for reward calibration) is independent of use_calibrated_weights (for weight
    calibration). DR estimators should receive the reward_calibrator whenever oracle coverage < 100%.
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        outcome_model: Optional[Any] = None,
        n_folds: int = 5,
        use_calibrated_weights: bool = True,
        weight_mode: str = "hajek",
        reward_calibrator: Optional[Any] = None,
        random_seed: int = 42,
        run_diagnostics: bool = True,
        **kwargs: Any,
    ):
        # Extract oua_jackknife BEFORE passing kwargs to parent
        if "oua_jackknife" in kwargs:
            oua_jackknife = kwargs.pop("oua_jackknife", True)  # Use pop to remove it
        else:
            oracle_config = kwargs.get("oracle_slice_config", {})
            if isinstance(oracle_config, dict):
                oua_jackknife = oracle_config.get("oua_jackknife", True)
            else:
                oua_jackknife = True

        # Pop var_cap if present - it's for weight calibration, not for base class
        # This is passed by ablation code but not used by DR estimators
        kwargs.pop("var_cap", None)

        # Pass OUA parameters to base class
        super().__init__(
            sampler=sampler,
            run_diagnostics=run_diagnostics,
            diagnostic_config=None,  # Will use defaults
            reward_calibrator=reward_calibrator,
            oua_jackknife=oua_jackknife,
            **kwargs,  # Passes remaining kwargs (e.g., oracle_slice_config)
        )

        self.n_folds = n_folds
        self.use_calibrated_weights = use_calibrated_weights
        self.random_seed = random_seed

        # Initialize the IPS estimator with appropriate mode
        self.ips_estimator: CalibratedIPS
        # Pass reward_calibrator to CalibratedIPS for DR-aware direction selection if calibrating
        ips_kwargs: Dict[str, Any] = {
            "calibrate_weights": use_calibrated_weights,
            "weight_mode": weight_mode,
            "run_diagnostics": run_diagnostics,
            "suppress_overlap_warnings": True,  # DR handles poor overlap, suppress IPS warnings
            "n_outer_folds": n_folds,  # Align with DR outcome folds
            "outer_cv_seed": random_seed,  # Align fold seeds for one-way clustering
        }
        if use_calibrated_weights and reward_calibrator is not None:
            ips_kwargs["reward_calibrator"] = reward_calibrator

        self.ips_estimator = CalibratedIPS(sampler, **ips_kwargs)

        logger.info(
            f"Using CalibratedIPS with calibrate_weights={use_calibrated_weights} for importance weights in DR"
        )

        # Oracle augmentation removed - using OUA jackknife only

        # Choose default outcome model based on available reward_calibrator
        if outcome_model is None:
            has_fold_models = False
            if reward_calibrator is not None and hasattr(
                reward_calibrator, "has_fold_models"
            ):
                has_fold_models = reward_calibrator.has_fold_models()

            if has_fold_models:
                # We have a cross-fitted reward_calibrator, use it for outcome model
                logger.info(
                    "Using CalibratorBackedOutcomeModel (reusing calibration models)"
                )
                outcome_model = CalibratorBackedOutcomeModel(
                    reward_calibrator, n_folds=n_folds  # type: ignore[arg-type]
                )
            else:
                # Check if any samples have cv_fold metadata
                has_cv_fold = any(
                    "cv_fold" in s.metadata
                    for s in sampler.dataset.samples[
                        : min(10, len(sampler.dataset.samples))
                    ]
                )

                if has_cv_fold:
                    logger.warning(
                        "Samples have cv_fold metadata but no reward_calibrator provided. "
                        "Consider passing reward_calibrator from calibrate_dataset() for optimal DR."
                    )

                # Fall back to standard isotonic outcome model
                # Pass reward_calibrator if available for proper index transformation
                outcome_model = IsotonicOutcomeModel(
                    n_folds=n_folds,
                    calibrator=reward_calibrator,  # Note: IsotonicOutcomeModel still uses 'calibrator' param name
                )
        self.outcome_model = outcome_model

        # Storage for fresh draws (added via add_fresh_draws)
        self._fresh_draws: Dict[str, FreshDrawDataset] = {}
        self._outcome_fitted = False

        # Store components for diagnostics
        self._dm_component: Dict[str, np.ndarray] = {}
        self._ips_correction: Dict[str, np.ndarray] = {}
        self._fresh_rewards: Dict[str, np.ndarray] = {}
        self._outcome_predictions: Dict[str, np.ndarray] = {}
        self._orthogonality_scores: Dict[str, Dict[str, Any]] = {}
        self._dm_ips_decompositions: Dict[str, Dict[str, Any]] = {}
        # Per-policy SE diagnostics for downstream CI construction
        self._se_diagnostics: Dict[str, Dict[str, Any]] = {}

        # Note: Fold assignments are now computed on-demand from prompt_ids
        # This ensures correct folds even for filtered data

    def add_fresh_draws(self, policy: str, fresh_draws: FreshDrawDataset) -> None:
        """Add pre-generated fresh draws for a target policy.

        Fresh draws must have complete coverage - every logged sample with
        a valid importance weight for this policy must have corresponding
        fresh draws.

        Args:
            policy: Target policy name
            fresh_draws: Pre-generated fresh draw dataset

        Raises:
            ValueError: If fresh draws don't have complete coverage
        """
        # Validate coverage
        validate_fresh_draws(fresh_draws, self.sampler.dataset, policy)

        # Store the fresh draws
        self._fresh_draws[policy] = fresh_draws

        logger.info(
            f"Added fresh draws for policy '{policy}': "
            f"{len(fresh_draws.samples)} samples, "
            f"{fresh_draws.draws_per_prompt} draws/prompt"
        )

    def _auto_load_fresh_draws(self) -> None:
        """Attempt to auto-load fresh draws from standard locations.

        Looks for fresh draws in:
        1. Same directory as dataset
        2. responses/ subdirectory
        3. fresh_draws/ subdirectory
        """
        # Skip if fresh draws already loaded for all policies
        if all(policy in self._fresh_draws for policy in self.sampler.target_policies):
            logger.debug(
                "Fresh draws already loaded for all policies, skipping auto-load"
            )
            return

        logger.info("Attempting to auto-load fresh draws...")

        # Try to infer data directory from sampler's dataset path if available
        data_dir = None

        # Check if sampler has dataset_path attribute or metadata
        if hasattr(self.sampler, "dataset_path"):
            data_dir = Path(self.sampler.dataset_path).parent
            logger.debug(f"Found dataset_path on sampler: {self.sampler.dataset_path}")
        elif (
            hasattr(self.sampler, "metadata")
            and "dataset_path" in self.sampler.metadata
        ):
            data_dir = Path(self.sampler.metadata["dataset_path"]).parent
            logger.debug(
                f"Found dataset_path in sampler metadata: {self.sampler.metadata['dataset_path']}"
            )
        else:
            # Try current directory and parent
            logger.debug(f"No dataset_path found, checking cwd: {Path.cwd()}")
            for potential_dir in [Path.cwd(), Path.cwd().parent]:
                if (potential_dir / "data").exists():
                    data_dir = potential_dir / "data"
                    logger.debug(f"Found data directory at: {data_dir}")
                    break

        if data_dir is None:
            logger.warning(
                "Could not determine data directory for auto-loading fresh draws"
            )
            return

        # Try to load fresh draws for each policy
        from ..data.fresh_draws import load_fresh_draws_auto

        for policy in self.sampler.target_policies:
            if policy in self._fresh_draws:
                # Already loaded
                continue

            try:
                fresh_draws = load_fresh_draws_auto(data_dir, policy, verbose=False)
                self._fresh_draws[policy] = fresh_draws
                logger.info(f"Auto-loaded fresh draws for policy '{policy}'")
            except Exception as e:
                logger.debug(f"Could not auto-load fresh draws for '{policy}': {e}")

    def _compute_policy_diagnostics(
        self, policy: str, estimate: float
    ) -> Dict[str, Any]:
        """Compute diagnostics for a single policy.

        This helper method ensures consistent diagnostic computation across
        all DR estimator subclasses.

        Args:
            policy: Policy name
            estimate: The DR estimate for this policy

        Returns:
            Dictionary of diagnostic metrics
        """
        return compute_dr_policy_diagnostics(
            dm_component=self._dm_component.get(policy, np.array([])),
            ips_correction=self._ips_correction.get(policy, np.array([])),
            dr_estimate=estimate,
            fresh_rewards=self._fresh_rewards.get(policy),  # Always use stored rewards
            outcome_predictions=self._outcome_predictions.get(policy),
            influence_functions=self._influence_functions.get(policy),
            unique_folds=list(range(self.n_folds)),
            policy=policy,
        )

    def fit(self) -> None:
        """Fit weight calibration (if applicable) and outcome model."""
        # First fit the IPS weights
        self.ips_estimator.fit()

        # Call estimate() to populate IPS diagnostics (needed for weight ESS, etc.)
        # This is cheap after fit() and ensures DR diagnostics include weight metrics
        self._ips_result = self.ips_estimator.estimate()

        # Then fit the outcome model on logged data
        self._fit_outcome_model()

        self._fitted = True

    def _fit_outcome_model(self) -> None:
        """Fit the outcome model on logged data."""
        # Get indices of samples that are valid for at least one policy
        valid_for_any: set[int] = set()
        for policy in self.sampler.target_policies:
            valid_indices = self.sampler._get_valid_indices(policy)
            valid_for_any.update(valid_indices)

        # Sort to maintain order
        valid_indices_list = sorted(valid_for_any)

        # Upfront validation: Check all samples have judge scores
        missing_judge_scores: list[tuple[int, str]] = []
        invalid_judge_scores: list[tuple[int, str]] = []
        for idx in valid_indices_list:
            sample = self.sampler.dataset.samples[idx]
            if sample.judge_score is None:
                missing_judge_scores.append((idx, sample.prompt_id))
            elif not isinstance(sample.judge_score, (int, float)):
                invalid_judge_scores.append((idx, sample.prompt_id))

        if missing_judge_scores:
            example_ids = [str(pid) for _, pid in missing_judge_scores[:3]]
            raise ValueError(
                f"DR requires judge_score for all samples. Missing {len(missing_judge_scores)} scores. "
                f"Example prompt_ids: {example_ids}. "
                f"Run calibrate_dataset(..., enable_cross_fit=True) with judge_field specified."
            )

        if invalid_judge_scores:
            example_ids = [str(pid) for _, pid in invalid_judge_scores[:3]]
            raise ValueError(
                f"DR requires numeric judge_score for all samples. {len(invalid_judge_scores)} invalid. "
                f"Example prompt_ids: {example_ids}."
            )

        # Collect logged data
        prompts = []
        responses = []
        rewards = []
        judge_scores = []
        valid_fold_assignments = []

        # Check if reward_calibrator has covariate names
        covariate_names: List[str] = []
        if self.reward_calibrator is not None and hasattr(
            self.reward_calibrator, "covariate_names"
        ):
            covariate_names = self.reward_calibrator.covariate_names or []

        # Collect covariates if specified
        covariate_values: Optional[List[List[float]]] = [] if covariate_names else None

        for idx in valid_indices_list:
            sample = self.sampler.dataset.samples[idx]
            prompts.append(sample.prompt)
            responses.append(sample.response)

            # Get calibrated reward (from full model)
            if sample.reward is not None:
                rewards.append(sample.reward)
            else:
                raise ValueError("All samples must have calibrated rewards for DR")

            # Get judge score from metadata
            if sample.judge_score is None:
                raise ValueError("All samples must have judge scores for DR")
            judge_scores.append(sample.judge_score)

            # Extract covariates if specified
            if covariate_values is not None:
                sample_covariates = []
                for cov_name in covariate_names:
                    if cov_name not in sample.metadata:
                        raise ValueError(
                            f"Covariate '{cov_name}' not found in metadata for sample {idx} "
                            f"(prompt_id={sample.prompt_id})"
                        )
                    cov_value = sample.metadata[cov_name]
                    if cov_value is None:
                        raise ValueError(
                            f"Covariate '{cov_name}' is None for sample {idx}. "
                            "All covariate values must be present."
                        )
                    try:
                        sample_covariates.append(float(cov_value))
                    except (TypeError, ValueError) as e:
                        raise ValueError(
                            f"Covariate '{cov_name}' has non-numeric value for sample {idx}: {e}"
                        )
                covariate_values.append(sample_covariates)

            # Get fold assignment using unified system
            # Note: We compute fold from prompt_id to handle filtered data correctly
            from ..data.folds import get_fold

            fold = get_fold(sample.prompt_id, self.n_folds, self.random_seed)
            valid_fold_assignments.append(fold)

        rewards_array = np.array(rewards)
        judge_scores_array = np.array(judge_scores)
        fold_assignments_array = (
            np.array(valid_fold_assignments) if valid_fold_assignments else None
        )

        # Convert covariates to array if extracted
        covariates_array = np.array(covariate_values) if covariate_values else None

        # Pass fold assignments and covariates to outcome model
        self.outcome_model.fit(
            prompts,
            responses,
            rewards_array,
            judge_scores_array,
            fold_assignments_array,
            covariates=covariates_array,
        )

        # Store the valid indices for later use
        self._outcome_valid_indices = valid_indices_list

        # Store covariate names for use in estimate()
        self._covariate_names = covariate_names

        # Precompute prompt_id to fold mapping for O(1) lookup in estimate()
        self._promptid_to_fold = {}
        if fold_assignments_array is not None:
            for idx, fold in zip(valid_indices_list, fold_assignments_array):
                sample = self.sampler.dataset.samples[idx]
                pid = str(sample.prompt_id)
                self._promptid_to_fold[pid] = int(fold)

        self._outcome_fitted = True

        if covariate_names:
            logger.info(
                f"Fitted outcome model on {len(prompts)} logged samples with {len(covariate_names)} covariates"
            )
        else:
            logger.info(f"Fitted outcome model on {len(prompts)} logged samples")

    def estimate(self) -> EstimationResult:
        """Compute DR estimates for all target policies.

        DR formula: V_DR(π') = E[g(X, A', S')] + E[W * (R - g(X, A, S))]
        Where the first term is the Direct Method and second is IPS correction.

        Will attempt to auto-load fresh draws if not already added.
        """
        self._validate_fitted()

        # Auto-load fresh draws if not already loaded
        self._auto_load_fresh_draws()

        estimates = []
        standard_errors = []
        n_samples_used = {}
        # Calibration-floor metrics per policy
        calibration_floor_meta: Dict[str, Dict[str, float]] = {}
        df_info = {}  # Track degrees of freedom per policy

        for policy in self.sampler.target_policies:
            # Check fresh draws are available
            if policy not in self._fresh_draws:
                raise ValueError(
                    f"No fresh draws for policy '{policy}'. "
                    f"Tried auto-loading but failed. Call add_fresh_draws() manually."
                )

            # Get components
            weights = self.ips_estimator.get_weights(policy)
            if weights is None:
                # Check if this is a no_overlap case
                # get_diagnostics() doesn't take policy argument, it returns all
                logger.warning(f"No weights for policy '{policy}', skipping")
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            # Get rewards (already filtered to valid samples)
            data = self.sampler.get_data_for_policy(policy)
            if data is None:
                logger.warning(f"No data for policy '{policy}', skipping")
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue
            logged_rewards = np.array([d["reward"] for d in data])

            # Sanity check: weights and logged data should be aligned
            if len(weights) != len(logged_rewards):
                raise ValueError(
                    f"Weights and logged data length mismatch for policy '{policy}': "
                    f"weights={len(weights)}, data={len(logged_rewards)}"
                )

            # Get logged data for outcome model
            logged_prompts = [d["prompt"] for d in data]
            logged_responses = [d["response"] for d in data]
            logged_scores = np.array([d.get("judge_score") for d in data])
            # Require prompt_ids for DR (no fallback to index)
            logged_prompt_ids = []
            for i, d in enumerate(data):
                if "prompt_id" not in d:
                    raise ValueError(
                        f"Data entry {i} for policy '{policy}' missing 'prompt_id'. "
                        f"DR estimation requires prompt_id to align with fresh draws."
                    )
                logged_prompt_ids.append(str(d["prompt_id"]))

            # Extract covariates if specified
            logged_covariates = None
            if hasattr(self, "_covariate_names") and self._covariate_names:
                covariate_values = []
                for d in data:
                    sample_covariates = []
                    for cov_name in self._covariate_names:
                        cov_value = d.get(cov_name)  # Use get() instead of in check
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

            # Get fold assignments using precomputed mapping (O(1) lookups)
            # Strict mode: error if any prompt_id is missing fold assignment
            valid_fold_ids_list = []
            if self._promptid_to_fold:
                unknown_pids = []
                for pid in logged_prompt_ids:
                    if pid not in self._promptid_to_fold:
                        unknown_pids.append(pid)
                    else:
                        valid_fold_ids_list.append(self._promptid_to_fold[pid])

                if unknown_pids:
                    raise ValueError(
                        f"Missing fold assignments for {len(unknown_pids)} samples in policy '{policy}'. "
                        f"Example prompt_ids: {unknown_pids[:3]}. "
                        f"Ensure calibration was done with enable_cross_fit=True or provide explicit fold assignments."
                    )
            else:
                raise ValueError(
                    f"No fold assignments available for DR estimation. "
                    f"Ensure calibration was done with enable_cross_fit=True."
                )
            valid_fold_ids = np.array(valid_fold_ids_list)

            # Get outcome model predictions for logged data (using cross-fitted models)
            # Both IsotonicOutcomeModel and BaseOutcomeModel-derived classes need fold_ids
            if hasattr(self.outcome_model, "predict"):
                # Our outcome models accept fold_ids and covariates for cross-fitting
                g_logged = self.outcome_model.predict(
                    logged_prompts,
                    logged_responses,
                    logged_scores,
                    valid_fold_ids,
                    covariates=logged_covariates,
                )
            else:
                # Fallback for other models
                g_logged = self.outcome_model.predict(
                    logged_prompts, logged_responses, logged_scores
                )

            # Get fresh draws
            fresh_dataset = self._fresh_draws[policy]

            # Collect fresh scores for each logged sample
            g_fresh_all = []
            fresh_draw_var_per_prompt_list = []  # For diagnostics

            for i, prompt_id in enumerate(logged_prompt_ids):
                # Get fresh judge scores for this prompt
                if prompt_id is None:
                    raise ValueError(f"Missing prompt_id for sample {i}")
                fresh_scores = fresh_dataset.get_scores_for_prompt_id(prompt_id)

                # Get fresh samples to validate fold assignments
                fresh_samples = fresh_dataset.get_samples_for_prompt_id(prompt_id)

                # Create dummy prompts/responses for outcome model interface
                fresh_prompts = [logged_prompts[i]] * len(fresh_scores)
                fresh_responses = [""] * len(fresh_scores)  # Not used in isotonic model

                # Use same fold for all fresh draws from this prompt
                fresh_fold_ids = np.full(len(fresh_scores), valid_fold_ids[i])

                # Validate that fresh draws have matching fold assignments if available
                for j, fresh_sample in enumerate(fresh_samples):
                    if (
                        fresh_sample.fold_id is not None
                        and fresh_sample.fold_id != valid_fold_ids[i]
                    ):
                        logger.warning(
                            f"Fold mismatch for prompt_id '{prompt_id}': "
                            f"logged fold={valid_fold_ids[i]}, fresh fold={fresh_sample.fold_id}"
                        )

                # Get covariate values for fresh draws
                fresh_covariates = None
                if hasattr(self, "_covariate_names") and self._covariate_names:
                    # Extract covariates from fresh draw metadata (response-level covariates)
                    fresh_cov_values = []
                    for fresh_sample in fresh_samples:
                        sample_covariates = []
                        for cov_name in self._covariate_names:
                            if cov_name not in fresh_sample.metadata:
                                raise ValueError(
                                    f"Covariate '{cov_name}' not found in fresh draw metadata "
                                    f"for prompt_id '{prompt_id}'. Available metadata: {list(fresh_sample.metadata.keys())}"
                                )
                            sample_covariates.append(fresh_sample.metadata[cov_name])
                        fresh_cov_values.append(sample_covariates)
                    fresh_covariates = np.array(fresh_cov_values)

                # Get predictions for fresh draws
                # Note: Our models need fold_ids for cross-fitting
                # They all use the same fold for each prompt's fresh draws
                if hasattr(self.outcome_model, "predict"):
                    g_fresh_prompt = self.outcome_model.predict(
                        fresh_prompts,
                        fresh_responses,
                        fresh_scores,
                        fresh_fold_ids,
                        covariates=fresh_covariates,
                    )
                else:
                    # Fallback for other models
                    g_fresh_prompt = self.outcome_model.predict(
                        fresh_prompts, fresh_responses, fresh_scores
                    )

                # Average over draws for this prompt
                g_fresh_all.append(g_fresh_prompt.mean())

                # Track variance for diagnostics
                if len(g_fresh_prompt) > 1:
                    fresh_draw_var_per_prompt_list.append(g_fresh_prompt.var())
                else:
                    fresh_draw_var_per_prompt_list.append(0.0)

            g_fresh = np.array(g_fresh_all)
            fresh_draw_var_per_prompt = np.array(fresh_draw_var_per_prompt_list)

            # Store for MC variance computation later
            # Also track actual draws per prompt (M_i) which may vary
            draws_per_prompt_list = []
            for prompt_id in logged_prompt_ids:
                if prompt_id is not None:
                    fresh_scores = fresh_dataset.get_scores_for_prompt_id(prompt_id)
                    draws_per_prompt_list.append(len(fresh_scores))
                else:
                    draws_per_prompt_list.append(1)  # Fallback

            if not hasattr(self, "_fresh_draw_stats"):
                self._fresh_draw_stats = {}
            self._fresh_draw_stats[policy] = {
                "variances": fresh_draw_var_per_prompt,
                "draws_per_prompt": np.array(
                    draws_per_prompt_list
                ),  # Now per-prompt M_i
                "n_prompts": len(fresh_draw_var_per_prompt),
            }

            # Sanity check: weights should have mean approximately 1.0 (only for Hajek/calibrated weights)
            weights_mean = weights.mean()
            # Only check mean ~ 1.0 when using Hajek/mean-one weights
            if (
                self.use_calibrated_weights
                and getattr(self.ips_estimator, "weight_mode", "hajek") == "hajek"
                and not (0.99 <= weights_mean <= 1.01)
            ):
                weights_min = weights.min()
                weights_max = weights.max()
                weights_std = weights.std()
                logger.warning(
                    f"Weights for policy '{policy}' deviate from expected mean=1.0: "
                    f"mean={weights_mean:.3f}, std={weights_std:.3f}, "
                    f"min={weights_min:.3e}, max={weights_max:.3e}. "
                    f"This may indicate calibration issues or poor policy overlap."
                )

            # DR estimate components
            dm_term = g_fresh.mean()  # Direct method term
            ips_correction_base = weights * (logged_rewards - g_logged)

            # No augmentation - OUA jackknife handles oracle uncertainty via variance
            ips_correction = ips_correction_base.mean()
            dr_estimate = dm_term + ips_correction

            # Store components for diagnostics (avoid recomputation later)
            self._dm_component[policy] = g_fresh
            self._ips_correction[policy] = ips_correction_base  # No augmentation
            self._fresh_rewards[policy] = logged_rewards  # Actually logged rewards
            self._outcome_predictions[policy] = g_logged

            # Compute standard error using influence function
            if_contributions = g_fresh + ips_correction_base - dr_estimate

            # IIC removed - influence functions used directly

            # Base SE from influence functions (using cluster-robust SE on outcome folds)
            from ..diagnostics.robust_inference import (
                cluster_robust_se,
                compose_se_components,
            )

            se_if = np.std(if_contributions, ddof=1) / np.sqrt(
                len(if_contributions)
            )  # fallback
            try:
                res_if = cluster_robust_se(
                    data=if_contributions,
                    cluster_ids=valid_fold_ids,  # outcome folds (one per prompt)
                    statistic_fn=lambda x: np.mean(x),
                    influence_fn=lambda x: x,  # IF already provided
                    alpha=0.05,
                )
                se_if = res_if["se"]

                # Store degrees of freedom for this policy
                df_cluster = res_if.get("df", len(if_contributions) - 1)

                # If OUA was applied, get oracle DF and take minimum
                df_final = df_cluster
                if self.oua_jackknife and self.reward_calibrator is not None:
                    try:
                        if hasattr(self.reward_calibrator, "get_fold_models_for_oua"):
                            fold_models = (
                                self.reward_calibrator.get_fold_models_for_oua()
                            )
                            if fold_models:
                                K = len(fold_models)
                                df_oracle = K - 1
                                df_final = min(df_cluster, df_oracle)
                    except Exception as e:
                        logger.debug(f"Could not get oracle DF for {policy}: {e}")

                # Ensure DF is at least 1
                df_final = max(df_final, 1)

                # Store DF info
                from scipy import stats

                t_crit = stats.t.ppf(1 - 0.05 / 2, df_final)
                df_info[policy] = {
                    "df": int(df_final),
                    "t_critical": float(t_crit),
                    "n_clusters": int(res_if.get("n_clusters", len(if_contributions))),
                }

                logger.debug(
                    f"Using cluster-robust SE for {policy}: "
                    f"naive={np.std(if_contributions, ddof=1) / np.sqrt(len(if_contributions)):.6f}, "
                    f"robust={se_if:.6f}, n_clusters={res_if['n_clusters']}, df={df_final}"
                )
            except Exception as e:
                logger.debug(f"cluster_robust_se failed for {policy}: {e}")

            base_se = se_if  # For backward compatibility with diagnostics below

            # Add Monte Carlo variance component from finite fresh draws
            mc_var = 0.0
            n = len(g_fresh) if policy in self._dm_component else 0

            if hasattr(self, "_fresh_draw_stats") and policy in self._fresh_draw_stats:
                stats = self._fresh_draw_stats[policy]
                fresh_var = np.asarray(
                    stats["variances"]
                )  # per-prompt sample var if M_i>=2, else 0
                M = np.asarray(stats["draws_per_prompt"])  # Ensure numpy array

                # Check if we can compute exact MC variance (all M >= 2)
                if np.all(M >= 2):
                    # Exact MC variance computation
                    mc_var = np.sum(fresh_var / np.maximum(M, 1)) / (len(data) ** 2)
                    fallback_used = False
                    fallback_method = "exact"
                    s2_total = None
                    s2_cap = None
                else:
                    # Automatic fallback for M=1 cases
                    # Conservative upper bound using total variance
                    # Mathematical justification: For a mixture distribution,
                    # Var(X) = E[Var(X|I)] + Var(E[X|I]) >= E[Var(X|I)]
                    # where I indexes the components (prompts).
                    # Thus the total variance upper-bounds the average within-prompt variance.
                    fallback_used = True
                    s2_total = float(np.var(g_fresh, ddof=1)) if n > 1 else 0.0
                    s2_cap = min(
                        s2_total, 0.25
                    )  # Cap at maximum possible variance for [0,1]

                    # Check if we have mixed M values (some M>=2, some M=1)
                    has_multi = np.any(M >= 2)
                    if has_multi:
                        # Combine exact for M>=2 and bound for M=1
                        mask_multi = M >= 2
                        M_array = np.asarray(
                            M
                        )  # Ensure it's a numpy array for indexing
                        exact_part = float(
                            np.sum(
                                fresh_var[mask_multi]
                                / np.maximum(M_array[mask_multi], 1)
                            )
                        ) / (n**2)
                        n_singles = int(np.sum(M < 2))
                        bound_part = float(np.sum(np.ones(n_singles) * s2_cap)) / (n**2)
                        mc_var = exact_part + bound_part
                        fallback_method = "upper_bound(mixed)"
                    else:
                        # All M=1, use pure bound
                        mc_var = s2_cap / n if n > 0 else 0.0
                        fallback_method = "upper_bound(total_var)"

                    logger.debug(
                        f"Using MC variance fallback for {policy}: "
                        f"{fallback_method}, s2_cap={s2_cap:.4f}"
                    )

                # Store MC diagnostics with fallback information
                if not hasattr(self, "_mc_diagnostics"):
                    self._mc_diagnostics = {}
                self._mc_diagnostics[policy] = {
                    "base_se": base_se,
                    "mc_var": mc_var,
                    "mc_share": (
                        mc_var / (base_se**2 + mc_var)
                        if (base_se**2 + mc_var) > 0
                        else 0
                    ),
                    "avg_draws_per_prompt": float(M.mean()),
                    "min_draws_per_prompt": int(M.min()),
                    "max_draws_per_prompt": int(M.max()),
                    "fallback_used": fallback_used,
                    "fallback_method": fallback_method,
                    "n_prompts": int(n),
                    "n_prompts_M1": int(np.sum(M < 2)),
                }

                # Add fallback-specific diagnostics if used
                if fallback_used:
                    if s2_total is not None:
                        self._mc_diagnostics[policy]["s2_total"] = float(s2_total)
                    if s2_cap is not None:
                        self._mc_diagnostics[policy]["s2_cap"] = float(s2_cap)

            # Add oracle uncertainty from finite-oracle jackknife
            se_oracle = 0.0

            # Skip oracle uncertainty at 100% coverage (no uncertainty when we have all labels)
            skip_oua = False
            try:
                if (
                    hasattr(self.sampler, "oracle_coverage")
                    and self.sampler.oracle_coverage == 1.0
                ):
                    skip_oua = True
                    logger.debug(
                        f"Skipping oracle uncertainty for {policy}: 100% oracle coverage"
                    )
            except Exception:
                pass  # Continue with normal OUA calculation if we can't check

            if not skip_oua:
                jack = self.get_oracle_jackknife(policy)
                if jack is not None and len(jack) >= 2:
                    K = len(jack)
                    mu = float(np.mean(jack))
                    se_oracle = float(np.sqrt((K - 1) / K * np.sum((jack - mu) ** 2)))
                    logger.debug(
                        f"Oracle SE for {policy}: {se_oracle:.6f} from {K} folds"
                    )

            # Total SE combining all sources: IF (cluster-robust), oracle, and fresh-draw MC
            se = compose_se_components(se_if, se_oracle, mc_var)

            # For backward compatibility, keep total_var
            total_var = se**2

            if mc_var > 0:
                logger.debug(
                    f"SE for '{policy}': base={base_se:.4f}, with MC={se:.4f} "
                    f"(MC adds {100*mc_var/total_var:.1f}% to variance)"
                )

            # Store influence functions (always needed for proper inference)
            self._influence_functions[policy] = if_contributions

            # Store sample indices for IF alignment in stacking
            self._store_sample_indices(policy, data)

            estimates.append(dr_estimate)
            standard_errors.append(se)
            n_samples_used[policy] = len(data)

            # Compute calibration-floor metrics (logged and fresh)
            try:
                cal_info = getattr(self.sampler.dataset, "metadata", {}).get(
                    "calibration_info", {}
                )
                f_min = float(cal_info.get("f_min", float("nan")))
                eps = 1e-6
                # Logged floor mass
                floor_mass_logged = (
                    float(np.mean(np.abs(logged_rewards - f_min) <= eps))
                    if np.isfinite(f_min)
                    else float("nan")
                )
                # Fresh floor mass (use global reward_calibrator on fresh scores)
                floor_mass_fresh = float("nan")
                if (
                    self.reward_calibrator is not None
                    and policy in self._fresh_draws
                    and np.isfinite(f_min)
                ):
                    fresh_dataset = self._fresh_draws[policy]
                    fresh_scores_all = []
                    for pid in set(d["prompt_id"] for d in data):
                        sc = fresh_dataset.get_scores_for_prompt_id(pid)
                        if sc is not None and len(sc) > 0:
                            fresh_scores_all.extend(list(sc))
                    if fresh_scores_all:
                        fresh_scores_arr = np.asarray(fresh_scores_all, dtype=float)
                        fresh_pred = np.clip(
                            self.reward_calibrator.predict(fresh_scores_arr), 0.0, 1.0
                        )
                        floor_mass_fresh = float(
                            np.mean(np.abs(fresh_pred - f_min) <= eps)
                        )
                calibration_floor_meta[policy] = {
                    "f_min": f_min,
                    "floor_mass_logged": floor_mass_logged,
                    "floor_mass_fresh": floor_mass_fresh,
                }
            except Exception:
                pass

            logger.info(
                f"DR estimate for policy '{policy}': {dr_estimate:.4f} ± {se:.4f} "
                f"(DM={dm_term:.4f}, IPS_corr={ips_correction:.4f})"
            )

            # Compute orthogonality score (new)
            ortho_result = compute_orthogonality_score(
                weights=weights,
                rewards=logged_rewards,
                outcome_predictions=g_logged,
                return_ci=True,
            )
            self._orthogonality_scores[policy] = ortho_result

            # Compute DM-IPS decomposition (new)
            decomp_result = compute_dm_ips_decomposition(
                g_hat=g_fresh,
                weights=weights,
                rewards=logged_rewards,
                q_hat=g_logged,
            )
            self._dm_ips_decompositions[policy] = decomp_result

        # Build DR diagnostics using stored components
        dr_diagnostics_per_policy: Dict[str, Dict[str, Any]] = {}

        for idx, policy in enumerate(self.sampler.target_policies):
            if policy not in self._dm_component or np.isnan(estimates[idx]):
                continue

            # Use helper method for consistent diagnostic computation
            dr_diagnostics_per_policy[policy] = self._compute_policy_diagnostics(
                policy, estimates[idx]
            )

        # Add DR-specific metadata
        dr_metadata = {
            "fresh_draws_policies": list(self._fresh_draws.keys()),
            "cross_fitted": True,
            "n_folds": self.n_folds,
        }

        # Add MC variance diagnostics if available
        if hasattr(self, "_mc_diagnostics"):
            dr_metadata["mc_variance_diagnostics"] = self._mc_diagnostics
            dr_metadata["mc_variance_included"] = True

        # Create overview
        dr_overview = {}
        if dr_diagnostics_per_policy:
            dr_overview = {
                "policies": list(dr_diagnostics_per_policy.keys()),
                "dm_vs_ips": {
                    p: (d["dm_mean"], d["ips_corr_mean"])
                    for p, d in dr_diagnostics_per_policy.items()
                },
                "worst_if_tail_ratio_99_5": max(
                    d.get("if_tail_ratio_99_5", 0.0)
                    for d in dr_diagnostics_per_policy.values()
                ),
            }

            # For TMLE specifically (will be overridden in subclass)
            if self.__class__.__name__ == "TMLEEstimator":
                dr_overview["tmle_score_abs_mean"] = {
                    p: abs(d["score_mean"])
                    for p, d in dr_diagnostics_per_policy.items()
                }

        # Build metadata (keep dr_diagnostics for backward compatibility with visualization)
        metadata: Dict[str, Any] = {
            "target_policies": list(self.sampler.target_policies),
            "weight_method": "calibrated" if self.use_calibrated_weights else "raw",
            "dr_diagnostics": dr_diagnostics_per_policy,  # Keep for visualization
            "dr_overview": dr_overview,
            "orthogonality_scores": self._orthogonality_scores,  # New: orthogonality diagnostics
            "dm_ips_decompositions": self._dm_ips_decompositions,  # New: DM-IPS breakdown
            "dr_influence": self._influence_functions,  # Store influence functions for analysis
        }
        # Attach SE diagnostics for ablation runner to use t-critical
        if self._se_diagnostics:
            metadata["_se_diagnostics"] = self._se_diagnostics

        # Get IPS diagnostics if available
        ips_diag = None
        if hasattr(self.ips_estimator, "get_diagnostics"):
            ips_diag = self.ips_estimator.get_diagnostics()

        # Build DR diagnostics directly
        dr_diagnostics = self._build_dr_diagnostics(
            estimates,
            standard_errors,
            n_samples_used,
            dr_diagnostics_per_policy,
            ips_diag,
        )

        # Attach calibration-floor meta if available
        if calibration_floor_meta:
            metadata["calibration_floor"] = calibration_floor_meta

        # Add dr_metadata to surface MC variance diagnostics
        metadata["dr_metadata"] = dr_metadata

        # Also expose MC variance at top-level for stacker convenience
        if hasattr(self, "_mc_diagnostics"):
            metadata["mc_variance_diagnostics"] = self._mc_diagnostics
            metadata["mc_variance_included"] = True

        # Attach compact core summary for empirical analysis (no UX change)
        try:
            core_summary: Dict[str, Dict[str, Any]] = {}
            ess = ips_diag.ess_per_policy if ips_diag else {}
            tails_dict: Dict[str, Optional[float]] = (
                getattr(ips_diag, "tail_indices", None) if ips_diag else None
            ) or {}
            hell_all = (
                getattr(ips_diag, "hellinger_affinity", None) if ips_diag else None
            )
            hell_per = (
                getattr(ips_diag, "hellinger_per_policy", None) if ips_diag else None
            )
            cal_floor = metadata.get("calibration_floor", {})
            for policy in self.sampler.target_policies:
                core_summary[policy] = {
                    "ess_fraction": float(ess.get(policy, 0.0)) if ess else None,
                    "tail_index": (
                        float(val)
                        if (val := tails_dict.get(policy)) is not None
                        else None
                    ),
                    "hellinger_affinity": (
                        float(hell_per[policy])
                        if hell_per
                        and policy in hell_per
                        and hell_per[policy] is not None
                        else (float(hell_all) if hell_all is not None else None)
                    ),
                }
                # Be explicit for mypy: calibration_floor is dict-like
                try:
                    cf = cast(Dict[str, Dict[str, Any]], cal_floor)
                    if policy in cf:
                        core_summary[policy].update(cf[policy])
                except Exception:
                    pass
                # DR orthogonality pass (if available)
                ortho = self._orthogonality_scores.get(policy)
                if ortho and all(k in ortho for k in ("ci_lower", "ci_upper")):
                    ci_l = (
                        float(ortho["ci_lower"])
                        if ortho["ci_lower"] is not None
                        else None
                    )
                    ci_u = (
                        float(ortho["ci_upper"])
                        if ortho["ci_upper"] is not None
                        else None
                    )
                    core_summary[policy]["orthogonality_pass"] = bool(
                        ci_l is not None and ci_u is not None and ci_l <= 0 <= ci_u
                    )
            metadata["core_summary"] = core_summary
        except Exception:
            pass

        # Mark that oracle variance is already included in standard_errors
        # (prevents base class from adding it again)
        metadata["se_components"] = {
            "includes_oracle_uncertainty": True,
            "includes_mc_variance": True,
        }

        # Add sample indices for IF alignment in stacking
        if hasattr(self, "_if_sample_indices"):
            metadata["if_sample_indices"] = self._if_sample_indices

        # Add degrees of freedom info
        if df_info:
            metadata["degrees_of_freedom"] = df_info

        base_result = EstimationResult(
            estimates=np.array(estimates),
            standard_errors=np.array(standard_errors),
            n_samples_used=n_samples_used,
            method="dr_base",
            influence_functions=self._influence_functions,
            diagnostics=dr_diagnostics,
            metadata=metadata,
        )

        # Apply OUA jackknife using base class method
        self._apply_oua_jackknife(base_result)

        return base_result

    def _build_dr_diagnostics(
        self,
        estimates: List[float],
        standard_errors: List[float],
        n_samples_used: Dict[str, int],
        dr_diagnostics_per_policy: Dict[str, Dict[str, Any]],
        ips_diagnostics: Optional[IPSDiagnostics],
    ) -> DRDiagnostics:
        """Build DRDiagnostics object from components.

        Args:
            estimates: List of estimates per policy
            standard_errors: List of SEs per policy
            n_samples_used: Dict of samples used per policy
            dr_diagnostics_per_policy: Detailed DR diagnostics
            ips_diagnostics: IPSDiagnostics from internal IPS estimator

        Returns:
            DRDiagnostics object
        """
        # Build estimates/SE dicts
        policies = list(self.sampler.target_policies)
        estimates_dict = {
            p: float(e) for p, e in zip(policies, estimates) if not np.isnan(e)
        }
        se_dict = {
            p: float(se) for p, se in zip(policies, standard_errors) if not np.isnan(se)
        }

        # Extract summary metrics from detailed diagnostics
        r2_values = []
        rmse_values = []
        if_tail_ratios = []

        for policy, diag in dr_diagnostics_per_policy.items():
            if "r2_oof" in diag and diag["r2_oof"] is not None:
                r2_values.append(diag["r2_oof"])
            if "residual_rmse" in diag and diag["residual_rmse"] is not None:
                rmse_values.append(diag["residual_rmse"])
            if "if_tail_ratio_99_5" in diag:
                if_tail_ratios.append(diag["if_tail_ratio_99_5"])
            else:
                # Use a default value if influence functions weren't computed
                if_tail_ratios.append(0.0)

        # Compute ranges
        outcome_r2_range = (min(r2_values), max(r2_values)) if r2_values else (0.0, 0.0)
        outcome_rmse_mean = np.mean(rmse_values) if rmse_values else 0.0
        worst_if_tail = max(if_tail_ratios) if if_tail_ratios else 0.0

        # Build DRDiagnostics
        if ips_diagnostics is not None:
            # Copy fields from IPS diagnostics
            diagnostics = DRDiagnostics(
                estimator_type=f"DR_{ips_diagnostics.estimator_type}",
                method=self.__class__.__name__.lower().replace("estimator", ""),
                n_samples_total=ips_diagnostics.n_samples_total,
                n_samples_valid=ips_diagnostics.n_samples_valid,
                n_policies=len(policies),
                policies=policies,
                estimates=estimates_dict,
                standard_errors=se_dict,
                n_samples_used=n_samples_used,
                # Weight fields from IPS
                weight_ess=ips_diagnostics.weight_ess,
                weight_status=ips_diagnostics.weight_status,
                ess_per_policy=ips_diagnostics.ess_per_policy,
                max_weight_per_policy=ips_diagnostics.max_weight_per_policy,
                weight_tail_ratio_per_policy=getattr(
                    ips_diagnostics, "weight_tail_ratio_per_policy", {}
                ),
                # Calibration fields (may be None)
                calibration_rmse=ips_diagnostics.calibration_rmse,
                calibration_r2=ips_diagnostics.calibration_r2,
                calibration_coverage=ips_diagnostics.calibration_coverage,
                n_oracle_labels=ips_diagnostics.n_oracle_labels,
                # DR-specific fields
                dr_cross_fitted=True,
                dr_n_folds=self.n_folds,
                outcome_r2_range=outcome_r2_range,
                outcome_rmse_mean=outcome_rmse_mean,
                worst_if_tail_ratio=worst_if_tail,
                dr_diagnostics_per_policy=dr_diagnostics_per_policy,
                dm_ips_decompositions=self._dm_ips_decompositions,
                orthogonality_scores=self._orthogonality_scores,
                influence_functions=self._influence_functions,
            )
        else:
            # No IPS diagnostics available, create minimal version
            from ..diagnostics import Status

            diagnostics = DRDiagnostics(
                estimator_type="DR",
                method=self.__class__.__name__.lower().replace("estimator", ""),
                n_samples_total=len(self.sampler.dataset.samples),
                n_samples_valid=self.sampler.n_valid_samples,
                n_policies=len(policies),
                policies=policies,
                estimates=estimates_dict,
                standard_errors=se_dict,
                n_samples_used=n_samples_used,
                # Minimal weight fields
                weight_ess=0.0,
                weight_status=Status.WARNING,
                ess_per_policy={},
                max_weight_per_policy={},
                weight_tail_ratio_per_policy={},
                # DR-specific fields
                dr_cross_fitted=True,
                dr_n_folds=self.n_folds,
                outcome_r2_range=outcome_r2_range,
                outcome_rmse_mean=outcome_rmse_mean,
                worst_if_tail_ratio=worst_if_tail,
                dr_diagnostics_per_policy=dr_diagnostics_per_policy,
                dm_ips_decompositions=self._dm_ips_decompositions,
                orthogonality_scores=self._orthogonality_scores,
                influence_functions=self._influence_functions,
            )

        return diagnostics

    def _store_sample_indices(self, policy: str, data: Any) -> None:
        """Store sample indices for IF alignment in stacking.

        This method extracts stable identifiers (prompt_ids) from the data
        and stores them for later use in aligning influence functions across
        ensemble components.

        Args:
            policy: Target policy name
            data: Data from get_data_for_policy (list of dicts with flattened metadata)
        """
        if not hasattr(self, "_if_sample_indices"):
            self._if_sample_indices = {}

        sample_indices = []
        for d in data:
            if isinstance(d, dict):
                if "prompt_id" in d:
                    # Best: use the stable prompt_id (already flattened from metadata)
                    sample_indices.append(d["prompt_id"])
                elif "prompt" in d and "response" in d:
                    # Fallback: use content hash (also stable)
                    content_hash = hash((d["prompt"], d["response"]))
                    sample_indices.append(f"hash_{content_hash}")
                else:
                    # Last resort: use index
                    sample_indices.append(f"idx_{len(sample_indices)}")
            else:
                # Handle if it's an object (shouldn't happen with get_data_for_policy)
                if hasattr(d, "metadata") and d.metadata and "prompt_id" in d.metadata:
                    sample_indices.append(d.metadata["prompt_id"])
                else:
                    content_hash = hash((d.prompt, d.response))
                    sample_indices.append(f"hash_{content_hash}")

        self._if_sample_indices[policy] = np.array(sample_indices, dtype=object)

        logger.debug(f"Stored {len(sample_indices)} sample indices for {policy}")

    def get_weights(self, policy: str) -> Optional[np.ndarray]:
        """Get importance weights for a policy.

        Args:
            policy: Target policy name

        Returns:
            Array of importance weights or None if not fitted
        """
        if not self._fitted:
            return None
        return self.ips_estimator.get_weights(policy)

    def get_weight_diagnostics(self) -> Optional[IPSDiagnostics]:
        """Get weight diagnostics from internal IPS estimator.

        This helper method provides easy access to weight diagnostics
        for DR estimators, which internally use an IPS estimator for weights.

        Returns:
            IPSDiagnostics object from the internal IPS estimator, or None
        """
        if hasattr(self.ips_estimator, "get_diagnostics"):
            diag = self.ips_estimator.get_diagnostics()
            # Ensure it's IPSDiagnostics (not some other type)
            if isinstance(diag, IPSDiagnostics):
                return diag
        return None

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the estimation.

        Returns:
            Dictionary with diagnostic metrics
        """
        diagnostics: Dict[str, Any] = {
            "weight_method": "calibrated" if self.use_calibrated_weights else "raw",
            "outcome_model": type(self.outcome_model).__name__,
            "n_folds": self.n_folds,
            "policies_with_fresh_draws": list(self._fresh_draws.keys()),
        }

        # Add IPS diagnostics if available
        if hasattr(self.ips_estimator, "get_diagnostics"):
            ips_diag = self.ips_estimator.get_diagnostics()
            # ips_diag is an IPSDiagnostics object, not a dict
            if ips_diag is not None:
                # Convert to dict for legacy compatibility
                diagnostics["ips_weight_ess"] = ips_diag.weight_ess
                diagnostics["ips_n_samples"] = ips_diag.n_samples_valid

        return diagnostics

    def get_oracle_jackknife(self, policy: str) -> Optional[np.ndarray]:
        """Compute leave-one-oracle-fold-out estimates for oracle uncertainty.

        This method computes K estimates, each leaving out one fold of oracle samples,
        to quantify uncertainty from the finite oracle slice used for calibration.

        Args:
            policy: Target policy name

        Returns:
            Array of K jackknife estimates, or None if not applicable

        Note:
            The jackknife variance is computed as: var_oracle = (K-1)/K * Var(estimates)
            This represents the additional uncertainty from learning f: judge → oracle
            from a finite sample of oracle labels.
        """
        # Check if we have the necessary components
        if not self._fitted:
            logger.warning("Estimator not fitted, cannot compute oracle jackknife")
            return None

        if self.reward_calibrator is None:
            logger.debug("No reward_calibrator available for oracle jackknife")
            return None

        # Use unified interface to get fold models
        if not hasattr(self.reward_calibrator, "get_fold_models_for_oua"):
            if self.oua_jackknife:
                raise ValueError(
                    "OUA jackknife is enabled but reward calibrator doesn't support it. "
                    "Ensure calibrate_dataset() uses enable_cross_fit=True."
                )
            return None

        fold_models = self.reward_calibrator.get_fold_models_for_oua()
        if not fold_models:
            if self.oua_jackknife:
                logger.warning(
                    "OUA jackknife is enabled but no fold models available. "
                    "This may happen if calibration mode doesn't support cross-fitting."
                )
            return None

        if policy not in self._fresh_draws:
            logger.warning(
                f"No fresh draws for policy {policy}, cannot compute oracle jackknife"
            )
            return None

        # Cache jackknife results to avoid recomputation
        if not hasattr(self, "_oracle_jackknife_cache"):
            self._oracle_jackknife_cache: Dict[str, np.ndarray] = {}

        if policy in self._oracle_jackknife_cache:
            return self._oracle_jackknife_cache[policy]

        try:
            # Get the number of folds from reward_calibrator
            n_folds = len(fold_models)
            jackknife_estimates = []

            # Get base components that don't change
            fresh_draw_data = self._fresh_draws[policy]
            data = self.sampler.get_data_for_policy(policy)
            if data is None:
                logger.warning(f"No data for policy {policy}")
                return None
            weights = self.ips_estimator.get_weights(policy)

            # For each fold, compute leave-that-fold-out estimate
            for fold_id in range(n_folds):
                # Use the model that was trained WITHOUT this fold's oracle samples
                # The fold_models[fold_id] was trained on all folds EXCEPT fold_id
                fold_model = fold_models.get(fold_id)
                if fold_model is None:
                    logger.debug(f"No fold model for fold {fold_id}")
                    continue

                # Get judge scores for logged data
                # Note: get_data_for_policy flattens judge_score to top level
                # Don't filter - we need to maintain alignment with weights and g_logged
                judge_scores_logged = np.array(
                    [d["judge_score"] for d in data], dtype=float
                )

                if len(judge_scores_logged) == 0:
                    logger.warning(f"No judge scores found in data for fold {fold_id}")
                    continue

                # Check if we need covariates
                needs_covariates = (
                    hasattr(self, "_covariate_names") and self._covariate_names
                )

                # Recalibrate logged rewards with leave-one-out model
                if needs_covariates:
                    # Use reward_calibrator's predict_oof with fold_ids for covariate support
                    logged_covariates = []
                    for d in data:
                        sample_covariates = []
                        for cov_name in self._covariate_names:
                            cov_value = d.get(cov_name)
                            if cov_value is None:
                                raise ValueError(
                                    f"Covariate '{cov_name}' not found or is None in data"
                                )
                            try:
                                sample_covariates.append(float(cov_value))  # type: ignore[arg-type]
                            except (TypeError, ValueError) as e:
                                raise ValueError(
                                    f"Covariate '{cov_name}' has non-numeric value: {e}"
                                )
                        logged_covariates.append(sample_covariates)
                    logged_covariates_array = np.array(logged_covariates)
                    fold_ids_logged = np.full(
                        len(judge_scores_logged), fold_id, dtype=int
                    )
                    logged_rewards_loo = np.clip(
                        self.reward_calibrator.predict_oof(
                            judge_scores_logged,
                            fold_ids_logged,
                            logged_covariates_array,
                        ),
                        0.0,
                        1.0,
                    )
                else:
                    # No covariates - use fold model directly
                    logged_rewards_loo = np.clip(
                        fold_model.predict(judge_scores_logged), 0.0, 1.0
                    )

                # Compute per-prompt fresh draw means (like main estimate() does)
                g_fresh_means = []
                for prompt_id in set(d["prompt_id"] for d in data):
                    prompt_fresh_scores = fresh_draw_data.get_scores_for_prompt_id(
                        prompt_id
                    )
                    if len(prompt_fresh_scores) > 0:
                        if needs_covariates:
                            # Extract covariates from fresh draw metadata
                            prompt_fresh_samples = (
                                fresh_draw_data.get_samples_for_prompt_id(prompt_id)
                            )
                            fresh_cov_values = []
                            for fresh_sample in prompt_fresh_samples:
                                sample_covariates = []
                                for cov_name in self._covariate_names:
                                    sample_covariates.append(
                                        fresh_sample.metadata[cov_name]
                                    )
                                fresh_cov_values.append(sample_covariates)
                            fresh_covariates_array = np.array(fresh_cov_values)
                            fresh_fold_ids = np.full(
                                len(prompt_fresh_scores), fold_id, dtype=int
                            )
                            # Predict with covariates using reward_calibrator
                            prompt_preds = self.reward_calibrator.predict_oof(
                                np.array(prompt_fresh_scores),
                                fresh_fold_ids,
                                fresh_covariates_array,
                            )
                        else:
                            # No covariates - use fold model directly
                            prompt_preds = fold_model.predict(
                                np.array(prompt_fresh_scores)
                            )
                        prompt_preds = np.clip(prompt_preds, 0.0, 1.0)
                        g_fresh_means.append(prompt_preds.mean())

                if len(g_fresh_means) == 0:
                    logger.warning(f"No fresh draw scores found for fold {fold_id}")
                    continue

                # Get outcome model predictions (these use cross-fitted models already)
                g_logged = self._outcome_predictions[policy]

                # Compute leave-one-out DR estimate with per-prompt averaging
                dm_term = float(np.mean(g_fresh_means))
                ips_correction = (weights * (logged_rewards_loo - g_logged)).mean()

                # Note: We're not recomputing augmentation for each fold
                # This is a simplification - proper implementation would recompute
                # augmentation with the leave-one-out reward_calibrator
                # For now, we ignore augmentation in jackknife to focus on main effect

                dr_estimate_loo = dm_term + ips_correction
                jackknife_estimates.append(dr_estimate_loo)

            if len(jackknife_estimates) < 2:
                logger.warning(
                    f"Not enough jackknife estimates for {policy}: {len(jackknife_estimates)}"
                )
                return None

            jackknife_array = np.array(jackknife_estimates)
            self._oracle_jackknife_cache[policy] = jackknife_array

            logger.debug(
                f"Oracle jackknife for {policy}: {len(jackknife_estimates)} estimates, "
                f"mean={jackknife_array.mean():.4f}, std={jackknife_array.std():.4f}"
            )

            return jackknife_array

        except Exception as e:
            logger.error(
                f"Failed to compute oracle jackknife for {policy}: {e}", exc_info=True
            )
            return None


class DRCPOEstimator(DREstimator):
    """DR-CPO: Default DR estimator using isotonic outcome model.

    This is the simplest DR variant that uses g(x,a,s) = f(s) where
    f is the isotonic calibration function learned from judge scores.

    For logged data: Uses cross-fitted predictions f^(-k)(S_i)
    For fresh draws: Uses cross-fitted predictions f^(-k)(S'_i)

    This is theoretically sound under the monotone sufficiency assumption
    (A-J2S) from the paper: E[Y | X, A, S] = μ(S) for some non-decreasing μ.

    By default uses IsotonicOutcomeModel with cross-fitting.
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        outcome_model: Optional[Any] = None,
        n_folds: int = 5,
        use_calibrated_weights: bool = True,
        weight_mode: str = "hajek",
        reward_calibrator: Optional[Any] = None,
        random_seed: int = 42,
        **kwargs: Any,
    ):
        # Pass everything to parent - it will choose the right outcome model
        super().__init__(
            sampler=sampler,
            outcome_model=outcome_model,
            n_folds=n_folds,
            use_calibrated_weights=use_calibrated_weights,
            weight_mode=weight_mode,
            reward_calibrator=reward_calibrator,
            random_seed=random_seed,
            **kwargs,
        )

    def estimate(self) -> EstimationResult:
        """Override to set correct method name."""
        result = super().estimate()
        result.method = "dr_cpo"
        return result
