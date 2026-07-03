"""
Robust inference utilities for Direct-mode evaluation.

Cluster bootstrap with calibrator refit, cluster-robust standard errors,
and paired policy comparisons.
"""

import numpy as np
from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Any,
    Callable,
    TYPE_CHECKING,
    Literal,
    Union,
)
from dataclasses import dataclass, field
from scipy import stats
import logging
import warnings

if TYPE_CHECKING:
    from ..data.fresh_draws import FreshDrawDataset

logger = logging.getLogger(__name__)

BoolLike = Union[bool, np.bool_]


# ========== Residual-Augmented Estimator (AIPW-style) ==========


def compute_augmented_estimate_per_policy(
    calibrated_full: np.ndarray,
    oracle_labels: np.ndarray,
    oracle_mask: np.ndarray,
    oof_predictions: np.ndarray,
    policy_indices: np.ndarray,
    n_policies: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute residual-augmented estimates (θ̂_aug) per policy.

    Implements AIPW-style debiasing:
        θ̂_aug = (1/n)Σᵢ f̂_full(Sᵢ) + (1/n)Σᵢ (Oᵢ/p)(Yᵢ - f̂_oof(Sᵢ))

    Where:
        - f̂_full: Calibrator fitted on ALL oracle-labeled data (lower variance)
        - f̂_oof: Cluster-OOF predictions (unbiased residuals)
        - Oᵢ: Oracle observation indicator
        - p: Oracle sampling probability (estimated per policy)
        - Yᵢ: Oracle label

    The residual term corrects for calibrator bias using the oracle samples,
    with OOF predictions to avoid overfitting.

    Args:
        calibrated_full: (n,) calibrated scores using full model
        oracle_labels: (n,) oracle labels (NaN for unlabeled)
        oracle_mask: (n,) boolean, True if sample has oracle label
        oof_predictions: (n,) OOF predictions for oracle samples (NaN for unlabeled)
        policy_indices: (n,) which policy each sample belongs to
        n_policies: Number of policies

    Returns:
        Tuple of:
        - augmented_estimates: (P,) augmented estimates per policy
        - diagnostics: Dict with per-policy diagnostics
    """
    augmented_estimates = np.zeros(n_policies)
    diagnostics: Dict[str, Any] = {
        "plug_in_estimates": [],
        "residual_corrections": [],
        "oracle_fractions": [],
        "mean_residuals": [],
    }

    for p in range(n_policies):
        p_mask = policy_indices == p
        n_p = np.sum(p_mask)

        if n_p == 0:
            augmented_estimates[p] = np.nan
            continue

        # Policy-specific oracle mask
        p_oracle_mask = p_mask & oracle_mask
        n_oracle_p = np.sum(p_oracle_mask)

        # Oracle sampling probability for this policy
        p_oracle = n_oracle_p / n_p if n_p > 0 else 0.0

        # Plug-in estimate: mean of full-model predictions
        plug_in = np.mean(calibrated_full[p_mask])

        # Residual correction term
        if n_oracle_p > 0 and p_oracle > 0:
            # Residuals: Y - f̂_oof(S) for oracle samples
            residuals = oracle_labels[p_oracle_mask] - oof_predictions[p_oracle_mask]

            # AIPW correction: (1/n) * Σ (1/p) * residual = (1/p) * mean(residual) * (n_oracle/n)
            # Simplifies to: mean(residual)
            residual_correction = np.mean(residuals)
        else:
            residuals = np.array([])
            residual_correction = 0.0

        # Augmented estimate
        augmented_estimates[p] = plug_in + residual_correction

        # Diagnostics
        diagnostics["plug_in_estimates"].append(float(plug_in))
        diagnostics["residual_corrections"].append(float(residual_correction))
        diagnostics["oracle_fractions"].append(float(p_oracle))
        diagnostics["mean_residuals"].append(
            float(np.mean(residuals)) if len(residuals) > 0 else 0.0
        )

    return augmented_estimates, diagnostics


def get_oof_predictions(
    calibrator: Any,
    judge_scores: np.ndarray,
    oracle_mask: np.ndarray,
    covariates: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Get cluster-out-of-fold predictions for oracle samples.

    For each oracle sample, returns the prediction from a calibrator
    trained on data NOT including that sample's cluster/fold.

    NOTE: The calibrator's _fold_ids are aligned with the oracle samples
    (since fit_cv was called with only oracle samples). We need to map
    these back to the original indices.

    Args:
        calibrator: Fitted JudgeCalibrator with fold models
        judge_scores: (n,) judge scores for all samples
        oracle_mask: (n,) boolean mask for oracle samples
        covariates: Optional (n, d) covariate array

    Returns:
        oof_predictions: (n,) OOF predictions (NaN for non-oracle samples)
    """
    n = len(judge_scores)
    oof_predictions = np.full(n, np.nan)

    # Get oracle samples
    oracle_indices = np.where(oracle_mask)[0]
    n_oracle = len(oracle_indices)
    oracle_judge = judge_scores[oracle_mask]
    oracle_cov = covariates[oracle_mask] if covariates is not None else None

    if not hasattr(calibrator, "_fold_ids") or calibrator._fold_ids is None:
        logger.warning("Calibrator has no fold info, using full model for OOF")
        oof_predictions[oracle_indices] = calibrator.predict(
            oracle_judge, covariates=oracle_cov
        )
        return oof_predictions

    # fold_ids is aligned with oracle samples (length = n_oracle)
    fold_ids = calibrator._fold_ids
    if len(fold_ids) != n_oracle:
        logger.warning(
            f"fold_ids length ({len(fold_ids)}) != n_oracle ({n_oracle}), "
            "using full model for OOF"
        )
        oof_predictions[oracle_indices] = calibrator.predict(
            oracle_judge, covariates=oracle_cov
        )
        return oof_predictions

    # Check if flexible calibrator or standard isotonic
    if (
        hasattr(calibrator, "_flexible_calibrator")
        and calibrator._flexible_calibrator is not None
    ):
        # Use flexible calibrator's OOF predictions
        oof_preds = calibrator._flexible_calibrator.predict(
            oracle_judge, fold_ids, oracle_cov
        )
        oof_predictions[oracle_indices] = np.clip(oof_preds, 0.0, 1.0)
    elif hasattr(calibrator, "_fold_models") and calibrator._fold_models:
        # Standard isotonic per-fold models
        # fold_ids is aligned with oracle samples, so iterate over oracle indices
        for fold_id, model in calibrator._fold_models.items():
            # Find which oracle samples belong to this fold
            fold_oracle_mask = fold_ids == fold_id
            if np.any(fold_oracle_mask):
                fold_oracle_indices = oracle_indices[fold_oracle_mask]
                fold_oracle_judge = oracle_judge[fold_oracle_mask]
                preds = model.predict(fold_oracle_judge)
                oof_predictions[fold_oracle_indices] = np.clip(preds, 0.0, 1.0)
    else:
        logger.warning("No fold models available, using full model")
        oof_predictions[oracle_indices] = calibrator.predict(
            oracle_judge, covariates=oracle_cov
        )

    return oof_predictions


# ========== Direct Mode Bootstrap Data Structures ==========


@dataclass
class DirectEvalTable:
    """Long-format evaluation table for Direct mode bootstrap.

    This structure enables efficient cluster bootstrap by:
    1. Storing all data in long format (one row per policy-prompt pair)
    2. Precomputing cluster-to-row mappings for O(1) lookup during resampling
    3. Supporting covariates for two-stage calibration

    The bootstrap resamples prompt clusters (shared across policies for paired
    designs), which preserves the correlation structure needed for valid
    pairwise comparisons.
    """

    # Core data arrays (all length n_total = sum of samples across policies)
    prompt_ids: np.ndarray  # (n_total,) cluster identifier (string hashes as int)
    prompt_id_strings: List[str]  # (n_total,) original prompt_id strings
    policy_indices: np.ndarray  # (n_total,) which policy (0, 1, ...)
    judge_scores: np.ndarray  # (n_total,) raw judge scores
    oracle_labels: np.ndarray  # (n_total,) NaN if unlabeled
    oracle_mask: np.ndarray  # (n_total,) boolean, True if labeled

    # Covariates (optional, for two-stage calibration)
    covariates: Optional[np.ndarray]  # (n_total, n_cov) or None
    covariate_names: Optional[List[str]]

    # Precomputed indices for efficient bootstrap (O(1) lookup)
    cluster_to_rows: Dict[int, np.ndarray] = field(default_factory=dict)

    # Metadata
    policy_names: List[str] = field(default_factory=list)
    n_clusters: int = 0
    n_policies: int = 0
    unique_clusters: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        """Compute derived fields after initialization."""
        if len(self.cluster_to_rows) == 0 and len(self.prompt_ids) > 0:
            # Build cluster-to-rows mapping
            self.unique_clusters = np.unique(self.prompt_ids)
            self.n_clusters = len(self.unique_clusters)
            self.n_policies = len(self.policy_names) if self.policy_names else 0

            # Precompute row indices for each cluster (single pass)
            # This avoids O(G*n) lookups during bootstrap
            for cluster_id in self.unique_clusters:
                self.cluster_to_rows[int(cluster_id)] = np.where(
                    self.prompt_ids == cluster_id
                )[0]


def build_direct_eval_table(
    fresh_draws_per_policy: Dict[str, "FreshDrawDataset"],
    covariate_names: Optional[List[str]] = None,
) -> DirectEvalTable:
    """Build evaluation table from fresh draws for efficient bootstrap.

    Creates a long-format table with precomputed cluster indices for O(1)
    bootstrap resampling. All policies are stacked vertically with a
    policy_indices column to identify which policy each row belongs to.

    Args:
        fresh_draws_per_policy: Dict mapping policy names to FreshDrawDataset
        covariate_names: Optional list of covariate names to extract from
                         sample.metadata (e.g., ["response_length"])

    Returns:
        DirectEvalTable ready for cluster bootstrap
    """
    if not fresh_draws_per_policy:
        raise ValueError("fresh_draws_per_policy cannot be empty")

    policy_names = list(fresh_draws_per_policy.keys())
    n_policies = len(policy_names)

    # Collect arrays from each policy
    all_prompt_ids: List[str] = []
    all_policy_indices: List[int] = []
    all_judge_scores: List[float] = []
    all_oracle_labels: List[float] = []
    all_covariates: List[List[float]] = []

    for policy_idx, policy_name in enumerate(policy_names):
        fd = fresh_draws_per_policy[policy_name]
        for sample in fd.samples:
            all_prompt_ids.append(sample.prompt_id)
            all_policy_indices.append(policy_idx)
            all_judge_scores.append(sample.judge_score)

            # Oracle label: use NaN if not present
            if sample.oracle_label is not None:
                all_oracle_labels.append(sample.oracle_label)
            else:
                all_oracle_labels.append(np.nan)

            # Extract covariates if requested
            if covariate_names:
                row_covs = []
                for cov_name in covariate_names:
                    if cov_name in sample.metadata:
                        row_covs.append(float(sample.metadata[cov_name]))
                    else:
                        row_covs.append(np.nan)
                all_covariates.append(row_covs)

    # Convert to numpy arrays
    prompt_id_strings = all_prompt_ids
    # Map prompt_ids to sequential integers (deterministic factorization)
    # Using dict-based factorization instead of hash() because:
    #   1. hash() is non-deterministic across Python processes (PYTHONHASHSEED)
    #   2. hash() % N can have collisions, incorrectly merging clusters
    # This approach guarantees zero collisions and deterministic mapping.
    unique_prompts = list(dict.fromkeys(all_prompt_ids))  # Preserves order
    prompt_to_idx = {pid: idx for idx, pid in enumerate(unique_prompts)}
    prompt_ids = np.array(
        [prompt_to_idx[pid] for pid in all_prompt_ids], dtype=np.int64
    )
    policy_indices = np.array(all_policy_indices, dtype=np.int32)
    judge_scores = np.array(all_judge_scores, dtype=np.float64)
    oracle_labels = np.array(all_oracle_labels, dtype=np.float64)
    oracle_mask = ~np.isnan(oracle_labels)

    # Covariates array
    covariates: Optional[np.ndarray] = None
    if covariate_names and all_covariates:
        covariates = np.array(all_covariates, dtype=np.float64)

    return DirectEvalTable(
        prompt_ids=prompt_ids,
        prompt_id_strings=prompt_id_strings,
        policy_indices=policy_indices,
        judge_scores=judge_scores,
        oracle_labels=oracle_labels,
        oracle_mask=oracle_mask,
        covariates=covariates,
        covariate_names=covariate_names,
        policy_names=policy_names,
        n_policies=n_policies,
    )


def make_calibrator_factory(
    mode: Literal["monotone", "two_stage"],
    covariate_names: Optional[List[str]] = None,
    seed: int = 42,
) -> Callable[[], Any]:
    """Create a factory function that produces fresh JudgeCalibrator instances.

    This factory pattern ensures each bootstrap replicate gets a completely
    fresh calibrator instance, avoiding any state leakage between replicates.

    The mode should be FIXED to the mode selected on full data, not "auto".
    This focuses bootstrap on capturing calibration/evaluation covariance
    without adding unnecessary variability from mode re-selection.

    Args:
        mode: Calibration mode ('monotone' or 'two_stage'). Should NOT be 'auto'
              during bootstrap - use the selected_mode from full-data calibration.
        covariate_names: Optional list of covariate names for two-stage calibration
        seed: Random seed for reproducibility

    Returns:
        Callable that creates a new JudgeCalibrator instance on each call
    """
    from ..calibration.judge import JudgeCalibrator

    def factory() -> Any:
        return JudgeCalibrator(
            random_seed=seed,
            calibration_mode=mode,  # Fixed mode, not "auto"
            covariate_names=covariate_names,
        )

    return factory


def cluster_bootstrap_direct_with_refit(
    eval_table: DirectEvalTable,
    calibrator_factory: Callable[[], Any],
    n_bootstrap: int = 2000,
    min_oracle_per_replicate: int = 30,
    alpha: float = 0.05,
    seed: int = 42,
    use_augmented_estimator: bool = True,
    calibration_policy_idx: Optional[int] = None,
    use_multipolicy_eif: Optional[BoolLike] = None,
) -> Dict[str, Any]:
    """Cluster bootstrap with calibrator refit for Direct mode.

    This bootstrap procedure captures:
    1. Prompt sampling variance (by resampling clusters)
    2. Calibrator uncertainty (by refitting on each replicate's oracle subset)
    3. Calibration/evaluation covariance (the key term missing from analytic SEs)

    The algorithm uses "resample-until-valid" to avoid conditioning on
    "easy bootstrap worlds" with fewer oracle labels, which would shrink CIs.

    When use_augmented_estimator=True (default), uses θ̂_aug (AIPW-style) which
    debiases the plug-in estimator using cluster-out-of-fold predictions:
        θ̂_aug = mean(f̂_full(S)) + mean(Y - f̂_oof(S))
    This corrects for calibrator bias while maintaining valid bootstrap inference.

    Args:
        eval_table: DirectEvalTable from build_direct_eval_table()
        calibrator_factory: Factory that creates fresh JudgeCalibrator instances
            (should have mode fixed to full-data selection, not "auto")
        n_bootstrap: Number of valid bootstrap replicates to collect (default 2000)
        min_oracle_per_replicate: Minimum oracle labels required per replicate (default 30)
        alpha: Significance level for confidence intervals (default 0.05)
        seed: Random seed for reproducibility
        use_augmented_estimator: If True (default), use θ̂_aug (AIPW-style debiasing).
            If False, use plug-in estimator (mean of calibrated scores).
        calibration_policy_idx: If provided, fit calibrator only on this policy's
            oracle samples (for transport experiments). Residual corrections in θ̂_aug
            still use all policies' oracle samples. If None, use all oracle samples
            for both calibration and residuals (default behavior).
        use_multipolicy_eif: Legacy compatibility shim. Multi-policy EIF has been
            removed. Passing True raises a ValueError. Passing False emits a
            deprecation warning and is otherwise ignored.

    Returns:
        Dictionary with:
        - bootstrap_matrix: (B, P) array of policy means per replicate
        - estimates: (P,) point estimates from full data
        - standard_errors: (P,) bootstrap standard errors
        - ci_lower, ci_upper: (P,) percentile confidence interval bounds
        - n_valid_replicates: number of successful replicates
        - n_attempts: total attempts made (for skip rate calculation)
        - skip_rate: fraction of attempts that were invalid
        - oracle_count_summary: min/p10/median oracle counts across replicates
        - metadata: additional diagnostic information
        - augmentation_diagnostics: (if use_augmented_estimator) per-policy diagnostics
    """
    if use_multipolicy_eif is not None and not isinstance(
        use_multipolicy_eif, (bool, np.bool_)
    ):
        raise TypeError(
            "use_multipolicy_eif must be a boolean or None. "
            f"Got {type(use_multipolicy_eif).__name__}."
        )

    if use_multipolicy_eif is not None and bool(use_multipolicy_eif):
        raise ValueError(
            "use_multipolicy_eif=True is no longer supported. "
            "CJE now uses per-policy residual correction only."
        )
    if use_multipolicy_eif is not None and not bool(use_multipolicy_eif):
        warnings.warn(
            "use_multipolicy_eif is deprecated and ignored. "
            "CJE now uses per-policy residual correction only.",
            FutureWarning,
            stacklevel=2,
        )

    rng = np.random.default_rng(seed)

    # Extract arrays from eval table
    prompt_ids = eval_table.prompt_ids
    prompt_id_strings = eval_table.prompt_id_strings
    policy_indices = eval_table.policy_indices
    judge_scores = eval_table.judge_scores
    oracle_labels = eval_table.oracle_labels
    oracle_mask = eval_table.oracle_mask
    covariates = eval_table.covariates
    unique_clusters = eval_table.unique_clusters
    cluster_to_rows = eval_table.cluster_to_rows
    n_policies = eval_table.n_policies
    n_clusters = eval_table.n_clusters

    # Compute separate masks for calibration vs residuals (transport experiments)
    # - calibration_oracle_mask: used for fitting the calibrator
    # - oracle_mask: used for computing residual corrections in θ̂_aug
    if calibration_policy_idx is not None:
        calibration_oracle_mask = oracle_mask & (
            policy_indices == calibration_policy_idx
        )
        n_cal_oracle = int(np.sum(calibration_oracle_mask))
        n_total_oracle = int(np.sum(oracle_mask))
        logger.info(
            f"Transport mode: calibrating on policy {calibration_policy_idx} "
            f"({n_cal_oracle} oracle samples), residuals on all policies "
            f"({n_total_oracle} oracle samples)"
        )
    else:
        calibration_oracle_mask = oracle_mask

    # Compute point estimates from full data (single calibrator for all policies)
    full_estimates = np.zeros(n_policies)
    augmentation_diagnostics: Optional[Dict[str, Any]] = None
    calibrated_full: Optional[np.ndarray] = None

    try:
        calibrator = calibrator_factory()

        # Pass prompt_id_strings for cluster-level fold assignment
        # This ensures OOF predictions use cluster-level cross-fitting
        # Use calibration_oracle_mask (may be subset if calibration_policy_idx is set)
        cal_oracle_prompt_ids = [
            prompt_id_strings[i] for i in np.where(calibration_oracle_mask)[0]
        ]

        n_cal_oracle_full = int(np.sum(calibration_oracle_mask))
        calibrator.fit_cv(
            judge_scores=judge_scores[calibration_oracle_mask],
            oracle_labels=oracle_labels[calibration_oracle_mask],
            n_folds=min(
                5, max(2, n_cal_oracle_full // 2)
            ),  # Reduce folds if low oracle
            prompt_ids=cal_oracle_prompt_ids,  # For cluster-level folds
            covariates=(
                covariates[calibration_oracle_mask] if covariates is not None else None
            ),
        )

        # Predict calibrated rewards for all samples (full model)
        calibrated_full = calibrator.predict(judge_scores, covariates=covariates)

        if use_augmented_estimator:
            # Use θ̂_aug (AIPW-style debiasing)
            # Get cluster-out-of-fold predictions for CALIBRATION oracle samples
            # (fold_ids will match calibration_oracle_mask length)
            oof_predictions = get_oof_predictions(
                calibrator, judge_scores, calibration_oracle_mask, covariates
            )

            # For NON-CALIBRATION oracle (target policies), use full-model predictions
            # This captures transport bias correction via the residual term
            if calibration_policy_idx is not None:
                non_calibration_oracle = oracle_mask & ~calibration_oracle_mask
                if np.any(non_calibration_oracle):
                    non_cal_indices = np.where(non_calibration_oracle)[0]
                    non_cal_judge = judge_scores[non_calibration_oracle]
                    non_cal_cov = (
                        covariates[non_calibration_oracle]
                        if covariates is not None
                        else None
                    )
                    oof_predictions[non_cal_indices] = calibrator.predict(
                        non_cal_judge, covariates=non_cal_cov
                    )

            # Compute augmented estimates using FULL oracle_mask (all policies)
            # The residual correction will capture transport bias for target policies
            full_estimates, augmentation_diagnostics = (
                compute_augmented_estimate_per_policy(
                    calibrated_full=calibrated_full,
                    oracle_labels=oracle_labels,
                    oracle_mask=oracle_mask,  # All policies for residuals
                    oof_predictions=oof_predictions,
                    policy_indices=policy_indices,
                    n_policies=n_policies,
                )
            )
            logger.info(
                f"Using augmented estimator: residual corrections = "
                f"{augmentation_diagnostics['residual_corrections']}"
            )
        else:
            # Use plug-in estimator (mean of calibrated scores)
            for p in range(n_policies):
                p_mask = policy_indices == p
                if np.any(p_mask):
                    full_estimates[p] = np.mean(calibrated_full[p_mask])

    except Exception as e:
        logger.warning(f"Full data calibration failed: {e}")
        full_estimates[:] = np.nan

    # Bootstrap loop with resample-until-valid
    bootstrap_matrix = np.zeros((n_bootstrap, n_policies))
    oracle_counts: List[int] = []
    valid_count = 0
    attempt = 0
    max_attempts = 5 * n_bootstrap  # Cap to prevent infinite loops

    while valid_count < n_bootstrap and attempt < max_attempts:
        attempt += 1

        # 1. Resample prompt clusters with replacement
        sampled_cluster_ids = rng.choice(unique_clusters, size=n_clusters, replace=True)

        # 2. Get all rows for sampled clusters
        bootstrap_rows: List[int] = []
        for cluster_id in sampled_cluster_ids:
            rows = cluster_to_rows.get(int(cluster_id), np.array([], dtype=int))
            bootstrap_rows.extend(rows.tolist())
        bootstrap_rows_arr = np.array(bootstrap_rows, dtype=int)

        if len(bootstrap_rows_arr) == 0:
            continue

        # 3. Extract bootstrap subset
        boot_judge = judge_scores[bootstrap_rows_arr]
        boot_oracle = oracle_labels[bootstrap_rows_arr]
        boot_oracle_mask = oracle_mask[bootstrap_rows_arr]
        boot_policy = policy_indices[bootstrap_rows_arr]
        boot_prompt_ids = [prompt_id_strings[i] for i in bootstrap_rows_arr]
        boot_covariates = (
            covariates[bootstrap_rows_arr] if covariates is not None else None
        )

        # 3b. Compute bootstrap calibration mask (subset if calibration_policy_idx set)
        boot_calibration_mask = calibration_oracle_mask[bootstrap_rows_arr]

        # 4. Check CALIBRATION oracle count - retry if too few
        # (we need enough calibration oracle for the calibrator to fit)
        n_cal_oracle_boot = int(np.sum(boot_calibration_mask))
        if n_cal_oracle_boot < min_oracle_per_replicate:
            continue

        # 5. Refit calibrator on bootstrap CALIBRATION oracle subset
        try:
            boot_calibrator = calibrator_factory()
            boot_cal_oracle_prompt_ids = [
                boot_prompt_ids[i] for i in np.where(boot_calibration_mask)[0]
            ]

            boot_calibrator.fit_cv(
                judge_scores=boot_judge[boot_calibration_mask],
                oracle_labels=boot_oracle[boot_calibration_mask],
                n_folds=min(5, n_cal_oracle_boot // 4),  # Reduce folds if low oracle
                prompt_ids=boot_cal_oracle_prompt_ids,  # For cluster-level folds
                covariates=(
                    boot_covariates[boot_calibration_mask]
                    if boot_covariates is not None
                    else None
                ),
            )
        except Exception as e:
            logger.debug(f"Bootstrap replicate {attempt} calibration failed: {e}")
            continue

        # 6. Predict calibrated rewards on bootstrap evaluation sample
        try:
            boot_rewards = boot_calibrator.predict(
                boot_judge, covariates=boot_covariates
            )
        except Exception as e:
            logger.debug(f"Bootstrap replicate {attempt} prediction failed: {e}")
            continue

        # 7. Compute policy estimates (augmented or plug-in)
        if use_augmented_estimator:
            # Use θ̂_aug for bootstrap replicate
            # Get OOF predictions for CALIBRATION oracle (fold_ids will match)
            boot_oof_preds = get_oof_predictions(
                boot_calibrator, boot_judge, boot_calibration_mask, boot_covariates
            )

            # For NON-CALIBRATION oracle, use full-model predictions
            if calibration_policy_idx is not None:
                boot_non_cal_mask = boot_oracle_mask & ~boot_calibration_mask
                if np.any(boot_non_cal_mask):
                    boot_non_cal_indices = np.where(boot_non_cal_mask)[0]
                    boot_non_cal_judge = boot_judge[boot_non_cal_mask]
                    boot_non_cal_cov = (
                        boot_covariates[boot_non_cal_mask]
                        if boot_covariates is not None
                        else None
                    )
                    boot_oof_preds[boot_non_cal_indices] = boot_calibrator.predict(
                        boot_non_cal_judge, covariates=boot_non_cal_cov
                    )

            # Compute augmented estimates using FULL boot_oracle_mask
            means_p, _ = compute_augmented_estimate_per_policy(
                calibrated_full=boot_rewards,
                oracle_labels=boot_oracle,
                oracle_mask=boot_oracle_mask,  # All policies for residuals
                oof_predictions=boot_oof_preds,
                policy_indices=boot_policy,
                n_policies=n_policies,
            )
        else:
            # Plug-in estimator via bincount (efficient)
            sum_p = np.bincount(boot_policy, weights=boot_rewards, minlength=n_policies)
            cnt_p = np.bincount(boot_policy, minlength=n_policies)

            # Handle zero counts (shouldn't happen in paired design, but be safe)
            with np.errstate(divide="ignore", invalid="ignore"):
                means_p = np.where(cnt_p > 0, sum_p / cnt_p, np.nan)

        bootstrap_matrix[valid_count, :] = means_p

        oracle_counts.append(n_cal_oracle_boot)  # Track calibration oracle count
        valid_count += 1

    # Check if we got enough valid replicates
    if valid_count < n_bootstrap:
        logger.warning(
            f"Bootstrap only collected {valid_count}/{n_bootstrap} valid replicates "
            f"after {attempt} attempts. Results may be less reliable."
        )
        # Trim matrix to actual valid count
        bootstrap_matrix = bootstrap_matrix[:valid_count, :]

    # Compute standard errors and CIs
    if valid_count > 1:
        standard_errors = np.nanstd(bootstrap_matrix, axis=0, ddof=1)
        ci_lower = np.nanpercentile(bootstrap_matrix, 100 * alpha / 2, axis=0)
        ci_upper = np.nanpercentile(bootstrap_matrix, 100 * (1 - alpha / 2), axis=0)
    else:
        standard_errors = np.full(n_policies, np.nan)
        ci_lower = np.full(n_policies, np.nan)
        ci_upper = np.full(n_policies, np.nan)

    # Oracle count summary
    oracle_summary = {}
    if oracle_counts:
        oracle_summary = {
            "min": int(np.min(oracle_counts)),
            "p10": int(np.percentile(oracle_counts, 10)),
            "median": int(np.median(oracle_counts)),
        }

    # Skip rate
    skip_rate = (attempt - valid_count) / attempt if attempt > 0 else 0.0

    # Use simple percentile intervals (BCa removed - negligible benefit, expensive)
    # The ~95% coverage comes from θ̂_aug + bootstrap refit, not BCa corrections

    policy_names = eval_table.policy_names

    return {
        "bootstrap_matrix": bootstrap_matrix,
        "estimates": full_estimates,
        "standard_errors": standard_errors,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_valid_replicates": valid_count,
        "n_attempts": attempt,
        "skip_rate": float(skip_rate),
        "oracle_count_summary": oracle_summary,
        "policy_names": policy_names,
        "n_clusters": n_clusters,
        "n_policies": n_policies,
        "alpha": alpha,
        "seed": seed,
        "min_oracle_per_replicate": min_oracle_per_replicate,
        "use_augmented_estimator": use_augmented_estimator,
        "augmentation_diagnostics": augmentation_diagnostics,
        "calibration_policy_idx": calibration_policy_idx,  # For transport experiments
    }


def compare_policies_bootstrap(
    bootstrap_result: Dict[str, Any],
    policy_a: int,
    policy_b: int,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute pairwise policy comparison from bootstrap results.

    This uses the stored bootstrap matrix to compute contrasts between policies.
    Because the same cluster resampling was used for both policies in each
    replicate, the correlation structure is preserved, yielding tighter CIs
    for paired designs than naive independence assumptions.

    Args:
        bootstrap_result: Result from cluster_bootstrap_direct_with_refit()
        policy_a: Index of first policy
        policy_b: Index of second policy
        alpha: Significance level for CI

    Returns:
        Dictionary with:
        - diff_estimate: point estimate of difference (a - b)
        - diff_se: bootstrap SE of difference
        - ci_lower, ci_upper: percentile CI bounds
        - p_value: two-sided bootstrap p-value
    """
    bootstrap_matrix = bootstrap_result["bootstrap_matrix"]
    estimates = bootstrap_result["estimates"]

    # Compute difference distribution
    diff_samples = bootstrap_matrix[:, policy_a] - bootstrap_matrix[:, policy_b]
    diff_estimate = estimates[policy_a] - estimates[policy_b]

    # Bootstrap SE and CI
    diff_se = np.nanstd(diff_samples, ddof=1)
    ci_lower = np.nanpercentile(diff_samples, 100 * alpha / 2)
    ci_upper = np.nanpercentile(diff_samples, 100 * (1 - alpha / 2))

    # Two-sided bootstrap p-value (fraction of replicates on opposite side of 0)
    n_valid = np.sum(~np.isnan(diff_samples))
    if diff_estimate >= 0:
        p_value = 2 * np.nanmean(diff_samples <= 0)
    else:
        p_value = 2 * np.nanmean(diff_samples >= 0)
    p_value = min(p_value, 1.0)  # Cap at 1.0

    return {
        "diff_estimate": float(diff_estimate),
        "diff_se": float(diff_se),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
        "policy_a": policy_a,
        "policy_b": policy_b,
        "n_valid_replicates": int(n_valid),
    }


# ========== Dependence-Robust Standard Errors ==========


def cluster_robust_se(
    data: np.ndarray,
    cluster_ids: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    influence_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute cluster-robust (sandwich) standard errors with CRV1 correction.

    For data with cluster structure (e.g., multiple obs per user),
    accounts for within-cluster correlation using the CRV1 variance estimator.

    Args:
        data: Input data array
        cluster_ids: Cluster membership for each observation
        statistic_fn: Function that computes the statistic
        influence_fn: Function that computes influence functions
        alpha: Significance level for CI

    Returns:
        Dictionary with robust standard errors and t-based confidence intervals
    """
    n = len(data)
    if n == 0:
        return {
            "estimate": float("nan"),
            "se": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_clusters": 0,
            "df": 0,
        }

    estimate = statistic_fn(data)

    # Build influence contributions
    if influence_fn is None:
        # Default: sample-mean statistic -> IF is (x_i - mean)
        if data.ndim != 1:
            raise ValueError(
                "For multi-dimensional data, provide influence_fn. "
                "Default influence function only works for 1-D data."
            )
        influences = (data - estimate).astype(float, copy=False)
    else:
        # Use provided influence function
        influences = influence_fn(data).astype(float, copy=False)

    # Center defensively for numerical stability
    influences = influences - np.mean(influences)

    # Get unique clusters
    clusters = np.asarray(cluster_ids, dtype=int)
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    df = max(G - 1, 1)

    if G < 2:
        # Fallback to naive SE if we don't have clustering
        se_naive = float(np.std(influences, ddof=1) / np.sqrt(n))
        t_crit = stats.t.ppf(1 - alpha / 2, df=max(n - 1, 1))
        return {
            "estimate": float(estimate),
            "se": se_naive,
            "ci_lower": float(estimate - t_crit * se_naive),
            "ci_upper": float(estimate + t_crit * se_naive),
            "n_clusters": int(G),
            "df": int(max(n - 1, 1)),
        }

    # Cluster totals of IF
    T = np.array(
        [np.sum(influences[clusters == g]) for g in unique_clusters], dtype=float
    )
    T = T - T.mean()  # Center across clusters

    # CRV1 variance for a mean-type estimator (with G/(G-1) factor):
    var_hat = (G / (G - 1)) * np.sum(T**2) / (n**2)
    se = float(np.sqrt(max(var_hat, 0.0)))

    # Confidence interval using t-distribution with G - 1 df
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_lower = estimate - t_crit * se
    ci_upper = estimate + t_crit * se

    return {
        "estimate": float(estimate),
        "se": se,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_clusters": int(G),
        "df": int(df),
    }
