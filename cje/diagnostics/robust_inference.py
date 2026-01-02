"""
Robust inference utilities for handling dependence and multiple testing.

Implements dependence-robust standard errors and FDR control as specified
in Section 9.4 of the CJE paper.
"""

import numpy as np
from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Any,
    Callable,
    Union,
    TYPE_CHECKING,
    Literal,
)
from dataclasses import dataclass, field
from scipy import stats
import logging

if TYPE_CHECKING:
    from ..data.fresh_draws import FreshDrawDataset

logger = logging.getLogger(__name__)


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
    # Hash prompt_ids to integers for efficient comparison
    prompt_ids = np.array(
        [hash(pid) % (2**31) for pid in all_prompt_ids], dtype=np.int64
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

    # Compute point estimates from full data (single calibrator for all policies)
    full_estimates = np.zeros(n_policies)
    augmentation_diagnostics: Optional[Dict[str, Any]] = None
    calibrated_full: Optional[np.ndarray] = None

    try:
        calibrator = calibrator_factory()

        # Pass prompt_id_strings for cluster-level fold assignment
        # This ensures OOF predictions use cluster-level cross-fitting
        oracle_prompt_ids = [prompt_id_strings[i] for i in np.where(oracle_mask)[0]]

        calibrator.fit_cv(
            judge_scores=judge_scores[oracle_mask],
            oracle_labels=oracle_labels[oracle_mask],
            n_folds=5,
            prompt_ids=oracle_prompt_ids,  # For cluster-level folds
            covariates=(covariates[oracle_mask] if covariates is not None else None),
        )

        # Predict calibrated rewards for all samples (full model)
        calibrated_full = calibrator.predict(judge_scores, covariates=covariates)

        if use_augmented_estimator:
            # Use θ̂_aug (AIPW-style debiasing)
            # Get cluster-out-of-fold predictions for oracle samples
            oof_predictions = get_oof_predictions(
                calibrator, judge_scores, oracle_mask, covariates
            )

            # Compute augmented estimates
            full_estimates, augmentation_diagnostics = (
                compute_augmented_estimate_per_policy(
                    calibrated_full=calibrated_full,
                    oracle_labels=oracle_labels,
                    oracle_mask=oracle_mask,
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

        # 4. Check oracle count - retry if too few
        n_oracle_boot = int(np.sum(boot_oracle_mask))
        if n_oracle_boot < min_oracle_per_replicate:
            continue

        # 5. Refit calibrator on bootstrap oracle subset (mode is fixed, not "auto")
        try:
            boot_calibrator = calibrator_factory()
            boot_oracle_prompt_ids = [
                boot_prompt_ids[i] for i in np.where(boot_oracle_mask)[0]
            ]

            boot_calibrator.fit_cv(
                judge_scores=boot_judge[boot_oracle_mask],
                oracle_labels=boot_oracle[boot_oracle_mask],
                n_folds=min(5, n_oracle_boot // 4),  # Reduce folds if low oracle count
                prompt_ids=boot_oracle_prompt_ids,  # For cluster-level folds
                covariates=(
                    boot_covariates[boot_oracle_mask]
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
            boot_oof_preds = get_oof_predictions(
                boot_calibrator, boot_judge, boot_oracle_mask, boot_covariates
            )
            means_p, _ = compute_augmented_estimate_per_policy(
                calibrated_full=boot_rewards,
                oracle_labels=boot_oracle,
                oracle_mask=boot_oracle_mask,
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
        oracle_counts.append(n_oracle_boot)
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
        "policy_names": eval_table.policy_names,
        "n_clusters": n_clusters,
        "n_policies": n_policies,
        "alpha": alpha,
        "seed": seed,
        "min_oracle_per_replicate": min_oracle_per_replicate,
        "use_augmented_estimator": use_augmented_estimator,
        "augmentation_diagnostics": augmentation_diagnostics,
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


def stationary_bootstrap_se(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 4000,
    mean_block_length: Optional[float] = None,
    alpha: float = 0.05,
    return_distribution: bool = False,
) -> Dict[str, Any]:
    """Compute standard errors using stationary bootstrap for time series.

    The stationary bootstrap (Politis & Romano, 1994) resamples blocks of
    random length with geometric distribution, preserving weak dependence.

    Args:
        data: Input data array (n_samples, ...)
        statistic_fn: Function that computes the statistic of interest
        n_bootstrap: Number of bootstrap iterations (default 4000)
        mean_block_length: Expected block length (auto if None)
        alpha: Significance level for CI (default 0.05 for 95% CI)
        return_distribution: If True, return bootstrap distribution

    Returns:
        Dictionary with:
        - 'estimate': Point estimate
        - 'se': Bootstrap standard error
        - 'ci_lower': Lower CI bound
        - 'ci_upper': Upper CI bound
        - 'mean_block_length': Block length used
        - 'distribution': Bootstrap distribution (if requested)

    References:
        Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
        Journal of the American Statistical Association, 89(428), 1303-1313.
    """
    n = len(data)

    # Compute point estimate
    estimate = statistic_fn(data)

    # Determine block length if not provided
    if mean_block_length is None:
        # Use first-order autocorrelation to tune block length
        # Rule of thumb: block_length ≈ n^(1/3) * (ρ/(1-ρ))^(2/3)
        if data.ndim == 1:
            acf_1 = np.corrcoef(data[:-1], data[1:])[0, 1] if n > 1 else 0.0
        else:
            # For multi-dimensional, use first column
            acf_1 = np.corrcoef(data[:-1, 0], data[1:, 0])[0, 1] if n > 1 else 0.0

        # Guard against NaN from constant series
        if not np.isfinite(acf_1):
            acf_1 = 0.0

        # Ensure reasonable bounds
        acf_1 = float(np.clip(acf_1, -0.99, 0.99))

        base = max(1, int(round(n ** (1.0 / 3.0))))
        if abs(acf_1) < 0.1:
            # Weak dependence, use smaller blocks
            mean_block_length = base
        else:
            # Stronger dependence, use larger blocks
            mean_block_length = max(
                1, int(round(base * (abs(acf_1) / (1 - abs(acf_1))) ** 0.67))
            )

        # Cap but never drop below 1
        cap = max(1, n // 4)
        mean_block_length = min(mean_block_length, cap)

    # Probability of starting a new block
    p = 1.0 / float(mean_block_length)

    # Bootstrap iterations
    bootstrap_estimates: List[float] = []

    for _ in range(n_bootstrap):
        # Proper stationary bootstrap: start at random position,
        # then with prob p jump to a new random start; otherwise continue.
        bootstrap_indices: List[int] = []
        i = np.random.randint(0, n)

        while len(bootstrap_indices) < n:
            # With probability p, jump to a new random start
            if np.random.random() < p:
                i = np.random.randint(0, n)
            bootstrap_indices.append(i)
            i = (i + 1) % n  # Wrap around

        bootstrap_sample = data[bootstrap_indices]

        # Compute statistic on bootstrap sample
        try:
            boot_stat = statistic_fn(bootstrap_sample)
            bootstrap_estimates.append(boot_stat)
        except Exception as e:
            # Skip failed iterations (can happen with small samples)
            logger.debug(f"Bootstrap iteration failed: {e}")
            continue

    bootstrap_estimates_array = np.array(bootstrap_estimates)

    if bootstrap_estimates_array.size < 2:
        # Safe fallback for very small n or if too many failures
        return {
            "estimate": float(estimate),
            "se": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "mean_block_length": float(mean_block_length),
            "n_bootstrap": int(bootstrap_estimates_array.size),
            "effective_samples": int(bootstrap_estimates_array.size),
        }

    # Compute standard error
    se = np.std(bootstrap_estimates_array, ddof=1)

    # Compute confidence interval (percentile method)
    ci_lower = np.percentile(bootstrap_estimates_array, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates_array, 100 * (1 - alpha / 2))

    result: Dict[str, Any] = {
        "estimate": float(estimate),
        "se": float(se),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "mean_block_length": float(mean_block_length),
        "n_bootstrap": len(bootstrap_estimates_array),
        "effective_samples": len(bootstrap_estimates_array),
    }

    if return_distribution:
        result["distribution"] = bootstrap_estimates_array

    return result


def moving_block_bootstrap_se(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 4000,
    block_length: Optional[int] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute standard errors using moving block bootstrap.

    The moving block bootstrap (Künsch, 1989) resamples fixed-length
    contiguous blocks, preserving local dependence structure.

    Args:
        data: Input data array
        statistic_fn: Function that computes the statistic
        n_bootstrap: Number of bootstrap iterations
        block_length: Fixed block length (auto if None)
        alpha: Significance level for CI

    Returns:
        Dictionary with bootstrap results

    References:
        Künsch, H. R. (1989). The jackknife and the bootstrap for general
        stationary observations. The Annals of Statistics, 17(3), 1217-1241.
    """
    n = len(data)

    # Compute point estimate
    estimate = statistic_fn(data)

    # Determine block length if not provided
    if block_length is None:
        # Standard choice: n^(1/3) for optimal MSE
        block_length = max(1, int(n ** (1.0 / 3.0)))

    # Number of blocks needed
    n_blocks = int(np.ceil(n / block_length))

    # Bootstrap iterations
    bootstrap_estimates: List[float] = []

    for _ in range(n_bootstrap):
        # Sample blocks with replacement
        bootstrap_indices = []

        for _ in range(n_blocks):
            # Random starting point for block
            start = np.random.randint(0, n - block_length + 1)
            block_indices = list(range(start, min(start + block_length, n)))
            bootstrap_indices.extend(block_indices)

        # Trim to original length
        bootstrap_indices = bootstrap_indices[:n]
        bootstrap_sample = data[bootstrap_indices]

        # Compute statistic
        try:
            boot_stat = statistic_fn(bootstrap_sample)
            bootstrap_estimates.append(boot_stat)
        except Exception as e:
            logger.debug(f"Bootstrap iteration failed: {e}")
            continue

    bootstrap_estimates_array = np.array(bootstrap_estimates)

    # Compute statistics
    se = np.std(bootstrap_estimates_array, ddof=1)
    ci_lower = np.percentile(bootstrap_estimates_array, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates_array, 100 * (1 - alpha / 2))

    return {
        "estimate": float(estimate),
        "se": float(se),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "block_length": int(block_length),
        "n_bootstrap": len(bootstrap_estimates_array),
    }


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


def two_way_cluster_se(
    influences: np.ndarray,
    clusters_a: np.ndarray,
    clusters_b: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Two-way cluster-robust SE via Cameron-Gelbach-Miller inclusion-exclusion.

    For cases where two clustering dimensions exist (e.g., weight folds and outcome folds),
    computes: Var_AB = Var_A + Var_B - Var_{A∩B}.

    Args:
        influences: Influence functions (already centered)
        clusters_a: First clustering dimension (e.g., weight folds)
        clusters_b: Second clustering dimension (e.g., outcome folds)
        alpha: Significance level for CI

    Returns:
        Dictionary with two-way cluster-robust SE and t-based CI
    """

    def _crv1(phi: np.ndarray, c: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Helper to get variance from cluster_robust_se."""
        res = cluster_robust_se(
            data=phi,
            cluster_ids=c,
            statistic_fn=lambda x: np.mean(x),
            influence_fn=lambda x: x,  # IF already provided
            alpha=alpha,
        )
        return res["se"] ** 2, res

    # Create intersection clusters deterministically (avoid hash collisions)
    ca = np.asarray(clusters_a, dtype=np.int64)
    cb = np.asarray(clusters_b, dtype=np.int64)
    pairs = np.column_stack([ca, cb])
    _, ab = np.unique(pairs, axis=0, return_inverse=True)
    ab = ab.astype(np.int64)

    # Compute variance components
    v_a, res_a = _crv1(influences, clusters_a)
    v_b, res_b = _crv1(influences, clusters_b)
    v_ab, _ = _crv1(influences, ab)

    # Inclusion-exclusion principle
    var_hat = max(v_a + v_b - v_ab, 0.0)
    se = float(np.sqrt(var_hat))

    # Use the larger df for conservative inference
    df = max(res_a.get("df", 1), res_b.get("df", 1))
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    # Note: mean of centered IF is ~0; caller should form CI around the actual estimate
    est = 0.0
    return {
        "se": se,
        "ci_lower": float(est - t_crit * se),
        "ci_upper": float(est + t_crit * se),
        "df": int(df),
        "n_clusters_a": int(res_a.get("n_clusters", 0)),
        "n_clusters_b": int(res_b.get("n_clusters", 0)),
    }


def compose_se_components(
    se_if: float, se_oracle: float = 0.0, mc_var: float = 0.0
) -> float:
    """Combine independent SE sources in quadrature.

    Args:
        se_if: Standard error from influence functions (possibly cluster-robust)
        se_oracle: Standard error from oracle uncertainty (e.g., jackknife)
        mc_var: Monte Carlo variance from fresh draws (already variance, not SE)

    Returns:
        Combined standard error
    """
    return float(np.sqrt(max(se_if**2 + se_oracle**2 + mc_var, 0.0)))


# ========== Multiple Testing Correction ==========


def benjamini_hochberg_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Apply Benjamini-Hochberg FDR correction for multiple testing.

    Controls the False Discovery Rate when testing multiple hypotheses,
    as required by Section 9.4 of the CJE paper.

    Args:
        p_values: Array of p-values from individual tests
        alpha: FDR level (default 0.05)
        labels: Optional labels for each test

    Returns:
        Dictionary with:
        - 'adjusted_p_values': BH-adjusted p-values
        - 'significant': Boolean mask of significant results
        - 'n_significant': Number of significant results
        - 'threshold': Largest p-value threshold used
        - 'summary': List of (label, p_value, adjusted_p, significant)

    References:
        Benjamini, Y., & Hochberg, Y. (1995). Controlling the false
        discovery rate. Journal of the Royal Statistical Society B.
    """
    n = len(p_values)

    if n == 0:
        return {
            "adjusted_p_values": np.array([]),
            "significant": np.array([], dtype=bool),
            "n_significant": 0,
            "threshold": 0.0,
            "summary": [],
        }

    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # BH adjustment: p_adj = min(1, p * n / rank)
    ranks = np.arange(1, n + 1)
    adjusted_p = np.minimum(1.0, sorted_p * n / ranks)

    # Enforce monotonicity (adjusted p-values should be non-decreasing)
    for i in range(n - 2, -1, -1):
        adjusted_p[i] = min(adjusted_p[i], adjusted_p[i + 1])

    # Find threshold (largest p where p <= alpha * rank / n)
    bh_threshold = 0.0
    significant_sorted = np.zeros(n, dtype=bool)

    for i in range(n - 1, -1, -1):
        if sorted_p[i] <= alpha * (i + 1) / n:
            bh_threshold = sorted_p[i]
            significant_sorted[: i + 1] = True
            break

    # Map back to original order
    adjusted_p_orig = np.zeros(n)
    significant_orig = np.zeros(n, dtype=bool)

    for i, orig_idx in enumerate(sorted_indices):
        adjusted_p_orig[orig_idx] = adjusted_p[i]
        significant_orig[orig_idx] = significant_sorted[i]

    # Create summary
    summary = []
    if labels is None:
        labels = [f"H{i+1}" for i in range(n)]

    for i in range(n):
        summary.append(
            {
                "label": labels[i],
                "p_value": float(p_values[i]),
                "adjusted_p": float(adjusted_p_orig[i]),
                "significant": bool(significant_orig[i]),
            }
        )

    # Sort summary by p-value for readability
    def get_p_value(item: Dict[str, Any]) -> float:
        return float(item["p_value"])

    summary.sort(key=get_p_value)

    return {
        "adjusted_p_values": adjusted_p_orig,
        "significant": significant_orig,
        "n_significant": int(np.sum(significant_orig)),
        "threshold": float(bh_threshold),
        "fdr_level": float(alpha),
        "n_tests": n,
        "summary": summary,
    }


def compute_simultaneous_bands(
    estimates: np.ndarray,
    standard_errors: np.ndarray,
    correlation_matrix: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute simultaneous confidence bands using max-t method.

    For a selected subset of policies, provides simultaneous coverage
    accounting for correlation between estimates.

    Args:
        estimates: Point estimates for each policy
        standard_errors: Standard errors for each estimate
        correlation_matrix: Correlation between estimates (identity if None)
        alpha: Significance level
        labels: Optional policy labels

    Returns:
        Dictionary with simultaneous confidence bands
    """
    k = len(estimates)

    if correlation_matrix is None:
        # Assume independence
        correlation_matrix = np.eye(k)

    # Standardize to get t-statistics
    t_stats = estimates / standard_errors

    # Critical value for simultaneous coverage
    # Using Bonferroni as conservative approximation
    # (Could use multivariate t simulation for exact)
    bonf_alpha = alpha / k
    z_crit = stats.norm.ppf(1 - bonf_alpha / 2)

    # Simultaneous bands
    lower_bands = estimates - z_crit * standard_errors
    upper_bands = estimates + z_crit * standard_errors

    # Check which are significantly different from 0
    significant = (lower_bands > 0) | (upper_bands < 0)

    if labels is None:
        labels = [f"Policy{i+1}" for i in range(k)]

    bands = []
    for i in range(k):
        bands.append(
            {
                "label": labels[i],
                "estimate": float(estimates[i]),
                "se": float(standard_errors[i]),
                "lower": float(lower_bands[i]),
                "upper": float(upper_bands[i]),
                "significant": bool(significant[i]),
            }
        )

    return {
        "bands": bands,
        "critical_value": float(z_crit),
        "n_policies": k,
        "bonferroni_alpha": float(bonf_alpha),
        "n_significant": int(np.sum(significant)),
    }


# ========== Integrated Robust Inference ==========


def compute_robust_inference(
    estimates: np.ndarray,
    influence_functions: Optional[np.ndarray] = None,
    data: Optional[np.ndarray] = None,
    method: str = "stationary_bootstrap",
    cluster_ids: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    n_bootstrap: int = 4000,
    apply_fdr: bool = True,
    fdr_alpha: float = 0.05,
    policy_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Comprehensive robust inference with dependence and multiplicity handling.

    Args:
        estimates: Point estimates for policies
        influence_functions: If provided, use for inference
        data: Raw data (if influence_functions not provided)
        method: "stationary_bootstrap", "moving_block", or "cluster"
        cluster_ids: For cluster-robust SEs
        alpha: Significance level for CIs
        n_bootstrap: Bootstrap iterations
        apply_fdr: Whether to apply FDR correction
        fdr_alpha: FDR control level
        policy_labels: Names for policies

    Returns:
        Dictionary with complete robust inference results
    """
    n_policies = len(estimates)

    # Compute robust SEs for each policy
    robust_ses: List[float] = []
    robust_cis: List[Tuple[float, float]] = []
    p_values: List[float] = []

    for i in range(n_policies):
        if method == "stationary_bootstrap":
            if influence_functions is not None:
                # Use influence functions
                result = stationary_bootstrap_se(
                    (
                        influence_functions[:, i]
                        if influence_functions.ndim > 1
                        else influence_functions
                    ),
                    lambda x: np.mean(x),
                    n_bootstrap=n_bootstrap,
                    alpha=alpha,
                )
            elif data is not None:
                # Use raw data
                result = stationary_bootstrap_se(
                    data,
                    lambda x: estimates[i],  # Placeholder - would need actual estimator
                    n_bootstrap=n_bootstrap,
                    alpha=alpha,
                )
            else:
                raise ValueError("Need either influence_functions or data")

        elif method == "moving_block":
            # Moving block bootstrap for time series data
            if influence_functions is not None:
                result = moving_block_bootstrap_se(
                    (
                        influence_functions[:, i]
                        if influence_functions.ndim > 1
                        else influence_functions
                    ),
                    lambda x: np.mean(x),
                    n_bootstrap=n_bootstrap,
                    alpha=alpha,
                )
            elif data is not None:
                result = moving_block_bootstrap_se(
                    data,
                    lambda x: estimates[i],  # Use the estimate
                    n_bootstrap=n_bootstrap,
                    alpha=alpha,
                )
            else:
                raise ValueError("Need either influence_functions or data")

        elif method == "cluster" and cluster_ids is not None:
            if influence_functions is not None:
                # Use cluster-robust SE with proper t-based inference
                if_data = (
                    influence_functions[:, i]
                    if influence_functions.ndim > 1
                    else influence_functions
                )
                result = cluster_robust_se(
                    data=if_data,
                    cluster_ids=cluster_ids,
                    statistic_fn=lambda x: np.mean(x),
                    influence_fn=lambda x: x,  # IF already provided
                    alpha=alpha,
                )
                robust_ses.append(result["se"])
                robust_cis.append((result["ci_lower"], result["ci_upper"]))

                # t-based p-value (not normal!)
                df = max(result.get("df", 1), 1)
                t_stat = estimates[i] / result["se"] if result["se"] > 0 else 0.0
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                p_values.append(float(p_val))
                continue
            else:
                raise ValueError("Need influence_functions for cluster method")
        else:
            # Fallback to classical
            if influence_functions is not None:
                se = np.std(influence_functions[:, i]) / np.sqrt(
                    len(influence_functions)
                )
            elif data is not None:
                se = np.std(data) / np.sqrt(len(data))
            else:
                raise ValueError("Need either influence_functions or data")
            result = {
                "se": se,
                "ci_lower": estimates[i] - 1.96 * se,
                "ci_upper": estimates[i] + 1.96 * se,
            }

        robust_ses.append(result["se"])
        robust_cis.append((result["ci_lower"], result["ci_upper"]))

        # Compute p-value for test that estimate != 0
        z_stat = estimates[i] / result["se"] if result["se"] > 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        p_values.append(p_val)

    robust_ses_array = np.array(robust_ses)
    p_values_array = np.array(p_values)

    # Apply FDR correction if requested
    fdr_results = None
    if apply_fdr and n_policies > 1:
        fdr_results = benjamini_hochberg_correction(
            p_values_array,
            alpha=fdr_alpha,
            labels=policy_labels,
        )

    return {
        "estimates": estimates,
        "robust_ses": robust_ses_array,
        "robust_cis": robust_cis,
        "p_values": p_values,
        "method": method,
        "fdr_results": fdr_results,
        "n_policies": n_policies,
        "inference_alpha": alpha,
        "fdr_alpha": fdr_alpha if apply_fdr else None,
    }
