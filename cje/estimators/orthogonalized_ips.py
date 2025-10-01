"""Orthogonalized Calibrated IPS estimator.

This provides first-order robustness to errors in both:
1. The reward calibrator f̂(S)
2. The weight calibrator m̂(S) = E[W|S]

Uses a SIMCal-anchored approach that preserves variance control
while adding orthogonalization corrections.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, cast
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .calibrated_ips import CalibratedIPS
from ..data.models import EstimationResult
from ..data.folds import get_fold
from ..diagnostics.robust_inference import cluster_robust_se

logger = logging.getLogger(__name__)


class OrthogonalizedCalibratedIPS(CalibratedIPS):
    """Calibrated IPS with first-order orthogonalization.

    This estimator extends CalibratedIPS with orthogonalization against
    both reward and weight nuisance functions, achieving:
    - First-order insensitivity to f̂(S) errors (reward calibration)
    - First-order insensitivity to m̂(S) errors (weight calibration)
    - Clean √n asymptotic behavior
    - Preserved variance gains from SIMCal

    The SIMCal-anchored formulation:
    V̂ = P_n[W̃·R] + P_n[(W-m̂^OOF)(R-f̂^OOF)] + P_n[f̂^OOF(W-W̃)]

    where:
    - W̃: SIMCal calibrated weights (variance-stabilized)
    - W: Raw importance weights
    - R: Calibrated rewards (global fit)
    - R^OOF, f̂^OOF: Out-of-fold calibrated rewards
    - m̂^OOF: Out-of-fold E[W|S]
    """

    def __init__(
        self,
        *args: Any,
        use_orthogonalization: bool = True,
        n_folds: int = 5,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialize OC-IPS estimator.

        Args:
            use_orthogonalization: Whether to apply orthogonalization.
                If False, behaves exactly like CalibratedIPS.
            n_folds: Number of folds for cross-fitting m̂(S).
            random_seed: Random seed for fold assignment.
            *args, **kwargs: Passed to CalibratedIPS
        """
        super().__init__(*args, **kwargs)
        self.use_orthogonalization = use_orthogonalization
        self.n_folds = n_folds
        self.random_seed = random_seed
        self._m_hat_oof_cache: Dict[str, np.ndarray] = {}
        self._orthogonalization_diagnostics: Dict[str, Dict] = {}

    def fit(self) -> None:
        """Fit the estimator with additional m̂^OOF computation."""
        # Run parent fit (handles weight calibration, OUA, etc.)
        super().fit()

        if not self.use_orthogonalization or not self.calibrate_weights:
            return

        # For each policy, fit m̂^OOF(S) for orthogonalization
        logger.debug("Fitting m̂^OOF for orthogonalization")

        for policy in self.sampler.target_policies:
            if policy in self._no_overlap_policies:
                continue

            # Get data for this policy
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                continue

            # Get raw weights W (before SIMCal)
            raw_weights = self.sampler.compute_importance_weights(
                policy, clip_weight=self.clip_weight, mode=self.weight_mode
            )
            if raw_weights is None:
                continue

            # Get judge scores
            judge_scores = np.array([d.get("judge_score", np.nan) for d in data])
            if np.all(np.isnan(judge_scores)):
                logger.warning(f"No judge scores for policy {policy}, skipping m̂^OOF")
                continue

            # Get fold assignments (local to this policy subset)
            n_folds = self.n_folds
            seed = self.random_seed
            prompt_ids = [d.get("prompt_id", f"sample_{i}") for i, d in enumerate(data)]
            fold_ids = np.array([get_fold(pid, n_folds, seed) for pid in prompt_ids])

            # Cross-fit m̂(S) = E[W|S]
            m_hat_oof = self._fit_m_hat_oof(raw_weights, judge_scores, fold_ids)
            self._m_hat_oof_cache[policy] = m_hat_oof

            logger.debug(
                f"Fitted m̂^OOF for {policy}: mean={m_hat_oof.mean():.3f}, "
                f"std={m_hat_oof.std():.3f}, range=[{m_hat_oof.min():.3f}, {m_hat_oof.max():.3f}]"
            )

    def _fit_m_hat_oof(
        self, weights: np.ndarray, judge_scores: np.ndarray, fold_ids: np.ndarray
    ) -> np.ndarray:
        """Cross-fit m̂(S) = E[W|S] for orthogonalization.

        Args:
            weights: Raw importance weights
            judge_scores: Judge scores S
            fold_ids: Fold assignments for cross-fitting

        Returns:
            m_hat_oof: Out-of-fold predictions of E[W|S]
        """
        m_hat_oof = np.zeros_like(weights)
        unique_folds = np.unique(fold_ids[fold_ids >= 0])

        if len(unique_folds) < 2:
            # Not enough folds for cross-fitting, use global fit
            logger.debug("Insufficient folds for cross-fitting m̂, using global fit")
            valid_mask = np.isfinite(judge_scores)
            if valid_mask.sum() > 1:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(judge_scores[valid_mask], weights[valid_mask])
                m_hat_oof[valid_mask] = iso.predict(judge_scores[valid_mask])
                m_hat_oof[~valid_mask] = weights[valid_mask].mean()
            else:
                m_hat_oof = np.ones_like(weights)
        else:
            # Cross-fit across folds
            for fold in unique_folds:
                train_mask = (
                    (fold_ids >= 0) & (fold_ids != fold) & np.isfinite(judge_scores)
                )
                test_mask = (fold_ids == fold) & np.isfinite(judge_scores)

                # Robust fold handling with minimum requirements
                min_train = 100  # Minimum training samples
                min_bins = 8  # Minimum unique score values

                unique_train_scores = (
                    np.unique(judge_scores[train_mask]).size
                    if train_mask.sum() > 0
                    else 0
                )

                if (
                    train_mask.sum() < min_train
                    or test_mask.sum() == 0
                    or unique_train_scores < min_bins
                ):
                    # Not enough data in this fold
                    if test_mask.sum() > 0:
                        m_hat_oof[test_mask] = (
                            weights[train_mask].mean() if train_mask.sum() > 0 else 1.0
                        )
                    continue

                # Fit isotonic regression on training folds
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(judge_scores[train_mask], weights[train_mask])

                # Predict on test fold
                m_hat_oof[test_mask] = iso.predict(judge_scores[test_mask])

            # Handle any samples with missing judge scores
            missing_mask = np.isnan(judge_scores)
            if missing_mask.sum() > 0:
                m_hat_oof[missing_mask] = weights[~missing_mask].mean()

        # Scale to match mean(W) not 1.0 for better orthogonality
        valid_mask = np.isfinite(judge_scores)
        muW = float(np.mean(weights[valid_mask])) if valid_mask.sum() > 0 else 1.0

        if muW <= 0:
            logger.warning(f"Non-positive mean(W)={muW:.3f}, falling back to 1.0")
            muW = 1.0

        if m_hat_oof.mean() > 1e-12:
            m_hat_oof = m_hat_oof * (muW / m_hat_oof.mean())
        else:
            logger.warning("m̂^OOF has near-zero mean, using ones")
            m_hat_oof = np.ones_like(m_hat_oof) * muW

        return m_hat_oof

    def estimate(self) -> EstimationResult:
        """Compute OC-IPS estimates with orthogonalization."""
        if not self.use_orthogonalization or not self.calibrate_weights:
            # Fall back to standard CalibratedIPS
            return super().estimate()

        self._validate_fitted()

        estimates = []
        standard_errors = []
        n_samples_used = {}
        influence_functions = {}

        for policy in self.sampler.target_policies:
            if policy in self._no_overlap_policies:
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                influence_functions[policy] = np.array([np.nan])
                logger.warning(f"Policy '{policy}' has no overlap - returning NaN")
                continue

            # Get data for this policy
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                logger.warning(f"No data for policy '{policy}'. Skipping.")
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                continue

            n = len(data)
            n_samples_used[policy] = n

            # Get weights
            W = self.sampler.compute_importance_weights(
                policy, clip_weight=self.clip_weight, mode=self.weight_mode
            )  # Raw weights
            W_tilde = self._weights_cache[policy]  # SIMCal calibrated weights

            # Defensive normalization to ensure mean-one
            mu = float(np.mean(W_tilde))
            if np.isfinite(mu) and mu > 0:
                W_tilde = W_tilde / mu
            else:
                logger.warning(
                    f"OC-IPS: Invalid SIMCal weights mean ({mu}), falling back to raw weights"
                )
                W_tilde = W

            # Get rewards
            R = np.array([d["reward"] for d in data])  # Global fit rewards

            # Initialize OOF prediction (default to global rewards if not available)
            f_oof = R.copy()  # Will hold the OOF calibrator prediction

            # Try to get OOF rewards using dataset indices (best), else fold-based OOF
            if self.reward_calibrator is not None:
                try:
                    # 1) Prefer dataset-index OOF if available
                    if hasattr(self.reward_calibrator, "predict_oof_by_index"):
                        # Build mapping from prompt_id to dataset index
                        ds_index_by_pid = {
                            str(s.prompt_id): i
                            for i, s in enumerate(self.sampler.dataset.samples)
                        }
                        ds_idx = np.array(
                            [
                                ds_index_by_pid.get(str(d.get("prompt_id")), -1)
                                for d in data
                            ],
                            dtype=int,
                        )

                        # Only proceed if all indices are valid
                        if np.all(ds_idx >= 0):
                            R_pred = self.reward_calibrator.predict_oof_by_index(ds_idx)
                            if R_pred is not None:
                                f_oof = np.asarray(R_pred, dtype=float)
                                logger.debug(
                                    f"Using index-based OOF rewards for {policy}"
                                )

                    # 2) Else try fold-based OOF using per-prompt folds
                    elif hasattr(self.reward_calibrator, "predict_oof"):
                        from ..data.folds import get_fold

                        n_folds = self.n_folds
                        seed = self.random_seed
                        judge_scores = np.array(
                            [d.get("judge_score", np.nan) for d in data], dtype=float
                        )
                        prompt_ids = [
                            d.get("prompt_id", f"sample_{i}")
                            for i, d in enumerate(data)
                        ]
                        fold_cal = np.array(
                            [get_fold(pid, n_folds, seed) for pid in prompt_ids],
                            dtype=int,
                        )

                        # Check if we have valid judge scores
                        valid_scores = np.isfinite(judge_scores)
                        if valid_scores.sum() > 0:
                            R_pred = self.reward_calibrator.predict_oof(
                                judge_scores, fold_cal
                            )
                            if R_pred is not None:
                                f_oof = np.asarray(R_pred, dtype=float)
                                logger.debug(
                                    f"Using fold-based OOF rewards for {policy}"
                                )

                    # 3) Fallback: in-fold predict with a warning
                    elif hasattr(self.reward_calibrator, "predict"):
                        judge_scores = np.array(
                            [d.get("judge_score", np.nan) for d in data], dtype=float
                        )
                        valid_scores = np.isfinite(judge_scores)
                        if valid_scores.sum() > 0:
                            R_pred = self.reward_calibrator.predict(judge_scores)
                            if R_pred is not None:
                                f_oof = np.asarray(R_pred, dtype=float)
                                logger.warning(
                                    f"OC-IPS: Using in-fold calibrator.predict() for {policy}; orthogonality guarantees may weaken."
                                )
                except Exception as e:
                    logger.debug(f"OC-IPS: OOF reward path failed for '{policy}': {e}")

            # Get m̂^OOF (default to mean(W) if not available)
            m_hat_oof = self._m_hat_oof_cache.get(policy, None)
            if m_hat_oof is None:
                muW = float(np.mean(W)) if np.isfinite(W).all() else 1.0
                m_hat_oof = np.full_like(W, fill_value=muW)

            # Compute OC-IPS (SIMCal-anchored, full three-term version)
            # V̂ = P_n[W̃·R] + P_n[(W-m̂)(R-f̂)] + P_n[f̂(W-W̃)]

            # 1. Baseline term (use actual rewards R, not OOF)
            baseline = W_tilde * R

            # 2. Orthogonalization term (doubly-robust correction)
            # This is the KEY term that makes us robust to both f̂ and m̂ errors
            # Uses observed reward R minus OOF prediction f_oof
            orthogonalization = (W - m_hat_oof) * (R - f_oof)

            # 3. Re-targeting term (aligns with SIMCal weights)
            retarget = f_oof * (W - W_tilde)

            # Total contribution before augmentation
            contrib = baseline + orthogonalization + retarget

            # Point estimate (no oracle augmentation)
            V_hat = float(contrib.mean())
            estimates.append(V_hat)

            # Influence function (perfectly aligned with estimator now)
            # φ = contrib - V̂  (we build IF directly from the per-sample contributions)
            phi = contrib - V_hat

            # IIC removed - use influence functions directly

            # Cluster-robust SE across folds (accounts for within-fold dependence)
            # Get fold assignments for clustering
            prompt_ids = [d.get("prompt_id", f"sample_{i}") for i, d in enumerate(data)]
            fold_ids = np.array(
                [get_fold(pid, self.n_folds, self.random_seed) for pid in prompt_ids]
            )

            # Compute cluster-robust SE
            res_if = cluster_robust_se(
                data=phi,
                cluster_ids=fold_ids,
                statistic_fn=lambda x: np.mean(x),
                influence_fn=lambda x: x,  # already centered IF
                alpha=0.05,
            )
            se = float(res_if["se"])
            standard_errors.append(se)
            influence_functions[policy] = phi

            # Store orthogonalization diagnostics with CIs
            orthog_se = float(np.std(orthogonalization, ddof=1) / np.sqrt(n))
            orthog_ci = 1.96 * orthog_se
            retarget_se = float(np.std(retarget, ddof=1) / np.sqrt(n))
            retarget_ci = 1.96 * retarget_se

            self._orthogonalization_diagnostics[policy] = {
                "orthog_residual": float(orthogonalization.mean()),
                "orthog_se": orthog_se,
                "orthog_ci_lower": float(orthogonalization.mean() - orthog_ci),
                "orthog_ci_upper": float(orthogonalization.mean() + orthog_ci),
                "retarget_residual": float(retarget.mean()),
                "retarget_se": retarget_se,
                "retarget_ci_lower": float(retarget.mean() - retarget_ci),
                "retarget_ci_upper": float(retarget.mean() + retarget_ci),
                "baseline_contrib": float(baseline.mean()),
                "uses_oof_calibrator": not np.array_equal(R, f_oof),
            }

            logger.debug(
                f"OC-IPS for {policy}: V̂={V_hat:.4f}, SE={se:.4f}, "
                f"orthog={orthogonalization.mean():.6f}±{orthog_se:.6f}, "
                f"retarget={retarget.mean():.6f}±{retarget_se:.6f}"
            )

        # Store influence functions for later use
        self._influence_functions = influence_functions

        # Create result
        result = EstimationResult(
            estimates=np.array(estimates),
            standard_errors=np.array(standard_errors),
            n_samples_used=n_samples_used,
            method="oc-ips",
            influence_functions=influence_functions,
            diagnostics=None,  # Will be set below
            metadata={
                "target_policies": list(self.sampler.target_policies),
                "calibrate": self.calibrate_weights,
                "use_orthogonalization": self.use_orthogonalization,
                "orthogonalization_diagnostics": self._orthogonalization_diagnostics,
                # augmentation_diagnostics removed - using OUA jackknife only
            },
        )

        # Apply OUA jackknife using base class method (uses our custom get_oracle_jackknife)
        self._apply_oua_jackknife(result)

        # Create diagnostics similar to parent class
        if self.run_diagnostics:
            from ..diagnostics import IPSDiagnostics, Status

            # Gather weight statistics
            ess_per_policy = {}
            max_weight_per_policy = {}
            overall_ess = 0.0

            for policy in self.sampler.target_policies:
                if policy in self._weights_cache:
                    weights = self._weights_cache[policy]
                    # Compute raw ESS (not divided by n)
                    ess = float((weights.sum() ** 2) / (weights**2).sum())
                    ess_normalized = ess / len(weights)  # ESS per sample
                    ess_per_policy[policy] = ess_normalized
                    max_weight_per_policy[policy] = float(weights.max())
                    overall_ess += ess_normalized

            if len(ess_per_policy) > 0:
                overall_ess = overall_ess / len(ess_per_policy)

            # Determine status based on ESS
            if overall_ess > 0.5:
                status = Status.GOOD
            elif overall_ess > 0.2:
                status = Status.WARNING
            else:
                status = Status.CRITICAL

            # Create diagnostics
            policies = list(self.sampler.target_policies)
            estimates_dict = {p: result.estimates[i] for i, p in enumerate(policies)}
            se_dict = {p: result.standard_errors[i] for i, p in enumerate(policies)}

            diagnostics = IPSDiagnostics(
                estimator_type="OrthogonalizedCalibratedIPS",
                method="oc-ips",
                n_samples_total=len(self.sampler.dataset.samples),
                n_samples_valid=self.sampler.n_valid_samples,
                n_policies=len(policies),
                policies=policies,
                estimates=estimates_dict,
                standard_errors=se_dict,
                n_samples_used=result.n_samples_used,
                weight_ess=overall_ess,
                weight_status=status,
                ess_per_policy=ess_per_policy,
                max_weight_per_policy=max_weight_per_policy,
            )

            result.diagnostics = diagnostics

        self._results = result
        return result

    def get_oracle_jackknife(self, policy: str) -> Optional[np.ndarray]:
        """Leave-one-oracle-fold jackknife estimates for OC-IPS.

        For each reward_calibrator fold model f^(−k), recompute the OC-IPS estimate using
        rewards R^(−k) = f^(−k)(S) with the same weights and orthogonalization terms.

        Returns an array of K estimates, or None if not applicable.
        """
        try:
            if self.reward_calibrator is None:
                return None

            # Use the unified method to get fold models
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

            # Get required data for this policy
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                return None

            # Get weights (raw and SIMCal calibrated)
            W = self.sampler.compute_importance_weights(
                policy, clip_weight=self.clip_weight, mode=self.weight_mode
            )
            W_tilde = self._weights_cache.get(policy)
            if W is None or W_tilde is None:
                return None

            # Defensive normalization for W_tilde
            mu = float(np.mean(W_tilde))
            if np.isfinite(mu) and mu > 0:
                W_tilde = W_tilde / mu
            else:
                logger.warning(
                    f"OC-IPS jackknife: Invalid SIMCal weights mean ({mu}), falling back to raw weights"
                )
                W_tilde = W

            # Get m̂^OOF (default to mean(W) if not available)
            m_hat_oof = self._m_hat_oof_cache.get(policy, None)
            if m_hat_oof is None:
                muW = float(np.mean(W)) if np.isfinite(W).all() else 1.0
                m_hat_oof = np.full_like(W, fill_value=muW)

            judge_scores = np.array([d.get("judge_score") for d in data], dtype=float)

            # Sanity check alignment
            if len(judge_scores) != len(W) or len(judge_scores) != len(W_tilde):
                return None

            jack: List[float] = []
            for fold_id, fold_model in fold_models.items():
                # Recompute rewards under leave-one-fold reward_calibrator
                # For FlexibleCalibrator, fold_model is IsotonicRegression
                rewards_loo = np.clip(fold_model.predict(judge_scores), 0.0, 1.0)

                # OUA jackknife: recompute OC-IPS with different calibrator
                # Mirror the estimator exactly: keep baseline on R, only swap f_oof → rewards_loo
                # Get actual rewards R (not OOF)
                R = np.array([d["reward"] for d in data])

                # V̂^(−k) = P_n[W̃·R] + P_n[(W-m̂)(R-R^(−k))] + P_n[R^(−k)(W-W̃)]
                baseline_loo = W_tilde * R  # Keep baseline on actual rewards
                orthog_loo = (W - m_hat_oof) * (R - rewards_loo)  # Now non-zero!
                retarget_loo = rewards_loo * (W - W_tilde)

                contrib_loo = baseline_loo + orthog_loo + retarget_loo
                jack.append(float(np.mean(contrib_loo)))

            return np.asarray(jack, dtype=float) if jack else None
        except Exception as e:
            logger.debug(f"get_oracle_jackknife failed for {policy}: {e}")
            return None
