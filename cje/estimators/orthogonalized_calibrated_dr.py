# orthogonalized_calibrated_dr.py
# -*- coding: utf-8 -*-
"""
Orthogonalized Calibrated DR-CPO (OC-DR-CPO).

This module implements the SIMCal-anchored, orthogonalized calibrated DR estimator:
    V̂_ODR
      = P_n[ g(X) + W̃ * { R - q(X,A) } ]
      + P_n[ (W - m̂^OOF) * (R^OOF - f̂^OOF) ]                 # orthogonalizer
      + P_n[ (R^OOF - q^OOF) * (W - W̃) ]                      # retarget-to-W

Properties:
- First-order insensitivity to errors in BOTH nuisances:
  (i) reward calibrator f̂(S) and (ii) weight calibrator m̂(S)=E[W|S].
- √n inference with cross-fitting (OOF) and the standard OUA jackknife add-on.
- Preserves SIMCal’s tail stability by anchoring on W̃.

Implementation notes:
- Reuses DREstimator's infrastructure (fresh draws, outcome model).
- Cross-fits m̂^OOF(S) locally per-policy subset via isotonic W~S, using per-policy folds.
- Fetches OOF rewards for the residual corrections by DATASET INDEX if available
  (calibrator.predict_oof_by_index). Falls back to fold-based OOF or plain predict() with a warning.
- Uses the same per-prompt fold mapping as the outcome model for q^OOF on logged data.

Diagnostics:
- Reports orthogonalization and retarget residuals (mean ± CI); both should include 0.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional, Tuple, cast

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .dr_base import (
    DREstimator,
)  # Base DR implementation (fresh draws, outcome model, OUA)
from ..data.precomputed_sampler import PrecomputedSampler
from ..data.models import EstimationResult
from ..data.folds import get_fold
from ..diagnostics.robust_inference import cluster_robust_se

logger = logging.getLogger(__name__)


class OrthogonalizedCalibratedDRCPO(DREstimator):
    """Orthogonalized Calibrated DR-CPO estimator (OC-DR-CPO).

    Estimator formula (per-sample contributions):
        contrib_i = g_fresh_i
                  + W_tilde_i * (f_oof_i - q_logged_oof_i)        # anchor baseline
                  + (W_i - W_tilde_i) * (f_oof_i - q_logged_oof_i) # retarget-to-W
                  + (W_i - m_hat_oof_i) * (R_logged_i - f_oof_i)   # orthogonalizer

    Where:
        - f_oof: Out-of-fold calibrated rewards (from reward_calibrator)
        - R_logged: Actual observed rewards
        - q_logged_oof: Out-of-fold outcome model predictions
        - m_hat_oof: Out-of-fold E[W|S] predictions
        - W_tilde: SIMCal-calibrated weights
        - W: Raw/Hájek importance weights

    The baseline + retarget terms satisfy the identity:
        W_tilde * (f_oof - q_oof) + (W - W_tilde) * (f_oof - q_oof) = W * (f_oof - q_oof)

    The orthogonalizer is asymptotically mean-zero under cross-fitting.

    Influence function:
        φ_i = contrib_i - V̂

    Notes
    -----
    * Anchors on SIMCal weights (W̃) for tail stability while maintaining the DR estimand.
    * First-order insensitive to errors in both f̂ and m̂ under cross-fitting.
    * Requires fresh draws (same as DR-CPO). Add via DREstimator.add_fresh_draws().
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        n_folds: int = 5,
        use_calibrated_weights: bool = True,  # MUST be True to anchor on SIMCal (recommended)
        weight_mode: str = "hajek",
        reward_calibrator: Optional[Any] = None,
        random_seed: int = 42,
        run_diagnostics: bool = True,
        use_orthogonalization: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            sampler: PrecomputedSampler with calibrated data
            n_folds: Outcome-model cross-fitting folds (reused by DREstimator)
            use_calibrated_weights: True => SIMCal anchor (recommended)
            weight_mode: 'hajek' (mean-one) or 'raw' for W (recommended: 'hajek')
            reward_calibrator: Reward calibrator f̂; used for R and OOF predictions
            random_seed: Seed for deterministic fold assignment
            run_diagnostics: Whether to compute diagnostics
            use_orthogonalization: If False, falls back to simple DR-CPO
            **kwargs: forwarded to DREstimator (e.g., oracle_slice_config)
        """
        super().__init__(
            sampler=sampler,
            outcome_model=None,  # DREstimator chooses model automatically
            n_folds=n_folds,
            use_calibrated_weights=use_calibrated_weights,
            weight_mode=weight_mode,
            reward_calibrator=reward_calibrator,
            random_seed=random_seed,
            run_diagnostics=run_diagnostics,
            **kwargs,
        )
        self.use_orthogonalization = use_orthogonalization
        self._m_hat_oof_cache: Dict[str, np.ndarray] = {}
        self._orthogonalization_diagnostics: Dict[str, Dict[str, Any]] = {}
        self._orthogonality_scores: Dict[str, Dict[str, Any]] = (
            {}
        )  # Add orthogonality score tracking

    # ---------- Fit: add m̂^OOF(S) per policy (local folds) ----------

    def fit(self) -> None:
        """Fit weights (SIMCal), outcome model, and m̂^OOF for each policy."""
        # Parent does: fit IPS (SIMCal) and outcome model; sets _promptid_to_fold
        super().fit()

        if not self.use_orthogonalization:
            return

        logger.debug("Fitting m̂^OOF(S) = E[W|S] per policy for ODR orthogonalization")

        for policy in self.sampler.target_policies:
            # Logged subset for this policy
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                continue

            # Raw/base W for the residual (use hajek mean-one to align with W̃)
            W = self.sampler.compute_importance_weights(
                policy, clip_weight=None, mode=self.ips_estimator.weight_mode
            )

            # Judge scores S
            S = np.array([d.get("judge_score", np.nan) for d in data], dtype=float)
            if np.all(~np.isfinite(S)):
                logger.warning(
                    f"No finite judge scores for policy '{policy}', skipping m̂^OOF."
                )
                continue

            # Local fold assignments (simple, robust): computed from prompt_id, independent of calibrator folds
            n_folds = self.n_folds
            seed = self.random_seed
            prompt_ids = [d.get("prompt_id", f"sample_{i}") for i, d in enumerate(data)]
            fold_ids = np.array(
                [get_fold(pid, n_folds, seed) for pid in prompt_ids], dtype=int
            )

            # Cross-fitted isotonic m̂(S) = E[W|S]
            m_hat_oof = self._fit_m_hat_oof(W, S, fold_ids)
            self._m_hat_oof_cache[policy] = m_hat_oof

            logger.debug(
                f"m̂^OOF for {policy}: mean={m_hat_oof.mean():.4f}, "
                f"std={m_hat_oof.std():.4f}, n={len(m_hat_oof)}"
            )

    def _fit_m_hat_oof(
        self, weights: np.ndarray, judge_scores: np.ndarray, fold_ids: np.ndarray
    ) -> np.ndarray:
        """Cross-fit m̂(S)=E[W|S] by isotonic regression with per-policy local folds."""
        m_hat_oof = np.zeros_like(weights, dtype=float)
        uniq = np.unique(fold_ids[fold_ids >= 0])

        # If we cannot cross-fit, fall back to global isotonic or a constant
        if uniq.size < 2:
            valid = np.isfinite(judge_scores)
            if valid.sum() > 1:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(judge_scores[valid], weights[valid])
                m_hat_oof[valid] = iso.predict(judge_scores[valid])
                m_hat_oof[~valid] = float(weights[valid].mean())
            else:
                m_hat_oof[:] = float(np.mean(weights)) if weights.size else 1.0
        else:
            # Conservative fold requirements
            min_train = 100
            min_bins = 8

            for f in uniq:
                train = (fold_ids >= 0) & (fold_ids != f) & np.isfinite(judge_scores)
                test = (fold_ids == f) & np.isfinite(judge_scores)

                if test.sum() == 0:
                    continue

                if (train.sum() < min_train) or (
                    np.unique(judge_scores[train]).size < min_bins
                ):
                    # Not enough signal; use pooled mean on test fold
                    m_hat_oof[test] = (
                        float(weights[train].mean()) if train.sum() else 1.0
                    )
                    continue

                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(judge_scores[train], weights[train])
                m_hat_oof[test] = iso.predict(judge_scores[test])

            # Handle NaN S
            missing = ~np.isfinite(judge_scores)
            if missing.any():
                m_hat_oof[missing] = float(weights[~missing].mean())

        # Scale m̂ to match mean(W) on this subset (better orthogonality than forcing mean=1)
        valid = np.isfinite(judge_scores)
        muW = float(weights[valid].mean()) if valid.any() else float(weights.mean())
        muW = muW if muW > 0 else 1.0

        mean_m = float(m_hat_oof.mean()) if m_hat_oof.size else 1.0
        if mean_m > 1e-12:
            m_hat_oof *= muW / mean_m
        else:
            logger.warning("m̂^OOF has near-zero mean; using constant μ_W.")
            m_hat_oof[:] = muW

        return m_hat_oof

    # ---------- Estimate: ODR contributions and IF ----------

    def estimate(self) -> EstimationResult:
        """Compute ODR-CPO estimates with orthogonalization (or DR-CPO fallback)."""
        self._validate_fitted()

        # Try to auto-load fresh draws (DREstimator provides this)
        self._auto_load_fresh_draws()

        estimates: List[float] = []
        ses: List[float] = []
        n_used: Dict[str, int] = {}
        ifs: Dict[str, np.ndarray] = {}
        self._orthogonalization_diagnostics = {}

        # Build a fast prompt_id -> dataset index map for OOF-by-index rewards
        ds_index_by_pid: Dict[str, int] = {
            str(s.prompt_id): i for i, s in enumerate(self.sampler.dataset.samples)
        }

        for policy in self.sampler.target_policies:
            # Require fresh draws
            if policy not in self._fresh_draws:
                raise ValueError(
                    f"No fresh draws registered for '{policy}'. "
                    f"Call add_fresh_draws(policy, fresh_draws) before estimate()."
                )

            # Logged subset for this policy
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                logger.warning(f"No valid logged data for policy '{policy}'. Skipping.")
                estimates.append(np.nan)
                ses.append(np.nan)
                n_used[policy] = 0
                continue

            n = len(data)
            n_used[policy] = n

            # Store sample indices for IF alignment in stacking
            self._store_sample_indices(policy, data)

            # SIMCal weights (anchor) and mean-one raw/Hájek W for the retarget term
            W_tilde = self.ips_estimator.get_weights(policy)
            if W_tilde is None:
                logger.warning(f"No weights for policy '{policy}'. Skipping.")
                estimates.append(np.nan)
                ses.append(np.nan)
                continue

            # Defensive normalization to ensure mean-one
            mu = float(np.mean(W_tilde))
            if np.isfinite(mu) and mu > 0:
                W_tilde = W_tilde / mu
            else:
                logger.warning(
                    f"OC-DR: Invalid SIMCal weights mean ({mu}), falling back to raw weights"
                )
                W_tilde = self.sampler.compute_importance_weights(
                    policy, clip_weight=None, mode=self.ips_estimator.weight_mode
                )

            W = self.sampler.compute_importance_weights(
                policy, clip_weight=None, mode=self.ips_estimator.weight_mode
            )
            if W is None:
                logger.warning(f"No raw weights for policy '{policy}'. Skipping.")
                estimates.append(np.nan)
                ses.append(np.nan)
                n_used[policy] = 0
                continue

            # Logged arrays
            R_logged = np.array([d["reward"] for d in data], dtype=float)
            S_logged = np.array([d.get("judge_score") for d in data], dtype=float)
            prompts = [d["prompt"] for d in data]
            responses = [d["response"] for d in data]
            pids = [str(d.get("prompt_id")) for d in data]

            # Outcome model OOF predictions on logged data (q^OOF)
            if not hasattr(self, "_promptid_to_fold") or not self._promptid_to_fold:
                raise ValueError(
                    "Missing fold assignments for outcome model. "
                    "Ensure fit() completed with cross-fitting."
                )
            fold_ids = np.array(
                [self._promptid_to_fold[pid] for pid in pids], dtype=int
            )
            q_logged_oof = self.outcome_model.predict(
                prompts, responses, S_logged, fold_ids
            )

            # Store outcome predictions for oracle jackknife (OUA compatibility)
            self._outcome_predictions[policy] = q_logged_oof

            # Fresh-draw DM vector (per-prompt averages, same fold as the prompt)
            fresh = self._fresh_draws[policy]
            g_fresh_list: List[float] = []
            fresh_var_list: List[float] = []
            for i, pid in enumerate(pids):
                scores_i = fresh.get_scores_for_prompt_id(pid)
                if len(scores_i) == 0:
                    g_fresh_list.append(0.0)
                    fresh_var_list.append(0.0)
                    continue
                # Outcome model expects same fold for the prompt's draws
                fold_vec = np.full(len(scores_i), fold_ids[i], dtype=int)
                preds_i = self.outcome_model.predict(
                    [prompts[i]] * len(scores_i),
                    [""] * len(scores_i),
                    np.asarray(scores_i, dtype=float),
                    fold_vec,
                )
                g_fresh_list.append(float(np.mean(preds_i)))
                fresh_var_list.append(
                    float(np.var(preds_i, ddof=1)) if len(preds_i) > 1 else 0.0
                )

            g_fresh = np.asarray(g_fresh_list, dtype=float)
            fresh_var = np.asarray(fresh_var_list, dtype=float)

            # OOF rewards for orthogonalization (R_logged and f̂^OOF)
            # Prefer by-index OOF; else fold-based OOF; else plain predict() (warn).
            # CRITICAL: f_oof holds the OOF calibrator prediction, R_logged is the actual reward
            f_oof = R_logged.copy()  # Will hold the OOF calibrator prediction
            used_true_oof = False

            if self.reward_calibrator is not None:
                try:
                    # 1) Try dataset-index OOF
                    if hasattr(self.reward_calibrator, "predict_oof_by_index"):
                        ds_idx = np.array(
                            [ds_index_by_pid[pid] for pid in pids], dtype=int
                        )
                        R_pred = self.reward_calibrator.predict_oof_by_index(ds_idx)
                        if R_pred is not None:
                            f_oof = np.asarray(R_pred, dtype=float)
                            used_true_oof = True
                    # 2) Else try fold-based OOF with prompt-based folds
                    elif hasattr(self.reward_calibrator, "predict_oof"):
                        n_folds = self.n_folds
                        seed = self.random_seed
                        fold_cal = np.array(
                            [get_fold(pid, n_folds, seed) for pid in pids], dtype=int
                        )
                        R_pred = self.reward_calibrator.predict_oof(S_logged, fold_cal)
                        if R_pred is not None:
                            f_oof = np.asarray(R_pred, dtype=float)
                            used_true_oof = True
                    # 3) Fallback to in-fold predict (warn)
                    elif hasattr(self.reward_calibrator, "predict"):
                        R_pred = self.reward_calibrator.predict(S_logged)
                        if R_pred is not None:
                            f_oof = np.asarray(R_pred, dtype=float)
                            logger.warning(
                                "ODR: Using in-fold calibrator.predict() (no OOF available). "
                                "Orthogonalization guarantee may weaken."
                            )
                except Exception as e:
                    logger.debug(
                        f"ODR: OOF reward prediction failed for '{policy}': {e}"
                    )

            # m̂^OOF cache (if missing, use 1's; estimator remains valid but loses orthogonality to m̂)
            m_hat_oof = self._m_hat_oof_cache.get(policy, np.ones_like(W, dtype=float))

            # ---------- Build contributions ----------
            # Baseline DR (anchored on W̃) - use f_oof for consistency with IF
            baseline_ips = W_tilde * (f_oof - q_logged_oof)

            # Orthogonalizer and retarget terms
            if self.use_orthogonalization:
                orthog = (W - m_hat_oof) * (
                    R_logged - f_oof
                )  # Observed reward minus OOF prediction
                retarget = (f_oof - q_logged_oof) * (W - W_tilde)
            else:
                orthog = np.zeros_like(W_tilde)
                retarget = np.zeros_like(W_tilde)

            # Total per-sample contribution and point estimate (no oracle augmentation)
            contrib = g_fresh + baseline_ips + orthog + retarget
            V_hat = float(np.mean(contrib))
            estimates.append(V_hat)

            # ---------- Influence function (perfectly aligned with estimator) ----------
            phi = contrib - V_hat

            # IIC removed - use influence functions directly

            # CRITICAL FIX: Use cluster-robust SE for fold dependence
            res_if = cluster_robust_se(
                data=phi,
                cluster_ids=fold_ids,
                statistic_fn=lambda x: np.mean(x),
                influence_fn=lambda x: x,  # already an IF
                alpha=0.05,
            )
            base_se = float(res_if["se"])

            # Compute MC variance component (mirrors DREstimator logic)
            draws_per_prompt = []
            for pid in pids:
                scores_i = fresh.get_scores_for_prompt_id(pid)
                draws_per_prompt.append(len(scores_i))
            M = np.asarray(draws_per_prompt, dtype=float)
            mc_var = float(np.sum(fresh_var / np.maximum(M, 1.0)) / (n**2))

            # Combined SE
            se = float(np.sqrt(base_se**2 + mc_var))
            ses.append(se)
            ifs[policy] = phi

            # ---------- Diagnostics ----------
            # Residual means and CIs (should include 0)
            def _mean_ci(v: np.ndarray) -> Tuple[float, float, float]:
                m = float(v.mean())
                s = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
                delta = 1.96 * s
                return m, m - delta, m + delta

            ortho_mean, ortho_lo, ortho_hi = _mean_ci(orthog)
            retgt_mean, retgt_lo, retgt_hi = _mean_ci(retarget)

            # Compute retarget identity error (should be ~0)
            base_plus_retarget = baseline_ips + retarget
            raw_ips = W * (f_oof - q_logged_oof)
            identity_error = float(np.mean(np.abs(base_plus_retarget - raw_ips)))

            # Debug assertions to catch regressions
            if __debug__:
                assert np.allclose(
                    base_plus_retarget, raw_ips, atol=1e-12, rtol=1e-12
                ), f"OC-DR-CPO identity failed for {policy}: error={identity_error}"

            # Check IF mean (should be ~0)
            if_mean = float(np.mean(phi))
            if __debug__:
                assert (
                    abs(if_mean) < 1e-9
                ), f"OC-DR-CPO IF not centered for {policy}: mean={if_mean}"

            self._orthogonalization_diagnostics[policy] = {
                "orthog_residual": ortho_mean,
                "orthog_ci_lower": ortho_lo,
                "orthog_ci_upper": ortho_hi,
                "retarget_residual": retgt_mean,
                "retarget_ci_lower": retgt_lo,
                "retarget_ci_upper": retgt_hi,
                "retarget_identity_error": identity_error,
                "if_mean": if_mean,
                "baseline_dm_mean": float(np.mean(g_fresh)),
                "baseline_ips_mean": float(np.mean(baseline_ips)),
                "uses_true_oof_rewards": bool(used_true_oof),
                "mc_variance": mc_var,
                "avg_draws_per_prompt": float(np.mean(M)) if len(M) > 0 else 0.0,
            }

            # Compute orthogonality score (same as parent DR-CPO)
            from cje.diagnostics.dr import compute_orthogonality_score

            ortho_result = compute_orthogonality_score(
                weights=W_tilde,  # Use calibrated weights for consistency with DR-CPO
                rewards=R_logged,
                outcome_predictions=q_logged_oof,
                return_ci=True,
            )
            self._orthogonality_scores[policy] = ortho_result

            logger.info(
                f"OC-DR-CPO[{policy}]: {V_hat:.4f} ± {se:.4f} | "
                f"orthog={ortho_mean:+.4e} [{ortho_lo:+.4e},{ortho_hi:+.4e}], "
                f"retarget={retgt_mean:+.4e} [{retgt_lo:+.4e},{retgt_hi:+.4e}]"
            )

        # Package result
        result = EstimationResult(
            estimates=np.asarray(estimates, dtype=float),
            standard_errors=np.asarray(ses, dtype=float),
            n_samples_used=n_used,
            method="oc_dr_cpo",
            influence_functions=ifs,
            diagnostics=None,  # The caller (suite) or parent infra can attach suites; we add metadata below
            metadata={
                "target_policies": list(self.sampler.target_policies),
                "orthogonalization_diagnostics": self._orthogonalization_diagnostics,
                "orthogonality_scores": self._orthogonality_scores,  # Include orthogonality scores
                # Add sample indices for IF alignment in stacking
                "if_sample_indices": getattr(self, "_if_sample_indices", {}),
                # oracle_augmentation removed - using OUA jackknife only
            },
        )

        # Apply OUA jackknife using base class method
        self._apply_oua_jackknife(result)

        # Store IFs on self for downstream tools
        self._influence_functions = ifs
        self._results = result
        return result

    def get_oracle_jackknife(self, policy: str) -> Optional[np.ndarray]:
        """Compute leave-one-oracle-fold-out estimates for OC-DR-CPO.

        For each fold's calibrator, recompute the full OC-DR-CPO estimate including
        all three terms (baseline, retarget, orthogonalizer).

        Returns:
            Array of K jackknife estimates, or None if not applicable
        """
        try:
            if self.reward_calibrator is None:
                return None

            if not hasattr(self.reward_calibrator, "get_fold_models_for_oua"):
                return None

            fold_models = self.reward_calibrator.get_fold_models_for_oua()
            if not fold_models:
                return None

            # Get necessary data
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                return None

            # Get weights (reuse cached values)
            W_tilde = self.ips_estimator.get_weights(policy)
            W = self.sampler.compute_importance_weights(
                policy, clip_weight=None, mode=self.ips_estimator.weight_mode
            )
            if W_tilde is None or W is None:
                return None

            # Normalize W_tilde
            mu = float(np.mean(W_tilde))
            W_tilde = W_tilde / mu if np.isfinite(mu) and mu > 0 else W

            # Get m_hat_oof (or default)
            m_hat = self._m_hat_oof_cache.get(policy)
            if m_hat is None:
                muW = float(np.mean(W)) if np.isfinite(W).all() else 1.0
                m_hat = np.full_like(W, muW)

            # Extract arrays
            S = np.array([d.get("judge_score", np.nan) for d in data], dtype=float)
            R = np.array([d["reward"] for d in data], dtype=float)
            pids = [str(d.get("prompt_id")) for d in data]
            fold_ids = np.array(
                [self._promptid_to_fold[pid] for pid in pids], dtype=int
            )

            # Get outcome predictions
            prompts = [d["prompt"] for d in data]
            responses = [d["response"] for d in data]
            q_oof = self.outcome_model.predict(prompts, responses, S, fold_ids)

            # Get fresh draw data
            fresh = self._fresh_draws.get(policy)
            if fresh is None:
                return None

            # Compute jackknife estimates
            jack = []
            for fold_id, f_model in fold_models.items():
                # Recalibrate rewards with this fold's model
                f_loo = np.clip(f_model.predict(S), 0.0, 1.0)

                # Compute fresh draw mean for this fold
                g_fresh_list = []
                for pid in set(pids):
                    scores_i = fresh.get_scores_for_prompt_id(pid)
                    if len(scores_i) > 0:
                        preds_i = np.clip(f_model.predict(np.array(scores_i)), 0.0, 1.0)
                        g_fresh_list.append(float(np.mean(preds_i)))

                if not g_fresh_list:
                    continue

                g_fresh_mean = float(np.mean(g_fresh_list))

                # Compute OC-DR-CPO estimate with all three terms
                baseline = W_tilde * (f_loo - q_oof)
                retarget = (W - W_tilde) * (f_loo - q_oof)
                orthog = (W - m_hat) * (R - f_loo) if self.use_orthogonalization else 0

                contrib = g_fresh_mean + np.mean(baseline + retarget + orthog)
                jack.append(float(contrib))

            return np.asarray(jack, dtype=float) if jack else None

        except Exception as e:
            logger.debug(f"OC-DR-CPO jackknife failed for {policy}: {e}")
            return None
