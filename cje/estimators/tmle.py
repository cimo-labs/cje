# cje/core/tmle.py
"""
TMLE estimator for policy evaluation with cross-fitted monotone outcome models.

This estimator properly inherits from DREstimator, eliminating code duplication
and ensuring consistent diagnostics with other DR methods.
"""

from __future__ import annotations
from typing import Dict, Optional, Any, List, Tuple, Union, cast
import logging
import numpy as np

from .dr_base import DREstimator
from .outcome_models import IsotonicOutcomeModel
from ..data.precomputed_sampler import PrecomputedSampler
from ..data.models import EstimationResult
from ..data.fresh_draws import FreshDrawDataset

logger = logging.getLogger(__name__)

_EPS = 1e-7  # numerical guard for logits/probabilities


def _expit(x: np.ndarray) -> np.ndarray:
    result = 1.0 / (1.0 + np.exp(-x))
    return np.asarray(result)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    result = np.log(p) - np.log(1.0 - p)
    return np.asarray(result)


class TMLEEstimator(DREstimator):
    """TMLE with cross-fitted isotonic outcome models.

    Now properly inherits from DREstimator for consistency with other DR methods.

    Args:
        sampler: PrecomputedSampler with calibrated rewards
        n_folds: Number of cross-fitting folds (default 5)
        link: 'logit' (default, for rewards in [0,1]) or 'identity'
        max_iter: Max Newton steps for logistic targeting
        tol: Convergence tolerance on the (weighted) score
        use_calibrated_weights: Use CalibratedIPS (default True)
        **kwargs: Passed through to DREstimator
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        n_folds: int = 5,
        link: str = "logit",
        max_iter: int = 50,
        tol: float = 1e-8,
        use_calibrated_weights: bool = True,
        weight_mode: str = "hajek",
        reward_calibrator: Optional[Any] = None,
        **kwargs: Any,
    ):
        # Initialize DR base with standard isotonic outcome model
        # Pass reward_calibrator for proper index transformation with two-stage calibration
        outcome_model = IsotonicOutcomeModel(
            n_folds=n_folds, calibrator=reward_calibrator
        )

        super().__init__(
            sampler=sampler,
            outcome_model=outcome_model,
            n_folds=n_folds,
            use_calibrated_weights=use_calibrated_weights,
            weight_mode=weight_mode,
            reward_calibrator=reward_calibrator,
            **kwargs,
        )

        if link not in {"logit", "identity"}:
            raise ValueError(f"link must be one of ['logit','identity'], got {link}")

        self.link = link
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        # Per-policy epsilon/diagnostics
        self._tmle_info: Dict[str, Dict[str, Any]] = {}

        # Initialize storage for diagnostics (inherited from DREstimator but ensure they exist)
        if not hasattr(self, "_dm_component"):
            self._dm_component: Dict[str, np.ndarray] = {}
        if not hasattr(self, "_ips_correction"):
            self._ips_correction: Dict[str, np.ndarray] = {}
        if not hasattr(self, "_fresh_rewards"):
            self._fresh_rewards: Dict[str, np.ndarray] = {}
        if not hasattr(self, "_outcome_predictions"):
            self._outcome_predictions: Dict[str, np.ndarray] = {}

    def estimate(self) -> EstimationResult:
        """Compute TMLE estimates for all target policies.

        Extends the base DR estimate by adding a targeting step.
        """
        self._validate_fitted()

        # Auto-load fresh draws if not already loaded
        self._auto_load_fresh_draws()

        estimates: List[float] = []
        standard_errors: List[float] = []
        n_samples_used: Dict[str, int] = {}
        self._tmle_info = {}

        for policy in self.sampler.target_policies:
            # Ensure fresh draws available
            if policy not in self._fresh_draws:
                raise ValueError(
                    f"No fresh draws registered for '{policy}'. "
                    f"Call add_fresh_draws(policy, fresh_draws) before estimate()."
                )

            # Get data and weights using base class methods
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                logger.warning(f"No valid data for policy '{policy}'. Skipping.")
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            weights = self.get_weights(policy)
            if weights is None or len(weights) != len(data):
                raise ValueError(
                    f"Weight/data mismatch for policy '{policy}': "
                    f"weights={None if weights is None else len(weights)}, data={len(data)}"
                )

            # Extract arrays
            rewards = np.array([d["reward"] for d in data], dtype=float)
            scores = np.array([d.get("judge_score") for d in data], dtype=float)
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

            # Get fold assignments - strict mode (no fallback)
            unknown_pids = [
                pid for pid in prompt_ids if pid not in self._promptid_to_fold
            ]
            if unknown_pids:
                raise ValueError(
                    f"Missing fold assignments for {len(unknown_pids)} samples in policy '{policy}'. "
                    f"Example prompt_ids: {unknown_pids[:3]}. "
                    f"Ensure calibration was done with enable_cross_fit=True."
                )
            fold_ids = np.array(
                [self._promptid_to_fold[pid] for pid in prompt_ids], dtype=int
            )

            # 1) Get initial cross-fitted predictions on logged data
            g_logged0 = self.outcome_model.predict(
                prompts, responses, scores, fold_ids, covariates=logged_covariates
            )

            # 2) Get initial predictions on fresh draws
            fresh_dataset = self._fresh_draws[policy]
            g_fresh0_all = []
            fresh_var_all = []

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

                g_fresh_prompt = self.outcome_model.predict(
                    fresh_prompts,
                    fresh_responses,
                    fresh_scores,
                    fresh_fold_ids,
                    covariates=fresh_covariates,
                )
                g_fresh0_all.append(g_fresh_prompt.mean())

                if len(g_fresh_prompt) > 1:
                    fresh_var_all.append(g_fresh_prompt.var())
                else:
                    fresh_var_all.append(0.0)

            g_fresh0 = np.array(g_fresh0_all)
            fresh_var = np.array(fresh_var_all)

            # 3) Targeting step: solve for ε and update Q0 → Q*
            if self.link == "logit":
                eps, info = self._solve_logistic_fluctuation(
                    g_logged0, rewards, weights
                )
                # Clever covariate is W, so update is ε·W
                g_logged_star = _expit(_logit(g_logged0) + eps * weights)
                # DO NOT shift the fresh draw predictions - only the logged term
                g_fresh_star = g_fresh0
            else:  # identity link
                eps, info = self._solve_identity_fluctuation(
                    g_logged0, rewards, weights
                )
                # Update with ε·W for identity link too
                g_logged_star = np.clip(g_logged0 + eps * weights, 0.0, 1.0)
                # DO NOT shift the fresh draw predictions
                g_fresh_star = g_fresh0

            # 4) TMLE estimate = DM + IPS correction (using targeted predictions)
            dm_term = float(g_fresh_star.mean())
            # IPS correction (no oracle augmentation)
            ips_corr_total = weights * (rewards - g_logged_star)
            ips_corr = float(np.mean(ips_corr_total))
            psi = dm_term + ips_corr

            # 5) Standard error via empirical IF
            if_contrib = g_fresh_star + ips_corr_total - psi

            # IIC removed - use influence functions directly

            se = (
                float(np.std(if_contrib, ddof=1) / np.sqrt(len(if_contrib)))
                if len(if_contrib) > 1
                else 0.0
            )

            # Store components for diagnostics (like parent DR does)
            self._dm_component[policy] = g_fresh0
            self._ips_correction[policy] = ips_corr_total
            self._fresh_rewards[policy] = rewards  # Actually logged rewards
            self._outcome_predictions[policy] = g_logged0

            # Store influence functions (always needed for proper inference)
            self._influence_functions[policy] = if_contrib

            # Store sample indices for IF alignment in stacking (using parent's helper)
            self._store_sample_indices(policy, data)

            estimates.append(psi)
            standard_errors.append(se)
            n_samples_used[policy] = len(rewards)

            # Keep diagnostics
            info.update(
                dict(
                    link=self.link,
                    epsilon=float(info.get("epsilon", 0.0)),
                    dm=float(dm_term),
                    ips_correction=float(ips_corr),
                    n=len(rewards),
                    mean_weight=float(weights.mean()),
                )
            )
            self._tmle_info[policy] = info

            logger.info(
                f"TMLE[{policy}]: {psi:.4f} ± {se:.4f} "
                f"(ε={info.get('epsilon', 0.0):+.4f}, DM={dm_term:.4f}, IPS_corr={ips_corr:.4f})"
            )

        # Use base class to compute DR diagnostics (with our modifications)
        # We'll override _compute_dr_diagnostics to use original predictions
        base_result = self._create_base_result(
            estimates, standard_errors, n_samples_used
        )

        # Add TMLE-specific metadata
        base_result.method = "tmle"
        if base_result.metadata is None:
            base_result.metadata = {}

        base_result.metadata.update(
            {
                "link": self.link,
                "targeting": self._tmle_info,
            }
        )

        # Add sample indices for IF alignment in stacking
        if hasattr(self, "_if_sample_indices"):
            base_result.metadata["if_sample_indices"] = self._if_sample_indices

        return base_result

    def _create_base_result(
        self,
        estimates: List[float],
        standard_errors: List[float],
        n_samples_used: Dict[str, int],
    ) -> EstimationResult:
        """Create base result with DR diagnostics computed on ORIGINAL predictions."""
        from ..diagnostics.dr import compute_dr_policy_diagnostics

        dr_diagnostics_per_policy = {}
        dr_calibration_data = {}

        for idx, policy in enumerate(self.sampler.target_policies):
            if policy not in self._fresh_draws or np.isnan(estimates[idx]):
                continue

            # Get data and components
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                continue

            weights = self.get_weights(policy)
            if weights is None:
                continue
            rewards = np.array([d["reward"] for d in data], dtype=float)
            scores = np.array([d.get("judge_score") for d in data], dtype=float)
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

            # Get fold assignments - strict mode (no fallback)
            unknown_pids = [
                pid for pid in prompt_ids if pid not in self._promptid_to_fold
            ]
            if unknown_pids:
                raise ValueError(
                    f"Missing fold assignments for {len(unknown_pids)} samples in policy '{policy}'. "
                    f"Example prompt_ids: {unknown_pids[:3]}. "
                    f"Ensure calibration was done with enable_cross_fit=True."
                )
            fold_ids = np.array(
                [self._promptid_to_fold[pid] for pid in prompt_ids], dtype=int
            )

            # Get ORIGINAL predictions (not targeted) for honest R²
            g_logged0 = self.outcome_model.predict(
                prompts, responses, scores, fold_ids, covariates=logged_covariates
            )

            # Get fresh predictions
            fresh_dataset = self._fresh_draws[policy]
            g_fresh0_all = []
            fresh_var_all = []

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

                g_fresh_prompt = self.outcome_model.predict(
                    fresh_prompts,
                    fresh_responses,
                    fresh_scores,
                    fresh_fold_ids,
                    covariates=fresh_covariates,
                )
                g_fresh0_all.append(g_fresh_prompt.mean())

                if len(g_fresh_prompt) > 1:
                    fresh_var_all.append(g_fresh_prompt.var())
                else:
                    fresh_var_all.append(0.0)

            g_fresh0 = np.array(g_fresh0_all)
            fresh_var = np.array(fresh_var_all)

            # Use base class helper for consistent diagnostic computation
            dr_diagnostics_per_policy[policy] = self._compute_policy_diagnostics(
                policy, estimates[idx]
            )

            # Store calibration data
            dr_calibration_data[policy] = {
                "g_logged": g_logged0,  # Original predictions
                "rewards": rewards,
            }

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
                "tmle_score_abs_mean": {
                    p: abs(d.get("score_mean", 0.0))
                    for p, d in dr_diagnostics_per_policy.items()
                },
                "tmle_max_score_z": max(
                    abs(d.get("score_z", 0.0))
                    for d in dr_diagnostics_per_policy.values()
                ),
            }

        # Get IPS diagnostics from the base estimator
        ips_diag = None
        if hasattr(self.ips_estimator, "get_diagnostics"):
            ips_diag = self.ips_estimator.get_diagnostics()

        # Build DRDiagnostics object
        diagnostics = self._build_dr_diagnostics(
            estimates=estimates,
            standard_errors=standard_errors,
            n_samples_used=n_samples_used,
            dr_diagnostics_per_policy=dr_diagnostics_per_policy,
            ips_diagnostics=ips_diag,
        )

        # Create metadata without influence functions (they're first-class now)
        metadata = {
            "cross_fitted": True,
            "n_folds": self.n_folds,
            "fresh_draws_policies": list(self._fresh_draws.keys()),
            "dr_diagnostics": dr_diagnostics_per_policy,  # Keep for backward compatibility
            "dr_overview": dr_overview,
            "dr_calibration_data": dr_calibration_data,
        }

        result = EstimationResult(
            estimates=np.array(estimates, dtype=float),
            standard_errors=np.array(standard_errors, dtype=float),
            n_samples_used=n_samples_used,
            method="tmle",
            influence_functions=self._influence_functions,
            diagnostics=diagnostics,  # Add the DRDiagnostics object
            metadata=metadata,
        )

        # Apply OUA jackknife using base class method
        self._apply_oua_jackknife(result)

        return result

    def _solve_logistic_fluctuation(
        self, q0_logged: np.ndarray, rewards: np.ndarray, weights: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """Solve for ε in logit(Q*) = logit(Q0) + ε·W using weighted logistic MLE.

        The clever covariate is W, so the fluctuation is ε·W, not just ε.
        Uses scale-aware convergence: |score| / sqrt(Fisher) < tol
        """
        # Guards
        q0 = np.clip(q0_logged, _EPS, 1.0 - _EPS)
        eta0 = _logit(q0)

        eps = 0.0
        converged = False
        score_val = None
        fisher_val = None
        normalized_score = None

        # If all weights ~0, skip targeting
        if float(np.sum(weights)) <= 0:
            return 0.0, dict(
                epsilon=0.0,
                converged=True,
                iters=0,
                score=0.0,
                fisher=0.0,
                normalized_score=0.0,
            )

        for t in range(self.max_iter):
            # CRITICAL FIX: clever covariate is W, so fluctuation is ε·W
            mu = _expit(eta0 + eps * weights)
            # weighted score (sum w*(y-mu))
            score = float(np.sum(weights * (rewards - mu)))
            # Fisher information for ε with clever covariate W
            fisher = float(np.sum((weights**2) * mu * (1.0 - mu)))

            score_val = score
            fisher_val = fisher

            # Scale-aware convergence: |score| / sqrt(Fisher)
            # This is the normalized score that accounts for parameter scale
            if fisher > 1e-12:
                normalized_score = abs(score) / np.sqrt(fisher)
                # Scale-aware tolerance (much more stringent and meaningful)
                if normalized_score <= self.tol:
                    converged = True
                    break
            else:
                # Fisher too small - declare converged if score is small
                if abs(score) <= self.tol:
                    converged = True
                    normalized_score = abs(score)  # Use raw score as fallback
                    break
                logger.warning(
                    "TMLE logistic fluctuation: near-singular Fisher; stopping early."
                )
                break

            # Newton update; cap step to avoid giant jumps
            step = score / fisher
            if abs(step) > 5.0:
                step = np.sign(step) * 5.0
            eps += step

        if not converged:
            logger.info(
                f"TMLE logistic fluctuation did not fully converge: "
                f"iters={self.max_iter}, normalized_score={normalized_score:.3e}, "
                f"|score|={abs(score_val) if score_val is not None else 0:.3e}, fisher={fisher_val:.3e}"
            )

        return float(eps), dict(
            epsilon=float(eps),
            converged=bool(converged),
            iters=int(t + 1),
            score=float(score_val if score_val is not None else 0.0),
            fisher=float(fisher_val if fisher_val is not None else 0.0),
            normalized_score=float(
                normalized_score if normalized_score is not None else 0.0
            ),
        )

    def _solve_identity_fluctuation(
        self, q0_logged: np.ndarray, rewards: np.ndarray, weights: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """Solve for ε in Q* = Q0 + ε·W via the EIF score equation.

        The clever covariate is W, so the update is ε·W.
        """
        num = float(np.sum(weights * (rewards - q0_logged)))
        den = float(np.sum(weights**2))  # CRITICAL FIX: denominator is sum(W²)
        if den <= 1e-12:
            return 0.0, dict(
                epsilon=0.0, converged=True, iters=1, score=num, fisher=den
            )
        eps = num / den
        return float(eps), dict(
            epsilon=float(eps),
            converged=True,
            iters=1,
            score=float(num),
            fisher=den,
        )
