"""Clean implementation of stacked DR estimator using oracle IC approach."""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any, cast
from concurrent.futures import ThreadPoolExecutor, as_completed

from cje.estimators.base_estimator import BaseCJEEstimator
from cje.data.models import EstimationResult
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.diagnostics.robust_inference import cluster_robust_se
from cje.data.folds import get_fold

logger = logging.getLogger(__name__)


class StackedDREstimator(BaseCJEEstimator):
    """
    Stacks DR estimators via influence function variance minimization.

    Uses the oracle IC approach: directly computes w^T φ(Z) where w are
    optimal weights. This is theoretically justified because weight learning
    is O_p(n^{-1}) and doesn't affect the asymptotic distribution.

    Key features:
    - Runs multiple DR estimators with shared resources
    - Computes optimal weights via regularized covariance
    - Simple, clean SE computation
    - No unnecessary CV machinery
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        estimators: Optional[List[str]] = None,
        covariance_regularization: float = 1e-4,
        min_weight: float = 0.0,
        weight_shrinkage: float = 0.0,  # No shrinkage - let optimizer find optimal weights
        parallel: bool = True,
        seed: int = 42,
        n_folds: int = 5,
        oua_jackknife: bool = True,
        force_uniform_weights: bool = False,  # Hidden flag for sanity checks
        **kwargs: Any,
    ):
        """Initialize the stacked estimator.

        Args:
            sampler: PrecomputedSampler with calibrated data
            estimators: List of estimator names to stack
                Default: ["dr-cpo", "tmle", "mrdr"] - core DR estimators only
            covariance_regularization: Ridge regularization for numerical stability
            min_weight: Minimum weight for any estimator (for stability)
            weight_shrinkage: Shrinkage toward uniform weights (0=none, 1=uniform)
            parallel: If True, run component estimators in parallel
            seed: Random seed for reproducibility
            n_folds: Number of folds for component cross-fitting
            oua_jackknife: If True, enable OUA for component estimators
            force_uniform_weights: If True, force uniform weights (for debugging)
            **kwargs: Additional arguments (reward_calibrator, etc.)
        """
        super().__init__(sampler)

        # Configuration
        self.estimators = estimators or [
            "dr-cpo",
            "tmle",
            "mrdr",
        ]
        self.covariance_regularization = covariance_regularization
        self.min_weight = min_weight
        self.weight_shrinkage = weight_shrinkage
        self.parallel = parallel
        self.seed = seed
        self.n_folds = n_folds
        self.oua_jackknife = oua_jackknife
        self.force_uniform_weights = force_uniform_weights

        # Extract additional configuration
        self.reward_calibrator = kwargs.pop("reward_calibrator", None)
        self.use_calibrated_weights = kwargs.pop("use_calibrated_weights", True)
        self.weight_mode = kwargs.pop("weight_mode", "hajek")

        # MC-aware stacking configuration (optional)
        # If enabled, incorporate per-component MC variance from fresh draws
        # into the stacking objective (rank-1 update). See _compute_optimal_weights.
        self.include_mc_in_objective: bool = bool(
            kwargs.pop("include_mc_in_objective", True)
        )
        self.mc_lambda: float = float(kwargs.pop("mc_lambda", 1.0))

        # Storage for results
        self.component_results: Dict[str, Optional[EstimationResult]] = {}
        self.component_estimators: Dict[str, Any] = {}
        self.weights_per_policy: Dict[str, np.ndarray] = {}
        self.diagnostics_per_policy: Dict[str, Dict] = {}
        self._influence_functions: Dict[str, np.ndarray] = {}

        # Use local RNG to avoid global side effects
        self.rng = np.random.default_rng(self.seed)

    def fit(self) -> None:
        """Fit is a no-op for stacked estimator (components fit themselves)."""
        self._fitted = True

    def add_fresh_draws(self, policy: str, fresh_draws_df: Any) -> None:
        """Add fresh draws for DR components.

        This is passed through to component estimators.

        Args:
            policy: The policy name
            fresh_draws_df: Fresh draws data
        """
        # Store fresh draws to pass to components
        if not hasattr(self, "_fresh_draws_per_policy"):
            self._fresh_draws_per_policy = {}
        self._fresh_draws_per_policy[policy] = fresh_draws_df

    def estimate(self) -> EstimationResult:
        """Run all component estimators and stack them optimally."""
        if not self._fitted:
            self.fit()

        # Step 1: Run all component estimators
        self._run_component_estimators()

        # Check for failures
        valid_estimators = [
            name
            for name, result in self.component_results.items()
            if result is not None
        ]

        if len(valid_estimators) == 0:
            raise RuntimeError("All component estimators failed")

        if len(valid_estimators) == 1:
            logger.warning(
                f"Only one valid estimator ({valid_estimators[0]}), using it directly"
            )
            result = self.component_results[valid_estimators[0]]
            if result is None:
                raise RuntimeError("Single valid estimator returned None")
            return result

        # Step 2: Stack estimates for each policy
        stacked_estimates = []
        stacked_ses = []
        stacked_ifs = {}

        for policy_idx, policy in enumerate(self.sampler.target_policies):
            stack_result = self._stack_policy(policy, policy_idx, valid_estimators)

            if stack_result is None:
                stacked_estimates.append(np.nan)
                stacked_ses.append(np.nan)
            else:
                estimate, se, stacked_if = stack_result
                stacked_estimates.append(estimate)
                stacked_ses.append(se)
                stacked_ifs[policy] = stacked_if
                self._influence_functions[policy] = stacked_if

        # Step 3: Create result with diagnostics
        metadata = self._build_metadata(valid_estimators)

        # Get n_samples_used from first valid component
        n_samples_used = {}
        if valid_estimators:
            first_result = self.component_results[valid_estimators[0]]
            if first_result:
                n_samples_used = first_result.n_samples_used

        result = EstimationResult(
            estimates=np.array(stacked_estimates),
            standard_errors=np.array(stacked_ses),
            n_samples_used=n_samples_used,
            method=f"StackedDR({', '.join(valid_estimators)})",
            influence_functions=stacked_ifs,
            metadata=metadata,
            diagnostics=None,
        )

        # Add any diagnostics from components
        result.diagnostics = self._build_diagnostics(valid_estimators, result)

        # Apply stacked-level oracle jackknife by linear combination of component jackknifes
        self._apply_stacked_oua(result)

        # Store DF info for t-based CIs (after OUA to account for oracle uncertainty)
        self._store_df_info(result)

        return result

    def _apply_stacked_oua(self, result: EstimationResult) -> None:
        """Augment stacked SEs with oracle jackknife (OUA) via linear combination.

        For fixed stacking weights w, the stacked jackknife path is ψ_stack^(−f) = Σ w_k ψ_k^(−f).
        We compute var_oracle_stack = (K-1)/K * Var_f(ψ_stack^(−f)) and add it to standard_errors in-place.
        """
        # Skip OUA when we have 100% oracle coverage (no oracle uncertainty)
        try:
            if (
                hasattr(self.sampler, "oracle_coverage")
                and self.sampler.oracle_coverage == 1.0
            ):
                # At 100% coverage, we use raw oracle labels, so no oracle uncertainty
                if result.metadata is None:
                    result.metadata = {}
                result.metadata.setdefault("se_components", {})
                result.metadata["se_components"][
                    "oracle_uncertainty_skipped"
                ] = "100% oracle coverage"
                return
        except Exception:
            pass  # Continue with normal OUA calculation if we can't check coverage

        try:
            policies = list(self.sampler.target_policies)
        except Exception:
            policies = (
                list(result.influence_functions.keys())
                if result.influence_functions
                else []
            )

        if not policies:
            return

        robust_ses: List[float] = []
        var_oracle_per_policy: Dict[str, float] = {}
        jackknife_counts: Dict[str, int] = {}
        contributors_map: Dict[str, List[str]] = {}

        for idx, policy in enumerate(policies):
            # Retrieve weights and component names used for this policy
            weights = self.weights_per_policy.get(policy)
            diag = self.diagnostics_per_policy.get(policy, {})
            used_names = diag.get("components", []) if isinstance(diag, dict) else []

            base_se = (
                float(result.standard_errors[idx])
                if idx < len(result.standard_errors)
                else float("nan")
            )
            var_oracle = 0.0
            K = 0
            contributors: List[str] = []

            if weights is not None and used_names:
                # Collect component jackknife arrays
                jack_list: List[Tuple[float, np.ndarray]] = []
                for w, name in zip(weights, used_names):
                    comp = self.component_estimators.get(name)
                    if not comp or not hasattr(comp, "get_oracle_jackknife"):
                        continue
                    try:
                        jarr = comp.get_oracle_jackknife(policy)
                    except Exception:
                        jarr = None
                    if jarr is None or len(jarr) == 0:
                        continue
                    jack_list.append((float(w), np.asarray(jarr, dtype=float)))
                    contributors.append(name)

                if jack_list:
                    # Align lengths by truncating to minimum K
                    K = min(arr.shape[0] for _, arr in jack_list)
                    if K >= 2:
                        stacked_jack = np.zeros(K, dtype=float)
                        for w, arr in jack_list:
                            stacked_jack += w * arr[:K]
                        mu = float(np.mean(stacked_jack))
                        var_oracle = ((K - 1) / K) * float(
                            np.mean((stacked_jack - mu) ** 2)
                        )

            # Update standard_errors in-place with oracle variance
            if (
                idx < len(result.standard_errors)
                and np.isfinite(base_se)
                and var_oracle >= 0.0
            ):
                result.standard_errors[idx] = float(np.sqrt(base_se**2 + var_oracle))

            var_oracle_per_policy[policy] = var_oracle
            jackknife_counts[policy] = int(K)
            contributors_map[policy] = contributors

        # If we computed anything meaningful, record it in metadata
        if any(k >= 2 for k in jackknife_counts.values()):
            # Merge into metadata
            if result.metadata is None:
                result.metadata = {}
            result.metadata.setdefault("se_components", {})
            result.metadata["se_components"].update(
                {
                    "includes_oracle_uncertainty": True,
                    "oracle_variance_per_policy": var_oracle_per_policy,
                    "oracle_jackknife_counts": jackknife_counts,
                    "oracle_contributors": contributors_map,
                }
            )
            # Keep backward compat key
            result.metadata.setdefault("oua", {})
            result.metadata["oua"].update(
                {
                    "source": "stacked-linear-combo",
                    "var_oracle_per_policy": var_oracle_per_policy,
                    "jackknife_counts": jackknife_counts,
                    "contributors": contributors_map,
                }
            )

    def _store_df_info(self, result: EstimationResult) -> None:
        """Store degrees of freedom information for t-based CI computation.

        This method stores DF information that EstimationResult.confidence_interval()
        will use to automatically compute t-based CIs.

        The degrees of freedom is determined by the limiting factor:
        - df_cluster from cluster-robust SE (typically n_folds - 1)
        - If OUA was applied: min(df_cluster, K - 1) where K is number of oracle folds

        Args:
            result: EstimationResult with estimates and standard_errors already populated
                   (including OUA adjustment if applicable)

        Side effects:
            - Stores DF info in result.metadata["degrees_of_freedom"]
        """
        from scipy import stats

        if not hasattr(self, "_df_info"):
            # No DF tracking (shouldn't happen but be defensive)
            logger.debug("No DF tracking available, skipping DF storage")
            return

        try:
            policies = list(self.sampler.target_policies)
        except Exception:
            policies = (
                list(result.influence_functions.keys())
                if result.influence_functions
                else []
            )

        if not policies:
            return

        df_info = {}

        for i, policy in enumerate(policies):
            if np.isnan(result.estimates[i]) or np.isnan(result.standard_errors[i]):
                continue

            # Get cluster DF
            base_df_info = self._df_info.get(policy)
            if not base_df_info:
                continue

            df_cluster = base_df_info.get("df", self.n_folds - 1)

            # If OUA was applied, get oracle DF and take minimum
            df_final = df_cluster
            oua_applied = False
            if (
                result.metadata
                and "se_components" in result.metadata
                and result.metadata["se_components"].get("includes_oracle_uncertainty")
            ):
                # Check if this policy had oracle jackknife contribution
                jackknife_counts = result.metadata.get("oua", {}).get(
                    "jackknife_counts", {}
                )
                K = jackknife_counts.get(policy, 0)
                if K >= 2:
                    df_oracle = K - 1
                    df_final = min(df_cluster, df_oracle)
                    oua_applied = True
                    logger.debug(
                        f"Policy {policy}: df_cluster={df_cluster}, "
                        f"df_oracle={df_oracle}, df_final={df_final}"
                    )

            # Ensure DF is at least 1
            df_final = max(df_final, 1)

            # Compute t-critical value for logging
            t_crit = stats.t.ppf(1 - 0.05 / 2, df_final)

            df_info[policy] = {
                "df": int(df_final),
                "t_critical": float(t_crit),
                "n_clusters": base_df_info.get("n_clusters", self.n_folds),
                "oua_applied": oua_applied,
            }

            logger.debug(
                f"Stored DF info for {policy}: df={df_final}, t_crit={t_crit:.3f}"
            )

        # Store in metadata
        if not isinstance(result.metadata, dict):
            result.metadata = {}
        result.metadata["degrees_of_freedom"] = df_info
        result.metadata["target_policies"] = policies

    def _stack_policy(
        self, policy: str, policy_idx: int, valid_estimators: List[str]
    ) -> Optional[Tuple[float, float, np.ndarray]]:
        """Stack estimates for a single policy.

        Returns:
            Tuple of (estimate, se, stacked_if) or None if failed
        """
        # Collect and align influence functions
        IF_matrix, used_names, common_indices = self._collect_aligned_ifs(
            policy, valid_estimators
        )

        if IF_matrix is None:
            logger.warning(f"No valid influence functions for policy {policy}")
            return None

        if IF_matrix.shape[1] == 0:
            logger.warning(f"No valid influence functions for policy {policy}")
            return None

        # Clean IF matrix: remove rows with NaN/Inf
        # Cast for mypy - we've already checked IF_matrix is not None above
        IF_matrix_clean = cast(np.ndarray, IF_matrix)

        # Track which rows are valid for proper alignment with common_indices
        valid_rows = np.all(np.isfinite(IF_matrix_clean), axis=1)
        if not np.all(valid_rows):
            n_dropped = (~valid_rows).sum()
            logger.debug(f"Dropping {n_dropped} rows with NaN/Inf for {policy}")
            IF_matrix_clean = IF_matrix_clean[valid_rows]
            # Also filter common_indices to match
            common_indices_array = np.array(sorted(common_indices))
            common_indices_clean = common_indices_array[valid_rows]
        else:
            common_indices_clean = np.array(sorted(common_indices))

        if IF_matrix_clean.shape[0] < 10:
            logger.warning(
                f"Too few valid rows ({IF_matrix_clean.shape[0]}) for {policy}"
            )
            return None

        # Harmonize scales if needed
        IF_matrix_clean = self._harmonize_if_scales(IF_matrix_clean, used_names)
        # NOTE: Sign alignment removed - it's inconsistent to flip IF signs without flipping estimates
        # IF_matrix_clean = self._align_if_signs(IF_matrix_clean)

        # Compute optimal weights with regularization
        weights, weight_diagnostics = self._compute_optimal_weights(
            IF_matrix_clean, used_names, policy
        )
        self.weights_per_policy[policy] = weights
        self.diagnostics_per_policy[policy] = weight_diagnostics

        # Oracle IC approach: simple and theoretically justified
        stacked_if = IF_matrix_clean @ weights

        # CRITICAL FIX: Use cluster-robust SE accounting for fold dependence
        # Recover prompt_ids in the same order used to build stacked_if
        prompt_ids = common_indices_clean.astype(object)

        # Get fold configuration from dataset metadata or use defaults
        n_folds = self.n_folds if hasattr(self, "n_folds") else 5
        fold_seed = 42  # Use consistent seed for fold assignment

        # Reproduce the fold hashing used everywhere else
        fold_ids = np.array(
            [get_fold(str(pid), n_folds, fold_seed) for pid in prompt_ids], dtype=int
        )

        # Compute cluster-robust SE
        res_if = cluster_robust_se(
            data=stacked_if,
            cluster_ids=fold_ids,
            statistic_fn=lambda x: np.mean(x),
            influence_fn=lambda x: x,  # already an IF
            alpha=0.05,
        )
        se_if = float(res_if["se"])

        # Store DF info for t-based CIs
        if not hasattr(self, "_df_info"):
            self._df_info: Dict[str, Dict[str, Any]] = {}

        df_cluster = res_if.get("df", res_if.get("n_clusters", len(stacked_if)) - 1)
        from scipy import stats

        t_crit = stats.t.ppf(1 - 0.05 / 2, df_cluster)
        self._df_info[policy] = {
            "df": int(df_cluster),
            "t_critical": float(t_crit),
            "n_clusters": int(res_if.get("n_clusters", len(stacked_if))),
        }

        # Add MC variance if present
        mc_var = self._aggregate_mc_variance(policy, used_names, weights)
        se = float(np.sqrt(se_if**2 + mc_var))

        # Compute point estimate
        component_estimates = []
        for est_name in used_names:
            result = self.component_results[est_name]
            if result:
                component_estimates.append(result.estimates[policy_idx])

        if component_estimates:
            estimate = np.dot(weights, component_estimates)
        else:
            estimate = np.nan

        # Store diagnostics
        weight_diagnostics["se_if"] = se_if
        weight_diagnostics["mc_var"] = mc_var
        weight_diagnostics["n_samples"] = len(stacked_if)
        weight_diagnostics["components"] = used_names

        # Report overlap fraction
        max_component_length = 0
        for name in valid_estimators:
            result = self.component_results[name]
            if result and result.influence_functions:
                ifs: Any = result.influence_functions.get(policy, [])
                if len(ifs) > max_component_length:
                    max_component_length = len(ifs)
        weight_diagnostics["overlap_fraction"] = len(stacked_if) / max(
            max_component_length, 1
        )

        return estimate, se, stacked_if

    def _solve_simplex_qp(self, Sigma: np.ndarray, max_iter: int = 50) -> np.ndarray:
        """Minimize α^T Σ α s.t. α ≥ 0, 1^T α = 1 using an active-set method."""
        K = Sigma.shape[0]
        active = set(range(K))  # start with all active
        for _ in range(max_iter):
            if not active:
                return np.ones(K) / K
            idx = sorted(active)
            S = Sigma[np.ix_(idx, idx)]
            ones = np.ones(len(idx))
            try:
                Sinv1 = np.linalg.solve(S, ones)
                denom = float(ones @ Sinv1)
                alpha_act = Sinv1 / (denom if abs(denom) > 1e-12 else len(idx))
            except np.linalg.LinAlgError:
                alpha_act = ones / len(idx)
            if np.all(alpha_act >= -1e-12):
                alpha = np.zeros(K)
                for j, k in enumerate(idx):
                    alpha[k] = max(0.0, float(alpha_act[j]))
                s = float(alpha.sum())
                return alpha / (s if s > 0 else 1.0)
            # remove the most negative weight and iterate
            remove_local = np.argmin(alpha_act)
            active.remove(idx[remove_local])
        # fallback
        return np.ones(K) / K

    def _compute_optimal_weights(
        self, IF_matrix: np.ndarray, used_names: List[str], policy: str
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute optimal stacking weights with regularization.

        Solves: min_α α^T Σ α  s.t.  α ≥ 0, 1^T α = 1

        Returns:
            weights: K-dimensional weight vector
            diagnostics: Dictionary with condition numbers, eigenvalues, etc.
        """
        K = IF_matrix.shape[1]

        # Check if forcing uniform weights (for debugging/sanity checks)
        if self.force_uniform_weights:
            w = np.ones(K) / K
            diagnostics = {
                "forced_uniform": True,
                "weights": w.tolist(),
            }
            return w, diagnostics

        # Compute covariance matrix
        centered_IF = IF_matrix - IF_matrix.mean(axis=0, keepdims=True)
        Sigma = np.cov(centered_IF.T)

        # Optionally incorporate per-component Monte Carlo variance as a
        # rank-1 update: Σ_total = Σ_IF + n · λ_mc · (s s^T), where s_k = sqrt(mc_var_k)
        mc_used = False
        if self.include_mc_in_objective and K > 0:
            try:
                s_vals: List[float] = []
                for name in used_names:
                    v = 0.0
                    comp_res = self.component_results.get(name)
                    if (
                        comp_res
                        and hasattr(comp_res, "metadata")
                        and isinstance(comp_res.metadata, dict)
                    ):
                        mcd = comp_res.metadata.get("mc_variance_diagnostics", {}) or {}
                        if (
                            isinstance(mcd, dict)
                            and policy in mcd
                            and isinstance(mcd[policy], dict)
                            and "mc_var" in mcd[policy]
                        ):
                            v = float(mcd[policy]["mc_var"]) or 0.0
                        else:
                            v = float(
                                comp_res.metadata.get("mc_variance", {}).get(
                                    policy, 0.0
                                )
                            )
                    s_vals.append(max(0.0, v) ** 0.5)

                # If all zeros, this has no effect; otherwise add rank-1 term
                s_vec = np.asarray(s_vals, dtype=float).reshape(-1, 1)
                n_rows = IF_matrix.shape[0]
                Sigma = Sigma + (max(1, n_rows) * float(self.mc_lambda)) * (
                    s_vec @ s_vec.T
                )
                mc_used = True
            except Exception:
                mc_used = False

        # Check condition number before regularization (for diagnostics)
        eigenvalues = np.linalg.eigvalsh(Sigma)
        condition_pre = eigenvalues.max() / max(eigenvalues.min(), 1e-10)

        # Add regularization for numerical stability
        Sigma_reg = Sigma + self.covariance_regularization * np.eye(K)

        # Ensure symmetry for numerical stability
        Sigma_reg = 0.5 * (Sigma_reg + Sigma_reg.T)

        # Check condition number after regularization
        eigenvalues_post = np.linalg.eigvalsh(Sigma_reg)
        condition_post = eigenvalues_post.max() / eigenvalues_post.min()

        # Guard against extreme conditioning
        if condition_post > 1e8:
            logger.warning(
                f"Condition number still too high ({condition_post:.2e}), using uniform weights"
            )
            w = np.ones(K) / K
        else:
            # Solve for optimal weights using simplex QP (ensures non-negative weights)
            w = self._solve_simplex_qp(Sigma_reg)

            # Apply shrinkage toward uniform if specified (preserves optimality better than hard floor)
            if self.weight_shrinkage > 0:
                u = np.ones(K) / K
                gamma = np.clip(self.weight_shrinkage, 0.0, 1.0)
                w = (1 - gamma) * w + gamma * u
                w = w / w.sum()

            # If min_weight is needed, use parametric shrinkage rather than hard floor
            # (hard floor breaks KKT optimality)
            if self.min_weight > 0:
                u = np.ones(K) / K
                # Heuristic mapping: ensure no weight below min_weight via shrinkage
                gamma_min = min(1.0, self.min_weight * K)
                if gamma_min > self.weight_shrinkage:
                    w = (1 - gamma_min) * w + gamma_min * u
                    w = w / w.sum()

        # Compute pairwise correlations between estimators
        if K > 1:
            correlations = np.corrcoef(centered_IF.T)
            max_corr = np.max(
                np.abs(correlations[np.triu_indices_from(correlations, k=1)])
            )
        else:
            max_corr = 0.0

        # Check for weights at boundary (near zero)
        min_threshold = 1e-6
        weights_at_zero = sum(1 for weight in w if weight <= min_threshold)

        # Warn if many weights are effectively zero
        if weights_at_zero > 2:
            logger.warning(
                f"{weights_at_zero} estimators have near-zero weight (<{min_threshold:.1e}). "
                f"Consider using fewer estimators or checking for redundancy."
            )

        # Warn if condition number is very high
        if condition_post > 1e8:
            logger.warning(
                f"Very high condition number ({condition_post:.2e}) even after regularization. "
                f"Estimators may be too similar for stable stacking."
            )

        diagnostics = {
            "condition_pre": float(condition_pre),
            "condition_post": float(condition_post),
            "eigenvalues": eigenvalues.tolist(),
            "eigenvalues_post": eigenvalues_post.tolist(),
            "min_eigenvalue": float(eigenvalues.min()),
            "max_eigenvalue": float(eigenvalues.max()),
            "regularization_used": float(self.covariance_regularization),
            "max_pairwise_correlation": float(max_corr),
            "weights": w.tolist(),
            "weight_shrinkage": (
                float(self.weight_shrinkage) if self.weight_shrinkage else 0.0
            ),
            "min_weight": float(w.min()),
            "max_weight": float(w.max()),
            "simplex_qp": True,  # Flag indicating we're using the constrained solver
            "weights_at_boundary": int(weights_at_zero),  # Number of weights near zero
            "effective_estimators": int(
                sum(1 for weight in w if weight > 0.05)
            ),  # Estimators with meaningful weight
            "mc_aware": bool(mc_used),
            "mc_lambda": float(self.mc_lambda),
        }

        return w, diagnostics

    def _run_component_estimators(self) -> None:
        """Run all component estimators in parallel or sequentially."""
        if self.parallel:
            self._run_parallel()
        else:
            self._run_sequential()

    def _run_parallel(self) -> None:
        """Run component estimators in parallel."""
        with ThreadPoolExecutor(max_workers=len(self.estimators)) as executor:
            futures = {}
            for est_name in self.estimators:
                future = executor.submit(self._run_single_estimator, est_name)
                futures[future] = est_name

            for future in as_completed(futures):
                est_name = futures[future]
                try:
                    result, estimator = future.result()
                    self.component_results[est_name] = result
                    self.component_estimators[est_name] = estimator
                    if result:
                        logger.info(f"Successfully ran {est_name}")
                except Exception as e:
                    logger.error(f"Failed to run {est_name}: {e}")
                    # Add more detailed error info to help debugging
                    import traceback

                    # Always log the full traceback for debugging oc-dr-cpo and tr-cpo-e issues
                    full_traceback = traceback.format_exc()
                    logger.error(f"Full traceback for {est_name}:\n{full_traceback}")
                    self.component_results[est_name] = None

    def _run_sequential(self) -> None:
        """Run component estimators sequentially."""
        for est_name in self.estimators:
            try:
                result, estimator = self._run_single_estimator(est_name)
                self.component_results[est_name] = result
                self.component_estimators[est_name] = estimator
                if result:
                    logger.info(f"Successfully ran {est_name}")
            except Exception as e:
                logger.error(f"Failed to run {est_name}: {e}")
                # Add more detailed error info to help debugging
                import traceback

                # Always log the full traceback for debugging oc-dr-cpo and tr-cpo-e issues
                full_traceback = traceback.format_exc()
                logger.error(f"Full traceback for {est_name}:\n{full_traceback}")
                self.component_results[est_name] = None

    def _run_single_estimator(self, est_name: str) -> Tuple[EstimationResult, Any]:
        """Run a single component estimator."""
        # Import estimators here to avoid circular imports
        from cje.estimators.dr_base import DRCPOEstimator
        from cje.estimators.tmle import TMLEEstimator
        from cje.estimators.mrdr import MRDREstimator
        from cje.estimators.orthogonalized_calibrated_dr import (
            OrthogonalizedCalibratedDRCPO,
        )
        from cje.estimators.tr_cpo import TRCPOEstimator

        estimator_map = {
            "dr-cpo": DRCPOEstimator,
            "tmle": TMLEEstimator,
            "mrdr": MRDREstimator,
            "oc-dr-cpo": OrthogonalizedCalibratedDRCPO,
            "tr-cpo-e": TRCPOEstimator,  # efficient TR-CPO (m̂(S))
        }

        if est_name not in estimator_map:
            raise ValueError(f"Unknown estimator: {est_name}")

        EstimatorClass = estimator_map[est_name]

        # Create estimator with shared configuration
        # TR-CPO has different constructor (doesn't accept use_calibrated_weights)
        if est_name == "tr-cpo-e":
            estimator = EstimatorClass(
                sampler=self.sampler,
                reward_calibrator=self.reward_calibrator,
                n_folds=self.n_folds,
                weight_mode=self.weight_mode,  # TR-CPO does accept weight_mode
                oua_jackknife=self.oua_jackknife,
                use_efficient_tr=True,  # Ensure we're using the efficient version
            )
        elif est_name == "mrdr":
            # MRDR has omega_mode parameter but also accepts standard DR parameters
            estimator = EstimatorClass(
                sampler=self.sampler,
                reward_calibrator=self.reward_calibrator,
                n_folds=self.n_folds,
                omega_mode="w",  # Use weights for omega
                use_calibrated_weights=self.use_calibrated_weights,
                weight_mode=self.weight_mode,
                # MRDR doesn't take oua_jackknife
            )
        else:
            # DR-CPO, TMLE, OC-DR-CPO all accept these parameters
            estimator = EstimatorClass(
                sampler=self.sampler,
                reward_calibrator=self.reward_calibrator,
                n_folds=self.n_folds,
                use_calibrated_weights=self.use_calibrated_weights,
                weight_mode=self.weight_mode,
                oua_jackknife=self.oua_jackknife,
            )

        # Pass fresh draws if available
        if hasattr(self, "_fresh_draws_per_policy"):
            for policy, fresh_draws in self._fresh_draws_per_policy.items():
                estimator.add_fresh_draws(policy, fresh_draws)

        # Run estimation
        result = estimator.fit_and_estimate()
        return result, estimator

    def _collect_aligned_ifs(
        self, policy: str, valid_estimators: List[str]
    ) -> Tuple[Optional[np.ndarray], List[str], set]:
        """Collect and align influence functions from components.

        Returns:
            IF_matrix: n x K matrix of aligned IFs
            used_names: List of estimator names that contributed IFs
            common_indices: Set of sample indices used for alignment
        """
        # Find common sample indices across all components
        common_indices = None
        for est_name in valid_estimators:
            result = self.component_results[est_name]
            if (
                result
                and result.influence_functions
                and policy in result.influence_functions
            ):
                if_data = result.influence_functions[policy]
                if if_data is not None and len(if_data) > 0:
                    # Get indices from metadata if available
                    indices = None
                    if hasattr(result, "metadata") and isinstance(
                        result.metadata, dict
                    ):
                        if_indices = result.metadata.get("if_sample_indices", {})
                        indices = if_indices.get(policy)

                    if indices is not None:
                        if common_indices is None:
                            common_indices = set(indices)
                        else:
                            common_indices = common_indices.intersection(indices)

        # Fallback if no indices provided: align by position up to min length
        if common_indices is None:
            logger.warning(
                f"No if_sample_indices provided by components for {policy}. "
                f"Falling back to position-based alignment (less reliable)."
            )
            # Find minimum length across all components with IFs
            min_length: Optional[int] = None
            for est_name in valid_estimators:
                result = self.component_results[est_name]
                if (
                    result
                    and result.influence_functions
                    and policy in result.influence_functions
                ):
                    if_data = result.influence_functions[policy]
                    if if_data is not None and len(if_data) > 0:
                        if min_length is None:
                            min_length = len(if_data)
                        else:
                            min_length = min(min_length, len(if_data))

            if min_length is None:
                logger.warning(f"No valid influence functions found for {policy}")
                return None, [], set()

            # Use position-based alignment up to min length
            IF_columns = []
            used_names = []
            for est_name in valid_estimators:
                result = self.component_results[est_name]
                if (
                    result
                    and result.influence_functions
                    and policy in result.influence_functions
                ):
                    if_data = result.influence_functions[policy]
                    if if_data is not None and len(if_data) >= min_length:
                        IF_columns.append(if_data[:min_length])
                        used_names.append(est_name)

            if len(IF_columns) == 0:
                return None, [], set()

            IF_matrix = np.column_stack(IF_columns)
            # Return pseudo-indices for position-based alignment
            return IF_matrix, used_names, set(range(min_length))

        if len(common_indices) == 0:
            logger.warning(f"No common samples found for {policy}")
            return None, [], set()

        # Collect aligned IFs
        IF_columns = []
        used_names = []

        for est_name in valid_estimators:
            result = self.component_results[est_name]
            if (
                result
                and result.influence_functions
                and policy in result.influence_functions
            ):
                if_data = result.influence_functions[policy]
                if if_data is not None and len(if_data) > 0:
                    # Get indices for alignment
                    if hasattr(result, "metadata") and isinstance(
                        result.metadata, dict
                    ):
                        if_indices = result.metadata.get("if_sample_indices", {})
                        indices = if_indices.get(policy)

                        if indices is not None:
                            # Create alignment mapping
                            idx_to_pos = {idx: i for i, idx in enumerate(indices)}
                            aligned_if = np.zeros(len(common_indices))

                            for j, common_idx in enumerate(sorted(common_indices)):
                                if common_idx in idx_to_pos:
                                    aligned_if[j] = if_data[idx_to_pos[common_idx]]

                            IF_columns.append(aligned_if)
                            used_names.append(est_name)

        if len(IF_columns) == 0:
            return None, [], set()

        IF_matrix = np.column_stack(IF_columns)
        return IF_matrix, used_names, common_indices

    def _harmonize_if_scales(
        self, IF_matrix: np.ndarray, used_names: List[str]
    ) -> np.ndarray:
        """Ensure all IFs are on the same scale."""
        # Simple check: if SEs differ by more than 50%, there might be a scale issue
        for i in range(IF_matrix.shape[1]):
            col_se = np.std(IF_matrix[:, i], ddof=1) / np.sqrt(len(IF_matrix[:, i]))
            expected_se = col_se  # This would come from component metadata

            # For now, we trust that components are providing correctly scaled IFs
            # Could add more sophisticated checks here if needed

        return IF_matrix

    def _aggregate_mc_variance(
        self, policy: str, used_names: List[str], weights: np.ndarray
    ) -> float:
        """Aggregate Monte Carlo variance from components.

        Conservative assumption: perfect correlation across components
        (they share the same fresh draws).
        """
        mc_vars = []

        for est_name in used_names:
            result = self.component_results[est_name]
            v = 0.0
            if (
                result
                and hasattr(result, "metadata")
                and isinstance(result.metadata, dict)
            ):
                # Prefer detailed diagnostics if present
                mcd = result.metadata.get("mc_variance_diagnostics", {})
                if policy in mcd and "mc_var" in mcd[policy]:
                    v = float(mcd[policy]["mc_var"])
                else:
                    # Legacy key (if ever set)
                    v = float(result.metadata.get("mc_variance", {}).get(policy, 0.0))
            mc_vars.append(v)

        if not mc_vars:
            return 0.0

        # Conservative assumption: perfect correlation across components
        # (they share the same fresh draws)
        return float((np.sum(weights * np.sqrt(mc_vars))) ** 2)

    def _build_metadata(self, valid_estimators: List[str]) -> Dict[str, Any]:
        """Build metadata for the result."""
        metadata = {
            "stacking_weights": self.weights_per_policy,
            "stacking_diagnostics": self.diagnostics_per_policy,
            "valid_estimators": valid_estimators,
            "failed_estimators": [
                e for e in self.estimators if e not in valid_estimators
            ],
            "n_folds": self.n_folds,
            "covariance_regularization": self.covariance_regularization,
        }

        # Add sample indices if available
        if_sample_indices: Dict[str, Any] = {}
        for name in valid_estimators:
            result = self.component_results[name]
            if (
                result
                and hasattr(result, "metadata")
                and isinstance(result.metadata, dict)
            ):
                comp_indices = result.metadata.get("if_sample_indices", {})
                if comp_indices and not if_sample_indices:
                    if_sample_indices = comp_indices
                    break

        if if_sample_indices:
            metadata["if_sample_indices"] = if_sample_indices

        return metadata

    def _build_diagnostics(
        self, valid_estimators: List[str], result: EstimationResult
    ) -> Optional[Any]:
        """Build DRDiagnostics by aggregating component metrics using stacking weights."""
        from ..diagnostics.models import DRDiagnostics, Status

        # Aggregate MC diagnostics from DR components (keep existing logic)
        mc_diagnostics: Dict[str, Dict[str, Any]] = {}
        for est_name in valid_estimators:
            if est_name in self.component_results:
                comp_result = self.component_results[est_name]
                if comp_result and hasattr(comp_result, "metadata"):
                    # Check for MC diagnostics in component metadata
                    if "mc_variance_diagnostics" in comp_result.metadata:
                        mc_diag = comp_result.metadata["mc_variance_diagnostics"]
                        # Store per-estimator MC diagnostics
                        for policy, diag in mc_diag.items():
                            if policy not in mc_diagnostics:
                                mc_diagnostics[policy] = {}
                            mc_diagnostics[policy][est_name] = diag

        # Add aggregated MC diagnostics to result metadata
        if mc_diagnostics:
            if "mc_diagnostics" not in result.metadata:
                result.metadata["mc_diagnostics"] = {}

            # Aggregate across estimators for each policy
            for policy in mc_diagnostics:
                # Use the first available estimator's MC diagnostics as representative
                # (they should all be the same since they use the same fresh draws)
                for est_name, diag in mc_diagnostics[policy].items():
                    result.metadata["mc_diagnostics"][policy] = diag
                    break  # Use first one

            # Extract M_min and M_max for top-level reporting
            all_M_min_values = []
            all_M_max_values = []
            for policy_diag in result.metadata["mc_diagnostics"].values():
                # Look for min_draws_per_prompt and max_draws_per_prompt
                if "min_draws_per_prompt" in policy_diag:
                    all_M_min_values.append(policy_diag["min_draws_per_prompt"])
                if "max_draws_per_prompt" in policy_diag:
                    all_M_max_values.append(policy_diag["max_draws_per_prompt"])
                # Also add M_min and M_max aliases for convenience
                if "min_draws_per_prompt" in policy_diag:
                    policy_diag["M_min"] = policy_diag["min_draws_per_prompt"]
                if "max_draws_per_prompt" in policy_diag:
                    policy_diag["M_max"] = policy_diag["max_draws_per_prompt"]

            if all_M_min_values:
                result.metadata["M_min"] = min(all_M_min_values)
            if all_M_max_values:
                result.metadata["M_max"] = max(all_M_max_values)

        # Now aggregate component diagnostics using stacking weights
        policies = list(self.sampler.target_policies)

        # Collect per-policy weighted metrics
        weighted_r2_per_policy = {}
        weighted_rmse_per_policy = {}
        max_tail_per_policy = {}

        for policy in policies:
            stacking_weights = result.metadata.get("stacking_weights", {}).get(
                policy, []
            )
            components = (
                result.metadata.get("stacking_diagnostics", {})
                .get(policy, {})
                .get("components", [])
            )

            if len(stacking_weights) == 0 or len(components) == 0:
                continue

            r2_values = []
            rmse_values = []
            tail_values = []

            for comp, alpha in zip(components, stacking_weights):
                if alpha < 0.001:  # Skip negligible weights
                    continue
                comp_result = self.component_results.get(comp)
                if comp_result and comp_result.diagnostics:
                    d = comp_result.diagnostics

                    # Get per-policy values if available
                    if (
                        hasattr(d, "dr_diagnostics_per_policy")
                        and policy in d.dr_diagnostics_per_policy
                    ):
                        pd = d.dr_diagnostics_per_policy[policy]
                        r2 = pd.get("r2_oof", 0)
                        rmse = pd.get("residual_rmse", 0)
                        tail = pd.get("if_tail_ratio_99_5", 0)
                    else:
                        # Fallback to aggregate values
                        r2 = (
                            d.outcome_r2_range[0]
                            if hasattr(d, "outcome_r2_range")
                            else 0
                        )
                        rmse = (
                            d.outcome_rmse_mean
                            if hasattr(d, "outcome_rmse_mean")
                            else 0
                        )
                        tail = (
                            d.worst_if_tail_ratio
                            if hasattr(d, "worst_if_tail_ratio")
                            else 0
                        )

                    r2_values.append(alpha * r2)
                    rmse_values.append(alpha * rmse)
                    tail_values.append(tail)

            if r2_values:
                weighted_r2_per_policy[policy] = sum(r2_values)
                weighted_rmse_per_policy[policy] = sum(rmse_values)
                max_tail_per_policy[policy] = max(tail_values) if tail_values else 0.0

        # Aggregate across policies
        if weighted_r2_per_policy:
            min_r2 = min(weighted_r2_per_policy.values())
            max_r2 = max(weighted_r2_per_policy.values())
            avg_rmse = np.mean(list(weighted_rmse_per_policy.values()))
            max_tail = max(max_tail_per_policy.values())
        else:
            min_r2, max_r2, avg_rmse, max_tail = 0.0, 0.0, 0.0, 0.0

        # Get weight diagnostics from first component (they all share same weights)
        weight_ess = 0.0
        ess_per_policy = {}
        max_weight_per_policy = {}
        weight_status = Status.WARNING

        for comp_result in self.component_results.values():
            if comp_result and comp_result.diagnostics:
                d = comp_result.diagnostics
                if hasattr(d, "weight_ess") and d.weight_ess > 0:
                    weight_ess = d.weight_ess
                    ess_per_policy = (
                        d.ess_per_policy if hasattr(d, "ess_per_policy") else {}
                    )
                    max_weight_per_policy = (
                        d.max_weight_per_policy
                        if hasattr(d, "max_weight_per_policy")
                        else {}
                    )
                    weight_status = (
                        d.weight_status
                        if hasattr(d, "weight_status")
                        else Status.WARNING
                    )
                    break  # Found weight diagnostics, use them

        # Create DRDiagnostics
        estimates_dict = {p: result.estimates[i] for i, p in enumerate(policies)}
        se_dict = {p: result.standard_errors[i] for i, p in enumerate(policies)}

        diagnostics = DRDiagnostics(
            estimator_type="Stacked-DR",
            method="stacked_dr",
            n_samples_total=(
                len(self.sampler.dataset.samples)
                if hasattr(self.sampler, "dataset")
                else 0
            ),
            n_samples_valid=(
                self.sampler.n_valid_samples
                if hasattr(self.sampler, "n_valid_samples")
                else 0
            ),
            n_policies=len(policies),
            policies=policies,
            estimates=estimates_dict,
            standard_errors=se_dict,
            n_samples_used=result.n_samples_used,
            # Weight metrics from components (all share same importance weights)
            weight_ess=weight_ess,
            weight_status=weight_status,
            ess_per_policy=ess_per_policy,
            max_weight_per_policy=max_weight_per_policy,
            # DR-specific metrics (weighted averages)
            dr_cross_fitted=True,
            dr_n_folds=self.n_folds,
            outcome_r2_range=(min_r2, max_r2),
            outcome_rmse_mean=avg_rmse,
            worst_if_tail_ratio=max_tail,
            # Store stacking-specific info in metadata (already in result.metadata)
        )

        return diagnostics
