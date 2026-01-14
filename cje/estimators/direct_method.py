"""Direct Method estimator for on-policy evaluation with fresh draws.

This estimator is for scenarios where you have:
- Fresh draws from multiple policies on the same prompts
- Judge scores for all outputs
- Oracle labels on a slice (for calibration)
- NO importance weights (no teacher-forced logprobs)

It computes the calibrated plug-in: V̂(πⱼ) = E[f̂(S)] for each policy.

Key differences from IPS/DR:
- No causal inference (not estimating counterfactual deployment)
- Direct comparison on evaluation set
- Simpler data requirements
- Paired comparisons when prompts match

Use this when you want: "Which policy is best on this eval set?"
Don't use for: "What would happen if we deployed π' in production?"
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging
from dataclasses import dataclass

from .base_estimator import BaseCJEEstimator
from ..data.models import EstimationResult
from ..diagnostics import IPSDiagnostics, Status

logger = logging.getLogger(__name__)


@dataclass
class PolicyData:
    """Data for a single policy in direct mode."""

    policy: str
    judge_scores: np.ndarray
    calibrated_rewards: np.ndarray
    prompt_ids: List[str]


class CalibratedDirectEstimator(BaseCJEEstimator):
    """Calibrated direct method for on-policy evaluation.

    Estimates V(πⱼ) = E_πⱼ[f*(S)] by averaging calibrated rewards over
    fresh draws from each policy.

    This is NOT off-policy evaluation - it evaluates each policy on the
    prompts you provided, without accounting for production context distribution
    or using importance weights.

    Args:
        target_policies: List of policy names to evaluate
        reward_calibrator: Optional calibrator to map judge scores to rewards.
            If None, uses raw judge scores (uncalibrated "naive" mode).
        paired_comparison: If True, use within-prompt differences when possible
        run_diagnostics: Whether to compute diagnostics
        oua_jackknife: Whether to use oracle uncertainty augmentation
        inference_method: How to compute standard errors. One of:
            - "bootstrap": Cluster bootstrap with calibrator refit (default when
              reward_calibrator is provided)
            - "cluster_robust": Cluster-robust SEs without bootstrap
            - "auto": Choose based on data characteristics
            Note: When reward_calibrator=None, defaults to "cluster_robust" since
            bootstrap would create a new calibrator, defeating uncalibrated mode.
        n_bootstrap: Number of bootstrap replicates (default 2000)
        bootstrap_seed: Random seed for bootstrap reproducibility
        use_augmented_estimator: If True, use AIPW-style debiasing in bootstrap
        calibration_policy: If provided, fit calibrator only on this policy's oracle
            samples (for transport experiments). Residual corrections in θ̂_aug
            still use all policies' oracle samples, enabling bias correction for
            policies where the calibrator doesn't transport. If None, use all
            oracle samples for both calibration and residuals (default).
        use_multipolicy_eif: If True, use multi-policy EIF which pools oracle labels
            across all policies with density ratio weighting w_p(z) = f_p(z)/g(z).
            This can yield ~50% RMSE reduction at 5-10% oracle coverage when
            multiple policies share the same calibration curve.

            **Requirement**: Shared calibration assumption - E[Y|Z=z, P=p] = E[Y|Z=z]
            for all policies p. This is STRONGER than mean transport (which only
            requires E[Y - f(Z)] = 0 per policy). Test with binned residual analysis
            before enabling.

            **Caveats**:
            - Only supported with calibration_mode='monotone' (not two_stage)
            - If a policy fails the shared calibration assumption, it will bias
              estimates for ALL policies (unlike per-policy residuals which isolate)
            - Diminishing returns at high oracle coverage (>25%)

            Default False (conservative - uses per-policy residual correction).

    Example:
        >>> # Fresh draws from multiple policies
        >>> estimator = CalibratedDirectEstimator(
        ...     target_policies=["policy_a", "policy_b"],
        ...     reward_calibrator=calibrator  # Optional
        ... )
        >>> estimator.add_fresh_draws("policy_a", fresh_draws_a)
        >>> estimator.add_fresh_draws("policy_b", fresh_draws_b)
        >>> result = estimator.fit_and_estimate()
    """

    def __init__(
        self,
        target_policies: List[str],
        reward_calibrator: Optional[Any] = None,
        paired_comparison: bool = True,
        run_diagnostics: bool = True,
        oua_jackknife: bool = True,
        inference_method: str = "bootstrap",
        n_bootstrap: int = 2000,
        bootstrap_seed: int = 42,
        calibration_data_path: Optional[str] = None,
        use_augmented_estimator: bool = True,
        calibration_policy: Optional[str] = None,
        use_multipolicy_eif: bool = False,
        **kwargs: Any,
    ):
        # Create a minimal dummy sampler for base class compatibility
        # TODO: Refactor base class to not require sampler
        from ..data.precomputed_sampler import PrecomputedSampler
        from ..data.models import Dataset, Sample

        # Create minimal dummy dataset
        dummy_sample = Sample(
            prompt_id="dummy",
            prompt="",
            response="",
            reward=0.5,
            base_policy_logprob=-1.0,
            target_policy_logprobs={p: -1.0 for p in target_policies},
            judge_score=None,
            oracle_label=None,
            metadata={},
        )
        dummy_dataset = Dataset(samples=[dummy_sample], target_policies=target_policies)
        # Suppress warnings from dummy sampler (we don't actually use it)
        import logging

        old_level = logging.getLogger("cje.data.precomputed_sampler").level
        logging.getLogger("cje.data.precomputed_sampler").setLevel(logging.ERROR)
        dummy_sampler = PrecomputedSampler(dummy_dataset)
        logging.getLogger("cje.data.precomputed_sampler").setLevel(old_level)

        super().__init__(
            sampler=dummy_sampler,
            run_diagnostics=run_diagnostics,
            reward_calibrator=reward_calibrator,
            oua_jackknife=oua_jackknife,
            **kwargs,
        )
        self.target_policies = target_policies
        self.paired_comparison = paired_comparison

        # Auto-detect: when reward_calibrator=None, bootstrap would create a new
        # calibrator internally, defeating "naive" (uncalibrated) mode.
        # Default to cluster_robust in this case.
        if reward_calibrator is None and inference_method == "bootstrap":
            logger.info(
                "reward_calibrator=None with inference_method='bootstrap' would create "
                "a calibrator during bootstrap. Defaulting to 'cluster_robust' for "
                "uncalibrated estimation. Pass inference_method='cluster_robust' explicitly "
                "to silence this message."
            )
            inference_method = "cluster_robust"

        self.inference_method = inference_method
        self.n_bootstrap = n_bootstrap
        self.bootstrap_seed = bootstrap_seed
        self.calibration_data_path = calibration_data_path
        self.use_augmented_estimator = use_augmented_estimator
        self.calibration_policy = calibration_policy  # For transport experiments
        self.use_multipolicy_eif = use_multipolicy_eif
        self._policy_data: Dict[str, PolicyData] = {}
        self._fresh_draws: Dict[str, Any] = {}  # Storage for fresh draws
        self._bootstrap_result: Optional[Dict[str, Any]] = (
            None  # Cache bootstrap results
        )

    def add_fresh_draws(self, policy: str, fresh_draws: Any) -> None:
        """Add fresh draws for a target policy.

        Args:
            policy: Target policy name
            fresh_draws: FreshDrawDataset with responses from the policy
        """
        self._fresh_draws[policy] = fresh_draws
        logger.info(
            f"Added fresh draws for policy '{policy}': "
            f"{len(fresh_draws.samples)} samples"
        )

    def _calibration_overlaps_evaluation(self) -> Tuple[bool, int]:
        """Check if calibration and evaluation data are coupled.

        Coupling occurs when oracle labels used for calibration come from
        prompts that are also in the evaluation set. This creates covariance
        between calibration error and evaluation error that additive variance
        decomposition doesn't capture.

        Returns:
            Tuple of (coupled: bool, overlap_count: int)
            - coupled: True if there's any cluster overlap
            - overlap_count: Number of clusters that appear in both sets
        """
        if self.calibration_data_path is not None:
            # Separate calibration data provided - check if there's actual overlap
            # (User may have provided truly independent calibration data)
            # For now, assume separate path means no coupling
            # TODO: Actually load and check cluster intersection
            return False, 0

        # Default case: oracle labels come from fresh draws (coupled)
        # Collect all prompt_ids with oracle labels from fresh draws
        cal_clusters: set = set()
        eval_clusters: set = set()

        for fd in self._fresh_draws.values():
            for sample in fd.samples:
                eval_clusters.add(sample.prompt_id)
                if sample.oracle_label is not None:
                    cal_clusters.add(sample.prompt_id)

        # Compute intersection
        overlap = cal_clusters & eval_clusters
        coupled = len(overlap) > 0

        return coupled, len(overlap)

    def _should_use_bootstrap(self) -> Tuple[bool, str]:
        """Determine if bootstrap inference should be used.

        Bootstrap is preferred when:
        1. inference_method == "bootstrap" (explicit request)
        2. inference_method == "auto" AND:
           - G < 20 clusters (cluster asymptotics unreliable), OR
           - Calibration is coupled with evaluation (covariance term needed)

        Returns:
            Tuple of (use_bootstrap: bool, reason: str)
        """
        if self.inference_method == "bootstrap":
            return True, "explicitly requested"

        if self.inference_method == "cluster_robust":
            return False, "cluster_robust explicitly requested"

        # Auto mode: check conditions
        if self.inference_method == "auto":
            # Check number of clusters
            n_clusters = len(
                set().union(
                    *[
                        set(
                            fd.prompt_ids
                            if hasattr(fd, "prompt_ids")
                            else [s.prompt_id for s in fd.samples]
                        )
                        for fd in self._fresh_draws.values()
                    ]
                )
            )

            if n_clusters < 20:
                return True, f"few clusters (G={n_clusters} < 20)"

            # Check coupling
            coupled, overlap = self._calibration_overlaps_evaluation()
            if coupled:
                return (
                    True,
                    f"calibration/evaluation coupled ({overlap} overlapping clusters)",
                )

            return False, "sufficient clusters and no coupling detected"

        # Unknown method - default to cluster_robust
        logger.warning(
            f"Unknown inference_method '{self.inference_method}', using cluster_robust"
        )
        return False, "unknown method, defaulting to cluster_robust"

    def fit(self) -> None:
        """Prepare data for each policy using fresh draws.

        Direct mode requires fresh draws for each target policy.
        """
        # Verify we have fresh draws for all policies
        missing_policies = set(self.target_policies) - set(self._fresh_draws.keys())
        if missing_policies:
            raise ValueError(
                f"Direct mode requires fresh draws for all target policies. "
                f"Missing fresh draws for: {missing_policies}. "
                f"Either provide fresh_draws_dir or use IPS/DR mode."
            )

        # Get data for each policy from fresh draws
        for policy in self.target_policies:
            fresh_draws = self._fresh_draws[policy]

            # Extract judge scores and compute calibrated rewards
            judge_scores = []
            rewards = []
            prompt_ids = []
            covariates_list = []

            # Check if calibrator expects covariates
            needs_covariates = False
            covariate_names: List[str] = []
            if self.reward_calibrator is not None and hasattr(
                self.reward_calibrator, "covariate_names"
            ):
                covariate_names = self.reward_calibrator.covariate_names or []
                needs_covariates = len(covariate_names) > 0

            for sample in fresh_draws.samples:
                # FreshDrawSample has judge_score as a direct field
                judge_score = sample.judge_score
                judge_scores.append(judge_score)
                prompt_ids.append(sample.prompt_id)

                # Extract covariates from metadata if needed
                if needs_covariates:
                    sample_covariates = []
                    for cov_name in covariate_names:
                        if cov_name not in sample.metadata:
                            raise ValueError(
                                f"Covariate '{cov_name}' not found in fresh draw metadata "
                                f"for policy '{policy}', sample {sample.prompt_id}. "
                                f"Available metadata: {list(sample.metadata.keys())}"
                            )
                        sample_covariates.append(sample.metadata[cov_name])
                    covariates_list.append(sample_covariates)

                # Calibrate judge score to reward if calibrator available
                if self.reward_calibrator is not None:
                    # Prepare covariates if needed
                    if needs_covariates:
                        # Use the covariates we just extracted
                        cov_array = np.array(
                            covariates_list[-1:]
                        )  # Last element as 2D array
                        reward = float(
                            np.clip(
                                self.reward_calibrator.predict(
                                    np.array([judge_score]), covariates=cov_array
                                )[0],
                                0.0,
                                1.0,
                            )
                        )
                    else:
                        # No covariates needed
                        reward = float(
                            np.clip(
                                self.reward_calibrator.predict(np.array([judge_score]))[
                                    0
                                ],
                                0.0,
                                1.0,
                            )
                        )
                else:
                    # No calibrator - use judge score directly
                    reward = float(judge_score)

                rewards.append(reward)

            self._policy_data[policy] = PolicyData(
                policy=policy,
                judge_scores=np.array(judge_scores),
                calibrated_rewards=np.array(rewards),
                prompt_ids=prompt_ids,
            )

            logger.info(
                f"Loaded fresh draws for policy '{policy}': {len(rewards)} samples"
            )

        self._fitted = True
        logger.info(
            f"Prepared data for {len(self._policy_data)} policies from fresh draws"
        )

    def estimate(self) -> EstimationResult:
        """Compute calibrated direct estimates for all policies.

        Returns:
            EstimationResult with:
                - estimates: Mean calibrated reward for each policy
                - standard_errors: Including oracle uncertainty via OUA (or bootstrap)
                - diagnostics: Simplified (no weight metrics)
                - metadata: Mode info and caveats
        """
        self._validate_fitted()

        # Check if bootstrap should be used
        use_bootstrap, bootstrap_reason = self._should_use_bootstrap()
        if use_bootstrap:
            return self._estimate_with_bootstrap(bootstrap_reason)

        # Standard estimation path (cluster-robust SE + OUA jackknife)
        estimates = []
        standard_errors = []
        n_samples_used = {}
        influence_functions = {}

        for policy in self.target_policies:
            if policy not in self._policy_data:
                logger.warning(f"No data for policy '{policy}', using NaN")
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            pdata = self._policy_data[policy]

            # Simple mean estimator
            estimate = float(np.mean(pdata.calibrated_rewards))

            # Influence function: ψ_i = R_i - V̂
            if_values = pdata.calibrated_rewards - estimate
            influence_functions[policy] = if_values

            # Determine SE method based on pairing structure
            n = len(pdata.calibrated_rewards)
            se_method = "standard"
            n_clusters = n
            df_cluster = n - 1  # Degrees of freedom for cluster-robust SE

            # Check if this is paired comparison with aligned prompts
            if self.paired_comparison and len(self._policy_data) > 1:
                # Check alignment across all policies
                prompt_sets = [set(pd.prompt_ids) for pd in self._policy_data.values()]
                prompts_aligned = all(ps == prompt_sets[0] for ps in prompt_sets)

                if prompts_aligned:
                    # Paired comparison: use cluster-robust SE by prompt
                    from ..diagnostics.robust_inference import cluster_robust_se

                    # Map prompt_ids to cluster indices
                    unique_prompts = sorted(set(pdata.prompt_ids))
                    prompt_to_cluster = {pid: i for i, pid in enumerate(unique_prompts)}
                    cluster_ids = np.array(
                        [prompt_to_cluster[pid] for pid in pdata.prompt_ids]
                    )

                    try:
                        res = cluster_robust_se(
                            data=if_values,
                            cluster_ids=cluster_ids,
                            statistic_fn=lambda x: np.mean(x),
                            influence_fn=lambda x: x,
                            alpha=0.05,
                        )
                        se = res["se"]
                        se_method = "cluster_robust"
                        n_clusters = res["n_clusters"]
                        df_cluster = res.get(
                            "df", n_clusters - 1
                        )  # Get DF from cluster-robust SE

                        logger.debug(
                            f"Using cluster-robust SE for {policy}: "
                            f"naive={np.std(if_values, ddof=1) / np.sqrt(n):.6f}, "
                            f"robust={se:.6f}, n_clusters={n_clusters}, df={df_cluster}"
                        )
                    except Exception as e:
                        # Fallback to standard SE if cluster-robust fails
                        logger.debug(
                            f"Cluster-robust SE failed for {policy}: {e}, using standard SE"
                        )
                        se = float(np.std(if_values, ddof=1) / np.sqrt(n))
                        se_method = "standard_fallback"
                        df_cluster = n - 1
                else:
                    # Prompts not fully aligned: use standard SE
                    se = float(np.std(if_values, ddof=1) / np.sqrt(n))
                    se_method = "standard_unpaired"
                    df_cluster = n - 1
            else:
                # Single policy or unpaired mode: use standard SE
                se = float(np.std(if_values, ddof=1) / np.sqrt(n))
                df_cluster = n - 1

            # Store SE method and DF for this policy (used in metadata and CI computation later)
            if not hasattr(self, "_se_methods"):
                self._se_methods = {}
                self._n_clusters = {}
                self._df_cluster = {}
            self._se_methods[policy] = se_method
            self._n_clusters[policy] = n_clusters
            self._df_cluster[policy] = df_cluster

            estimates.append(estimate)
            standard_errors.append(se)
            n_samples_used[policy] = n

            logger.info(
                f"Direct estimate for '{policy}': {estimate:.4f} ± {se:.4f} "
                f"(n={n}, method={se_method})"
            )

        # Build diagnostics
        diagnostics = self._build_diagnostics(
            estimates, standard_errors, n_samples_used
        )

        # Build metadata
        metadata = {
            "mode": "direct",
            "estimand": "on-policy evaluation on provided prompts",
            "caveat": "Does not estimate counterfactual deployment value. Evaluates each policy on the evaluation set.",
            "target_policies": list(self.target_policies),
            "paired_comparison": self.paired_comparison,
            "se_components": {
                "includes_oracle_uncertainty": False,  # Will be set to True by _apply_oua_jackknife()
                "includes_mc_variance": False,
            },
            "se_methods": getattr(self, "_se_methods", {}),
            "n_clusters": getattr(self, "_n_clusters", {}),
        }

        # Check if prompts are aligned across policies
        if self.paired_comparison and len(self._policy_data) > 1:
            prompt_sets = [set(pd.prompt_ids) for pd in self._policy_data.values()]
            if all(ps == prompt_sets[0] for ps in prompt_sets):
                metadata["prompts_aligned"] = True
                metadata["n_prompts"] = len(prompt_sets[0])
                logger.info(
                    f"Prompts aligned across all {len(self._policy_data)} policies. "
                    f"Paired comparisons available."
                )
            else:
                metadata["prompts_aligned"] = False
                logger.warning(
                    "Prompts not fully aligned across policies. "
                    "Paired comparisons not available."
                )

        result = EstimationResult(
            estimates=np.array(estimates),
            standard_errors=np.array(standard_errors),
            n_samples_used=n_samples_used,
            method="calibrated_direct",
            influence_functions=influence_functions,
            diagnostics=diagnostics,
            metadata=metadata,
        )

        # Apply OUA jackknife using base class method
        self._apply_oua_jackknife(result)

        # Store DF info for t-based CIs (computed automatically by EstimationResult.confidence_interval())
        self._store_df_info(result)

        return result

    def _build_diagnostics(
        self,
        estimates: List[float],
        standard_errors: List[float],
        n_samples_used: Dict[str, int],
    ) -> IPSDiagnostics:
        """Build simplified diagnostics for direct mode.

        Note: No weight metrics (ESS, tail indices) since we don't use weights.
        """
        policies = list(self.target_policies)

        # Build estimate dicts
        estimates_dict = {
            p: float(e) for p, e in zip(policies, estimates) if not np.isnan(e)
        }
        se_dict = {
            p: float(se) for p, se in zip(policies, standard_errors) if not np.isnan(se)
        }

        # Get calibration info (if calibrator was provided)
        cal_info = {}
        if self.reward_calibrator and hasattr(
            self.reward_calibrator, "get_calibration_info"
        ):
            cal_info = self.reward_calibrator.get_calibration_info()

        # Count total samples from fresh draws
        total_samples = sum(
            len(self._fresh_draws[p].samples)
            for p in self.target_policies
            if p in self._fresh_draws
        )
        valid_samples = sum(n_samples_used.values())

        diagnostics = IPSDiagnostics(
            estimator_type="Direct",
            method="calibrated_direct",
            n_samples_total=total_samples,
            n_samples_valid=valid_samples,
            n_policies=len(policies),
            policies=policies,
            estimates=estimates_dict,
            standard_errors=se_dict,
            n_samples_used=n_samples_used,
            # No weight metrics for direct mode
            weight_ess=1.0,  # Conceptually, direct mode has perfect "overlap"
            weight_status=Status.GOOD,
            ess_per_policy={p: 1.0 for p in policies},
            max_weight_per_policy={p: 1.0 for p in policies},
            # Calibration metrics (if available)
            calibration_rmse=cal_info.get("rmse"),
            calibration_r2=cal_info.get("r2"),
            calibration_coverage=cal_info.get("oracle_coverage"),
            n_oracle_labels=cal_info.get("n_oracle_labels"),
        )

        return diagnostics

    def get_oracle_jackknife(self, policy: str) -> Optional[np.ndarray]:
        """Compute leave-one-fold-out estimates for oracle uncertainty.

        Args:
            policy: Policy name

        Returns:
            Array of K jackknife estimates, or None if not applicable
        """
        if not self._fitted:
            logger.warning("Estimator not fitted")
            return None

        if self.reward_calibrator is None:
            logger.debug("No reward_calibrator for OUA")
            return None

        if policy not in self._policy_data:
            logger.warning(f"No data for policy {policy}")
            return None

        # Use unified interface to get fold models
        if not hasattr(self.reward_calibrator, "get_fold_models_for_oua"):
            if self.oua_jackknife:
                raise ValueError(
                    "OUA jackknife enabled but calibrator doesn't support it. "
                    "Ensure calibrate_dataset() uses enable_cross_fit=True."
                )
            return None

        fold_models = self.reward_calibrator.get_fold_models_for_oua()
        if not fold_models:
            if self.oua_jackknife:
                logger.warning("OUA enabled but no fold models available")
            return None

        # Cache to avoid recomputation
        if not hasattr(self, "_oracle_jackknife_cache"):
            self._oracle_jackknife_cache: Dict[str, np.ndarray] = {}

        if policy in self._oracle_jackknife_cache:
            return self._oracle_jackknife_cache[policy]

        try:
            pdata = self._policy_data[policy]
            n_folds = len(fold_models)
            jackknife_estimates = []

            # Check if we need covariates
            needs_covariates = False
            covariate_names: List[str] = []
            covariates_array: Optional[np.ndarray] = None
            if hasattr(self.reward_calibrator, "covariate_names"):
                covariate_names = self.reward_calibrator.covariate_names or []
                needs_covariates = len(covariate_names) > 0

            # Extract covariates from fresh draws if needed
            if needs_covariates:
                fresh_draws = self._fresh_draws[policy]
                covariates_list = []
                for sample in fresh_draws.samples:
                    sample_covariates = []
                    for cov_name in covariate_names:
                        if cov_name not in sample.metadata:
                            raise ValueError(
                                f"Covariate '{cov_name}' not found in fresh draw metadata "
                                f"for policy '{policy}' during OUA jackknife"
                            )
                        sample_covariates.append(sample.metadata[cov_name])
                    covariates_list.append(sample_covariates)
                covariates_array = np.array(covariates_list)

            # For each fold, recompute estimate with leave-one-out calibrator
            for fold_id in range(n_folds):
                fold_model = fold_models.get(fold_id)
                if fold_model is None:
                    logger.debug(f"No fold model for fold {fold_id}")
                    continue

                # Recalibrate rewards with LOO model
                # Note: fold_models are raw sklearn objects, not JudgeCalibrator wrappers
                # If covariates are needed, we need to use the FlexibleCalibrator's predict
                if needs_covariates:
                    # Use the calibrator's predict_oof method with fold_ids to get LOO predictions
                    # Create fold_ids array marking all samples as this fold (for LOO)
                    fold_ids = np.full(len(pdata.judge_scores), fold_id, dtype=int)
                    rewards_loo = np.clip(
                        self.reward_calibrator.predict_oof(
                            pdata.judge_scores, fold_ids, covariates_array
                        ),
                        0.0,
                        1.0,
                    )
                else:
                    # No covariates - use fold model directly
                    rewards_loo = np.clip(
                        fold_model.predict(pdata.judge_scores), 0.0, 1.0
                    )

                # Compute LOO estimate
                estimate_loo = float(np.mean(rewards_loo))
                jackknife_estimates.append(estimate_loo)

            if len(jackknife_estimates) < 2:
                logger.warning(
                    f"Not enough jackknife estimates for {policy}: "
                    f"{len(jackknife_estimates)}"
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
            logger.error(f"Failed to compute oracle jackknife for {policy}: {e}")
            return None

    def _store_df_info(self, result: EstimationResult) -> None:
        """Store degrees of freedom information for t-based CI computation.

        This method stores DF information that EstimationResult.confidence_interval()
        will use to automatically compute t-based CIs.

        The degrees of freedom is determined by the limiting factor:
        - If cluster-robust SE was used: df from clustering (typically n_clusters - 1)
        - If OUA jackknife was applied: min(df_cluster, K - 1) where K is number of oracle folds

        Args:
            result: EstimationResult with estimates and standard_errors already populated
                   (including OUA adjustment if applicable)

        Side effects:
            - Stores DF info in result.metadata["degrees_of_freedom"]
        """
        from scipy import stats

        if not hasattr(self, "_df_cluster"):
            # No DF tracking (shouldn't happen but be defensive)
            logger.debug("No DF tracking available, skipping DF storage")
            return

        df_info = {}

        for i, policy in enumerate(self.target_policies):
            if np.isnan(result.estimates[i]) or np.isnan(result.standard_errors[i]):
                continue

            # Get cluster DF
            df_cluster = self._df_cluster.get(policy, len(result.estimates) - 1)

            # If OUA was applied, get oracle DF and take minimum
            df_final = df_cluster
            if self.oua_jackknife and self.reward_calibrator is not None:
                try:
                    # Get number of oracle folds from calibrator
                    if hasattr(self.reward_calibrator, "get_fold_models_for_oua"):
                        fold_models = self.reward_calibrator.get_fold_models_for_oua()
                        if fold_models:
                            K = len(fold_models)
                            df_oracle = K - 1
                            df_final = min(df_cluster, df_oracle)
                            logger.debug(
                                f"Policy {policy}: df_cluster={df_cluster}, "
                                f"df_oracle={df_oracle}, df_final={df_final}"
                            )
                except Exception as e:
                    logger.debug(f"Could not get oracle DF for {policy}: {e}")

            # Ensure DF is at least 1
            df_final = max(df_final, 1)

            # Compute t-critical value for logging
            t_crit = stats.t.ppf(1 - 0.05 / 2, df_final)

            df_info[policy] = {
                "df": int(df_final),
                "t_critical": float(t_crit),
                "se_method": self._se_methods.get(policy, "standard"),
                "n_clusters": self._n_clusters.get(policy, len(result.estimates)),
            }

            logger.debug(
                f"Stored DF info for {policy}: df={df_final}, t_crit={t_crit:.3f}, "
                f"method={self._se_methods.get(policy, 'standard')}"
            )

        # Store in metadata
        if not isinstance(result.metadata, dict):
            result.metadata = {}
        result.metadata["degrees_of_freedom"] = df_info

    def _estimate_with_bootstrap(self, bootstrap_reason: str) -> EstimationResult:
        """Compute estimates using cluster bootstrap with calibrator refit.

        This method is used when bootstrap inference is preferred over
        analytic cluster-robust SEs. It captures:
        1. Prompt sampling variance
        2. Calibrator uncertainty
        3. Calibration/evaluation covariance (key term missing from OUA)

        Args:
            bootstrap_reason: Reason why bootstrap was selected (for metadata)

        Returns:
            EstimationResult with bootstrap-based confidence intervals
        """
        from ..diagnostics.robust_inference import (
            DirectEvalTable,
            build_direct_eval_table,
            cluster_bootstrap_direct_with_refit,
            make_calibrator_factory,
        )

        logger.info(f"Using cluster bootstrap inference ({bootstrap_reason})")

        # Build evaluation table from fresh draws
        eval_table = build_direct_eval_table(
            fresh_draws_per_policy=self._fresh_draws,
            covariate_names=(
                self.reward_calibrator.covariate_names
                if self.reward_calibrator
                and hasattr(self.reward_calibrator, "covariate_names")
                else None
            ),
        )

        # Get the calibration mode from the fitted calibrator
        # (Fixed to full-data selection, not "auto")
        from typing import Literal, cast

        if self.reward_calibrator is not None:
            mode_str = getattr(self.reward_calibrator, "selected_mode", None)
            if mode_str is None:
                # Fallback to calibration_mode if selected_mode not available
                mode_str = getattr(
                    self.reward_calibrator, "calibration_mode", "monotone"
                )
            # Never use "auto" in bootstrap - it should already be resolved
            if mode_str == "auto" or mode_str not in ("monotone", "two_stage"):
                mode_str = "monotone"
        else:
            mode_str = "monotone"

        selected_mode = cast(Literal["monotone", "two_stage"], mode_str)

        # Create calibrator factory with fixed mode
        calibrator_factory = make_calibrator_factory(
            mode=selected_mode,
            covariate_names=(
                self.reward_calibrator.covariate_names
                if self.reward_calibrator
                and hasattr(self.reward_calibrator, "covariate_names")
                else None
            ),
            seed=self.bootstrap_seed,
        )

        # Compute adaptive min_oracle_per_replicate based on available oracle data
        # This prevents bootstrap from failing at low oracle coverage (e.g., 5-10%)
        n_oracle_total = int(np.sum(eval_table.oracle_mask))
        # Use ~1/3 of total oracle as minimum, with floor of 10 and ceiling of 30
        min_oracle_per_replicate = max(10, min(30, n_oracle_total // 3))
        logger.info(
            f"Bootstrap: {n_oracle_total} oracle samples, min_per_replicate={min_oracle_per_replicate}"
        )

        # Compute calibration_policy_idx for transport experiments
        # This separates calibration oracle (single policy) from residual oracle (all policies)
        calibration_policy_idx = None
        if self.calibration_policy:
            try:
                calibration_policy_idx = eval_table.policy_names.index(
                    self.calibration_policy
                )
                logger.info(
                    f"Transport mode: calibrating on policy '{self.calibration_policy}' "
                    f"(idx={calibration_policy_idx})"
                )
            except ValueError:
                logger.warning(
                    f"calibration_policy '{self.calibration_policy}' not in policy_names "
                    f"{eval_table.policy_names}, using all oracle for calibration"
                )

        # Run bootstrap
        bootstrap_result = cluster_bootstrap_direct_with_refit(
            eval_table=eval_table,
            calibrator_factory=calibrator_factory,
            n_bootstrap=self.n_bootstrap,
            min_oracle_per_replicate=min_oracle_per_replicate,
            alpha=0.05,
            seed=self.bootstrap_seed,
            use_augmented_estimator=self.use_augmented_estimator,
            calibration_policy_idx=calibration_policy_idx,
            use_multipolicy_eif=self.use_multipolicy_eif,
        )

        # Cache result for pairwise comparisons
        self._bootstrap_result = bootstrap_result

        # ESTIMATOR CONSISTENCY: Use bootstrap's theta_hat as reported estimate.
        # The bootstrap refits calibrator on fresh draws oracle, so we must use its
        # point estimates for consistency. Using self._policy_data (logged-data calibrator)
        # would create an estimator mismatch.
        estimates = bootstrap_result["estimates"]

        # Guardrail: detect and log estimator mismatch for diagnostics
        estimates_logged_list: List[float] = []
        for policy in self.target_policies:
            if policy in self._policy_data:
                pdata = self._policy_data[policy]
                estimates_logged_list.append(float(np.mean(pdata.calibrated_rewards)))
            else:
                estimates_logged_list.append(float("nan"))
        estimates_logged_calibrator = np.array(estimates_logged_list)

        # Compute mismatch between the two estimators
        estimator_mismatch = estimates - estimates_logged_calibrator
        max_mismatch = np.nanmax(np.abs(estimator_mismatch))
        if max_mismatch > 0.01:
            logger.warning(
                f"Estimator mismatch detected: max|Δ| = {max_mismatch:.4f}. "
                f"This is expected when logged-data and fresh-draws calibrators differ. "
                f"Using bootstrap's theta_hat for consistency."
            )

        # Use bootstrap SEs (these correctly capture calibrator uncertainty)
        standard_errors = bootstrap_result["standard_errors"]

        # Build n_samples_used
        n_samples_used = {}
        for i, policy in enumerate(self.target_policies):
            if policy in self._fresh_draws:
                n_samples_used[policy] = len(self._fresh_draws[policy].samples)
            else:
                n_samples_used[policy] = 0

        # Build influence functions (for compatibility)
        influence_functions = {}
        for policy in self.target_policies:
            if policy in self._policy_data:
                pdata = self._policy_data[policy]
                policy_idx = self.target_policies.index(policy)
                if not np.isnan(estimates[policy_idx]):
                    influence_functions[policy] = (
                        pdata.calibrated_rewards - estimates[policy_idx]
                    )

        # Build diagnostics
        diagnostics = self._build_diagnostics(
            list(estimates), list(standard_errors), n_samples_used
        )

        # Build metadata with comprehensive bootstrap info
        coupled, overlap = self._calibration_overlaps_evaluation()
        metadata = {
            "mode": "direct",
            "estimand": "on-policy evaluation on provided prompts",
            "caveat": "Does not estimate counterfactual deployment value. Evaluates each policy on the evaluation set.",
            "target_policies": list(self.target_policies),
            "paired_comparison": self.paired_comparison,
            "inference": {
                "method": "cluster_bootstrap_refit",
                "n_bootstrap_requested": self.n_bootstrap,
                "n_bootstrap_valid": bootstrap_result["n_valid_replicates"],
                "n_attempts": bootstrap_result["n_attempts"],
                "skip_rate": bootstrap_result["skip_rate"],
                "seed": self.bootstrap_seed,
                "n_clusters": bootstrap_result["n_clusters"],
                "cluster_id_field": "prompt_id",
                "coupled": coupled,
                "coupling_overlap": overlap,
                "bootstrap_refit_mode": selected_mode,
                "min_oracle_per_replicate": bootstrap_result[
                    "min_oracle_per_replicate"
                ],
                "oracle_count_summary": bootstrap_result["oracle_count_summary"],
                "bootstrap_reason": bootstrap_reason,
            },
            "bootstrap_ci": {
                "lower": [float(x) for x in bootstrap_result["ci_lower"]],
                "upper": [float(x) for x in bootstrap_result["ci_upper"]],
                "method": "percentile",
                "alpha": 0.05,
            },
            "se_components": {
                "includes_oracle_uncertainty": True,  # Bootstrap captures this
                "includes_mc_variance": False,
                "via_bootstrap": True,
            },
            "estimator_consistency": {
                "theta_hat_source": "bootstrap_refit",  # Using bootstrap's estimate for consistency
                "theta_hat_logged_calibrator": [
                    float(x) for x in estimates_logged_calibrator
                ],
                "estimator_mismatch": [float(x) for x in estimator_mismatch],
                "max_mismatch": float(max_mismatch),
            },
        }

        # Check if prompts are aligned across policies
        if self.paired_comparison and len(self._policy_data) > 1:
            prompt_sets = [set(pd.prompt_ids) for pd in self._policy_data.values()]
            if all(ps == prompt_sets[0] for ps in prompt_sets):
                metadata["prompts_aligned"] = True
                metadata["n_prompts"] = len(prompt_sets[0])
            else:
                metadata["prompts_aligned"] = False

        result = EstimationResult(
            estimates=np.array(estimates),
            standard_errors=np.array(standard_errors),
            n_samples_used=n_samples_used,
            method="calibrated_direct_bootstrap",
            influence_functions=influence_functions,
            diagnostics=diagnostics,
            metadata=metadata,
        )

        # Log summary with SE-based CIs (not bootstrap percentile CIs)
        from scipy import stats

        z_crit = stats.norm.ppf(0.975)  # 1.96 for 95% CI
        for i, policy in enumerate(self.target_policies):
            if not np.isnan(estimates[i]):
                ci_lower = estimates[i] - z_crit * standard_errors[i]
                ci_upper = estimates[i] + z_crit * standard_errors[i]
                logger.info(
                    f"Bootstrap estimate for '{policy}': {estimates[i]:.4f} ± {standard_errors[i]:.4f} "
                    f"(95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])"
                )

        return result
