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
        reward_calibrator: Optional calibrator to map judge scores to rewards
        paired_comparison: If True, use within-prompt differences when possible
        run_diagnostics: Whether to compute diagnostics

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
        self._policy_data: Dict[str, PolicyData] = {}
        self._fresh_draws: Dict[str, Any] = {}  # Storage for fresh draws

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
                - standard_errors: Including oracle uncertainty via OUA
                - diagnostics: Simplified (no weight metrics)
                - metadata: Mode info and caveats
        """
        self._validate_fitted()

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
