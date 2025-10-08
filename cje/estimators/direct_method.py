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

            for sample in fresh_draws.samples:
                # FreshDrawSample has judge_score as a direct field
                judge_score = sample.judge_score
                judge_scores.append(judge_score)
                prompt_ids.append(sample.prompt_id)

                # Calibrate judge score to reward if calibrator available
                if self.reward_calibrator is not None:
                    reward = float(
                        np.clip(
                            self.reward_calibrator.predict(np.array([judge_score]))[0],
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

            # Standard error from influence function (base SE)
            # Oracle uncertainty will be added by _apply_oua_jackknife() later
            n = len(pdata.calibrated_rewards)
            se = float(np.std(if_values, ddof=1) / np.sqrt(n))

            estimates.append(estimate)
            standard_errors.append(se)
            n_samples_used[policy] = n

            logger.info(
                f"Direct estimate for '{policy}': {estimate:.4f} ± {se:.4f} " f"(n={n})"
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

            # For each fold, recompute estimate with leave-one-out calibrator
            for fold_id in range(n_folds):
                fold_model = fold_models.get(fold_id)
                if fold_model is None:
                    logger.debug(f"No fold model for fold {fold_id}")
                    continue

                # Recalibrate rewards with LOO model
                rewards_loo = np.clip(fold_model.predict(pdata.judge_scores), 0.0, 1.0)

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
