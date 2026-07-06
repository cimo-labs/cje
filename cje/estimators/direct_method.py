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

from ..data.models import CIInfo, EstimationResult
from ..diagnostics.models import DirectDiagnostics, Status
from ..diagnostics.gates import BOUNDARY_CARD_STATUS_TO_STATUS, worst_status
from ..diagnostics.robust_inference import (
    combine_cluster_and_oracle,
    oracle_jackknife_estimates,
    oracle_jackknife_variance,
)

logger = logging.getLogger(__name__)


@dataclass
class PolicyData:
    """Data for a single policy in direct mode."""

    policy: str
    judge_scores: np.ndarray
    calibrated_rewards: np.ndarray
    prompt_ids: List[str]


class CalibratedDirectEstimator:
    """Calibrated direct method for on-policy evaluation.

    Estimates V(πⱼ) = E_πⱼ[f*(S)] by averaging calibrated rewards over
    fresh draws from each policy.

    This is NOT off-policy evaluation - it evaluates each policy on the
    prompts you provided, without accounting for production context distribution
    or using importance weights.

    Supported inference methods are:
    - "bootstrap"
    - "cluster_robust"
    - "auto"

    Args:
        target_policies: List of policy names to evaluate
        reward_calibrator: Optional calibrator to map judge scores to rewards.
            If None, uses raw judge scores (uncalibrated "naive" mode).
        paired_comparison: If True, use within-prompt differences when possible
        oua_jackknife: Whether to include calibration uncertainty via the oracle jackknife
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

    _VALID_INFERENCE_METHODS = {"bootstrap", "cluster_robust", "auto"}

    def __init__(
        self,
        target_policies: List[str],
        reward_calibrator: Optional[Any] = None,
        paired_comparison: bool = True,
        oua_jackknife: bool = True,
        inference_method: str = "bootstrap",
        n_bootstrap: int = 2000,
        bootstrap_seed: int = 42,
        use_augmented_estimator: bool = True,
    ):
        self.target_policies = list(target_policies)
        self.reward_calibrator = reward_calibrator
        self.oua_jackknife = oua_jackknife
        self.paired_comparison = paired_comparison
        self._fitted = False
        self._results: Optional[EstimationResult] = None

        normalized_inference = inference_method.strip().lower()
        if normalized_inference not in self._VALID_INFERENCE_METHODS:
            allowed = ", ".join(sorted(self._VALID_INFERENCE_METHODS))
            raise ValueError(
                f"Invalid inference_method '{inference_method}'. "
                f"Expected one of: {allowed}. "
                "If you want oracle jackknife augmentation, set "
                "oua_jackknife=True with a valid inference_method."
            )

        # Auto-detect: when reward_calibrator=None, bootstrap would create a new
        # calibrator internally, defeating "naive" (uncalibrated) mode.
        # Default to cluster_robust in this case.
        if reward_calibrator is None and normalized_inference == "bootstrap":
            logger.info(
                "reward_calibrator=None with inference_method='bootstrap' would create "
                "a calibrator during bootstrap. Defaulting to 'cluster_robust' for "
                "uncalibrated estimation. Pass inference_method='cluster_robust' explicitly "
                "to silence this message."
            )
            normalized_inference = "cluster_robust"

        self.inference_method = normalized_inference
        self.n_bootstrap = n_bootstrap
        self.bootstrap_seed = bootstrap_seed
        self.use_augmented_estimator = use_augmented_estimator
        self._policy_data: Dict[str, PolicyData] = {}
        self._fresh_draws: Dict[str, Any] = {}  # Storage for fresh draws

    @property
    def is_fitted(self) -> bool:
        """Check if estimator has been fitted."""
        return self._fitted

    def _validate_fitted(self) -> None:
        """Ensure estimator is fitted before making predictions."""
        if not self._fitted:
            raise RuntimeError("Estimator must be fitted before calling estimate()")

    def fit_and_estimate(self) -> EstimationResult:
        """Convenience method to fit and estimate in one call."""
        self.fit()
        return self.estimate()

    def get_diagnostics(self) -> Optional[Any]:
        """Get the diagnostics from the last estimation.

        Returns:
            Diagnostics if estimate() has been called, None otherwise
        """
        if self._results and self._results.diagnostics:
            return self._results.diagnostics
        return None

    @property
    def _method_name(self) -> str:
        """Method label reflecting whether rewards are actually calibrated.

        Without a reward calibrator the estimator averages raw judge scores,
        so labeling the result 'calibrated_direct' would be misleading.
        """
        return (
            "calibrated_direct"
            if self.reward_calibrator is not None
            else "naive_direct"
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
        # Oracle labels come from fresh draws (coupled when any overlap)
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

        # Should be unreachable due to validation in __init__
        raise ValueError(
            f"Invalid inference_method '{self.inference_method}'. "
            f"Expected one of: {', '.join(sorted(self._VALID_INFERENCE_METHODS))}"
        )

    def _eval_oracle_count(self) -> int:
        """Number of oracle-labeled samples in the evaluation fresh draws."""
        return sum(
            1
            for fd in self._fresh_draws.values()
            for sample in fd.samples
            if sample.oracle_label is not None
        )

    def _oracle_coverage_for_oua(self) -> Optional[float]:
        """Oracle coverage consulted by the 100%-coverage OUA skip.

        The skip exists because rewards on labeled eval rows are the
        labels themselves, so the IF variance already carries the reward
        noise. It therefore applies only when the oracle labels live in
        THIS evaluation data: with a separate calibration source and
        label-free fresh draws, the calibrator's coverage describes the
        calibration dataset, not the eval set — eval rewards are f̂(S_eval)
        and the calibrator's finite-sample uncertainty must still be added
        via the jackknife, so report no coverage and keep the OUA active.
        """
        coverage = getattr(self.reward_calibrator, "oracle_coverage", None)
        if coverage is None:
            return None
        coverage_value = float(coverage)
        if coverage_value >= 1.0 and self._eval_oracle_count() == 0:
            return None
        return coverage_value

    def _bootstrap_fallback_reason(self) -> Optional[str]:
        """Why bootstrap cannot run on this data (None when it can).

        The bootstrap path refits the calibrator on the EVALUATION data's
        oracle labels. With a separate calibration source and NO oracle
        labels in the fresh draws, its eval table has zero oracle rows:
        the full-data refit fails and every replicate is skipped below
        min_oracle_per_replicate, so NaN estimates came back quietly
        (pre-0.4.0 behavior).

        Falling back to cluster-robust + oracle-jackknife is statistically
        sound here: with a separate calibration source and zero eval-oracle
        rows, calibration and evaluation are independent samples, so the
        calibration/evaluation covariance term the bootstrap exists to
        capture is exactly zero. The additive decomposition — cluster-robust
        evaluation variance plus the oracle jackknife (which carries the
        calibrator's oracle uncertainty via its cross-fitted fold models) —
        is then the correct SE.
        """
        if self._eval_oracle_count() > 0:
            return None

        if self.reward_calibrator is not None:
            logger.warning(
                "bootstrap requires oracle labels in the evaluation data; "
                "falling back to cluster-robust SEs with oracle-jackknife "
                "(the fresh draws carry no oracle labels, so the bootstrap "
                "refit would skip every replicate and return NaN estimates; "
                "the fitted calibrator's oracle uncertainty is included via "
                "the jackknife)"
            )
            return "no_oracle_labels_in_evaluation_data"

        # No calibrator either (naive mode reached via inference "auto"):
        # there is nothing to refit, so bootstrap is equally impossible.
        logger.warning(
            "bootstrap requires oracle labels in the evaluation data; "
            "falling back to cluster-robust SEs (no oracle labels and no "
            "fitted calibrator — results are uncalibrated judge-score means)"
        )
        return "no_oracle_labels_and_no_calibrator"

    def _covariate_matrix(self, policy: str) -> Optional[np.ndarray]:
        """Extract the calibrator's covariates from a policy's fresh draws.

        Returns None when the calibrator needs no covariates. Raises an
        actionable error when a needed covariate is missing from a sample's
        metadata — CJE never NaN-fills missing covariates.
        """
        covariate_names: List[str] = []
        if self.reward_calibrator is not None and hasattr(
            self.reward_calibrator, "covariate_names"
        ):
            covariate_names = self.reward_calibrator.covariate_names or []
        if not covariate_names:
            return None

        fresh_draws = self._fresh_draws[policy]
        rows: List[List[float]] = []
        for sample in fresh_draws.samples:
            row = []
            for cov_name in covariate_names:
                if cov_name not in sample.metadata:
                    raise ValueError(
                        f"Covariate '{cov_name}' not found in fresh draw metadata "
                        f"for policy '{policy}', sample {sample.prompt_id}. "
                        f"Available metadata: {list(sample.metadata.keys())}"
                    )
                row.append(sample.metadata[cov_name])
            rows.append(row)
        return np.array(rows)

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
                f"Provide fresh_draws_dir or fresh_draws_data."
            )

        # Get data for each policy from fresh draws
        for policy in self.target_policies:
            fresh_draws = self._fresh_draws[policy]

            judge_scores = np.array(
                [sample.judge_score for sample in fresh_draws.samples], dtype=float
            )
            prompt_ids = [sample.prompt_id for sample in fresh_draws.samples]
            covariates = self._covariate_matrix(policy)

            # Calibrate judge scores to rewards (one vectorized call per
            # policy); without a calibrator use judge scores directly.
            if self.reward_calibrator is not None:
                rewards = np.clip(
                    self.reward_calibrator.predict(judge_scores, covariates=covariates),
                    0.0,
                    1.0,
                )
            else:
                rewards = judge_scores.copy()

            self._policy_data[policy] = PolicyData(
                policy=policy,
                judge_scores=judge_scores,
                calibrated_rewards=np.asarray(rewards, dtype=float),
                prompt_ids=prompt_ids,
            )

            logger.info(
                f"Loaded fresh draws for policy '{policy}': {len(rewards)} samples"
            )

        self._fitted = True
        logger.info(
            f"Prepared data for {len(self._policy_data)} policies from fresh draws"
        )

    def _assemble_result(
        self,
        estimates: List[float],
        standard_errors: List[float],
        n_samples_used: Dict[str, int],
        influence_functions: Dict[str, np.ndarray],
        method: str,
        extra_metadata: Dict[str, Any],
    ) -> EstimationResult:
        """Shared result assembly for the analytic and bootstrap paths.

        Builds diagnostics, the common Direct-mode metadata (including
        se_methods / n_clusters on BOTH paths), the prompts_aligned check,
        and the reliability gates, then constructs the EstimationResult.
        """
        diagnostics = self._build_diagnostics(
            list(estimates), list(standard_errors), n_samples_used
        )

        metadata: Dict[str, Any] = {
            "mode": "direct",
            "estimand": "on-policy evaluation on provided prompts",
            "caveat": "Does not estimate counterfactual deployment value. Evaluates each policy on the evaluation set.",
            "target_policies": list(self.target_policies),
            "paired_comparison": self.paired_comparison,
            "se_methods": getattr(self, "_se_methods", {}),
            "n_clusters": getattr(self, "_n_clusters", {}),
        }
        metadata.update(extra_metadata)

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

        # Boundary gate: coverage badges + reliability gates for the CLI
        self._attach_reliability_metadata(metadata, diagnostics)

        # Typed CI record (metadata mirrors stay the serialized source of
        # truth). Bootstrap results carry their precomputed percentile CIs
        # here; the analytic path fills ci_info in _store_df_info once the
        # OUA-adjusted degrees of freedom are known.
        ci_info: Optional[CIInfo] = None
        boot_ci = extra_metadata.get("bootstrap_ci")
        if isinstance(boot_ci, dict) and boot_ci.get("method") == "percentile":
            ci_info = CIInfo(
                method="percentile",
                alpha=float(boot_ci.get("alpha", 0.05)),
                lower=[float(v) for v in boot_ci["lower"]],
                upper=[float(v) for v in boot_ci["upper"]],
                df_per_policy=None,
            )

        result = EstimationResult(
            estimates=np.array(estimates),
            standard_errors=np.array(standard_errors),
            n_samples_used=n_samples_used,
            method=method,
            influence_functions=influence_functions,
            diagnostics=diagnostics,
            metadata=metadata,
            ci_info=ci_info,
        )
        self._results = result
        return result

    def estimate(self) -> EstimationResult:
        """Compute calibrated direct estimates for all policies.

        Returns:
            EstimationResult with:
                - estimates: Mean calibrated reward for each policy
                - standard_errors: Including calibration uncertainty via the oracle jackknife (or bootstrap)
                - diagnostics: Simplified (no weight metrics)
                - metadata: Mode info and caveats
        """
        self._validate_fitted()

        # Check if bootstrap should be used — and whether it CAN run on
        # this data (zero eval-oracle rows would skip every replicate and
        # quietly return NaN estimates; see _bootstrap_fallback_reason).
        use_bootstrap, bootstrap_reason = self._should_use_bootstrap()
        bootstrap_fallback: Optional[str] = None
        if use_bootstrap:
            bootstrap_fallback = self._bootstrap_fallback_reason()
            if bootstrap_fallback is None:
                return self._estimate_with_bootstrap(bootstrap_reason)

        # Standard estimation path (cluster-robust SE + oracle jackknife)
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

        extra_metadata: Dict[str, Any] = {
            "se_components": {
                "includes_oracle_uncertainty": False,  # Will be set to True by _apply_oua_jackknife()
                "includes_mc_variance": False,
            },
        }

        # Record a bootstrap-to-cluster-robust downgrade so the SE basis
        # is visible in results, not just in a log line
        if bootstrap_fallback is not None:
            extra_metadata["inference"] = {
                "method": "cluster_robust",
                "requested_method": self.inference_method,
                "fallback_reason": bootstrap_fallback,
            }

        result = self._assemble_result(
            estimates=estimates,
            standard_errors=standard_errors,
            n_samples_used=n_samples_used,
            influence_functions=influence_functions,
            method=self._method_name,
            extra_metadata=extra_metadata,
        )

        # Apply oracle-jackknife inference (adds calibration uncertainty)
        self._apply_oua_jackknife(result)

        # Store DF info for t-based CIs (computed automatically by EstimationResult.confidence_interval())
        self._store_df_info(result)

        return result

    def _apply_oua_jackknife(self, result: EstimationResult) -> None:
        """Apply the oracle jackknife for calibration-aware inference.

        This method adds oracle uncertainty to standard_errors in-place, accounting
        for finite-sample uncertainty in the learned reward calibrator f̂(S).

        Args:
            result: EstimationResult with standard_errors to augment
        """
        if not (self.oua_jackknife and self.reward_calibrator is not None):
            return

        # Skip oracle-jackknife augmentation at 100% oracle coverage
        # (the calibrator knows its own coverage; at 100% there is no
        # calibration uncertainty to add). _oracle_coverage_for_oua returns
        # None when the calibrator's coverage describes a different dataset
        # than the evaluation data.
        try:
            coverage = self._oracle_coverage_for_oua()

            if coverage is not None and coverage >= 1.0:
                if isinstance(result.metadata, dict):
                    result.metadata.setdefault("se_components", {})
                    result.metadata["se_components"][
                        "oracle_uncertainty_skipped"
                    ] = "100% oracle coverage"
                return
        except Exception:
            pass  # Continue with the default path if we can't check coverage

        # Check if oracle variance is already included (e.g., by the
        # bootstrap inference path, which refits the calibrator per replicate)
        if isinstance(result.metadata, dict) and result.metadata.get(
            "se_components", {}
        ).get("includes_oracle_uncertainty"):
            # Oracle variance already included in standard_errors
            return

        try:
            var_oracle_map: Dict[str, float] = {}
            jk_counts: Dict[str, int] = {}

            for i, policy in enumerate(self.target_policies):
                var_orc = 0.0
                K = 0
                jack = self.get_oracle_jackknife(policy)
                if (
                    jack is not None
                    and len(jack) >= 2
                    and i < len(result.standard_errors)
                ):
                    K = len(jack)
                    var_orc = oracle_jackknife_variance(jack)

                var_oracle_map[policy] = var_orc
                jk_counts[policy] = K

                # Update standard_errors in place (add oracle variance)
                if i < len(result.standard_errors):
                    se_base = float(result.standard_errors[i])
                    df_cluster = getattr(self, "_df_cluster", {}).get(policy, 1)
                    se_total, _ = combine_cluster_and_oracle(
                        se_base, df_cluster, var_orc, K
                    )
                    result.standard_errors[i] = se_total

            # Record that oracle uncertainty has been added
            if isinstance(result.metadata, dict):
                result.metadata.setdefault("se_components", {})
                result.metadata["se_components"].update(
                    {
                        "includes_oracle_uncertainty": True,
                        "oracle_variance_per_policy": var_oracle_map,
                        "oracle_jackknife_counts": jk_counts,
                    }
                )
        except Exception as e:
            logger.debug(f"Calibration-aware oracle jackknife failed: {e}")

    def _compute_policy_boundary_card(self, policy: str) -> Optional[Dict[str, Any]]:
        """Coverage badge (paper REFUSE-LEVEL gate) for one policy.

        Compares the policy's fresh-draw judge scores against the oracle
        calibration S-range the reward calibrator stored at fit time;
        >= 5% out-of-range mass refuses level claims (threshold canonical
        in gates.py). The shared helper emits the REFUSE-LEVEL warning.
        """
        if self.reward_calibrator is None:
            return None
        pdata = self._policy_data.get(policy)
        if pdata is None:
            return None

        try:
            from ..diagnostics.reward_boundary import boundary_card_dict

            return boundary_card_dict(
                self.reward_calibrator,
                S_policy=pdata.judge_scores,
                R_policy=pdata.calibrated_rewards,
                warn_label=policy,
            )
        except Exception as e:
            logger.debug(f"Could not compute boundary card for policy '{policy}': {e}")
            return None

    def _build_diagnostics(
        self,
        estimates: List[float],
        standard_errors: List[float],
        n_samples_used: Dict[str, int],
    ) -> DirectDiagnostics:
        """Build Direct-mode diagnostics.

        No weight metrics (ESS, tail indices) since we don't use weights.
        The identification risk that matters here is coverage: each
        policy's boundary card (the paper's coverage badge) is computed
        against the calibrator's oracle S-range, and a REFUSE-LEVEL badge
        sets that policy's status to CRITICAL (the shared boundary helper
        emits the loud warning).
        """
        policies = list(self.target_policies)

        # Build estimate dicts
        estimates_dict = {
            p: float(e) for p, e in zip(policies, estimates) if not np.isnan(e)
        }
        se_dict = {
            p: float(se) for p, se in zip(policies, standard_errors) if not np.isnan(se)
        }

        # Get calibration info (if calibrator was provided).
        # JudgeCalibrator implements get_calibration_info() since 0.4.0;
        # before that this guard never fired and the calibration fields
        # below were always None.
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

        # Coverage badges (paper REFUSE-LEVEL gate): judge-score mass
        # outside the oracle calibration range refuses level claims
        # (CRITICAL); a CAUTION card (boundary saturation) yields WARNING.
        # The card-status -> Status ladder is canonical in gates.py.
        boundary_cards: Dict[str, Dict[str, Any]] = {}
        status_per_policy: Dict[str, Status] = {p: Status.GOOD for p in policies}
        for policy in policies:
            card = self._compute_policy_boundary_card(policy)
            if card is None:
                continue
            boundary_cards[policy] = card
            card_status = BOUNDARY_CARD_STATUS_TO_STATUS.get(
                str(card.get("status")), Status.GOOD
            )
            status_per_policy[policy] = worst_status(
                status_per_policy[policy], card_status
            )

        diagnostics = DirectDiagnostics(
            estimator_type="Direct",
            method=self._method_name,
            n_samples_total=total_samples,
            n_samples_valid=valid_samples,
            policies=policies,
            estimates=estimates_dict,
            standard_errors=se_dict,
            n_samples_used=n_samples_used,
            status_per_policy=status_per_policy,
            boundary_cards=boundary_cards if boundary_cards else None,
            # Calibration metrics (fit-time, if a calibrator was provided)
            calibration_rmse=cal_info.get("rmse"),
            calibration_coverage=cal_info.get("coverage_at_01"),
            n_oracle_labels=cal_info.get("n_oracle_labels"),
        )

        return diagnostics

    def _attach_reliability_metadata(
        self, metadata: Dict[str, Any], diagnostics: DirectDiagnostics
    ) -> None:
        """Record boundary cards and reliability gates in result metadata.

        The CLI's best-policy announcement reads
        ``metadata["reliability_gates"][policy]["flagged"]``: a REFUSE-LEVEL
        coverage badge demotes the policy from the trophy line.
        """
        if not diagnostics.boundary_cards:
            return
        metadata["boundary_cards"] = diagnostics.boundary_cards
        gates: Dict[str, Dict[str, Any]] = {}
        for policy, card in diagnostics.boundary_cards.items():
            refuse_level = card.get("status") == "REFUSE-LEVEL"
            reasons = []
            if refuse_level:
                reasons.append(
                    f"boundary: {card.get('out_of_range', 0.0):.1%} of judge "
                    f"scores outside the oracle calibration range"
                )
            gates[policy] = {
                "flagged": refuse_level,
                "refused": False,  # estimates are still reported
                "refuse_level_claims": refuse_level,
                "reasons": reasons,
            }
        metadata["reliability_gates"] = gates

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
            logger.debug("No reward_calibrator for oracle-jackknife inference")
            return None

        if policy not in self._policy_data:
            logger.warning(f"No data for policy {policy}")
            return None

        # Use unified interface to get fold models
        if not hasattr(self.reward_calibrator, "get_fold_models_for_oua"):
            if self.oua_jackknife:
                raise ValueError(
                    "Calibration-aware oracle jackknife is enabled but the calibrator doesn't support it. "
                    "Ensure calibrate_dataset() uses enable_cross_fit=True."
                )
            return None

        fold_models = self.reward_calibrator.get_fold_models_for_oua()
        if not fold_models:
            if self.oua_jackknife:
                logger.warning(
                    "Calibration-aware oracle jackknife is enabled but no fold models are available"
                )
            return None

        # Cache to avoid recomputation
        if not hasattr(self, "_oracle_jackknife_cache"):
            self._oracle_jackknife_cache: Dict[str, np.ndarray] = {}

        if policy in self._oracle_jackknife_cache:
            return self._oracle_jackknife_cache[policy]

        try:
            pdata = self._policy_data[policy]
            covariates_array = self._covariate_matrix(policy)

            jackknife_array = oracle_jackknife_estimates(
                self.reward_calibrator, pdata.judge_scores, covariates_array
            )
            if jackknife_array is None:
                logger.warning(
                    f"Not enough jackknife estimates for {policy} "
                    f"(need at least 2 fold models)"
                )
                return None

            self._oracle_jackknife_cache[policy] = jackknife_array

            logger.debug(
                f"Oracle jackknife for {policy}: {len(jackknife_array)} estimates, "
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
        - If oracle-jackknife inference was applied: min(df_cluster, K - 1)

        Args:
            result: EstimationResult with estimates and standard_errors already populated
                   (including oracle-jackknife adjustment if applicable)

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

            # If oracle-jackknife inference was applied, cap DF by the
            # oracle fold count (shared combining rule)
            K = 0
            if self.oua_jackknife and self.reward_calibrator is not None:
                try:
                    if hasattr(self.reward_calibrator, "get_fold_models_for_oua"):
                        fold_models = self.reward_calibrator.get_fold_models_for_oua()
                        if fold_models:
                            K = len(fold_models)
                            logger.debug(
                                f"Policy {policy}: df_cluster={df_cluster}, "
                                f"df_oracle={K - 1}"
                            )
                except Exception as e:
                    logger.debug(f"Could not get oracle DF for {policy}: {e}")

            _, df_final = combine_cluster_and_oracle(0.0, df_cluster, 0.0, K)

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

        # Store in metadata (serialized mirror) and as the typed CI record
        if not isinstance(result.metadata, dict):
            result.metadata = {}
        result.metadata["degrees_of_freedom"] = df_info
        result.ci_info = CIInfo(
            method="t",
            alpha=0.05,
            lower=None,
            upper=None,
            df_per_policy=df_info,
        )

    def _estimate_with_bootstrap(self, bootstrap_reason: str) -> EstimationResult:
        """Compute estimates using cluster bootstrap with calibrator refit.

        This method is used when bootstrap inference is preferred over
        analytic cluster-robust SEs. It captures:
        1. Prompt sampling variance
        2. Calibrator uncertainty
        3. Calibration/evaluation covariance (the key term missing from oracle-jackknife-only inference)

        Args:
            bootstrap_reason: Reason why bootstrap was selected (for metadata)

        Returns:
            EstimationResult with bootstrap-based confidence intervals
        """
        from ..diagnostics.robust_inference import (
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

        # Run bootstrap. (Transport experiments use
        # cluster_bootstrap_direct_with_refit's calibration_policy_idx
        # directly; the estimator always calibrates on all oracle rows.)
        bootstrap_result = cluster_bootstrap_direct_with_refit(
            eval_table=eval_table,
            calibrator_factory=calibrator_factory,
            n_bootstrap=self.n_bootstrap,
            min_oracle_per_replicate=min_oracle_per_replicate,
            alpha=0.05,
            seed=self.bootstrap_seed,
            use_augmented_estimator=self.use_augmented_estimator,
        )

        # ESTIMATOR CONSISTENCY: Use bootstrap's theta_hat as reported estimate.
        # The bootstrap refits calibrator on fresh draws oracle, so we must use its
        # point estimates for consistency. Using self._policy_data (logged-data calibrator)
        # would create an estimator mismatch (expected with small oracle samples).
        estimates = bootstrap_result["estimates"]

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

        # Bootstrap-specific metadata
        coupled, overlap = self._calibration_overlaps_evaluation()
        extra_metadata: Dict[str, Any] = {
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
                # Variance decomposition for budget planning (from dual-estimate bootstrap)
                "oracle_variance_per_policy": bootstrap_result.get(
                    "variance_decomposition", {}
                ).get("var_cal_per_policy", {}),
            },
        }

        result = self._assemble_result(
            estimates=list(estimates),
            standard_errors=list(standard_errors),
            n_samples_used=n_samples_used,
            influence_functions=influence_functions,
            method=f"{self._method_name}_bootstrap",
            extra_metadata=extra_metadata,
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
