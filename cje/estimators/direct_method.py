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
    CalibrationProvenance,
    DirectEvalTable,
    DirectPointEstimate,
    LabelDesign,
    build_direct_eval_table,
    combine_cluster_and_oracle,
    compute_direct_point_estimate,
    direct_oracle_jackknife_estimates,
    oracle_jackknife_variance,
    residual_predictions_for_evaluation,
    validate_calibration_provenance,
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
        inference_method: Optional[str] = None,
        n_bootstrap: int = 2000,
        bootstrap_seed: int = 42,
        use_augmented_estimator: bool = True,
        calibration_provenance: Optional[CalibrationProvenance] = None,
        label_design: Optional[LabelDesign] = None,
    ):
        self.target_policies = list(target_policies)
        self.reward_calibrator = reward_calibrator
        self.oua_jackknife = oua_jackknife
        self.paired_comparison = paired_comparison
        self._fitted = False
        self._results: Optional[EstimationResult] = None

        normalized_inference = (
            inference_method.strip().lower()
            if inference_method is not None
            else ("bootstrap" if reward_calibrator is not None else "cluster_robust")
        )
        if normalized_inference not in self._VALID_INFERENCE_METHODS:
            allowed = ", ".join(sorted(self._VALID_INFERENCE_METHODS))
            raise ValueError(
                f"Invalid inference_method '{inference_method}'. "
                f"Expected one of: {allowed}. "
                "If you want oracle jackknife augmentation, set "
                "oua_jackknife=True with a valid inference_method."
            )

        self.inference_method = normalized_inference
        self.n_bootstrap = n_bootstrap
        self.bootstrap_seed = bootstrap_seed
        self.use_augmented_estimator = use_augmented_estimator
        if calibration_provenance is not None and reward_calibrator is None:
            raise ValueError(
                "calibration_provenance requires a fitted reward_calibrator"
            )
        self._provenance_explicit = calibration_provenance is not None
        self._provenance_linkage_validated = self._provenance_explicit
        self.calibration_provenance = calibration_provenance
        if self.calibration_provenance is None and reward_calibrator is not None:
            try:
                self.calibration_provenance = (
                    CalibrationProvenance.from_fitted_calibrator(reward_calibrator)
                )
            except ValueError:
                self.calibration_provenance = None
        if (
            self.calibration_provenance is not None
            and self._provenance_explicit
            and reward_calibrator is not None
        ):
            validate_calibration_provenance(
                self.calibration_provenance, reward_calibrator
            )
        self.label_design = label_design or LabelDesign()
        self._policy_data: Dict[str, PolicyData] = {}
        self._fresh_draws: Dict[str, Any] = {}  # Storage for fresh draws
        self._eval_table: Optional[Any] = None
        self._last_point: Optional[DirectPointEstimate] = None

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
        routes = self._point_routes()
        if routes and all(route == "direct_oracle" for route in routes):
            return "direct_oracle"
        if self.reward_calibrator is not None:
            return "calibrated_direct"
        if any(route == "direct_oracle" for route in routes):
            return "mixed_direct"
        return "naive_direct"

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

    def _ordered_fresh_draws(self) -> Dict[str, Any]:
        """Return evaluation draws in the public target-policy order."""
        return {policy: self._fresh_draws[policy] for policy in self.target_policies}

    def _build_eval_table(
        self, *, couple_policy_prompts: bool = True
    ) -> DirectEvalTable:
        """Build a row table in target order, optionally decoupling policies."""
        covariate_names = (
            self.reward_calibrator.covariate_names
            if self.reward_calibrator is not None
            and hasattr(self.reward_calibrator, "covariate_names")
            else None
        )
        table = build_direct_eval_table(
            fresh_draws_per_policy=self._ordered_fresh_draws(),
            covariate_names=covariate_names,
        )
        if couple_policy_prompts or len(self.target_policies) < 2:
            return table

        cluster_by_policy_prompt: Dict[Tuple[int, str], int] = {}
        independent_cluster_ids: List[int] = []
        for policy_index, prompt_id in zip(
            table.policy_indices, table.prompt_id_strings
        ):
            key = (int(policy_index), str(prompt_id))
            if key not in cluster_by_policy_prompt:
                cluster_by_policy_prompt[key] = len(cluster_by_policy_prompt)
            independent_cluster_ids.append(cluster_by_policy_prompt[key])

        return DirectEvalTable(
            prompt_ids=np.asarray(independent_cluster_ids, dtype=np.int64),
            prompt_id_strings=list(table.prompt_id_strings),
            policy_indices=table.policy_indices,
            judge_scores=table.judge_scores,
            oracle_labels=table.oracle_labels,
            oracle_mask=table.oracle_mask,
            covariates=table.covariates,
            covariate_names=table.covariate_names,
            row_keys=list(table.row_keys) if table.row_keys is not None else None,
            row_keys_synthesized=table.row_keys_synthesized,
            policy_names=list(table.policy_names),
        )

    def _point_routes(self) -> List[str]:
        if self._last_point is None:
            return []
        return [str(route) for route in self._last_point.diagnostics.get("routes", [])]

    def _route_for_policy(self, policy: str) -> Optional[str]:
        try:
            index = self.target_policies.index(policy)
        except ValueError:
            return None
        routes = self._point_routes()
        return routes[index] if index < len(routes) else None

    def _all_policies_have_complete_oracle_coverage(self) -> bool:
        return bool(self.target_policies) and all(
            bool(self._fresh_draws[policy].samples)
            and all(
                sample.oracle_label is not None
                for sample in self._fresh_draws[policy].samples
            )
            for policy in self.target_policies
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
        # Explicit provenance is authoritative.  Legacy direct constructor
        # calls lack row roles, so retain the historical heuristic only for
        # their metadata/auto-routing.
        cal_clusters: set = set()
        eval_clusters: set = set()

        for fd in self._fresh_draws.values():
            for sample in fd.samples:
                eval_clusters.add(sample.prompt_id)
                if not self._provenance_explicit and sample.oracle_label is not None:
                    cal_clusters.add(sample.prompt_id)

        if self._provenance_explicit and self.calibration_provenance is not None:
            for role, prompt_id in zip(
                self.calibration_provenance.row_roles or [],
                self.calibration_provenance.prompt_ids,
            ):
                if role == "evaluation":
                    cal_clusters.add(prompt_id)

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
        n_total = sum(len(fd.samples) for fd in self._fresh_draws.values())
        if n_total == 0:
            return None
        n_labeled = self._eval_oracle_count()
        if n_labeled == 0:
            return None
        return n_labeled / n_total

    def _bootstrap_fallback_reason(self) -> Optional[str]:
        """Why bootstrap cannot run on this data (None when it can).

        Refit bootstrap requires the exact calibration rows. Explicit
        provenance is preferred; JudgeCalibrator retains compact fit rows for
        backwards-compatible direct construction. If neither is available,
        return the strongest valid analytic artifact rather than inventing a
        bootstrap calibration sample.
        """
        if (
            self.reward_calibrator is not None
            and self.calibration_provenance is not None
        ):
            return None

        if self.reward_calibrator is not None:
            logger.warning(
                "bootstrap requires the exact rows used to fit the calibrator; "
                "falling back to cluster-robust SEs with oracle-jackknife "
                "because calibration provenance is unavailable"
            )
            return "calibration_provenance_unavailable"

        if self._all_policies_have_complete_oracle_coverage():
            # The bootstrap point functional routes every policy directly to
            # observed oracle outcomes, so no calibrator or fit provenance is
            # needed.
            return None

        # No calibrator and at least one policy needs a judge-score plug-in.
        logger.warning(
            "bootstrap without a fitted calibrator requires complete evaluation "
            "oracle coverage for every policy; falling back to cluster-robust "
            "inference for the mixed/uncalibrated routes"
        )
        return "calibrator_unavailable_for_non_oracle_routes"

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

        self._eval_table = self._build_eval_table()
        self._resolve_legacy_evaluation_linkage()

        self._fitted = True
        logger.info(
            f"Prepared data for {len(self._policy_data)} policies from fresh draws"
        )

    def _resolve_legacy_evaluation_linkage(self) -> None:
        """Link retained legacy fit rows only when every match is unique.

        High-level analysis always passes explicit provenance. This narrow
        compatibility bridge handles older direct constructors that fitted a
        calibrator from the same labeled fresh draws but could not declare row
        roles. Ambiguous or partial matches remain external.
        """
        if (
            self._provenance_explicit
            or self.calibration_provenance is None
            or self._eval_table is None
            or self._eval_table.row_keys is None
        ):
            return
        table = self._eval_table
        matched_keys: List[Optional[Tuple[str, str, int]]] = []
        used_rows: set = set()
        for score, label, prompt_id in zip(
            self.calibration_provenance.judge_scores,
            self.calibration_provenance.oracle_labels,
            self.calibration_provenance.prompt_ids,
        ):
            candidates = [
                row
                for row in range(len(table.judge_scores))
                if row not in used_rows
                and table.oracle_mask[row]
                and table.prompt_id_strings[row] == prompt_id
                and np.isclose(table.judge_scores[row], score)
                and np.isclose(table.oracle_labels[row], label)
            ]
            if len(candidates) != 1:
                return
            row = candidates[0]
            used_rows.add(row)
            matched_keys.append(table.row_keys[row])

        self.calibration_provenance = CalibrationProvenance(
            judge_scores=self.calibration_provenance.judge_scores,
            oracle_labels=self.calibration_provenance.oracle_labels,
            prompt_ids=self.calibration_provenance.prompt_ids,
            covariates=self.calibration_provenance.covariates,
            row_roles=["evaluation"] * len(matched_keys),
            evaluation_keys=matched_keys,
            sample_weights=self.calibration_provenance.sample_weights,
        )
        self._provenance_linkage_validated = True

    def _effective_augmentation(self) -> bool:
        """Whether residual augmentation has a defensible provenance contract."""
        if not self.use_augmented_estimator:
            return False
        if self.label_design.kind == "targeted_unknown":
            return True  # The point routine records an explicit plug-in route.
        if self._eval_table is not None:
            full_coverage = []
            for policy_index in range(self._eval_table.n_policies):
                rows = self._eval_table.policy_indices == policy_index
                full_coverage.append(
                    bool(np.any(rows) and np.all(self._eval_table.oracle_mask[rows]))
                )
            if all(full_coverage):
                return False
        if self.reward_calibrator is None:
            return False
        if self._provenance_linkage_validated:
            return True
        logger.warning(
            "Residual augmentation was requested without explicit calibration "
            "provenance. Returning the calibrated plug-in estimator; pass "
            "CalibrationProvenance to distinguish external and evaluation-linked "
            "fit rows."
        )
        return False

    def _compute_point_estimate(self) -> DirectPointEstimate:
        """Run the single point-estimator implementation on observed data."""
        if self._eval_table is None:
            raise RuntimeError("Evaluation table is unavailable; call fit() first")
        eval_table = self._eval_table
        if self.reward_calibrator is None:
            calibrated = eval_table.judge_scores.astype(float, copy=True)
            residual_predictions = calibrated.copy()
        else:
            calibrated = np.clip(
                self.reward_calibrator.predict(
                    eval_table.judge_scores, covariates=eval_table.covariates
                ),
                0.0,
                1.0,
            )
            residual_predictions = residual_predictions_for_evaluation(
                self.reward_calibrator,
                calibrated,
                eval_table,
                (
                    self.calibration_provenance
                    if self._provenance_linkage_validated
                    else None
                ),
            )

        point = compute_direct_point_estimate(
            calibrated,
            eval_table,
            residual_predictions,
            self.label_design,
            use_augmented_estimator=self._effective_augmentation(),
        )
        self._last_point = point
        return point

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
            "label_design": self.label_design.kind,
        }
        metadata.update(extra_metadata)
        if self.label_design.kind == "targeted_unknown":
            metadata.setdefault("limitations", []).append(
                "Oracle labels were targeted with unknown inclusion "
                "probabilities; residual augmentation was disabled and partial-"
                "coverage policies use the calibrated plug-in estimator."
            )

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
        point = self._compute_point_estimate()
        estimates = []
        standard_errors = []
        n_samples_used = {}
        influence_functions = {}

        for policy_index, policy in enumerate(self.target_policies):
            if policy not in self._policy_data:
                logger.warning(f"No data for policy '{policy}', using NaN")
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            pdata = self._policy_data[policy]

            estimate = float(point.estimates[policy_index])
            pseudo_outcomes = point.pseudo_outcomes[policy_index]

            # Row contributions from the same point estimator used by the
            # bootstrap (raw oracle, augmented, or plug-in by policy route).
            if_values = pseudo_outcomes - estimate
            influence_functions[policy] = if_values

            # Prompt is the sampling unit on every analytic path, including a
            # single policy and unaligned policy samples.
            n = len(pseudo_outcomes)
            from ..diagnostics.robust_inference import cluster_robust_se

            unique_prompts = sorted(set(pdata.prompt_ids))
            prompt_to_cluster = {pid: i for i, pid in enumerate(unique_prompts)}
            cluster_ids = np.asarray(
                [prompt_to_cluster[pid] for pid in pdata.prompt_ids], dtype=int
            )
            n_clusters = len(unique_prompts)
            df_cluster = max(n_clusters - 1, 0)
            se_method = "cluster_robust"
            if n_clusters < 2:
                se = float("nan")
                se_method = "unavailable_one_cluster"
                logger.warning(
                    f"Policy '{policy}' has one unique prompt cluster; returning "
                    "the point estimate with SE unavailable. Row-level IID "
                    "inference is not a valid fallback."
                )
            else:
                res = cluster_robust_se(
                    data=if_values,
                    cluster_ids=cluster_ids,
                    statistic_fn=lambda x: np.mean(x),
                    influence_fn=lambda x: x,
                    alpha=0.05,
                )
                se = float(res["se"])
                n_clusters = int(res["n_clusters"])
                df_cluster = int(res["df"])

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
            "point_estimator": point.diagnostics,
            "calibration_provenance": (
                self.calibration_provenance.summary()
                if self.calibration_provenance is not None
                else None
            ),
            "calibration_provenance_explicit": self._provenance_explicit,
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

        # Store per-pair difference SEs (pairing-aware sampling SE + OUA
        # variance of the difference) for compare_policies. Uses the
        # pre-OUA sampling SEs — the OUA term enters per pair as the
        # jackknife variance of the DIFFERENCE, not per policy.
        if len(self.target_policies) > 1:
            self._store_pairwise_inference(result, sampling_ses=standard_errors)

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

        # Check if oracle variance is already included (e.g., by the
        # bootstrap inference path, which refits the calibrator per replicate)
        if isinstance(result.metadata, dict) and result.metadata.get(
            "se_components", {}
        ).get("includes_oracle_uncertainty"):
            # Oracle variance already included in standard_errors
            return

        var_oracle_map: Dict[str, float] = {}
        jk_counts: Dict[str, int] = {}
        status_by_policy: Dict[str, str] = {}
        skipped_by_policy: Dict[str, str] = {}
        any_applied = False

        for i, policy in enumerate(self.target_policies):
            route = self._route_for_policy(policy)
            var_orc = 0.0
            K = 0
            if route == "direct_oracle":
                status_by_policy[policy] = "skipped_direct_oracle"
                skipped_by_policy[policy] = "100% evaluation oracle coverage"
            else:
                try:
                    jack = self.get_oracle_jackknife(policy)
                except Exception as exc:
                    logger.debug("Oracle jackknife unavailable for %s: %s", policy, exc)
                    jack = None
                if jack is not None and len(jack) >= 2:
                    K = len(jack)
                    var_orc = oracle_jackknife_variance(jack)
                    status_by_policy[policy] = "applied"
                    any_applied = True
                    if i < len(result.standard_errors):
                        se_base = float(result.standard_errors[i])
                        df_cluster = getattr(self, "_df_cluster", {}).get(policy, 1)
                        se_total, _ = combine_cluster_and_oracle(
                            se_base, df_cluster, var_orc, K
                        )
                        result.standard_errors[i] = se_total
                else:
                    status_by_policy[policy] = "unavailable"
                    skipped_by_policy[policy] = getattr(
                        self,
                        "_oracle_jackknife_unavailable_reason",
                        "fewer than two fitted calibration fold models",
                    )

            var_oracle_map[policy] = float(var_orc)
            jk_counts[policy] = int(K)

        if isinstance(result.metadata, dict):
            components = result.metadata.setdefault("se_components", {})
            components.update(
                {
                    "includes_oracle_uncertainty": any_applied,
                    "oracle_variance_per_policy": var_oracle_map,
                    "oracle_jackknife_counts": jk_counts,
                    "oracle_jackknife_status_per_policy": status_by_policy,
                    "oracle_uncertainty_skipped_per_policy": skipped_by_policy,
                }
            )
            if skipped_by_policy and len(skipped_by_policy) == len(
                self.target_policies
            ):
                unique_statuses = set(status_by_policy.values())
                if unique_statuses == {"skipped_direct_oracle"}:
                    components["oracle_uncertainty_skipped"] = (
                        "100% evaluation oracle coverage"
                    )
                elif "unavailable" in unique_statuses:
                    components["oracle_uncertainty_skipped"] = (
                        "oracle jackknife unavailable"
                    )

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
                emit_warning=self._route_for_policy(policy) != "direct_oracle",
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
            card = dict(card)
            applies_to_current_estimate = (
                self._route_for_policy(policy) != "direct_oracle"
            )
            card["applies_to_current_estimate"] = applies_to_current_estimate
            boundary_cards[policy] = card
            if not applies_to_current_estimate:
                continue
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
            calibration_rmse=cal_info.get("oof_rmse", cal_info.get("rmse")),
            calibration_coverage=cal_info.get(
                "oof_coverage_at_01", cal_info.get("coverage_at_01")
            ),
            calibration_tolerance=(
                0.1
                if cal_info.get("oof_coverage_at_01", cal_info.get("coverage_at_01"))
                is not None
                else None
            ),
            n_oracle_labels=cal_info.get("n_oracle_labels"),
        )

        return diagnostics

    def _attach_reliability_metadata(
        self, metadata: Dict[str, Any], diagnostics: DirectDiagnostics
    ) -> None:
        """Record boundary cards and reliability gates in result metadata.

        The CLI's best-policy announcement reads
        ``metadata["reliability_gates"][policy]["flagged"]``: a REFUSE-LEVEL
        coverage badge qualifies the policy's point-winner announcement.
        """
        if not diagnostics.boundary_cards:
            return
        metadata["boundary_cards"] = diagnostics.boundary_cards
        gates: Dict[str, Dict[str, Any]] = {}
        for policy, card in diagnostics.boundary_cards.items():
            applies_to_current_estimate = bool(
                card.get("applies_to_current_estimate", True)
            )
            refuse_level = (
                card.get("status") == "REFUSE-LEVEL" and applies_to_current_estimate
            )
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

    def _oracle_jackknife_matrix(self) -> Optional[np.ndarray]:
        """Recompute the reported point functional under each fold model."""
        cached = getattr(self, "_oracle_jackknife_matrix_cache", None)
        if isinstance(cached, np.ndarray):
            return cached
        self._oracle_jackknife_unavailable_reason = "unknown"
        if not self._fitted or self.reward_calibrator is None:
            self._oracle_jackknife_unavailable_reason = "calibrator_unavailable"
            return None
        if self._eval_table is None:
            self._oracle_jackknife_unavailable_reason = "evaluation_table_unavailable"
            return None
        if not hasattr(
            self.reward_calibrator, "get_fold_models_for_oua"
        ) or not hasattr(self.reward_calibrator, "predict_oof"):
            self._oracle_jackknife_unavailable_reason = (
                "calibrator_does_not_expose_cross_fitted_models"
            )
            return None

        fold_models = self.reward_calibrator.get_fold_models_for_oua()
        n_folds = sum(model is not None for model in fold_models.values())
        if n_folds < 2:
            self._oracle_jackknife_unavailable_reason = (
                "fewer_than_two_fitted_calibration_fold_models"
            )
            return None

        table = self._eval_table
        try:
            matrix = direct_oracle_jackknife_estimates(
                self.reward_calibrator,
                table,
                self.label_design,
                use_augmented_estimator=self._effective_augmentation(),
            )
        except Exception as exc:
            self._oracle_jackknife_unavailable_reason = (
                f"point_functional_recomputation_failed: {exc}"
            )
            logger.warning(
                "Calibration-aware oracle jackknife could not recompute the "
                "point functional: %s",
                exc,
            )
            return None

        if matrix is None:
            self._oracle_jackknife_unavailable_reason = (
                "fewer_than_two_fitted_calibration_fold_models"
            )
            return None
        if matrix.shape != (n_folds, len(self.target_policies)) or not np.all(
            np.isfinite(matrix)
        ):
            self._oracle_jackknife_unavailable_reason = (
                "non_finite_or_misaligned_jackknife_estimates"
            )
            return None
        self._oracle_jackknife_matrix_cache = matrix
        self._oracle_jackknife_unavailable_reason = "available"
        return matrix

    def get_oracle_jackknife(self, policy: str) -> Optional[np.ndarray]:
        """Return leave-fold estimates of this policy's reported functional."""
        if policy not in self.target_policies or policy not in self._policy_data:
            self._oracle_jackknife_unavailable_reason = "policy_unavailable"
            return None
        matrix = self._oracle_jackknife_matrix()
        if matrix is None:
            return None
        return matrix[:, self.target_policies.index(policy)].copy()

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
        components = (
            result.metadata.get("se_components", {})
            if isinstance(result.metadata, dict)
            else {}
        )
        jk_counts = components.get("oracle_jackknife_counts", {})
        jk_statuses = components.get("oracle_jackknife_status_per_policy", {})

        for i, policy in enumerate(self.target_policies):
            if np.isnan(result.estimates[i]) or np.isnan(result.standard_errors[i]):
                df_info[policy] = {
                    "available": False,
                    "df": None,
                    "reason": "standard_error_unavailable",
                    "se_method": self._se_methods.get(policy, "unavailable"),
                    "n_clusters": self._n_clusters.get(policy, 0),
                    "oracle_jackknife_status": jk_statuses.get(policy, "not_requested"),
                    "oracle_jackknife_folds": int(jk_counts.get(policy, 0)),
                }
                continue

            # Get cluster DF
            df_cluster = self._df_cluster.get(policy, len(result.estimates) - 1)

            # If oracle-jackknife inference was applied, cap DF by the
            # oracle fold count (shared combining rule)
            status = str(jk_statuses.get(policy, "not_requested"))
            K = int(jk_counts.get(policy, 0)) if status == "applied" else 0

            _, df_final = combine_cluster_and_oracle(0.0, df_cluster, 0.0, K)

            # Compute t-critical value for logging
            t_crit = stats.t.ppf(1 - 0.05 / 2, df_final)

            df_info[policy] = {
                "available": True,
                "df": int(df_final),
                "t_critical": float(t_crit),
                "se_method": self._se_methods.get(policy, "standard"),
                "n_clusters": self._n_clusters.get(policy, len(result.estimates)),
                "oracle_jackknife_status": status,
                "oracle_jackknife_folds": K,
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

    def _store_pairwise_inference(
        self, result: EstimationResult, sampling_ses: List[float]
    ) -> None:
        """Store per-pair difference inference for the analytic path.

        Writes ``result.metadata["pairwise_inference"]`` with one entry per
        (i < j) policy pair, keyed ``"i-j"`` in estimate order, that
        ``EstimationResult.compare_policies`` consumes as its
        "paired_if_oua" basis. Each entry combines:

        - a sampling SE with an explicit pairing ``basis``:
          "index_paired" (identical ordered prompt_id lists — per-row IF
          differences, the pre-0.5.1 SE now justified by verified row
          alignment), "prompt_paired" (same prompts as multisets but
          different order — IFs aggregated per prompt_id, paired per
          prompt), or "independent" (partial/no prompt overlap —
          independent combination of the per-policy sampling SEs);
        - the oracle-jackknife variance of the DIFFERENCE,
          Var_jk(jack_i - jack_j), from the cached per-policy
          leave-one-fold estimate vectors (shared fold order across
          policies), so shared-calibrator error cancels where it should
          (F̂_i ≈ F̂_j on near-tie pairs makes this term ≈ 0 — the
          analytic near-tie no-op is correct, unlike the bootstrap path's
          residual-correction noise);

        as se = sqrt(se_sampling² + var_oua_diff) with
        df = max(min(df_pairs, K - 1), 1) via combine_cluster_and_oracle.

        Args:
            result: Assembled result (post-OUA standard errors).
            sampling_ses: PRE-OUA per-policy sampling SEs in
                target_policies order (the OUA component enters per pair
                via the jackknife difference variance, never double-counted
                from the per-policy SEs).
        """
        policies = self.target_policies
        if len(policies) < 2:
            return

        # Per-policy jackknife vectors (None disables the pair's OUA term),
        # mirroring the OUA skip conditions in _apply_oua_jackknife.
        jack_by_policy: Dict[str, Optional[np.ndarray]] = {p: None for p in policies}
        if self.oua_jackknife and self.reward_calibrator is not None:
            for policy in policies:
                try:
                    jack_by_policy[policy] = self.get_oracle_jackknife(policy)
                except Exception as e:
                    logger.debug(
                        f"Pairwise OUA: oracle jackknife failed for " f"'{policy}': {e}"
                    )

        pairwise: Dict[str, Dict[str, Any]] = {}
        for i in range(len(policies)):
            for j in range(i + 1, len(policies)):
                try:
                    entry = self._pairwise_entry(
                        result, i, j, sampling_ses, jack_by_policy
                    )
                except Exception as e:
                    logger.debug(f"Pairwise inference failed for pair {i}-{j}: {e}")
                    entry = None
                if entry is not None:
                    pairwise[f"{i}-{j}"] = entry

        if pairwise and isinstance(result.metadata, dict):
            result.metadata["pairwise_inference"] = pairwise

    def _pairwise_entry(
        self,
        result: EstimationResult,
        i: int,
        j: int,
        sampling_ses: List[float],
        jack_by_policy: Dict[str, Optional[np.ndarray]],
    ) -> Optional[Dict[str, Any]]:
        """Build one JSON-safe metadata["pairwise_inference"] entry.

        Returns None when the pair cannot support difference inference
        (NaN estimates, missing data, or a degenerate SE) — the pair is
        then simply absent and compare_policies falls through to its
        legacy basis.
        """
        policies = self.target_policies
        p1, p2 = policies[i], policies[j]
        if i >= len(result.estimates) or j >= len(result.estimates):
            return None
        if np.isnan(result.estimates[i]) or np.isnan(result.estimates[j]):
            return None
        pd1 = self._policy_data.get(p1)
        pd2 = self._policy_data.get(p2)
        if pd1 is None or pd2 is None:
            return None

        if not self.paired_comparison:
            if i >= len(sampling_ses) or j >= len(sampling_ses):
                return None
            se_sampling = float(
                np.sqrt(float(sampling_ses[i]) ** 2 + float(sampling_ses[j]) ** 2)
            )
            df_pairs = min(
                int(getattr(self, "_df_cluster", {}).get(p1, 1)),
                int(getattr(self, "_df_cluster", {}).get(p2, 1)),
            )
            basis = "independent_requested"
            n_pairs = 0
        else:
            if not result.influence_functions:
                return None
            if1 = np.asarray(result.influence_functions.get(p1), dtype=float)
            if2 = np.asarray(result.influence_functions.get(p2), dtype=float)
            if if1.shape != (len(pd1.prompt_ids),) or if2.shape != (
                len(pd2.prompt_ids),
            ):
                return None

            # Cluster contributions to each policy mean. Taking their
            # difference over the prompt union handles repeated draws and
            # partial overlap while retaining requested prompt covariance.
            contributions1: Dict[str, float] = {}
            contributions2: Dict[str, float] = {}
            for value, prompt_id in zip(if1, pd1.prompt_ids):
                contributions1[prompt_id] = contributions1.get(prompt_id, 0.0) + float(
                    value
                ) / len(if1)
            for value, prompt_id in zip(if2, pd2.prompt_ids):
                contributions2[prompt_id] = contributions2.get(prompt_id, 0.0) + float(
                    value
                ) / len(if2)
            union_prompts = sorted(set(contributions1) | set(contributions2))
            G = len(union_prompts)
            if G < 2:
                return None
            cluster_differences = np.asarray(
                [
                    contributions1.get(prompt_id, 0.0)
                    - contributions2.get(prompt_id, 0.0)
                    for prompt_id in union_prompts
                ],
                dtype=float,
            )
            cluster_differences -= np.mean(cluster_differences)
            se_sampling = float(np.sqrt((G / (G - 1)) * np.sum(cluster_differences**2)))
            df_pairs = G - 1
            overlap = set(contributions1) & set(contributions2)
            if len(overlap) == G:
                basis = "prompt_cluster_paired"
                n_pairs = G
            elif overlap:
                basis = "prompt_cluster_partial_overlap"
                n_pairs = len(overlap)
            else:
                basis = "prompt_cluster_disjoint"
                n_pairs = 0

        # Oracle-uncertainty component: jackknife variance of the
        # DIFFERENCE of the per-policy leave-one-fold estimate vectors
        # (fold order is shared — both come from the same calibrator's
        # fold models via get_oracle_jackknife)
        jack1 = jack_by_policy.get(p1)
        jack2 = jack_by_policy.get(p2)
        var_oua_diff = 0.0
        n_folds = 0
        if (
            jack1 is not None
            and jack2 is not None
            and len(jack1) == len(jack2)
            and len(jack1) >= 2
        ):
            var_oua_diff = float(
                oracle_jackknife_variance(
                    np.asarray(jack1, dtype=float) - np.asarray(jack2, dtype=float)
                )
            )
            n_folds = len(jack1)
        if (
            self._route_for_policy(p1) == "direct_oracle"
            and self._route_for_policy(p2) == "direct_oracle"
        ):
            n_folds = 0
            var_oua_diff = 0.0

        se_total, df_final = combine_cluster_and_oracle(
            se_sampling, df_pairs, var_oua_diff, n_folds
        )
        if not np.isfinite(se_total) or se_total <= 0:
            return None

        return {
            "policy1": p1,
            "policy2": p2,
            "se": float(se_total),
            "df": int(df_final),
            "basis": basis,
            "se_sampling": float(se_sampling),
            "var_oua_diff": float(var_oua_diff),
            "n_pairs": int(n_pairs),
            "oua_folds": int(n_folds),
        }

    @staticmethod
    def _per_prompt_means(
        values: np.ndarray, prompt_ids: List[str]
    ) -> Dict[str, float]:
        """Mean of `values` per prompt_id (for prompt-paired difference SEs)."""
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for value, pid in zip(values, prompt_ids):
            sums[pid] = sums.get(pid, 0.0) + float(value)
            counts[pid] = counts.get(pid, 0) + 1
        return {pid: sums[pid] / counts[pid] for pid in sums}

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
            cluster_bootstrap_direct_with_refit,
            make_calibrator_factory,
        )

        logger.info(f"Using cluster bootstrap inference ({bootstrap_reason})")

        # Build evaluation table from fresh draws
        eval_table = self._build_eval_table(
            couple_policy_prompts=self.paired_comparison
        )

        point = self._compute_point_estimate()

        # Preserve auto mode when it was requested: mode selection is part of
        # the estimator and is rerun in every weighted bootstrap world.
        from typing import Literal, cast

        if self.reward_calibrator is not None:
            if self._provenance_explicit:
                mode_str = getattr(
                    self.reward_calibrator, "calibration_mode", "monotone"
                )
            else:
                mode_str = getattr(
                    self.reward_calibrator, "selected_mode", None
                ) or getattr(self.reward_calibrator, "calibration_mode", "monotone")
            if mode_str not in ("monotone", "two_stage", "auto"):
                mode_str = "monotone"
        else:
            mode_str = "monotone"

        selected_mode = cast(Literal["monotone", "two_stage", "auto"], mode_str)

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

        # Positive weights retain every calibration cluster, so there is no
        # minimum-label rejection threshold and exactly B worlds are run.
        bootstrap_result = cluster_bootstrap_direct_with_refit(
            eval_table=eval_table,
            calibrator_factory=calibrator_factory,
            n_bootstrap=self.n_bootstrap,
            alpha=0.05,
            seed=self.bootstrap_seed,
            use_augmented_estimator=self._effective_augmentation(),
            calibration_provenance=self.calibration_provenance,
            label_design=self.label_design,
            point_calibrator=self.reward_calibrator,
        )

        # ESTIMATOR CONSISTENCY: Use bootstrap's theta_hat as reported estimate.
        # The bootstrap refits calibrator on fresh draws oracle, so we must use its
        # point estimates for consistency. Using self._policy_data (logged-data calibrator)
        # would create an estimator mismatch (expected with small oracle samples).
        estimates = bootstrap_result["estimates"]
        if not np.allclose(estimates, point.estimates, equal_nan=True):
            raise RuntimeError(
                "Internal estimator mismatch: analytic and bootstrap point "
                "implementations returned different values"
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
                        point.pseudo_outcomes[policy_idx] - estimates[policy_idx]
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
                "bootstrap_scheme": bootstrap_result["bootstrap_scheme"],
                "oracle_count_summary": bootstrap_result["oracle_count_summary"],
                "bootstrap_reason": bootstrap_reason,
                "label_design": self.label_design.kind,
                "effective_estimator_routes": point.diagnostics["routes"],
                "paired_comparison": self.paired_comparison,
                "prompt_weight_coupling": (
                    "shared_by_prompt"
                    if self.paired_comparison
                    else "independent_by_policy_prompt"
                ),
                "policy_cluster_counts": bootstrap_result["policy_cluster_counts"],
                "inference_unavailable_policies": bootstrap_result[
                    "inference_unavailable_policies"
                ],
                "inference_unavailable_reason": bootstrap_result[
                    "inference_unavailable_reason"
                ],
            },
            "bootstrap_ci": {
                "lower": [float(x) for x in bootstrap_result["ci_lower"]],
                "upper": [float(x) for x in bootstrap_result["ci_upper"]],
                "method": "percentile",
                "alpha": 0.05,
            },
            "se_components": {
                "includes_oracle_uncertainty": bool(
                    self.reward_calibrator is not None
                    and any(
                        route != "direct_oracle"
                        for route in point.diagnostics["routes"]
                    )
                ),
                "includes_mc_variance": False,
                "via_bootstrap": True,
                "calibration_variance_decomposition": {
                    "available": False,
                    "reason": (
                        "joint bootstrap does not separately identify calibration "
                        "variance"
                    ),
                },
            },
            "point_estimator": point.diagnostics,
            "calibration_provenance": bootstrap_result["provenance_summary"],
            "calibration_provenance_explicit": self._provenance_explicit,
            "inference_unavailable_policies": bootstrap_result[
                "inference_unavailable_policies"
            ],
        }

        result = self._assemble_result(
            estimates=list(estimates),
            standard_errors=list(standard_errors),
            n_samples_used=n_samples_used,
            influence_functions=influence_functions,
            method=f"{self._method_name}_bootstrap",
            extra_metadata=extra_metadata,
        )

        # Attach the paired (B, P) replicate matrix — one joint cluster
        # resample + calibrator refit per row — so compare_policies can do
        # honest paired difference inference (its "paired_bootstrap" path).
        # Attach-only: the per-policy estimates/SEs/CIs above are untouched.
        # Columns follow eval_table.policy_names == self.target_policies
        # (the same ordering assumption the estimates above already rely on).
        bootstrap_matrix = np.asarray(bootstrap_result["bootstrap_matrix"], dtype=float)
        if bootstrap_matrix.ndim == 2 and bootstrap_matrix.shape[1] == len(
            self.target_policies
        ):
            result.bootstrap_samples = bootstrap_matrix

        # Log the exact percentile interval returned to users.
        for i, policy in enumerate(self.target_policies):
            if not np.isnan(estimates[i]):
                ci_lower = bootstrap_result["ci_lower"][i]
                ci_upper = bootstrap_result["ci_upper"][i]
                logger.info(
                    f"Bootstrap estimate for '{policy}': {estimates[i]:.4f} ± {standard_errors[i]:.4f} "
                    f"(95% percentile CI: [{ci_lower:.4f}, {ci_upper:.4f}])"
                )

        return result
