"""High-level analysis service.

Encapsulates the end-to-end workflow and uses the estimator registry.
The public API still exposes analyze_dataset(...) for simplicity.
"""

from typing import Any, Dict, List, Optional
import logging
from pathlib import Path

from .config import AnalysisConfig
from .factory import validate_estimator_name
from ..data import load_dataset_from_jsonl
from ..data.models import Dataset, EstimationResult
from ..calibration import calibrate_dataset

logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self) -> None:
        pass

    def _metadata_estimator_config(
        self, estimator_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Drop legacy compatibility keys that are no longer operational."""
        sanitized = dict(estimator_config)
        sanitized.pop("use_multipolicy_eif", None)
        return sanitized

    def _build_covariate_list(
        self, config: AnalysisConfig, dataset: Optional[Dataset] = None
    ) -> Optional[List[str]]:
        """
        Build the final covariate list, prepending auto-computable covariates if needed.

        Args:
            config: Analysis configuration
            dataset: Optional dataset to validate against (checks for response field)

        Returns:
            List of covariate names to use, or None if no covariates specified
        """
        # Start with user-specified covariates
        covariates = list(config.calibration_covariates or [])

        # Prepend response_length if flag is set
        if config.include_response_length:
            # Validate that dataset has response field
            if dataset is not None:
                for i, sample in enumerate(dataset.samples):
                    if not hasattr(sample, "response") or sample.response is None:
                        raise ValueError(
                            f"include_response_length=True requires all samples to have a 'response' field. "
                            f"Sample {i} (prompt_id={sample.prompt_id}) is missing this field or it is None."
                        )

            # Add response_length to the front of the list if not already there
            if "response_length" not in covariates:
                covariates.insert(0, "response_length")

        return covariates if covariates else None

    def _direct_calibration_folds(self, n_oracle: int, n_folds: int = 5) -> int:
        """Choose the number of calibration folds for Direct mode.

        Cross-fitted calibration needs at least 2 oracle samples per fold.
        When fewer than 2 * n_folds oracle labels are available (but at
        least 4), reduce the fold count instead of failing outright.

        Args:
            n_oracle: Number of oracle-labeled samples available for calibration
            n_folds: Desired number of CV folds

        Returns:
            Number of folds to use (possibly reduced)
        """
        if n_oracle >= n_folds * 2 or n_oracle < 4:
            # Enough labels for the requested folds, or too few to calibrate
            # at all (let the calibrator raise its actionable error).
            return n_folds
        reduced_folds = max(2, n_oracle // 2)
        logger.warning(
            f"Only {n_oracle} oracle-labeled samples available; reducing "
            f"calibration folds from {n_folds} to {reduced_folds}. Results "
            f"will be noisier — provide at least {n_folds * 2} oracle labels "
            f"for stable calibration."
        )
        return reduced_folds

    def run(self, config: AnalysisConfig) -> EstimationResult:
        """Run the single Direct-mode flow.

        validate config -> load draws -> resolve calibration source ->
        calibrate -> CalibratedDirectEstimator -> covariates -> estimate ->
        denormalize -> metadata.

        OPE (IPS/DR) modes were removed in 0.4.0; analyze_dataset raises the
        migration error for logged_data_path before a config is ever built.
        """
        from ..data.fresh_draws import (
            NormalizationInfo,
            compute_response_covariates,
            discover_policies_from_fresh_draws,
            fresh_draws_from_dict,
            load_fresh_draws_auto,
        )
        from ..data.models import Dataset, Sample
        from ..estimators.direct_method import CalibratedDirectEstimator

        # --- 1. Validate config
        if config.fresh_draws_dir is None and config.fresh_draws_data is None:
            raise ValueError(
                "Must provide fresh_draws_dir or fresh_draws_data for Direct mode"
            )

        chosen_estimator = config.estimator.lower() if config.estimator else "auto"
        if chosen_estimator == "auto":
            chosen_estimator = "direct"
        else:
            # Raises the migration error for removed OPE estimator names.
            validate_estimator_name(chosen_estimator)

        # --- 2. Load fresh draws (in-memory dict OR directory); norm_info
        # tracks auto-normalization for the inverse transform in stage 5
        norm_info: Optional[NormalizationInfo] = None

        if config.fresh_draws_data is not None:
            if config.verbose:
                logger.info("Using in-memory fresh_draws_data")
            fresh_draws_dict, norm_info = fresh_draws_from_dict(
                config.fresh_draws_data, verbose=config.verbose, auto_normalize=True
            )
            target_policies = sorted(fresh_draws_dict.keys())
        else:
            assert config.fresh_draws_dir is not None  # validated in stage 1
            draws_dir = Path(config.fresh_draws_dir)
            target_policies = discover_policies_from_fresh_draws(draws_dir)
            fresh_draws_dict = {}
            for policy in target_policies:
                fresh_draws_dict[policy] = load_fresh_draws_auto(
                    draws_dir, policy, verbose=config.verbose
                )

        if config.verbose:
            logger.info(
                f"Found {len(target_policies)} policies: {', '.join(target_policies)}"
            )

        all_fresh_draws = []
        for fd in fresh_draws_dict.values():
            all_fresh_draws.extend(fd.samples)

        # --- 3. Resolve the calibration source and calibrate:
        # calibration_data_path (optionally pooled with labeled draws via
        # _combine_oracle_sources) OR oracle-in-draws
        oracle_sources_metadata = None
        calibration_result = None
        covariate_names = None
        oracle_coverage = 0.0

        if config.calibration_data_path:
            if config.verbose:
                logger.info(
                    f"Loading calibration dataset from {config.calibration_data_path}"
                )
            calibration_dataset = load_dataset_from_jsonl(config.calibration_data_path)
            if config.verbose:
                logger.info(
                    f"Loaded calibration dataset: {calibration_dataset.n_samples} samples"
                )

            if config.combine_oracle_sources:
                if config.verbose:
                    logger.info(
                        "Combining oracle sources from calibration data and fresh draws"
                    )
                calibration_dataset_for_rewards, oracle_sources_metadata = (
                    self._combine_oracle_sources(
                        calibration_dataset,
                        None,  # No logged dataset in Direct mode
                        fresh_draws_dict,
                        target_policies,
                        config.judge_field,
                        config.oracle_field,
                        config.verbose,
                    )
                )
            else:
                # Use ONLY calibration_data_path for learning calibration
                if config.verbose:
                    logger.info(
                        "Using calibration data exclusively (combine_oracle_sources=False)"
                    )
                calibration_dataset_for_rewards = calibration_dataset
                n_calib_oracle = sum(
                    1 for s in calibration_dataset.samples if s.oracle_label is not None
                )
                oracle_sources_metadata = {
                    "calibration_data": {
                        "n_oracle": n_calib_oracle,
                        "coverage": n_calib_oracle / calibration_dataset.n_samples,
                    },
                    "total_oracle": n_calib_oracle,
                    "combine_enabled": False,
                }

            # Covariate list (validates response fields when
            # include_response_length is set), then learn the calibration
            covariate_names = self._build_covariate_list(
                config, calibration_dataset_for_rewards
            )
            n_oracle_for_calibration = sum(
                1
                for s in calibration_dataset_for_rewards.samples
                if s.oracle_label is not None
            )
            _, calibration_result = calibrate_dataset(
                calibration_dataset_for_rewards,
                judge_field=config.judge_field,
                oracle_field=config.oracle_field,
                enable_cross_fit=True,
                n_folds=self._direct_calibration_folds(n_oracle_for_calibration),
                covariate_names=covariate_names,
            )
        else:
            # Oracle-in-draws: learn calibration from labeled fresh draws
            n_with_oracle = sum(
                1 for s in all_fresh_draws if s.oracle_label is not None
            )
            oracle_coverage = (
                n_with_oracle / len(all_fresh_draws) if all_fresh_draws else 0
            )

            if oracle_coverage > 0:
                if config.verbose:
                    logger.info(
                        f"Found {n_with_oracle}/{len(all_fresh_draws)} samples with oracle labels ({oracle_coverage:.1%})"
                    )
                    logger.info("Learning calibration from fresh draws")

                # Convert FreshDrawSample to Sample (dummy fields) so the
                # draws can serve as a calibration dataset
                calibration_samples = []
                for fd_sample in all_fresh_draws:
                    sample = Sample(
                        prompt_id=fd_sample.prompt_id,
                        prompt="",  # Not needed for calibration
                        response=fd_sample.response or "",
                        reward=None,  # Will be calibrated
                        base_policy_logprob=-1.0,  # Dummy value
                        target_policy_logprobs={p: -1.0 for p in target_policies},
                        judge_score=fd_sample.judge_score,
                        oracle_label=fd_sample.oracle_label,
                        metadata={},
                    )
                    calibration_samples.append(sample)
                fresh_dataset = Dataset(
                    samples=calibration_samples, target_policies=target_policies
                )

                covariate_names = self._build_covariate_list(config, fresh_dataset)
                _, calibration_result = calibrate_dataset(
                    fresh_dataset,
                    judge_field=config.judge_field,
                    oracle_field=config.oracle_field,
                    enable_cross_fit=True,
                    n_folds=self._direct_calibration_folds(n_with_oracle),
                    covariate_names=covariate_names,
                )
            else:
                logger.warning(
                    "No oracle labels found — returning UNCALIBRATED judge-score "
                    "means; CIs do not account for judge bias. Results will be "
                    "labeled method='naive_direct'. Add oracle labels to the "
                    "fresh draws or provide calibration_data_path to calibrate."
                )

        # --- 4. Build the estimator, attach draws (computing covariates
        # when the calibrator expects them), and estimate
        estimator_obj = CalibratedDirectEstimator(
            target_policies=target_policies,
            reward_calibrator=(
                calibration_result.calibrator if calibration_result else None
            ),
            run_diagnostics=True,
            oua_jackknife=(
                calibration_result is not None
            ),  # Include calibration uncertainty if calibrated
            **config.estimator_config,
        )

        for policy in target_policies:
            fd = fresh_draws_dict[policy]
            if covariate_names:
                fd = compute_response_covariates(fd, covariate_names=covariate_names)
            estimator_obj.add_fresh_draws(policy, fd)

        results = estimator_obj.fit_and_estimate()

        # --- 5. Inverse-transform results back to the original scale if
        # normalization was applied
        if norm_info and norm_info.oracle_label_scale:
            oracle_scale = norm_info.oracle_label_scale
            # Transform estimates and standard errors back to original scale
            results.estimates = oracle_scale.inverse_array(results.estimates)
            # Standard errors scale linearly with the range
            scale_factor = oracle_scale.max_val - oracle_scale.min_val
            results.standard_errors = results.standard_errors * scale_factor

            # Also denormalize bootstrap CIs if present
            if "bootstrap_ci" in results.metadata:
                boot_ci = results.metadata["bootstrap_ci"]
                boot_ci["lower"] = [
                    float(v * scale_factor + oracle_scale.min_val)
                    for v in boot_ci["lower"]
                ]
                boot_ci["upper"] = [
                    float(v * scale_factor + oracle_scale.min_val)
                    for v in boot_ci["upper"]
                ]

            # Add normalization metadata
            results.metadata["normalization"] = norm_info.to_dict()

            if config.verbose:
                logger.info(
                    f"Inverse-transformed results to original oracle scale "
                    f"[{oracle_scale.min_val}, {oracle_scale.max_val}]"
                )
        elif norm_info:
            # No oracle labels, use judge scale for inverse transform
            judge_scale = norm_info.judge_score_scale
            results.estimates = judge_scale.inverse_array(results.estimates)
            scale_factor = judge_scale.max_val - judge_scale.min_val
            results.standard_errors = results.standard_errors * scale_factor

            # Also denormalize bootstrap CIs if present
            if "bootstrap_ci" in results.metadata:
                boot_ci = results.metadata["bootstrap_ci"]
                boot_ci["lower"] = [
                    float(v * scale_factor + judge_scale.min_val)
                    for v in boot_ci["lower"]
                ]
                boot_ci["upper"] = [
                    float(v * scale_factor + judge_scale.min_val)
                    for v in boot_ci["upper"]
                ]

            results.metadata["normalization"] = norm_info.to_dict()
            results.metadata["normalization"]["results_scale"] = "judge_original"

            if config.verbose:
                logger.info(
                    f"Inverse-transformed results to original judge scale "
                    f"[{judge_scale.min_val}, {judge_scale.max_val}] (no oracle labels)"
                )

        # --- 6. Metadata (mode/mode_selection keys kept for 0.3.0-output compat)
        results.metadata["mode"] = "direct"
        results.metadata["estimator"] = chosen_estimator
        results.metadata["target_policies"] = target_policies
        if config.fresh_draws_dir:
            results.metadata["fresh_draws_dir"] = config.fresh_draws_dir
        if config.fresh_draws_data is not None:
            results.metadata["fresh_draws_source"] = "in_memory"

        # Calibration source metadata
        if config.calibration_data_path:
            results.metadata["calibration"] = (
                "from_calibration_data_combined"
                if config.combine_oracle_sources
                else "from_calibration_data_only"
            )
            results.metadata["calibration_data_path"] = config.calibration_data_path
        else:
            results.metadata["calibration"] = (
                "from_fresh_draws" if calibration_result else "none"
            )
            if calibration_result:
                # Only set oracle_coverage for fresh-draws-only calibration
                results.metadata["oracle_coverage"] = oracle_coverage

        metadata_estimator_config = self._metadata_estimator_config(
            config.estimator_config
        )
        if metadata_estimator_config:
            results.metadata["estimator_config"] = metadata_estimator_config

        if oracle_sources_metadata:
            results.metadata["oracle_sources"] = oracle_sources_metadata

        results.metadata["mode_selection"] = {
            "mode": "direct",
            "estimator": chosen_estimator,
            "logprob_coverage": 0.0,  # Direct-only mode has no logged data
            "has_fresh_draws": True,
            "has_logged_data": False,
            "reason": "Direct mode is the only mode in 0.4.x",
        }

        # Add calibrator for transportability audits
        if calibration_result:
            results.calibrator = calibration_result.calibrator

        return results

    def _combine_oracle_sources(
        self,
        calibration_dataset: Optional[Dataset],
        logged_dataset: Optional[Dataset],
        fresh_draws_per_policy: Optional[Dict[str, Any]],
        target_policies: List[str],
        judge_field: str,
        oracle_field: str,
        verbose: bool = False,
    ) -> tuple[Dataset, Dict[str, Any]]:
        """
        Combine oracle (judge, oracle) pairs from multiple sources.

        Oracle labels attach to RESPONSES, not prompts: several sources (or
        several policies' fresh draws) can legitimately contribute different
        pairs for the same prompt_id, and ALL of them are kept as calibration
        pairs. Only true duplicates — same response identity (source; for
        fresh draws also policy and draw index), same prompt, identical judge
        and oracle values — are deduped. Cross-source disagreements on the
        same prompt (oracle diff > 0.05) are reported as conflicts but both
        pairs still enter calibration.

        Args:
            calibration_dataset: Optional calibration dataset with oracle labels
            logged_dataset: Optional logged dataset (can be None in Direct mode)
            fresh_draws_per_policy: Optional dict of fresh draws by policy
            target_policies: List of target policy names (for dataset construction)
            judge_field: Field name for judge scores
            oracle_field: Field name for oracle labels
            verbose: Whether to log progress

        Returns:
            Tuple of (combined_dataset, oracle_sources_metadata)
        """
        from ..data.models import Sample

        # Accumulate every (prompt_id, source, judge, oracle) pair
        pairs: List[tuple[str, str, float, float]] = []
        seen_pairs: set = set()
        oracle_by_prompt: Dict[str, List[tuple[str, float]]] = {}
        conflicts: List[Dict[str, Any]] = []

        def _add_pair(
            prompt_id: str, source: str, judge_val: float, oracle_val: float
        ) -> bool:
            """Add a calibration pair; returns False for true duplicates.

            The source string identifies the response, not just the source
            family: fresh-draw pairs arrive as "fresh_draws:<policy>:draw<i>",
            so distinct policies' (or draws') labels for one prompt never
            collapse even when their (judge, oracle) values coincide — which
            is common with binary or rubric scores. Conflicts are still
            detected at the source-FAMILY level (calibration_data vs
            fresh_draws vs logged_data): different responses within one
            family legitimately earn different labels.
            """
            key = (source, prompt_id, judge_val, oracle_val)
            if key in seen_pairs:
                return False
            # Cross-source-family conflict check (pairs are kept either way)
            source_family = source.split(":", 1)[0]
            for prev_source, prev_oracle in oracle_by_prompt.get(prompt_id, []):
                prev_family = prev_source.split(":", 1)[0]
                if (
                    prev_family != source_family
                    and abs(prev_oracle - oracle_val) > 0.05
                ):
                    conflicts.append(
                        {
                            "prompt_id": prompt_id,
                            "existing_value": float(prev_oracle),
                            "existing_source": prev_source,
                            "new_value": float(oracle_val),
                            "new_source": source,
                            "difference": abs(prev_oracle - oracle_val),
                        }
                    )
                    break
            seen_pairs.add(key)
            pairs.append((prompt_id, source, judge_val, oracle_val))
            oracle_by_prompt.setdefault(prompt_id, []).append((source, oracle_val))
            return True

        # Logged data
        n_from_logged = 0
        if logged_dataset:
            for sample in logged_dataset.samples:
                oracle_val = (
                    sample.oracle_label
                    if oracle_field == "oracle_label"
                    else sample.metadata.get(oracle_field)
                )
                if oracle_val is not None:
                    judge_val = (
                        sample.judge_score
                        if judge_field == "judge_score"
                        else sample.metadata.get(judge_field)
                    )
                    if judge_val is not None:
                        if _add_pair(
                            sample.prompt_id,
                            "logged_data",
                            float(judge_val),
                            float(oracle_val),
                        ):
                            n_from_logged += 1

        # Fresh draws (every policy's labeled draws count — labels attach to
        # responses, so K policies can contribute K pairs for one prompt)
        n_from_fresh = 0
        if fresh_draws_per_policy:
            for policy, fd_dataset in fresh_draws_per_policy.items():
                for fd_sample in fd_dataset.samples:
                    if (
                        fd_sample.oracle_label is not None
                        and fd_sample.judge_score is not None
                    ):
                        # Identify the response (policy + draw), not just the
                        # family: a bare "fresh_draws" source collapsed
                        # distinct policies' draws for one prompt whenever
                        # their (judge, oracle) values coincided.
                        draw_idx = getattr(fd_sample, "draw_idx", 0)
                        if _add_pair(
                            fd_sample.prompt_id,
                            f"fresh_draws:{policy}:draw{draw_idx}",
                            float(fd_sample.judge_score),
                            float(fd_sample.oracle_label),
                        ):
                            n_from_fresh += 1

        # Calibration data
        n_from_calib = 0
        if calibration_dataset:
            for sample in calibration_dataset.samples:
                oracle_val = (
                    sample.oracle_label
                    if oracle_field == "oracle_label"
                    else sample.metadata.get(oracle_field)
                )
                if oracle_val is not None:
                    judge_val = (
                        sample.judge_score
                        if judge_field == "judge_score"
                        else sample.metadata.get(judge_field)
                    )
                    if judge_val is not None:
                        if _add_pair(
                            sample.prompt_id,
                            "calibration_data",
                            float(judge_val),
                            float(oracle_val),
                        ):
                            n_from_calib += 1

        # Log conflicts if any (loud by default — cross-source disagreement is
        # a data-quality signal the user must see)
        if conflicts:
            logger.warning(
                f"Found {len(conflicts)} prompts with conflicting oracle labels "
                f"across sources (diff > 0.05). All pairs are kept for "
                f"calibration (oracle labels attach to responses, not prompts); "
                f"large cross-source differences may indicate inconsistent "
                f"oracle definitions."
            )
            # Log top 5 conflicts
            for conflict in conflicts[:5]:
                logger.debug(
                    f"  {conflict['prompt_id']}: {conflict['existing_source']}={conflict['existing_value']:.3f} "
                    f"vs {conflict['new_source']}={conflict['new_value']:.3f} (diff={conflict['difference']:.3f})"
                )

        # Build combined dataset with every accumulated pair
        combined_samples = []
        # Use target policies parameter - cast to Dict[str, Optional[float]] for variance
        target_policies_dict: Dict[str, Optional[float]] = {
            policy: -1.0 for policy in target_policies
        }

        for prompt_id, source, judge_val, oracle_val in pairs:
            # Create Sample with judge and oracle
            sample = Sample(
                prompt_id=prompt_id,
                prompt="",  # Not needed for calibration
                response="",
                reward=None,  # Will be calibrated
                base_policy_logprob=-1.0,  # Dummy
                target_policy_logprobs=target_policies_dict.copy(),  # Match logged dataset policies
                judge_score=judge_val,
                oracle_label=oracle_val,
                metadata={"source": source},
            )
            combined_samples.append(sample)

        if verbose:
            logger.info(
                f"Combined oracle sources: {len(combined_samples)} total pairs "
                f"(calib={n_from_calib}, fresh={n_from_fresh}, logged={n_from_logged})"
            )

        # Build metadata (counts reflect pairs that actually enter calibration)
        oracle_sources_metadata = {
            "calibration_data": {
                "n_oracle": n_from_calib,
                "coverage": (
                    n_from_calib / calibration_dataset.n_samples
                    if calibration_dataset
                    else 0.0
                ),
            },
            "logged_data": {
                "n_oracle": n_from_logged,
                "coverage": (
                    n_from_logged / logged_dataset.n_samples if logged_dataset else 0.0
                ),
            },
            "fresh_draws": {
                "n_oracle": n_from_fresh,
                "coverage": None,  # Can't compute without knowing total fresh draws
            },
            "total_oracle": len(combined_samples),
            "n_conflicts": len(conflicts),
        }

        if conflicts:
            # Limit to top 10 for metadata size
            oracle_sources_metadata["conflicts"] = conflicts[:10]

        # Create combined dataset using target_policies parameter
        combined_dataset = Dataset(
            samples=combined_samples,
            target_policies=target_policies,
            metadata={"combined_oracle_sources": True},
        )

        return combined_dataset, oracle_sources_metadata

    def _check_distribution_mismatch(
        self,
        calibration_dataset: Dataset,
        evaluation_dataset: Dataset,
        judge_field: str = "judge_score",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Check if calibration and evaluation data have different judge score distributions.

        Uses Kolmogorov-Smirnov test to compare distributions.

        Args:
            calibration_dataset: Dataset used for learning calibration
            evaluation_dataset: Dataset being evaluated
            judge_field: Field containing judge scores
            verbose: Whether to log warnings

        Returns:
            Dict with KS test results and mismatch warnings
        """
        import numpy as np
        from scipy.stats import ks_2samp

        # Extract judge scores from both datasets
        calib_scores = []
        for sample in calibration_dataset.samples:
            score = (
                sample.judge_score
                if judge_field == "judge_score"
                else sample.metadata.get(judge_field)
            )
            if score is not None:
                calib_scores.append(float(score))

        eval_scores = []
        for sample in evaluation_dataset.samples:
            score = (
                sample.judge_score
                if judge_field == "judge_score"
                else sample.metadata.get(judge_field)
            )
            if score is not None:
                eval_scores.append(float(score))

        if not calib_scores or not eval_scores:
            return {
                "ks_statistic": None,
                "p_value": None,
                "warning": "Insufficient data for distribution comparison",
            }

        calib_array = np.array(calib_scores)
        eval_array = np.array(eval_scores)

        # Perform KS test
        ks_stat, p_value = ks_2samp(calib_array, eval_array)

        # Check for significant mismatch (p < 0.05 indicates different distributions)
        has_mismatch = p_value < 0.05

        # Compute distribution statistics for context
        calib_stats = {
            "mean": float(np.mean(calib_array)),
            "std": float(np.std(calib_array)),
            "q25": float(np.percentile(calib_array, 25)),
            "median": float(np.percentile(calib_array, 50)),
            "q75": float(np.percentile(calib_array, 75)),
        }

        eval_stats = {
            "mean": float(np.mean(eval_array)),
            "std": float(np.std(eval_array)),
            "q25": float(np.percentile(eval_array, 25)),
            "median": float(np.percentile(eval_array, 50)),
            "q75": float(np.percentile(eval_array, 75)),
        }

        if has_mismatch and verbose:
            logger.warning(
                f"Distribution mismatch detected (KS test p={p_value:.4f} < 0.05). "
                f"Calibration data may not be representative of evaluation data. "
                f"Calib mean={calib_stats['mean']:.3f}, Eval mean={eval_stats['mean']:.3f}"
            )

        return {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "has_mismatch": has_mismatch,
            "calibration_stats": calib_stats,
            "evaluation_stats": eval_stats,
            "warning": (
                "Calibration data distribution differs significantly from evaluation data"
                if has_mismatch
                else None
            ),
        }
