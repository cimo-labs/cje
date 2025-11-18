"""High-level analysis service.

Encapsulates the end-to-end workflow and uses the estimator registry.
The public API still exposes analyze_dataset(...) for simplicity.
"""

from typing import Any, Dict, List, Optional
import logging
from pathlib import Path

from .config import AnalysisConfig
from .factory import create_estimator
from .mode_detection import detect_analysis_mode
from ..data import load_dataset_from_jsonl
from ..data.models import Dataset, EstimationResult
from ..data.precomputed_sampler import PrecomputedSampler
from ..calibration import calibrate_dataset

logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self) -> None:
        pass

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

    def run(self, config: AnalysisConfig) -> EstimationResult:
        # Determine estimator choice early
        chosen_estimator = config.estimator.lower() if config.estimator else "auto"

        # Check if we're in Direct-only mode (no logged data)
        is_direct_only = (
            config.logged_data_path is None and config.fresh_draws_dir is not None
        )

        if is_direct_only:
            # Direct-only mode: fresh draws without logged data
            if chosen_estimator == "auto":
                chosen_estimator = "direct"
            elif chosen_estimator not in {"direct", "calibrated-direct"}:
                raise ValueError(
                    f"Without logged_data_path, only Direct mode is supported. "
                    f"Got estimator={chosen_estimator}. Either provide logged_data_path "
                    f"or use estimator='direct'."
                )

            return self._run_direct_only(config, chosen_estimator)

        # IPS/DR mode: requires logged data
        if config.logged_data_path is None:
            raise ValueError(
                "Must provide logged_data_path for IPS/DR modes, or "
                "provide fresh_draws_dir for Direct mode"
            )

        return self._run_with_logged_data(config, chosen_estimator)

    def _run_direct_only(
        self, config: AnalysisConfig, chosen_estimator: str
    ) -> EstimationResult:
        """Run Direct mode without logged data (fresh draws only)."""
        from ..data.fresh_draws import (
            discover_policies_from_fresh_draws,
            load_fresh_draws_auto,
        )
        from ..data.models import Dataset, Sample

        if config.verbose:
            logger.info("Direct mode: Using fresh draws only (no logged data)")

        # Discover policies from fresh draws directory
        if config.fresh_draws_dir is None:
            raise ValueError("fresh_draws_dir is required for Direct mode")
        target_policies = discover_policies_from_fresh_draws(
            Path(config.fresh_draws_dir)
        )

        if config.verbose:
            logger.info(
                f"Discovered {len(target_policies)} policies: {', '.join(target_policies)}"
            )

        # Load all fresh draws for calibration
        fresh_draws_dict = {}
        all_fresh_draws = []
        fresh_draws_path = Path(config.fresh_draws_dir)  # Already checked above
        for policy in target_policies:
            fd = load_fresh_draws_auto(fresh_draws_path, policy, verbose=config.verbose)
            fresh_draws_dict[policy] = fd
            all_fresh_draws.extend(fd.samples)

        # NEW: Handle calibration_data_path and oracle combining
        oracle_sources_metadata = None
        calibration_result = None
        calibration_dataset_for_rewards = None
        covariate_names = None  # Initialize before if/else branches

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

            # Combine or use exclusively based on combine_oracle_sources flag
            if config.combine_oracle_sources:
                if config.verbose:
                    logger.info(
                        "Combining oracle sources from calibration data and fresh draws"
                    )

                # Combine oracle sources (no logged data in Direct mode)
                combined_dataset, oracle_sources_metadata = (
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
                calibration_dataset_for_rewards = combined_dataset
            else:
                # Use ONLY calibration_data_path for learning calibration
                if config.verbose:
                    logger.info(
                        "Using calibration data exclusively (combine_oracle_sources=False)"
                    )
                calibration_dataset_for_rewards = calibration_dataset

                # Still track metadata
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

            # Build covariate list (with validation for include_response_length)
            covariate_names = self._build_covariate_list(
                config, calibration_dataset_for_rewards
            )

            # Learn calibration from combined/calibration-only dataset
            _, calibration_result = calibrate_dataset(
                calibration_dataset_for_rewards,
                judge_field=config.judge_field,
                oracle_field=config.oracle_field,
                enable_cross_fit=True,
                n_folds=5,
                covariate_names=covariate_names,
            )

        else:
            # Original behavior: Check if fresh draws have oracle labels for calibration
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

                # Convert FreshDrawSample to Sample for calibration
                calibration_samples = []
                for fd_sample in all_fresh_draws:
                    # Create dummy Sample with required fields for calibration
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

                # Create temporary dataset from fresh draws for calibration
                fresh_dataset = Dataset(
                    samples=calibration_samples, target_policies=target_policies
                )

                # Build covariate list (with validation for include_response_length)
                covariate_names = self._build_covariate_list(config, fresh_dataset)

                # Learn calibration from fresh draws
                _, calibration_result = calibrate_dataset(
                    fresh_dataset,
                    judge_field=config.judge_field,
                    oracle_field=config.oracle_field,
                    enable_cross_fit=True,
                    n_folds=5,
                    covariate_names=covariate_names,
                )

        from ..estimators.direct_method import CalibratedDirectEstimator

        estimator_obj = CalibratedDirectEstimator(
            target_policies=target_policies,
            reward_calibrator=(
                calibration_result.calibrator if calibration_result else None
            ),
            run_diagnostics=True,
            oua_jackknife=calibration_result is not None,  # Include OUA if calibrated
            **config.estimator_config,
        )

        # Load fresh draws for each policy
        fresh_draws_path = Path(config.fresh_draws_dir)  # Already checked above

        # Compute covariates for fresh draws if needed
        from ..data.fresh_draws import compute_response_covariates

        for policy in target_policies:
            fd = load_fresh_draws_auto(fresh_draws_path, policy, verbose=config.verbose)

            # Compute covariates if calibrator expects them
            if covariate_names:
                fd = compute_response_covariates(fd, covariate_names=covariate_names)

            estimator_obj.add_fresh_draws(policy, fd)

        results = estimator_obj.fit_and_estimate()

        # Add metadata
        results.metadata["mode"] = "direct"
        results.metadata["estimator"] = chosen_estimator
        results.metadata["target_policies"] = target_policies
        results.metadata["fresh_draws_dir"] = config.fresh_draws_dir

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
            if calibration_result and not config.calibration_data_path:
                # Only set oracle_coverage for fresh-draws-only calibration
                oracle_coverage = (
                    sum(1 for s in all_fresh_draws if s.oracle_label is not None)
                    / len(all_fresh_draws)
                    if all_fresh_draws
                    else 0
                )
                results.metadata["oracle_coverage"] = oracle_coverage

        if config.estimator_config:
            results.metadata["estimator_config"] = config.estimator_config

        # NEW: Add oracle sources metadata if available
        if oracle_sources_metadata:
            results.metadata["oracle_sources"] = oracle_sources_metadata

        # Add mode_selection metadata
        results.metadata["mode_selection"] = {
            "mode": "direct",
            "estimator": chosen_estimator,
            "logprob_coverage": 0.0,  # Direct-only mode has no logged data
            "has_fresh_draws": True,
            "has_logged_data": False,
            "reason": "Direct-only mode: Fresh draws without logged data",
        }

        return results

    def _run_direct_with_calibration(
        self,
        config: AnalysisConfig,
        chosen_estimator: str,
        target_policies: List[str],
        calibration_result: Optional[Any],
    ) -> EstimationResult:
        """Run Direct mode with logged data for calibration."""
        from ..data.fresh_draws import load_fresh_draws_auto
        from ..estimators.direct_method import CalibratedDirectEstimator

        if not config.fresh_draws_dir:
            raise ValueError(
                "Direct mode requires fresh_draws_dir. "
                "Provide fresh draws or use IPS/DR mode."
            )

        if config.verbose:
            logger.info(
                "Direct mode: Using logged data for calibration, fresh draws for evaluation"
            )

        # Create Direct estimator with calibrator from logged data
        estimator_obj = CalibratedDirectEstimator(
            target_policies=target_policies,
            reward_calibrator=(
                calibration_result.calibrator if calibration_result else None
            ),
            run_diagnostics=True,
            oua_jackknife=True,  # Include oracle uncertainty
            **config.estimator_config,
        )

        # Build covariate list (need to compute for fresh draws if present)
        covariate_names = self._build_covariate_list(config, dataset=None)

        # Load fresh draws for each policy
        fresh_draws_path = Path(config.fresh_draws_dir)  # Already checked above

        # Compute covariates for fresh draws if needed
        from ..data.fresh_draws import compute_response_covariates

        for policy in target_policies:
            fd = load_fresh_draws_auto(fresh_draws_path, policy, verbose=config.verbose)

            # Compute covariates if needed
            if covariate_names:
                fd = compute_response_covariates(fd, covariate_names=covariate_names)

            estimator_obj.add_fresh_draws(policy, fd)

        results = estimator_obj.fit_and_estimate()

        # Add metadata
        results.metadata["mode"] = "direct"
        results.metadata["logged_data_path"] = config.logged_data_path
        results.metadata["estimator"] = chosen_estimator
        results.metadata["target_policies"] = target_policies
        results.metadata["fresh_draws_dir"] = config.fresh_draws_dir
        results.metadata["calibration"] = "from_logged_data"
        results.metadata["judge_field"] = config.judge_field
        results.metadata["oracle_field"] = config.oracle_field
        if config.estimator_config:
            results.metadata["estimator_config"] = config.estimator_config

        # Add mode_selection metadata
        # Note: This is Direct mode with logged data for calibration
        # Logprob coverage is not computed here (would need to scan dataset)
        results.metadata["mode_selection"] = {
            "mode": "direct",
            "estimator": chosen_estimator,
            "logprob_coverage": None,  # Not computed for Direct mode
            "has_fresh_draws": True,
            "has_logged_data": True,
            "reason": "Direct mode: Using logged data for calibration, fresh draws for evaluation",
        }

        return results

    def _run_with_logged_data(
        self, config: AnalysisConfig, chosen_estimator: str
    ) -> EstimationResult:
        """Run IPS/DR/Direct mode with logged data."""
        if config.logged_data_path is None:
            raise ValueError("logged_data_path is required")

        if config.verbose:
            logger.info(f"Loading dataset from {config.logged_data_path}")

        dataset = load_dataset_from_jsonl(config.logged_data_path)

        if config.verbose:
            logger.info(f"Loaded {dataset.n_samples} samples")
            logger.info(f"Target policies: {', '.join(dataset.target_policies)}")

        # NEW: Handle calibration_data_path and oracle combining
        oracle_sources_metadata = None
        calibration_dataset_for_rewards = dataset  # Default: use logged data

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

            # If combining oracle sources, load fresh draws early and combine
            if config.combine_oracle_sources:
                if config.verbose:
                    logger.info("Combining oracle sources from all available data")

                # Load fresh draws if available
                fresh_draws_dict = None
                if config.fresh_draws_dir:
                    fresh_draws_dict = self._load_all_fresh_draws(config)

                # Combine oracle sources
                combined_dataset, oracle_sources_metadata = (
                    self._combine_oracle_sources(
                        calibration_dataset,
                        dataset,
                        fresh_draws_dict,
                        dataset.target_policies,
                        config.judge_field,
                        config.oracle_field,
                        config.verbose,
                    )
                )
                calibration_dataset_for_rewards = combined_dataset
            else:
                # Use ONLY calibration_data_path for learning calibration
                if config.verbose:
                    logger.info(
                        "Using calibration data exclusively (combine_oracle_sources=False)"
                    )
                calibration_dataset_for_rewards = calibration_dataset

                # Still track metadata
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

            # Check for distribution mismatch between calibration and evaluation data
            if config.verbose:
                logger.info(
                    "Checking calibration data distribution vs. evaluation data"
                )

            distribution_check = self._check_distribution_mismatch(
                calibration_dataset, dataset, config.judge_field, config.verbose
            )

            # Add to metadata
            if oracle_sources_metadata:
                oracle_sources_metadata["distribution_mismatch"] = distribution_check

        # Build covariate list
        covariate_names = self._build_covariate_list(
            config, calibration_dataset_for_rewards
        )

        calibrated_dataset, calibration_result = self._prepare_rewards(
            calibration_dataset_for_rewards,
            config.judge_field,
            config.oracle_field,
            covariate_names,
            config.verbose,
        )

        # Auto mode detection
        detected_mode: Optional[str] = None
        logprob_coverage: float = 0.0
        mode_explanation: str = ""
        is_auto_mode: bool = chosen_estimator == "auto"

        if is_auto_mode:
            mode, mode_explanation, logprob_coverage = detect_analysis_mode(
                calibrated_dataset, config.fresh_draws_dir
            )
            detected_mode = mode
            if config.verbose:
                logger.info(f"Mode detection: {mode_explanation}")

            # Map mode to default estimator
            mode_to_estimator = {
                "ips": "calibrated-ips",
                "dr": "stacked-dr",
                "direct": "direct",
            }
            chosen_estimator = mode_to_estimator[mode]

            if config.verbose:
                logger.info(f"Selected estimator: {chosen_estimator} for {mode} mode")
        else:
            if config.verbose:
                logger.info(f"Using explicitly specified estimator: {chosen_estimator}")

            # Infer mode from estimator if manually specified
            if chosen_estimator in {"direct", "calibrated-direct"}:
                detected_mode = "direct"
            elif chosen_estimator in {"calibrated-ips", "raw-ips"}:
                detected_mode = "ips"
            elif chosen_estimator in {
                "stacked-dr",
                "dr-cpo",
                "oc-dr-cpo",
                "mrdr",
                "tmle",
                "tr-cpo",
                "tr-cpo-e",
            }:
                detected_mode = "dr"

        # Branch: Direct mode vs IPS/DR mode
        if chosen_estimator in {"direct", "calibrated-direct"}:
            return self._run_direct_with_calibration(
                config, chosen_estimator, dataset.target_policies, calibration_result
            )

        # IPS/DR mode: create sampler
        sampler = PrecomputedSampler(calibrated_dataset)
        if config.verbose:
            logger.info(f"Valid samples after filtering: {sampler.n_valid_samples}")

        estimator_obj = create_estimator(
            chosen_estimator,
            sampler,
            config.estimator_config,
            calibration_result,
            config.verbose,
        )

        # Estimators that can use fresh draws
        dr_estimators = {
            "dr-cpo",
            "oc-dr-cpo",
            "mrdr",
            "tmle",
            "tr-cpo",
            "tr-cpo-e",
            "stacked-dr",
        }

        # DR estimators require fresh draws
        if chosen_estimator in dr_estimators:
            from ..data.fresh_draws import (
                load_fresh_draws_auto,
                compute_response_covariates,
            )

            if not config.fresh_draws_dir:
                raise ValueError(
                    "DR estimators require fresh draws. Provide fresh_draws_dir."
                )

            for policy in sampler.target_policies:
                fd = load_fresh_draws_auto(
                    Path(config.fresh_draws_dir), policy, verbose=config.verbose
                )

                # Compute covariates if needed
                if covariate_names:
                    fd = compute_response_covariates(
                        fd, covariate_names=covariate_names
                    )

                estimator_obj.add_fresh_draws(policy, fd)

        results: EstimationResult = estimator_obj.fit_and_estimate()

        # Minimal metadata enrichment
        results.metadata["logged_data_path"] = config.logged_data_path
        results.metadata["estimator"] = chosen_estimator
        if detected_mode:
            results.metadata["mode"] = detected_mode
        results.metadata["target_policies"] = list(sampler.target_policies)
        if config.estimator_config:
            results.metadata["estimator_config"] = config.estimator_config
        results.metadata["judge_field"] = config.judge_field
        results.metadata["oracle_field"] = config.oracle_field

        # NEW: Add oracle sources metadata if available
        if oracle_sources_metadata:
            results.metadata["oracle_sources"] = oracle_sources_metadata

        # Add mode_selection metadata
        results.metadata["mode_selection"] = {
            "mode": detected_mode or "unknown",
            "estimator": chosen_estimator,
            "logprob_coverage": logprob_coverage if is_auto_mode else None,
            "has_fresh_draws": config.fresh_draws_dir is not None,
            "has_logged_data": True,
            "reason": (
                mode_explanation
                if is_auto_mode
                else f"Explicitly specified: {chosen_estimator}"
            ),
        }

        return results

    def _load_all_fresh_draws(self, config: AnalysisConfig) -> Dict[str, Any]:
        """Load fresh draws for all policies (helper for oracle combining)."""
        from ..data.fresh_draws import (
            discover_policies_from_fresh_draws,
            load_fresh_draws_auto,
        )

        if config.fresh_draws_dir is None:
            raise ValueError("fresh_draws_dir is required")

        fresh_draws_path = Path(config.fresh_draws_dir)
        policies = discover_policies_from_fresh_draws(fresh_draws_path)
        fresh_draws = {}
        for policy in policies:
            fd = load_fresh_draws_auto(fresh_draws_path, policy, verbose=config.verbose)
            fresh_draws[policy] = fd
        return fresh_draws

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
        Combine oracle labels from multiple sources with priority handling.

        Priority order (highest to lowest):
        1. calibration_dataset
        2. fresh_draws_per_policy
        3. logged_dataset

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
        import numpy as np

        # Track oracle samples by prompt_id: {prompt_id: (oracle, source, judge_score)}
        oracle_map: Dict[str, tuple[float, str, float]] = {}
        conflicts: List[Dict[str, Any]] = []

        # Priority 3: Logged data (lowest priority)
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
                        oracle_map[sample.prompt_id] = (
                            float(oracle_val),
                            "logged_data",
                            float(judge_val),
                        )
                        n_from_logged += 1

        # Priority 2: Fresh draws
        n_from_fresh = 0
        if fresh_draws_per_policy:
            for policy, fd_dataset in fresh_draws_per_policy.items():
                for fd_sample in fd_dataset.samples:
                    if (
                        fd_sample.oracle_label is not None
                        and fd_sample.judge_score is not None
                    ):
                        if fd_sample.prompt_id in oracle_map:
                            old_oracle, old_source, _ = oracle_map[fd_sample.prompt_id]
                            # Check for conflict
                            if abs(old_oracle - fd_sample.oracle_label) > 0.05:
                                conflicts.append(
                                    {
                                        "prompt_id": fd_sample.prompt_id,
                                        "old_value": float(old_oracle),
                                        "old_source": old_source,
                                        "new_value": float(fd_sample.oracle_label),
                                        "new_source": "fresh_draws",
                                        "difference": abs(
                                            old_oracle - fd_sample.oracle_label
                                        ),
                                    }
                                )
                        oracle_map[fd_sample.prompt_id] = (
                            float(fd_sample.oracle_label),
                            "fresh_draws",
                            float(fd_sample.judge_score),
                        )
                        n_from_fresh += 1

        # Priority 1: Calibration data (highest - overwrites all)
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
                        if sample.prompt_id in oracle_map:
                            old_oracle, old_source, _ = oracle_map[sample.prompt_id]
                            # Check for conflict
                            if abs(old_oracle - oracle_val) > 0.05:
                                conflicts.append(
                                    {
                                        "prompt_id": sample.prompt_id,
                                        "old_value": float(old_oracle),
                                        "old_source": old_source,
                                        "new_value": float(oracle_val),
                                        "new_source": "calibration_data",
                                        "difference": abs(old_oracle - oracle_val),
                                    }
                                )
                        oracle_map[sample.prompt_id] = (
                            float(oracle_val),
                            "calibration_data",
                            float(judge_val),
                        )
                        n_from_calib += 1

        # Log conflicts if any
        if conflicts and verbose:
            logger.warning(
                f"Found {len(conflicts)} prompts with conflicting oracle labels (diff > 0.05). "
                f"Using priority: calibration_data > fresh_draws > logged_data"
            )
            # Log top 5 conflicts
            for conflict in conflicts[:5]:
                logger.debug(
                    f"  {conflict['prompt_id']}: {conflict['old_source']}={conflict['old_value']:.3f} "
                    f"→ {conflict['new_source']}={conflict['new_value']:.3f} (diff={conflict['difference']:.3f})"
                )

        # Build combined dataset with all unique oracle samples
        combined_samples = []
        # Use target policies parameter - cast to Dict[str, Optional[float]] for variance
        target_policies_dict: Dict[str, Optional[float]] = {
            policy: -1.0 for policy in target_policies
        }

        for prompt_id, (oracle_val, source, judge_val) in oracle_map.items():
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
                f"Combined oracle sources: {len(combined_samples)} total "
                f"(calib={n_from_calib}, fresh={n_from_fresh}, logged={n_from_logged})"
            )

        # Build metadata
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
            "priority_order": ["calibration_data", "fresh_draws", "logged_data"],
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

    def _prepare_rewards(
        self,
        dataset: Dataset,
        judge_field: str,
        oracle_field: str,
        covariate_names: Optional[List[str]],
        verbose: bool,
    ) -> tuple[Dataset, Optional[Any]]:
        n_total = len(dataset.samples)
        rewards_exist = sum(1 for s in dataset.samples if s.reward is not None)

        if rewards_exist == n_total and n_total > 0:
            if verbose:
                logger.info("Using pre-computed rewards for all samples")
            return dataset, None
        elif 0 < rewards_exist < n_total:
            logger.warning(
                f"Detected partial rewards ({rewards_exist}/{n_total}). "
                "Recalibrating to produce consistent rewards for all samples."
            )

        if verbose:
            logger.info("Calibrating judge scores with oracle labels")
            if covariate_names:
                logger.info(f"Using covariates: {covariate_names}")

        calibrated_dataset, cal_result = calibrate_dataset(
            dataset,
            judge_field=judge_field,
            oracle_field=oracle_field,
            enable_cross_fit=True,
            n_folds=5,
            covariate_names=covariate_names,
        )
        return calibrated_dataset, cal_result
