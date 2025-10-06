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
        target_policies = discover_policies_from_fresh_draws(
            Path(config.fresh_draws_dir)
        )

        if config.verbose:
            logger.info(
                f"Discovered {len(target_policies)} policies: {', '.join(target_policies)}"
            )

        # Load all fresh draws and check for oracle labels to enable calibration
        all_fresh_draws = []
        for policy in target_policies:
            fd = load_fresh_draws_auto(
                Path(config.fresh_draws_dir), policy, verbose=config.verbose
            )
            all_fresh_draws.extend(fd.samples)

        # Check if fresh draws have oracle labels for calibration
        n_with_oracle = sum(1 for s in all_fresh_draws if s.oracle_label is not None)
        oracle_coverage = n_with_oracle / len(all_fresh_draws) if all_fresh_draws else 0

        calibration_result = None
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

            # Learn calibration from fresh draws
            _, calibration_result = calibrate_dataset(
                fresh_dataset,
                judge_field=config.judge_field,
                oracle_field=config.oracle_field,
                enable_cross_fit=True,
                n_folds=5,
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
        for policy in target_policies:
            fd = load_fresh_draws_auto(
                Path(config.fresh_draws_dir), policy, verbose=config.verbose
            )
            estimator_obj.add_fresh_draws(policy, fd)

        results = estimator_obj.fit_and_estimate()

        # Add metadata
        results.metadata["mode"] = "direct"
        results.metadata["estimator"] = chosen_estimator
        results.metadata["target_policies"] = target_policies
        results.metadata["fresh_draws_dir"] = config.fresh_draws_dir
        results.metadata["calibration"] = (
            "from_fresh_draws" if calibration_result else "none"
        )
        if calibration_result:
            results.metadata["oracle_coverage"] = oracle_coverage
        if config.estimator_config:
            results.metadata["estimator_config"] = config.estimator_config

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

        # Load fresh draws for each policy
        for policy in target_policies:
            fd = load_fresh_draws_auto(
                Path(config.fresh_draws_dir), policy, verbose=config.verbose
            )
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
        if config.verbose:
            logger.info(f"Loading dataset from {config.logged_data_path}")

        dataset = load_dataset_from_jsonl(config.logged_data_path)

        if config.verbose:
            logger.info(f"Loaded {dataset.n_samples} samples")
            logger.info(f"Target policies: {', '.join(dataset.target_policies)}")

        calibrated_dataset, calibration_result = self._prepare_rewards(
            dataset, config.judge_field, config.oracle_field, config.verbose
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
            from ..data.fresh_draws import load_fresh_draws_auto

            if not config.fresh_draws_dir:
                raise ValueError(
                    "DR estimators require fresh draws. Provide fresh_draws_dir."
                )

            for policy in sampler.target_policies:
                fd = load_fresh_draws_auto(
                    Path(config.fresh_draws_dir), policy, verbose=config.verbose
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

    def _prepare_rewards(
        self, dataset: Dataset, judge_field: str, oracle_field: str, verbose: bool
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

        calibrated_dataset, cal_result = calibrate_dataset(
            dataset,
            judge_field=judge_field,
            oracle_field=oracle_field,
            enable_cross_fit=True,
            n_folds=5,
        )
        return calibrated_dataset, cal_result
