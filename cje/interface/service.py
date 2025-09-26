"""High-level analysis service.

Encapsulates the end-to-end workflow and uses the estimator registry.
The public API still exposes analyze_dataset(...) for simplicity.
"""

from typing import Any, Dict, Optional
import logging
from pathlib import Path

from .config import AnalysisConfig
from .factory import create_estimator
from ..data import load_dataset_from_jsonl
from ..data.models import Dataset, EstimationResult
from ..data.precomputed_sampler import PrecomputedSampler
from ..calibration import calibrate_dataset

logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self) -> None:
        pass

    def run(self, config: AnalysisConfig) -> EstimationResult:
        if config.verbose:
            logger.info(f"Loading dataset from {config.dataset_path}")

        dataset = load_dataset_from_jsonl(config.dataset_path)

        if config.verbose:
            logger.info(f"Loaded {dataset.n_samples} samples")
            logger.info(f"Target policies: {', '.join(dataset.target_policies)}")

        calibrated_dataset, calibration_result = self._prepare_rewards(
            dataset, config.judge_field, config.oracle_field, config.verbose
        )

        sampler = PrecomputedSampler(calibrated_dataset)
        if config.verbose:
            logger.info(f"Valid samples after filtering: {sampler.n_valid_samples}")

        # Determine estimator (support 'auto' default)
        chosen_estimator = (
            config.estimator.lower() if config.estimator else "calibrated-ips"
        )
        if chosen_estimator == "auto":
            chosen_estimator = (
                "stacked-dr" if config.fresh_draws_dir else "calibrated-ips"
            )

        estimator_obj = create_estimator(
            chosen_estimator,
            sampler,
            config.estimator_config,
            calibration_result,
            config.verbose,
        )

        # DR estimators require explicit fresh draws directory
        if chosen_estimator in (
            "dr-cpo",
            "oc-dr-cpo",
            "mrdr",
            "tmle",
            "tr-cpo",
            "tr-cpo-e",
            "stacked-dr",
        ):
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
        results.metadata["dataset_path"] = config.dataset_path
        results.metadata["estimator"] = chosen_estimator
        results.metadata["target_policies"] = list(sampler.target_policies)
        if config.estimator_config:
            results.metadata["estimator_config"] = config.estimator_config
        results.metadata["judge_field"] = config.judge_field
        results.metadata["oracle_field"] = config.oracle_field

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
