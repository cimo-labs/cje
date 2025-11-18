"""
High-level analysis functions for CJE.

This module provides simple, one-line analysis functions that handle
the complete CJE workflow automatically.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import numpy as np

from ..data.models import Dataset, EstimationResult
from .config import AnalysisConfig
from .service import AnalysisService

logger = logging.getLogger(__name__)


def analyze_dataset(
    logged_data_path: Optional[str] = None,
    fresh_draws_dir: Optional[str] = None,
    calibration_data_path: Optional[str] = None,
    combine_oracle_sources: bool = True,
    estimator: str = "auto",
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
    calibration_covariates: Optional[List[str]] = None,
    include_response_length: bool = False,
    estimator_config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> EstimationResult:
    """
    Analyze policies using logged data and/or fresh draws.

    This high-level function handles:
    - Data loading and validation
    - Automatic reward calibration (judge → oracle mapping)
    - Oracle source combining (pooling labels from multiple sources)
    - Estimator selection and configuration
    - Fresh draw loading for DR/Direct estimators
    - Complete analysis workflow

    Args:
        logged_data_path: Path to logged data JSONL file (responses from base/production policy).
            Required for: IPS mode (must have logprobs), DR mode.
            Optional for: Direct mode (if provided, used for calibration only).
        fresh_draws_dir: Directory containing fresh draw response files.
            Required for: DR mode, Direct mode.
            Optional for: IPS mode (ignored).
        calibration_data_path: Path to dedicated calibration dataset with oracle labels.
            Use this to learn judge→oracle mapping from a curated oracle set separate
            from your evaluation data. If combine_oracle_sources=True (default), will
            pool with oracle labels from logged_data and fresh_draws for maximum efficiency.
        combine_oracle_sources: Whether to pool oracle labels from all sources
            (calibration_data + logged_data + fresh_draws). Default True for data efficiency.
            Set False to use ONLY calibration_data_path for learning calibration.
            Priority order when combining: calibration_data > fresh_draws > logged_data.
        estimator: Estimator type. Options:
            - "auto" (default): Automatically select based on available data
            - "calibrated-ips": Importance sampling (requires logged_data_path with logprobs)
            - "stacked-dr": Doubly robust (requires both logged_data_path and fresh_draws_dir)
            - "direct": On-policy evaluation (requires fresh_draws_dir)
        judge_field: Metadata field containing judge scores (default "judge_score")
        oracle_field: Metadata field containing oracle labels (default "oracle_label")
        calibration_covariates: Optional list of metadata field names to use as covariates
            in two-stage reward calibration (e.g., ["response_length", "domain"]).
            Helps handle confounding where judge scores at fixed S have different oracle
            outcomes based on observable features like response length or domain.
            Only works with two_stage or auto calibration mode.
        include_response_length: Automatically include response length (word count) as a covariate.
            Computed as len(response.split()). Requires all samples (logged data, fresh draws,
            and calibration data) to have a 'response' field. If True, 'response_length' is
            automatically prepended to calibration_covariates. Convenient for handling length bias.
        estimator_config: Optional configuration dict for the estimator
        verbose: Whether to print progress messages

    Returns:
        EstimationResult with estimates, standard errors, and metadata.

        New metadata fields when using calibration_data_path:
        - results.metadata["oracle_sources"]: Breakdown of oracle labels by source
        - results.metadata["oracle_conflicts"]: Prompts with conflicting oracle values
        - results.metadata["distribution_mismatch"]: KS test results

    Raises:
        ValueError: If required data is missing for the selected estimator

    Example - Basic usage:
        >>> # IPS mode: Logged data only
        >>> results = analyze_dataset(logged_data_path="logs.jsonl")

        >>> # DR mode: Both logged data and fresh draws
        >>> results = analyze_dataset(
        ...     logged_data_path="logs.jsonl",
        ...     fresh_draws_dir="responses/"
        ... )

        >>> # Direct mode: Fresh draws only
        >>> results = analyze_dataset(
        ...     fresh_draws_dir="responses/",
        ...     estimator="direct"
        ... )

    Example - Dedicated calibration set:
        >>> # Learn calibration from curated oracle set
        >>> results = analyze_dataset(
        ...     logged_data_path="production_logs.jsonl",
        ...     calibration_data_path="human_labels.jsonl",  # 1000 samples, high quality
        ...     estimator="calibrated-ips"
        ... )
        >>> print(f"Oracle sources: {results.metadata['oracle_sources']}")

    Example - Combined oracle sources:
        >>> # Pool oracle labels from multiple sources
        >>> results = analyze_dataset(
        ...     logged_data_path="eval_data.jsonl",           # 100 oracle labels
        ...     fresh_draws_dir="responses/",                  # 200 oracle labels
        ...     calibration_data_path="certified_labels.jsonl", # 500 oracle labels
        ...     combine_oracle_sources=True,                   # Use all 800 labels
        ...     verbose=True
        ... )
    """
    # Validate that at least one data source is provided
    if logged_data_path is None and fresh_draws_dir is None:
        raise ValueError(
            "Must provide at least one of: logged_data_path, fresh_draws_dir"
        )

    # Delegate to the AnalysisService with typed config
    cfg = AnalysisConfig(
        logged_data_path=logged_data_path,
        fresh_draws_dir=fresh_draws_dir,
        calibration_data_path=calibration_data_path,
        combine_oracle_sources=combine_oracle_sources,
        estimator=estimator,
        judge_field=judge_field,
        oracle_field=oracle_field,
        calibration_covariates=calibration_covariates,
        include_response_length=include_response_length,
        estimator_config=estimator_config or {},
        verbose=verbose,
    )
    service = AnalysisService()
    return service.run(cfg)

    # Note: detailed workflow remains implemented in AnalysisService
