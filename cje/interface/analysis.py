"""
High-level analysis functions for CJE.

This module provides simple, one-line analysis functions that handle
the complete CJE workflow automatically.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
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
    timestamp_field: Optional[str] = None,
    check_drift: bool = False,
    estimator: str = "auto",
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
    estimator_config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> EstimationResult:
    """
    Analyze policies using logged data and/or fresh draws.

    This high-level function handles:
    - Data loading and validation
    - Automatic reward calibration (judge → oracle mapping)
    - Oracle source combining (pooling labels from multiple sources)
    - Temporal drift detection
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
        timestamp_field: Metadata field containing timestamps (Unix int or ISO string).
            If provided with check_drift=True, enables automatic temporal drift detection
            using Kendall tau correlation over time batches.
        check_drift: Enable temporal drift detection. Requires timestamp_field to be set.
            Computes sequential drift across time batches and adds diagnostics to
            results.metadata["drift_diagnostics"].
        estimator: Estimator type. Options:
            - "auto" (default): Automatically select based on available data
            - "calibrated-ips": Importance sampling (requires logged_data_path with logprobs)
            - "stacked-dr": Doubly robust (requires both logged_data_path and fresh_draws_dir)
            - "direct": On-policy evaluation (requires fresh_draws_dir)
        judge_field: Metadata field containing judge scores (default "judge_score")
        oracle_field: Metadata field containing oracle labels (default "oracle_label")
        estimator_config: Optional configuration dict for the estimator
        verbose: Whether to print progress messages

    Returns:
        EstimationResult with estimates, standard errors, and metadata.

        New metadata fields when using calibration_data_path:
        - results.metadata["oracle_sources"]: Breakdown of oracle labels by source
        - results.metadata["oracle_conflicts"]: Prompts with conflicting oracle values
        - results.metadata["distribution_mismatch"]: KS test results
        - results.metadata["calibration_staleness"]: Time gap warnings

        New metadata fields when using check_drift=True:
        - results.metadata["drift_diagnostics"]: Temporal stability metrics

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

    Example - Drift detection:
        >>> # Monitor judge stability over time
        >>> results = analyze_dataset(
        ...     logged_data_path="logs_q1.jsonl",
        ...     timestamp_field="timestamp",
        ...     check_drift=True
        ... )
        >>> drift = results.metadata["drift_diagnostics"]
        >>> if drift["has_drift"]:
        ...     print(f"Drift detected at batches: {drift['drift_points']}")

    Example - Combined features:
        >>> # Use dedicated calibration + auto-combine + drift detection
        >>> results = analyze_dataset(
        ...     logged_data_path="eval_data.jsonl",           # 100 oracle labels
        ...     fresh_draws_dir="responses/",                  # 200 oracle labels
        ...     calibration_data_path="certified_labels.jsonl", # 500 oracle labels
        ...     combine_oracle_sources=True,                   # Use all 800 labels
        ...     timestamp_field="timestamp",
        ...     check_drift=True,
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
        timestamp_field=timestamp_field,
        check_drift=check_drift,
        estimator=estimator,
        judge_field=judge_field,
        oracle_field=oracle_field,
        estimator_config=estimator_config or {},
        verbose=verbose,
    )
    service = AnalysisService()
    return service.run(cfg)

    # Note: detailed workflow remains implemented in AnalysisService
