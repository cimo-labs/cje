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
        EstimationResult with estimates, standard errors, and metadata

    Raises:
        ValueError: If required data is missing for the selected estimator

    Example:
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
    """
    # Validate that at least one data source is provided
    if logged_data_path is None and fresh_draws_dir is None:
        raise ValueError(
            "Must provide at least one of: logged_data_path, fresh_draws_dir"
        )

    # Delegate to the AnalysisService with typed config
    cfg = AnalysisConfig(
        logged_data_path=logged_data_path,
        estimator=estimator,
        judge_field=judge_field,
        oracle_field=oracle_field,
        estimator_config=estimator_config or {},
        fresh_draws_dir=fresh_draws_dir,
        verbose=verbose,
    )
    service = AnalysisService()
    return service.run(cfg)

    # Note: detailed workflow remains implemented in AnalysisService
