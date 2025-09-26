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
    dataset_path: str,
    estimator: str = "stacked-dr",
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
    estimator_config: Optional[Dict[str, Any]] = None,
    fresh_draws_dir: Optional[str] = None,
    verbose: bool = False,
) -> EstimationResult:
    """
    Analyze a CJE dataset with automatic workflow orchestration.

    This high-level function handles:
    - Data loading and validation
    - Automatic reward handling (pre-computed, oracle direct, or calibration)
    - Estimator selection and configuration
    - Fresh draw loading for DR estimators
    - Complete analysis workflow

    Args:
        dataset_path: Path to JSONL dataset file
        estimator: Estimator type (default "stacked-dr"; also "calibrated-ips", "raw-ips", "dr-cpo", "mrdr", "tmle")
        judge_field: Metadata field containing judge scores
        oracle_field: Metadata field containing oracle labels
        estimator_config: Optional configuration dict for the estimator
        fresh_draws_dir: Directory containing fresh draw response files (for DR)
        verbose: Whether to print progress messages

    Returns:
        EstimationResult with estimates, standard errors, and metadata

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If dataset is invalid or estimation fails

    Example:
        >>> # Simple usage
        >>> results = analyze_dataset("my_data.jsonl")
        >>> print(f"Best estimate: {results.estimates.max():.3f}")

        >>> # Advanced usage with DR
        >>> results = analyze_dataset(
        ...     "my_data.jsonl",
        ...     estimator="dr-cpo",
        ...     estimator_config={"n_folds": 10},
        ...     fresh_draws_dir="responses/"
        ... )
    """
    # Delegate to the AnalysisService with typed config
    cfg = AnalysisConfig(
        dataset_path=dataset_path,
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
