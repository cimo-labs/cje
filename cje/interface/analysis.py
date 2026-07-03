"""
High-level analysis functions for CJE.

This module provides simple, one-line analysis functions that handle
the complete CJE workflow automatically.
"""

import logging
from typing import Optional, Dict, Any, List

from ..data.models import EstimationResult
from .config import AnalysisConfig
from .service import AnalysisService

logger = logging.getLogger(__name__)

# Exact migration copy for the removed OPE (IPS/DR) entry point. Single source
# for the API and the CLI; pinned verbatim by test_migration_errors.py.
LOGGED_DATA_PATH_REMOVED_MESSAGE = """\
Off-policy evaluation was removed in cje-eval 0.4.0; 'logged_data_path' is no longer accepted.

CJE is now Direct-mode only: generate fresh draws from each policy, judge them,
and label a small oracle slice.

  * Have fresh draws?  Pass fresh_draws_dir=... or fresh_draws_data=... .
  * Your logged data has judge_score + oracle_label?  It still works as the
    calibration source: pass calibration_data_path="<your logged data>.jsonl".
  * Need IPS/DR from logged propensities?  Pin the frozen OPE line:
        pip install "cje-eval==0.3.*"
    (maintained on the 0.3.x branch; docs at the v0.3.0 tag; requires
    Python <=3.12 — on 3.13 use a 3.12 env for OPE)."""


def analyze_dataset(
    logged_data_path: Optional[str] = None,
    fresh_draws_dir: Optional[str] = None,
    fresh_draws_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
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
    Analyze policies using fresh draws (Direct mode).

    This high-level function handles:
    - Data loading and validation
    - Automatic reward calibration (judge → oracle mapping)
    - Oracle source combining (pooling labels from multiple sources)
    - Complete analysis workflow

    Args:
        logged_data_path: REMOVED in 0.4.0 along with the OPE (IPS/DR) modes.
            Accepted only so that passing a value raises a ValueError with the
            migration guidance. Logged data with judge scores and oracle labels
            still works as the calibration source via calibration_data_path.
        fresh_draws_dir: Directory containing fresh draw response files.
        fresh_draws_data: In-memory alternative to fresh_draws_dir. Dict mapping policy names
            to lists of records. Each record needs: prompt_id, judge_score. Optional: oracle_label.
            Example: {"policy_a": [{"prompt_id": "1", "judge_score": 0.8}, ...], ...}
        calibration_data_path: Path to dedicated calibration dataset with oracle labels.
            Use this to learn judge→oracle mapping from a curated oracle set separate
            from your evaluation data. If combine_oracle_sources=True (default), will
            pool with oracle labels from fresh_draws for maximum efficiency.
        combine_oracle_sources: Whether to pool oracle labels from all sources
            (calibration_data + fresh_draws). Default True for data efficiency.
            Set False to use ONLY calibration_data_path for learning calibration.
            When combining, every source's (judge, oracle) pair enters calibration
            (labels attach to responses); only true duplicates are deduped.
        estimator: Estimator type. Options:
            - "auto" (default): resolves to "direct"
            - "direct" / "calibrated-direct": On-policy evaluation (requires fresh_draws_dir
              or fresh_draws_data)
            Removed OPE names (calibrated-ips, raw-ips, dr-cpo, mrdr, tmle,
            stacked-dr) raise a ValueError pointing at the 0.3.x releases.
        judge_field: Metadata field containing judge scores (default "judge_score")
        oracle_field: Metadata field containing oracle labels (default "oracle_label")
        calibration_covariates: Optional list of metadata field names to use as covariates
            in two-stage reward calibration (e.g., ["response_length", "domain"]).
            Helps handle confounding where judge scores at fixed S have different oracle
            outcomes based on observable features like response length or domain.
            Only works with two_stage or auto calibration mode.
        include_response_length: Automatically include response length (word count) as a covariate.
            Computed as len(response.split()). Requires all samples (fresh draws
            and calibration data) to have a 'response' field. If True, 'response_length' is
            automatically prepended to calibration_covariates. Convenient for handling length bias.
        estimator_config: Optional configuration dict for the estimator.
        verbose: Whether to print progress messages

    Returns:
        EstimationResult with estimates, standard errors, and metadata.

        New metadata fields when using calibration_data_path:
        - results.metadata["oracle_sources"]: Breakdown of oracle labels by source
        - results.metadata["oracle_sources"]["conflicts"]: Cross-source oracle disagreements

    Raises:
        ValueError: If required data is missing for the selected estimator

    Example - Basic usage:
        >>> # Direct mode: Fresh draws only
        >>> results = analyze_dataset(fresh_draws_dir="responses/")

    Example - Dedicated calibration set:
        >>> # Learn calibration from curated oracle set
        >>> results = analyze_dataset(
        ...     fresh_draws_dir="responses/",
        ...     calibration_data_path="human_labels.jsonl",  # 1000 samples, high quality
        ... )
        >>> print(f"Oracle sources: {results.metadata['oracle_sources']}")
    """
    if logged_data_path is not None:
        raise ValueError(LOGGED_DATA_PATH_REMOVED_MESSAGE)

    # Validate that at least one data source is provided
    if fresh_draws_dir is None and fresh_draws_data is None:
        raise ValueError(
            "Must provide at least one of: fresh_draws_dir, fresh_draws_data"
        )

    # Delegate to the AnalysisService with typed config (logged_data_path is
    # accepted above solely to raise the migration error; the config never
    # sees it).
    cfg = AnalysisConfig(
        fresh_draws_dir=fresh_draws_dir,
        fresh_draws_data=fresh_draws_data,
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
