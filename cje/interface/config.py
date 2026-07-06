"""Typed configuration models for the CJE interface.

These models provide a stable, validated contract between the CLI and
the analysis service while preserving backward-compatible function APIs.
"""

from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field, field_validator


class AnalysisConfig(BaseModel):
    fresh_draws_dir: Optional[str] = Field(
        None,
        description="Directory with per-policy fresh draw files, or a single "
        "JSONL file whose records carry a target_policy field.",
    )
    fresh_draws_data: Optional[Dict[str, List[Dict[str, Any]]]] = Field(
        None,
        description="In-memory fresh draws data. Dict mapping policy names to lists of records. "
        "Each record needs at minimum: prompt_id, judge_score. Optional: oracle_label, response. "
        "Alternative to fresh_draws_dir for programmatic usage.",
    )
    calibration_data_path: Optional[str] = Field(
        None,
        description="Path to dedicated calibration dataset with oracle labels. "
        "Used to learn judge→oracle mapping separately from evaluation data.",
    )
    combine_oracle_sources: bool = Field(
        True,
        description="Pool oracle labels from all sources (calibration + logged + fresh) "
        "for maximum data efficiency. Set False to use only calibration_data_path.",
    )
    estimator: str = Field(
        "auto",
        description="Estimator name: auto (resolves to direct), direct/calibrated-direct.",
    )
    judge_field: str = Field("judge_score")
    oracle_field: str = Field("oracle_label")
    calibration_covariates: Optional[List[str]] = Field(
        None,
        description="List of metadata field names to use as covariates in two-stage calibration. "
        "E.g., ['response_length', 'domain'] to handle length bias or domain-specific miscalibration. "
        "Only works with calibration_mode='two_stage' or 'auto'.",
    )
    include_response_length: bool = Field(
        False,
        description="Automatically include response length (word count) as a covariate. "
        "Computed as len(response.split()). Requires all samples to have 'response' field. "
        "If True, 'response_length' is prepended to calibration_covariates.",
    )
    estimator_config: Dict[str, Any] = Field(default_factory=dict)
    verbose: bool = Field(False)

    @field_validator("estimator")
    @classmethod
    def normalize_estimator(cls, v: str) -> str:
        return v.strip()
