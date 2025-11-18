"""Typed configuration models for the CJE interface.

These models provide a stable, validated contract between CLI/Hydra and
the analysis service while preserving backward-compatible function APIs.
"""

from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field, field_validator


class AnalysisConfig(BaseModel):
    logged_data_path: Optional[str] = Field(
        None, description="Path to logged data JSONL (from base/production policy)"
    )
    fresh_draws_dir: Optional[str] = Field(
        None, description="Directory with fresh draws from target policies"
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
        description="Estimator name: auto, calibrated-ips, stacked-dr, direct, etc.",
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

    @field_validator("logged_data_path", "fresh_draws_dir")
    @classmethod
    def validate_at_least_one_source(cls, v: Optional[str], info: Any) -> Optional[str]:
        """Ensure at least one data source is provided."""
        # Note: This validation happens after both fields are set
        # We'll do the actual validation in the service
        return v
