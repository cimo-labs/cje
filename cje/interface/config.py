"""Typed configuration models for the CJE interface.

These models provide a stable, validated contract between CLI/Hydra and
the analysis service while preserving backward-compatible function APIs.
"""

from typing import Any, Dict, Optional
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
    timestamp_field: Optional[str] = Field(
        None,
        description="Metadata field containing timestamps (Unix int or ISO string) "
        "for temporal drift detection.",
    )
    check_drift: bool = Field(
        False,
        description="Enable temporal drift detection. Requires timestamp_field to be set.",
    )
    estimator: str = Field(
        "auto",
        description="Estimator name: auto, calibrated-ips, stacked-dr, direct, etc.",
    )
    judge_field: str = Field("judge_score")
    oracle_field: str = Field("oracle_label")
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
