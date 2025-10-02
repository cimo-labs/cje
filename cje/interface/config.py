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
