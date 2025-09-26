"""Typed configuration models for the CJE interface.

These models provide a stable, validated contract between CLI/Hydra and
the analysis service while preserving backward-compatible function APIs.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class AnalysisConfig(BaseModel):
    dataset_path: str = Field(..., description="Path to JSONL dataset")
    estimator: str = Field(
        "stacked-dr",
        description="Estimator name (see interface.factory for options)",
    )
    judge_field: str = Field("judge_score")
    oracle_field: str = Field("oracle_label")
    estimator_config: Dict[str, Any] = Field(default_factory=dict)
    fresh_draws_dir: Optional[str] = Field(
        None, description="Directory with fresh draws (for DR estimators)"
    )
    verbose: bool = Field(False)

    @field_validator("estimator")
    @classmethod
    def normalize_estimator(cls, v: str) -> str:
        return v.strip()
