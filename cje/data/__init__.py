"""Data loading and preparation utilities.

This module contains:
- Data models: Pydantic models for type safety
- DatasetLoader: Pure data loading functionality
"""

from .models import (
    Sample,
    Dataset,
    EstimationResult,
    InferenceUnavailableError,
    ResultUnits,
)
from .loaders import DatasetLoader
from .validation import (
    validate_direct_data,
)
from .folds import (
    get_fold,
    get_folds_for_prompts,
)
from .fresh_draws import (
    FreshDrawSample,
    FreshDrawDataset,
    fresh_draws_from_dict,
    NormalizationInfo,
)
from .normalization import ScaleDeclaration, ScaleInfo, detect_range

from typing import Optional, List


# Convenience function
def load_dataset_from_jsonl(
    file_path: str,
    target_policies: Optional[List[str]] = None,
    *,
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
    judge_scale: ScaleDeclaration = None,
    oracle_scale: ScaleDeclaration = None,
    strict: bool = False,
    on_invalid: Optional[str] = None,
) -> Dataset:
    """Load Dataset from JSONL file.

    Convenience function using the default loader.
    """
    return DatasetLoader(
        judge_field=judge_field,
        oracle_field=oracle_field,
        judge_scale=judge_scale,
        oracle_scale=oracle_scale,
        strict=strict,
        on_invalid=on_invalid,
    ).load_from_jsonl(file_path, target_policies)


__all__ = [
    # Data loading
    "DatasetLoader",
    # Data models
    "Sample",
    "Dataset",
    "EstimationResult",
    "InferenceUnavailableError",
    "ResultUnits",
    # Fresh draws
    "FreshDrawSample",
    "FreshDrawDataset",
    "fresh_draws_from_dict",
    "NormalizationInfo",
    # Normalization
    "ScaleDeclaration",
    "ScaleInfo",
    "detect_range",
    # Validation
    "validate_direct_data",
    # Fold management
    "get_fold",
    "get_folds_for_prompts",
    # Convenience function
    "load_dataset_from_jsonl",
]
