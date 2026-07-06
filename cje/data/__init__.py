"""Data loading and preparation utilities.

This module contains:
- Data models: Pydantic models for type safety
- DatasetLoader: Pure data loading functionality
"""

from .models import (
    Sample,
    Dataset,
    EstimationResult,
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
from .normalization import ScaleInfo, detect_range

from typing import Optional, List


# Convenience function
def load_dataset_from_jsonl(
    file_path: str, target_policies: Optional[List[str]] = None
) -> Dataset:
    """Load Dataset from JSONL file.

    Convenience function using the default loader.
    """
    return DatasetLoader().load_from_jsonl(file_path, target_policies)


__all__ = [
    # Data loading
    "DatasetLoader",
    # Data models
    "Sample",
    "Dataset",
    "EstimationResult",
    # Fresh draws
    "FreshDrawSample",
    "FreshDrawDataset",
    "fresh_draws_from_dict",
    "NormalizationInfo",
    # Normalization
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
