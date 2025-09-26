"""Data loading and preparation utilities.

This module contains:
- Data models: Pydantic models for type safety
- PrecomputedSampler: Load data with log probs and rewards
- DatasetFactory: SOLID-compliant data loading with optional calibration
- DatasetLoader: Pure data loading functionality
- Reward Utils: Utility functions for calibrated rewards
"""

from .precomputed_sampler import PrecomputedSampler
from .reward_utils import (
    add_rewards_to_existing_data,
)
from .models import (
    Sample,
    Dataset,
    EstimationResult,
    LogProbStatus,
    LogProbResult,
)
from .factory import DatasetFactory, default_factory
from .loaders import DatasetLoader, JsonlDataSource, InMemoryDataSource
from .validation import (
    validate_cje_data,
    validate_for_precomputed_sampler,
)
from .folds import (
    get_fold,
    get_folds_for_prompts,
    get_folds_for_dataset,
    get_folds_with_oracle_balance,
)

from typing import Optional, List


# Convenience function
def load_dataset_from_jsonl(
    file_path: str, target_policies: Optional[List[str]] = None
) -> Dataset:
    """Load Dataset from JSONL file.

    Convenience function using the default factory.
    """
    return default_factory.create_from_jsonl(file_path, target_policies)


__all__ = [
    # Data loading
    "PrecomputedSampler",
    "DatasetFactory",
    "DatasetLoader",
    "default_factory",
    "JsonlDataSource",
    "InMemoryDataSource",
    # Data models
    "Sample",
    "Dataset",
    "EstimationResult",
    "LogProbStatus",
    "LogProbResult",
    # Utilities
    "add_rewards_to_existing_data",
    # Validation
    "validate_cje_data",
    "validate_for_precomputed_sampler",
    # Fold management
    "get_fold",
    "get_folds_for_prompts",
    "get_folds_for_dataset",
    "get_folds_with_oracle_balance",
    # Convenience function
    "load_dataset_from_jsonl",
]
