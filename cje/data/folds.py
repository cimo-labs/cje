"""Unified fold assignment for cross-validation.

Core principle: Use prompt_id hashing for stable fold assignment
that survives filtering and works across all components.

All cross-validation in CJE MUST use these functions.
"""

import hashlib
import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Dataset


def get_fold(prompt_id: str, n_folds: int = 5, seed: int = 42) -> int:
    """Get fold assignment for a single prompt_id.

    This is THE authoritative way to assign folds in CJE.
    Uses stable hashing that:
    - Survives sample filtering
    - Works with fresh draws (same prompt_id → same fold)
    - Ensures consistency across all components

    Args:
        prompt_id: Unique identifier for the prompt
        n_folds: Number of folds for cross-validation
        seed: Random seed for reproducibility

    Returns:
        Fold index in [0, n_folds)

    Example:
        >>> get_fold("prompt_123")  # Always returns same fold
        2
        >>> get_fold("prompt_123", n_folds=10)  # Different for different n_folds
        7
    """
    if not prompt_id:
        raise ValueError("prompt_id cannot be empty")
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")

    hash_input = f"{prompt_id}-{seed}-{n_folds}".encode()
    hash_bytes = hashlib.blake2b(hash_input, digest_size=8).digest()
    return int.from_bytes(hash_bytes, "big") % n_folds


def get_folds_for_prompts(
    prompt_ids: List[str], n_folds: int = 5, seed: int = 42
) -> np.ndarray:
    """Get fold assignments for multiple prompt_ids.

    Args:
        prompt_ids: List of prompt identifiers
        n_folds: Number of folds
        seed: Random seed

    Returns:
        Array of fold indices, shape (len(prompt_ids),)
    """
    if not prompt_ids:
        return np.array([], dtype=int)

    return np.array([get_fold(pid, n_folds, seed) for pid in prompt_ids])


def get_folds_for_dataset(
    dataset: "Dataset", n_folds: int = 5, seed: int = 42
) -> np.ndarray:
    """Get fold assignments for all samples in a dataset.

    Args:
        dataset: Dataset with samples containing prompt_ids
        n_folds: Number of folds
        seed: Random seed

    Returns:
        Array of fold indices aligned with dataset.samples
    """
    prompt_ids = [s.prompt_id for s in dataset.samples]
    return get_folds_for_prompts(prompt_ids, n_folds, seed)


def get_folds_with_oracle_balance(
    prompt_ids: List[str], oracle_mask: np.ndarray, n_folds: int = 5, seed: int = 42
) -> np.ndarray:
    """Get folds with balanced oracle sample distribution.

    Ensures oracle samples are evenly distributed across folds
    (important for small oracle subsets). Unlabeled samples
    use standard hash-based assignment.

    Args:
        prompt_ids: All prompt identifiers
        oracle_mask: Boolean mask indicating oracle samples
        n_folds: Number of folds
        seed: Random seed

    Returns:
        Array of fold indices with balanced oracle distribution

    Note:
        This is primarily for JudgeCalibrator backward compatibility.
        New code should use get_folds_for_prompts() directly.
    """
    n = len(prompt_ids)
    if n == 0:
        return np.array([], dtype=int)

    if len(oracle_mask) != n:
        raise ValueError(
            f"oracle_mask length ({len(oracle_mask)}) must match "
            f"prompt_ids length ({n})"
        )

    folds = np.zeros(n, dtype=int)

    # Oracle samples: round-robin for perfect balance
    oracle_indices = np.where(oracle_mask)[0]
    if len(oracle_indices) > 0:
        # Shuffle oracle indices for randomization
        rng = np.random.RandomState(seed)
        oracle_indices = oracle_indices.copy()
        rng.shuffle(oracle_indices)

        for i, idx in enumerate(oracle_indices):
            folds[idx] = i % n_folds

    # Unlabeled samples: standard hash-based
    unlabeled = ~oracle_mask
    if np.any(unlabeled):
        unlabeled_ids = [prompt_ids[i] for i in range(n) if unlabeled[i]]
        unlabeled_folds = get_folds_for_prompts(unlabeled_ids, n_folds, seed)
        folds[unlabeled] = unlabeled_folds

    return folds
