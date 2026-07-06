"""Unified fold assignment for cross-validation.

Core principle: Use prompt_id hashing for stable fold assignment
that survives filtering and works across all components.

All cross-validation in CJE MUST use these functions.
"""

import hashlib
import numpy as np
from typing import List


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
        3
        >>> get_fold("prompt_123", n_folds=10)  # Different for different n_folds
        6
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
