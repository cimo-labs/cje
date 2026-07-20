"""Stable hash-based fold assignment utilities.

`get_fold` / `get_folds_for_prompts` compute deterministic
`hash(prompt_id) % n_folds` assignments: stable under filtering (a
prompt_id maps to the same fold no matter what else is in the data), useful
for generic reproducible splits.

They do NOT predict calibration folds. Since 0.6.0,
`JudgeCalibrator.fit_cv` assigns whole oracle prompt clusters to folds via
a seeded-hash sort with balanced round-robin assignment, auto-reducing the
fold count when labeled clusters are scarce, and records the assignments
actually used (`CalibrationResult.fold_ids`,
`calibration_info["n_folds"]`). Calibration fold membership depends on the
whole oracle cluster set, so read the recorded assignments instead of
recomputing them here.
"""

import hashlib
import numpy as np
from typing import List


def get_fold(prompt_id: str, n_folds: int = 5, seed: int = 42) -> int:
    """Get a stable hash-based fold assignment for a single prompt_id.

    Uses stable hashing that:
    - Survives sample filtering
    - Works with fresh draws (same prompt_id → same fold)

    Note: calibration folds are NOT assigned this way (see the module
    docstring) — this is for generic reproducible splits.

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
