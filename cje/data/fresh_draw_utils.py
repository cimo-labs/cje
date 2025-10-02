"""Utilities for computing fresh draw statistics for Monte Carlo uncertainty."""

from typing import Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .fresh_draws import FreshDrawDataset


def compute_fresh_draw_prompt_stats(
    fresh_dataset: "FreshDrawDataset",
) -> Dict[str, Dict[str, Any]]:
    """Compute per-prompt statistics for fresh draws.

    Args:
        fresh_dataset: FreshDrawDataset with samples containing judge scores

    Returns:
        Dictionary mapping prompt_id to stats dict with keys:
        - 'mean': Sample mean of judge scores
        - 'var': Unbiased sample variance (ddof=1) if M_i > 1, else 0.0
        - 'n': Number of fresh draws (M_i)
    """
    scores_by_prompt: Dict[str, list] = {}

    # Group scores by prompt_id
    for sample in fresh_dataset.samples:
        pid = str(sample.prompt_id)
        # Get judge score from sample
        if hasattr(sample, "judge_score") and sample.judge_score is not None:
            score = float(sample.judge_score)
        elif hasattr(sample, "reward") and sample.reward is not None:
            # Try reward as fallback
            score = float(sample.reward)
        else:
            score = 0.0

        scores_by_prompt.setdefault(pid, []).append(score)

    # Compute statistics per prompt
    stats: Dict[str, Dict[str, Any]] = {}
    for pid, scores_list in scores_by_prompt.items():
        x = np.array(scores_list, dtype=np.float64)
        m = x.size

        # Compute mean and variance
        mean = float(x.mean()) if m > 0 else 0.0
        var = float(x.var(ddof=1)) if m > 1 else 0.0  # Unbiased variance

        stats[pid] = {"mean": mean, "var": var, "n": m}

    return stats
