"""Mode detection for CJE analysis.

Determines whether to use Direct, IPS, or DR mode based on available data.
"""

import logging
from typing import Dict, Tuple, Optional, List, TypedDict
from pathlib import Path

from ..data.models import Dataset

logger = logging.getLogger(__name__)


class ModeSelectionInfo(TypedDict):
    """Information about mode selection for metadata."""

    mode: str
    estimator: str
    logprob_coverage: float
    has_fresh_draws: bool
    has_logged_data: bool
    explanation: str


def detect_analysis_mode(
    dataset: Dataset,
    fresh_draws_dir: Optional[str] = None,
) -> Tuple[str, str, float]:
    """Detect the appropriate analysis mode for logged data.

    NOTE: This is only called when logged_data_path is provided.
    Direct-only mode (fresh_draws_dir without logged_data) is handled separately.

    Mode detection uses a simple 3-rule system based on available data:
        1. fresh_draws present → DR mode (doubly robust)
        2. fresh_draws absent → IPS mode (importance sampling)
        3. no logged data at all → Error (handled by caller)

    Logprob coverage is still computed and returned for diagnostics, but it does NOT
    affect mode selection. Low coverage will be flagged in warnings but won't prevent
    using IPS/DR modes.

    Returns:
        Tuple of (mode_name, explanation, logprob_coverage)

        Mode names returned:
        - "ips": Importance sampling mode (logged data, no fresh draws)
        - "dr": Doubly robust mode (logged data AND fresh draws)

        The logprob_coverage value is returned for populating mode_selection metadata
        and diagnostic warnings.

    Examples:
        >>> # Case 1: Logged data, no fresh draws → IPS mode
        >>> dataset = load_dataset("logs.jsonl")
        >>> mode, msg = detect_analysis_mode(dataset, None)
        >>> # Returns: ("ips", "IPS mode: Reweighting logged samples...")

        >>> # Case 2: Logged data + fresh draws → DR mode
        >>> mode, msg = detect_analysis_mode(dataset, "responses/")
        >>> # Returns: ("dr", "DR mode: Combining importance weighting...")
    """
    # Count samples with valid logprobs
    n_total = len(dataset.samples)
    n_valid_logprobs = 0

    for sample in dataset.samples:
        # Check if has base_policy_logprob
        if sample.base_policy_logprob is None:
            continue

        # Check if has valid target_policy_logprobs for declared policies
        all_targets_valid = True
        for policy in dataset.target_policies:
            if policy not in sample.target_policy_logprobs:
                all_targets_valid = False
                break
            if sample.target_policy_logprobs[policy] is None:
                all_targets_valid = False
                break

        if all_targets_valid:
            n_valid_logprobs += 1

    logprob_coverage = n_valid_logprobs / n_total if n_total > 0 else 0.0
    has_fresh_draws = fresh_draws_dir is not None and Path(fresh_draws_dir).exists()

    # Error if no valid logprobs and no fresh draws
    if logprob_coverage == 0 and not has_fresh_draws:
        raise ValueError(
            f"Insufficient data: No samples have complete logprobs "
            f"and no fresh draws provided. Cannot proceed with any analysis mode.\n\n"
            f"To fix, choose one:\n"
            f"  1. Ensure samples have base_policy_logprob and target_policy_logprobs → enables IPS/DR mode\n"
            f"     (see cje/teacher_forcing/ for teacher-forced logprob computation)\n"
            f"  2. Provide fresh draws (--fresh-draws-dir) → enables Direct mode\n"
            f"     (on-policy evaluation without counterfactual inference)\n"
        )

    # Simplified mode routing based on data availability
    if has_fresh_draws:
        # Has fresh draws: use DR mode
        mode = "dr"
        if logprob_coverage > 0:
            coverage_note = (
                f"{logprob_coverage:.1%} of logged samples have valid logprobs. "
            )
            if logprob_coverage < 0.5:
                coverage_note += f"⚠️ Low coverage - only {n_valid_logprobs}/{n_total} samples will be used for importance weighting. "
            explanation = (
                f"DR mode: {coverage_note}"
                f"Combining importance weighting with outcome models from fresh draws for best accuracy."
            )
        else:
            # Has fresh draws but no logprobs - essentially Direct mode via DR estimator
            logger.warning(
                "No valid logprobs found in logged data. DR mode will rely entirely on outcome model."
            )
            explanation = (
                "DR mode: No valid logprobs in logged data. "
                "Using fresh draws for outcome modeling. "
                "Note: Without logprobs, this is equivalent to Direct mode."
            )
    else:
        # No fresh draws: use IPS mode
        mode = "ips"
        if logprob_coverage < 0.5:
            explanation = (
                f"IPS mode: {logprob_coverage:.1%} of samples have valid logprobs "
                f"({n_valid_logprobs}/{n_total}). "
                f"⚠️ Low coverage - results may be less reliable. "
                f"Tip: Provide --fresh-draws-dir for more robust DR estimates."
            )
        else:
            explanation = (
                f"IPS mode: {logprob_coverage:.1%} of samples have valid logprobs. "
                f"Reweighting logged samples to estimate target policies via importance sampling. "
                f"Tip: Provide --fresh-draws-dir for more accurate DR estimates."
            )

    return mode, explanation, logprob_coverage


def check_multi_policy_format(dataset: Dataset) -> bool:
    """Check if dataset is in multi-policy format (suitable for direct mode).

    Multi-policy format means:
    - Multiple unique policies in the data
    - Samples grouped by prompt_id with different policies
    - Typically used for head-to-head comparison

    Returns:
        True if dataset appears to be multi-policy format
    """
    if len(dataset.target_policies) <= 1:
        return False

    # Check if we have samples with different policies on same prompts
    prompt_to_policies: Dict[str, List[str]] = {}

    for sample in dataset.samples:
        prompt_id = sample.prompt_id
        # Infer policy from metadata if available
        policy = sample.metadata.get("policy")
        if policy:
            if prompt_id not in prompt_to_policies:
                prompt_to_policies[prompt_id] = []
            prompt_to_policies[prompt_id].append(policy)

    # If we have prompts with multiple policies, it's multi-policy format
    multi_policy_prompts = sum(
        1 for policies in prompt_to_policies.values() if len(set(policies)) > 1
    )

    return multi_policy_prompts > 0
