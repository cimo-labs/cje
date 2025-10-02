"""Mode detection for CJE analysis.

Determines whether to use Direct, IPS, or DR mode based on available data.
"""

import logging
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from ..data.models import Dataset

logger = logging.getLogger(__name__)


def detect_analysis_mode(
    dataset: Dataset,
    fresh_draws_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """Detect the appropriate analysis mode for logged data.

    NOTE: This is only called when logged_data_path is provided.
    Direct-only mode (fresh_draws_dir without logged_data) is handled separately.

    Returns:
        Tuple of (mode_name, explanation)

        Mode names returned:
        - "ips": Importance sampling mode (logged data with logprobs, no fresh draws)
        - "dr": Doubly robust mode (logged data with logprobs AND fresh draws)
        - "direct": Direct evaluation mode (fresh draws available, insufficient logprobs for IPS/DR)

    Logic:
        1. Check if we have valid logprobs (base_policy_logprob + target_policy_logprobs)
        2. Check if we have fresh draws directory
        3. Select mode based on what's available

    Examples:
        >>> # Case 1: Logged data with logprobs only → IPS mode
        >>> dataset = load_dataset("logs.jsonl")
        >>> mode, msg = detect_analysis_mode(dataset, None)
        >>> # Returns: ("ips", "Using IPS mode...")

        >>> # Case 2: Logged data with logprobs + fresh draws → DR mode
        >>> mode, msg = detect_analysis_mode(dataset, "responses/")
        >>> # Returns: ("dr", "Using DR mode...")

        >>> # Case 3: Logged data but no logprobs, has fresh draws → Direct mode
        >>> mode, msg = detect_analysis_mode(dataset, "responses/")
        >>> # Returns: ("direct", "Using Direct mode with calibration...")
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

    # Decision logic: Choose between IPS, DR, or Direct (with calibration)

    if has_fresh_draws and logprob_coverage >= 0.5:
        # Has both fresh draws and logprobs: use DR mode
        mode = "dr"
        explanation = (
            f"DR mode: {logprob_coverage:.1%} of samples have valid logprobs "
            f"and fresh draws are available. This combines importance weighting with "
            f"outcome models for best accuracy."
        )

    elif has_fresh_draws and logprob_coverage < 0.1:
        # Has fresh draws but few/no logprobs: use Direct mode with calibration from logged data
        mode = "direct"
        if logprob_coverage > 0:
            explanation = (
                f"Direct mode with calibration: Only {logprob_coverage:.1%} of samples have logprobs "
                f"(insufficient for IPS/DR), but fresh draws are available. Using logged data for "
                f"calibration only, computing on-policy evaluation on fresh draws. "
                f"Note: This does NOT estimate counterfactual deployment value."
            )
        else:
            explanation = (
                "Direct mode with calibration: No logprobs detected. Using logged data for "
                "calibration, evaluating fresh draws from target policies. "
                "Note: This does NOT estimate counterfactual deployment value."
            )

    elif logprob_coverage >= 0.5:
        # Has logprobs but no fresh draws: use IPS mode
        mode = "ips"
        explanation = (
            f"IPS mode: {logprob_coverage:.1%} of samples have valid logprobs. "
            f"Reweighting logged samples to estimate target policies via importance sampling. "
            f"Tip: Provide --fresh-draws-dir for more accurate DR estimates."
        )

    elif has_fresh_draws and 0.1 <= logprob_coverage < 0.5:
        # Ambiguous: some logprobs, has fresh draws - prefer Direct mode
        mode = "direct"
        explanation = (
            f"Direct mode with calibration: {logprob_coverage:.1%} of samples have logprobs "
            f"(below 50% threshold for reliable IPS/DR). Using logged data for calibration, "
            f"evaluating fresh draws. Warning: Mixed data - consider computing logprobs for all "
            f"samples to enable DR mode."
        )

    else:
        # Insufficient data: no fresh draws and too few logprobs
        raise ValueError(
            f"Insufficient data: only {logprob_coverage:.1%} of samples have logprobs, "
            f"and no fresh draws provided. Cannot proceed with any analysis mode.\n\n"
            f"Options:\n"
            f"  1. Compute teacher-forced logprobs for all samples (see cje/teacher_forcing/)\n"
            f"  2. Provide fresh draws from target policies (--fresh-draws-dir)\n"
            f"  3. Ensure your data has either:\n"
            f"     - base_policy_logprob + target_policy_logprobs (for IPS/DR)\n"
            f"     - Fresh samples with judge scores (for Direct mode)\n\n"
            f"Need at least 50% logprob coverage for IPS mode, or fresh draws for Direct mode."
        )

    return mode, explanation


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
