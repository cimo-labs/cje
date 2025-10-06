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

    Mode detection is based on **logprob coverage**:
        logprob_coverage = (samples with complete logprobs) / total_samples

    A sample has "complete logprobs" if:
        - base_policy_logprob is not None
        - target_policy_logprobs[policy] is not None for ALL target policies

    Decision rules (simplified 4-rule system):
        1. fresh_draws present + coverage ≥50% → DR mode (doubly robust)
        2. fresh_draws absent + coverage ≥50% → IPS mode (importance sampling)
        3. fresh_draws present + coverage <50% → Direct mode (on-policy evaluation)
        4. Otherwise (no fresh draws + coverage <50%) → Error (insufficient data)

    Returns:
        Tuple of (mode_name, explanation, logprob_coverage)

        Mode names returned:
        - "ips": Importance sampling mode (logged data with logprobs, no fresh draws)
        - "dr": Doubly robust mode (logged data with logprobs AND fresh draws)
        - "direct": Direct evaluation mode (fresh draws available, insufficient logprobs for IPS/DR)

        The logprob_coverage value is returned for populating mode_selection metadata.

    Examples:
        >>> # Case 1: 100% logprob coverage, no fresh draws → IPS mode
        >>> dataset = load_dataset("logs.jsonl")  # All samples have logprobs
        >>> mode, msg = detect_analysis_mode(dataset, None)
        >>> # Returns: ("ips", "IPS mode: 100.0% of samples have valid logprobs...")

        >>> # Case 2: 80% logprob coverage + fresh draws → DR mode
        >>> mode, msg = detect_analysis_mode(dataset, "responses/")
        >>> # Returns: ("dr", "DR mode: 80.0% of samples have valid logprobs...")

        >>> # Case 3: 20% logprob coverage + fresh draws → Direct mode
        >>> mode, msg = detect_analysis_mode(dataset, "responses/")
        >>> # Returns: ("direct", "Direct mode: Only 20.0% of samples have logprobs...")
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

    elif logprob_coverage >= 0.5:
        # Rule 2: Has logprobs but no fresh draws: use IPS mode
        mode = "ips"
        explanation = (
            f"IPS mode: {logprob_coverage:.1%} of samples have valid logprobs. "
            f"Reweighting logged samples to estimate target policies via importance sampling. "
            f"Tip: Provide --fresh-draws-dir for more accurate DR estimates."
        )

    elif has_fresh_draws:
        # Rule 3: Has fresh draws but insufficient logprobs (<50%): use Direct mode
        mode = "direct"
        if logprob_coverage == 0:
            explanation = (
                "Direct mode: No logprobs detected. Using fresh draws for on-policy evaluation. "
                f"Note: This estimates on-policy value, not counterfactual deployment value. "
                f"Tip: Compute logprobs for ≥50% of samples to enable DR mode."
            )
        else:
            explanation = (
                f"Direct mode: Only {logprob_coverage:.1%} of samples have logprobs "
                f"(need ≥50% for IPS/DR). Using fresh draws for on-policy evaluation. "
                f"Note: This estimates on-policy value, not counterfactual deployment value. "
                f"Tip: Compute logprobs for ≥50% of samples to enable DR mode."
            )

    else:
        # Rule 4: No fresh draws and insufficient logprobs (<50%): error
        raise ValueError(
            f"Insufficient data: only {logprob_coverage:.1%} of samples have logprobs "
            f"and no fresh draws provided. Cannot proceed with any analysis mode.\n\n"
            f"To fix, choose one:\n"
            f"  1. Compute logprobs for ≥50% of samples → enables IPS/DR mode\n"
            f"     (see cje/teacher_forcing/ for teacher-forced logprob computation)\n"
            f"  2. Provide fresh draws (--fresh-draws-dir) → enables Direct mode\n"
            f"     (on-policy evaluation on target policies)\n\n"
            f"Current coverage: {logprob_coverage:.1%} (need ≥50% for IPS/DR)"
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
