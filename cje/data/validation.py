"""Data validation utilities for CJE.

This module provides functions to validate that input data has the required
fields for different CJE use cases.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def validate_cje_data(
    data: List[Dict[str, Any]],
    reward_field: Optional[str] = None,
    judge_field: Optional[str] = None,
    oracle_field: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """Validate that data has required fields for CJE analysis.

    This function checks that the data has the core required fields
    (prompt, response, base_policy_logprob, target_policy_logprobs)
    and appropriate evaluation fields (either reward or judge scores).

    Args:
        data: List of data records to validate
        reward_field: Field name containing pre-calibrated rewards
        judge_field: Field name containing judge scores
        oracle_field: Field name containing oracle labels

    Returns:
        Tuple of (is_valid, list_of_issues)

    Example:
        >>> data = load_jsonl("data.jsonl")
        >>> is_valid, issues = validate_cje_data(
        ...     data,
        ...     judge_field="judge_score",
        ...     oracle_field="oracle_label"
        ... )
        >>> if not is_valid:
        ...     for issue in issues:
        ...         print(f"⚠️  {issue}")
    """
    issues = []

    if not data:
        issues.append("Data is empty")
        return False, issues

    # Check core fields in first sample (assume homogeneous)
    sample = data[0]
    core_fields = [
        "prompt_id",
        "prompt",
        "response",
        "base_policy_logprob",
        "target_policy_logprobs",
    ]

    for field in core_fields:
        if field not in sample:
            issues.append(f"Missing required field: {field}")

    # Check that target_policy_logprobs is a dict
    if "target_policy_logprobs" in sample:
        if not isinstance(sample["target_policy_logprobs"], dict):
            issues.append("target_policy_logprobs must be a dictionary")
        elif not sample["target_policy_logprobs"]:
            issues.append("target_policy_logprobs cannot be empty")

    # Check evaluation fields - scan a larger sample for robustness
    # Check up to 100 samples or 10% of data, whichever is smaller
    sample_size = min(100, max(10, len(data) // 10))
    has_reward = reward_field and reward_field in sample

    # Check for judge field - accept it either at top level or in metadata
    # Also validate that values are numeric and non-None
    judge_samples_checked = 0
    valid_judge_samples = 0
    invalid_judge_values = []

    if judge_field:
        for i, rec in enumerate(data[:sample_size]):
            judge_samples_checked += 1
            judge_val = None

            # Check top level first
            if judge_field in rec:
                judge_val = rec[judge_field]
            # Then check metadata
            elif "metadata" in rec and judge_field in rec["metadata"]:
                judge_val = rec["metadata"][judge_field]

            # Validate the value
            if judge_val is not None:
                if isinstance(judge_val, (int, float)):
                    valid_judge_samples += 1
                else:
                    invalid_judge_values.append((i, type(judge_val).__name__))

    has_judge = judge_field and (valid_judge_samples > 0)

    # Report invalid judge values if found
    if invalid_judge_values:
        examples = invalid_judge_values[:3]  # Show first 3 examples
        issues.append(
            f"Judge field '{judge_field}' has non-numeric values. "
            f"Examples: {examples}. Values must be numeric (int or float)."
        )

    if not has_reward and not has_judge:
        issues.append(
            "No evaluation field found. Need either:\n"
            "  - A 'reward' field with pre-calibrated values, OR\n"
            "  - Judge scores in metadata for calibration"
        )

    # If using judge scores, oracle labels are REQUIRED for calibration
    if has_judge and not has_reward:
        # Judge scores without rewards require oracle labels for calibration
        if not oracle_field:
            issues.append(
                "Judge scores require oracle labels for calibration. "
                "Provide oracle_field parameter pointing to oracle labels."
            )
        else:
            # Check for oracle labels - accept at top level or in metadata
            # Also validate that values are numeric
            oracle_count = 0
            invalid_oracle_values = []

            for i, rec in enumerate(data):
                oracle_val = None

                # Check top level first
                if oracle_field in rec:
                    oracle_val = rec[oracle_field]
                # Then check metadata
                elif "metadata" in rec and oracle_field in rec["metadata"]:
                    oracle_val = rec["metadata"][oracle_field]

                # Validate the value
                if oracle_val is not None:
                    if isinstance(oracle_val, (int, float)):
                        oracle_count += 1
                    else:
                        invalid_oracle_values.append((i, type(oracle_val).__name__))

            # Report invalid oracle values if found
            if invalid_oracle_values:
                examples = invalid_oracle_values[:3]  # Show first 3 examples
                issues.append(
                    f"Oracle field '{oracle_field}' has non-numeric values. "
                    f"Examples: {examples}. Values must be numeric (int or float)."
                )

            if oracle_count == 0:
                issues.append(
                    f"No valid oracle labels found in field '{oracle_field}'. "
                    "Judge scores require oracle labels for calibration. "
                    "Need at least 10 samples with oracle labels (50-100 recommended). "
                    "Check that oracle values are numeric and non-None."
                )
            elif oracle_count < 10:
                issues.append(
                    f"Too few oracle samples ({oracle_count}). "
                    "Absolute minimum is 10 samples. "
                    "Strongly recommend 50-100+ for robust calibration."
                )
            elif oracle_count < 50:
                logger.warning(
                    f"Found {oracle_count} oracle samples. "
                    f"Consider adding more (50-100 recommended) for better calibration."
                )
            else:
                logger.info(f"Found {oracle_count} oracle samples for calibration")

    # Check data consistency across samples
    n_samples = len(data)
    valid_base_lp = sum(1 for rec in data if rec.get("base_policy_logprob") is not None)

    if valid_base_lp < n_samples:
        pct_missing = 100 * (n_samples - valid_base_lp) / n_samples
        issues.append(
            f"{n_samples - valid_base_lp}/{n_samples} samples "
            f"({pct_missing:.1f}%) have missing base_policy_logprob"
        )

    # Check target policies consistency
    if data and "target_policy_logprobs" in data[0]:
        first_policies = set(data[0]["target_policy_logprobs"].keys())
        inconsistent = []

        for i, rec in enumerate(data[1:11], 1):  # Check first 10
            if "target_policy_logprobs" in rec:
                policies = set(rec["target_policy_logprobs"].keys())
                if policies != first_policies:
                    inconsistent.append(i)

        if inconsistent:
            issues.append(
                f"Inconsistent target policies in samples {inconsistent}. "
                f"Expected: {first_policies}"
            )

    is_valid = len(issues) == 0
    return is_valid, issues


def validate_for_precomputed_sampler(
    data: List[Dict[str, Any]], reward_field: str = "reward"
) -> Tuple[bool, List[str]]:
    """Validate data specifically for PrecomputedSampler.

    PrecomputedSampler requires rewards to be already present,
    either as pre-calibrated values or from judge calibration.

    Args:
        data: List of data records
        reward_field: Field name containing rewards

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # First check basic CJE requirements including the reward field
    is_valid, base_issues = validate_cje_data(data, reward_field=reward_field)
    issues.extend(base_issues)

    # Check for rewards
    if not data:
        issues.append("Data is empty")
        return False, issues

    has_rewards = all(
        reward_field in rec and rec[reward_field] is not None
        for rec in data[: min(100, len(data))]
    )

    if not has_rewards:
        issues.append(
            f"PrecomputedSampler requires '{reward_field}' field. "
            "Either provide pre-calibrated rewards or use calibrate_dataset() first."
        )

    return len(issues) == 0, issues
