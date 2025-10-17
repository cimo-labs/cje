"""Precomputed sampler for CJE estimation.

This module provides the PrecomputedSampler class which manages data access
for CJE estimators. It maintains multiple data representations for different
purposes:

1. Original samples (dataset.samples): Complete Sample objects with metadata
2. Formatted data (formatted_data): Filtered samples for weight computation
3. Policy data (get_data_for_policy): Transformed dicts with flattened metadata

Example Data Flow:
-----------------
    # Original Sample object
    sample.judge_score = 0.8  # Top-level field
    sample.reward = 0.75

    # After get_data_for_policy("gpt-4")
    data["judge_score"] = 0.8  # From sample.judge_score
    data["reward"] = 0.75      # From sample.reward
    data["cv_fold"] = 2        # Computed from prompt_id

Key Differences:
---------------
    # WRONG - looking in metadata
    data = sampler.get_data_for_policy("gpt-4")
    score = data[0]["metadata"]["judge_score"]  # ❌ KeyError!

    # RIGHT - judge_score is top-level in dict
    data = sampler.get_data_for_policy("gpt-4")
    score = data[0]["judge_score"]  # ✓ Correct

    # ALSO RIGHT - accessing original samples directly
    score = sampler.dataset.samples[0].judge_score  # ✓ Correct

See PolicyDataDict for the complete structure returned by get_data_for_policy().
"""

from typing import Dict, List, Optional, Any, Union, TypedDict, cast
import numpy as np
import logging

from .models import Dataset
from .factory import DatasetFactory
from .folds import get_folds_for_prompts, get_fold

logger = logging.getLogger(__name__)


class PolicyDataDict(TypedDict, total=False):
    """Structure returned by get_data_for_policy().

    This is a flattened representation that combines data from:
    - Sample fields (reward, prompt, response, etc.)
    - Sample.metadata (judge_score, oracle_label)
    - Computed fields (cv_fold)

    Note: judge_score is moved from metadata to top-level for convenience.
    """

    # Required fields
    reward: float
    base_policy_logprob: float
    policy_logprob: float
    prompt: str
    response: str
    prompt_id: str

    # Optional fields (from metadata or computed)
    judge_score: Optional[float]
    oracle_label: Optional[float]
    cv_fold: int


class PrecomputedSampler:
    """Wrapper around Dataset that provides CJE-specific operations.

    This class takes a Dataset and adds methods needed for importance sampling
    estimation like weight computation, filtering, and diagnostic checks.

    Data Representations:
    --------------------
    The sampler maintains three data representations:

    1. self.dataset.samples: Original Sample objects with full metadata
       - Access: sampler.dataset.samples
       - Use for: Accessing complete metadata, oracle labels

    2. self.formatted_data: Filtered/validated samples for weight computation
       - Access: Internal only (use get_data_for_policy instead)
       - Use for: Weight computation (pre-filtered for efficiency)

    3. get_data_for_policy(): Transformed dicts with flattened structure
       - Access: sampler.get_data_for_policy(policy)
       - Use for: Main data access in estimators
       - Structure: See PolicyDataDict

    Important: get_data_for_policy() transforms the data by:
    - Filtering to samples with valid logprobs for the policy
    - Flattening metadata.judge_score to top-level
    - Adding computed cv_fold field
    """

    def __init__(
        self,
        data_or_dataset: Union[Dataset, List[Dict[str, Any]]],
        target_policies: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Initialize with either a Dataset or raw data.

        Args:
            data_or_dataset: Either a Dataset instance or raw data list
            target_policies: Target policy names (only used if data_or_dataset is a list)
            **kwargs: Additional arguments passed to DatasetFactory

        Raises:
            ValueError: If any samples are missing rewards
        """
        if isinstance(data_or_dataset, Dataset):
            self.dataset = data_or_dataset
        else:
            # Create Dataset from raw data using factory
            factory = DatasetFactory()
            self.dataset = factory.create_from_data(
                data_or_dataset, target_policies=target_policies
            )

        # Validate that all samples have rewards
        samples_without_rewards = [
            i for i, sample in enumerate(self.dataset.samples) if sample.reward is None
        ]
        if samples_without_rewards:
            raise ValueError(
                f"PrecomputedSampler requires all samples to have rewards. "
                f"Found {len(samples_without_rewards)} samples without rewards. "
                f"Please calibrate your dataset first using calibrate_dataset()."
            )

        self.target_policies = self.dataset.target_policies

        # Prepare formatted data
        self.formatted_data = self._format_for_estimators()

    @classmethod
    def from_jsonl(
        cls, file_path: str, target_policies: Optional[List[str]] = None, **kwargs: Any
    ) -> "PrecomputedSampler":
        """Create sampler from JSONL file.

        Args:
            file_path: Path to JSONL file
            target_policies: Optional list of target policy names
            **kwargs: Additional arguments passed to DatasetFactory

        Returns:
            PrecomputedSampler instance
        """
        factory = DatasetFactory()
        dataset = factory.create_from_jsonl(file_path, target_policies)
        return cls(dataset)

    def _format_for_estimators(self) -> List[Dict[str, Any]]:
        """Format data for CJE estimators.

        Returns list of dicts with:
        - context: prompt text
        - response: generated text
        - base_policy_logprob: base policy log prob
        - reward: calibrated reward
        - target_policy_logprobs: dict of target log probs
        """
        formatted = []
        self._formatted_to_dataset_idx = []  # Track mapping for O(1) lookup
        n_missing_base = 0
        n_missing_target = {policy: 0 for policy in self.target_policies}

        for i, sample in enumerate(self.dataset.samples):
            # Skip samples without valid base log prob
            if sample.base_policy_logprob is None:
                n_missing_base += 1
                continue

            # Check all required target policies have valid log probs
            valid_targets = {}
            skip_record = False
            for policy in self.target_policies:
                logp = sample.target_policy_logprobs.get(policy)
                if logp is None:
                    n_missing_target[policy] += 1
                    skip_record = True
                    break
                valid_targets[policy] = logp

            if skip_record:
                continue

            formatted.append(
                {
                    "context": sample.prompt,
                    "response": sample.response,
                    "base_policy_logprob": sample.base_policy_logprob,
                    "reward": sample.reward,
                    "target_policy_logprobs": valid_targets,
                }
            )
            self._formatted_to_dataset_idx.append(i)

        # Report filtering statistics
        n_total = len(self.dataset.samples)
        n_valid = len(formatted)
        n_filtered = n_total - n_valid

        if n_filtered > 0:
            filter_pct = (n_filtered / n_total) * 100
            logger.warning(
                f"Filtered {n_filtered}/{n_total} samples ({filter_pct:.1f}%) due to missing log probabilities:\n"
                f"  - Missing base_policy_logprob: {n_missing_base}\n"
                f"  - Missing target policy logprobs: {n_missing_target}"
            )

            if filter_pct > 50:
                logger.error(
                    f"WARNING: More than 50% of samples filtered! Only {n_valid}/{n_total} samples remain. "
                    f"This may significantly impact estimation quality."
                )

        if not formatted:
            raise ValueError(
                f"No valid records after filtering! All {n_total} samples had invalid log probabilities.\n"
                f"  - Missing base_policy_logprob: {n_missing_base}\n"
                f"  - Missing target policy logprobs: {n_missing_target}"
            )

        if n_valid < 10:
            logger.warning(
                f"Only {n_valid} valid samples available for estimation. "
                f"Results may be unreliable with such small sample size."
            )

        return formatted

    def get_data_for_policy(self, target_policy: str) -> Optional[List[PolicyDataDict]]:
        """Get formatted data for a specific target policy.

        ⚠️ IMPORTANT: This method transforms the data structure:

        Transformations applied:
        1. Filters to only samples with valid logprobs for this policy
        2. Flattens metadata fields to top-level (e.g., metadata.judge_score → judge_score)
        3. Adds computed fields (cv_fold from prompt_id)
        4. Returns different structure than dataset.samples

        Args:
            target_policy: Name of target policy

        Returns:
            List of PolicyDataDict with structure:
            {
                "reward": float,                    # From sample.reward
                "base_policy_logprob": float,       # From sample.base_policy_logprob
                "policy_logprob": float,            # From sample.target_policy_logprobs[policy]
                "prompt": str,                      # From sample.prompt
                "response": str,                    # From sample.response
                "prompt_id": str,                   # From sample.prompt_id
                "judge_score": Optional[float],     # From sample.judge_score
                "oracle_label": Optional[float],    # From sample.oracle_label
                "cv_fold": int,                     # Computed from prompt_id
            }

            Returns None if policy not in target_policies.

        Example:
            >>> data = sampler.get_data_for_policy("gpt-4")
            >>> judge_scores = [d["judge_score"] for d in data]  # Top-level access
            >>> # NOT d["metadata"]["judge_score"] - already flattened!

        Note: This ensures consistency with weight computation by using the same
        filtered samples (formatted_data) that were used to compute weights.
        """
        if target_policy not in self.target_policies:
            return None

        # Use the same formatted_data that was used for weights to ensure consistency
        policy_data = []
        for i, record in enumerate(self.formatted_data):
            # Check if this record has the target policy logprob
            if target_policy in record["target_policy_logprobs"]:
                # Get the corresponding sample for metadata
                sample = self.dataset.samples[self._get_sample_index(i)]

                # Build policy data dict with core fields
                data_dict = {
                    "reward": record["reward"],
                    "base_policy_logprob": record["base_policy_logprob"],
                    "policy_logprob": record["target_policy_logprobs"][target_policy],
                    "prompt": record["context"],
                    "response": record["response"],
                    "prompt_id": sample.prompt_id,
                    "judge_score": sample.judge_score,
                    "oracle_label": sample.oracle_label,
                    # Compute cv_fold on-demand from prompt_id
                    # Use metadata if available, else defaults
                    "cv_fold": get_fold(
                        sample.prompt_id,
                        self.dataset.metadata.get("n_folds", 5),
                        self.dataset.metadata.get("fold_seed", 42),
                    ),
                }

                # Flatten ALL covariates from metadata (e.g., response_length, domain, etc.)
                # This ensures any covariate computed during calibration is available
                # at the top level for DR estimators to use
                if sample.metadata:
                    for key, value in sample.metadata.items():
                        # Only add if not already in data_dict (don't overwrite core fields)
                        # Skip keys that are already flattened (judge_score, oracle_label)
                        if key not in data_dict and key not in (
                            "judge_score",
                            "oracle_label",
                            "prompt_id",
                        ):
                            data_dict[key] = value

                policy_data.append(data_dict)

        return cast(List[PolicyDataDict], policy_data) if policy_data else None

    def __len__(self) -> int:
        """Return the number of valid samples in formatted data."""
        return self.n_valid_samples

    def _get_valid_indices(self, target_policy: str) -> np.ndarray:
        """Get indices of valid samples for a target policy.

        Returns indices into the original dataset.samples array.
        """
        valid_indices = []
        for i, sample in enumerate(self.dataset.samples):
            # Check if sample has valid data for this policy
            if (
                sample.base_policy_logprob is not None
                and sample.target_policy_logprobs.get(target_policy) is not None
            ):
                valid_indices.append(i)
        return np.array(valid_indices)

    def _get_sample_index(self, formatted_index: int) -> int:
        """Map from formatted_data index back to dataset.samples index.

        This is needed because formatted_data filters out invalid samples.
        """
        if formatted_index >= len(self._formatted_to_dataset_idx):
            raise IndexError(f"Formatted index {formatted_index} out of range")
        return self._formatted_to_dataset_idx[formatted_index]

    def compute_log_ratios(self, target_policy: str) -> np.ndarray:
        """Compute log importance ratios (log p_target - log p_base).

        Args:
            target_policy: Name of target policy

        Returns:
            Array of log ratios (may contain -inf for zero weights)
        """
        if target_policy not in self.target_policies:
            raise ValueError(f"Unknown target policy: {target_policy}")

        log_ratios = np.array(
            [
                record["target_policy_logprobs"][target_policy]
                - record["base_policy_logprob"]
                for record in self.formatted_data
            ],
            dtype=np.float64,
        )

        # NaN -> -inf (zero weight)
        log_ratios[np.isnan(log_ratios)] = -np.inf

        return log_ratios

    def compute_raw_weights(self, target_policy: str) -> np.ndarray:
        """Compute raw importance weights WITHOUT scaling.

        Returns truly raw weights: exp(log_p_target - log_p_base)
        Only guards against overflow to inf, no scaling applied.

        Args:
            target_policy: Name of target policy

        Returns:
            Array of raw importance weights
        """
        log_ratios = self.compute_log_ratios(target_policy)

        # Clamp only to avoid overflow to inf, keep underflow (->0) natural
        max_log = np.log(np.finfo(np.float64).max)  # ~709.78
        clamped = np.minimum(log_ratios, max_log)

        # Report extreme values
        n_clamped = np.sum(log_ratios > max_log)
        if n_clamped > 0:
            logger.debug(
                f"Clamped {n_clamped} extreme log-ratios for {target_policy} "
                f"(max was {np.max(log_ratios[np.isfinite(log_ratios)]):.1f}) to prevent overflow"
            )

        weights = np.exp(clamped)

        # Clean up non-finite values (from -inf log ratios)
        weights[~np.isfinite(weights)] = 0.0

        return np.asarray(weights)

    def compute_hajek_weights(self, target_policy: str) -> np.ndarray:
        """Compute mean-one (SNIPS/Hájek) weights using stable log-sum-exp.

        These weights have mean exactly 1.0 and are computed in a numerically
        stable way using the log-sum-exp trick.

        Args:
            target_policy: Name of target policy

        Returns:
            Array of Hájek weights with mean=1.0
        """
        log_ratios = self.compute_log_ratios(target_policy)
        n = len(log_ratios)

        # Handle all -inf case (all weights would be zero)
        finite_mask = np.isfinite(log_ratios)
        if not finite_mask.any():
            logger.warning(
                f"All log-ratios are -inf for {target_policy}; returning zeros"
            )
            return np.zeros_like(log_ratios)

        # Log-sum-exp trick: subtract max for numerical stability
        max_log = np.max(log_ratios[finite_mask])

        # Compute stable exponentials
        stable_exp = np.zeros_like(log_ratios)
        stable_exp[finite_mask] = np.exp(log_ratios[finite_mask] - max_log)

        # Sum of weights
        sum_weights = stable_exp.sum()
        if sum_weights == 0.0:
            logger.warning(
                f"Sum of weights is zero for {target_policy}; returning zeros"
            )
            return np.zeros_like(log_ratios)

        # Normalize to mean=1: w_i = n * exp(lr_i) / sum(exp(lr))
        hajek_weights = (n * stable_exp) / sum_weights

        # Verify mean is 1.0 (within floating point precision)
        actual_mean = hajek_weights.mean()
        if abs(actual_mean - 1.0) > 1e-10:
            logger.debug(
                f"Hájek weights for {target_policy} have mean {actual_mean:.12f} (expected 1.0)"
            )

        return np.asarray(hajek_weights)

    def compute_importance_weights(
        self,
        target_policy: str,
        clip_weight: Optional[float] = None,
        mode: str = "hajek",
    ) -> np.ndarray:
        """Compute importance weights for a target policy with optional clipping.

        Args:
            target_policy: Name of target policy
            clip_weight: Maximum weight value for variance control.
                        If None (default), no clipping is applied.
                        Set to a finite value (e.g., 100.0) to clip weights.
            mode: "hajek" for mean-one weights (default), "raw" for unnormalized

        Returns:
            Array of importance weights
        """
        # Get weights based on mode
        if mode == "hajek":
            weights_array = self.compute_hajek_weights(target_policy)
        elif mode == "raw":
            weights_array = self.compute_raw_weights(target_policy)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'hajek' or 'raw'")

        # Apply clipping if requested (after Hájek normalization for interpretability)
        if clip_weight is not None and np.isfinite(clip_weight):
            n_clipped = np.sum(weights_array > clip_weight)
            if n_clipped > 0:
                max_weight = np.max(weights_array)
                weights_array = np.minimum(weights_array, clip_weight)
                # Restore mean-one after clipping to keep Hájek unbiasedness
                if mode == "hajek":
                    s = weights_array.sum()
                    if s > 0:
                        weights_array *= len(weights_array) / s
                logger.info(
                    f"Clipped {n_clipped}/{len(weights_array)} weights for {target_policy} "
                    f"to {clip_weight} (max was {max_weight:.2f})"
                    + (
                        f", re-normalized to restore mean-one"
                        if mode == "hajek"
                        else ""
                    )
                )

        # Log statistics
        logger.debug(
            f"Weight statistics for '{target_policy}': "
            f"mean={weights_array.mean():.3f}, std={weights_array.std():.3f}, "
            f"min={weights_array.min():.3f}, max={weights_array.max():.3f}"
        )

        return np.asarray(weights_array)

    def get_rewards(self) -> np.ndarray:
        """Get array of calibrated rewards."""
        return np.array([s.reward for s in self.dataset.samples])

    def get_judge_scores(self) -> Optional[np.ndarray]:
        """Get array of judge scores from metadata.

        ⚠️ Note: This returns judge scores for the FILTERED samples (those with
        valid logprobs), not all samples in the dataset. The order matches
        the samples that would be used for weight computation.

        To get judge scores for a specific policy's data:
            >>> data = sampler.get_data_for_policy(policy)
            >>> scores = [d["judge_score"] for d in data]  # Already flattened

        Returns:
            Array of judge scores if available in all valid samples, None if any missing.
            Length matches len(formatted_data), not len(dataset.samples).
        """
        # Only get judge scores for valid (formatted) samples
        judge_scores = []
        for idx in self._formatted_to_dataset_idx:
            sample = self.dataset.samples[idx]
            score = sample.judge_score
            if score is None:
                return None  # Not all samples have judge scores
            judge_scores.append(score)
        return np.array(judge_scores)

    def get_contexts(self) -> List[str]:
        """Get list of contexts/prompts."""
        return [s.prompt for s in self.dataset.samples]

    def get_responses(self) -> List[str]:
        """Get list of responses."""
        return [s.response for s in self.dataset.samples]

    def get_folds_for_policy(
        self, policy: str, n_folds: Optional[int] = None, seed: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Get consistent fold assignments for policy's valid samples.

        Returns folds for the FILTERED samples that align with
        get_data_for_policy(). This ensures folds match the actual
        data used for estimation.

        Args:
            policy: Target policy name
            n_folds: Number of cross-validation folds (uses metadata if None)
            seed: Random seed for reproducibility (uses metadata if None)

        Returns:
            Fold assignments for valid samples, or None if no data
        """
        # Use metadata values if not provided
        if n_folds is None:
            n_folds = self.dataset.metadata.get("n_folds", 5)
        if seed is None:
            seed = self.dataset.metadata.get("fold_seed", 42)

        data = self.get_data_for_policy(policy)
        if data is None:
            return None

        prompt_ids = [d["prompt_id"] for d in data]
        return get_folds_for_prompts(prompt_ids, n_folds, seed)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get dataset metadata.

        Returns dataset metadata including dataset_path if available.
        """
        return self.dataset.metadata

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset.

        Note: This returns the total number of samples in the dataset.
        For the number of samples with valid log probabilities that will
        be used for estimation, use n_valid_samples.
        """
        return self.dataset.n_samples

    @property
    def n_valid_samples(self) -> int:
        """Number of valid samples with all required log probabilities.

        This is the number of samples that will actually be used for estimation,
        after filtering out samples with missing log probabilities.
        """
        return len(self.formatted_data)

    @property
    def n_policies(self) -> int:
        """Number of target policies."""
        return len(self.target_policies)

    @property
    def oracle_coverage(self) -> Optional[float]:
        """Get oracle coverage (fraction of samples with oracle labels).

        Returns:
            Oracle coverage in [0, 1] if available, None if no information.

        Note:
            This first checks calibration metadata (most reliable),
            then falls back to scanning samples for oracle_label fields.
        """
        # First try: Check calibration metadata (most reliable)
        cal_info = self.dataset.metadata.get("calibration_info", {})
        if "n_oracle" in cal_info and "n_total" in cal_info:
            n_total = cal_info["n_total"]
            if n_total > 0:
                return float(cal_info["n_oracle"]) / float(n_total)

        # Second try: Scan samples for oracle labels
        # This handles cases where dataset wasn't created via calibrate_dataset
        n_with_oracle = 0
        n_total = len(self.dataset.samples)

        if n_total == 0:
            return None

        for sample in self.dataset.samples:
            if sample.oracle_label is not None:
                n_with_oracle += 1

        # If we found any oracle labels, return the coverage
        if n_with_oracle > 0:
            return float(n_with_oracle) / float(n_total)

        # No oracle information found
        return None

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        dataset_summary = self.dataset.summary()
        filter_rate = (
            1.0 - (self.n_valid_samples / self.n_samples) if self.n_samples > 0 else 0.0
        )

        return {
            "n_samples": self.n_samples,  # Keep for backwards compatibility
            "n_samples_valid": self.n_valid_samples,
            "n_samples_total": self.n_samples,  # Same as n_samples
            "n_samples_filtered": self.n_samples - self.n_valid_samples,
            "filter_rate": filter_rate,
            "n_policies": self.n_policies,
            "target_policies": self.target_policies,
            "reward_mean": dataset_summary["reward_mean"],
            "reward_std": dataset_summary["reward_std"],
            "valid_samples_per_policy": dataset_summary["valid_samples_per_policy"],
        }
