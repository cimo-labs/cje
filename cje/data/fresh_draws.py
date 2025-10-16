"""Data models and utilities for fresh draws used in DR estimation."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .models import Dataset


class FreshDrawSample(BaseModel):
    """A single fresh draw sample for DR estimation.

    Represents a fresh response sampled from a target policy,
    evaluated by the judge.
    """

    prompt_id: str = Field(..., description="ID to align with logged data")
    target_policy: str = Field(..., description="Policy that generated this response")
    judge_score: float = Field(..., ge=0, le=1, description="Judge evaluation score")
    oracle_label: Optional[float] = Field(
        None, ge=0, le=1, description="Ground truth oracle label (for calibration)"
    )
    response: Optional[str] = Field(None, description="Generated response (optional)")
    draw_idx: int = Field(
        ..., ge=0, description="Draw index for this prompt (0, 1, 2...)"
    )
    fold_id: Optional[int] = Field(
        None, description="CV fold assignment (should match logged data)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., computed covariates)",
    )

    @field_validator("judge_score")
    def validate_judge_score(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError(f"Judge score must be in [0, 1], got {v}")
        return v


class FreshDrawDataset(BaseModel):
    """Collection of fresh draws for a target policy.

    Contains pre-generated fresh samples from a target policy,
    evaluated by a judge, for use in DR estimation.
    """

    target_policy: str = Field(..., description="Target policy name")
    draws_per_prompt: int = Field(..., ge=1, description="Number of draws per prompt")
    samples: List[FreshDrawSample] = Field(..., min_length=1)

    @field_validator("samples")
    def validate_samples(
        cls, v: List[FreshDrawSample], info: Any
    ) -> List[FreshDrawSample]:
        """Ensure samples are consistent."""
        if "target_policy" in info.data:
            policy = info.data["target_policy"]
            for sample in v:
                if sample.target_policy != policy:
                    raise ValueError(
                        f"Sample has policy '{sample.target_policy}' "
                        f"but dataset is for '{policy}'"
                    )
        return v

    @property
    def n_samples(self) -> int:
        """Total number of fresh draw samples."""
        return len(self.samples)

    def get_prompt_ids(self) -> List[str]:
        """Get unique prompt IDs in dataset."""
        return sorted(set(s.prompt_id for s in self.samples))

    def get_scores_for_prompt_id(self, prompt_id: str) -> np.ndarray:
        """Get judge scores for a specific prompt.

        Args:
            prompt_id: The prompt ID to get scores for

        Returns:
            Array of judge scores for this prompt, sorted by draw_idx
        """
        # Sort by draw_idx for reproducibility
        matching_samples = sorted(
            [s for s in self.samples if s.prompt_id == prompt_id],
            key=lambda s: s.draw_idx,
        )

        if not matching_samples:
            raise ValueError(f"No samples found for prompt_id '{prompt_id}'")

        # Allow variable draws per prompt (don't enforce exact count)
        # Just log if different from expected
        if self.draws_per_prompt and len(matching_samples) != self.draws_per_prompt:
            logger.debug(
                f"Prompt '{prompt_id}' has {len(matching_samples)} draws "
                f"(dataset average: {self.draws_per_prompt})"
            )

        return np.array([s.judge_score for s in matching_samples])

    def get_samples_for_prompt_id(self, prompt_id: str) -> List[FreshDrawSample]:
        """Get all samples for a specific prompt.

        Args:
            prompt_id: The prompt ID to get samples for

        Returns:
            List of samples for this prompt
        """
        samples = [s for s in self.samples if s.prompt_id == prompt_id]

        if not samples:
            raise ValueError(f"No samples found for prompt_id '{prompt_id}'")

        # Sort by draw_idx to ensure consistent ordering
        return sorted(samples, key=lambda s: s.draw_idx)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Convert dataset to arrays for efficient computation.

        Returns:
            Dict with 'prompt_ids' and 'judge_scores' arrays
        """
        # Sort samples by (prompt_id, draw_idx) for consistent ordering
        sorted_samples = sorted(self.samples, key=lambda s: (s.prompt_id, s.draw_idx))

        prompt_ids = []
        judge_scores = []

        for sample in sorted_samples:
            prompt_ids.append(sample.prompt_id)
            judge_scores.append(sample.judge_score)

        return {
            "prompt_ids": np.array(prompt_ids),
            "judge_scores": np.array(judge_scores),
        }

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics for the dataset."""
        scores = np.array([s.judge_score for s in self.samples])
        unique_prompts = self.get_prompt_ids()

        return {
            "target_policy": self.target_policy,
            "n_samples": len(self.samples),
            "n_prompts": len(unique_prompts),
            "draws_per_prompt": self.draws_per_prompt,
            "judge_score_mean": float(scores.mean()),
            "judge_score_std": float(scores.std()),
            "judge_score_min": float(scores.min()),
            "judge_score_max": float(scores.max()),
        }


# ============================================================================
# Utility functions for fresh draws
# ============================================================================


def load_fresh_draws_from_jsonl(path: str) -> Dict[str, FreshDrawDataset]:
    """Load fresh draws from JSONL file, grouped by policy.

    This function delegates to FreshDrawLoader in the loaders module
    for consistency with other data loading operations.

    Expected JSONL format:
    {"prompt_id": "0", "target_policy": "premium", "judge_score": 0.85, "draw_idx": 0}
    {"prompt_id": "0", "target_policy": "premium", "judge_score": 0.82, "draw_idx": 1}
    {"prompt_id": "1", "target_policy": "premium", "judge_score": 0.90, "draw_idx": 0}

    Args:
        path: Path to JSONL file containing fresh draws

    Returns:
        Dict mapping policy names to FreshDrawDataset objects
    """
    from .loaders import FreshDrawLoader

    return FreshDrawLoader.load_from_jsonl(path)


def validate_fresh_draws(
    fresh_draws: FreshDrawDataset,
    logged_dataset: "Dataset",
    policy: str,
) -> None:
    """Validate fresh draws have complete coverage for a policy.

    Args:
        fresh_draws: Fresh draw dataset to validate
        logged_dataset: Logged dataset to check coverage against
        policy: Target policy name

    Raises:
        ValueError: If fresh draws don't have complete coverage
    """
    # Get valid samples for this policy from logged data
    valid_samples = [
        s for s in logged_dataset.samples if s.get_importance_weight(policy) is not None
    ]

    if not valid_samples:
        raise ValueError(f"No valid logged samples for policy '{policy}'")

    # Get prompt IDs
    logged_ids = {s.prompt_id for s in valid_samples}
    fresh_ids = set(fresh_draws.get_prompt_ids())

    # Check coverage
    missing = logged_ids - fresh_ids
    extra = fresh_ids - logged_ids

    if missing:
        raise ValueError(
            f"Fresh draws missing for {len(missing)} prompts:\n"
            f"  First 5 missing: {list(missing)[:5]}\n"
            f"DR requires fresh draws for ALL samples with valid importance weights."
        )

    if extra:
        logger.warning(
            f"Fresh draws contain {len(extra)} extra prompts not in logged data. "
            f"These will be ignored."
        )

    # Check draws per prompt (allow variable M_i)
    draw_counts = []
    for prompt_id in logged_ids:
        try:
            prompt_id_str = str(prompt_id) if prompt_id is not None else ""
            scores = fresh_draws.get_scores_for_prompt_id(prompt_id_str)
            draw_counts.append(len(scores))

            # Warn if significantly different from expected
            if (
                fresh_draws.draws_per_prompt
                and len(scores) != fresh_draws.draws_per_prompt
            ):
                logger.debug(
                    f"Prompt '{prompt_id}' has {len(scores)} draws "
                    f"(dataset average: {fresh_draws.draws_per_prompt})"
                )
        except ValueError as e:
            raise ValueError(f"Validation failed: {e}")

    # Compute statistics
    min_draws = min(draw_counts) if draw_counts else 0
    max_draws = max(draw_counts) if draw_counts else 0
    avg_draws = sum(draw_counts) / len(draw_counts) if draw_counts else 0

    logger.info(
        f"Fresh draws validated: {len(fresh_ids)} prompts, "
        f"draws/prompt: min={min_draws}, avg={avg_draws:.1f}, max={max_draws}"
    )


def create_synthetic_fresh_draws(
    logged_dataset: "Dataset",
    target_policy: str,
    draws_per_prompt: int = 5,
    score_correlation: float = 0.8,
    seed: Optional[int] = None,
) -> FreshDrawDataset:
    """Create synthetic fresh draws for testing.

    Generates correlated judge scores based on logged data,
    useful for testing DR without actual API calls.

    Args:
        logged_dataset: Logged dataset to base fresh draws on
        target_policy: Target policy name
        draws_per_prompt: Number of draws per prompt
        score_correlation: Correlation with logged judge scores (0-1)
        seed: Random seed for reproducibility

    Returns:
        Synthetic FreshDrawDataset
    """
    if seed is not None:
        np.random.seed(seed)

    # Get valid samples for this policy
    valid_samples = [
        s
        for s in logged_dataset.samples
        if s.get_importance_weight(target_policy) is not None
    ]

    if not valid_samples:
        raise ValueError(f"No valid samples for policy '{target_policy}'")

    samples: List[FreshDrawSample] = []
    for sample in valid_samples:
        prompt_id = sample.prompt_id

        # Must have judge_score to generate synthetic draws
        if sample.judge_score is None:
            raise ValueError(
                f"Sample {prompt_id} has no judge_score. "
                f"Synthetic draws require judge scores to generate correlated samples."
            )

        base_score = sample.judge_score

        for draw_idx in range(draws_per_prompt):
            # Generate correlated score
            noise = np.random.normal(0, 0.1 * (1 - score_correlation))
            score = np.clip(base_score + noise, 0, 1)

            fresh_sample = FreshDrawSample(
                prompt_id=prompt_id,
                target_policy=target_policy,
                judge_score=float(score),
                oracle_label=None,
                response=f"Synthetic response for {prompt_id} draw {draw_idx}",
                draw_idx=draw_idx,
                fold_id=None,
            )
            samples.append(fresh_sample)

    dataset = FreshDrawDataset(
        target_policy=target_policy,
        draws_per_prompt=draws_per_prompt,
        samples=samples,
    )

    logger.info(
        f"Created synthetic fresh draws: {len(samples)} samples, "
        f"{len(valid_samples)} prompts, {draws_per_prompt} draws/prompt"
    )

    return dataset


def load_fresh_draws_auto(
    data_dir: Path,
    policy: str,
    verbose: bool = False,
) -> FreshDrawDataset:
    """
    Load fresh draws from files.

    This function tries to load fresh draws from standard locations:
    1. {data_dir}/{policy}_responses.jsonl
    2. {data_dir}/responses/{policy}_responses.jsonl
    3. {data_dir}/{policy}_fresh.jsonl
    4. {data_dir}/fresh_draws/{policy}.jsonl

    Args:
        data_dir: Directory to search for fresh draw files
        policy: Target policy name
        verbose: Whether to log detailed information

    Returns:
        FreshDrawDataset for the specified policy

    Raises:
        FileNotFoundError: If no fresh draw file found
    """
    # Standard file patterns to check
    possible_files = [
        data_dir / f"{policy}_responses.jsonl",
        data_dir / "responses" / f"{policy}_responses.jsonl",
        data_dir / f"{policy}_fresh.jsonl",
        data_dir / "fresh_draws" / f"{policy}.jsonl",
    ]

    # Try to load from each possible location
    for file_path in possible_files:
        if file_path.exists():
            if verbose:
                logger.info(f"Loading fresh draws from {file_path}")

            try:
                # Load the file
                fresh_samples = []
                with open(file_path, "r") as f:
                    for idx, line in enumerate(f):
                        data = json.loads(line)

                        # Get prompt_id - check top-level first, then metadata, then auto-generate
                        prompt_id = data.get("prompt_id") or data.get(
                            "metadata", {}
                        ).get("prompt_id")
                        if prompt_id is None:
                            # Auto-generate from prompt hash for consistency with logged data
                            # This ensures fresh draws will map to the same prompt_id
                            prompt = data.get("prompt", "")
                            if prompt:
                                import hashlib

                                # Use first 12 chars of SHA256 for readable but unique ID
                                prompt_hash = hashlib.sha256(
                                    prompt.encode()
                                ).hexdigest()[:12]
                                prompt_id = f"prompt_{prompt_hash}"
                            else:
                                # Fallback to index if no prompt either
                                prompt_id = f"fresh_{policy}_{idx:06d}"
                                logger.warning(
                                    f"Fresh draw record {idx} for policy '{policy}' missing both "
                                    f"'prompt_id' and 'prompt'. Using index-based ID '{prompt_id}'. "
                                    f"This will NOT align with logged data for DR mode. "
                                    f"Add explicit prompt_id or prompt text for stability."
                                )

                        # Handle different formats
                        # Check for judge_score properly - don't use 'or' for numeric fields
                        if "judge_score" in data and data["judge_score"] is not None:
                            judge_score = data["judge_score"]
                        elif (
                            "metadata" in data
                            and "judge_score" in data["metadata"]
                            and data["metadata"]["judge_score"] is not None
                        ):
                            judge_score = data["metadata"]["judge_score"]
                        else:
                            # Never fabricate missing data - fail clearly
                            raise ValueError(
                                f"Missing judge_score for prompt_id={prompt_id} "
                                f"in {file_path}. Fresh draws require judge scores."
                            )

                        # Extract oracle_label if present (for calibration)
                        oracle_label = None
                        if "oracle_label" in data and data["oracle_label"] is not None:
                            oracle_label = data["oracle_label"]
                        elif (
                            "metadata" in data
                            and "oracle_label" in data["metadata"]
                            and data["metadata"]["oracle_label"] is not None
                        ):
                            oracle_label = data["metadata"]["oracle_label"]

                        fresh_sample = FreshDrawSample(
                            prompt_id=str(prompt_id),
                            target_policy=policy,
                            response=data.get("response", ""),
                            judge_score=judge_score,
                            oracle_label=oracle_label,
                            draw_idx=data.get("draw_idx", 0),
                            fold_id=data.get("fold_id"),
                        )
                        fresh_samples.append(fresh_sample)

                # Create dataset
                fresh_dataset = FreshDrawDataset(
                    target_policy=policy,
                    draws_per_prompt=1,  # Will be updated based on actual data
                    samples=fresh_samples,
                )

                # Update draws_per_prompt based on actual data
                prompt_counts: Dict[str, int] = {}
                for sample in fresh_samples:
                    prompt_counts[sample.prompt_id] = (
                        prompt_counts.get(sample.prompt_id, 0) + 1
                    )
                if prompt_counts:
                    fresh_dataset.draws_per_prompt = max(prompt_counts.values())

                if verbose:
                    logger.info(
                        f"Loaded {len(fresh_samples)} fresh draws for {policy} "
                        f"({len(prompt_counts)} unique prompts)"
                    )

                return fresh_dataset

            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

    # No file found - raise error with helpful message
    searched_paths = "\n  ".join(str(p) for p in possible_files)
    raise FileNotFoundError(
        f"No fresh draw file found for policy '{policy}'. Searched:\n  {searched_paths}\n"
        f"Fresh draws must be generated from real teacher forcing responses."
    )


def save_fresh_draws_to_jsonl(
    datasets: Dict[str, FreshDrawDataset],
    path: str,
) -> None:
    """Save fresh draw datasets to JSONL file.

    Args:
        datasets: Dict mapping policy names to FreshDrawDataset objects
        path: Output path for JSONL file
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w") as f:
        for policy, dataset in datasets.items():
            for sample in dataset.samples:
                record = {
                    "prompt_id": sample.prompt_id,
                    "target_policy": sample.target_policy,
                    "judge_score": sample.judge_score,
                    "draw_idx": sample.draw_idx,
                }
                if sample.response is not None:
                    record["response"] = sample.response

                f.write(json.dumps(record) + "\n")

    total_samples = sum(len(d.samples) for d in datasets.values())
    logger.info(f"Saved {total_samples} fresh draws to {path_obj}")


def discover_policies_from_fresh_draws(fresh_draws_dir: Path) -> List[str]:
    """Discover target policies from fresh draws directory.

    Looks for files matching patterns:
    - {policy}_responses.jsonl
    - {policy}.jsonl

    Args:
        fresh_draws_dir: Directory containing fresh draw files

    Returns:
        List of discovered policy names

    Raises:
        ValueError: If no fresh draw files found
    """
    fresh_draws_path = Path(fresh_draws_dir)
    if not fresh_draws_path.exists():
        raise ValueError(f"Fresh draws directory not found: {fresh_draws_dir}")

    policies = []

    # Pattern 1: {policy}_responses.jsonl
    for path in fresh_draws_path.glob("*_responses.jsonl"):
        policy = path.stem.replace("_responses", "")
        policies.append(policy)

    # Pattern 2: {policy}.jsonl (if no _responses files found)
    if not policies:
        for path in fresh_draws_path.glob("*.jsonl"):
            # Skip files that don't look like policy files
            if path.stem not in ["dataset", "data", "logs"]:
                policies.append(path.stem)

    if not policies:
        raise ValueError(
            f"No fresh draw files found in {fresh_draws_dir}. "
            f"Expected files like 'policy_a_responses.jsonl' or 'policy_a.jsonl'"
        )

    logger.info(f"Discovered {len(policies)} policies from fresh draws: {policies}")
    return sorted(policies)


def compute_response_covariates(
    fresh_draws: FreshDrawDataset,
    covariate_names: Optional[List[str]] = None,
) -> FreshDrawDataset:
    """Compute covariates for fresh draws based on response text.

    This function computes response-level covariates (like response_length)
    and stores them in each sample's metadata field. This is needed for
    DR estimators and Direct Method to use covariates properly.

    Args:
        fresh_draws: FreshDrawDataset to augment with covariates
        covariate_names: List of covariate names to compute. Currently supported:
            - "response_length": word count (len(response.split())) - matches calibration

    Returns:
        New FreshDrawDataset with covariates computed and stored in metadata

    Example:
        >>> fresh_draws = load_fresh_draws_auto(data_dir, "policy_a")
        >>> fresh_draws_with_covs = compute_response_covariates(
        ...     fresh_draws, covariate_names=["response_length"]
        ... )
        >>> # Now fresh_draws_with_covs.samples[i].metadata["response_length"] exists
    """
    if covariate_names is None:
        covariate_names = []

    if not covariate_names:
        logger.debug("No covariate names specified, returning unchanged")
        return fresh_draws

    # Compute covariates for each sample
    updated_samples = []
    for sample in fresh_draws.samples:
        # Create a copy of the sample's metadata
        new_metadata = dict(sample.metadata) if sample.metadata else {}

        for cov_name in covariate_names:
            if cov_name == "response_length":
                # Compute response_length matching the formula in calibration/dataset.py
                # Uses word count (len(response.split())) to match calibration exactly
                if sample.response is not None:
                    # CRITICAL: Must match calibration/dataset.py AUTO_COMPUTABLE_COVARIATES
                    # which uses: lambda sample: float(len(sample.response.split()))
                    word_count = len(sample.response.split())
                    new_metadata["response_length"] = float(word_count)
                else:
                    raise ValueError(
                        f"Cannot compute response_length for sample {sample.prompt_id} "
                        f"draw {sample.draw_idx}: response is None"
                    )
            else:
                raise ValueError(
                    f"Unsupported covariate: {cov_name}. "
                    f"Currently supported: ['response_length']"
                )

        # Create new sample with updated metadata
        updated_sample = sample.model_copy(update={"metadata": new_metadata})
        updated_samples.append(updated_sample)

    # Create new dataset with updated samples
    updated_dataset = FreshDrawDataset(
        target_policy=fresh_draws.target_policy,
        draws_per_prompt=fresh_draws.draws_per_prompt,
        samples=updated_samples,
    )

    logger.info(
        f"Computed {len(covariate_names)} covariates for {len(updated_samples)} "
        f"fresh draw samples (policy={fresh_draws.target_policy})"
    )

    return updated_dataset
