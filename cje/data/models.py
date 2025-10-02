"""Data models for CJE using Pydantic."""

from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING, ForwardRef, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import numpy as np


class LogProbStatus(Enum):
    """Status of log probability computation."""

    SUCCESS = "success"
    API_ERROR = "api_error"
    TOKEN_BOUNDARY_ERROR = "token_boundary_error"
    TOKEN_LIMIT_EXCEEDED = "token_limit_exceeded"
    EMPTY_RESPONSE = "empty_response"


class LogProbResult(BaseModel):
    """Result of log probability computation with explicit error handling."""

    value: Optional[float] = Field(
        None, description="Log probability value if successful"
    )
    status: LogProbStatus = Field(
        LogProbStatus.API_ERROR, description="Computation status"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @property
    def is_valid(self) -> bool:
        """Check if computation succeeded."""
        return self.status == LogProbStatus.SUCCESS and self.value is not None


class Sample(BaseModel):
    """A single sample for CJE analysis."""

    prompt_id: str = Field(..., description="Unique identifier for the prompt")
    prompt: str = Field(..., description="Input prompt/context")
    response: str = Field(..., description="Generated response")
    reward: Optional[float] = Field(
        None, ge=0, le=1, description="Calibrated reward [0,1]"
    )
    base_policy_logprob: Optional[float] = Field(
        None, description="Log prob under base policy"
    )
    target_policy_logprobs: Dict[str, Optional[float]] = Field(
        ..., description="Log probs under target policies (None for failures)"
    )
    judge_score: Optional[float] = Field(
        None, ge=0, le=1, description="Judge evaluation score [0,1]"
    )
    oracle_label: Optional[float] = Field(
        None, ge=0, le=1, description="Ground truth oracle label [0,1]"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (timestamps, model info, etc.)",
    )

    @field_validator("base_policy_logprob")
    def validate_base_policy_logprob(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v > 0:
            raise ValueError(f"Log probability must be <= 0, got {v}")
        return v

    @field_validator("target_policy_logprobs")
    def validate_target_policy_logprobs(
        cls, v: Dict[str, Optional[float]]
    ) -> Dict[str, Optional[float]]:
        for policy, logprob in v.items():
            if logprob is not None and logprob > 0:
                raise ValueError(
                    f"Log probability for {policy} must be <= 0, got {logprob}"
                )
        return v

    def get_importance_weight(self, target_policy: str) -> Optional[float]:
        """Compute importance weight for a target policy."""
        if self.base_policy_logprob is None:
            return None
        target_lp = self.target_policy_logprobs.get(target_policy)
        if target_lp is None:
            return None
        return float(np.exp(target_lp - self.base_policy_logprob))


class Dataset(BaseModel):
    """A dataset for CJE analysis.

    This is a pure data container following the Single Responsibility Principle.
    For loading data, use DatasetFactory or DatasetLoader.
    """

    samples: List[Sample] = Field(..., min_length=1)
    target_policies: List[str] = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("target_policies")
    def validate_policies_exist(cls, v: List[str], info: Any) -> List[str]:
        """Ensure target policies exist in samples."""
        if "samples" in info.data:
            all_policies = set()
            for sample in info.data["samples"]:
                all_policies.update(sample.target_policy_logprobs.keys())

            missing = set(v) - all_policies
            if missing:
                raise ValueError(f"Target policies not found in data: {missing}")
        return v

    def filter_valid_samples(self, target_policy: str) -> List[Sample]:
        """Get samples with valid data for a specific target policy."""
        valid_samples = []
        for sample in self.samples:
            if (
                sample.base_policy_logprob is not None
                and sample.target_policy_logprobs.get(target_policy) is not None
            ):
                valid_samples.append(sample)
        return valid_samples

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def summary(self) -> Dict[str, Any]:
        """Get dataset summary statistics."""
        rewards = [s.reward for s in self.samples if s.reward is not None]
        valid_counts = {policy: 0 for policy in self.target_policies}

        for sample in self.samples:
            for policy in self.target_policies:
                if sample.get_importance_weight(policy) is not None:
                    valid_counts[policy] += 1

        return {
            "n_samples": self.n_samples,
            "target_policies": self.target_policies,
            "reward_mean": np.mean(rewards) if rewards else None,
            "reward_std": np.std(rewards) if rewards else None,
            "valid_samples_per_policy": valid_counts,
        }


class EstimationResult(BaseModel):
    """Result from a CJE estimator.

    Influence functions are first-class outputs for statistical inference.
    Diagnostics contain quality metrics and health indicators.
    Metadata contains configuration and context.
    """

    # Core results
    estimates: np.ndarray = Field(..., description="Point estimates for each policy")
    standard_errors: np.ndarray = Field(..., description="Standard errors")
    n_samples_used: Dict[str, int] = Field(..., description="Valid samples per policy")
    method: str = Field(..., description="Estimation method used")

    # First-class statistical artifact
    influence_functions: Optional[Dict[str, np.ndarray]] = Field(
        None,
        description="Influence functions for each policy (when store_influence=True)",
    )

    # Quality metrics
    diagnostics: Optional[Union["IPSDiagnostics", "DRDiagnostics"]] = Field(
        None, description="Diagnostic information (IPSDiagnostics or DRDiagnostics)"
    )

    # Configuration and context
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration parameters and context (dataset path, timestamp, etc.)",
    )

    model_config = {"arbitrary_types_allowed": True}

    def confidence_interval(self, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Compute confidence intervals (returns lower and upper arrays).

        Args:
            alpha: Significance level (default 0.05 for 95% CI)

        Returns:
            Tuple of (lower_bounds, upper_bounds) as numpy arrays
        """
        from scipy import stats

        z = stats.norm.ppf(1 - alpha / 2)
        lower = self.estimates - z * self.standard_errors
        upper = self.estimates + z * self.standard_errors
        return lower, upper

    def ci(self, alpha: float = 0.05) -> List[Tuple[float, float]]:
        """Convenience method for confidence intervals as list of tuples.

        Args:
            alpha: Significance level (default 0.05 for 95% CI)

        Returns:
            List of (lower, upper) tuples, one per policy

        Example:
            >>> result.ci()
            [(0.701, 0.745), (0.680, 0.720)]
        """
        lower, upper = self.confidence_interval(alpha)
        return [(float(l), float(u)) for l, u in zip(lower, upper)]

    def best_policy(self) -> int:
        """Get index of best policy by point estimate."""
        return int(np.argmax(self.estimates))

    def compare_policies(
        self, idx1: int, idx2: int, alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Compare two policies using influence functions when available."""
        diff = self.estimates[idx1] - self.estimates[idx2]

        # Use influence functions for proper variance estimation
        if self.influence_functions and "target_policies" in self.metadata:
            policies = self.metadata["target_policies"]
            if idx1 < len(policies) and idx2 < len(policies):
                p1 = policies[idx1]
                p2 = policies[idx2]

                if p1 in self.influence_functions and p2 in self.influence_functions:
                    # Compute variance of difference using influence functions
                    if1 = self.influence_functions[p1]
                    if2 = self.influence_functions[p2]

                    # Ensure same length (should be aligned)
                    if len(if1) == len(if2):
                        diff_if = if1 - if2
                        se_diff = float(np.std(diff_if, ddof=1) / np.sqrt(len(diff_if)))
                    else:
                        # Fall back to conservative estimate if lengths mismatch
                        se_diff = np.sqrt(
                            self.standard_errors[idx1] ** 2
                            + self.standard_errors[idx2] ** 2
                        )
                else:
                    # Fall back if policies not found
                    se_diff = np.sqrt(
                        self.standard_errors[idx1] ** 2
                        + self.standard_errors[idx2] ** 2
                    )
            else:
                # Fall back if indices out of range
                se_diff = np.sqrt(
                    self.standard_errors[idx1] ** 2 + self.standard_errors[idx2] ** 2
                )
        else:
            # Conservative estimate ignoring covariance
            se_diff = np.sqrt(
                self.standard_errors[idx1] ** 2 + self.standard_errors[idx2] ** 2
            )

        z_score = diff / se_diff if se_diff > 0 else 0

        from scipy import stats

        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return {
            "difference": diff,
            "se_difference": se_diff,
            "z_score": z_score,
            "p_value": p_value,
            "significant": p_value < alpha,
            "used_influence": self.influence_functions is not None
            and se_diff
            != np.sqrt(
                self.standard_errors[idx1] ** 2 + self.standard_errors[idx2] ** 2
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        ci_lower, ci_upper = self.confidence_interval()

        result = {
            "method": self.method,
            "estimates": self.estimates.tolist(),
            "standard_errors": self.standard_errors.tolist(),
            "n_samples_used": self.n_samples_used,
            "confidence_intervals": {
                "alpha": 0.05,
                "lower": ci_lower.tolist(),
                "upper": ci_upper.tolist(),
            },
        }

        # Add influence functions if present (convert to lists for JSON)
        if self.influence_functions:
            result["influence_functions"] = {
                policy: ifs.tolist() for policy, ifs in self.influence_functions.items()
            }

        # Add diagnostics if present
        if self.diagnostics:
            result["diagnostics"] = self.diagnostics.to_dict()

        # Add metadata if non-empty
        if self.metadata:
            result["metadata"] = self.metadata

        # Add per-policy results if policies are specified
        if "target_policies" in self.metadata:
            policies = self.metadata["target_policies"]
            result["per_policy_results"] = {}
            for i, policy in enumerate(policies):
                result["per_policy_results"][policy] = {
                    "estimate": float(self.estimates[i]),
                    "standard_error": float(self.standard_errors[i]),
                    "ci_lower": float(ci_lower[i]),
                    "ci_upper": float(ci_upper[i]),
                    "n_samples": self.n_samples_used.get(policy, 0),
                }

        # Include diagnostics if available
        if "diagnostics" in self.metadata:
            result["diagnostics"] = self.metadata["diagnostics"]

        return result


# Import at the end to resolve forward references
from ..diagnostics import IPSDiagnostics, DRDiagnostics

# Update forward references - compatible with both Pydantic v1 and v2
if hasattr(EstimationResult, "model_rebuild"):
    # Pydantic v2
    EstimationResult.model_rebuild()
elif hasattr(EstimationResult, "update_forward_refs"):
    # Pydantic v1
    EstimationResult.update_forward_refs()
