"""Data models for CJE using Pydantic."""

import logging
import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import date, datetime, time
from enum import Enum
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union, cast
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
import numpy as np

logger = logging.getLogger(__name__)

RESULT_SCHEMA_VERSION = "cje.estimation_result/v2"

# metadata["calibration_label_sources"] entry for labels that came from
# calibration_data_path (the others are policy names).
_CALIBRATION_DATA_SOURCE = "calibration_data"

# compare_policies keys re-derived from the result on every call; stored
# portable comparisons omit them so a reversed pair is never mislabelled.
_DERIVED_COMPARISON_KEYS = (
    "gate_flagged",
    "policy1",
    "policy2",
    "transport_unverified",
    "conditional_on_transport",
)


def _join_names(names: List[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _format_p_value(p_value: float) -> str:
    """'p=0.22'; three decimals below 0.1 so values near 0.05 stay
    distinguishable ('p=0.049', 'p=0.052', 'p=0.004'); 'p<0.001'."""
    if not np.isfinite(p_value):
        return "p=nan"
    if p_value < 0.001:
        return "p<0.001"
    if abs(p_value - 0.05) < 0.0005:
        return f"p={p_value:.4f}"
    if p_value < 0.1:
        return f"p={p_value:.3f}"
    return f"p={p_value:.2f}"


def _validate_alpha(alpha: Any) -> float:
    """Return a finite significance level strictly between zero and one."""
    if isinstance(alpha, bool) or not isinstance(
        alpha, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"alpha must be a finite number in (0, 1), got {alpha!r}")
    numeric = float(alpha)
    if not np.isfinite(numeric) or not 0.0 < numeric < 1.0:
        raise ValueError(f"alpha must be a finite number in (0, 1), got {alpha!r}")
    return numeric


class InferenceUnavailableError(RuntimeError):
    """Raised when a serialized result omitted required inference state."""


class ResultUnits(BaseModel):
    """Explicit public and internal units for a result and its artifacts."""

    estimand: str
    output_scale: Dict[str, Any]
    judge_input_scale: Dict[str, Any]
    internal_scale: Dict[str, Any] = Field(
        default_factory=lambda: {"min": 0.0, "max": 1.0, "is_identity": True}
    )


@dataclass
class CIInfo:
    """Typed record of how a result's confidence intervals were computed.

    Written by the estimator at result-assembly time. The metadata mirrors
    (``bootstrap_ci`` / ``degrees_of_freedom``) remain the serialized source
    of truth this release; CIInfo is the raise-free typed accessor.

    Attributes:
        method: "percentile" (bootstrap percentile CIs, precomputed) or
            "t" (t-based CIs computed from per-policy degrees of freedom).
        alpha: Significance level the stored intervals were computed at
            (only binding for method="percentile"; t-based CIs are computed
            at the caller's alpha).
        lower: Per-policy lower bounds (method="percentile" only).
        upper: Per-policy upper bounds (method="percentile" only).
        df_per_policy: Per-policy degrees-of-freedom info, same shape as
            ``metadata["degrees_of_freedom"]`` (method="t" only).
    """

    method: str
    alpha: float
    lower: Optional[List[float]] = None
    upper: Optional[List[float]] = None
    df_per_policy: Optional[Dict[str, Any]] = None


@dataclass
class GateResult:
    """Typed view of one policy's reliability gate.

    Derived from ``metadata["reliability_gates"]`` (the source of truth this
    release) via ``EstimationResult.gates``.
    """

    policy: str
    flagged: bool
    refuse_level_claims: bool
    reasons: List[str]


@dataclass
class PolicyVerdict:
    """Result of ``EstimationResult.best_policy()``.

    Attributes:
        name: The selected policy. With ``reliable_only=True`` (default)
            this is the best policy that PASSED the reliability gates; a
            flagged raw argmax is demoted to ``runner_up``. Callers may
            explicitly request the raw point-estimate argmax with
            ``reliable_only=False``, qualified through ``flagged``.
        index: Index of ``name`` in the estimates array.
        estimate: Point estimate of ``name``.
        flagged: Whether the returned policy failed the reliability gates.
        all_flagged: True when no policy with a usable estimate passed the
            gates (the returned argmax should not be crowned).
        runner_up: The raw point-estimate argmax when it was flagged and
            demoted in favor of ``name`` (it beat ``name`` on point
            estimate but failed the gates). None otherwise.
        runner_up_reasons: Why ``runner_up`` was flagged (gate reasons and/or
            CRITICAL diagnostics status). None unless ``runner_up`` is set.
        decision_ready: True only when ``name`` beats every other usable
            policy in the paired comparison (each 95% CI wholly above 0,
            and above any PASS-audit transport margin it relies on), no
            policy in the comparison relies on an unverified calibration
            transfer (``metadata["transport_unverified"]``) or failed the
            gates, ``name`` was not reached by demoting a flagged leader,
            and the result records label provenance (0.9.2+). Requiring
            every pair is an intersection-union test, so no multiplicity
            adjustment applies. True is not a guarantee: read the gates and
            audits before acting.
        decision_note: One sentence saying why, and what to do next.
    """

    name: str
    index: int
    estimate: float
    flagged: bool
    all_flagged: bool = False
    runner_up: Optional[str] = None
    runner_up_reasons: Optional[List[str]] = None
    decision_ready: bool = False
    decision_note: str = ""


class Sample(BaseModel):
    """A single sample for CJE analysis."""

    prompt_id: str = Field(..., description="Unique identifier for the prompt")
    prompt: str = Field(..., description="Input prompt/context")
    response: str = Field(..., description="Generated response")
    reward: Optional[float] = Field(
        None, ge=0, le=1, description="Calibrated reward [0,1]"
    )
    judge_score: Optional[float] = Field(
        None, ge=0, le=1, description="Judge evaluation score [0,1]"
    )
    oracle_label: Optional[float] = Field(
        None, ge=0, le=1, description="Ground truth oracle label [0,1]"
    )
    row_id: Optional[str] = Field(
        default=None, description="Stable source-local row identity"
    )
    observation_id: Optional[str] = Field(
        default=None, description="Optional cross-source response identity"
    )
    source_id: Optional[str] = Field(default=None, description="Logical input source")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (timestamps, model info, etc.)",
    )


class Dataset(BaseModel):
    """A dataset for CJE analysis.

    This is a pure data container following the Single Responsibility Principle.
    For loading data, use DatasetLoader.
    """

    samples: List[Sample] = Field(..., min_length=1)
    # May be empty: calibration-only datasets (judge + oracle pairs, e.g. a
    # minimal calibration_data_path file) carry no policy information.
    target_policies: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.samples)


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
    # Raw pseudo-outcome deviations. For augmented policies under
    # representative labels the analytic SE also scales the labelled
    # clusters' CRV1 part by n_L / (n_L - q) (issue #60), so it no longer
    # equals sqrt(CRV1(influence) + oracle variance) for those policies.
    influence_functions: Optional[Dict[str, np.ndarray]] = Field(
        None,
        description=(
            "Influence functions for each policy (when store_influence=True): "
            "raw pseudo-outcome deviations. For augmented policies under "
            "representative labels the analytic SE is not their CRV1 SE plus "
            "the oracle variance; see metadata['degrees_of_freedom']."
        ),
    )

    # Paired bootstrap replicates (written by the bootstrap inference path).
    # Each row is one joint cluster resample + calibrator refit, so column
    # differences carry the full paired uncertainty (including calibrator
    # noise) that per-policy plug-in influence functions miss.
    bootstrap_samples: Optional[np.ndarray] = Field(
        default=None,
        description=(
            "Bootstrap replicate matrix of shape (B, P); columns follow "
            "metadata['target_policies']. Used by compare_policies for "
            "paired difference inference. Included only in full-detail "
            "serialization."
        ),
    )

    # Quality metrics
    diagnostics: Optional["DirectDiagnostics"] = Field(
        None, description="Diagnostic information (DirectDiagnostics)"
    )

    # Calibrator for transportability audits
    calibrator: Optional[Any] = Field(
        default=None,
        description="Fitted calibrator (sklearn-compatible) for transportability audits",
    )

    # Configuration and context
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration parameters and context (dataset path, timestamp, etc.)",
    )

    # Typed CI record written by the estimator (metadata mirrors stay the
    # serialized source of truth this release)
    ci_info: Optional[CIInfo] = Field(
        default=None,
        description="How confidence intervals were computed (typed accessor)",
    )
    units: Optional[ResultUnits] = Field(
        default=None,
        description="Estimand and scale contract shared by every result artifact",
    )

    # best_policy() logs its demotion warning once per result instance;
    # summary()/renderers re-deriving the verdict must not repeat it.
    _demotion_warned: bool = PrivateAttr(default=False)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("n_samples_used", mode="before")
    @classmethod
    def validate_raw_sample_counts(cls, value: Any) -> Any:
        """Reject booleans before Pydantic can coerce them to integers."""
        if not isinstance(value, dict):
            return value
        for policy, count in value.items():
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
                raise ValueError(f"n_samples_used[{policy!r}] must be an integer")
        return value

    @model_validator(mode="after")
    def validate_result_contract(self) -> "EstimationResult":
        estimates = np.asarray(self.estimates, dtype=float)
        standard_errors = np.asarray(self.standard_errors, dtype=float)
        if estimates.ndim != 1 or standard_errors.ndim != 1:
            raise ValueError("estimates and standard_errors must be one-dimensional")
        if estimates.shape != standard_errors.shape:
            raise ValueError(
                "estimates and standard_errors must have the same length "
                f"({len(estimates)} != {len(standard_errors)})"
            )
        if np.any(np.isinf(estimates)) or np.any(np.isinf(standard_errors)):
            raise ValueError("estimates and standard_errors cannot contain infinity")
        finite_se = standard_errors[np.isfinite(standard_errors)]
        if np.any(finite_se < 0):
            raise ValueError("standard_errors must be nonnegative")
        policies = self.target_policies
        if policies:
            if len(policies) != len(estimates):
                raise ValueError(
                    "metadata['target_policies'] must align with estimates "
                    f"({len(policies)} != {len(estimates)})"
                )
            if len(set(policies)) != len(policies):
                raise ValueError("metadata['target_policies'] must be unique")
            if set(self.n_samples_used) != set(policies):
                raise ValueError(
                    "n_samples_used keys must exactly match target_policies"
                )
        for policy, count in self.n_samples_used.items():
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
                raise ValueError(f"n_samples_used[{policy!r}] must be an integer")
            if int(count) < 0:
                raise ValueError(f"n_samples_used[{policy!r}] must be nonnegative")

        if self.bootstrap_samples is not None:
            matrix = np.asarray(self.bootstrap_samples, dtype=float)
            if matrix.ndim != 2 or matrix.shape[1] != len(estimates):
                raise ValueError(
                    "bootstrap_samples must have shape (replicates, policies) "
                    f"with {len(estimates)} policy columns, got {matrix.shape}"
                )
            if np.any(np.isinf(matrix)):
                raise ValueError("bootstrap_samples cannot contain infinity")
        if self.influence_functions is not None:
            if policies and not set(self.influence_functions).issubset(set(policies)):
                raise ValueError("influence_functions contains an unknown policy")
            for policy, values in self.influence_functions.items():
                influence = np.asarray(values, dtype=float)
                if influence.ndim != 1:
                    raise ValueError(
                        f"influence_functions[{policy!r}] must be one-dimensional"
                    )
                if np.any(np.isinf(influence)):
                    raise ValueError(
                        f"influence_functions[{policy!r}] cannot contain infinity"
                    )
        if self.ci_info is not None:
            if not 0 < float(self.ci_info.alpha) < 1:
                raise ValueError("ci_info.alpha must lie in (0, 1)")
            if self.ci_info.lower is not None and len(self.ci_info.lower) != len(
                estimates
            ):
                raise ValueError("ci_info.lower must align with estimates")
            if self.ci_info.upper is not None and len(self.ci_info.upper) != len(
                estimates
            ):
                raise ValueError("ci_info.upper must align with estimates")
            for name, bounds in (
                ("lower", self.ci_info.lower),
                ("upper", self.ci_info.upper),
            ):
                if bounds is not None and np.any(
                    np.isinf(np.asarray(bounds, dtype=float))
                ):
                    raise ValueError(f"ci_info.{name} cannot contain infinity")
        self.estimates = estimates
        self.standard_errors = standard_errors
        return self

    @property
    def target_policies(self) -> List[str]:
        """Policy names in estimate order (read from metadata; [] if absent)."""
        if not isinstance(self.metadata, dict):
            return []
        policies = self.metadata.get("target_policies")
        if not policies:
            return []
        return [str(p) for p in policies]

    @property
    def gates(self) -> Dict[str, GateResult]:
        """Typed per-policy reliability gates.

        Derived from ``metadata["reliability_gates"]`` (the source of truth
        this release). Empty dict when no gates were recorded.
        """
        raw = (
            self.metadata.get("reliability_gates")
            if isinstance(self.metadata, dict)
            else None
        )
        gates: Dict[str, GateResult] = {}
        if not isinstance(raw, dict):
            return gates
        for policy, info in raw.items():
            if not isinstance(info, dict):
                continue
            gates[str(policy)] = GateResult(
                policy=str(policy),
                flagged=bool(info.get("flagged", False)),
                refuse_level_claims=bool(info.get("refuse_level_claims", False)),
                reasons=[str(r) for r in (info.get("reasons") or [])],
            )
        return gates

    def _warn_alpha_mismatch(self, alpha: float, stored_alpha: float) -> None:
        """Warn when a caller's alpha cannot be honored by stored bootstrap CIs."""
        if abs(alpha - stored_alpha) > 1e-9:
            logger.warning(
                f"confidence_interval(alpha={alpha}) requested, but bootstrap "
                f"percentile CIs were precomputed at alpha={stored_alpha}; "
                f"returning the stored intervals. Re-run the analysis to "
                f"bootstrap at a different alpha."
            )

    def _t_based_ci(
        self, alpha: float, df_info: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """t-based CIs from per-policy DF info (z fallback per policy)."""
        from scipy import stats

        policies = self.target_policies
        z = stats.norm.ppf(1 - alpha / 2)
        lower_list: List[float] = []
        upper_list: List[float] = []
        for i in range(len(self.estimates)):
            policy = policies[i] if i < len(policies) else None
            crit = float(z)
            if policy is not None and df_info.get(policy) is not None:
                policy_info = df_info[policy]
                df = policy_info.get("df") if isinstance(policy_info, dict) else None
                if df is not None and df > 0:
                    # Use t-critical value with finite DF
                    crit = float(stats.t.ppf(1 - alpha / 2, df))
            lower_list.append(float(self.estimates[i] - crit * self.standard_errors[i]))
            upper_list.append(float(self.estimates[i] + crit * self.standard_errors[i]))
        return np.array(lower_list), np.array(upper_list)

    def confidence_interval(self, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Compute confidence intervals (returns lower and upper arrays).

        Priority order:
        1. ci_info — the typed CI record written by the estimator:
           precomputed bootstrap percentile CIs, or t-based CIs from
           per-policy degrees of freedom
        2. Bootstrap percentile CIs from metadata (older results)
        3. t-based CIs from metadata degrees-of-freedom info (older results)
        4. z-based CIs for large-sample approximation (fallback)

        Note: Bootstrap percentile CIs are precomputed at their stored alpha;
        a differing caller alpha logs a warning and returns the stored
        intervals (re-run the analysis to bootstrap at a different alpha).

        Args:
            alpha: Significance level (default 0.05 for 95% CI)

        Returns:
            Tuple of (lower_bounds, upper_bounds) as numpy arrays
        """
        from scipy import stats

        alpha = _validate_alpha(alpha)

        # Priority 1: the typed CI record written at result-assembly time
        if self.ci_info is not None:
            ci = self.ci_info
            if (
                ci.method == "percentile"
                and ci.lower is not None
                and ci.upper is not None
            ):
                self._warn_alpha_mismatch(alpha, ci.alpha)
                return np.array(ci.lower, dtype=float), np.array(ci.upper, dtype=float)
            if ci.method == "t" and ci.df_per_policy is not None:
                return self._t_based_ci(alpha, ci.df_per_policy)

        # Priority 2 (legacy metadata sniffing): bootstrap percentile CIs.
        # Bootstrap with θ̂_aug provides ~95% coverage via AIPW-style debiasing
        if isinstance(self.metadata, dict) and "bootstrap_ci" in self.metadata:
            boot_ci = self.metadata["bootstrap_ci"]
            if boot_ci.get("method") == "percentile":
                stored_alpha = boot_ci.get("alpha")
                if stored_alpha is not None:
                    self._warn_alpha_mismatch(alpha, float(stored_alpha))
                # Use pre-computed percentile intervals
                lower = np.array(boot_ci["lower"])
                upper = np.array(boot_ci["upper"])
                return lower, upper

        # Priority 3 (legacy): t-based CIs with degrees of freedom
        if (
            isinstance(self.metadata, dict)
            and "degrees_of_freedom" in self.metadata
            and self.metadata["degrees_of_freedom"] is not None
            and "target_policies" in self.metadata
        ):
            df_info = self.metadata["degrees_of_freedom"]
            return self._t_based_ci(alpha, df_info)

        # Priority 4: z-based CIs (asymptotically valid for large n)
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
        try:
            _validate_alpha(alpha)
        except ValueError as exc:
            raise ValueError(
                f"{exc}: ci(alpha) takes a significance level (0.05 for 95% "
                "intervals) and returns one (lower, upper) tuple per policy in "
                "target_policies order; by name, use "
                "dict(zip(result.target_policies, result.ci()))."
            ) from None
        lower, upper = self.confidence_interval(alpha)
        return [(float(l), float(u)) for l, u in zip(lower, upper)]

    def _unreliable_policies(self) -> Set[str]:
        """Policies flagged by the reliability gates or CRITICAL status."""
        flagged = {name for name, gate in self.gates.items() if gate.flagged}
        status_per_policy = getattr(self.diagnostics, "status_per_policy", None)
        if status_per_policy:
            for policy, status in status_per_policy.items():
                value = getattr(status, "value", status)
                if value == "critical":
                    flagged.add(policy)
        return flagged

    def _flagged_reasons(self, policy: str) -> List[str]:
        """Why ``policy`` failed the reliability gates (for loud demotions)."""
        reasons: List[str] = []
        gate = self.gates.get(policy)
        if gate is not None and gate.flagged:
            reasons.extend(gate.reasons)
        status_per_policy = getattr(self.diagnostics, "status_per_policy", None)
        if status_per_policy:
            status = status_per_policy.get(policy)
            if getattr(status, "value", status) == "critical":
                reasons.append("diagnostics status CRITICAL")
        return reasons or ["flagged by the reliability gates"]

    def best_policy(self, reliable_only: bool = True) -> PolicyVerdict:
        """Best policy, demoting winners that failed the reliability gates.

        The raw point-estimate argmax can be an adversarial policy the gates
        flagged as unreliable (verified on the bundled arena sample, where
        the 'unhelpful' policy won the raw argmax). With reliable_only=True
        (default) the verdict names the best policy that PASSED the gates.
        The demotion is never silent: the flagged raw argmax is recorded in
        ``runner_up`` with its gate reasons in ``runner_up_reasons``, and
        ``summary()`` reports the divergence. If every policy is flagged,
        the argmax is returned with ``all_flagged=True`` — do not crown it.
        Pass ``reliable_only=False`` for the raw argmax with its gate flag.

        ``decision_ready`` is True only when the verdict beats every other
        usable policy in the paired comparison with no unverified
        calibration transfer, gate failure or gate demotion involved (see
        ``PolicyVerdict``); ``decision_note`` says why and what to do. A
        demotion is never a decision: the demoted leader may be the better
        policy. The verdict itself ranks point estimates.

        Note: This replaced the 0.4.x ``best_policy() -> int`` (naive
        argmax index) in 0.5.0. Use ``verdict.index`` for the old value.

        Args:
            reliable_only: When True (default), select only among policies
                that passed the reliability gates (and are not CRITICAL).
                When False, return the raw point-estimate argmax with its
                limitations.

        Returns:
            PolicyVerdict (name, index, estimate, flagged, all_flagged,
            runner_up, runner_up_reasons, decision_ready, decision_note).

        Raises:
            ValueError: If there are no estimates, no target_policies
                metadata, or every estimate is NaN.
        """
        verdict = self._best_policy_verdict(reliable_only)
        ready, note = self._decision_readiness(verdict)
        verdict.decision_ready = ready
        verdict.decision_note = note
        return verdict

    def _best_policy_verdict(self, reliable_only: bool) -> PolicyVerdict:
        """Point-estimate verdict with gate demotion (see ``best_policy``)."""
        policies = self.target_policies
        if len(self.estimates) == 0 or not policies:
            raise ValueError(
                "best_policy() requires estimates and metadata['target_policies']"
            )
        estimates = np.asarray(self.estimates, dtype=float)
        if np.all(np.isnan(estimates)):
            raise ValueError(
                "No usable estimates: every policy estimate is NaN " "(see diagnostics)"
            )

        flagged = self._unreliable_policies()
        best_idx = int(np.nanargmax(estimates))
        best_name = policies[best_idx]
        best_flagged = best_name in flagged

        usable = [
            (i, policy)
            for i, policy in enumerate(policies)
            if i < len(estimates) and not np.isnan(estimates[i])
        ]
        reliable = [(i, policy) for i, policy in usable if policy not in flagged]
        all_flagged = not reliable

        if not reliable_only or not best_flagged:
            return PolicyVerdict(
                name=best_name,
                index=best_idx,
                estimate=float(estimates[best_idx]),
                flagged=best_flagged,
                all_flagged=all_flagged,
                runner_up=None,
            )

        if all_flagged:
            # No policy passed the gates: return the argmax, loudly marked.
            return PolicyVerdict(
                name=best_name,
                index=best_idx,
                estimate=float(estimates[best_idx]),
                flagged=True,
                all_flagged=True,
                runner_up=None,
            )

        # Demote the flagged argmax: the trophy goes to the best gate-passing
        # policy (ties broken by policy name for deterministic behavior). The
        # divergence is loud — the raw argmax and why it was flagged travel
        # with the verdict.
        _, rel_name = max((float(estimates[i]), policy) for i, policy in reliable)
        rel_idx = policies.index(rel_name)
        runner_up_reasons = self._flagged_reasons(best_name)
        if not self._demotion_warned:
            self._demotion_warned = True
            logger.warning(
                f"best_policy(): raw argmax '{best_name}' was flagged "
                f"({'; '.join(runner_up_reasons)}); returning best reliable "
                f"policy '{rel_name}'. Pass reliable_only=False for the raw "
                f"argmax."
            )
        return PolicyVerdict(
            name=rel_name,
            index=rel_idx,
            estimate=float(estimates[rel_idx]),
            flagged=False,
            all_flagged=False,
            runner_up=best_name,
            runner_up_reasons=runner_up_reasons,
        )

    # ------------------------------------------------------------------
    # Borrowed calibration (estimates that rely on calibration transport)
    # ------------------------------------------------------------------

    def _transport_provenance_known(self) -> bool:
        """Whether the result records which policies rely on transport.

        ``analyze_dataset`` (0.9.2+) writes ``metadata["transport_unverified"]``.
        Results saved by earlier versions, or built by calling an estimator
        directly, do not; for them the reliance is unknown, not absent.
        """
        return isinstance(self.metadata, dict) and isinstance(
            self.metadata.get("transport_unverified"), (list, tuple)
        )

    def _transport_unverified(self) -> List[str]:
        """Policies whose estimate relies on an unverified calibration transfer.

        Read from ``metadata["transport_unverified"]`` (written by
        ``analyze_dataset``); empty when the result does not record it (see
        ``_transport_provenance_known``).
        """
        if not self._transport_provenance_known():
            return []
        return [str(policy) for policy in self.metadata["transport_unverified"]]

    def _transport_status(self, policy: str) -> str:
        audits = self.metadata.get("transport_audits")
        audit = audits.get(policy) if isinstance(audits, dict) else None
        if isinstance(audit, dict) and audit.get("status"):
            return str(audit["status"])
        return "NOT_CHECKED"

    def _own_label_count(self, policy: str) -> int:
        own = self.metadata.get("own_oracle_labels_by_policy")
        if isinstance(own, dict):
            try:
                return int(own.get(policy, 0) or 0)
            except (TypeError, ValueError):
                return 0
        return 0

    def _point_route(self, policy: str) -> Optional[str]:
        point = self.metadata.get("point_estimator")
        routes = point.get("routes") if isinstance(point, dict) else None
        policies = self.target_policies
        if not isinstance(routes, (list, tuple)) or policy not in policies:
            return None
        index = policies.index(policy)
        return str(routes[index]) if index < len(routes) else None

    def _relies_on_transport(self, policy: str) -> bool:
        """Whether ``policy``'s estimate assumes a calibration transfers to it.

        True with no oracle labels of its own. With its own labels, the
        augmented (residual-corrected) and direct-oracle routes do not rely
        on transport; a plug-in route does, unless the calibration was fit on
        this policy's labels alone.
        """
        if self._own_label_count(policy) == 0:
            return True
        if self._point_route(policy) in ("augmented", "direct_oracle"):
            return False
        raw = self.metadata.get("calibration_label_sources")
        sources = [str(s) for s in raw] if isinstance(raw, (list, tuple)) else []
        return sources != [policy]

    def _calibration_source_phrase(self) -> str:
        """Name the labels the shared calibration was fit on.

        "production's calibration" for one labeled policy; otherwise "the
        calibration fit on ..." naming the policies and/or the external
        calibration data.
        """
        raw = self.metadata.get("calibration_label_sources")
        sources = [str(s) for s in raw] if isinstance(raw, (list, tuple)) else []
        if not sources:
            own = self.metadata.get("own_oracle_labels_by_policy")
            if isinstance(own, dict):
                sources = sorted(str(p) for p, n in own.items() if n)
        external = _CALIBRATION_DATA_SOURCE in sources
        policies = [s for s in sources if s != _CALIBRATION_DATA_SOURCE]
        if not external and len(policies) == 1:
            return f"{policies[0]}'s calibration"
        parts: List[str] = []
        if external:
            parts.append("calibration_data_path")
        if policies:
            noun = "'s labels" if len(policies) == 1 else " labels"
            parts.append(_join_names(policies) + noun)
        if not parts:
            return "the shared calibration"
        return "the calibration fit on " + " and ".join(parts)

    def _calibration_marker(self, policy: str) -> str:
        """'borrowed calibration' (no own labels) or 'uncorrected calibration'."""
        if self._own_label_count(policy) == 0:
            return "borrowed calibration"
        return "uncorrected calibration"

    def _transport_actions(self, policies: List[str]) -> str:
        """What would verify these policies' calibration, by audit status."""
        by_status: Dict[str, List[str]] = {}
        for policy in policies:
            by_status.setdefault(self._transport_status(policy), []).append(policy)
        sentences: List[str] = []
        for status in sorted(by_status):
            group = by_status[status]
            names = _join_names(group)
            whose = f"of {names}" if len(group) == 1 else f"of each of {names}"
            uncorrected = [p for p in group if self._own_label_count(p) > 0]
            label = (
                f"label >=20 random responses {whose}"
                if not uncorrected
                else (
                    f"attach >=20 random labels {whose} and keep the default "
                    "augmented route (use_augmented_estimator on, label_design "
                    "'representative' or 'known_propensity')"
                )
            )
            if status == "FAIL":
                sentences.append(
                    f"Transport audit FAILED for {names}: {label}, and re-run "
                    "without the failed probes in transport= (keep that audit as "
                    "a record); do not re-audit to escape a FAIL."
                )
            elif status == "NOT_GRADED":
                sentences.append(
                    f"{label[0].upper()}{label[1:]}, or grade the transport audit "
                    "with a delta_max margin (now NOT_GRADED)."
                )
            elif status == "NOT_CHECKED":
                sentences.append(
                    f"{label[0].upper()}{label[1:]}, or audit transport on a "
                    "held-out random sample of them (size it with "
                    "plan_transport_audits)."
                )
            else:
                sentences.append(
                    f"{label[0].upper()}{label[1:]}, or extend the held-out "
                    f"transport audit (now {status}; plan_transport_audits)."
                )
        return " ".join(sentences)

    def _borrowed_calibration_line(self, policy: str) -> str:
        """The named summary line for one transport-unverified policy."""
        if self._own_label_count(policy) == 0:
            opening = f"{policy}: no labels of its own"
        else:
            route = self._point_route(policy) or "unknown"
            opening = (
                f"{policy}: its own labels do not correct its estimate (route {route})"
            )
        return (
            f"{opening}; its estimate and every difference involving it assume "
            f"{self._calibration_source_phrase()} transfers, which the CI and "
            "p-value do not cover (the true difference can have either sign). "
            f"{self._transport_actions([policy])}"
        )

    def _transport_limitation(self, policy: str) -> Optional[str]:
        """The transport caveat printed next to ``policy``, or None."""
        status = self._transport_status(policy)
        if not self._transport_provenance_known():
            # Pre-0.9.2 results: which policies have own labels is unknown,
            # so keep the conservative 0.9.1 display.
            return None if status == "PASS" else f"residual transport {status}"
        if policy in self._transport_unverified():
            return f"{self._calibration_marker(policy)} (residual transport {status})"
        if status not in ("PASS", "NOT_CHECKED"):
            return f"residual transport {status}"
        return None

    def _paired_interval(
        self, i: int, j: int, alpha: float = 0.05
    ) -> Tuple[Dict[str, Any], float, float, float, float]:
        """compare_policies(i, j) plus (difference, lower, upper, p) as floats.

        z-test paths carry no stored CI; the interval matching their p-value
        is returned. Raises what ``_compare_policies_impl`` raises.
        """
        from scipy import stats

        comparison = self._compare_policies_impl(i, j, alpha, warn=False)
        difference = float(comparison["difference"])
        lower, upper = comparison.get("ci_lower"), comparison.get("ci_upper")
        if lower is None or upper is None:
            z = float(stats.norm.ppf(1 - alpha / 2))
            se = float(comparison["se_difference"])
            lower, upper = difference - z * se, difference + z * se
        return (
            comparison,
            difference,
            float(lower),
            float(upper),
            float(comparison["p_value"]),
        )

    def _decision_readiness(self, verdict: PolicyVerdict) -> Tuple[bool, str]:
        """``decision_ready`` / ``decision_note`` for a best_policy verdict.

        Ready only when the verdict beats every other usable policy in the
        paired comparison (each 95% CI wholly above 0), no policy in that set
        relies on an unverified calibration transfer or failed the gates, the
        verdict was not reached by demoting a flagged leader, and each
        difference clears the margins of any transport audit it relies on.
        Requiring every comparison to succeed is an intersection-union test,
        so no multiplicity adjustment is needed for the "beats every other
        policy" claim.
        """
        if verdict.all_flagged:
            return False, "Not decision-ready: no policy passed the reliability gates."
        if verdict.flagged:
            return False, (
                f"Not decision-ready: {verdict.name} failed the reliability gates."
            )
        if verdict.runner_up is not None:
            reasons = "; ".join(verdict.runner_up_reasons or []) or "gates"
            return False, (
                f"Not decision-ready: the point-estimate leader "
                f"{verdict.runner_up} failed the reliability gates ({reasons}); "
                f"{verdict.name} is the best gate-passing policy, not a tested "
                f"winner. Resolve {verdict.runner_up}'s gate before ranking them."
            )
        if not self._transport_provenance_known():
            return False, (
                "Not decision-ready: this result does not record which policies "
                "have oracle labels of their own (saved before cje-eval 0.9.2, or "
                "built without analyze_dataset), so borrowed calibration cannot "
                "be ruled out. Re-run analyze_dataset."
            )
        policies = self.target_policies
        estimates = np.asarray(self.estimates, dtype=float)
        others = [
            policy
            for i, policy in enumerate(policies)
            if i < len(estimates)
            and not np.isnan(estimates[i])
            and policy != verdict.name
        ]
        if not others:
            return True, "Only one usable policy; there is nothing to compare."

        unverified = set(self._transport_unverified())
        blockers = [p for p in [verdict.name, *others] if p in unverified]
        if blockers:
            names = _join_names(blockers)
            return False, (
                f"Not decision-ready: the ranking involves {names}, whose "
                f"calibration transfer is unverified (it assumes "
                f"{self._calibration_source_phrase()} transfers). Name no winner "
                f"and no lean until that changes. "
                f"{self._transport_actions(blockers)}"
            )
        flagged_others = [
            p
            for p in others
            if (gate := self.gates.get(p)) is not None
            and (gate.flagged or gate.refuse_level_claims)
        ]
        if flagged_others:
            names = _join_names(flagged_others)
            return False, (
                f"Not decision-ready: {names} failed the reliability gates, so "
                f"{verdict.name} cannot be ranked against "
                f"{'it' if len(flagged_others) == 1 else 'them'}."
            )

        k = policies.index(verdict.name)
        closest: Optional[Tuple[float, str, float, float, float]] = None
        for rival in others:
            try:
                _, difference, lower, upper, p_value = self._paired_interval(
                    k, policies.index(rival)
                )
            except (InferenceUnavailableError, ValueError) as exc:
                return False, (
                    f"Not decision-ready: the paired comparison of {verdict.name} "
                    f"and {rival} is unavailable ({exc})."
                )
            # A PASS audit bounds a policy's residual by its delta_max, which
            # the paired CI does not include: the difference must clear it.
            margin = 0.0
            for policy in (verdict.name, rival):
                if self._relies_on_transport(policy):
                    audit = (self.metadata.get("transport_audits") or {}).get(policy)
                    delta = audit.get("delta_max") if isinstance(audit, dict) else None
                    margin += float(delta) if delta is not None else 0.0
            if not np.isfinite(lower) or lower <= margin:
                if margin > 0 and np.isfinite(lower) and lower > 0:
                    return False, (
                        f"Not decision-ready: the paired 95% CI for {verdict.name} - "
                        f"{rival} is [{lower:+.3f}, {upper:+.3f}], inside the "
                        f"transport margin ({margin:.3f}) its PASS audits allow, "
                        "so calibration error could reverse it."
                    )
                return False, (
                    f"Not decision-ready: the paired 95% CI for {verdict.name} - "
                    f"{rival} is [{lower:+.3f}, {upper:+.3f}] "
                    f"({_format_p_value(p_value)}), so {verdict.name} is not shown "
                    f"to beat {rival} at this sample size (which is not evidence "
                    "the two are equal). Size more labels with plan_for_mde, or "
                    "report no reliable winner."
                )
            if closest is None or lower < closest[0]:
                closest = (lower, rival, upper, difference, p_value)
        assert closest is not None
        lower, rival, upper, _, p_value = closest
        scope = (
            f"{rival}"
            if len(others) == 1
            else f"every other policy (closest: {rival}, 95% CI "
            f"[{lower:+.3f}, {upper:+.3f}])"
        )
        p_text = f" ({_format_p_value(p_value)})" if len(others) == 1 else ""
        return True, (
            f"{verdict.name} beats {scope} in the paired comparison{p_text}; "
            "check gates and audits before acting."
        )

    def _paired_summary_lines(
        self, alpha: float = 0.05, focus: Optional[str] = None
    ) -> List[str]:
        """The summary()'s paired-difference block (unadjusted p-values).

        With more than 10 pairs, only the pairs involving ``focus`` (the best
        policy by point estimate) are printed, with a pointer to
        ``compare_all_policies()`` for the rest; the closing line then speaks
        only of the printed pairs.
        """
        policies = self.target_policies
        n = len(self.estimates)
        if n < 2 or len(policies) != n:
            return []
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        hidden = 0
        if len(pairs) > 10 and focus in policies:
            k = policies.index(focus)
            shown = [(i, j) for i, j in pairs if k in (i, j)]
            hidden = len(pairs) - len(shown)
            pairs = shown
        lines = [
            (
                "Paired differences (p unadjusted):"
                if not hidden
                else f"Paired differences with {focus} (p unadjusted):"
            )
        ]
        every_ci_includes_zero = True
        # Policies relying on an unverified transfer in pairs whose CI
        # excludes 0, and whether any such pair rests on verified estimates.
        separated_unverified: List[str] = []
        separated_unconditionally = False
        first_error: Optional[str] = None
        for i, j in pairs:
            label = f"{policies[i]} - {policies[j]}"
            try:
                comparison, difference, lower, upper, p_value = self._paired_interval(
                    i, j, alpha
                )
            except (InferenceUnavailableError, ValueError) as exc:
                every_ci_includes_zero = False
                first_error = first_error or str(exc)
                lines.append(f"  {label}: unavailable")
                continue
            unverified = [str(p) for p in comparison.get("transport_unverified") or []]
            if not all(np.isfinite([difference, lower, upper])):
                every_ci_includes_zero = False
            elif lower > 0 or upper < 0:
                every_ci_includes_zero = False
                if unverified:
                    separated_unverified.extend(
                        p for p in unverified if p not in separated_unverified
                    )
                else:
                    separated_unconditionally = True
            # A difference inherits a flagged input's unreliability,
            # however small its p-value: mark the pair, as the policy
            # lines above mark the policy.
            flagged = [str(p) for p in comparison.get("gate_flagged") or []]
            flag = f"  [gate-flagged: {', '.join(flagged)}]" if flagged else ""
            if unverified:
                kinds = {self._calibration_marker(p) for p in unverified}
                kind = kinds.pop() if len(kinds) == 1 else "unverified calibration"
                flag += f"  [{kind}: {', '.join(unverified)}]"
            lines.append(
                f"  {label}: {difference:+.3f}  "
                f"{100 * (1 - alpha):.0f}% CI [{lower:+.3f}, {upper:+.3f}]  "
                f"{_format_p_value(p_value)}{flag}"
            )
        if first_error:
            lines.append(f"  ({first_error})")
        if hidden:
            lines.append(
                f"  ({hidden} more pairs not shown: compare_all_policies() lists all)"
            )
        if n >= 3:
            lines.append(
                "  With 3+ policies, read Benjamini-Hochberg p_adjusted from "
                'compare_all_policies(adjust="bh").'
            )
        scope = "every paired CI" if not hidden else f"every paired CI with {focus}"
        if every_ci_includes_zero:
            lines.append(
                f"No reliable winner: {scope} includes 0 "
                "(not evidence that they are equal)"
            )
        elif separated_unverified and not separated_unconditionally:
            names = _join_names(separated_unverified)
            single = len(separated_unverified) == 1
            lines.append(
                f"No decision-ready winner: {scope} that excludes 0 involves "
                f"{names}, whose calibration transfer is unverified (see "
                f"{'its line' if single else 'their lines'} above). Name no winner "
                "and no lean until that changes."
            )
        return lines

    def summary(self) -> str:
        """Compact text summary: estimates, CIs, gates, paired differences.

        Lines, in order: one per policy (estimate and 95% CI, marked
        ``[borrowed calibration]`` when it has no oracle labels of its own,
        or ``[uncorrected calibration]`` when a plug-in route leaves its own
        labels unused for correction, in both cases without a PASS transport
        audit); one line per such policy saying its estimate and differences
        assume the calibration transfers, and what would verify it; the
        best policy by point estimate (a ranking, not a test) with its
        limitations; with two or more policies, the paired differences
        ``a - b`` with 95% CI and unadjusted p-value (every pair, or with
        more than 10 pairs only those with the best point estimate), marked
        ``[gate-flagged: ...]`` or ``[borrowed calibration: ...]``, plus "No
        reliable winner" when every printed paired CI includes 0, or "No
        decision-ready winner" when every printed pair whose CI excludes 0
        involves an unverified calibration transfer; and the diagnostics
        status (at least ``warning`` for such a policy).

        Example:
            >>> print(results.summary())
        """
        policies = self.target_policies
        if not policies:
            return (
                f"CJE Estimation Results (method: {self.method}): "
                f"{len(self.estimates)} estimate(s); no target_policies metadata"
            )

        ci_lower, ci_upper = self.confidence_interval()
        gates = self.gates
        width = max(len(p) for p in policies)
        unverified = [p for p in self._transport_unverified() if p in policies]

        lines = [f"CJE Estimation Results (method: {self.method})"]
        for i, policy in enumerate(policies):
            gate = gates.get(policy)
            flag = ""
            if gate is not None and gate.flagged:
                flag = "  [gate: FLAGGED]"
            if policy in unverified:
                flag += f"  [{self._calibration_marker(policy)}]"
            lines.append(
                f"  {policy:<{width}s}  {self.estimates[i]:.3f}  "
                f"95% CI [{ci_lower[i]:.3f}, {ci_upper[i]:.3f}]{flag}"
            )
        # Say what an unverified calibration transfer means before any ranking.
        for policy in unverified:
            lines.append(self._borrowed_calibration_line(policy))

        try:
            verdict = self.best_policy()
        except ValueError:
            lines.append("Best policy: none (no usable estimates)")
        else:
            # Show the raw point-estimate winner (the demoted runner_up when
            # the gates flagged it) with its limitations, then state the
            # returned reliable winner when the two diverge.
            display = verdict.runner_up if verdict.runner_up else verdict.name
            limitations = []
            if verdict.flagged or verdict.runner_up is not None:
                limitations.append(
                    "no policy passed the reliability gates"
                    if verdict.all_flagged
                    else "flagged by the reliability gates"
                )
            if self.metadata.get("calibration_status") == "UNCALIBRATED":
                limitations.append("UNCALIBRATED raw judge-score mean")
            # NOT_CHECKED matters only for an estimate that relies on
            # transport; an observed audit result (FAIL / INCONCLUSIVE) is
            # always shown.
            transport = self._transport_limitation(display)
            if transport:
                limitations.append(transport)
            lines.append(
                f"Best by point estimate: {display} (point estimate, not a test)"
            )
            if limitations:
                lines.append("Limitations: " + "; ".join(limitations))
            if verdict.runner_up is not None:
                reasons = "; ".join(verdict.runner_up_reasons or [])
                lines.append(
                    f"Best reliable policy: {verdict.name} — raw argmax "
                    f"{verdict.runner_up} was flagged ({reasons}); pass "
                    f"reliable_only=False for the raw argmax"
                )

        try:
            focus = self.best_policy(reliable_only=False).name
        except ValueError:
            focus = None
        lines.extend(self._paired_summary_lines(focus=focus))

        if self.diagnostics is not None:
            status = self.diagnostics.overall_status
            if unverified:
                from ..diagnostics import Status
                from ..diagnostics.gates import worst_status

                status = worst_status(status, Status.WARNING)
            lines.append(f"Status: {status.value}")
        return "\n".join(lines)

    # Minimum NaN-filtered paired deltas required for the paired-bootstrap
    # comparison path; below this the percentile CI and sign-test p-value
    # are too coarse and compare_policies falls through to the next basis.
    _MIN_PAIRED_BOOTSTRAP_DELTAS = 100

    def compare_policies(
        self,
        policy1_idx: Union[int, str],
        policy2_idx: Union[int, str],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Compare two policies with the most honest difference SE available.

        Each policy is given by name (``"candidate"``) or by integer index
        into ``target_policies`` (sorted by name, not your dict's order);
        ``compare_policies("a", "b")`` equals ``compare_policies(i, j)``.
        ``difference`` is policy1 minus policy2.

        Dispatches over four inference bases, best-first; the winning basis
        is reported in the returned ``method`` key:

        1. ``paired_bootstrap`` — when ``bootstrap_samples`` is present with
           both policies' columns and at least 100 NaN-filtered paired
           replicate deltas. Each bootstrap replicate is one joint cluster
           resample plus one calibrator refit, so the per-replicate deltas
           M[:, i] - M[:, j] carry the full paired uncertainty of the
           difference — including the calibrator/residual-correction noise
           that per-policy plug-in influence functions miss (the source of
           anti-conservative CIs on near-tie pairs). ``difference`` stays
           the point-estimate difference; ``se_difference`` is the delta
           standard deviation; ``ci_lower``/``ci_upper`` are percentile
           bounds; ``p_value`` is an add-one smoothed two-sided sign test,
           min(1, 2*min(1+#{d<=0}, 1+#{d>=0})/(B+1)), floored at 2/(B+1) —
           it cannot reach 0 at finite B. Because the p-value (sign test)
           and the CI (percentile) are different functionals of the same
           deltas, they may disagree by O(1/B) exactly at the significance
           boundary.
        2. ``paired_if_oua`` — when the analytic (cluster-robust) path
           stored ``metadata["pairwise_inference"]`` for this pair:
           a t-test using the stored difference SE (pairing-aware sampling
           SE + oracle-jackknife variance of the difference) and degrees of
           freedom; the pairing ``basis`` is included in the result.
        3. ``paired_if_legacy`` — the pre-0.5.1 influence-function z-test
           (per-sample IF differences), kept byte-identical for
           deserialized older results that carry influence functions but
           no bootstrap matrix or pairwise-inference metadata. Its SE
           contains no calibrator noise — anti-conservative on near-tie
           pairs.
        4. ``independent_conservative`` — sqrt(se1^2 + se2^2), ignoring
           covariance (conservative for positively correlated policies).

        Args:
            policy1_idx: The first policy: its name, or its index in
                ``metadata["target_policies"]``.
            policy2_idx: The second policy (name or index).
            alpha: Significance level for ``significant`` and (paths 1-2)
                the returned CI.

        Returns:
            Dict with ``policy1``/``policy2`` (names), ``difference``,
            ``se_difference``, ``z_score``, ``p_value``, ``significant``,
            ``used_influence``, ``method``, ``gate_flagged`` (names among the
            pair whose reliability gates are flagged — a comparison
            involving a flagged policy inherits that unreliability, however
            honest the SE), ``transport_unverified`` (names among the pair
            whose estimate relies on an unverified calibration transfer)
            and ``conditional_on_transport`` (True when that list is
            non-empty: the difference assumes a calibration transfers,
            which neither the CI nor the p-value covers; None when the
            result does not record label provenance, as for results saved
            before 0.9.2; ``significant`` keeps its meaning); paths 1-2 add
            ``ci_lower``/``ci_upper``/``alpha`` (plus ``n_replicates`` for
            the bootstrap path and ``df``/``basis`` for the analytic path).

        Raises:
            ValueError: An unknown policy name.
            TypeError: A policy given as neither a name nor an integer.
            IndexError: An integer index out of range.
        """
        return self._compare_policies_impl(policy1_idx, policy2_idx, alpha, warn=True)

    def _resolve_policy_index(self, name: str, value: Any) -> int:
        """Accept a policy name or an integer index; return the index."""
        n_policies = len(self.estimates)
        if isinstance(value, str):
            policies = self.target_policies
            if value in policies:
                return policies.index(value)
            folded = value.strip().casefold()
            near = [p for p in policies if p.strip().casefold() == folded]
            hint = f" Did you mean {near[0]!r}?" if near else ""
            raise ValueError(
                f"{name}={value!r} is not a policy in this result.{hint} "
                f"Policies: {policies}"
            )
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(
                f"{name} must be an integer index or a policy name, got {value!r}"
            )
        if int(value) < 0 or int(value) >= n_policies:
            raise IndexError(
                f"{name}={value} is out of range for {n_policies} policies"
            )
        return int(value)

    def _compare_policies_impl(
        self,
        policy1_idx: Union[int, str],
        policy2_idx: Union[int, str],
        alpha: float,
        *,
        warn: bool,
    ) -> Dict[str, Any]:
        """``compare_policies`` body; ``warn=False`` keeps summary() quiet."""
        policy1_idx = self._resolve_policy_index("policy1_idx", policy1_idx)
        policy2_idx = self._resolve_policy_index("policy2_idx", policy2_idx)
        alpha = _validate_alpha(alpha)

        unavailable_raw = self.metadata.get("inference_unavailable_policies", [])
        unavailable = (
            {str(policy) for policy in unavailable_raw}
            if isinstance(unavailable_raw, (list, tuple, set))
            else set()
        )
        policies = self.target_policies
        requested = {
            policies[index]
            for index in (policy1_idx, policy2_idx)
            if index < len(policies)
        }
        blocked = sorted(requested & unavailable)
        if blocked:
            raise InferenceUnavailableError(
                "Pairwise inference is unavailable for "
                + "; ".join(self._inference_unavailable_detail(p) for p in blocked)
                + "."
            )

        serialized_state = self.metadata.get("_serialized_pairwise_state")
        if isinstance(serialized_state, dict):
            capability = serialized_state.get("capability")
            if capability == "unavailable":
                raise InferenceUnavailableError(
                    "Paired inference was omitted from this serialized result. "
                    "Re-run analysis or export with detail='portable'/'full'."
                )
            if capability == "stored_alpha_only":
                lo, hi = (
                    (policy1_idx, policy2_idx)
                    if policy1_idx <= policy2_idx
                    else (policy2_idx, policy1_idx)
                )
                pair_key = f"{lo}-{hi}"
                recomputable_pairs = serialized_state.get("recomputable_pairs", [])
                if pair_key in recomputable_pairs:
                    recomputed_comparison = self._compare_legacy(
                        policy1_idx, policy2_idx, alpha
                    )
                    return self._annotate_gate_flags(
                        recomputed_comparison, policy1_idx, policy2_idx, warn=warn
                    )

                stored_alpha = float(serialized_state.get("alpha", 0.05))
                if abs(alpha - stored_alpha) > 1e-12:
                    raise InferenceUnavailableError(
                        f"This result stores paired comparisons only at "
                        f"alpha={stored_alpha}; requested alpha={alpha}."
                    )
                raw = serialized_state.get("comparisons", {}).get(pair_key)
                if not isinstance(raw, dict):
                    raise InferenceUnavailableError(
                        f"No stored paired comparison exists for {lo}-{hi}."
                    )
                serialized_result = dict(raw)
                if policy1_idx > policy2_idx:
                    serialized_result["difference"] = -float(
                        serialized_result["difference"]
                    )
                    serialized_result["z_score"] = -float(serialized_result["z_score"])
                    if (
                        "ci_lower" in serialized_result
                        and "ci_upper" in serialized_result
                    ):
                        lower = -float(serialized_result["ci_upper"])
                        upper = -float(serialized_result["ci_lower"])
                        serialized_result["ci_lower"] = lower
                        serialized_result["ci_upper"] = upper
                return self._annotate_gate_flags(
                    serialized_result, policy1_idx, policy2_idx, warn=warn
                )

        comparison = self._compare_paired_bootstrap(
            policy1_idx, policy2_idx, alpha, warn=warn
        )
        if comparison is None:
            comparison = self._compare_pairwise_inference(
                policy1_idx, policy2_idx, alpha
            )
        if comparison is None:
            comparison = self._compare_legacy(policy1_idx, policy2_idx, alpha)
        return self._annotate_gate_flags(
            comparison, policy1_idx, policy2_idx, warn=warn
        )

    def _inference_unavailable_detail(self, policy: str) -> str:
        """Why a policy has no interval (``inference_unavailable_reasons``)."""
        reasons = self.metadata.get("inference_unavailable_reasons")
        reason = reasons.get(policy) if isinstance(reasons, dict) else None
        if reason != "too_few_labelled_clusters":
            return f"{policy!r}: fewer than two independent evaluation clusters"
        df_info = self.metadata.get("degrees_of_freedom")
        info = df_info.get(policy) if isinstance(df_info, dict) else None
        info = info if isinstance(info, dict) else {}
        n_labelled = info.get("n_labelled_clusters")
        fitted = info.get("fitted_parameters")
        if isinstance(n_labelled, int) and isinstance(fitted, int):
            noun = "prompt" if n_labelled == 1 else "prompts"
            return (
                f"{policy!r}: its residual correction has {n_labelled} labelled "
                f"{noun}; at least {fitted + 1} are needed for an interval"
            )
        return (
            f"{policy!r}: its residual correction has too few labelled prompts "
            "for an interval"
        )

    def _annotate_gate_flags(
        self, result: Dict[str, Any], idx1: int, idx2: int, *, warn: bool = True
    ) -> Dict[str, Any]:
        """Name the pair and surface reliability-gate and transport caveats.

        A difference CI cannot repair a biased input: when a policy's
        estimate is unreliable (flagged gate / refused level claims, e.g.
        after a failed transport audit), every comparison involving it
        inherits that unreliability regardless of how honest the
        difference SE is. Adds ``policy1``/``policy2`` (names),
        ``gate_flagged`` (list of flagged policy names, possibly empty; warns
        when non-empty), ``transport_unverified`` (pair members relying on
        an unverified calibration transfer) and ``conditional_on_transport``
        (whether that list is non-empty; None when provenance is unknown).
        """
        flagged: List[str] = []
        policies = self.target_policies
        gates = self.gates
        for idx in (idx1, idx2):
            if 0 <= idx < len(policies):
                gate = gates.get(policies[idx])
                if gate is not None and (gate.flagged or gate.refuse_level_claims):
                    flagged.append(policies[idx])
        names = [
            policies[idx] if 0 <= idx < len(policies) else str(idx)
            for idx in (idx1, idx2)
        ]
        result["policy1"], result["policy2"] = names
        unverified = set(self._transport_unverified())
        pair_unverified: List[str] = []
        for policy in names:
            if policy in unverified and policy not in pair_unverified:
                pair_unverified.append(policy)
        result["transport_unverified"] = pair_unverified
        result["conditional_on_transport"] = (
            bool(pair_unverified) if self._transport_provenance_known() else None
        )
        result["gate_flagged"] = flagged
        if flagged and warn:
            logger.warning(
                "compare_policies: %s flagged by reliability gates — the "
                "difference estimate inherits that unreliability (a paired "
                "SE cannot correct a biased point estimate). Treat this "
                "comparison per the gates discipline.",
                ", ".join(repr(p) for p in flagged),
            )
        return result

    def _compare_paired_bootstrap(
        self, idx1: int, idx2: int, alpha: float, *, warn: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Paired difference inference from the bootstrap replicate matrix.

        Returns None (caller falls through to the next basis) when the
        matrix is absent, either column is missing, or fewer than
        _MIN_PAIRED_BOOTSTRAP_DELTAS NaN-filtered paired deltas remain.
        """
        if self.bootstrap_samples is None:
            return None
        matrix = np.asarray(self.bootstrap_samples, dtype=float)
        if matrix.ndim != 2:
            return None
        n_cols = int(matrix.shape[1])
        if not (0 <= idx1 < n_cols and 0 <= idx2 < n_cols):
            return None

        deltas = matrix[:, idx1] - matrix[:, idx2]
        valid = deltas[~np.isnan(deltas)]
        n_valid = int(valid.size)
        if n_valid < self._MIN_PAIRED_BOOTSTRAP_DELTAS:
            if not warn:
                return None
            logger.warning(
                f"compare_policies: only {n_valid} valid paired bootstrap "
                f"deltas (< {self._MIN_PAIRED_BOOTSTRAP_DELTAS}); falling "
                f"through to the next inference basis."
            )
            return None

        difference = float(self.estimates[idx1] - self.estimates[idx2])
        se_difference = float(np.nanstd(deltas, ddof=1))
        ci_lower = float(np.nanpercentile(deltas, 100 * alpha / 2))
        ci_upper = float(np.nanpercentile(deltas, 100 * (1 - alpha / 2)))

        # Add-one smoothed two-sided sign test over the replicate deltas.
        # Floor is 2/(B+1): a finite bootstrap can never certify p=0.
        n_nonpos = int(np.sum(valid <= 0.0))
        n_nonneg = int(np.sum(valid >= 0.0))
        p_value = min(1.0, 2.0 * min(1 + n_nonpos, 1 + n_nonneg) / (n_valid + 1))

        z_score = difference / se_difference if se_difference > 0 else 0.0

        return {
            "difference": difference,
            "se_difference": se_difference,
            "z_score": float(z_score),
            "p_value": float(p_value),
            "significant": bool(p_value < alpha),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "alpha": float(alpha),
            "n_replicates": n_valid,
            "used_influence": False,
            "method": "paired_bootstrap",
        }

    def _compare_pairwise_inference(
        self, idx1: int, idx2: int, alpha: float
    ) -> Optional[Dict[str, Any]]:
        """t-based comparison from metadata["pairwise_inference"] (analytic path).

        The estimator stores one entry per unordered pair, keyed "i-j" with
        i < j in estimate order, carrying the pairing-aware sampling SE
        combined with the oracle-jackknife variance of the difference.
        Returns None when the pair is absent or its stored SE is unusable.
        """
        if not isinstance(self.metadata, dict):
            return None
        pairwise = self.metadata.get("pairwise_inference")
        if not isinstance(pairwise, dict):
            return None
        lo, hi = (idx1, idx2) if idx1 <= idx2 else (idx2, idx1)
        entry = pairwise.get(f"{lo}-{hi}")
        if not isinstance(entry, dict):
            return None

        se_raw = entry.get("se")
        df_raw = entry.get("df")
        if not isinstance(se_raw, (int, float)) or not isinstance(df_raw, (int, float)):
            return None
        se_difference = float(se_raw)
        df_unfloored = float(df_raw)
        if (
            not np.isfinite(se_difference)
            or se_difference <= 0
            or not np.isfinite(df_unfloored)
        ):
            return None
        df = max(df_unfloored, 1.0)

        from scipy import stats

        difference = float(self.estimates[idx1] - self.estimates[idx2])
        t_stat = difference / se_difference
        p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df)))
        t_crit = float(stats.t.ppf(1 - alpha / 2, df))
        basis = str(entry.get("basis", "unknown"))

        return {
            "difference": difference,
            "se_difference": se_difference,
            "z_score": float(t_stat),
            "p_value": p_value,
            "significant": bool(p_value < alpha),
            "ci_lower": difference - t_crit * se_difference,
            "ci_upper": difference + t_crit * se_difference,
            "alpha": float(alpha),
            "df": df,
            "basis": basis,
            "used_influence": basis
            in {
                "index_paired",
                "prompt_paired",
                "prompt_cluster_paired",
            },
            "method": "paired_if_oua",
        }

    def _compare_legacy(self, idx1: int, idx2: int, alpha: float) -> Dict[str, Any]:
        """Pre-0.5.1 comparison: IF z-test when possible, else independent SEs.

        Numerics are byte-identical to the 0.5.0 compare_policies. The
        ``method`` key distinguishes the paired IF z-test
        ("paired_if_legacy") from the independent-SE fallback
        ("independent_conservative").
        """
        diff = self.estimates[idx1] - self.estimates[idx2]
        used_paired_if = False

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
                        used_paired_if = True
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
            "significant": bool(p_value < alpha),
            "used_influence": bool(
                self.influence_functions is not None
                and se_diff
                != np.sqrt(
                    self.standard_errors[idx1] ** 2 + self.standard_errors[idx2] ** 2
                )
            ),
            "method": (
                "paired_if_legacy" if used_paired_if else "independent_conservative"
            ),
        }

    def compare_all_policies(
        self, alpha: float = 0.05, adjust: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Compare every (i < j) policy pair via ``compare_policies``.

        Args:
            alpha: Significance level per comparison (and for
                ``significant_adjusted`` when ``adjust`` is set).
            adjust: None (default) for raw per-pair p-values, or "bh" to
                apply the Benjamini-Hochberg step-up procedure across all
                pairs, adding ``p_adjusted`` and ``significant_adjusted``
                to each comparison (FDR control at ``alpha`` for the
                many-pair audit setting).

        Returns:
            List of ``compare_policies`` dicts (one per pair, in (i, j)
            index order, each naming its ``policy1``/``policy2``).
        """
        alpha = _validate_alpha(alpha)
        if adjust not in (None, "bh"):
            raise ValueError(f"adjust must be None or 'bh', got {adjust!r}")

        n = len(self.estimates)

        comparisons: List[Dict[str, Any]] = []
        for i in range(n):
            for j in range(i + 1, n):
                comparisons.append(self.compare_policies(i, j, alpha=alpha))

        if adjust == "bh":
            # Benjamini-Hochberg step-up: p_adj_(k) = min_{l >= k} p_(l)*m/l,
            # clipped at 1, over the finite p-values (NaN p-values — e.g.
            # pairs with a NaN estimate — keep p_adjusted=NaN, not significant).
            p_values = [float(c["p_value"]) for c in comparisons]
            finite = [k for k, p in enumerate(p_values) if np.isfinite(p)]
            m = len(finite)
            order = sorted(finite, key=lambda k: p_values[k])
            running = 1.0
            adjusted: Dict[int, float] = {}
            for pos in range(m - 1, -1, -1):
                k = order[pos]
                running = min(running, p_values[k] * m / (pos + 1))
                adjusted[k] = min(1.0, running)
            for k, comparison in enumerate(comparisons):
                p_adj = adjusted.get(k, float("nan"))
                comparison["p_adjusted"] = float(p_adj)
                comparison["significant_adjusted"] = bool(
                    np.isfinite(p_adj) and p_adj < alpha
                )

        return comparisons

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        if not self.metadata.get("target_policies"):
            return "<pre>EstimationResult (no policies available)</pre>"

        policies = self.metadata["target_policies"]
        ci_lower, ci_upper = self.confidence_interval()

        # Build HTML table rows
        rows = []
        rows.append(
            "<tr><th>Policy</th><th>Estimate</th><th>Std Error</th><th>95% CI</th></tr>"
        )

        for i, policy in enumerate(policies):
            est = self.estimates[i]
            se = self.standard_errors[i]
            ci_str = f"[{ci_lower[i]:.3f}, {ci_upper[i]:.3f}]"
            rows.append(
                f"<tr><td>{escape(str(policy))}</td><td>{est:.3f}</td>"
                f"<td>{se:.3f}</td><td>{ci_str}</td></tr>"
            )

        # Build full HTML
        html_parts = [
            '<div style="font-family: monospace;">',
            "<h4>CJE Estimation Results</h4>",
            f"<p><b>Method:</b> {escape(str(self.method))} | "
            f"<b>Policies:</b> {len(policies)}</p>",
            '<table style="border-collapse: collapse; margin-top: 10px; border: 1px solid #ddd;">',
            '<thead style="background-color: #f0f0f0;">',
        ]
        html_parts.extend(rows[:1])  # Header row
        html_parts.append("</thead>")
        html_parts.append("<tbody>")
        html_parts.extend(rows[1:])  # Data rows
        html_parts.append("</tbody>")
        html_parts.append("</table>")

        # Add diagnostic summary if available
        if self.diagnostics:
            html_parts.append(
                f'<p style="margin-top: 10px;"><b>Status:</b> {self.diagnostics.overall_status.value}</p>'
            )

        html_parts.append("</div>")
        return "".join(html_parts)

    def plot_estimates(
        self,
        base_policy_stats: Optional[Dict[str, float]] = None,
        oracle_values: Optional[Dict[str, float]] = None,
        policy_labels: Optional[Dict[str, str]] = None,
        save_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Plot policy estimates with confidence intervals.

        Convenience wrapper around plot_policy_estimates() that extracts
        data from this result object.

        Args:
            base_policy_stats: Optional dict with "mean" and "se" for base policy.
                Example: {"mean": 0.72, "se": 0.01}
            oracle_values: Optional dict of oracle values for comparison.
                Example: {"policy_a": 0.75, "policy_b": 0.68}
            policy_labels: Optional dict mapping policy names to display labels.
                Example: {"prompt_v1": "Conversational tone"}
            save_path: Optional path to save plot (e.g., "results/estimates.png")
            **kwargs: Additional arguments passed to plot_policy_estimates()
                (e.g., figsize=(10, 6), title="My Results")

        Returns:
            Matplotlib figure object

        Example:
            >>> result = analyze_dataset(fresh_draws_dir="data.jsonl")
            >>> result.plot_estimates(
            ...     policy_labels={"prompt_v1": "Conversational tone"},
            ...     save_path="estimates.png"
            ... )
        """
        from ..visualization import plot_policy_estimates

        # Extract policies
        policies = self.metadata.get("target_policies", [])
        if not policies:
            raise ValueError("No target_policies found in metadata")

        # Build estimates and standard_errors dicts
        estimates = {}
        standard_errors = {}
        confidence_intervals: Dict[str, Tuple[float, float]] = {}

        # Add base policy if provided
        if base_policy_stats:
            if "mean" not in base_policy_stats:
                raise ValueError("base_policy_stats must contain 'mean' key")
            estimates["base"] = base_policy_stats["mean"]
            standard_errors["base"] = base_policy_stats.get("se", 0.0)

        # Add target policies
        ci_lower, ci_upper = self.confidence_interval(alpha=0.05)
        for i, policy in enumerate(policies):
            estimates[policy] = float(self.estimates[i])
            standard_errors[policy] = float(self.standard_errors[i])
            confidence_intervals[policy] = (
                float(ci_lower[i]),
                float(ci_upper[i]),
            )

        # Call visualization function
        from pathlib import Path

        return plot_policy_estimates(
            estimates=estimates,
            standard_errors=standard_errors,
            confidence_intervals=confidence_intervals,
            oracle_values=oracle_values,
            policy_labels=policy_labels,
            save_path=Path(save_path) if save_path else None,
            **kwargs,
        )

    def to_dict(self, detail: str = "portable") -> Dict[str, Any]:
        """Return a recursive, versioned serialization payload.

        ``summary`` explicitly disables paired comparison after loading.
        ``portable`` stores pairwise results at alpha 0.05 without the large
        bootstrap matrix. ``full`` includes the matrix and influence arrays.
        """
        if detail not in {"summary", "portable", "full"}:
            raise ValueError("detail must be 'summary', 'portable', or 'full'")
        interval_alpha = 0.05
        if self.ci_info is not None and self.ci_info.method == "percentile":
            interval_alpha = float(self.ci_info.alpha)
        elif isinstance(self.metadata.get("bootstrap_ci"), dict):
            bootstrap_ci = self.metadata["bootstrap_ci"]
            if (
                bootstrap_ci.get("method") == "percentile"
                and bootstrap_ci.get("alpha") is not None
            ):
                interval_alpha = float(bootstrap_ci["alpha"])
        ci_lower, ci_upper = self.confidence_interval(alpha=interval_alpha)
        if self.ci_info is not None:
            interval_method = self.ci_info.method
        elif isinstance(self.metadata.get("bootstrap_ci"), dict):
            interval_method = str(
                self.metadata["bootstrap_ci"].get("method", "percentile")
            )
        elif isinstance(self.metadata.get("degrees_of_freedom"), dict):
            interval_method = "t"
        else:
            interval_method = "z"

        source_pairwise_state = self.metadata.get("_serialized_pairwise_state")
        clean_metadata = dict(self.metadata)
        clean_metadata.pop("_serialized_pairwise_state", None)
        result: Dict[str, Any] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "serialization_detail": detail,
            "method": self.method,
            "estimates": self.estimates,
            "standard_errors": self.standard_errors,
            "n_samples_used": self.n_samples_used,
            "target_policies": list(self.target_policies),
            "confidence_intervals": {
                "method": interval_method,
                "alpha": interval_alpha,
                "lower": ci_lower,
                "upper": ci_upper,
            },
            "metadata": clean_metadata,
            "units": self.units,
            "capabilities": {
                "calibrator_prediction": "unavailable_after_json",
            },
        }

        if self.bootstrap_samples is not None:
            matrix = np.asarray(self.bootstrap_samples)
            result["bootstrap_samples_summary"] = {
                "n_replicates": int(matrix.shape[0]),
                "n_policies": int(matrix.shape[1]),
            }
            if detail == "full":
                result["bootstrap_samples"] = matrix

        if detail == "full" and self.influence_functions:
            result["influence_functions"] = self.influence_functions

        if self.diagnostics:
            result["diagnostics"] = self.diagnostics.to_dict()
        if self.ci_info is not None:
            result["ci_info"] = asdict(self.ci_info)

        pairwise_state: Dict[str, Any] = {"capability": "unavailable"}
        source_capability = (
            source_pairwise_state.get("capability")
            if isinstance(source_pairwise_state, dict)
            else None
        )
        if detail == "summary":
            pairwise_state = {"capability": "unavailable"}
        elif isinstance(source_pairwise_state, dict) and source_capability in {
            "unavailable",
            "stored_alpha_only",
        }:
            # Serialization can discard state, never recreate state that a
            # prior summary/portable load explicitly marked unavailable.
            pairwise_state = dict(source_pairwise_state)
        elif detail == "full":
            pairwise_state = {"capability": "full"}
        elif detail == "portable" and len(self.estimates) > 1:
            comparisons: Dict[str, Dict[str, Any]] = {}
            recomputable_pairs: List[str] = []
            for i in range(len(self.estimates)):
                for j in range(i + 1, len(self.estimates)):
                    try:
                        comparison = self.compare_policies(i, j, alpha=0.05)
                    except (InferenceUnavailableError, ValueError):
                        continue
                    if comparison.get("method") == "paired_if_legacy":
                        continue
                    pair_key = f"{i}-{j}"
                    if comparison.get("method") == "independent_conservative":
                        recomputable_pairs.append(pair_key)
                        continue
                    comparison = dict(comparison)
                    for derived_key in _DERIVED_COMPARISON_KEYS:
                        comparison.pop(derived_key, None)
                    comparisons[pair_key] = comparison
            if comparisons or recomputable_pairs:
                pairwise_state = {
                    "capability": "stored_alpha_only",
                    "alpha": 0.05,
                    "comparisons": comparisons,
                    "recomputable_pairs": recomputable_pairs,
                }
        result["pairwise_inference_state"] = pairwise_state
        result["capabilities"]["paired_comparison"] = pairwise_state["capability"]

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

        return cast(Dict[str, Any], _json_safe(result))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EstimationResult":
        """Reconstruct a result without inventing omitted inference state."""
        schema = payload.get("schema_version")
        metadata = dict(payload.get("metadata") or {})
        policies = metadata.get("target_policies") or payload.get("target_policies")
        if policies and "target_policies" not in metadata:
            metadata["target_policies"] = list(policies)

        pairwise_state = payload.get("pairwise_inference_state")
        if not isinstance(pairwise_state, dict):
            pairwise_state = {"capability": "unavailable"}
        if schema != RESULT_SCHEMA_VERSION:
            pairwise_state = {
                "capability": "unavailable",
                "reason": "legacy_unversioned_result",
            }
            metadata["serialization_compatibility"] = "legacy_unverified"
        metadata["_serialized_pairwise_state"] = pairwise_state

        def _numeric_array(values: Any) -> np.ndarray:
            return np.asarray(
                [np.nan if value is None else float(value) for value in values],
                dtype=float,
            )

        ci_payload = payload.get("ci_info")
        ci_info = CIInfo(**ci_payload) if isinstance(ci_payload, dict) else None
        if ci_info is None and isinstance(payload.get("confidence_intervals"), dict):
            ci = payload["confidence_intervals"]
            if ci.get("method") == "percentile" and "lower" in ci and "upper" in ci:
                ci_info = CIInfo(
                    method="percentile",
                    alpha=float(ci.get("alpha", 0.05)),
                    lower=[np.nan if v is None else float(v) for v in ci["lower"]],
                    upper=[np.nan if v is None else float(v) for v in ci["upper"]],
                )

        diagnostics = _diagnostics_from_dict(payload.get("diagnostics"))
        influence = payload.get("influence_functions")
        if isinstance(influence, dict):
            influence = {
                str(policy): _numeric_array(values)
                for policy, values in influence.items()
            }
        else:
            influence = None
        bootstrap = payload.get("bootstrap_samples")
        if bootstrap is not None:
            bootstrap = np.asarray(
                [
                    [np.nan if value is None else float(value) for value in row]
                    for row in bootstrap
                ],
                dtype=float,
            )
        units_payload = payload.get("units")
        units = (
            ResultUnits.model_validate(units_payload)
            if isinstance(units_payload, dict)
            else None
        )
        return cls(
            estimates=_numeric_array(payload["estimates"]),
            standard_errors=_numeric_array(payload["standard_errors"]),
            n_samples_used={
                str(policy): count
                for policy, count in payload["n_samples_used"].items()
            },
            method=str(payload["method"]),
            influence_functions=influence,
            bootstrap_samples=bootstrap,
            diagnostics=diagnostics,
            calibrator=None,
            metadata=metadata,
            ci_info=ci_info,
            units=units,
        )

    @classmethod
    def from_json(cls, path: Any) -> "EstimationResult":
        """Load a versioned result JSON file."""
        with open(Path(path)) as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("EstimationResult JSON must contain an object")
        return cls.from_dict(payload)


def _json_safe(value: Any) -> Any:
    """Recursively convert supported scientific/Pydantic values to JSON data."""
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump())
    if is_dataclass(value):
        return _json_safe(asdict(value))  # type: ignore[arg-type]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=str)]
    raise TypeError(
        f"Unsupported value in EstimationResult serialization: "
        f"{type(value).__name__}"
    )


def _diagnostics_from_dict(payload: Any) -> Optional["DirectDiagnostics"]:
    if not isinstance(payload, dict):
        return None
    from ..diagnostics import DirectDiagnostics, Status

    values = dict(payload)
    values.pop("overall_status", None)
    raw_statuses = values.get("status_per_policy")
    if isinstance(raw_statuses, dict):
        converted = {}
        for policy, value in raw_statuses.items():
            try:
                converted[str(policy)] = Status(value)
            except (TypeError, ValueError):
                continue
        values["status_per_policy"] = converted
    allowed = {field.name for field in fields(DirectDiagnostics)}
    return DirectDiagnostics(
        **{key: value for key, value in values.items() if key in allowed}
    )


# Import at the end to resolve forward references
from ..diagnostics import DirectDiagnostics

# Update forward references - compatible with both Pydantic v1 and v2
if hasattr(EstimationResult, "model_rebuild"):
    # Pydantic v2
    EstimationResult.model_rebuild()
elif hasattr(EstimationResult, "update_forward_refs"):
    # Pydantic v1
    EstimationResult.update_forward_refs()
