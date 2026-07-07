"""Data models for CJE using Pydantic."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set, Tuple
from pydantic import BaseModel, Field
import numpy as np

logger = logging.getLogger(__name__)


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
        name: The winning policy. With reliable_only=True this is the best
            policy that passed the reliability gates; the raw point-estimate
            argmax wins only when it passed, or when every policy is flagged
            (all_flagged=True).
        index: Index of ``name`` in the estimates array.
        estimate: Point estimate of ``name``.
        flagged: Whether the returned policy failed the reliability gates.
        all_flagged: True when no policy with a usable estimate passed the
            gates (the returned argmax should not be crowned).
        runner_up: With reliable_only=True, the raw point-estimate argmax
            when it was flagged and demoted in favor of ``name`` (it beat
            ``name`` on point estimate but failed the gates). None otherwise.
    """

    name: str
    index: int
    estimate: float
    flagged: bool
    all_flagged: bool = False
    runner_up: Optional[str] = None


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
    influence_functions: Optional[Dict[str, np.ndarray]] = Field(
        None,
        description="Influence functions for each policy (when store_influence=True)",
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
            "paired difference inference. Omitted from to_dict()/JSON export."
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

    model_config = {"arbitrary_types_allowed": True}

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

    def best_policy(self, reliable_only: bool = True) -> PolicyVerdict:
        """Best policy, demoting winners that failed the reliability gates.

        The raw point-estimate argmax can be an adversarial policy the
        gates flagged as unreliable (verified on the bundled arena sample,
        where the 'unhelpful' policy won the raw argmax). With
        reliable_only=True (default) the verdict names the best policy
        that PASSED the gates; a flagged argmax is recorded as the
        demoted ``runner_up``. If every policy is flagged, the argmax is
        returned with ``all_flagged=True`` — do not crown it.

        Note: This replaced the 0.4.x ``best_policy() -> int`` (naive
        argmax index) in 0.5.0. Use ``verdict.index`` for the old value.

        Args:
            reliable_only: When True, only policies that passed the
                reliability gates (and are not CRITICAL) can win. When
                False, return the raw point-estimate argmax with its
                gate flag.

        Returns:
            PolicyVerdict (name, index, estimate, flagged, all_flagged,
            runner_up).

        Raises:
            ValueError: If there are no estimates, no target_policies
                metadata, or every estimate is NaN.
        """
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

        # Demote the flagged argmax: the trophy goes to the best reliable
        # policy (ties broken by policy name, matching the CLI's behavior).
        _, rel_name = max((float(estimates[i]), policy) for i, policy in reliable)
        rel_idx = policies.index(rel_name)
        return PolicyVerdict(
            name=rel_name,
            index=rel_idx,
            estimate=float(estimates[rel_idx]),
            flagged=False,
            all_flagged=False,
            runner_up=best_name,
        )

    def summary(self) -> str:
        """Compact text summary: per-policy estimates, CIs, gates, best policy.

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

        lines = [f"CJE Estimation Results (method: {self.method})"]
        for i, policy in enumerate(policies):
            gate = gates.get(policy)
            flag = ""
            if gate is not None and gate.flagged:
                flag = "  [gate: FLAGGED]"
            lines.append(
                f"  {policy:<{width}s}  {self.estimates[i]:.3f}  "
                f"95% CI [{ci_lower[i]:.3f}, {ci_upper[i]:.3f}]{flag}"
            )

        try:
            verdict = self.best_policy()
        except ValueError:
            lines.append("Best policy: none (no usable estimates)")
        else:
            if verdict.all_flagged:
                lines.append(
                    f"Best policy: none reliable (best by point estimate: "
                    f"{verdict.name}, flagged by the reliability gates)"
                )
            elif verdict.runner_up is not None:
                lines.append(
                    f"Best reliable policy: {verdict.name} (best by point "
                    f"estimate {verdict.runner_up} failed the reliability gates)"
                )
            else:
                lines.append(f"Best policy: {verdict.name}")

        if self.diagnostics is not None:
            lines.append(f"Status: {self.diagnostics.overall_status.value}")
        return "\n".join(lines)

    # Minimum NaN-filtered paired deltas required for the paired-bootstrap
    # comparison path; below this the percentile CI and sign-test p-value
    # are too coarse and compare_policies falls through to the next basis.
    _MIN_PAIRED_BOOTSTRAP_DELTAS = 100

    def compare_policies(
        self, policy1_idx: int, policy2_idx: int, alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Compare two policies with the most honest difference SE available.

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
            policy1_idx: Index of the first policy (order of
                ``metadata["target_policies"]``).
            policy2_idx: Index of the second policy.
            alpha: Significance level for ``significant`` and (paths 1-2)
                the returned CI.

        Returns:
            Dict with ``difference``, ``se_difference``, ``z_score``,
            ``p_value``, ``significant``, ``used_influence``, ``method``,
            and ``gate_flagged`` (names among the pair whose reliability
            gates are flagged — a comparison involving a flagged policy
            inherits that unreliability, however honest the SE); paths 1-2
            add ``ci_lower``/``ci_upper``/``alpha`` (plus ``n_replicates``
            for the bootstrap path and ``df``/``basis`` for the analytic
            path).
        """
        result = self._compare_paired_bootstrap(policy1_idx, policy2_idx, alpha)
        if result is None:
            result = self._compare_pairwise_inference(policy1_idx, policy2_idx, alpha)
        if result is None:
            result = self._compare_legacy(policy1_idx, policy2_idx, alpha)
        return self._annotate_gate_flags(result, policy1_idx, policy2_idx)

    def _annotate_gate_flags(
        self, result: Dict[str, Any], idx1: int, idx2: int
    ) -> Dict[str, Any]:
        """Surface reliability-gate refusals on the pair.

        A difference CI cannot repair a biased input: when a policy's
        estimate is unreliable (flagged gate / refused level claims, e.g.
        after a failed transport audit), every comparison involving it
        inherits that unreliability regardless of how honest the
        difference SE is. Adds ``gate_flagged`` (list of flagged policy
        names, possibly empty) and warns when non-empty.
        """
        flagged: List[str] = []
        policies = self.target_policies
        gates = self.gates
        for idx in (idx1, idx2):
            if 0 <= idx < len(policies):
                gate = gates.get(policies[idx])
                if gate is not None and (gate.flagged or gate.refuse_level_claims):
                    flagged.append(policies[idx])
        result["gate_flagged"] = flagged
        if flagged:
            logger.warning(
                "compare_policies: %s flagged by reliability gates — the "
                "difference estimate inherits that unreliability (a paired "
                "SE cannot correct a biased point estimate). Treat this "
                "comparison per the gates discipline.",
                ", ".join(repr(p) for p in flagged),
            )
        return result

    def _compare_paired_bootstrap(
        self, idx1: int, idx2: int, alpha: float
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
        df = max(int(df_raw), 1)
        if not np.isfinite(se_difference) or se_difference <= 0:
            return None

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
            "used_influence": basis != "independent",
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
            List of comparison dicts (one per pair, in (i, j) index order),
            each carrying ``policy1``/``policy2`` names in addition to the
            ``compare_policies`` keys.
        """
        if adjust not in (None, "bh"):
            raise ValueError(f"adjust must be None or 'bh', got {adjust!r}")

        policies = self.target_policies
        n = len(self.estimates)

        comparisons: List[Dict[str, Any]] = []
        for i in range(n):
            for j in range(i + 1, n):
                comparison = self.compare_policies(i, j, alpha=alpha)
                comparison["policy1"] = policies[i] if i < len(policies) else str(i)
                comparison["policy2"] = policies[j] if j < len(policies) else str(j)
                comparisons.append(comparison)

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
                f"<tr><td>{policy}</td><td>{est:.3f}</td><td>{se:.3f}</td><td>{ci_str}</td></tr>"
            )

        # Build full HTML
        html_parts = [
            '<div style="font-family: monospace;">',
            "<h4>CJE Estimation Results</h4>",
            f"<p><b>Method:</b> {self.method} | <b>Policies:</b> {len(policies)}</p>",
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
            >>> result = analyze_dataset("data.jsonl")
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

        # Add base policy if provided
        if base_policy_stats:
            if "mean" not in base_policy_stats:
                raise ValueError("base_policy_stats must contain 'mean' key")
            estimates["base"] = base_policy_stats["mean"]
            standard_errors["base"] = base_policy_stats.get("se", 0.0)

        # Add target policies
        for i, policy in enumerate(policies):
            estimates[policy] = float(self.estimates[i])
            standard_errors[policy] = float(self.standard_errors[i])

        # Call visualization function
        from pathlib import Path

        return plot_policy_estimates(
            estimates=estimates,
            standard_errors=standard_errors,
            oracle_values=oracle_values,
            policy_labels=policy_labels,
            save_path=Path(save_path) if save_path else None,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        The ``bootstrap_samples`` replicate matrix is intentionally omitted
        (large, and a 2-D array would break flat JSON serialization); only
        a small shape summary is recorded when it is present.
        """
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

        # Summary only — the (B, P) matrix itself is not serialized
        if self.bootstrap_samples is not None:
            matrix = np.asarray(self.bootstrap_samples)
            if matrix.ndim == 2:
                result["bootstrap_samples_summary"] = {
                    "n_replicates": int(matrix.shape[0]),
                    "n_policies": int(matrix.shape[1]),
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

        return result


# Import at the end to resolve forward references
from ..diagnostics import DirectDiagnostics

# Update forward references - compatible with both Pydantic v1 and v2
if hasattr(EstimationResult, "model_rebuild"):
    # Pydantic v2
    EstimationResult.model_rebuild()
elif hasattr(EstimationResult, "update_forward_refs"):
    # Pydantic v1
    EstimationResult.update_forward_refs()
