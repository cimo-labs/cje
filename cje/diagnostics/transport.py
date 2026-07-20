"""Residual transport diagnostics for a fitted judge calibrator.

Scalar score-range support is handled by :mod:`cje.diagnostics.reward_boundary`.
This module answers a different question using held-out oracle probes: is the
target-policy mean residual precise enough to place inside or outside a
predeclared practical-bias margin?
"""

from dataclasses import dataclass, field
import logging
import warnings
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats


logger = logging.getLogger(__name__)

TransportStatus = Literal["PASS", "FAIL", "INCONCLUSIVE", "NOT_GRADED", "NOT_CHECKED"]


@dataclass(frozen=True)
class TransportAuditConfig:
    """Optional held-out transport probes for ``analyze_dataset``.

    Probe keys are target-policy names. Each probe row uses the analysis call's
    judge/oracle field names and public scales. ``delta_max_by_policy`` is
    deliberately per policy: omitting a margin keeps that audit descriptive
    and returns ``NOT_GRADED`` rather than borrowing an arbitrary threshold.

    Unit contract: ``analyze_dataset`` converts probe oracle labels from the
    call's public oracle scale into the result OUTPUT scale before auditing,
    so every margin in ``delta_max_by_policy`` is in OUTPUT units — the units
    of ``result.estimates`` (``output_scale`` when declared) — NOT the units
    the probe ``oracle_label`` values were supplied in. When the output scale
    differs from the oracle scale, a margin stated in probe-label units will
    be silently wrong; restate it on the output scale.
    """

    probes_by_policy: Mapping[str, Sequence[Any]]
    delta_max_by_policy: Mapping[str, float] = field(default_factory=dict)
    bins: int = 10
    alpha: float = 0.05
    family_size: Optional[int] = None
    min_effective_clusters: float = 20.0

    def __post_init__(self) -> None:
        if not isinstance(self.probes_by_policy, Mapping):
            raise TypeError(
                "probes_by_policy must map target-policy names to probe rows"
            )
        if not isinstance(self.delta_max_by_policy, Mapping):
            raise TypeError(
                "delta_max_by_policy must map target-policy names to positive values"
            )
        if any(
            not isinstance(policy, str) or not policy
            for policy in self.probes_by_policy
        ):
            raise ValueError("probe policy names must be non-empty strings")
        if any(
            not isinstance(policy, str) or not policy
            for policy in self.delta_max_by_policy
        ):
            raise ValueError("margin policy names must be non-empty strings")
        normalized_probes: Dict[str, Tuple[Any, ...]] = {}
        for policy, rows in self.probes_by_policy.items():
            if isinstance(rows, (str, bytes)):
                raise TypeError(
                    f"probes_by_policy[{policy!r}] must be a sequence of rows"
                )
            try:
                normalized_probes[policy] = tuple(rows)
            except TypeError as exc:
                raise TypeError(
                    f"probes_by_policy[{policy!r}] must be a sequence of rows"
                ) from exc
        object.__setattr__(self, "probes_by_policy", normalized_probes)
        resolved_family = (
            self.family_size
            if self.family_size is not None
            else max(len(self.probes_by_policy), 1)
        )
        if resolved_family < max(len(self.probes_by_policy), 1):
            raise ValueError(
                "family_size cannot be smaller than the number of configured "
                "policy probes"
            )
        _validate_options(
            self.alpha,
            self.bins,
            resolved_family,
            None,
            self.min_effective_clusters,
        )
        normalized_margins: Dict[str, float] = {}
        for policy, margin in self.delta_max_by_policy.items():
            if isinstance(margin, bool):
                raise ValueError(
                    f"delta_max_by_policy[{policy!r}] must be finite and positive, "
                    f"got {margin}"
                )
            try:
                numeric_margin = float(margin)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"delta_max_by_policy[{policy!r}] must be numeric, got "
                    f"{margin!r}"
                ) from exc
            if not np.isfinite(numeric_margin) or numeric_margin <= 0:
                raise ValueError(
                    f"delta_max_by_policy[{policy!r}] must be finite and positive, "
                    f"got {margin}"
                )
            normalized_margins[policy] = numeric_margin
        object.__setattr__(self, "delta_max_by_policy", normalized_margins)

    @property
    def resolved_family_size(self) -> int:
        """Multiplicity family size used for every configured probe audit."""
        return (
            self.family_size
            if self.family_size is not None
            else max(len(self.probes_by_policy), 1)
        )


@dataclass
class TransportDiagnostics:
    """Result of a held-out residual transport audit.

    ``PASS`` means the Bonferroni-adjusted confidence interval is wholly
    inside the declared ``delta_max`` margin. ``FAIL`` means it is wholly
    outside that margin. An overlapping interval is ``INCONCLUSIVE``; no
    declared margin is ``NOT_GRADED``. ``NOT_CHECKED`` is reserved for
    high-level result records when no independent probe was supplied; it is
    not fabricated by this low-level audit. None of these states is a
    scalar-support diagnostic.

    Below the effective-cluster floor a decisive out-of-margin interval
    still grades ``FAIL``; only ``PASS`` requires the floor, so an
    under-sized probe cannot defeat the hard gate.

    Units: ``delta_hat``, ``delta_ci``, ``delta_se``, ``delta_max``,
    ``ci_half_width``, ``min_margin_for_pass``, and ``detectable_bias_80``
    are all in the units of the probe ``oracle_label`` values that were
    audited (for ``analyze_dataset`` audits, the result OUTPUT scale).

    ``simultaneous_confidence_level`` is the family-wise coverage of the
    Bonferroni intervals (``1 - alpha``); ``per_audit_confidence_level`` is
    the nominal level of each individual interval (``1 - alpha /
    family_size``).

    ``coverage`` is retained as a deprecated alias for
    ``probe_bin_occupancy`` so older serialized consumers keep working. The
    metric is display-only and never affects the verdict.
    """

    status: TransportStatus
    delta_hat: float
    delta_ci: Tuple[float, float]
    delta_se: float
    decile_residuals: List[float]
    decile_counts: List[int]
    recommended_action: str
    n_probe: int
    group_label: Optional[str] = None
    probe_bin_occupancy: float = 0.0
    coverage: Optional[float] = None
    delta_max: Optional[float] = None
    n_clusters: int = 0
    effective_clusters: float = 0.0
    alpha: float = 0.05
    family_size: int = 1
    simultaneous_confidence_level: float = 0.95
    per_audit_confidence_level: float = 0.95
    ci_half_width: float = float("nan")
    min_margin_for_pass: float = float("nan")
    detectable_bias_80: float = float("nan")
    weighted: bool = False
    reason_code: str = ""
    cluster_field: str = "prompt_id"
    covariate_names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        # Compatibility for callers constructing the old dataclass by hand.
        if self.coverage is not None and self.probe_bin_occupancy == 0.0:
            self.probe_bin_occupancy = float(self.coverage)
        self.coverage = float(self.probe_bin_occupancy)
        if self.status == "WARN":  # type: ignore[comparison-overlap]
            self.status = "INCONCLUSIVE"
            if not self.reason_code:
                self.reason_code = "legacy_warn"

    def summary(self) -> str:
        """Return a concise, claim-calibrated summary."""
        parts = [f"Residual transport: {self.status}"]
        if self.group_label:
            parts.append(f"Group: {self.group_label}")
        parts.append(f"N={self.n_probe} ({self.n_clusters} clusters)")
        parts.append(
            f"delta: {self.delta_hat:+.3f} "
            f"(CI: [{self.delta_ci[0]:+.3f}, {self.delta_ci[1]:+.3f}])"
        )
        if self.delta_max is not None:
            parts.append(f"margin: +/-{self.delta_max:.3f}")
        if self.status != "PASS":
            parts.append(f"Action: {self.recommended_action}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "status": self.status,
            "delta_hat": float(self.delta_hat),
            "delta_ci": [float(self.delta_ci[0]), float(self.delta_ci[1])],
            "delta_se": float(self.delta_se),
            "delta_max": self.delta_max,
            "decile_residuals": [
                float(r) if np.isfinite(r) else None for r in self.decile_residuals
            ],
            "decile_counts": [int(c) for c in self.decile_counts],
            "probe_bin_occupancy": float(self.probe_bin_occupancy),
            # Deprecated compatibility alias; not scalar support coverage.
            "coverage": float(self.probe_bin_occupancy),
            "recommended_action": self.recommended_action,
            "reason_code": self.reason_code,
            "n_probe": int(self.n_probe),
            "n_clusters": int(self.n_clusters),
            "effective_clusters": float(self.effective_clusters),
            "alpha": float(self.alpha),
            "family_size": int(self.family_size),
            "simultaneous_confidence_level": float(self.simultaneous_confidence_level),
            "per_audit_confidence_level": float(self.per_audit_confidence_level),
            "ci_half_width": float(self.ci_half_width),
            "min_margin_for_pass": float(self.min_margin_for_pass),
            "detectable_bias_80": float(self.detectable_bias_80),
            "weighted": bool(self.weighted),
            "cluster_field": self.cluster_field,
            "covariate_names": self.covariate_names,
            "group_label": self.group_label,
        }

    def plot(self, ax: Optional[Any] = None, figsize: tuple = (10, 5)) -> Any:
        """Plot residual diagnostics (requires the ``viz`` extra)."""
        from ..visualization.transport import plot_transport_diagnostics

        return plot_transport_diagnostics(self, ax=ax, figsize=figsize)


def audit_transportability(
    calibrator: Any,
    probe_samples: Sequence[Any],
    bins: int = 10,
    group_label: Optional[str] = None,
    alpha: float = 0.05,
    *,
    delta_max: Optional[float] = None,
    cluster_ids: Optional[Union[Sequence[Any], np.ndarray]] = None,
    sample_weights: Optional[Union[Sequence[float], np.ndarray]] = None,
    covariates: Optional[Any] = None,
    family_size: int = 1,
    min_effective_clusters: float = 20.0,
) -> TransportDiagnostics:
    """Audit target-policy mean residuals on held-out oracle probes.

    The confidence interval uses a prompt-cluster sandwich variance with a
    finite-cluster t critical value. ``family_size`` applies a Bonferroni
    adjustment across every policy/group audit used in the same decision.
    Probe rows must not have been used to fit ``calibrator``.

    Args:
        calibrator: A fitted object exposing ``predict``.
        probe_samples: Dict or ``Sample`` records with judge and oracle values.
        bins: Score-quantile bins for display only.
        group_label: Optional policy/group label.
        alpha: Family-wise error rate.
        delta_max: Maximum practically acceptable absolute mean residual.
            Stated in the same units as the probe ``oracle_label`` values:
            this audit grades ``oracle_label - calibrator.predict(...)``
            directly, with no rescaling. With ``None``, diagnostics are
            ``NOT_GRADED`` and can never ``PASS`` or ``FAIL``.
        cluster_ids: Independent cluster IDs. Defaults to record ``prompt_id``.
        sample_weights: Positive analysis weights, such as inverse inclusion
            probabilities for a disproportionate probe design.
        covariates: Probe covariate matrix passed to the fitted calibrator.
            When omitted, named covariates are extracted from record metadata.
        family_size: Number of predeclared policy/group audits in the family.
        min_effective_clusters: Minimum Kish-effective clusters for grading.

    Returns:
        Structured residual-audit diagnostics.
    """
    _validate_options(alpha, bins, family_size, delta_max, min_effective_clusters)
    if delta_max is None:
        warnings.warn(
            "Since 0.6.0, audits without delta_max are NOT_GRADED and can "
            "never PASS or FAIL; 0.5.x returned PASS/WARN/FAIL under a "
            "zero-null test. Pass delta_max (practical-equivalence margin) "
            "to grade.",
            FutureWarning,
            stacklevel=2,
        )
    if len(probe_samples) == 0:
        raise ValueError("probe_samples must contain at least one labeled row")

    scores, labels, inferred_clusters, inferred_weights = _extract_probe_fields(
        probe_samples
    )
    n_probe = len(labels)
    clusters = _validated_clusters(
        cluster_ids if cluster_ids is not None else inferred_clusters, n_probe
    )
    weights = _validated_weights(
        sample_weights if sample_weights is not None else inferred_weights, n_probe
    )
    covariate_matrix, covariate_names = _validated_covariates(
        calibrator, probe_samples, covariates, n_probe
    )

    predictions = _predict(calibrator, scores, covariate_matrix)
    if predictions.shape != labels.shape:
        raise ValueError(
            f"calibrator returned shape {predictions.shape}, expected {labels.shape}"
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError("calibrator predictions contain NaN or infinity")
    residuals = labels - predictions

    delta_hat, delta_se, n_clusters, effective_clusters = _clustered_mean_se(
        residuals, clusters, weights
    )
    df = max(int(np.floor(effective_clusters)) - 1, 1)
    per_audit_alpha = alpha / family_size
    t_crit = float(stats.t.ppf(1 - per_audit_alpha / 2, df))
    ci_half_width = t_crit * delta_se
    delta_ci = (delta_hat - ci_half_width, delta_hat + ci_half_width)
    # Bonferroni: each interval is at level 1 - alpha/K; the family-wise
    # (simultaneous) coverage of all K intervals together is 1 - alpha.
    per_audit_confidence_level = 1.0 - per_audit_alpha
    simultaneous_confidence_level = 1.0 - alpha

    decile_residuals, decile_counts, bin_occupancy = _binned_residuals(
        scores, residuals, bins
    )
    status, action, reason_code = _classify_status(
        delta_ci=delta_ci,
        delta_max=delta_max,
        effective_clusters=effective_clusters,
        min_effective_clusters=min_effective_clusters,
    )
    min_margin = max(abs(delta_ci[0]), abs(delta_ci[1]))
    detectable_bias_80 = float((t_crit + stats.norm.ppf(0.80)) * delta_se)

    logger.info(
        "Residual transport audit: %s | delta=%+.3f | CI=[%+.3f, %+.3f] "
        "| margin=%s | clusters=%d (effective %.1f)",
        status,
        delta_hat,
        delta_ci[0],
        delta_ci[1],
        "not declared" if delta_max is None else f"+/-{delta_max:.3f}",
        n_clusters,
        effective_clusters,
    )

    return TransportDiagnostics(
        status=status,
        delta_hat=delta_hat,
        delta_ci=delta_ci,
        delta_se=delta_se,
        delta_max=delta_max,
        decile_residuals=decile_residuals,
        decile_counts=decile_counts,
        probe_bin_occupancy=bin_occupancy,
        recommended_action=action,
        reason_code=reason_code,
        n_probe=n_probe,
        n_clusters=n_clusters,
        effective_clusters=effective_clusters,
        alpha=alpha,
        family_size=family_size,
        simultaneous_confidence_level=simultaneous_confidence_level,
        per_audit_confidence_level=per_audit_confidence_level,
        ci_half_width=ci_half_width,
        min_margin_for_pass=min_margin,
        detectable_bias_80=detectable_bias_80,
        weighted=sample_weights is not None or not np.allclose(weights, 1.0),
        group_label=group_label,
        covariate_names=covariate_names,
    )


def _validate_options(
    alpha: float,
    bins: int,
    family_size: int,
    delta_max: Optional[float],
    min_effective_clusters: float,
) -> None:
    if not 0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    if not isinstance(bins, int) or bins < 1:
        raise ValueError(f"bins must be a positive integer, got {bins!r}")
    if (
        isinstance(family_size, bool)
        or not isinstance(family_size, int)
        or family_size < 1
    ):
        raise ValueError(f"family_size must be a positive integer, got {family_size!r}")
    if delta_max is not None and (not np.isfinite(delta_max) or delta_max <= 0):
        raise ValueError(f"delta_max must be finite and positive, got {delta_max}")
    if not np.isfinite(min_effective_clusters) or min_effective_clusters < 2:
        raise ValueError("min_effective_clusters must be finite and at least 2")


def _record_field(sample: Any, name: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        if name in sample:
            return sample[name]
        metadata = sample.get("metadata") or {}
        return metadata.get(name, default)
    if hasattr(sample, name):
        value = getattr(sample, name)
        if value is not None:
            return value
    metadata = getattr(sample, "metadata", None) or {}
    return metadata.get(name, default)


def _extract_probe_fields(
    probe_samples: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    scores: List[float] = []
    labels: List[float] = []
    clusters: List[str] = []
    weights: List[float] = []
    for i, sample in enumerate(probe_samples):
        score = _record_field(sample, "judge_score")
        label = _record_field(sample, "oracle_label")
        sample_id = _record_field(sample, "prompt_id", f"sample_{i}")
        if score is None:
            raise ValueError(f"Sample {sample_id} missing judge_score")
        if label is None:
            raise ValueError(f"Sample {sample_id} missing oracle_label")
        scores.append(float(score))
        labels.append(float(label))
        clusters.append(str(sample_id))
        weights.append(float(_record_field(sample, "sample_weight", 1.0)))

    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=float)
    if not np.all(np.isfinite(score_array)):
        raise ValueError("probe judge scores contain NaN or infinity")
    if not np.all(np.isfinite(label_array)):
        raise ValueError("probe oracle labels contain NaN or infinity")
    return score_array, label_array, clusters, np.asarray(weights, dtype=float)


def _validated_clusters(
    cluster_ids: Union[Sequence[Any], np.ndarray], n_probe: int
) -> np.ndarray:
    if len(cluster_ids) != n_probe:
        raise ValueError(
            f"cluster_ids length ({len(cluster_ids)}) must match probes ({n_probe})"
        )
    clusters = np.asarray([str(value) for value in cluster_ids], dtype=object)
    if any(value in ("", "None") for value in clusters):
        raise ValueError("cluster_ids must be non-empty and non-missing")
    return clusters


def _validated_weights(
    weights: Union[Sequence[float], np.ndarray], n_probe: int
) -> np.ndarray:
    if len(weights) != n_probe:
        raise ValueError(
            f"sample_weights length ({len(weights)}) must match probes ({n_probe})"
        )
    result = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("sample_weights must be finite and strictly positive")
    return result


def _validated_covariates(
    calibrator: Any,
    probe_samples: Sequence[Any],
    covariates: Optional[Any],
    n_probe: int,
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    names = list(getattr(calibrator, "covariate_names", []) or [])
    if covariates is None and names:
        rows = []
        for i, sample in enumerate(probe_samples):
            row = []
            for name in names:
                value = _record_field(sample, name)
                if value is None:
                    raise ValueError(
                        f"Probe row {i} is missing fitted covariate {name!r}"
                    )
                row.append(value)
            rows.append(row)
        covariates = rows
    if covariates is None:
        return None, names or None
    matrix = np.asarray(covariates, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[0] != n_probe:
        raise ValueError(
            f"covariates must have shape ({n_probe}, k), got {matrix.shape}"
        )
    if names and matrix.shape[1] != len(names):
        raise ValueError(
            f"covariates has {matrix.shape[1]} columns but calibrator expects "
            f"{len(names)} ({', '.join(names)})"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariates contain NaN or infinity")
    return matrix, names or None


def _predict(
    calibrator: Any, scores: np.ndarray, covariates: Optional[np.ndarray]
) -> np.ndarray:
    if covariates is None:
        predicted = calibrator.predict(scores)
    else:
        predicted = calibrator.predict(scores, covariates=covariates)
    return np.asarray(predicted, dtype=float).reshape(-1)


def _clustered_mean_se(
    residuals: np.ndarray, clusters: np.ndarray, weights: np.ndarray
) -> Tuple[float, float, int, float]:
    total_weight = float(np.sum(weights))
    delta_hat = float(np.sum(weights * residuals) / total_weight)
    unique_clusters, inverse = np.unique(clusters, return_inverse=True)
    n_clusters = len(unique_clusters)
    cluster_weights = np.bincount(inverse, weights=weights)
    effective_clusters = float(total_weight**2 / np.sum(np.square(cluster_weights)))
    if n_clusters < 2:
        return delta_hat, float("inf"), n_clusters, effective_clusters
    cluster_scores = np.bincount(inverse, weights=weights * (residuals - delta_hat))
    variance = float(
        n_clusters
        / (n_clusters - 1)
        * np.sum(np.square(cluster_scores))
        / total_weight**2
    )
    return delta_hat, float(np.sqrt(max(variance, 0.0))), n_clusters, effective_clusters


def _binned_residuals(
    scores: np.ndarray, residuals: np.ndarray, bins: int
) -> Tuple[List[float], List[int], float]:
    bin_edges = np.unique(np.quantile(scores, np.linspace(0, 1, bins + 1)))
    if len(bin_edges) < 2:
        bin_edges = np.array([scores.min() - 1e-6, scores.max() + 1e-6])
    indices = np.digitize(scores, bin_edges[1:-1])
    actual_bins = len(bin_edges) - 1
    means: List[float] = []
    counts: List[int] = []
    for bin_index in range(actual_bins):
        mask = indices == bin_index
        count = int(mask.sum())
        counts.append(count)
        means.append(float(residuals[mask].mean()) if count else float("nan"))
    occupancy = float(sum(count >= 3 for count in counts) / max(actual_bins, 1))
    return means, counts, occupancy


def _classify_status(
    *,
    delta_ci: Tuple[float, float],
    delta_max: Optional[float],
    effective_clusters: float,
    min_effective_clusters: float,
) -> Tuple[TransportStatus, str, str]:
    if delta_max is None:
        return (
            "NOT_GRADED",
            "declare a practical residual margin before using this audit as a gate",
            "margin_not_declared",
        )
    lower, upper = delta_ci
    # A CI wholly outside the margin FAILs even below the effective-cluster
    # floor: the floor exists because cluster-robust intervals under-cover on
    # few clusters, which withholds PASS, but an entire interval beyond the
    # margin is decisive evidence of unacceptable bias, not low power.
    if lower > delta_max or upper < -delta_max:
        return (
            "FAIL",
            "do not reuse this calibration; collect target labels and refit",
            "unacceptable_mean_residual",
        )
    if effective_clusters < min_effective_clusters:
        return (
            "INCONCLUSIVE",
            "collect more independent oracle-probe clusters",
            "insufficient_effective_clusters",
        )
    if lower >= -delta_max and upper <= delta_max:
        return (
            "PASS",
            "the mean residual is established within the declared margin",
            "",
        )
    return (
        "INCONCLUSIVE",
        "collect more independent oracle-probe clusters to resolve the margin",
        "interval_crosses_margin",
    )


def compute_residuals(
    calibrator: Any,
    data: List[Dict[str, Any]],
    sort_by: Optional[Literal["residual", "abs_residual"]] = "residual",
    *,
    covariates: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Compute row residuals for inspection; this function does not grade them."""
    if not data:
        return []
    scores, labels, _, _ = _extract_probe_fields(data)
    matrix, _ = _validated_covariates(calibrator, data, covariates, len(data))
    predictions = _predict(calibrator, scores, matrix)
    results: List[Dict[str, Any]] = []
    for sample, calibrated, label in zip(data, predictions, labels):
        enriched = dict(sample)
        enriched["calibrated"] = float(calibrated)
        enriched["residual"] = float(label - calibrated)
        results.append(enriched)
    if sort_by == "residual":
        results.sort(key=lambda row: row["residual"])
    elif sort_by == "abs_residual":
        results.sort(key=lambda row: abs(row["residual"]), reverse=True)
    elif sort_by is not None:
        raise ValueError("sort_by must be 'residual', 'abs_residual', or None")
    return results
