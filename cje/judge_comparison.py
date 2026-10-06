"""Compare several judges against one oracle on the same labelled rows.

Passing judges to ``analyze_dataset`` as if they were policies fits one
calibrator to all of them and detects their score scales jointly, and the
"difference" it reports between them is about zero by construction (issue
#71). ``compare_judges`` instead fits one cross-fitted ``JudgeCalibrator`` per
judge, on folds shared by every judge, and reports each judge's out-of-fold
calibration quality and, against a reference judge, the label multiplier
implied by the within-policy R². Intervals come from a paired prompt-cluster
bootstrap that refits every judge in every replicate.
"""

import logging
import math
import warnings
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, cast

import numpy as np

from .array_api import _factorize_clusters
from .calibration.flexible_calibrator import validate_oracle_labels
from .calibration.judge import JudgeCalibrator
from .diagnostics.robust_inference import get_oof_predictions

logger = logging.getLogger(__name__)

Interval = Tuple[float, float]

# Below this many labelled prompt clusters the comparison still runs, with a
# warning (the same 20-cluster line calibrated_mean_ci's "auto" rule uses).
_FEW_LABELLED_CLUSTERS = 20

# Per-judge statistics, in the column order of the bootstrap arrays.
_STATS = ("oof_rmse", "r2_pooled", "r2_within", "var_f", "var_residual")


@dataclass(frozen=True)
class JudgeQuality:
    """One judge's out-of-fold calibration quality (a row of ``table``).

    Residuals are ``r = y - o`` on the labelled rows, where ``o`` is the
    judge's out-of-fold calibrated prediction. Every ``*_ci`` is a percentile
    interval from the paired prompt-cluster bootstrap.

    Attributes:
        judge: Judge name.
        n_labelled_rows: Labelled rows used (the same rows for every judge).
        n_labelled_clusters: Labelled prompt clusters (the fold and
            resampling unit).
        selected_mode: Calibration mode of the full-sample fit
            ("monotone" or "two_stage"); bootstrap refits keep it.
        covariates_used: Whether the full-sample fit used the covariates.
        oof_rmse: Root mean squared out-of-fold residual over the labelled
            rows (uncentred; ``fit_cv``'s ``oof_rmse``).
        r2_pooled: ``1 - SS(r) / SS(y)`` over all labelled rows as one group,
            with ``r`` and ``y`` centred.
        r2_within: The same with ``r`` and ``y`` centred within each policy.
            This is the R² that sets label savings. It equals ``r2_pooled``
            when ``policy_ids`` is None.
        var_f: Within-policy variance of the full-sample calibrated
            prediction over every row of the labelled policies (``Var(f)``).
        var_residual: Within-policy variance of the out-of-fold residual over
            the labelled rows, ``(1 - r2_within)`` times the within-policy
            variance of the labels (``Var(Y - f)``).
    """

    judge: str
    n_labelled_rows: int
    n_labelled_clusters: int
    selected_mode: str
    covariates_used: bool
    oof_rmse: float
    oof_rmse_ci: Interval
    r2_pooled: float
    r2_pooled_ci: Interval
    r2_within: float
    r2_within_ci: Interval
    var_f: float
    var_residual: float


@dataclass(frozen=True)
class JudgePair:
    """One judge against the reference judge (a row of ``pairwise``).

    Every ``*_ci`` is a percentile interval from the paired prompt-cluster
    bootstrap: each replicate reweights the prompt clusters once and refits
    both judges on the same weights.

    Attributes:
        judge: The compared judge, ``J``.
        reference: The reference judge.
        r2_within_diff: ``r2_within[J] - r2_within[reference]``.
        oof_rmse_diff: ``oof_rmse[J] - oof_rmse[reference]``.
        label_multiplier: ``(1 - R²_reference) / (1 - R²_J)`` with the
            within-policy R²: the labels the reference judge needs per label
            of ``J`` for an equal interval width on a policy mean, when
            unlabelled rows are plentiful. Above 1, ``J`` saves labels. It
            treats labelled rows as independent; NaN when ``J``'s
            out-of-fold residuals have zero variance.
        variance_ratio_at_n: With ``n_unlabeled`` = U, the predicted variance
            ratio ``V_reference / V_J`` of a policy mean, where ``V = Var(f) /
            N + Var(Y - f) / n``, ``n`` is the observed labelled rows per
            policy and ``N = n + U``. None without ``n_unlabeled``.
        label_multiplier_at_n: The labels the reference needs per label of
            ``J`` for ``V_reference = V_J`` at the same ``N``. It is capped at
            ``N / n`` (every row labelled), and tends to ``label_multiplier``
            as U grows. None without ``n_unlabeled``.
        label_multiplier_at_n_capped: True when the point value hit that cap:
            under the model the reference cannot match ``J`` even with every
            row labelled. None without ``n_unlabeled``.
    """

    judge: str
    reference: str
    r2_within_diff: float
    r2_within_diff_ci: Interval
    oof_rmse_diff: float
    oof_rmse_diff_ci: Interval
    label_multiplier: float
    label_multiplier_ci: Interval
    variance_ratio_at_n: Optional[float] = None
    variance_ratio_at_n_ci: Optional[Interval] = None
    label_multiplier_at_n: Optional[float] = None
    label_multiplier_at_n_ci: Optional[Interval] = None
    label_multiplier_at_n_capped: Optional[bool] = None


@dataclass
class JudgeComparison:
    """Result of `compare_judges`.

    Attributes:
        table: One `JudgeQuality` per judge, in input order.
        pairwise: One `JudgePair` per non-reference judge, in input order.
        reference: The reference judge.
        alpha: Significance level of every interval.
        calibrators: Full-sample `JudgeCalibrator` per judge, fitted on that
            judge's own score scale (reusable, e.g. with `transport_audit`).
        diagnostics: Fold, bootstrap, policy and planning details.
    """

    table: List[JudgeQuality]
    pairwise: List[JudgePair]
    reference: str
    alpha: float
    calibrators: Dict[str, JudgeCalibrator]
    diagnostics: Dict[str, Any]

    def row(self, judge: str) -> JudgeQuality:
        """The table row of one judge."""
        for quality in self.table:
            if quality.judge == judge:
                return quality
        raise KeyError(f"No judge named {judge!r}.")

    def pair(self, judge: str) -> JudgePair:
        """The comparison of one non-reference judge with the reference."""
        for pair in self.pairwise:
            if pair.judge == judge:
                return pair
        raise KeyError(f"No pairwise comparison for judge {judge!r}.")

    def summary(self) -> str:
        """Fixed-width table plus one block per judge compared."""
        level = f"{100 * (1 - self.alpha):g}%"
        first = self.table[0]
        lines = [
            f"Judge comparison (reference: {self.reference}; "
            f"{first.n_labelled_rows} labelled rows in "
            f"{first.n_labelled_clusters} prompt clusters; {level} intervals "
            f"from {self.diagnostics['bootstrap']['n_bootstrap']} paired "
            "prompt-cluster bootstrap refits)",
            f"{'judge':<14} {'mode':<10} {'OOF RMSE':<23} {'R² within':<23} "
            f"{'R² pooled':<10} {'Var(f)':<9} {'Var(Y-f)':<9}",
        ]
        for q in self.table:
            lines.append(
                f"{q.judge:<14} {q.selected_mode:<10} "
                f"{_fmt_ci(q.oof_rmse, q.oof_rmse_ci):<23} "
                f"{_fmt_ci(q.r2_within, q.r2_within_ci):<23} "
                f"{_fmt(q.r2_pooled):<10} {_fmt(q.var_f, 4):<9} "
                f"{_fmt(q.var_residual, 4):<9}"
            )
        planned = self.diagnostics.get("planned")
        for p in self.pairwise:
            lines.append(
                f"{p.judge} vs {p.reference}: R² within "
                f"{_fmt_ci(p.r2_within_diff, p.r2_within_diff_ci, sign=True)}, "
                f"OOF RMSE {_fmt_ci(p.oof_rmse_diff, p.oof_rmse_diff_ci, sign=True)}"
            )
            lines.append(
                f"  {p.reference} (reference) needs "
                f"{_fmt_ci(p.label_multiplier, p.label_multiplier_ci, 2)} labels "
                f"per label of {p.judge} for equal interval width (within policy, "
                "plentiful unlabelled rows)"
            )
            if planned is not None and p.label_multiplier_at_n is not None:
                assert p.variance_ratio_at_n is not None
                capped = " (capped: every row labelled)" * bool(
                    p.label_multiplier_at_n_capped
                )
                lines.append(
                    f"  at {planned['n_unlabeled']} unlabelled rows per policy: "
                    "variance ratio "
                    f"{_fmt_ci(p.variance_ratio_at_n, p.variance_ratio_at_n_ci, 2)}; "
                    f"{p.reference} needs "
                    f"{_fmt_ci(p.label_multiplier_at_n, p.label_multiplier_at_n_ci, 2)}"
                    f" labels per label of {p.judge}{capped}"
                )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict: NaN becomes None, tuples become lists.

        Calibrators are omitted. ``pd.DataFrame(cmp.to_dict()["table"])``
        gives the table as a DataFrame.
        """
        return cast(
            Dict[str, Any],
            _json_safe(
                {
                    "reference": self.reference,
                    "alpha": self.alpha,
                    "table": [asdict(q) for q in self.table],
                    "pairwise": [asdict(p) for p in self.pairwise],
                    "diagnostics": self.diagnostics,
                }
            ),
        )


def compare_judges(
    scores_by_judge: Mapping[str, Any],
    oracle_labels: Any,
    cluster_ids: Optional[Any],
    *,
    policy_ids: Optional[Any] = None,
    covariates: Optional[Any] = None,
    judge_scales: Optional[Mapping[str, Tuple[float, float]]] = None,
    reference: Optional[str] = None,
    n_unlabeled: Optional[int] = None,
    alpha: float = 0.05,
    n_folds: int = 5,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> JudgeComparison:
    """Compare several judges against one oracle on the same labelled rows.

    Each judge gets its own cross-fitted `JudgeCalibrator` (two-stage when
    covariates are given, otherwise auto-selected, as in
    `calibrated_mean_ci`). Folds are a seeded hash over the labelled
    prompt-cluster ids, so every judge gets the same folds and the
    out-of-fold predictions are paired. Use this instead of passing judges to
    `analyze_dataset` as policies, which pools them into one calibrator.

    Per judge, ``table`` reports the out-of-fold RMSE, the out-of-fold R²
    pooled and within policy, and the variance components ``Var(f)`` and
    ``Var(Y - f)``. Against the reference, ``pairwise`` reports the
    differences in within-policy R² and RMSE and the label multiplier
    ``(1 - R²_reference) / (1 - R²_J)``, plus finite-N versions when
    ``n_unlabeled`` is given. Every interval comes from a paired bootstrap:
    each replicate draws one positive Exp(1) weight per prompt cluster (the
    library's refit-bootstrap scheme), shares it across judges and policies,
    and refits each judge's calibrator with the judge's full-sample mode held
    fixed. That is ``n_bootstrap`` fits per judge.

    Assumptions and scope:

    - Labels are a representative (equal-probability) sample of the rows,
      the same rows for every judge. Stratified, oversampled or targeted
      labels are not supported.
    - The multiplier concerns a policy mean (a level) corrected by labels at
      weight one, with labelled rows treated as independent. With several
      labelled rows per prompt and judge errors shared within a prompt, the
      multiplier in labelled prompts can differ from this row-level ratio;
      the intervals resample prompts but describe the row-level ratio.
    - One calibrator per judge is fitted across all policies, as
      `analyze_dataset` fits one.
    - Folds hash the cluster-id strings, so results move with the seed and
      with how ids are spelled. Pass identical ``cluster_ids`` (or None in
      both) to compare with `calibrated_mean_ci`.

    Args:
        scores_by_judge: ``{name: (n,) scores}``. Each judge keeps its own
            scale; there is no scale detection across judges. Scores must be
            finite (nothing is imputed).
        oracle_labels: (n,) outcome shared by every judge, in [0, 1]; NaN
            for unlabelled rows.
        cluster_ids: Required. (n,) prompt ids, the fold and resampling unit.
            None means independent rows, exactly as `calibrated_mean_ci` with
            ``cluster_ids=None``.
        policy_ids: Optional (n,) policy labels. R² and the variance
            components are then computed within policy; policies without
            labelled rows are excluded, with a warning.
        covariates: Optional (n, d) numeric matrix; every judge then fits
            two-stage calibration with them.
        judge_scales: Optional ``{name: (lo, hi)}`` declared score range per
            judge. Scores outside it raise ValueError. Calibration is
            invariant to increasing affine rescaling, so a declared scale
            changes no statistic and the calibrators keep the raw scale.
        reference: Reference judge; default the first key.
        n_unlabeled: Unlabelled rows per policy in a planned evaluation that
            keeps the observed labelled rows per policy. Enables the
            ``*_at_n`` fields.
        alpha: Significance level of every interval (default 0.05).
        n_folds: Cross-fitting folds (reduced when labelled clusters are
            scarce; the resolved count is in ``diagnostics["n_folds"]``).
        n_bootstrap: Bootstrap replicates (default 2000).
        seed: Seed for the folds and the bootstrap weights.

    Returns:
        JudgeComparison with ``table``, ``pairwise``, the full-sample
        calibrators and diagnostics.

    Raises:
        ValueError: On empty or misaligned inputs, non-finite scores or
            covariates, labels outside [0, 1], labels that vary within no
            policy, an unknown reference, scores outside a declared scale,
            or invalid ``alpha``, ``n_folds``, ``n_bootstrap`` or
            ``n_unlabeled``.
        TypeError: When ``n_bootstrap``, ``n_folds``, ``n_unlabeled`` or
            ``seed`` is not an integer.
        RuntimeError: When a bootstrap refit fails (naming the replicate and
            judge) or the judges' folds differ.

    Example:
        >>> import numpy as np
        >>> from cje import compare_judges
        >>> rng = np.random.default_rng(0)
        >>> q = rng.normal(size=600)
        >>> y = (rng.uniform(size=600) < 1 / (1 + np.exp(-2 * q))).astype(float)
        >>> labels = np.where(rng.uniform(size=600) < 0.5, y, np.nan)
        >>> scores = {
        ...     "noisy": q + rng.normal(0, 1.0, 600),
        ...     "sharp": 5 + 2 * (q + rng.normal(0, 0.4, 600)),
        ... }
        >>> cmp = compare_judges(scores, labels, None, n_bootstrap=50)
        >>> print(cmp.summary())  # doctest: +SKIP
    """
    _check_options(alpha, n_folds, n_bootstrap, seed, n_unlabeled)
    names, scores = _validate_judges(scores_by_judge)
    n = len(scores[names[0]])
    if reference is None:
        reference = names[0]
    elif reference not in scores:
        raise ValueError(f"reference {reference!r} is not one of the judges {names}.")
    scales = _validate_scales(judge_scales, scores)

    labels = np.asarray(oracle_labels, dtype=float)
    if labels.shape != (n,):
        raise ValueError(
            f"oracle_labels must have shape ({n},) to match the judge scores; "
            f"got {labels.shape}. Use NaN for unlabelled rows."
        )
    mask = ~np.isnan(labels)
    if not np.any(mask):
        raise ValueError(
            "No labelled rows: oracle_labels is all NaN. Comparing judges needs "
            "oracle labels on a shared, representative subset of rows."
        )
    validate_oracle_labels(labels[mask])

    cluster_codes, cluster_strings = _factorize_clusters(cluster_ids, n)
    n_clusters = int(cluster_codes.max()) + 1
    cov = _validate_covariates(covariates, n)
    policy_codes, policy_names = _factorize_policies(policy_ids, n)

    labelled_policies = sorted(set(policy_codes[mask].tolist()))
    unlabelled_policies = [
        policy_names[p] for p in range(len(policy_names)) if p not in labelled_policies
    ]
    if unlabelled_policies:
        warnings.warn(
            "Policies without labelled rows are excluded from every statistic: "
            f"{unlabelled_policies}.",
            UserWarning,
            stacklevel=2,
        )
    group_of = {p: g for g, p in enumerate(labelled_policies)}
    n_groups = len(labelled_policies)
    lab_idx = np.flatnonzero(mask)
    used_idx = np.flatnonzero(np.isin(policy_codes, labelled_policies))
    g_lab = np.asarray([group_of[p] for p in policy_codes[lab_idx]], dtype=np.int64)
    g_used = np.asarray([group_of[p] for p in policy_codes[used_idx]], dtype=np.int64)
    y_lab = labels[lab_idx]
    ones = np.ones(n)
    if _centred_ss(y_lab, ones[lab_idx], g_lab, n_groups) <= 0.0:
        raise ValueError(
            "The labelled outcomes vary within no policy, so R² is undefined. "
            "Label more rows, or rows whose outcomes differ."
        )
    n_labelled_clusters = len(np.unique(cluster_codes[lab_idx]))
    if n_labelled_clusters < _FEW_LABELLED_CLUSTERS:
        warnings.warn(
            f"Only {n_labelled_clusters} labelled prompt clusters: the "
            "out-of-fold R², the multipliers and their bootstrap intervals are "
            f"noisy below {_FEW_LABELLED_CLUSTERS}. Label more prompts before "
            "choosing a judge.",
            UserWarning,
            stacklevel=2,
        )

    # 1. Point fits: one cross-fitted calibrator per judge, on shared folds.
    cov_names = [f"cov_{j}" for j in range(cov.shape[1])] if cov is not None else None
    calibrators: Dict[str, JudgeCalibrator] = {}
    point_folds: Optional[np.ndarray] = None
    refit_modes: Dict[str, str] = {}
    point_stats = np.empty((len(names), len(_STATS)))
    for k, name in enumerate(names):
        calibrator = JudgeCalibrator(
            random_seed=seed,
            calibration_mode="two_stage" if cov is not None else "auto",
            covariate_names=cov_names,
        )
        fit = calibrator.fit_cv(
            scores[name],
            labels[mask],
            mask,
            n_folds=n_folds,
            prompt_ids=cluster_strings,
            covariates=cov,
        )
        assert fit.fold_ids is not None
        if point_folds is None:
            point_folds = fit.fold_ids
        elif not np.array_equal(fit.fold_ids, point_folds):
            raise RuntimeError(
                f"Judge {name!r} was fitted on different cross-fitting folds "
                f"from judge {names[0]!r}; the out-of-fold predictions would not "
                "be paired."
            )
        calibrators[name] = calibrator
        # The bootstrap keeps the full-sample mode, as calibrated_mean_ci's
        # bootstrap does: two-stage whenever covariates were given.
        refit_modes[name] = (
            "two_stage"
            if cov is not None
            else (
                calibrator.selected_mode
                if calibrator.selected_mode in ("monotone", "two_stage")
                else "monotone"
            )
        )
        oof = get_oof_predictions(
            calibrator, scores[name], mask, covariates=cov, oracle_fold_ids=fit.fold_ids
        )
        point_stats[k] = _judge_statistics(
            y_lab,
            oof[lab_idx],
            ones[lab_idx],
            g_lab,
            fit.calibrated_scores[used_idx],
            ones[used_idx],
            g_used,
            n_groups,
        )
    assert point_folds is not None
    resolved_folds = calibrators[names[0]].n_folds

    # 2. Paired bootstrap: one weight per prompt cluster per replicate, shared
    # by every judge, so a judge's row does not depend on the other judges.
    rng = np.random.default_rng(seed)
    boot_stats = np.empty((n_bootstrap, len(names), len(_STATS)))
    for b in range(n_bootstrap):
        weights = rng.exponential(scale=1.0, size=n_clusters)[cluster_codes]
        for k, name in enumerate(names):
            try:
                refit = JudgeCalibrator(
                    random_seed=seed,
                    calibration_mode=cast(
                        Literal["monotone", "two_stage"], refit_modes[name]
                    ),
                    covariate_names=cov_names,
                )
                fit = refit.fit_cv(
                    scores[name],
                    labels[mask],
                    mask,
                    n_folds=n_folds,
                    prompt_ids=cluster_strings,
                    covariates=cov,
                    quiet=True,
                    sample_weight=weights[mask],
                )
                assert fit.fold_ids is not None
                if not np.array_equal(fit.fold_ids, point_folds):
                    raise RuntimeError("the refit's folds differ from the point fit's")
                oof = get_oof_predictions(
                    refit,
                    scores[name],
                    mask,
                    covariates=cov,
                    oracle_fold_ids=fit.fold_ids,
                )
                boot_stats[b, k] = _judge_statistics(
                    y_lab,
                    oof[lab_idx],
                    weights[lab_idx],
                    g_lab,
                    fit.calibrated_scores[used_idx],
                    weights[used_idx],
                    g_used,
                    n_groups,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Bootstrap replicate {b} failed for judge {name!r}; no "
                    f"replicate is retried or discarded. Original error: {exc}"
                ) from exc

    # 3. Table and pairwise comparisons.
    def ci(values: np.ndarray) -> Interval:
        lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return (float(lo), float(hi))

    column = {name: i for i, name in enumerate(_STATS)}
    table = []
    for k, name in enumerate(names):
        cal = calibrators[name]
        point = point_stats[k]
        table.append(
            JudgeQuality(
                judge=name,
                n_labelled_rows=int(len(lab_idx)),
                n_labelled_clusters=int(n_labelled_clusters),
                selected_mode=str(cal.selected_mode),
                covariates_used=bool(cal.covariates_used),
                oof_rmse=float(point[column["oof_rmse"]]),
                oof_rmse_ci=ci(boot_stats[:, k, column["oof_rmse"]]),
                r2_pooled=float(point[column["r2_pooled"]]),
                r2_pooled_ci=ci(boot_stats[:, k, column["r2_pooled"]]),
                r2_within=float(point[column["r2_within"]]),
                r2_within_ci=ci(boot_stats[:, k, column["r2_within"]]),
                var_f=float(point[column["var_f"]]),
                var_residual=float(point[column["var_residual"]]),
            )
        )

    n_bar = len(lab_idx) / n_groups
    big_n = None if n_unlabeled is None else n_bar + int(n_unlabeled)
    ref = names.index(reference)
    pairwise = []
    for k, name in enumerate(names):
        if name == reference:
            continue
        point_pair = _pair_values(point_stats[k], point_stats[ref], n_bar, big_n)
        boot_pair = _pair_values(boot_stats[:, k], boot_stats[:, ref], n_bar, big_n)
        fields: Dict[str, Any] = {}
        for key, value in point_pair.items():
            if key == "label_multiplier_at_n_capped":
                fields[key] = bool(value)
            else:
                fields[key] = float(value)
                fields[f"{key}_ci"] = ci(boot_pair[key])
        pairwise.append(JudgePair(judge=name, reference=reference, **fields))

    diagnostics: Dict[str, Any] = {
        "n_rows": int(n),
        "n_rows_used": int(len(used_idx)),
        "n_clusters": int(n_clusters),
        "n_labelled_rows": int(len(lab_idx)),
        "n_labelled_clusters": int(n_labelled_clusters),
        "n_folds": int(resolved_folds),
        "seed": int(seed),
        "fold_assignment": "balanced_seeded_hash_over_labelled_clusters",
        "cluster_ids_given": cluster_ids is not None,
        "policy_ids_given": policy_ids is not None,
        "policies": [policy_names[p] for p in labelled_policies],
        "policies_without_labels": unlabelled_policies,
        "judge_scales": {name: list(scale) for name, scale in scales.items()},
        "multiplier_basis": "row_level_within_policy",
        "bootstrap": {
            "scheme": "positive_exponential_cluster_weights",
            "n_bootstrap": int(n_bootstrap),
            "refit": True,
            "refit_modes": refit_modes,
        },
        "planned": (
            None
            if big_n is None
            else {
                "n_unlabeled": int(cast(int, n_unlabeled)),
                "labelled_rows_per_policy": float(n_bar),
                "rows_per_policy": float(big_n),
            }
        ),
    }
    logger.info(
        f"compare_judges: {len(names)} judge(s), reference {reference!r}, "
        f"{len(lab_idx)} labelled rows in {n_labelled_clusters} prompt clusters, "
        f"{n_groups} polic{'y' if n_groups == 1 else 'ies'}, "
        f"{n_bootstrap} bootstrap refits per judge"
    )
    return JudgeComparison(
        table=table,
        pairwise=pairwise,
        reference=reference,
        alpha=float(alpha),
        calibrators=calibrators,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _centred_ss(
    x: np.ndarray, w: np.ndarray, groups: np.ndarray, n_groups: int
) -> float:
    """Weighted sum of squares of ``x`` about its weighted group means."""
    total = np.bincount(groups, weights=w, minlength=n_groups)
    means = np.bincount(groups, weights=w * x, minlength=n_groups) / total
    return float(np.sum(w * (x - means[groups]) ** 2))


def _judge_statistics(
    y: np.ndarray,
    oof: np.ndarray,
    w_lab: np.ndarray,
    g_lab: np.ndarray,
    f_used: np.ndarray,
    w_used: np.ndarray,
    g_used: np.ndarray,
    n_groups: int,
) -> Tuple[float, float, float, float, float]:
    """One judge's weighted statistics, in the order of ``_STATS``."""
    resid = y - oof
    weight_lab = float(np.sum(w_lab))
    single = np.zeros(len(y), dtype=np.int64)
    ss_r_within = _centred_ss(resid, w_lab, g_lab, n_groups)
    r2_within = 1.0 - ss_r_within / _centred_ss(y, w_lab, g_lab, n_groups)
    if n_groups == 1:
        r2_pooled = r2_within
    else:
        r2_pooled = 1.0 - _centred_ss(resid, w_lab, single, 1) / _centred_ss(
            y, w_lab, single, 1
        )
    return (
        float(np.sqrt(np.sum(w_lab * resid**2) / weight_lab)),
        float(r2_pooled),
        float(r2_within),
        _centred_ss(f_used, w_used, g_used, n_groups) / float(np.sum(w_used)),
        ss_r_within / weight_lab,
    )


def _pair_values(
    judge: np.ndarray, reference: np.ndarray, n_bar: float, big_n: Optional[float]
) -> Dict[str, np.ndarray]:
    """Pairwise statistics from per-judge statistics (last axis: ``_STATS``)."""
    col = {name: i for i, name in enumerate(_STATS)}
    out = {
        "r2_within_diff": judge[..., col["r2_within"]]
        - reference[..., col["r2_within"]],
        "oof_rmse_diff": judge[..., col["oof_rmse"]] - reference[..., col["oof_rmse"]],
        # (1 - R²_ref) / (1 - R²_J): both R² share the labels' within-policy SS.
        "label_multiplier": _ratio(
            reference[..., col["var_residual"]], judge[..., col["var_residual"]]
        ),
    }
    if big_n is not None:
        ratio, multiplier, capped = _finite_n(
            reference[..., col["var_f"]],
            reference[..., col["var_residual"]],
            judge[..., col["var_f"]],
            judge[..., col["var_residual"]],
            n_bar,
            big_n,
        )
        out["variance_ratio_at_n"] = ratio
        out["label_multiplier_at_n"] = multiplier
        out["label_multiplier_at_n_capped"] = capped
    return out


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise ratio, NaN where the denominator is not positive."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    safe = np.where(den > 0.0, den, 1.0)
    return np.asarray(np.where(den > 0.0, num / safe, np.nan), dtype=float)


def _finite_n(
    var_f_ref: np.ndarray,
    var_res_ref: np.ndarray,
    var_f_j: np.ndarray,
    var_res_j: np.ndarray,
    n_bar: float,
    big_n: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Variance ratio and label multiplier at ``N`` rows with ``n`` labelled.

    ``V = Var(f) / N + Var(Y - f) / n``. The reference's labels ``m`` solve
    ``Var_ref(f) / N + Var_ref(Y - f) / m = V_J``, capped at ``m = N``.
    """
    v_ref = var_f_ref / big_n + var_res_ref / n_bar
    v_j = var_f_j / big_n + var_res_j / n_bar
    room = v_j - var_f_ref / big_n
    capped = room <= var_res_ref / big_n
    safe_room = np.where(capped, 1.0, room)
    multiplier = np.where(capped, big_n / n_bar, var_res_ref / safe_room / n_bar)
    return _ratio(v_ref, v_j), np.asarray(multiplier, dtype=float), capped


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _check_int(value: Any, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    return int(value)


def _check_options(
    alpha: float, n_folds: int, n_bootstrap: int, seed: int, n_unlabeled: Optional[int]
) -> None:
    if (
        not isinstance(alpha, Real)
        or isinstance(alpha, bool)
        or not 0.0 < float(alpha) < 1.0
    ):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}.")
    if _check_int(n_folds, "n_folds") < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}.")
    if _check_int(n_bootstrap, "n_bootstrap") < 2:
        raise ValueError(f"n_bootstrap must be at least 2, got {n_bootstrap}.")
    _check_int(seed, "seed")
    if n_unlabeled is not None and _check_int(n_unlabeled, "n_unlabeled") < 0:
        raise ValueError(f"n_unlabeled must be non-negative, got {n_unlabeled}.")


def _validate_judges(
    scores_by_judge: Mapping[str, Any],
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    if not isinstance(scores_by_judge, Mapping) or len(scores_by_judge) == 0:
        raise ValueError(
            "scores_by_judge must be a non-empty mapping of judge name to (n,) "
            "scores."
        )
    names: List[str] = []
    scores: Dict[str, np.ndarray] = {}
    n: Optional[int] = None
    for name, values in scores_by_judge.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"Judge names must be non-empty strings, got {name!r}.")
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or len(array) == 0:
            raise ValueError(
                f"Scores of judge {name!r} must be a non-empty 1-D array; got "
                f"shape {array.shape}."
            )
        if n is None:
            n = len(array)
        elif len(array) != n:
            raise ValueError(
                f"Judge {name!r} has {len(array)} scores but judge {names[0]!r} "
                f"has {n}; every judge must score the same rows."
            )
        bad = int(np.sum(~np.isfinite(array)))
        if bad:
            raise ValueError(
                f"Judge {name!r} has {bad} non-finite score(s) (NaN/inf). CJE "
                "never imputes scores: restrict every judge to the rows they all "
                "scored."
            )
        names.append(name)
        scores[name] = array
    return names, scores


def _validate_scales(
    judge_scales: Optional[Mapping[str, Tuple[float, float]]],
    scores: Dict[str, np.ndarray],
) -> Dict[str, Tuple[float, float]]:
    if judge_scales is None:
        return {}
    if not isinstance(judge_scales, Mapping):
        raise ValueError("judge_scales must be a mapping of judge name to (lo, hi).")
    out: Dict[str, Tuple[float, float]] = {}
    for name, scale in judge_scales.items():
        if name not in scores:
            raise ValueError(
                f"judge_scales names {name!r}, which is not one of the judges "
                f"{list(scores)}."
            )
        try:
            lo, hi = (float(v) for v in scale)
        except (TypeError, ValueError):
            raise ValueError(
                f"judge_scales[{name!r}] must be a (lo, hi) pair of numbers, got "
                f"{scale!r}."
            ) from None
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(
                f"judge_scales[{name!r}] must satisfy lo < hi with finite values, "
                f"got ({lo:g}, {hi:g})."
            )
        outside = int(np.sum((scores[name] < lo) | (scores[name] > hi)))
        if outside:
            raise ValueError(
                f"Judge {name!r} has {outside} score(s) outside its declared scale "
                f"[{lo:g}, {hi:g}] (observed [{scores[name].min():g}, "
                f"{scores[name].max():g}])."
            )
        out[name] = (lo, hi)
    return out


def _validate_covariates(covariates: Optional[Any], n: int) -> Optional[np.ndarray]:
    if covariates is None:
        return None
    cov = np.asarray(covariates, dtype=float)
    if cov.ndim == 1:
        cov = cov.reshape(-1, 1)
    if cov.ndim != 2 or len(cov) != n:
        raise ValueError(
            f"covariates must be (n, d) with n={n}; got shape {cov.shape}."
        )
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariates contains non-finite values (NaN/inf).")
    return cov


def _factorize_policies(
    policy_ids: Optional[Any], n: int
) -> Tuple[np.ndarray, List[str]]:
    """Integer policy codes in first-appearance order, plus their names."""
    if policy_ids is None:
        return np.zeros(n, dtype=np.int64), ["all"]
    strings = [str(p) for p in np.asarray(policy_ids, dtype=object).reshape(-1)]
    if len(strings) != n:
        raise ValueError(
            f"policy_ids length ({len(strings)}) must match the judge scores ({n})."
        )
    names = list(dict.fromkeys(strings))
    code_of = {p: i for i, p in enumerate(names)}
    return np.asarray([code_of[p] for p in strings], dtype=np.int64), names


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float], digits: int = 3, sign: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def _fmt_ci(
    value: Optional[float],
    interval: Optional[Interval],
    digits: int = 3,
    sign: bool = False,
) -> str:
    if value is None:
        return "n/a"
    if interval is None:
        return _fmt(value, digits, sign)
    return (
        f"{_fmt(value, digits, sign)} [{_fmt(interval[0], digits, sign)}, "
        f"{_fmt(interval[1], digits, sign)}]"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (Integral, np.integer)):
        return int(value)
    if isinstance(value, (Real, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value
