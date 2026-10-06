"""Array-first API: the single-policy calibrated-mean primitive.

This is CJE's documented bottom layer, in the spirit of ppi_py: plain numpy
arrays in (judge scores plus a partially-labeled oracle slice), a calibrated
mean and confidence interval out. It wraps the exact internals
`CalibratedDirectEstimator` uses — `JudgeCalibrator.fit_cv` for judge→oracle
calibration, the cluster bootstrap with calibrator refit, and cluster-robust
SEs augmented by the oracle jackknife — and introduces no new statistics.

For multi-policy paired comparisons, use `analyze_dataset` (fresh-draw files)
or `CalibratedDirectEstimator` directly.
"""

import logging
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast
import warnings

import numpy as np
from scipy import stats

from .calibration.flexible_calibrator import validate_oracle_labels
from .calibration.judge import CalibrationResult, JudgeCalibrator
from .data.normalization import (
    ScaledCalibrator,
    ScaleInfo,
    coerce_scale,
    validate_values_on_scale,
)
from .diagnostics.reward_boundary import boundary_card_dict
from .diagnostics.robust_inference import (
    DirectEvalTable,
    LabelDesign,
    cluster_bootstrap_direct_with_refit,
    cluster_robust_se,
    combine_cluster_and_oracle,
    compute_direct_point_estimate,
    correction_fitted_parameters,
    direct_oracle_jackknife_estimates,
    get_oof_predictions,
    labelled_cluster_crv1,
    labelled_cluster_df,
    labelled_cluster_variance,
    make_calibrator_factory,
    oracle_jackknife_variance,
)
from .diagnostics.transport import TransportDiagnostics, audit_transportability

logger = logging.getLogger(__name__)

_VALID_INFERENCE = ("auto", "bootstrap", "cluster_robust")


class _DefaultInference(str):
    """Typed marker for an omitted inference option in the public signature."""


class _DefaultInt(int):
    """Typed marker for an omitted integer option in the public signature."""


_DEFAULT_INFERENCE = _DefaultInference("cluster_robust")
_DEFAULT_N_BOOTSTRAP = _DefaultInt(2000)


@dataclass
class CalibratedMeanResult:
    """Result of `calibrated_mean_ci`.

    Attributes:
        estimate: Direct oracle mean at complete label coverage, otherwise the
            calibrated mean reward for the sample.
        se: Standard error (bootstrap SD, or cluster-robust SE augmented
            with the oracle-jackknife calibration variance).
        ci: (lower, upper) confidence interval at the requested alpha —
            percentile bootstrap, or t-based for cluster-robust inference.
        n: Number of evaluation samples.
        n_oracle: Number of oracle-labeled samples available.
        method: Inference method actually used ("bootstrap" or "cluster_robust").
        calibrator: The fitted `JudgeCalibrator` (reusable, e.g. for
            `transport_audit` on a new sample). This is explicitly None when
            complete oracle coverage makes calibration unnecessary and
            ``fit_calibrator`` was not requested; check it before requesting
            calibrator-dependent capabilities. With ``oracle_scale`` it is a
            `ScaledCalibrator` that predicts in the declared oracle units.
        diagnostics: Dict with calibration quality, the coverage badge
            (`boundary_card`), and inference details. With ``oracle_scale``
            every value is in the declared oracle units.
    """

    estimate: float
    se: float
    ci: Tuple[float, float]
    n: int
    n_oracle: int
    method: str
    calibrator: Optional[Any]
    diagnostics: Dict[str, Any]

    def summary(self) -> str:
        """One-line human-readable summary."""
        label = (
            "Oracle mean"
            if self.diagnostics.get("estimator_route") == "direct_oracle"
            else "Calibrated mean"
        )
        return (
            f"{label}: {self.estimate:.4f} (SE {self.se:.4f}, "
            f"CI [{self.ci[0]:.4f}, {self.ci[1]:.4f}], n={self.n}, "
            f"n_oracle={self.n_oracle}, {self.method})"
        )


def _factorize_clusters(
    cluster_ids: Optional[Any], n: int
) -> Tuple[np.ndarray, List[str]]:
    """Map cluster labels to sequential int codes plus per-row strings.

    Default (cluster_ids=None): each row is its own cluster, matching
    CalibratedDirectEstimator's behavior for independent prompts.
    """
    if cluster_ids is None:
        return np.arange(n, dtype=np.int64), [f"row_{i}" for i in range(n)]
    strings = [str(c) for c in np.asarray(cluster_ids, dtype=object)]
    if len(strings) != n:
        raise ValueError(
            f"cluster_ids length ({len(strings)}) must match "
            f"judge_scores length ({n})."
        )
    unique = list(dict.fromkeys(strings))  # order of first appearance
    code_of = {c: i for i, c in enumerate(unique)}
    codes = np.array([code_of[c] for c in strings], dtype=np.int64)
    return codes, strings


def _validate_inputs(
    judge_scores: Any,
    oracle_labels: Any,
    oracle_mask: Optional[Any],
    covariates: Optional[Any],
    scale: Optional[ScaleInfo] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Validate and canonicalize arrays. Fails loudly; never fabricates data.

    With a declared ``scale`` the labelled values must lie in it exactly; they
    are mapped to [0, 1] with ``scale.normalize_array`` (unlabelled entries are
    left untouched) and then pass the same unit-interval check, which also
    rejects infinite labels. Nothing is clipped.
    """
    judge = np.asarray(judge_scores, dtype=float)
    if judge.ndim != 1 or len(judge) == 0:
        raise ValueError("judge_scores must be a non-empty 1-D array.")
    if not np.all(np.isfinite(judge)):
        raise ValueError(
            "judge_scores contains non-finite values (NaN/inf). "
            "Filter or fix these rows explicitly — CJE never imputes scores."
        )
    n = len(judge)

    labels = np.asarray(oracle_labels, dtype=float)
    if labels.shape != judge.shape:
        raise ValueError(
            f"oracle_labels length ({labels.shape}) must match "
            f"judge_scores length ({judge.shape}). Use NaN (or oracle_mask) "
            f"for unlabeled samples."
        )

    if oracle_mask is None:
        mask = ~np.isnan(labels)
    else:
        mask_arr = np.asarray(oracle_mask)
        if mask_arr.dtype != np.bool_:
            raise ValueError(
                "oracle_mask must be a boolean array (True = has oracle label). "
                "Integer index arrays are not accepted here."
            )
        if mask_arr.shape != judge.shape:
            raise ValueError(
                f"oracle_mask length ({mask_arr.shape}) must match "
                f"judge_scores length ({judge.shape})."
            )
        mask = mask_arr
        if np.any(np.isnan(labels[mask])):
            n_bad = int(np.sum(np.isnan(labels[mask])))
            raise ValueError(
                f"oracle_mask selects {n_bad} sample(s) whose oracle_labels "
                f"are NaN. Every masked-in sample needs a real label."
            )

    n_oracle = int(np.sum(mask))
    if n_oracle == 0:
        raise ValueError(
            "No oracle labels available: oracle_labels is all-NaN (or "
            "oracle_mask is all-False). Calibration needs a labeled slice — "
            "provide oracle labels for at least a subset of samples."
        )
    if scale is not None:
        # Exact bounds (no tolerance), so every in-scale label maps into
        # [0, 1] without clipping; infinite labels pass this finite-only check
        # unchanged and are rejected below.
        validate_values_on_scale(
            labels[mask], scale, field_name="oracle_labels", tolerance=0.0
        )
        labels = labels.copy()
        labels[mask] = scale.normalize_array(labels[mask])
    # Same check and message as JudgeCalibrator.fit_cv.
    validate_oracle_labels(labels[mask])

    cov: Optional[np.ndarray] = None
    if covariates is not None:
        cov = np.asarray(covariates, dtype=float)
        if cov.ndim == 1:
            cov = cov.reshape(-1, 1)
        if cov.ndim != 2 or len(cov) != n:
            raise ValueError(
                f"covariates must be (n, d) with n={n}; got shape {cov.shape}."
            )
        if not np.all(np.isfinite(cov)):
            raise ValueError("covariates contains non-finite values (NaN/inf).")

    return judge, labels, mask, cov


def _direct_oracle_mean_ci(
    judge: np.ndarray,
    labels: np.ndarray,
    cluster_codes: np.ndarray,
    cluster_strings: List[str],
    *,
    alpha: float,
    inference: str,
    n_bootstrap: int,
    seed: int,
) -> CalibratedMeanResult:
    """Estimate a fully observed oracle mean without fitting a calibrator."""
    n = len(labels)
    n_clusters = int(len(np.unique(cluster_codes)))
    if inference == "auto":
        resolved = "bootstrap" if n_clusters < 20 else "cluster_robust"
        reason = (
            "auto: direct oracle mean with few clusters"
            if resolved == "bootstrap"
            else "auto: direct oracle mean with sufficient clusters"
        )
    elif inference == "cluster_robust":
        resolved = inference
        reason = "cluster_robust requested/default"
    else:
        resolved = inference
        reason = "explicitly requested"

    diagnostics: Dict[str, Any] = {
        "alpha": alpha,
        "inference_reason": reason,
        "n_clusters": n_clusters,
        "estimator_route": "direct_oracle",
        "calibration": {
            "mode": "not_required",
            "selected_mode": None,
            "n_oracle": n,
            "oracle_coverage": 1.0,
            "calibrator_available": False,
            "covariates_used": False,
            "n_folds_without_covariates": 0,
        },
    }

    if n_clusters < 2:
        estimate = float(np.mean(labels))
        se = float("nan")
        ci = (float("nan"), float("nan"))
        method = resolved
        diagnostics["inference_available"] = False
        if resolved == "bootstrap":
            diagnostics["bootstrap"] = {
                "n_bootstrap_requested": n_bootstrap,
                "n_valid_replicates": 0,
                "unavailable_reason": "fewer_than_two_independent_clusters",
                "seed": seed,
            }
        else:
            diagnostics["cluster_robust"] = {
                "se_cluster": float("nan"),
                "df": 0,
                "unavailable_reason": "fewer_than_two_independent_clusters",
                "oracle_jackknife_folds": 0,
                "var_oracle": 0.0,
                "oua_skipped_at_full_coverage": True,
            }
    elif resolved == "bootstrap":
        table = DirectEvalTable(
            prompt_ids=cluster_codes,
            prompt_id_strings=cluster_strings,
            policy_indices=np.zeros(n, dtype=np.int32),
            judge_scores=judge,
            oracle_labels=labels,
            oracle_mask=np.ones(n, dtype=bool),
            covariates=None,
            covariate_names=None,
            policy_names=["policy"],
        )
        boot = cluster_bootstrap_direct_with_refit(
            eval_table=table,
            calibrator_factory=make_calibrator_factory(mode="monotone", seed=seed),
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            seed=seed,
            use_augmented_estimator=False,
        )
        estimate = float(boot["estimates"][0])
        se = float(boot["standard_errors"][0])
        ci = (float(boot["ci_lower"][0]), float(boot["ci_upper"][0]))
        method = "bootstrap"
        diagnostics["bootstrap"] = {
            "refit_mode": None,
            "n_bootstrap_requested": n_bootstrap,
            "n_valid_replicates": int(boot["n_valid_replicates"]),
            "n_attempts": int(boot["n_attempts"]),
            "skip_rate": float(boot["skip_rate"]),
            "oracle_count_summary": boot["oracle_count_summary"],
            "seed": seed,
        }
    else:
        res = cluster_robust_se(
            data=labels,
            cluster_ids=cluster_codes,
            statistic_fn=lambda x: float(np.mean(x)),
            influence_fn=lambda x: x - float(np.mean(x)),
            alpha=alpha,
        )
        estimate = float(res["estimate"])
        se = float(res["se"])
        df = float(res["df"])
        t_crit = float(stats.t.ppf(1 - alpha / 2, df))
        ci = (estimate - t_crit * se, estimate + t_crit * se)
        method = "cluster_robust"
        diagnostics["cluster_robust"] = {
            "se_cluster": se,
            "df": df,
            "oracle_jackknife_folds": 0,
            "var_oracle": 0.0,
            "oua_skipped_at_full_coverage": True,
        }

    return CalibratedMeanResult(
        estimate=estimate,
        se=se,
        ci=ci,
        n=n,
        n_oracle=n,
        method=method,
        calibrator=None,
        diagnostics=diagnostics,
    )


def _correction_weight_summary(
    rule: str, point_diagnostics: Dict[str, Any]
) -> Dict[str, Any]:
    """The single policy's correction weight, its reason and route."""
    weights = point_diagnostics.get("correction_weights") or [float("nan")]
    reasons = point_diagnostics.get("correction_weight_reasons") or [None]
    routes = point_diagnostics.get("routes") or [None]
    if any(point_diagnostics.get("labelled_outcomes_constant") or []):
        logger.warning(
            "Every labelled outcome is identical: the residual correction and its "
            "standard error rest on no outcome variation in the labels, so treat "
            "the interval as provisional and label more rows if the outcome is rare."
        )
    return {
        "rule": rule,
        "weight": float(weights[0]),
        "reason": reasons[0],
        "route": routes[0],
    }


def _fit_judge_calibrator(
    judge: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    cov: Optional[np.ndarray],
    cluster_strings: List[str],
    *,
    n_folds: int,
    seed: int,
) -> Tuple[JudgeCalibrator, CalibrationResult, str, Optional[List[str]]]:
    """Fit the cross-fitted judge calibrator on unit-scale labels.

    Two-stage when covariates are given, otherwise auto-selected. Shared by the
    partial-coverage estimate and the descriptive complete-coverage fit, so
    both fit the same model on the same folds.
    """
    cov_names = [f"cov_{j}" for j in range(cov.shape[1])] if cov is not None else None
    mode: str = "two_stage" if cov is not None else "auto"
    calibrator = JudgeCalibrator(
        random_seed=seed,
        calibration_mode=cast(Literal["monotone", "two_stage", "auto"], mode),
        covariate_names=cov_names,
    )
    # Mask semantics: full-length scores/mask, compact labels (fit_cv's
    # boolean-mask contract).
    cal_result = calibrator.fit_cv(
        judge_scores=judge,
        oracle_labels=labels[mask],
        oracle_mask=mask,
        n_folds=n_folds,
        prompt_ids=cluster_strings,
        covariates=cov,
    )
    return calibrator, cal_result, mode, cov_names


def _oof_fit_statistics(
    calibrator: JudgeCalibrator,
    cal_result: CalibrationResult,
    judge: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    cov: Optional[np.ndarray],
) -> Dict[str, Optional[float]]:
    """Out-of-fold R^2 and correlation of the calibrator on the labelled rows.

    ``oof_r2 = 1 - sum((Y - f_oof)^2) / sum((Y - mean(Y))^2)`` (may be
    negative; None when the labels are constant) and ``oof_correlation`` is
    the Pearson correlation of ``f_oof`` and ``Y`` (None when either side is
    constant). ``f_oof`` is each row's prediction from the fold model that did
    not see its cluster. Both are invariant to an affine rescaling of the
    labels.
    """
    oof = get_oof_predictions(
        calibrator,
        judge,
        mask,
        covariates=cov,
        oracle_fold_ids=cal_result.fold_ids,
    )[mask]
    y = labels[mask]
    oof_r2: Optional[float] = None
    oof_correlation: Optional[float] = None
    # Exact-constant checks: the mean of identical floats need not equal them,
    # so a sum of squares cannot detect constant labels.
    if np.ptp(y) > 0.0:
        total = float(np.sum((y - np.mean(y)) ** 2))
        oof_r2 = 1.0 - float(np.sum((y - oof) ** 2)) / total
        if np.ptp(oof) > 0.0:
            oof_correlation = float(np.corrcoef(oof, y)[0, 1])
    return {"oof_r2": oof_r2, "oof_correlation": oof_correlation}


# Units of every numeric leaf of `CalibratedMeanResult.diagnostics` under
# `oracle_scale=(lo, hi)`. Paths are key tuples below `diagnostics`; "*" is any
# list position. Levels map to lo + span * x, spreads to span * x and
# variances to span**2 * x. Unchanged leaves are unitless (fractions, weights,
# R^2, correlations, df), counts, seeds, judge-score units (judge scores are
# never rescaled) or the scale declaration itself. A numeric leaf that is not
# listed raises instead of being reported on the wrong scale; the tests walk
# every path to keep this list complete.
_ORACLE_SCALE_LEVEL_PATHS = frozenset(
    {
        ("cluster_robust", "point_estimator", "plug_in_estimates", "*"),
    }
)
_ORACLE_SCALE_SPREAD_PATHS = frozenset(
    {
        ("calibration", "rmse"),
        ("calibration", "oof_rmse"),
        ("calibration", "coverage_tolerance"),
        ("boundary_card", "partial_id_width"),
        ("cluster_robust", "se_cluster"),
        ("cluster_robust", "se_cluster_unadjusted"),
        ("cluster_robust", "point_estimator", "residual_corrections", "*"),
    }
)
_ORACLE_SCALE_VARIANCE_PATHS = frozenset(
    {
        ("cluster_robust", "var_oracle"),
    }
)
_ORACLE_SCALE_UNCHANGED_PATHS = frozenset(
    {
        ("alpha",),
        ("n_clusters",),
        ("oracle_scale", "min"),
        ("oracle_scale", "max"),
        ("calibration", "coverage_at_01"),
        ("calibration", "oof_coverage_at_01"),
        ("calibration", "oof_r2"),
        ("calibration", "oof_correlation"),
        ("calibration", "n_oracle"),
        ("calibration", "oracle_coverage"),
        ("calibration", "n_folds"),
        ("calibration", "n_folds_without_covariates"),
        ("boundary_card", "out_of_range"),
        ("boundary_card", "saturation"),
        ("boundary_card", "oracle_s_range", "*"),
        ("correction_weight", "weight"),
        ("bootstrap", "n_bootstrap_requested"),
        ("bootstrap", "n_valid_replicates"),
        ("bootstrap", "n_attempts"),
        ("bootstrap", "skip_rate"),
        ("bootstrap", "seed"),
        ("bootstrap", "oracle_count_summary", "min"),
        ("bootstrap", "oracle_count_summary", "p10"),
        ("bootstrap", "oracle_count_summary", "median"),
        ("cluster_robust", "df"),
        ("cluster_robust", "oracle_jackknife_folds"),
        ("cluster_robust", "n_labelled_clusters"),
        ("cluster_robust", "fitted_parameters"),
        ("cluster_robust", "labelled_variance_inflation"),
        ("cluster_robust", "labelled_variance_share"),
        ("cluster_robust", "labels_coupled_fraction"),
        ("cluster_robust", "point_estimator", "oracle_fractions", "*"),
        ("cluster_robust", "point_estimator", "correction_weight_min_labels"),
        ("cluster_robust", "point_estimator", "correction_weight_min_minority"),
        ("cluster_robust", "point_estimator", "correction_weights", "*"),
        ("cluster_robust", "point_estimator", "labelled_rows", "*"),
        ("cluster_robust", "point_estimator", "labelled_clusters", "*"),
    }
)


def _is_numeric_leaf(value: Any) -> bool:
    return isinstance(value, (Integral, float, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _diagnostics_to_oracle_units(
    node: Any, scale: ScaleInfo, path: Tuple[str, ...] = ()
) -> Any:
    """Copy of a diagnostics tree with each numeric leaf in oracle units."""
    if isinstance(node, dict):
        return {
            key: _diagnostics_to_oracle_units(value, scale, path + (str(key),))
            for key, value in node.items()
        }
    if isinstance(node, (list, tuple)):
        mapped = [_diagnostics_to_oracle_units(v, scale, path + ("*",)) for v in node]
        return tuple(mapped) if isinstance(node, tuple) else mapped
    if not _is_numeric_leaf(node):
        return node
    if path in _ORACLE_SCALE_LEVEL_PATHS:
        return float(scale.inverse(float(node)))
    if path in _ORACLE_SCALE_SPREAD_PATHS:
        return float(node) * scale.span
    if path in _ORACLE_SCALE_VARIANCE_PATHS:
        return float(node) * scale.span**2
    if path in _ORACLE_SCALE_UNCHANGED_PATHS:
        return node
    raise RuntimeError(
        f"calibrated_mean_ci has no oracle_scale unit for diagnostics path "
        f"{'/'.join(path)!r}; refusing to report it on the wrong scale. "
        "Please report this as a CJE bug."
    )


def _result_to_oracle_units(
    result: CalibratedMeanResult, scale: ScaleInfo
) -> CalibratedMeanResult:
    """Map a unit-scale result, its diagnostics and calibrator to oracle units."""
    diagnostics = dict(result.diagnostics)
    calibration = diagnostics.get("calibration")
    if isinstance(calibration, dict) and "coverage_at_01" in calibration:
        # coverage_at_01 counts |prediction - label| <= 0.1 on the unit scale.
        diagnostics["calibration"] = {**calibration, "coverage_tolerance": 0.1}
    diagnostics = _diagnostics_to_oracle_units(diagnostics, scale)
    diagnostics["oracle_scale"] = scale.to_dict()
    calibrator = result.calibrator
    if calibrator is not None:
        calibrator = ScaledCalibrator(calibrator, judge_scale=None, output_scale=scale)
    return CalibratedMeanResult(
        estimate=float(scale.inverse(result.estimate)),
        se=float(result.se) * scale.span,
        ci=(
            float(scale.inverse(result.ci[0])),
            float(scale.inverse(result.ci[1])),
        ),
        n=result.n,
        n_oracle=result.n_oracle,
        method=result.method,
        calibrator=calibrator,
        diagnostics=diagnostics,
    )


def calibrated_mean_ci(
    judge_scores: Any,
    oracle_labels: Any,
    oracle_mask: Optional[Any] = None,
    *,
    cluster_ids: Optional[Any] = None,
    covariates: Optional[Any] = None,
    alpha: float = 0.05,
    n_folds: int = 5,
    inference: str = _DEFAULT_INFERENCE,
    n_bootstrap: int = _DEFAULT_N_BOOTSTRAP,
    seed: int = 42,
    correction_weight: str = "one",
    oracle_scale: Optional[Tuple[float, float]] = None,
    fit_calibrator: Union[bool, np.bool_] = False,
) -> CalibratedMeanResult:
    """Calibrated mean of judge scores against a partial oracle slice, with CI.

    With complete oracle coverage, estimates the oracle mean directly without
    fitting a calibrator (``fit_calibrator=True`` also fits a descriptive one
    that never enters the estimate). Otherwise fits a judge→oracle calibrator
    on the labeled subset (cross-fitted `JudgeCalibrator.fit_cv`; two-stage when
    covariates are given, otherwise auto-selected between monotone and
    two-stage) and estimates the mean calibrated reward over ALL samples.
    Inference matches
    `CalibratedDirectEstimator`:

    - "cluster_robust": CRV1 cluster-robust SE of the augmented
      pseudo-outcome mean, combined with the delete-one-oracle-fold jackknife
      variance (t-based CI; default). The residual correction is a mean over
      the labelled clusters that fits ``q`` parameters on them (the mean
      residual, plus the slope for the tuned weight), so the labelled
      clusters' share of the CRV1 variance is scaled by ``n_L / (n_L - q)``
      and the interval takes ``n_L - q`` degrees of freedom, capped by the
      approximate Welch--Satterthwaite df with the jackknife's ``K - 1`` and
      by the unadjusted interval's Welch df, so it never narrows
      (``diagnostics["cluster_robust"]["df_method"] == "labelled_clusters"``;
      ``se_cluster`` is the adjusted sampling SE and ``se_cluster_unadjusted``
      the plain CRV1 SE).
    - "bootstrap": cluster bootstrap with per-replicate calibrator refit
      (AIPW-style augmented estimate; percentile CI). Captures calibrator
      uncertainty and the calibration/evaluation covariance.
    - "auto": the estimator's rule — bootstrap when there are < 20 clusters or
      when calibration is coupled with evaluation. Partial oracle coverage here
      is coupled and resolves to bootstrap; complete coverage needs no
      calibrator and uses cluster-robust inference once there are >=20 clusters.

    Representative labels are assumed: the labelled rows must be an
    equal-probability sample of the n rows, because the calibrator is fitted
    on them and the residual correction averages over them unweighted.
    Stratified or oversampled labels (equal quotas per bucket, rare buckets
    oversampled) bias the estimate, and nothing here undoes that tilt; there
    is no strata or weights argument. For a stratified design defined before
    labelling, call this function once per stratum h on that stratum's rows
    and combine with the population shares W_h = N_h / N: estimate
    ``sum_h W_h * mu_h`` and standard error ``sqrt(sum_h W_h**2 * SE_h**2)``,
    with a normal interval. Each stratum then needs its own labelled slice
    (at least four labelled clusters). For known unequal inclusion
    probabilities, use ``analyze_dataset(label_design="known_propensity")``.

    Args:
        judge_scores: (n,) raw judge scores for every evaluation sample.
        oracle_labels: (n,) oracle labels in [0, 1], or in ``oracle_scale``
            when one is declared; NaN for unlabeled samples. Labels outside
            that range, or infinite, raise ValueError.
        oracle_mask: Optional (n,) boolean mask marking labeled samples.
            Default: ``~np.isnan(oracle_labels)``. When provided, labels
            outside the mask are ignored entirely.
        cluster_ids: Optional (n,) cluster labels (e.g. prompt ids) for
            dependent draws. Default: each row is its own cluster.
        covariates: Optional (n, d) covariate matrix; triggers two-stage
            calibration (passed through to the calibrator and the bootstrap).
            Two-stage needs at least 20 labelled rows to use them. Below that
            the calibrator falls back to judge-score-only monotone calibration
            (``selected_mode`` "monotone"), and cross-fitting folds whose
            training complement has fewer than 20 labelled rows ignore them
            (about 25 labels with 5 folds avoids both). Either case warns and
            is reported in ``diagnostics["calibration"]["covariates_used"]``
            and ``["n_folds_without_covariates"]``. With every row labelled
            no calibrator is fitted and the covariates are ignored, with a
            warning, unless ``fit_calibrator=True``.
        alpha: Significance level for the CI (default 0.05).
        n_folds: CV folds for the full-data calibrator, bootstrap refits, and
            oracle jackknife. Fold count is reduced when cluster support is
            insufficient.
        inference: "cluster_robust" (default) | "bootstrap" | "auto".
        n_bootstrap: Bootstrap replicates (default 2000 on the bootstrap
            path). Supplying this without ``inference`` selects bootstrap for
            backward compatibility and emits a warning.
        seed: Seed for fold assignment and the bootstrap.
        correction_weight: Weight on the calibrated prediction inside the
            residual correction: ``"one"`` (default; the plain augmented
            estimator, the 0.8.x behaviour) or ``"tuned"`` (opt-in; the
            power-tuned PPI++ weight, estimated from the labeled rows and
            clipped to [0, 1]; falls back to one below 20 labeled prompts, when
            fewer than 5 labeled prompts differ from the most common outcome,
            or when the predictions are constant). Reported under
            ``diagnostics["correction_weight"]`` on every inference path
            (weight NaN and reason None when every row is labelled).
        oracle_scale: Optional ``(lo, hi)`` declaring the bounded scale of
            ``oracle_labels`` (default None: labels must be in [0, 1]).
            Labelled values must lie in ``[lo, hi]``; they are mapped to
            ``(y - lo) / (hi - lo)`` internally and nothing is clipped.
            ``estimate`` and ``ci`` come back as ``lo + (hi - lo) * value``
            and ``se`` times ``hi - lo``; every diagnostic is mapped by its
            units (levels like the estimate, spreads such as RMSE and SEs
            times ``hi - lo``, ``var_oracle`` times ``(hi - lo) ** 2``;
            fractions, weights, R^2, correlations, df, counts and judge-score
            ranges unchanged), and ``diagnostics["oracle_scale"]`` records the
            declaration. ``coverage_at_01`` then counts predictions within
            ``diagnostics["calibration"]["coverage_tolerance"]`` (``0.1 * (hi
            - lo)``). Any returned calibrator is a `ScaledCalibrator` whose
            ``predict``/``predict_oof`` return these units, so a
            `transport_audit` with it takes probe labels and ``delta_max`` in
            them too. Judge scores are never rescaled.
        fit_calibrator: With every row labelled, also fit the cross-fitted
            calibrator (two-stage when covariates are given, otherwise
            auto-selected, as at partial coverage) and return it in
            ``result.calibrator`` for reuse, e.g. in `transport_audit`. It is
            descriptive only: ``estimate``, ``se``, ``ci``, ``method`` and
            ``diagnostics["estimator_route"] == "direct_oracle"`` are those of
            the default call, no boundary card is attached, and the
            ignored-covariates warning is not issued.
            ``diagnostics["calibration"]`` reports ``rmse``, ``oof_rmse``,
            ``coverage_at_01``, ``oof_coverage_at_01``, ``oof_r2`` (``1 -
            SSE/SST`` of the out-of-fold predictions, may be negative) and
            ``oof_correlation``, with ``used_for_estimate`` False. They
            describe how well judge (plus covariates) predict the label on
            this sample; they are not a label-savings estimate for another
            sample or policy. Needs at least four independent clusters
            (ValueError naming ``fit_calibrator`` otherwise). No effect at
            partial coverage, where the calibrator is always fitted.

    Returns:
        CalibratedMeanResult with estimate, se, ci, and diagnostics. Partial
        coverage also includes a reusable calibrator and its score-support
        badge; complete coverage returns ``calibrator=None`` because the
        direct oracle mean does not require a calibration model, unless
        ``fit_calibrator=True``.

    Example (matches the README's array-API section):
        >>> import numpy as np
        >>> from cje import calibrated_mean_ci
        >>> rng = np.random.default_rng(0)
        >>> scores = rng.uniform(size=400)
        >>> labels = np.full(400, np.nan)
        >>> labeled = rng.choice(400, size=100, replace=False)
        >>> labels[labeled] = np.clip(
        ...     scores[labeled] + rng.normal(0, 0.1, size=100), 0, 1
        ... )
        >>> result = calibrated_mean_ci(scores, labels)
        >>> print(result.summary())  # doctest: +SKIP
    """
    if correction_weight not in ("one", "tuned"):
        raise ValueError(
            f"correction_weight must be 'one' or 'tuned', got {correction_weight!r}"
        )
    inference_explicit = not isinstance(inference, _DefaultInference)
    n_bootstrap_explicit = not isinstance(n_bootstrap, _DefaultInt)
    if not inference_explicit and n_bootstrap_explicit:
        warnings.warn(
            "n_bootstrap was supplied without inference; selecting "
            "inference='bootstrap' for backward compatibility. Set inference "
            "explicitly to silence this warning.",
            UserWarning,
            stacklevel=2,
        )
        inference = "bootstrap"
    resolved_inference = (
        "cluster_robust" if isinstance(inference, _DefaultInference) else inference
    )
    if resolved_inference not in _VALID_INFERENCE:
        raise ValueError(
            f"Invalid inference '{resolved_inference}'. Expected one of: "
            f"{', '.join(_VALID_INFERENCE)}."
        )
    if resolved_inference == "cluster_robust" and n_bootstrap_explicit:
        warnings.warn(
            "n_bootstrap does not apply to inference='cluster_robust' and "
            "will be ignored.",
            UserWarning,
            stacklevel=2,
        )
        resolved_n_bootstrap = 2000
    else:
        resolved_n_bootstrap = (
            2000 if isinstance(n_bootstrap, _DefaultInt) else n_bootstrap
        )
        if not isinstance(resolved_n_bootstrap, Integral) or isinstance(
            resolved_n_bootstrap, bool
        ):
            raise TypeError("n_bootstrap must be an integer")
        resolved_n_bootstrap = int(resolved_n_bootstrap)
        if resolved_n_bootstrap < 2:
            raise ValueError("n_bootstrap must be at least 2")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if not isinstance(fit_calibrator, (bool, np.bool_)):
        raise TypeError(
            f"fit_calibrator must be True or False, got {fit_calibrator!r}."
        )
    fit_calibrator = bool(fit_calibrator)
    scale = coerce_scale(oracle_scale, field_name="oracle_scale")

    judge, labels, mask, cov = _validate_inputs(
        judge_scores, oracle_labels, oracle_mask, covariates, scale=scale
    )
    n = len(judge)
    n_oracle = int(np.sum(mask))
    cluster_codes, cluster_strings = _factorize_clusters(cluster_ids, n)
    n_clusters = int(len(np.unique(cluster_codes)))

    if n_oracle == n:
        if cov is not None and not fit_calibrator:
            warnings.warn(
                "covariates were passed but every row is labelled, so "
                "calibrated_mean_ci returns the direct outcome mean, fits no "
                "calibrator and ignores the covariates.",
                UserWarning,
                stacklevel=2,
            )
        full = _direct_oracle_mean_ci(
            judge,
            labels,
            cluster_codes,
            cluster_strings,
            alpha=alpha,
            inference=resolved_inference,
            n_bootstrap=resolved_n_bootstrap,
            seed=seed,
        )
        # Every row is labelled, so no residual correction and no weight apply.
        full.diagnostics["correction_weight"] = _correction_weight_summary(
            correction_weight,
            {
                "routes": ["direct_oracle"],
                "correction_weights": [float("nan")],
                "correction_weight_reasons": [None],
            },
        )
        if fit_calibrator:
            # Descriptive only, fitted after the direct estimate is final: it
            # never enters estimate, se, ci or route, and no boundary card is
            # attached because nothing here depends on calibrated rewards.
            try:
                full_calibrator, full_fit, full_mode, _ = _fit_judge_calibrator(
                    judge,
                    labels,
                    mask,
                    cov,
                    cluster_strings,
                    n_folds=n_folds,
                    seed=seed,
                )
            except ValueError as exc:
                raise ValueError(
                    f"fit_calibrator=True could not fit a calibrator: {exc}"
                ) from exc
            full.calibrator = full_calibrator
            full.diagnostics["calibration"] = {
                "mode": full_mode,
                "selected_mode": full_calibrator.selected_mode,
                "rmse": float(full_fit.calibration_rmse),
                "oof_rmse": full_fit.oof_rmse,
                "coverage_at_01": float(full_fit.coverage_at_01),
                "oof_coverage_at_01": full_fit.oof_coverage_at_01,
                **_oof_fit_statistics(
                    full_calibrator, full_fit, judge, labels, mask, cov
                ),
                "n_oracle": n,
                "oracle_coverage": 1.0,
                "n_folds": int(full_calibrator.n_folds),
                "calibrator_available": True,
                "used_for_estimate": False,
                "covariates_used": bool(full_calibrator.covariates_used),
                "n_folds_without_covariates": int(
                    full_calibrator.n_folds_without_covariates or 0
                ),
            }
        if scale is not None:
            full = _result_to_oracle_units(full, scale)
        return full

    calibrator, cal_result, mode, cov_names = _fit_judge_calibrator(
        judge, labels, mask, cov, cluster_strings, n_folds=n_folds, seed=seed
    )
    rewards = np.clip(calibrator.predict(judge, covariates=cov), 0.0, 1.0)

    # Resolve "auto" with CalibratedDirectEstimator's rule. The oracle slice is
    # drawn from the evaluation sample itself, so calibration and evaluation
    # are always coupled here.
    if resolved_inference == "auto":
        if n_clusters < 20:
            reason = f"auto: few clusters (G={n_clusters} < 20)"
        else:
            reason = (
                "auto: calibration/evaluation coupled (oracle labels live "
                "inside the evaluation sample)"
            )
        resolved = "bootstrap"
    elif resolved_inference == "cluster_robust":
        resolved = resolved_inference
        reason = "cluster_robust requested/default"
    else:
        resolved = resolved_inference
        reason = "explicitly requested"

    diagnostics: Dict[str, Any] = {
        "alpha": alpha,
        "inference_reason": reason,
        "n_clusters": n_clusters,
        "calibration": {
            "mode": mode,
            "selected_mode": calibrator.selected_mode,
            "rmse": float(cal_result.calibration_rmse),
            "oof_rmse": cal_result.oof_rmse,
            "coverage_at_01": float(cal_result.coverage_at_01),
            "n_oracle": n_oracle,
            "oracle_coverage": calibrator.oracle_coverage,
            "calibrator_available": True,
            "covariates_used": bool(calibrator.covariates_used),
            "n_folds_without_covariates": int(
                calibrator.n_folds_without_covariates or 0
            ),
        },
    }
    boundary = boundary_card_dict(calibrator, judge, rewards)
    if boundary is not None:
        diagnostics["boundary_card"] = boundary

    labels_nan = np.where(mask, labels, np.nan)
    table = DirectEvalTable(
        prompt_ids=cluster_codes,
        prompt_id_strings=cluster_strings,
        policy_indices=np.zeros(n, dtype=np.int32),
        judge_scores=judge,
        oracle_labels=labels_nan,
        oracle_mask=mask,
        covariates=cov,
        covariate_names=cov_names,
        policy_names=["policy"],
    )

    if resolved == "bootstrap":
        # Fix the bootstrap refit mode to the full-data selection (never "auto").
        # With covariates the requested fit is two-stage even when too few
        # labelled rows made the full-data fit fall back to monotone: a
        # monotone refit cannot accept covariates, and each replicate falls
        # back in the same way.
        if cov is not None:
            boot_mode = "two_stage"
        else:
            boot_mode = calibrator.selected_mode or "monotone"
        if boot_mode not in ("monotone", "two_stage"):
            boot_mode = "monotone"
        factory = make_calibrator_factory(
            mode=cast(Literal["monotone", "two_stage"], boot_mode),
            covariate_names=cov_names,
            seed=seed,
        )
        boot = cluster_bootstrap_direct_with_refit(
            eval_table=table,
            calibrator_factory=factory,
            n_bootstrap=resolved_n_bootstrap,
            alpha=alpha,
            seed=seed,
            point_calibrator=calibrator,
            n_folds=n_folds,
            correction_weight=correction_weight,
        )
        estimate = float(boot["estimates"][0])
        se = float(boot["standard_errors"][0])
        ci = (float(boot["ci_lower"][0]), float(boot["ci_upper"][0]))
        boot_point = boot.get("augmentation_diagnostics") or {}
        diagnostics["correction_weight"] = _correction_weight_summary(
            correction_weight, boot_point
        )
        diagnostics["bootstrap"] = {
            "refit_mode": boot_mode,
            "n_bootstrap_requested": resolved_n_bootstrap,
            "n_valid_replicates": int(boot["n_valid_replicates"]),
            "n_attempts": int(boot["n_attempts"]),
            "skip_rate": float(boot["skip_rate"]),
            "oracle_count_summary": boot["oracle_count_summary"],
            "seed": seed,
        }
        method = "bootstrap"
    else:
        residual_predictions = get_oof_predictions(
            calibrator,
            judge,
            mask,
            covariates=cov,
            oracle_fold_ids=cal_result.fold_ids,
        )
        residual_predictions[~mask] = rewards[~mask]
        point = compute_direct_point_estimate(
            rewards,
            table,
            residual_predictions,
            LabelDesign("representative"),
            correction_weight=correction_weight,
        )
        estimate = float(point.estimates[0])
        pseudo_outcomes = point.pseudo_outcomes[0]
        diagnostics["correction_weight"] = _correction_weight_summary(
            correction_weight, point.diagnostics
        )
        influence_values = pseudo_outcomes - estimate
        res = cluster_robust_se(
            data=influence_values,
            cluster_ids=cluster_codes,
            statistic_fn=lambda x: float(np.mean(x)),
            influence_fn=lambda x: x,
            alpha=alpha,
        )
        se_crv1 = float(res["se"])
        se_base = se_crv1
        df_crv1 = float(res["df"])
        df = df_crv1

        # The residual correction is a mean over the labelled clusters, which
        # fits q parameters on them: inflate the labelled clusters' CRV1 part
        # by n_L / (n_L - q) and take n_L - q degrees of freedom (issue #60).
        # The labels here also fitted the calibrator (always coupled).
        labelled: Dict[str, Any] = {}
        df_labelled: Optional[int] = None
        if point.diagnostics["routes"][0] == "augmented":
            q = correction_fitted_parameters(
                point.diagnostics["correction_weight_reasons"][0]
            )
            split = labelled_cluster_crv1(influence_values, cluster_codes, mask)
            n_labelled = int(split["n_labelled_clusters"])
            var_sampling, df_labelled = labelled_cluster_variance(
                split["v_labelled"], split["v_unlabelled"], n_labelled, q
            )
            crv1 = split["v_labelled"] + split["v_unlabelled"]
            labelled = {
                "df_method": "labelled_clusters",
                "n_labelled_clusters": n_labelled,
                "fitted_parameters": q,
                "labelled_variance_inflation": (
                    n_labelled / df_labelled
                    if df_labelled is not None
                    else float("nan")
                ),
                "labelled_variance_share": (
                    float(split["v_labelled"] / crv1) if crv1 > 0 else float("nan")
                ),
                "labels_coupled": True,
                "labels_coupled_fraction": 1.0,
            }
            if df_labelled is not None:
                se_base = float(np.sqrt(var_sampling))
                df = float(df_labelled)

        if labelled and df_labelled is None:
            # Defensive: fit_cv needs at least four labelled clusters and the
            # tuned slope at least 20, so n_L - q >= 3 here.
            logger.warning(
                f"{labelled['n_labelled_clusters']} labelled cluster(s) leave no "
                "degrees of freedom for a residual correction that fits "
                f"{labelled['fitted_parameters']} parameter(s); returning the "
                "point estimate with SE unavailable."
            )
            se = float("nan")
            ci = (float("nan"), float("nan"))
            diagnostics["inference_available"] = False
            diagnostics["cluster_robust"] = {
                "se_cluster": float("nan"),
                "se_cluster_unadjusted": se_crv1,
                "df": 0,
                "unavailable_reason": "too_few_labelled_clusters",
                **labelled,
                "oracle_jackknife_folds": 0,
                "var_oracle": 0.0,
                "oua_skipped_at_full_coverage": False,
                "point_estimator": point.diagnostics,
            }
        else:
            var_oracle = 0.0
            n_jack = 0
            jackknife = direct_oracle_jackknife_estimates(
                calibrator,
                table,
                LabelDesign("representative"),
                correction_weight=correction_weight,
            )
            if jackknife is not None:
                jack = jackknife[:, 0]
                n_jack = len(jack)
                var_oracle = oracle_jackknife_variance(jack)
            se, df_welch = combine_cluster_and_oracle(se_base, df, var_oracle, n_jack)
            df_cap_applied = False
            if labelled:
                df, df_cap_applied = labelled_cluster_df(
                    se_base,
                    df,
                    var_oracle,
                    n_jack,
                    se_unadjusted=se_crv1,
                    df_unadjusted=df_crv1,
                )
            else:
                df = df_welch
            t_crit = float(stats.t.ppf(1 - alpha / 2, df))
            ci = (estimate - t_crit * se, estimate + t_crit * se)
            diagnostics["cluster_robust"] = {
                "se_cluster": se_base,
                "df": df,
                "df_method": ("welch_satterthwaite" if var_oracle > 0.0 else "cluster"),
                "oracle_jackknife_folds": n_jack,
                "var_oracle": float(var_oracle),
                "oua_skipped_at_full_coverage": False,
                "point_estimator": point.diagnostics,
            }
            if labelled:
                diagnostics["cluster_robust"].update(
                    {
                        "se_cluster_unadjusted": se_crv1,
                        **labelled,
                        "oracle_df_cap_applied": bool(df_cap_applied),
                    }
                )
        method = "cluster_robust"

    result = CalibratedMeanResult(
        estimate=estimate,
        se=se,
        ci=ci,
        n=n,
        n_oracle=n_oracle,
        method=method,
        calibrator=calibrator,
        diagnostics=diagnostics,
    )
    if scale is not None:
        result = _result_to_oracle_units(result, scale)
    logger.info(
        f"calibrated_mean_ci [{method}]: {result.estimate:.4f} ± {result.se:.4f} "
        f"(CI [{result.ci[0]:.4f}, {result.ci[1]:.4f}], n={n}, n_oracle={n_oracle})"
    )
    return result


def transport_audit(
    judge_scores: Any,
    oracle_labels: Any,
    calibrator: Any,
    *,
    bins: int = 10,
    group_label: Optional[str] = None,
    alpha: float = 0.05,
    delta_max: Optional[float] = None,
    cluster_ids: Optional[Any] = None,
    sample_weights: Optional[Any] = None,
    covariates: Optional[Any] = None,
    family_size: int = 1,
    min_effective_clusters: float = 20.0,
) -> TransportDiagnostics:
    """Audit mean residual transport on a held-out oracle probe sample.

    Array-first wrapper around `cje.diagnostics.transport.audit_transportability`:
    estimates E[Y - f_hat(S, X)] with prompt-clustered uncertainty. Rows with
    NaN oracle labels are excluded. With an explicit ``delta_max``, PASS means
    the entire simultaneous CI is inside ``[-delta_max, +delta_max]``; FAIL
    means the CI is disjoint; overlap is INCONCLUSIVE. Without a margin the
    result is NOT_GRADED. Score-bin residuals are display-only.

    Args:
        judge_scores: (m,) judge scores for the probe sample.
        oracle_labels: (m,) oracle labels for the probe; NaN rows are dropped.
        calibrator: A fitted calibrator with `.predict()` — e.g. the
            `calibrator` returned by `calibrated_mean_ci`.
        bins: Number of score-quantile bins for the residual breakdown.
        group_label: Optional label (e.g. "policy:gpt-5.6-mini").
        alpha: Significance level for the audit CI (default 0.05 → 95% CI,
            Bonferroni-adjusted across ``family_size`` audits).
        delta_max: Practical absolute mean-residual margin in oracle units.
        cluster_ids: Prompt/independence-cluster ID per row. Defaults to one
            independent cluster per row.
        sample_weights: Optional positive analysis weights.
        covariates: Optional probe covariate matrix passed to the calibrator.
        family_size: Number of policy/group audits in the decision family.
        min_effective_clusters: Minimum effective clusters needed for grading.

    Returns:
        Structured residual-audit diagnostics.

    Example:
        >>> import numpy as np
        >>> from cje import calibrated_mean_ci, transport_audit
        >>> rng = np.random.default_rng(1)
        >>> scores = rng.uniform(size=400)
        >>> labels = np.where(
        ...     rng.uniform(size=400) < 0.3,
        ...     np.clip(scores + rng.normal(0, 0.1, size=400), 0, 1),
        ...     np.nan,
        ... )
        >>> result = calibrated_mean_ci(scores, labels, inference="cluster_robust")
        >>> probe_scores = rng.uniform(size=200)
        >>> probe_labels = np.clip(probe_scores + rng.normal(0, 0.1, 200), 0, 1)
        >>> audit = transport_audit(
        ...     probe_scores,
        ...     probe_labels,
        ...     result.calibrator,
        ...     delta_max=0.05,
        ...     cluster_ids=np.arange(200),
        ... )
        >>> audit.status in ("PASS", "FAIL", "INCONCLUSIVE")
        True
    """
    judge = np.asarray(judge_scores, dtype=float)
    labels = np.asarray(oracle_labels, dtype=float)
    if judge.ndim != 1 or len(judge) == 0:
        raise ValueError("judge_scores must be a non-empty 1-D array.")
    if labels.shape != judge.shape:
        raise ValueError(
            f"oracle_labels length ({labels.shape}) must match "
            f"judge_scores length ({judge.shape})."
        )
    if not np.all(np.isfinite(judge)):
        raise ValueError("judge_scores contains non-finite values (NaN/inf).")

    if np.any(np.isinf(labels)):
        raise ValueError("oracle_labels contains infinity.")
    mask = ~np.isnan(labels)
    n_probe = int(np.sum(mask))
    if n_probe < 1:
        raise ValueError(
            f"transport_audit needs at least 1 labeled probe sample, "
            f"got {n_probe}. Provide oracle labels for the probe."
        )

    def _masked_optional(values: Optional[Any], name: str) -> Optional[np.ndarray]:
        if values is None:
            return None
        array = np.asarray(values)
        if array.shape[0] != len(judge):
            raise ValueError(
                f"{name} length ({array.shape[0]}) must match judge_scores "
                f"length ({len(judge)})."
            )
        return cast(np.ndarray, array[mask])

    masked_clusters = _masked_optional(cluster_ids, "cluster_ids")
    masked_weights = _masked_optional(sample_weights, "sample_weights")
    masked_covariates = _masked_optional(covariates, "covariates")

    probe_records = [
        {
            "prompt_id": str(i),
            "judge_score": float(s),
            "oracle_label": float(y),
        }
        for i, (s, y) in enumerate(zip(judge[mask], labels[mask]))
    ]
    return audit_transportability(
        calibrator,
        probe_records,
        bins=bins,
        group_label=group_label,
        alpha=alpha,
        delta_max=delta_max,
        cluster_ids=masked_clusters,
        sample_weights=masked_weights,
        covariates=masked_covariates,
        family_size=family_size,
        min_effective_clusters=min_effective_clusters,
    )
