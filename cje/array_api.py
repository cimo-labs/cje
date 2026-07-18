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
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import numpy as np
from scipy import stats

from .calibration.judge import JudgeCalibrator
from .diagnostics.reward_boundary import boundary_card_dict
from .diagnostics.robust_inference import (
    DirectEvalTable,
    LabelDesign,
    cluster_bootstrap_direct_with_refit,
    cluster_robust_se,
    combine_cluster_and_oracle,
    compute_direct_point_estimate,
    direct_oracle_jackknife_estimates,
    get_oof_predictions,
    make_calibrator_factory,
    oracle_jackknife_variance,
)
from .diagnostics.transport import TransportDiagnostics, audit_transportability

logger = logging.getLogger(__name__)

_VALID_INFERENCE = ("auto", "bootstrap", "cluster_robust")


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
            complete oracle coverage makes calibration unnecessary; check it
            before requesting calibrator-dependent capabilities.
        diagnostics: Dict with calibration quality, the coverage badge
            (`boundary_card`), and inference details.
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Validate and canonicalize arrays. Fails loudly; never fabricates data."""
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
    labeled = labels[mask]
    if np.any((labeled < 0.0) | (labeled > 1.0)):
        raise ValueError(
            "oracle_labels must lie in [0, 1] (calibrated rewards are clipped "
            "to that range). Rescale your labels before calling."
        )

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
        df = int(res["df"])
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


def calibrated_mean_ci(
    judge_scores: Any,
    oracle_labels: Any,
    oracle_mask: Optional[Any] = None,
    *,
    cluster_ids: Optional[Any] = None,
    covariates: Optional[Any] = None,
    alpha: float = 0.05,
    n_folds: int = 5,
    inference: str = "auto",
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> CalibratedMeanResult:
    """Calibrated mean of judge scores against a partial oracle slice, with CI.

    With complete oracle coverage, estimates the oracle mean directly without
    fitting a calibrator. Otherwise fits a judge→oracle calibrator on the
    labeled subset (cross-fitted `JudgeCalibrator.fit_cv`; two-stage when
    covariates are given, otherwise auto-selected between monotone and
    two-stage) and estimates the mean calibrated reward over ALL samples.
    Inference matches
    `CalibratedDirectEstimator`:

    - "bootstrap": cluster bootstrap with per-replicate calibrator refit
      (AIPW-style augmented estimate; percentile CI). Captures calibrator
      uncertainty and the calibration/evaluation covariance.
    - "cluster_robust": CRV1 cluster-robust SE of the augmented
      pseudo-outcome mean, combined with the delete-one-oracle-fold jackknife
      variance (t-based CI).
    - "auto": the estimator's rule — bootstrap when there are < 20 clusters or
      when calibration is coupled with evaluation. Partial oracle coverage here
      is coupled and resolves to bootstrap; complete coverage needs no
      calibrator and uses cluster-robust inference once there are >=20 clusters.

    Args:
        judge_scores: (n,) raw judge scores for every evaluation sample.
        oracle_labels: (n,) oracle labels in [0, 1]; NaN for unlabeled samples.
        oracle_mask: Optional (n,) boolean mask marking labeled samples.
            Default: ``~np.isnan(oracle_labels)``. When provided, labels
            outside the mask are ignored entirely.
        cluster_ids: Optional (n,) cluster labels (e.g. prompt ids) for
            dependent draws. Default: each row is its own cluster.
        covariates: Optional (n, d) covariate matrix; triggers two-stage
            calibration (passed through to the calibrator and the bootstrap).
        alpha: Significance level for the CI (default 0.05).
        n_folds: CV folds for the full-data calibrator, bootstrap refits, and
            oracle jackknife. Fold count is reduced when cluster support is
            insufficient.
        inference: "auto" | "bootstrap" | "cluster_robust".
        n_bootstrap: Bootstrap replicates (bootstrap path only).
        seed: Seed for fold assignment and the bootstrap.

    Returns:
        CalibratedMeanResult with estimate, se, ci, and diagnostics. Partial
        coverage also includes a reusable calibrator and its score-support
        badge; complete coverage returns ``calibrator=None`` because the
        direct oracle mean does not require a calibration model.

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
    if inference not in _VALID_INFERENCE:
        raise ValueError(
            f"Invalid inference '{inference}'. Expected one of: "
            f"{', '.join(_VALID_INFERENCE)}."
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    judge, labels, mask, cov = _validate_inputs(
        judge_scores, oracle_labels, oracle_mask, covariates
    )
    n = len(judge)
    n_oracle = int(np.sum(mask))
    cluster_codes, cluster_strings = _factorize_clusters(cluster_ids, n)
    n_clusters = int(len(np.unique(cluster_codes)))

    if n_oracle == n:
        return _direct_oracle_mean_ci(
            judge,
            labels,
            cluster_codes,
            cluster_strings,
            alpha=alpha,
            inference=inference,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

    # Fit the full-data calibrator (mask semantics: full-length scores/mask,
    # compact labels — fit_cv's boolean-mask contract).
    cov_names = [f"cov_{j}" for j in range(cov.shape[1])] if cov is not None else None
    mode: str = "two_stage" if cov is not None else "auto"
    calibrator = JudgeCalibrator(
        random_seed=seed,
        calibration_mode=cast(Literal["monotone", "two_stage", "auto"], mode),
        covariate_names=cov_names,
    )
    cal_result = calibrator.fit_cv(
        judge_scores=judge,
        oracle_labels=labels[mask],
        oracle_mask=mask,
        n_folds=n_folds,
        prompt_ids=cluster_strings,
        covariates=cov,
    )
    rewards = np.clip(calibrator.predict(judge, covariates=cov), 0.0, 1.0)

    # Resolve "auto" with CalibratedDirectEstimator's rule. The oracle slice is
    # drawn from the evaluation sample itself, so calibration and evaluation
    # are always coupled here.
    if inference == "auto":
        if n_clusters < 20:
            reason = f"auto: few clusters (G={n_clusters} < 20)"
        else:
            reason = (
                "auto: calibration/evaluation coupled (oracle labels live "
                "inside the evaluation sample)"
            )
        resolved = "bootstrap"
    else:
        resolved = inference
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
        boot_mode = calibrator.selected_mode or "monotone"
        if boot_mode not in ("monotone", "two_stage"):
            boot_mode = "monotone"
        factory = make_calibrator_factory(
            mode=cast(Literal["monotone", "two_stage"], boot_mode),
            covariate_names=cov_names,
            seed=seed,
        )
        # Adaptive floor, same rule as CalibratedDirectEstimator.
        min_oracle_per_replicate = max(10, min(30, n_oracle // 3))
        boot = cluster_bootstrap_direct_with_refit(
            eval_table=table,
            calibrator_factory=factory,
            n_bootstrap=n_bootstrap,
            min_oracle_per_replicate=min_oracle_per_replicate,
            alpha=alpha,
            seed=seed,
            point_calibrator=calibrator,
            n_folds=n_folds,
        )
        estimate = float(boot["estimates"][0])
        se = float(boot["standard_errors"][0])
        ci = (float(boot["ci_lower"][0]), float(boot["ci_upper"][0]))
        diagnostics["bootstrap"] = {
            "refit_mode": boot_mode,
            "n_bootstrap_requested": n_bootstrap,
            "n_valid_replicates": int(boot["n_valid_replicates"]),
            "n_attempts": int(boot["n_attempts"]),
            "skip_rate": float(boot["skip_rate"]),
            "min_oracle_per_replicate": min_oracle_per_replicate,
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
        )
        estimate = float(point.estimates[0])
        pseudo_outcomes = point.pseudo_outcomes[0]
        influence_values = pseudo_outcomes - estimate
        res = cluster_robust_se(
            data=influence_values,
            cluster_ids=cluster_codes,
            statistic_fn=lambda x: float(np.mean(x)),
            influence_fn=lambda x: x,
            alpha=alpha,
        )
        se_base = float(res["se"])
        df = int(res["df"])

        var_oracle = 0.0
        n_jack = 0
        jackknife = direct_oracle_jackknife_estimates(
            calibrator,
            table,
            LabelDesign("representative"),
        )
        if jackknife is not None:
            jack = jackknife[:, 0]
            n_jack = len(jack)
            var_oracle = oracle_jackknife_variance(jack)
        se, df = combine_cluster_and_oracle(se_base, df, var_oracle, n_jack)
        t_crit = float(stats.t.ppf(1 - alpha / 2, df))
        ci = (estimate - t_crit * se, estimate + t_crit * se)
        diagnostics["cluster_robust"] = {
            "se_cluster": se_base,
            "df": df,
            "oracle_jackknife_folds": n_jack,
            "var_oracle": float(var_oracle),
            "oua_skipped_at_full_coverage": False,
            "point_estimator": point.diagnostics,
        }
        method = "cluster_robust"

    logger.info(
        f"calibrated_mean_ci [{method}]: {estimate:.4f} ± {se:.4f} "
        f"(CI [{ci[0]:.4f}, {ci[1]:.4f}], n={n}, n_oracle={n_oracle})"
    )
    return CalibratedMeanResult(
        estimate=estimate,
        se=se,
        ci=ci,
        n=n,
        n_oracle=n_oracle,
        method=method,
        calibrator=calibrator,
        diagnostics=diagnostics,
    )


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
