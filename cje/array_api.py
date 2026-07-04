"""Array-first API: the single-policy calibrated-mean primitive.

This is CJE's documented bottom layer, in the spirit of ppi_py: plain numpy
arrays in (judge scores plus a partially-labeled oracle slice), a calibrated
mean with an honest confidence interval out. It wraps the exact internals
`CalibratedDirectEstimator` uses — `JudgeCalibrator.fit_cv` for judge→oracle
calibration, the cluster bootstrap with calibrator refit, and cluster-robust
SEs augmented by the oracle jackknife — and introduces no new statistics.

For multi-policy paired comparisons, use `analyze_dataset` (fresh-draw files)
or `CalibratedDirectEstimator` directly.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import numpy as np
from scipy import stats

from .calibration.judge import JudgeCalibrator
from .diagnostics.reward_boundary import boundary_card
from .diagnostics.robust_inference import (
    DirectEvalTable,
    cluster_bootstrap_direct_with_refit,
    cluster_robust_se,
    make_calibrator_factory,
)
from .diagnostics.transport import TransportDiagnostics, audit_transportability
from .estimators.base_estimator import oracle_jackknife_variance

logger = logging.getLogger(__name__)

_VALID_INFERENCE = ("auto", "bootstrap", "cluster_robust")


@dataclass
class CalibratedMeanResult:
    """Result of `calibrated_mean_ci`.

    Attributes:
        estimate: Calibrated mean reward for the sample.
        se: Standard error (bootstrap SD, or cluster-robust SE augmented
            with the oracle-jackknife calibration variance).
        ci: (lower, upper) confidence interval at the requested alpha —
            percentile bootstrap, or t-based for cluster-robust inference.
        n: Number of evaluation samples.
        n_oracle: Number of oracle-labeled samples used for calibration.
        method: Inference method actually used ("bootstrap" or "cluster_robust").
        calibrator: The fitted `JudgeCalibrator` (reusable, e.g. for
            `transport_audit` on a new sample).
        diagnostics: Dict with calibration quality, the coverage badge
            (`boundary_card`), and inference details.
    """

    estimate: float
    se: float
    ci: Tuple[float, float]
    n: int
    n_oracle: int
    method: str
    calibrator: Any
    diagnostics: Dict[str, Any]

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"Calibrated mean: {self.estimate:.4f} (SE {self.se:.4f}, "
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


def _boundary_diagnostics(
    calibrator: Any, judge: np.ndarray, rewards: np.ndarray
) -> Optional[Dict[str, Any]]:
    """Coverage badge vs the calibrator's oracle S-range (paper's REFUSE-LEVEL gate)."""
    s_range = getattr(calibrator, "oracle_s_range", None)
    r_range = getattr(calibrator, "oracle_reward_range", None)
    if s_range is None or r_range is None:
        return None
    card = boundary_card(
        S_policy=judge,
        S_oracle=np.asarray(s_range, dtype=float),
        R_policy=rewards,
        R_min=float(r_range[0]),
        R_max=float(r_range[1]),
    )
    card_dict: Dict[str, Any] = asdict(card)
    card_dict["oracle_s_range"] = [float(s_range[0]), float(s_range[1])]
    if card.status == "REFUSE-LEVEL":
        logger.warning(
            f"REFUSE-LEVEL coverage badge: {card.out_of_range:.1%} of judge "
            f"scores fall outside the oracle calibration range "
            f"[{s_range[0]:.3f}, {s_range[1]:.3f}]. Do not ship level "
            f"(absolute) claims from this estimate."
        )
    return card_dict


def _oracle_jackknife_estimates(
    calibrator: Any, judge: np.ndarray, cov: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Leave-one-oracle-fold estimates of the mean (CalibratedDirectEstimator's recipe)."""
    fold_models = calibrator.get_fold_models_for_oua()
    if not fold_models:
        return None
    jack: List[float] = []
    for fold_id in range(len(fold_models)):
        if fold_models.get(fold_id) is None:
            continue
        fold_ids = np.full(len(judge), fold_id, dtype=int)
        rewards_loo = np.clip(calibrator.predict_oof(judge, fold_ids, cov), 0.0, 1.0)
        jack.append(float(np.mean(rewards_loo)))
    if len(jack) < 2:
        return None
    return np.asarray(jack)


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

    Fits a judge→oracle calibrator on the labeled subset (cross-fitted
    `JudgeCalibrator.fit_cv`; two-stage when covariates are given, otherwise
    auto-selected between monotone and two-stage) and estimates the mean
    calibrated reward over ALL samples. Inference matches
    `CalibratedDirectEstimator`:

    - "bootstrap": cluster bootstrap with per-replicate calibrator refit
      (AIPW-style augmented estimate; percentile CI). Captures calibrator
      uncertainty and the calibration/evaluation covariance.
    - "cluster_robust": CRV1 cluster-robust SE of the plug-in mean, augmented
      with the delete-one-oracle-fold jackknife variance (t-based CI).
    - "auto": the estimator's rule — bootstrap when there are < 20 clusters or
      when calibration is coupled with evaluation. The oracle slice here always
      lives inside the evaluation sample, so "auto" resolves to bootstrap.

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
        n_folds: CV folds for the full-data calibrator and oracle jackknife.
            (The bootstrap's internal refits use the estimator's adaptive rule.)
        inference: "auto" | "bootstrap" | "cluster_robust".
        n_bootstrap: Bootstrap replicates (bootstrap path only).
        seed: Seed for fold assignment and the bootstrap.

    Returns:
        CalibratedMeanResult with estimate, se, ci, and diagnostics (including
        the coverage badge against the calibrator's oracle S-range).

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
        },
    }
    boundary = _boundary_diagnostics(calibrator, judge, rewards)
    if boundary is not None:
        diagnostics["boundary_card"] = boundary

    if resolved == "bootstrap":
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
        res = cluster_robust_se(
            data=rewards,
            cluster_ids=cluster_codes,
            statistic_fn=lambda x: float(np.mean(x)),
            influence_fn=lambda x: x - float(np.mean(x)),
            alpha=alpha,
        )
        estimate = float(res["estimate"])
        se_base = float(res["se"])
        df = int(res["df"])

        # Oracle-jackknife augmentation (skipped at 100% oracle coverage,
        # mirroring BaseCJEEstimator._apply_oua_jackknife).
        var_oracle = 0.0
        n_jack = 0
        coverage = getattr(calibrator, "oracle_coverage", None)
        if coverage is None or coverage < 1.0:
            jack = _oracle_jackknife_estimates(calibrator, judge, cov)
            if jack is not None:
                n_jack = len(jack)
                var_oracle = oracle_jackknife_variance(jack)
        fold_models = calibrator.get_fold_models_for_oua()
        if fold_models and len(fold_models) >= 2:
            df = min(df, len(fold_models) - 1)
        df = max(df, 1)

        se = float(np.sqrt(se_base**2 + var_oracle))
        t_crit = float(stats.t.ppf(1 - alpha / 2, df))
        ci = (estimate - t_crit * se, estimate + t_crit * se)
        diagnostics["cluster_robust"] = {
            "se_cluster": se_base,
            "df": df,
            "oracle_jackknife_folds": n_jack,
            "var_oracle": float(var_oracle),
            "oua_skipped_at_full_coverage": bool(
                coverage is not None and coverage >= 1.0
            ),
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
) -> TransportDiagnostics:
    """Test whether a fitted calibrator transports to a new probe sample.

    Array-first wrapper around `cje.diagnostics.transport.audit_transportability`:
    checks the unbiasedness condition E[Y - f̂(S)] = 0 on the probe and returns
    PASS/WARN/FAIL with decile residuals. Rows with NaN oracle labels are
    excluded (the audit needs labeled probes only). The decile residuals are
    display-only — gate on the pooled CI, never on per-decile values.

    Args:
        judge_scores: (m,) judge scores for the probe sample.
        oracle_labels: (m,) oracle labels for the probe; NaN rows are dropped.
        calibrator: A fitted calibrator with `.predict()` — e.g. the
            `calibrator` returned by `calibrated_mean_ci`.
        bins: Number of score-quantile bins for the residual breakdown.
        group_label: Optional label (e.g. "policy:gpt-5.6-mini").
        alpha: Significance level for the audit CI (default 0.05 → 95% CI,
            t critical values with df = n_probe - 1).

    Returns:
        TransportDiagnostics (status, delta_hat, delta_ci, decile residuals).

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
        >>> audit = transport_audit(probe_scores, probe_labels, result.calibrator)
        >>> audit.status in ("PASS", "WARN", "FAIL")
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

    mask = ~np.isnan(labels)
    n_probe = int(np.sum(mask))
    if n_probe < 2:
        raise ValueError(
            f"transport_audit needs at least 2 labeled probe samples, "
            f"got {n_probe}. Provide oracle labels for the probe."
        )

    probe_records = [
        {"judge_score": float(s), "oracle_label": float(y)}
        for s, y in zip(judge[mask], labels[mask])
    ]
    return audit_transportability(
        calibrator, probe_records, bins=bins, group_label=group_label, alpha=alpha
    )
