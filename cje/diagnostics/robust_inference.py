"""
Robust inference utilities for Direct-mode evaluation.

Cluster bootstrap with calibrator refit, cluster-robust standard errors,
and paired policy comparisons.
"""

import numpy as np
from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Any,
    Callable,
    TYPE_CHECKING,
    Literal,
)
from dataclasses import dataclass, field
from scipy import stats
import logging

if TYPE_CHECKING:
    from ..data.fresh_draws import FreshDrawDataset

logger = logging.getLogger(__name__)


EvaluationKey = Tuple[str, str, int]


@dataclass
class CalibrationProvenance:
    """Exact rows used to fit a Direct-mode reward calibrator.

    ``row_roles`` distinguishes calibration observations sampled in an
    external frame from oracle labels attached to evaluation rows.  Evaluation
    rows carry an exact ``(policy, prompt_id, draw_idx)`` key so their
    calibration and evaluation weights remain coupled in bootstrap worlds.
    Judge scores may use any finite raw scale; oracle labels remain in [0, 1].
    """

    judge_scores: np.ndarray
    oracle_labels: np.ndarray
    prompt_ids: List[str]
    covariates: Optional[np.ndarray] = None
    row_roles: Optional[List[Literal["external", "evaluation"]]] = None
    evaluation_keys: Optional[List[Optional[EvaluationKey]]] = None
    sample_weights: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.judge_scores = np.asarray(self.judge_scores, dtype=float).copy()
        self.oracle_labels = np.asarray(self.oracle_labels, dtype=float).copy()
        self.prompt_ids = list(self.prompt_ids)
        n = len(self.judge_scores)
        if self.judge_scores.shape != (n,) or self.oracle_labels.shape != (n,):
            raise ValueError("Calibration scores and labels must be 1-D and aligned")
        if len(self.prompt_ids) != n:
            raise ValueError(
                f"Calibration prompt_ids length ({len(self.prompt_ids)}) must "
                f"match rows ({n})"
            )
        if n == 0:
            raise ValueError("Calibration provenance cannot be empty")
        if not np.all(np.isfinite(self.judge_scores)) or not np.all(
            np.isfinite(self.oracle_labels)
        ):
            raise ValueError("Calibration scores and labels must be finite")
        if np.any((self.oracle_labels < 0) | (self.oracle_labels > 1)):
            raise ValueError("Calibration provenance oracle_labels must lie in [0, 1]")
        if self.covariates is not None:
            self.covariates = np.asarray(self.covariates, dtype=float).copy()
            if self.covariates.ndim != 2 or len(self.covariates) != n:
                raise ValueError(
                    "Calibration covariates must be a 2-D array aligned to rows"
                )
            if not np.all(np.isfinite(self.covariates)):
                raise ValueError("Calibration covariates must be finite")
        if self.sample_weights is not None:
            self.sample_weights = np.asarray(self.sample_weights, dtype=float).copy()
            if self.sample_weights.shape != (n,):
                raise ValueError(
                    "Calibration sample_weights must be 1-D and aligned to rows"
                )
            if not np.all(np.isfinite(self.sample_weights)) or np.any(
                self.sample_weights <= 0
            ):
                raise ValueError(
                    "Calibration sample_weights must be finite and positive"
                )

        if self.row_roles is None:
            self.row_roles = ["external"] * n
        else:
            self.row_roles = list(self.row_roles)
        if len(self.row_roles) != n or any(
            role not in ("external", "evaluation") for role in self.row_roles
        ):
            raise ValueError(
                "row_roles must align to calibration rows and contain only "
                "'external' or 'evaluation'"
            )
        if self.evaluation_keys is None:
            self.evaluation_keys = [None] * n
        else:
            self.evaluation_keys = list(self.evaluation_keys)
        if len(self.evaluation_keys) != n:
            raise ValueError("evaluation_keys must align to calibration rows")
        seen_evaluation_keys: set = set()
        for row, (role, key) in enumerate(zip(self.row_roles, self.evaluation_keys)):
            if role == "evaluation" and key is None:
                raise ValueError("Every evaluation calibration row needs an exact key")
            if role == "external" and key is not None:
                raise ValueError(
                    "External calibration rows cannot carry evaluation keys"
                )
            if key is None:
                continue
            if (
                not isinstance(key, tuple)
                or len(key) != 3
                or not isinstance(key[0], str)
                or not isinstance(key[1], str)
                or not isinstance(key[2], (int, np.integer))
            ):
                raise ValueError(
                    "Evaluation keys must be (policy, prompt_id, draw_idx) tuples"
                )
            normalized_key = (key[0], key[1], int(key[2]))
            if normalized_key[1] != self.prompt_ids[row]:
                raise ValueError(
                    f"Evaluation key prompt {normalized_key[1]!r} does not match "
                    f"calibration prompt_id {self.prompt_ids[row]!r} at row {row}"
                )
            if normalized_key in seen_evaluation_keys:
                raise ValueError(
                    f"Evaluation key {normalized_key!r} appears twice in provenance"
                )
            seen_evaluation_keys.add(normalized_key)
            self.evaluation_keys[row] = normalized_key

    @classmethod
    def from_fitted_calibrator(cls, calibrator: Any) -> "CalibrationProvenance":
        """Compatibility bridge using the exact retained fit rows as external.

        Direct API callers predating the explicit provenance contract can still
        obtain a valid independent-frame bootstrap.  The estimator records that
        coupling was unspecified; high-level analysis supplies explicit roles.
        """
        scores = getattr(calibrator, "_fit_judge_scores", None)
        labels = getattr(calibrator, "_fit_oracle_labels", None)
        prompt_ids = getattr(calibrator, "_fit_prompt_ids", None)
        if scores is None or labels is None or prompt_ids is None:
            raise ValueError(
                "Bootstrap refit requires calibration_provenance or a calibrator "
                "fitted by JudgeCalibrator.fit_cv with retained fit rows."
            )
        return cls(
            judge_scores=np.asarray(scores, dtype=float),
            oracle_labels=np.asarray(labels, dtype=float),
            prompt_ids=list(prompt_ids),
            covariates=getattr(calibrator, "_fit_covariates", None),
            row_roles=["external"] * len(scores),
            evaluation_keys=[None] * len(scores),
            sample_weights=getattr(calibrator, "_fit_sample_weight", None),
        )

    def summary(self) -> Dict[str, Any]:
        roles = list(self.row_roles or [])
        return {
            "n_rows": len(self.judge_scores),
            "n_clusters": len(set(self.prompt_ids)),
            "n_external": roles.count("external"),
            "n_evaluation": roles.count("evaluation"),
            "has_covariates": self.covariates is not None,
            "has_sample_weights": self.sample_weights is not None,
        }


@dataclass
class LabelDesign:
    """Oracle-label observation design for residual augmentation."""

    kind: Literal["representative", "known_propensity", "targeted_unknown"] = (
        "representative"
    )
    propensities: Optional[Dict[str, np.ndarray]] = None

    def __post_init__(self) -> None:
        if self.kind not in (
            "representative",
            "known_propensity",
            "targeted_unknown",
        ):
            raise ValueError(f"Unknown label design: {self.kind}")
        if self.kind == "known_propensity" and not self.propensities:
            raise ValueError(
                "known_propensity label design requires per-policy propensities"
            )
        if self.kind != "known_propensity" and self.propensities is not None:
            raise ValueError("propensities are only valid with kind='known_propensity'")


@dataclass
class DirectPointEstimate:
    """Output of the single Direct point-estimator implementation."""

    estimates: np.ndarray
    pseudo_outcomes: Dict[int, np.ndarray]
    diagnostics: Dict[str, Any]


# ========== Residual-Augmented Estimator (AIPW-style) ==========


def compute_augmented_estimate_per_policy(
    calibrated_full: np.ndarray,
    oracle_labels: np.ndarray,
    oracle_mask: np.ndarray,
    oof_predictions: np.ndarray,
    policy_indices: np.ndarray,
    n_policies: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute residual-augmented estimates (θ̂_aug) per policy.

    Implements AIPW-style debiasing:
        θ̂_aug = (1/n)Σᵢ f̂_full(Sᵢ) + (1/n)Σᵢ (Oᵢ/p)(Yᵢ - f̂_oof(Sᵢ))

    Where:
        - f̂_full: Calibrator fitted on ALL oracle-labeled data (lower variance)
        - f̂_oof: Cluster-OOF predictions (unbiased residuals)
        - Oᵢ: Oracle observation indicator
        - p: Oracle sampling probability (estimated per policy)
        - Yᵢ: Oracle label

    The residual term corrects for calibrator bias using the oracle samples,
    with OOF predictions to avoid overfitting.

    Args:
        calibrated_full: (n,) calibrated scores using full model
        oracle_labels: (n,) oracle labels (NaN for unlabeled)
        oracle_mask: (n,) boolean, True if sample has oracle label
        oof_predictions: (n,) OOF predictions for oracle samples (NaN for unlabeled)
        policy_indices: (n,) which policy each sample belongs to
        n_policies: Number of policies

    Returns:
        Tuple of:
        - augmented_estimates: (P,) augmented estimates per policy
        - diagnostics: Dict with per-policy diagnostics
    """
    augmented_estimates = np.zeros(n_policies)
    diagnostics: Dict[str, Any] = {
        "plug_in_estimates": [],
        "residual_corrections": [],
        "oracle_fractions": [],
        "mean_residuals": [],
    }

    for p in range(n_policies):
        p_mask = policy_indices == p
        n_p = np.sum(p_mask)

        if n_p == 0:
            augmented_estimates[p] = np.nan
            continue

        # Policy-specific oracle mask
        p_oracle_mask = p_mask & oracle_mask
        n_oracle_p = np.sum(p_oracle_mask)

        # Oracle sampling probability for this policy
        p_oracle = n_oracle_p / n_p if n_p > 0 else 0.0

        # Plug-in estimate: mean of full-model predictions
        plug_in = np.mean(calibrated_full[p_mask])

        # Residual correction term
        if n_oracle_p > 0 and p_oracle > 0:
            # Residuals: Y - f̂_oof(S) for oracle samples
            residuals = oracle_labels[p_oracle_mask] - oof_predictions[p_oracle_mask]

            # AIPW correction: (1/n) * Σ (1/p) * residual = (1/p) * mean(residual) * (n_oracle/n)
            # Simplifies to: mean(residual)
            residual_correction = np.mean(residuals)
        else:
            residuals = np.array([])
            residual_correction = 0.0

        # Augmented estimate
        augmented_estimates[p] = plug_in + residual_correction

        # Diagnostics
        diagnostics["plug_in_estimates"].append(float(plug_in))
        diagnostics["residual_corrections"].append(float(residual_correction))
        diagnostics["oracle_fractions"].append(float(p_oracle))
        diagnostics["mean_residuals"].append(
            float(np.mean(residuals)) if len(residuals) > 0 else 0.0
        )

    return augmented_estimates, diagnostics


def get_oof_predictions(
    calibrator: Any,
    judge_scores: np.ndarray,
    oracle_mask: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    oracle_fold_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Get cluster-out-of-fold predictions for oracle samples.

    For each oracle sample, returns the prediction from a calibrator
    trained on data NOT including that sample's cluster/fold, via the
    calibrator's single OOF entry point (`predict_oof`).

    Args:
        calibrator: Fitted JudgeCalibrator with fold models
        judge_scores: (n,) judge scores for all samples
        oracle_mask: (n,) boolean mask for oracle samples
        covariates: Optional (n, d) covariate array
        oracle_fold_ids: Fold assignments returned by the ``fit_cv`` call that
            fitted this calibrator. This may be row-aligned with all
            ``judge_scores`` or compact and aligned with
            ``judge_scores[oracle_mask]``.

    Returns:
        oof_predictions: (n,) OOF predictions (NaN for non-oracle samples)

    Raises:
        ValueError: If ``oracle_fold_ids`` is missing or misaligned with the
            oracle rows — silently substituting full-model predictions here
            would break the out-of-fold contract of θ̂_aug.
    """
    n = len(judge_scores)
    oof_predictions = np.full(n, np.nan)

    # Get oracle samples
    oracle_indices = np.where(oracle_mask)[0]
    n_oracle = len(oracle_indices)
    oracle_judge = judge_scores[oracle_mask]
    oracle_cov = covariates[oracle_mask] if covariates is not None else None

    if oracle_fold_ids is None:
        raise ValueError(
            "get_oof_predictions requires oracle_fold_ids (the fold_ids "
            "returned by the fit_cv call that fitted the calibrator). "
            "Full-model predictions are not a valid substitute for "
            "out-of-fold predictions."
        )
    oracle_fold_ids = np.asarray(oracle_fold_ids)
    if len(oracle_fold_ids) == n:
        oracle_fold_ids = oracle_fold_ids[oracle_mask]
    elif len(oracle_fold_ids) != n_oracle:
        raise ValueError(
            f"oracle_fold_ids length ({len(oracle_fold_ids)}) matches neither "
            f"judge_scores ({n}) nor the oracle rows ({n_oracle}); the "
            "calibrator was fitted on different data than oracle_mask selects."
        )

    oof_predictions[oracle_indices] = np.clip(
        calibrator.predict_oof(oracle_judge, oracle_fold_ids, oracle_cov),
        0.0,
        1.0,
    )
    return oof_predictions


# ========== Oracle-Uncertainty (OUA) Jackknife Recipes ==========


def oracle_jackknife_variance(jack: np.ndarray) -> float:
    """Delete-one-fold jackknife variance of the calibration (OUA) component.

    Var_cal = (K-1)/K * Σ_k (ψ^(−k) − ψ̄)²

    This is the standard delete-a-group jackknife (paper Alg. 6). Note the sum,
    not the mean, over folds: dividing by K here understates the variance by a
    factor of K.

    Args:
        jack: Array of K leave-one-oracle-fold estimates

    Returns:
        Jackknife variance estimate (0.0 if fewer than 2 folds)
    """
    jack = np.asarray(jack, dtype=float)
    K = len(jack)
    if K < 2:
        return 0.0
    psi_bar = float(np.mean(jack))
    return (K - 1) / K * float(np.sum((jack - psi_bar) ** 2))


def oracle_jackknife_estimates(
    calibrator: Any,
    judge_scores: np.ndarray,
    covariates: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Leave-one-oracle-fold estimates of the mean calibrated reward.

    For each oracle fold k, recalibrates the sample with the fold-k model —
    routed through the calibrator's `predict_oof`, which applies the
    mode-appropriate transform (two-stage fold models expect the rank index,
    not raw judge scores) — and records the mean reward.

    Args:
        calibrator: Fitted calibrator exposing `get_fold_models_for_oua` and
            `predict_oof`
        judge_scores: (n,) judge scores for the evaluation sample
        covariates: Optional (n, d) covariate matrix

    Returns:
        Array of K leave-one-fold mean estimates, or None when fewer than
        2 fold models are available
    """
    fold_models = calibrator.get_fold_models_for_oua()
    if not fold_models:
        return None
    judge_scores = np.asarray(judge_scores)
    jack: List[float] = []
    for fold_id in sorted(fold_models):
        if fold_models.get(fold_id) is None:
            continue
        fold_ids = np.full(len(judge_scores), fold_id, dtype=int)
        rewards_loo = np.clip(
            calibrator.predict_oof(judge_scores, fold_ids, covariates), 0.0, 1.0
        )
        jack.append(float(np.mean(rewards_loo)))
    if len(jack) < 2:
        return None
    return np.asarray(jack)


def combine_cluster_and_oracle(
    se_base: float,
    df_cluster: int,
    jackknife_variance: float,
    n_jackknife_folds: int = 0,
) -> Tuple[float, int]:
    """Combine a cluster-robust SE with the oracle-jackknife variance.

    The additive decomposition used by cluster-robust inference:

        se_total = sqrt(se_base² + Var_cal)

    with degrees of freedom capped by the jackknife fold count when the
    oracle component is informative (K >= 2 fold models):

        df = max(min(df_cluster, K - 1), 1)

    Args:
        se_base: Cluster-robust (or standard) SE of the evaluation mean
        df_cluster: Degrees of freedom from clustering (typically G - 1)
        jackknife_variance: Var_cal from `oracle_jackknife_variance` (0.0
            when the OUA component is skipped or unavailable)
        n_jackknife_folds: Number of oracle fold models (caps df at K - 1
            when >= 2; fewer than 2 folds carry no jackknife variance and
            do not constrain df)

    Returns:
        Tuple of (se_total, df)
    """
    se_total = float(np.sqrt(se_base**2 + max(jackknife_variance, 0.0)))
    df = int(df_cluster)
    if n_jackknife_folds >= 2:
        df = min(df, n_jackknife_folds - 1)
    return se_total, max(df, 1)


# ========== Direct Mode Bootstrap Data Structures ==========


@dataclass
class DirectEvalTable:
    """Long-format evaluation table for Direct mode bootstrap.

    This structure enables efficient cluster bootstrap by:
    1. Storing all data in long format (one row per policy-prompt pair)
    2. Precomputing cluster-to-row mappings for O(1) lookup during resampling
    3. Supporting covariates for two-stage calibration

    The bootstrap resamples prompt clusters (shared across policies for paired
    designs), which preserves the correlation structure needed for valid
    pairwise comparisons.
    """

    # Core data arrays (all length n_total = sum of samples across policies)
    prompt_ids: np.ndarray  # (n_total,) cluster identifier (string hashes as int)
    prompt_id_strings: List[str]  # (n_total,) original prompt_id strings
    policy_indices: np.ndarray  # (n_total,) which policy (0, 1, ...)
    judge_scores: np.ndarray  # (n_total,) raw judge scores
    oracle_labels: np.ndarray  # (n_total,) NaN if unlabeled
    oracle_mask: np.ndarray  # (n_total,) boolean, True if labeled

    # Covariates (optional, for two-stage calibration)
    covariates: Optional[np.ndarray]  # (n_total, n_cov) or None
    covariate_names: Optional[List[str]]
    row_keys: Optional[List[EvaluationKey]] = None
    row_keys_synthesized: bool = False

    # Precomputed indices for efficient bootstrap (O(1) lookup)
    cluster_to_rows: Dict[int, np.ndarray] = field(default_factory=dict)

    # Metadata
    policy_names: List[str] = field(default_factory=list)
    n_clusters: int = 0
    n_policies: int = 0
    unique_clusters: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        """Compute derived fields after initialization."""
        n_rows = len(self.prompt_ids)
        for name, values in (
            ("prompt_id_strings", self.prompt_id_strings),
            ("policy_indices", self.policy_indices),
            ("judge_scores", self.judge_scores),
            ("oracle_labels", self.oracle_labels),
            ("oracle_mask", self.oracle_mask),
        ):
            if len(values) != n_rows:
                raise ValueError(f"{name} must align to DirectEvalTable rows")
        if self.row_keys is None and self.policy_names:
            occurrence: Dict[Tuple[str, str], int] = {}
            synthesized: List[EvaluationKey] = []
            for policy_index, prompt_id in zip(
                self.policy_indices, self.prompt_id_strings
            ):
                policy = self.policy_names[int(policy_index)]
                group = (policy, prompt_id)
                draw_index = occurrence.get(group, 0)
                occurrence[group] = draw_index + 1
                synthesized.append((policy, prompt_id, draw_index))
            self.row_keys = synthesized
            self.row_keys_synthesized = True
        if self.row_keys is not None and len(self.row_keys) != n_rows:
            raise ValueError("row_keys must align to DirectEvalTable rows")
        if self.row_keys is not None:
            for row, key in enumerate(self.row_keys):
                if (
                    not isinstance(key, tuple)
                    or len(key) != 3
                    or not isinstance(key[0], str)
                    or not isinstance(key[1], str)
                    or not isinstance(key[2], (int, np.integer))
                ):
                    raise ValueError(
                        "row_keys must contain (policy, prompt_id, draw_idx) tuples"
                    )
                if key[1] != self.prompt_id_strings[row]:
                    raise ValueError(
                        f"row key prompt {key[1]!r} does not match prompt_id "
                        f"{self.prompt_id_strings[row]!r} at row {row}"
                    )
                policy_index = int(self.policy_indices[row])
                if self.policy_names and (
                    policy_index < 0
                    or policy_index >= len(self.policy_names)
                    or key[0] != self.policy_names[policy_index]
                ):
                    raise ValueError(
                        f"row key policy {key[0]!r} does not match policy index "
                        f"{policy_index} at row {row}"
                    )
        if len(self.cluster_to_rows) == 0 and len(self.prompt_ids) > 0:
            # Build cluster-to-rows mapping
            self.unique_clusters = np.unique(self.prompt_ids)
            self.n_clusters = len(self.unique_clusters)
            self.n_policies = len(self.policy_names) if self.policy_names else 0

            # Precompute row indices for each cluster (single pass)
            # This avoids O(G*n) lookups during bootstrap
            for cluster_id in self.unique_clusters:
                self.cluster_to_rows[int(cluster_id)] = np.where(
                    self.prompt_ids == cluster_id
                )[0]


def build_direct_eval_table(
    fresh_draws_per_policy: Dict[str, "FreshDrawDataset"],
    covariate_names: Optional[List[str]] = None,
) -> DirectEvalTable:
    """Build evaluation table from fresh draws for efficient bootstrap.

    Creates a long-format table with precomputed cluster indices for O(1)
    bootstrap resampling. All policies are stacked vertically with a
    policy_indices column to identify which policy each row belongs to.

    Args:
        fresh_draws_per_policy: Dict mapping policy names to FreshDrawDataset
        covariate_names: Optional list of covariate names to extract from
                         sample.metadata (e.g., ["response_length"])

    Returns:
        DirectEvalTable ready for cluster bootstrap
    """
    if not fresh_draws_per_policy:
        raise ValueError("fresh_draws_per_policy cannot be empty")

    policy_names = list(fresh_draws_per_policy.keys())
    n_policies = len(policy_names)

    # Collect arrays from each policy
    all_prompt_ids: List[str] = []
    all_policy_indices: List[int] = []
    all_judge_scores: List[float] = []
    all_oracle_labels: List[float] = []
    all_covariates: List[List[float]] = []
    all_row_keys: List[EvaluationKey] = []

    for policy_idx, policy_name in enumerate(policy_names):
        fd = fresh_draws_per_policy[policy_name]
        for sample in fd.samples:
            all_prompt_ids.append(sample.prompt_id)
            all_policy_indices.append(policy_idx)
            all_judge_scores.append(sample.judge_score)
            all_row_keys.append(
                (policy_name, sample.prompt_id, int(getattr(sample, "draw_idx", 0)))
            )

            # Oracle label: use NaN if not present
            if sample.oracle_label is not None:
                all_oracle_labels.append(sample.oracle_label)
            else:
                all_oracle_labels.append(np.nan)

            # Extract covariates if requested. Missing covariates raise the
            # same actionable error as CalibratedDirectEstimator.fit() —
            # NaN-filling would silently feed fabricated covariates to the
            # bootstrap's calibrator refits.
            if covariate_names:
                row_covs = []
                for cov_name in covariate_names:
                    if cov_name not in sample.metadata:
                        raise ValueError(
                            f"Covariate '{cov_name}' not found in fresh draw metadata "
                            f"for policy '{policy_name}', sample {sample.prompt_id}. "
                            f"Available metadata: {list(sample.metadata.keys())}"
                        )
                    row_covs.append(float(sample.metadata[cov_name]))
                all_covariates.append(row_covs)

    # Convert to numpy arrays
    prompt_id_strings = all_prompt_ids
    # Map prompt_ids to sequential integers (deterministic factorization)
    # Using dict-based factorization instead of hash() because:
    #   1. hash() is non-deterministic across Python processes (PYTHONHASHSEED)
    #   2. hash() % N can have collisions, incorrectly merging clusters
    # This approach guarantees zero collisions and deterministic mapping.
    unique_prompts = list(dict.fromkeys(all_prompt_ids))  # Preserves order
    prompt_to_idx = {pid: idx for idx, pid in enumerate(unique_prompts)}
    prompt_ids = np.array(
        [prompt_to_idx[pid] for pid in all_prompt_ids], dtype=np.int64
    )
    policy_indices = np.array(all_policy_indices, dtype=np.int32)
    judge_scores = np.array(all_judge_scores, dtype=np.float64)
    oracle_labels = np.array(all_oracle_labels, dtype=np.float64)
    oracle_mask = ~np.isnan(oracle_labels)

    # Covariates array
    covariates: Optional[np.ndarray] = None
    if covariate_names and all_covariates:
        covariates = np.array(all_covariates, dtype=np.float64)

    return DirectEvalTable(
        prompt_ids=prompt_ids,
        prompt_id_strings=prompt_id_strings,
        policy_indices=policy_indices,
        judge_scores=judge_scores,
        oracle_labels=oracle_labels,
        oracle_mask=oracle_mask,
        covariates=covariates,
        covariate_names=covariate_names,
        row_keys=all_row_keys,
        policy_names=policy_names,
        n_policies=n_policies,
    )


def validate_calibration_provenance(
    provenance: CalibrationProvenance, calibrator: Any
) -> None:
    """Verify that provenance describes the supplied fitted calibrator."""
    fit_scores = getattr(calibrator, "_fit_judge_scores", None)
    fit_labels = getattr(calibrator, "_fit_oracle_labels", None)
    fit_prompts = getattr(calibrator, "_fit_prompt_ids", None)
    if fit_scores is None or fit_labels is None or fit_prompts is None:
        raise ValueError(
            "Cannot validate calibration provenance: fitted calibrator does not "
            "expose its compact fit rows."
        )
    if not np.array_equal(np.asarray(fit_scores), provenance.judge_scores):
        raise ValueError(
            "calibration_provenance judge_scores do not match the rows used to "
            "fit reward_calibrator"
        )
    if not np.array_equal(np.asarray(fit_labels), provenance.oracle_labels):
        raise ValueError(
            "calibration_provenance oracle_labels do not match the rows used to "
            "fit reward_calibrator"
        )
    if list(fit_prompts) != list(provenance.prompt_ids):
        raise ValueError(
            "calibration_provenance prompt_ids/order do not match the rows used "
            "to fit reward_calibrator"
        )
    fit_covariates = getattr(calibrator, "_fit_covariates", None)
    if (fit_covariates is None) != (provenance.covariates is None):
        raise ValueError(
            "calibration_provenance covariates do not match reward_calibrator"
        )
    if fit_covariates is not None:
        assert provenance.covariates is not None
        if not np.array_equal(np.asarray(fit_covariates), provenance.covariates):
            raise ValueError(
                "calibration_provenance covariate values/order do not match "
                "reward_calibrator"
            )
    fit_sample_weight = getattr(calibrator, "_fit_sample_weight", None)
    if (fit_sample_weight is None) != (provenance.sample_weights is None):
        raise ValueError(
            "calibration_provenance sample_weights do not match reward_calibrator"
        )
    if fit_sample_weight is not None:
        assert provenance.sample_weights is not None
        if not np.array_equal(np.asarray(fit_sample_weight), provenance.sample_weights):
            raise ValueError(
                "calibration_provenance sample_weight values/order do not match "
                "reward_calibrator"
            )


def resolve_label_propensities(
    eval_table: DirectEvalTable,
    label_design: LabelDesign,
    observation_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return row-aligned label inclusion probabilities for augmentation."""
    propensities = np.full(len(eval_table.judge_scores), np.nan, dtype=float)
    if label_design.kind == "targeted_unknown":
        return propensities
    weights = (
        np.ones(len(eval_table.judge_scores), dtype=float)
        if observation_weights is None
        else np.asarray(observation_weights, dtype=float)
    )
    if weights.shape != (len(eval_table.judge_scores),):
        raise ValueError("observation_weights must align to evaluation rows")

    for policy_index, policy in enumerate(eval_table.policy_names):
        rows = np.where(eval_table.policy_indices == policy_index)[0]
        if label_design.kind == "representative":
            observed = eval_table.oracle_mask[rows]
            observed_weight = float(np.sum(weights[rows][observed]))
            if observed_weight > 0:
                propensities[rows] = observed_weight / float(np.sum(weights[rows]))
            continue

        assert label_design.propensities is not None
        if policy not in label_design.propensities:
            raise ValueError(f"Missing known label propensities for policy '{policy}'")
        values = np.asarray(label_design.propensities[policy], dtype=float)
        if values.shape != (len(rows),):
            raise ValueError(
                f"Label propensities for policy '{policy}' must have shape "
                f"({len(rows)},), got {values.shape}"
            )
        if not np.all(np.isfinite(values)) or np.any((values <= 0) | (values > 1)):
            raise ValueError(
                f"Label propensities for policy '{policy}' must be finite in (0, 1]"
            )
        propensities[rows] = values
    return propensities


def residual_predictions_for_evaluation(
    calibrator: Any,
    calibrated_full: np.ndarray,
    eval_table: DirectEvalTable,
    provenance: Optional[CalibrationProvenance],
    calibration_fold_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build residual predictions, using OOF models for linked fit rows.

    Oracle rows that were not used to fit the calibrator use the full-model
    prediction, which is independent of their labels.  Evaluation rows that
    also entered calibration use the model excluding their prompt fold.
    """
    predictions = np.asarray(calibrated_full, dtype=float).copy()
    if provenance is None or not any(
        role == "evaluation" for role in provenance.row_roles or []
    ):
        return predictions
    if eval_table.row_keys is None:
        raise ValueError(
            "Evaluation-linked calibration provenance requires DirectEvalTable.row_keys"
        )

    key_to_row: Dict[EvaluationKey, int] = {}
    for row, key in enumerate(eval_table.row_keys):
        if key in key_to_row:
            raise ValueError(
                f"Evaluation key {key!r} is duplicated; provenance linkage is ambiguous"
            )
        key_to_row[key] = row

    fold_ids = calibration_fold_ids
    if fold_ids is None:
        fold_ids = getattr(calibrator, "_oracle_fold_ids", None)
    if fold_ids is None or len(fold_ids) != len(provenance.judge_scores):
        raise ValueError(
            "Evaluation-linked augmentation requires calibration fold ids aligned "
            "to calibration_provenance"
        )

    eval_rows: List[int] = []
    linked_folds: List[int] = []
    seen_keys: set = set()
    for provenance_row, (role, provenance_key) in enumerate(
        zip(provenance.row_roles or [], provenance.evaluation_keys or [])
    ):
        if role != "evaluation":
            continue
        assert provenance_key is not None
        if provenance_key in seen_keys:
            raise ValueError(
                f"Evaluation key {provenance_key!r} appears twice in provenance"
            )
        seen_keys.add(provenance_key)
        if provenance_key not in key_to_row:
            raise ValueError(
                f"Calibration provenance references evaluation row "
                f"{provenance_key!r}, "
                "but that row is absent from fresh draws"
            )
        eval_row = key_to_row[provenance_key]
        if (
            provenance_key[1] != eval_table.prompt_id_strings[eval_row]
            or provenance.prompt_ids[provenance_row]
            != eval_table.prompt_id_strings[eval_row]
        ):
            raise ValueError(
                f"Calibration provenance prompt for {provenance_key!r} does not "
                "match the evaluation row"
            )
        if not eval_table.oracle_mask[eval_row]:
            raise ValueError(
                f"Calibration provenance links {provenance_key!r}, but its "
                "evaluation row "
                "has no oracle label"
            )
        if not np.isclose(
            eval_table.judge_scores[eval_row], provenance.judge_scores[provenance_row]
        ) or not np.isclose(
            eval_table.oracle_labels[eval_row], provenance.oracle_labels[provenance_row]
        ):
            raise ValueError(
                f"Calibration provenance values for {provenance_key!r} do not match the "
                "evaluation row"
            )
        if provenance.covariates is not None:
            if eval_table.covariates is None or not np.array_equal(
                eval_table.covariates[eval_row],
                provenance.covariates[provenance_row],
            ):
                raise ValueError(
                    f"Calibration provenance covariates for {provenance_key!r} "
                    "do not match the evaluation row"
                )
        eval_rows.append(eval_row)
        linked_folds.append(int(fold_ids[provenance_row]))

    if eval_rows:
        rows_array = np.asarray(eval_rows, dtype=int)
        covariates = (
            eval_table.covariates[rows_array]
            if eval_table.covariates is not None
            else None
        )
        predictions[rows_array] = calibrator.predict_oof(
            eval_table.judge_scores[rows_array],
            np.asarray(linked_folds, dtype=int),
            covariates,
        )
    return np.clip(predictions, 0.0, 1.0)


def compute_direct_point_estimate(
    calibrated_full: np.ndarray,
    eval_table: DirectEvalTable,
    residual_predictions: Optional[np.ndarray],
    label_design: LabelDesign,
    use_augmented_estimator: bool = True,
    observation_weights: Optional[np.ndarray] = None,
    label_propensities: Optional[np.ndarray] = None,
) -> DirectPointEstimate:
    """Compute the Direct estimand on one (possibly weighted) data world.

    This is the only point-estimator implementation used by analytic and
    bootstrap inference.  Per-policy routing is:

    * complete evaluation oracle coverage: weighted raw-oracle mean;
    * partial coverage with a valid label design: calibrated plug-in plus a
      Horvitz-Thompson residual correction;
    * targeted/unknown labeling or augmentation disabled: calibrated plug-in.
    """
    calibrated = np.asarray(calibrated_full, dtype=float)
    n_rows = len(eval_table.judge_scores)
    if calibrated.shape != (n_rows,):
        raise ValueError("calibrated_full must align to evaluation rows")
    weights = (
        np.ones(n_rows, dtype=float)
        if observation_weights is None
        else np.asarray(observation_weights, dtype=float)
    )
    if (
        weights.shape != (n_rows,)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0)
    ):
        raise ValueError(
            "observation_weights must be finite, positive, and row-aligned"
        )

    if label_propensities is None:
        label_propensities = resolve_label_propensities(
            eval_table, label_design, observation_weights=weights
        )
    else:
        label_propensities = np.asarray(label_propensities, dtype=float)
        if label_propensities.shape != (n_rows,):
            raise ValueError("label_propensities must align to evaluation rows")

    estimates = np.full(eval_table.n_policies, np.nan, dtype=float)
    pseudo_outcomes: Dict[int, np.ndarray] = {}
    diagnostics: Dict[str, Any] = {
        "routes": [],
        "plug_in_estimates": [],
        "residual_corrections": [],
        "oracle_fractions": [],
        "label_design": label_design.kind,
        "augmentation_requested": bool(use_augmented_estimator),
        "augmentation_effective": False,
    }

    for policy_index in range(eval_table.n_policies):
        rows = np.where(eval_table.policy_indices == policy_index)[0]
        if len(rows) == 0:
            diagnostics["routes"].append("no_data")
            diagnostics["plug_in_estimates"].append(float("nan"))
            diagnostics["residual_corrections"].append(0.0)
            diagnostics["oracle_fractions"].append(0.0)
            continue
        policy_weights = weights[rows]
        observed = eval_table.oracle_mask[rows]
        oracle_fraction = float(
            np.sum(policy_weights[observed]) / np.sum(policy_weights)
        )

        if np.all(observed):
            values = eval_table.oracle_labels[rows].astype(float, copy=True)
            estimate = float(np.average(values, weights=policy_weights))
            route = "direct_oracle"
            plug_in = float(np.average(calibrated[rows], weights=policy_weights))
            correction = estimate - plug_in
        else:
            values = calibrated[rows].astype(float, copy=True)
            plug_in = float(np.average(values, weights=policy_weights))
            correction = 0.0
            route = "plug_in"
            can_augment = (
                use_augmented_estimator
                and label_design.kind != "targeted_unknown"
                and np.any(observed)
            )
            if can_augment:
                if residual_predictions is None:
                    raise ValueError(
                        "Residual augmentation requires residual_predictions"
                    )
                residual_predictions_array = np.asarray(
                    residual_predictions, dtype=float
                )
                if residual_predictions_array.shape != (n_rows,):
                    raise ValueError(
                        "residual_predictions must align to evaluation rows"
                    )
                residual = (
                    eval_table.oracle_labels[rows][observed]
                    - residual_predictions_array[rows][observed]
                )
                if label_design.kind == "representative":
                    correction = float(
                        np.average(residual, weights=policy_weights[observed])
                    )
                    propensity = oracle_fraction
                    values += correction
                    values[observed] += (residual - correction) / propensity
                    estimate = plug_in + correction
                else:
                    prop = label_propensities[rows]
                    if np.any(~np.isfinite(prop[observed])) or np.any(
                        (prop[observed] <= 0) | (prop[observed] > 1)
                    ):
                        raise ValueError(
                            "Observed oracle rows require finite label propensities "
                            "in (0, 1]"
                        )
                    values[observed] += residual / prop[observed]
                    estimate = float(np.average(values, weights=policy_weights))
                    correction = estimate - plug_in
                route = "augmented"
                diagnostics["augmentation_effective"] = True
            else:
                estimate = plug_in
                if use_augmented_estimator and label_design.kind == "targeted_unknown":
                    route = "plug_in_targeted_unknown"

        estimates[policy_index] = estimate
        pseudo_outcomes[policy_index] = values
        diagnostics["routes"].append(route)
        diagnostics["plug_in_estimates"].append(plug_in)
        diagnostics["residual_corrections"].append(float(correction))
        diagnostics["oracle_fractions"].append(oracle_fraction)

    return DirectPointEstimate(estimates, pseudo_outcomes, diagnostics)


def direct_oracle_jackknife_estimates(
    calibrator: Any,
    eval_table: DirectEvalTable,
    label_design: LabelDesign,
    use_augmented_estimator: bool = True,
) -> Optional[np.ndarray]:
    """Recompute the Direct estimand under each leave-oracle-fold model."""
    fold_models = calibrator.get_fold_models_for_oua()
    if not fold_models:
        return None

    jackknife: List[np.ndarray] = []
    for fold_id in sorted(fold_models):
        if fold_models.get(fold_id) is None:
            continue
        fold_ids = np.full(len(eval_table.judge_scores), fold_id, dtype=int)
        predictions = np.clip(
            calibrator.predict_oof(
                eval_table.judge_scores, fold_ids, eval_table.covariates
            ),
            0.0,
            1.0,
        )
        point = compute_direct_point_estimate(
            predictions,
            eval_table,
            predictions,
            label_design,
            use_augmented_estimator=use_augmented_estimator,
        )
        jackknife.append(point.estimates)

    if len(jackknife) < 2:
        return None
    return np.vstack(jackknife)


def make_calibrator_factory(
    mode: Literal["monotone", "two_stage", "auto"],
    covariate_names: Optional[List[str]] = None,
    seed: int = 42,
) -> Callable[[], Any]:
    """Create a factory function that produces fresh JudgeCalibrator instances.

    This factory pattern ensures each bootstrap replicate gets a completely
    fresh calibrator instance, avoiding any state leakage between replicates.

    Passing ``"auto"`` re-runs mode selection in every weighted bootstrap
    world, matching the estimator users requested on the original data.

    Args:
        mode: Calibration mode ('monotone', 'two_stage', or 'auto').
        covariate_names: Optional list of covariate names for two-stage calibration
        seed: Random seed for reproducibility

    Returns:
        Callable that creates a new JudgeCalibrator instance on each call
    """
    from ..calibration.judge import JudgeCalibrator

    def factory() -> Any:
        return JudgeCalibrator(
            random_seed=seed,
            calibration_mode=mode,
            covariate_names=covariate_names,
        )

    return factory


def cluster_bootstrap_direct_with_refit(
    eval_table: DirectEvalTable,
    calibrator_factory: Callable[[], Any],
    n_bootstrap: int = 2000,
    min_oracle_per_replicate: int = 30,
    alpha: float = 0.05,
    seed: int = 42,
    use_augmented_estimator: bool = True,
    calibration_policy_idx: Optional[int] = None,
    calibration_provenance: Optional[CalibrationProvenance] = None,
    label_design: Optional[LabelDesign] = None,
    point_calibrator: Optional[Any] = None,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """Positive prompt-cluster-weight bootstrap with calibrator refit.

    Every requested replicate is evaluated exactly once.  Exponential
    mean-one weights keep all clusters present, so fold sufficiency is stable
    and there is no retry-conditioned bootstrap distribution.  Calibration
    rows linked to evaluation observations reuse that observation's prompt
    weight; external calibration rows on prompts shared with the evaluation
    frame reuse that prompt cluster's weight, and only prompt-disjoint
    external clusters receive independent weights.

    ``min_oracle_per_replicate`` remains an accepted compatibility argument but
    is not used: positive weights preserve the original calibration support.
    Any invalid replicate aborts inference with an actionable error.
    """
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    rng = np.random.default_rng(seed)
    label_design = label_design or LabelDesign()
    if calibration_provenance is not None and calibration_policy_idx is not None:
        raise ValueError(
            "calibration_policy_idx cannot be combined with explicit "
            "calibration_provenance; encode the intended fit rows directly"
        )

    full_coverage = []
    for policy_index in range(eval_table.n_policies):
        policy_rows = eval_table.policy_indices == policy_index
        full_coverage.append(
            bool(np.any(policy_rows) and np.all(eval_table.oracle_mask[policy_rows]))
        )
    needs_calibrator = not all(full_coverage)

    # Compatibility for the lower-level transport bootstrap: when no explicit
    # provenance is supplied, the requested policy's labeled evaluation rows
    # are the calibration fit sample.
    if needs_calibrator and calibration_provenance is None:
        mask = eval_table.oracle_mask.copy()
        if calibration_policy_idx is not None:
            mask &= eval_table.policy_indices == calibration_policy_idx
        rows = np.where(mask)[0]
        if len(rows) > 0:
            if eval_table.row_keys is None:
                raise ValueError(
                    "Implicit evaluation calibration requires evaluation row keys"
                )
            calibration_provenance = CalibrationProvenance(
                judge_scores=eval_table.judge_scores[rows],
                oracle_labels=eval_table.oracle_labels[rows],
                prompt_ids=[eval_table.prompt_id_strings[i] for i in rows],
                covariates=(
                    eval_table.covariates[rows]
                    if eval_table.covariates is not None
                    else None
                ),
                row_roles=["evaluation"] * len(rows),
                evaluation_keys=[eval_table.row_keys[i] for i in rows],
            )
    effective_augmentation = use_augmented_estimator
    if needs_calibrator and calibration_provenance is None:
        raise ValueError(
            "Partial-oracle Direct bootstrap needs calibration provenance with "
            "at least four independent prompt clusters"
        )

    # Point estimate uses the supplied fitted calibrator, ensuring analytic and
    # bootstrap inference report the same estimator. Lower-level callers may
    # omit it, in which case we fit once on the exact provenance rows.
    point_fit_result: Optional[Any] = None
    if point_calibrator is None and needs_calibrator:
        assert calibration_provenance is not None
        point_calibrator = calibrator_factory()
        point_fit_result = point_calibrator.fit_cv(
            calibration_provenance.judge_scores,
            calibration_provenance.oracle_labels,
            n_folds=n_folds,
            prompt_ids=calibration_provenance.prompt_ids,
            covariates=calibration_provenance.covariates,
            sample_weight=calibration_provenance.sample_weights,
        )
    elif (
        needs_calibrator
        and point_calibrator is not None
        and calibration_provenance is not None
    ):
        validate_calibration_provenance(calibration_provenance, point_calibrator)

    if not needs_calibrator:
        calibrated_point = np.clip(
            eval_table.judge_scores.astype(float, copy=True), 0.0, 1.0
        )
        residual_point = calibrated_point.copy()
    elif point_calibrator is None:
        calibrated_point = eval_table.judge_scores.astype(float, copy=True)
        residual_point = calibrated_point.copy()
    else:
        calibrated_point = np.clip(
            point_calibrator.predict(
                eval_table.judge_scores, covariates=eval_table.covariates
            ),
            0.0,
            1.0,
        )
        point_fold_ids = (
            point_fit_result.fold_ids if point_fit_result is not None else None
        )
        residual_point = residual_predictions_for_evaluation(
            point_calibrator,
            calibrated_point,
            eval_table,
            calibration_provenance,
            calibration_fold_ids=point_fold_ids,
        )

    point = compute_direct_point_estimate(
        calibrated_point,
        eval_table,
        residual_point,
        label_design,
        use_augmented_estimator=effective_augmentation,
    )

    # Exact key lookup couples evaluation-linked calibration rows to their
    # prompt bootstrap weight.
    key_to_eval_row: Dict[EvaluationKey, int] = {}
    if eval_table.row_keys is not None:
        for row, key in enumerate(eval_table.row_keys):
            if key in key_to_eval_row:
                raise ValueError(f"Duplicate evaluation key {key!r}")
            key_to_eval_row[key] = row

    # Map evaluation prompts to their bootstrap cluster so external-role
    # calibration rows sharing a prompt with the evaluation frame reuse that
    # cluster's weight draw — resampling the shared prompt as one unit
    # preserves the calibration-evaluation covariance. Tables from
    # build_direct_eval_table factorize prompts globally, so each prompt maps
    # to exactly one cluster and this lookup is exact; only the decoupled
    # policy-prompt tables built for unpaired multi-policy runs give a prompt
    # several clusters, and there setdefault keeps the first-seen one. Truly
    # disjoint external prompts keep independent weights.
    prompt_to_eval_cluster: Dict[str, int] = {}
    for cluster, prompt in zip(eval_table.prompt_ids, eval_table.prompt_id_strings):
        prompt_to_eval_cluster.setdefault(str(prompt), int(cluster))

    external_clusters: List[str] = []
    shared_external_clusters: List[str] = []
    if calibration_provenance is not None:
        all_external_clusters = sorted(
            {
                prompt_id
                for prompt_id, role in zip(
                    calibration_provenance.prompt_ids,
                    calibration_provenance.row_roles or [],
                )
                if role == "external"
            }
        )
        for prompt_id in all_external_clusters:
            if str(prompt_id) in prompt_to_eval_cluster:
                shared_external_clusters.append(prompt_id)
            else:
                external_clusters.append(prompt_id)

    bootstrap_matrix = np.empty((n_bootstrap, eval_table.n_policies), dtype=float)
    for replicate in range(n_bootstrap):
        eval_cluster_draw = rng.exponential(
            scale=1.0, size=len(eval_table.unique_clusters)
        )
        eval_cluster_weights = {
            int(cluster): float(weight)
            for cluster, weight in zip(eval_table.unique_clusters, eval_cluster_draw)
        }
        eval_weights = np.asarray(
            [eval_cluster_weights[int(cluster)] for cluster in eval_table.prompt_ids],
            dtype=float,
        )

        boot_calibrator: Optional[Any] = None
        boot_fold_ids: Optional[np.ndarray] = None
        if needs_calibrator:
            assert calibration_provenance is not None
            external_draw = rng.exponential(scale=1.0, size=len(external_clusters))
            external_weights = dict(zip(external_clusters, external_draw))
            for shared_prompt in shared_external_clusters:
                external_weights[shared_prompt] = eval_cluster_weights[
                    prompt_to_eval_cluster[str(shared_prompt)]
                ]
            calibration_weights = np.empty(
                len(calibration_provenance.judge_scores), dtype=float
            )
            base_calibration_weights = (
                np.ones(len(calibration_provenance.judge_scores), dtype=float)
                if calibration_provenance.sample_weights is None
                else calibration_provenance.sample_weights
            )
            for i, (role, prompt_id, calibration_key) in enumerate(
                zip(
                    calibration_provenance.row_roles or [],
                    calibration_provenance.prompt_ids,
                    calibration_provenance.evaluation_keys or [],
                )
            ):
                if role == "external":
                    bootstrap_weight = external_weights[prompt_id]
                else:
                    assert calibration_key is not None
                    if calibration_key not in key_to_eval_row:
                        raise ValueError(
                            f"Evaluation-linked calibration key "
                            f"{calibration_key!r} is absent"
                        )
                    bootstrap_weight = eval_weights[key_to_eval_row[calibration_key]]
                calibration_weights[i] = base_calibration_weights[i] * bootstrap_weight

            boot_calibrator = calibrator_factory()
            try:
                fit_result = boot_calibrator.fit_cv(
                    calibration_provenance.judge_scores,
                    calibration_provenance.oracle_labels,
                    n_folds=n_folds,
                    prompt_ids=calibration_provenance.prompt_ids,
                    covariates=calibration_provenance.covariates,
                    quiet=True,
                    sample_weight=calibration_weights,
                )
                boot_fold_ids = fit_result.fold_ids
                calibrated_boot = np.clip(
                    boot_calibrator.predict(
                        eval_table.judge_scores,
                        covariates=eval_table.covariates,
                    ),
                    0.0,
                    1.0,
                )
                residual_boot = residual_predictions_for_evaluation(
                    boot_calibrator,
                    calibrated_boot,
                    eval_table,
                    calibration_provenance,
                    calibration_fold_ids=boot_fold_ids,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Positive-weight bootstrap replicate {replicate} failed; "
                    "no replicates were discarded or retried. Check calibration "
                    f"fold support/model compatibility. Original error: {exc}"
                ) from exc
        else:
            calibrated_boot = calibrated_point
            residual_boot = residual_point

        replicate_point = compute_direct_point_estimate(
            calibrated_boot,
            eval_table,
            residual_boot,
            label_design,
            use_augmented_estimator=effective_augmentation,
            observation_weights=eval_weights,
        )
        if not np.all(np.isfinite(replicate_point.estimates)):
            raise RuntimeError(
                f"Positive-weight bootstrap replicate {replicate} produced "
                "non-finite estimates; no retry was attempted."
            )
        bootstrap_matrix[replicate] = replicate_point.estimates

    standard_errors = np.std(bootstrap_matrix, axis=0, ddof=1)
    ci_lower = np.percentile(bootstrap_matrix, 100 * alpha / 2, axis=0)
    ci_upper = np.percentile(bootstrap_matrix, 100 * (1 - alpha / 2), axis=0)
    policy_cluster_counts = np.asarray(
        [
            len(
                np.unique(
                    eval_table.prompt_ids[eval_table.policy_indices == policy_index]
                )
            )
            for policy_index in range(eval_table.n_policies)
        ],
        dtype=int,
    )
    unavailable = policy_cluster_counts < 2
    standard_errors[unavailable] = np.nan
    ci_lower[unavailable] = np.nan
    ci_upper[unavailable] = np.nan
    bootstrap_matrix[:, unavailable] = np.nan
    n_calibration = (
        len(calibration_provenance.judge_scores)
        if calibration_provenance is not None
        else 0
    )
    oracle_summary = {
        "min": n_calibration,
        "p10": n_calibration,
        "median": n_calibration,
    }

    return {
        "bootstrap_matrix": bootstrap_matrix,
        "estimates": point.estimates,
        "standard_errors": standard_errors,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_valid_replicates": n_bootstrap,
        "n_attempts": n_bootstrap,
        "skip_rate": 0.0,
        "oracle_count_summary": oracle_summary,
        "policy_names": eval_table.policy_names,
        "n_clusters": eval_table.n_clusters,
        "n_policies": eval_table.n_policies,
        "policy_cluster_counts": policy_cluster_counts.tolist(),
        "inference_unavailable_policies": [
            eval_table.policy_names[index] for index in np.where(unavailable)[0]
        ],
        "inference_unavailable_reason": (
            "fewer_than_two_independent_clusters" if np.any(unavailable) else None
        ),
        "alpha": alpha,
        "seed": seed,
        "min_oracle_per_replicate": None,
        "use_augmented_estimator": effective_augmentation,
        "augmentation_diagnostics": point.diagnostics,
        "calibration_policy_idx": calibration_policy_idx,
        "bootstrap_scheme": "positive_exponential_cluster_weights",
        "provenance_summary": (
            calibration_provenance.summary()
            if calibration_provenance is not None
            else {"n_rows": 0, "n_clusters": 0}
        ),
        "label_design": label_design.kind,
    }


# ========== Dependence-Robust Standard Errors ==========


def cluster_robust_se(
    data: np.ndarray,
    cluster_ids: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    influence_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute cluster-robust (sandwich) standard errors with CRV1 correction.

    For data with cluster structure (e.g., multiple obs per user),
    accounts for within-cluster correlation using the CRV1 variance estimator.

    Args:
        data: Input data array
        cluster_ids: Cluster membership for each observation
        statistic_fn: Function that computes the statistic
        influence_fn: Function that computes influence functions
        alpha: Significance level for CI

    Returns:
        Dictionary with robust standard errors and t-based confidence intervals
    """
    n = len(data)
    if n == 0:
        return {
            "estimate": float("nan"),
            "se": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_clusters": 0,
            "df": 0,
        }

    estimate = statistic_fn(data)

    # Build influence contributions
    if influence_fn is None:
        # Default: sample-mean statistic -> IF is (x_i - mean)
        if data.ndim != 1:
            raise ValueError(
                "For multi-dimensional data, provide influence_fn. "
                "Default influence function only works for 1-D data."
            )
        influences = (data - estimate).astype(float, copy=False)
    else:
        # Use provided influence function
        influences = influence_fn(data).astype(float, copy=False)

    # Center defensively for numerical stability
    influences = influences - np.mean(influences)

    # Get unique clusters
    clusters = np.asarray(cluster_ids, dtype=int)
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    df = max(G - 1, 1)

    if G < 2:
        raise ValueError(
            "Cluster-robust inference requires at least two independent "
            "clusters; row-level IID SE is not a valid fallback."
        )

    # Cluster totals of IF
    T = np.array(
        [np.sum(influences[clusters == g]) for g in unique_clusters], dtype=float
    )
    T = T - T.mean()  # Center across clusters

    # CRV1 variance for a mean-type estimator (with G/(G-1) factor):
    var_hat = (G / (G - 1)) * np.sum(T**2) / (n**2)
    se = float(np.sqrt(max(var_hat, 0.0)))

    # Confidence interval using t-distribution with G - 1 df
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_lower = estimate - t_crit * se
    ci_upper = estimate + t_crit * se

    return {
        "estimate": float(estimate),
        "se": se,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_clusters": int(G),
        "df": int(df),
    }
