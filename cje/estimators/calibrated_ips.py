"""Calibrated Inverse Propensity Scoring (IPS) estimator with stacked SIMCal.

This is the core CJE estimator that uses stacked Score-Indexed Monotone Calibration
(SIMCal) to stabilize IPS in heavy-tail regimes. It combines {baseline, increasing,
decreasing} candidates via convex optimization to minimize OOF influence function
variance, then blends toward uniform to meet variance/ESS constraints.
"""

import numpy as np
from typing import Dict, Optional, Set, Any, List, cast, Tuple
import logging

from .base_estimator import BaseCJEEstimator
from ..data.models import EstimationResult
from ..data.precomputed_sampler import PrecomputedSampler
from ..diagnostics import IPSDiagnostics, Status
from ..diagnostics import compute_weight_diagnostics
from ..calibration.simcal import SIMCalibrator, SimcalConfig

logger = logging.getLogger(__name__)


class CalibratedIPS(BaseCJEEstimator):
    """IPS estimator with optional SIMCal weight calibration.

    Can operate in two modes:
    1. calibrate_weights=True (default): Uses stacked Score-Indexed Monotone Calibration (SIMCal)
       to reduce variance and heavy-tail pathologies in importance weights
    2. calibrate_weights=False: Uses raw importance weights directly (equivalent to traditional IPS)

    Features when calibrated:
    - Stacked calibration combining multiple candidates optimally
    - OOF influence function variance minimization
    - ESS floor and variance cap constraints
    - Judge score-indexed calibration for better alignment
    - Automatic DR-aware calibration when reward_calibrator available

    Features in both modes:
    - Oracle slice augmentation for honest confidence intervals
    - Comprehensive diagnostics
    - Optional weight clipping
    - Outer cross-validation for honest inference (default enabled)

    Args:
        sampler: PrecomputedSampler with data
        calibrate_weights: Whether to apply SIMCal weight calibration (default True)
        weight_mode: "hajek" for mean-one normalized weights, "raw" for unnormalized (default "hajek")
        clip_weight: Maximum weight value before calibration (default None = no clipping)
        ess_floor: Minimum ESS as fraction of n (default 0.2 = 20% ESS) [only used if calibrate_weights=True]
        var_cap: Maximum allowed variance of calibrated weights (default 1.0 = no variance increase) [only used if calibrate_weights=True]
        reward_calibrator: Optional JudgeCalibrator for OUA in variance estimation
        include_baseline: Whether to include raw weights in the stack (default False) [only used if calibrate_weights=True]
        baseline_shrink: Shrinkage toward baseline for stability (default 0.0) [only used if calibrate_weights=True]
        refuse_unreliable: Whether to refuse (return NaN) for unreliable estimates (default False)
        suppress_overlap_warnings: Whether to suppress overlap warnings (default False, used when IPS is internal to DR)
        use_outer_cv: Whether to use outer CV for honest weight learning (default True)
        n_outer_folds: Number of outer folds for honest inference (default 5)
        outer_cv_seed: Random seed for outer CV folds (default 1042, set to match DR folds for alignment)
        **kwargs: Additional arguments passed to BaseCJEEstimator (e.g., oracle_slice_config)
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        calibrate_weights: bool = True,
        weight_mode: str = "hajek",
        clip_weight: Optional[float] = None,
        ess_floor: Optional[float] = 0.2,
        var_cap: Optional[float] = 1.0,
        reward_calibrator: Optional[Any] = None,
        include_baseline: bool = False,
        baseline_shrink: float = 0.0,
        run_diagnostics: bool = True,
        refuse_unreliable: bool = False,
        suppress_overlap_warnings: bool = False,
        oua_jackknife: bool = True,
        use_outer_cv: bool = True,
        n_outer_folds: int = 5,
        outer_cv_seed: int = 1042,
        **kwargs: Any,
    ):
        # Pass OUA parameters to base class
        super().__init__(
            sampler=sampler,
            run_diagnostics=run_diagnostics,
            diagnostic_config=None,  # Will use defaults
            reward_calibrator=reward_calibrator,
            oua_jackknife=oua_jackknife,
            **kwargs,  # Passes oracle_slice_config if provided
        )
        self.calibrate_weights = calibrate_weights
        self.weight_mode = weight_mode
        self.clip_weight = clip_weight
        self.ess_floor = ess_floor if calibrate_weights else None
        self.var_cap = var_cap if calibrate_weights else None
        self.include_baseline = include_baseline if calibrate_weights else True
        self.baseline_shrink = baseline_shrink if calibrate_weights else 0.0
        self.refuse_unreliable = refuse_unreliable
        self.suppress_overlap_warnings = suppress_overlap_warnings
        self.use_outer_cv = use_outer_cv
        self.n_outer_folds = n_outer_folds
        self.outer_cv_seed = outer_cv_seed
        self._no_overlap_policies: Set[str] = set()
        self._calibration_info: Dict[str, Dict] = {}  # Store calibration details
        self._diagnostics: Optional[IPSDiagnostics] = None

    def fit(self) -> None:
        """Fit weights for all target policies (with or without calibration)."""
        for policy in self.sampler.target_policies:
            # Get raw weights (with optional pre-clipping and weight mode)
            raw_weights = self.sampler.compute_importance_weights(
                policy, clip_weight=self.clip_weight, mode=self.weight_mode
            )
            if raw_weights is None:
                continue

            # Check for no overlap (all weights are zero)
            if np.all(raw_weights == 0):
                logger.warning(
                    f"Policy '{policy}' has no overlap with base policy (all weights zero)."
                )
                self._no_overlap_policies.add(policy)
                continue

            # If not calibrating, just use raw weights
            if not self.calibrate_weights:
                logger.debug(
                    f"Raw IPS weights for policy '{policy}': "
                    f"mean={raw_weights.mean():.3f}, std={raw_weights.std():.3f}, "
                    f"min={raw_weights.min():.3f}, max={raw_weights.max():.3f}"
                )

                # Cache raw weights
                self._weights_cache[policy] = raw_weights

                # Oracle augmentation removed - using OUA jackknife only

                continue  # Skip calibration for this policy

            # ========== Calibration path (original code) ==========
            logger.debug(
                f"Calibrating weights for policy '{policy}': "
                f"n_samples={len(raw_weights)}, raw_mean={raw_weights.mean():.3f}"
            )

            # Get data and judge scores for this policy
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                logger.warning(f"No data for policy '{policy}'. Skipping.")
                continue

            # Policy-subset judge scores (aligned with raw_weights)
            S_policy = np.asarray(
                [d.get("judge_score", np.nan) for d in data], dtype=float
            )
            if np.all(np.isnan(S_policy)):
                raise ValueError(
                    f"Judge scores are required for SIMCal calibration of policy '{policy}'. "
                    "Ensure samples have 'judge_score' field."
                )

            # Get rewards for this policy (always needed for influence functions)
            rewards = np.array([d["reward"] for d in data], dtype=float)
            rewards_oof = None  # Will be populated if reward_calibrator available

            # Try to get cross-fitted rewards if reward_calibrator available (for SIMCal ordering)
            # Get OOF predictions for this policy subset
            g_oof = None
            fold_ids: Optional[np.ndarray] = (
                None  # Initialize before conditional blocks
            )

            if self.reward_calibrator is not None:
                try:
                    # Option 1: Use index-based OOF for policy subset
                    if hasattr(self.reward_calibrator, "predict_oof_by_index"):
                        # Build mapping checking for duplicates
                        ds_index_by_pid = {}
                        dups = set()
                        for i, s in enumerate(self.sampler.dataset.samples):
                            pid = str(s.prompt_id)
                            if pid in ds_index_by_pid:
                                dups.add(pid)
                            ds_index_by_pid[pid] = i

                        pids = [str(d.get("prompt_id")) for d in data]

                        # Check if any duplicates affect our data
                        if dups.intersection(pids):
                            logger.debug(
                                f"Duplicate prompt_ids detected for policy '{policy}'; falling back to fold-based OOF"
                            )
                            g_oof = None
                        else:
                            ds_idx = np.asarray(
                                [ds_index_by_pid.get(pid, -1) for pid in pids],
                                dtype=int,
                            )
                        if g_oof is None and np.all(ds_idx >= 0):
                            g_oof = self.reward_calibrator.predict_oof_by_index(ds_idx)
                            if g_oof is not None:
                                logger.debug(
                                    f"Using index-based cross-fitted rewards (g^OOF) as SIMCal ordering for policy '{policy}'"
                                )
                                # Also use OOF rewards as IF targets if available
                                if len(g_oof) == len(rewards):
                                    rewards_oof = np.asarray(g_oof, dtype=float)

                    # Option 2: Use fold-based OOF with policy subset
                    if g_oof is None and hasattr(self.reward_calibrator, "predict_oof"):
                        from ..data.folds import get_fold

                        n_folds = self.sampler.dataset.metadata.get("n_folds", 5)
                        seed = self.sampler.dataset.metadata.get("fold_seed", 42)
                        fold_ids = np.asarray(
                            [
                                get_fold(
                                    str(d.get("prompt_id", f"sample_{i}")),
                                    n_folds,
                                    seed,
                                )
                                for i, d in enumerate(data)
                            ],
                            dtype=int,
                        )
                        g_oof = self.reward_calibrator.predict_oof(S_policy, fold_ids)
                        if g_oof is not None:
                            logger.debug(
                                f"Using fold-based cross-fitted rewards (g^OOF) as SIMCal ordering for policy '{policy}'"
                            )
                            rewards_oof = np.asarray(g_oof, dtype=float)
                except Exception as e:
                    logger.debug(f"SIMCal ordering OOF failed for '{policy}': {e}")
                    g_oof = None

            # Determine the ordering index for SIMCal
            # Use cross-fitted calibrated rewards if available, otherwise raw judge scores
            # Ensure alignment with raw_weights length
            ordering_index = (
                g_oof
                if (g_oof is not None and len(g_oof) == len(raw_weights))
                else S_policy
            )

            # For residuals (used by DR methods), we need policy-specific computations
            # This is handled separately by DR estimators that have access to the subset mapping
            residuals = None

            # Run stacked SIMCal calibration
            cfg = SimcalConfig(
                ess_floor=self.ess_floor,
                var_cap=self.var_cap,
                include_baseline=self.include_baseline,
                baseline_shrink=self.baseline_shrink,
                random_seed=self.outer_cv_seed,  # Use outer_cv_seed for consistency
            )

            if self.use_outer_cv:
                # Use outer CV for honest inference
                calibrated, calib_info = self._calibrate_with_outer_cv(
                    raw_weights=raw_weights,
                    ordering_index=ordering_index,
                    rewards=rewards,
                    rewards_oof=rewards_oof,
                    fold_ids=fold_ids,
                    cfg=cfg,
                    policy=policy,
                )
            else:
                # Standard single-pass calibration
                sim = SIMCalibrator(cfg)
                calibrated, calib_info = sim.transform(
                    raw_weights,
                    ordering_index,  # Now uses g_oof when available, judge_scores otherwise
                    rewards=(
                        rewards_oof if rewards_oof is not None else rewards
                    ),  # Prefer OOF rewards for SIMCal
                    residuals=residuals,  # None for IPS (DR estimators handle this separately)
                    fold_ids=(
                        fold_ids
                        if fold_ids is not None
                        else self._build_default_folds(len(rewards), seed=42)
                    ),  # Always provide fold IDs
                )

            # Cache results
            self._weights_cache[policy] = calibrated
            self._calibration_info[policy] = calib_info

            # Oracle augmentation removed - using OUA jackknife only

        self._fitted = True

    def _calibrate_with_outer_cv(
        self,
        raw_weights: np.ndarray,
        ordering_index: np.ndarray,
        rewards: np.ndarray,
        rewards_oof: Optional[np.ndarray],
        fold_ids: Optional[np.ndarray],
        cfg: SimcalConfig,
        policy: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calibrate weights using outer cross-validation for honest inference.

        Args:
            raw_weights: Raw importance weights
            ordering_index: Score index for ordering (judge scores or g_oof)
            rewards: In-sample rewards
            rewards_oof: Out-of-fold rewards (if available)
            fold_ids: Inner fold IDs for SIMCal's internal CV
            cfg: SIMCal configuration
            policy: Policy name (for deterministic folds)

        Returns:
            Tuple of (calibrated weights, info dict)
        """
        n = len(raw_weights)

        # Create deterministic outer folds based on prompt_ids
        # Use different seed offset to avoid collision with inner folds
        data = self.sampler.get_data_for_policy(policy)
        if data and len(data) == n:
            from ..data.folds import get_fold

            prompt_ids = [
                str(d.get("prompt_id", f"sample_{i}")) for i, d in enumerate(data)
            ]
            outer_fold_ids = np.array(
                [
                    get_fold(
                        pid, self.n_outer_folds, seed=self.outer_cv_seed
                    )  # Can be aligned with DR folds via outer_cv_seed param
                    for pid in prompt_ids
                ],
                dtype=int,
            )
        else:
            # Fallback to sklearn if prompt_ids not available
            from sklearn.model_selection import KFold

            outer_kf = KFold(n_splits=self.n_outer_folds, shuffle=True, random_state=42)
            outer_fold_ids = np.zeros(n, dtype=int)
            for fold_idx, (_, test_idx) in enumerate(outer_kf.split(np.arange(n))):
                outer_fold_ids[test_idx] = fold_idx

        # Initialize arrays for results
        calibrated_weights = np.zeros(n)
        mixture_weights_list = []
        gamma_list = []
        per_fold_train_estimates = {}
        per_fold_test_weights = {}

        # Process each outer fold
        for v in range(self.n_outer_folds):
            test_mask = outer_fold_ids == v
            train_mask = ~test_mask

            if np.sum(train_mask) < 10 or np.sum(test_mask) < 2:
                # Skip if fold is too small
                logger.warning(f"Outer fold {v} too small, using raw weights")
                calibrated_weights[test_mask] = raw_weights[test_mask]
                # Use train subset estimate for tiny folds (not global)
                sw = float(np.sum(raw_weights[train_mask]))
                if sw > 0:
                    psi_train = float(
                        np.sum(raw_weights[train_mask] * rewards[train_mask]) / sw
                    )
                else:
                    psi_train = (
                        float(np.mean(rewards[train_mask]))
                        if np.any(train_mask)
                        else 0.0
                    )
                per_fold_train_estimates[v] = {
                    "psi": psi_train,
                    "mean_w": (
                        float(np.mean(raw_weights[train_mask]))
                        if np.any(train_mask)
                        else 1.0
                    ),
                }
                continue

            # Extract training data
            w_train = raw_weights[train_mask]
            s_train = ordering_index[train_mask]
            r_train = (
                rewards_oof[train_mask]
                if rewards_oof is not None
                else rewards[train_mask]
            )

            # Compute and store training estimate and mean weight for honest IFs
            sw = float(np.sum(w_train))
            psi_train = (
                float(np.sum(w_train * r_train) / sw)
                if sw > 0
                else float(np.mean(r_train))
            )
            per_fold_train_estimates[v] = {
                "psi": psi_train,
                "mean_w": float(np.mean(w_train)) if w_train.size else 1.0,
            }

            # Inner fold IDs for SIMCal's internal CV (subset of original fold_ids)
            if fold_ids is not None:
                inner_folds_train = fold_ids[train_mask]
                # Renumber to be contiguous
                unique_inner = np.unique(inner_folds_train)
                fold_map = {old: new for new, old in enumerate(unique_inner)}
                inner_folds_train = np.array([fold_map[f] for f in inner_folds_train])
            else:
                inner_folds_train = None

            # Fit SIMCal on training folds
            sim = SIMCalibrator(cfg)
            try:
                sim.fit(w_train, s_train, rewards=r_train, fold_ids=inner_folds_train)

                # Apply to test fold
                w_test = raw_weights[test_mask]
                s_test = ordering_index[test_mask]
                w_test_cal = sim.predict(w_test, s_test)

                # Store calibrated weights for this fold
                calibrated_weights[test_mask] = w_test_cal
                per_fold_test_weights[v] = w_test_cal

                # Collect diagnostics
                mixture_weights_list.append(sim._mixture_weights)
                gamma_list.append(sim._gamma)

            except Exception as e:
                logger.warning(
                    f"Outer fold {v} calibration failed: {e}, using raw weights"
                )
                calibrated_weights[test_mask] = raw_weights[test_mask]
                per_fold_test_weights[v] = raw_weights[test_mask]

        # Ensure mean-one normalization
        calibrated_weights = calibrated_weights / calibrated_weights.mean()

        # Build aggregated info dict (JSON-safe)
        # Filter out None values from mixture_weights_list
        valid_weights = [w for w in mixture_weights_list if w is not None]
        info = {
            "mixture_weights": (
                np.mean(valid_weights, axis=0).tolist() if valid_weights else [1.0]
            ),
            "candidates": (
                ["increasing", "decreasing"]
                if not self.include_baseline
                else ["baseline", "increasing", "decreasing"]
            ),
            "gamma": float(np.mean(gamma_list)) if gamma_list else 0.0,
            "var_before": float(np.var(raw_weights)),
            "var_after": float(np.var(calibrated_weights)),
            # ESS as fraction (0-1) - normalize to mean-one first for correct formula
            "ess_before_frac": float(
                1.0 / (1.0 + np.var(raw_weights / max(raw_weights.mean(), 1e-12)))
            ),
            "ess_after_frac": float(
                1.0
                / (
                    1.0
                    + np.var(calibrated_weights / max(calibrated_weights.mean(), 1e-12))
                )
            ),
            # ESS as count - actual effective sample size
            "ess_before": float((np.sum(raw_weights) ** 2) / np.sum(raw_weights**2)),
            "ess_after": float(
                (np.sum(calibrated_weights) ** 2) / np.sum(calibrated_weights**2)
            ),
            "n_outer_folds": self.n_outer_folds,
            "outer_cv": True,
            "baseline_shrink": self.baseline_shrink,
            # Cache for honest IF computation (convert to lists for JSON safety)
            "outer_fold_ids": outer_fold_ids.tolist(),
            "per_fold_train_estimates": per_fold_train_estimates,
            "per_fold_test_weights": {
                v: w.tolist() for v, w in per_fold_test_weights.items()
            },
        }

        return calibrated_weights, info

    def _build_honest_train_IF(
        self,
        w_train: np.ndarray,
        R_train: np.ndarray,
        psi_train: float,
        mean_w_train: float,
    ) -> np.ndarray:
        """Build training influence functions using training estimates.

        Args:
            w_train: Training weights
            R_train: Training rewards (OOF if available)
            psi_train: Training fold estimate
            mean_w_train: Training fold mean weight

        Returns:
            Training influence functions
        """
        denom = mean_w_train if mean_w_train > 0 else 1.0
        result = (
            w_train * (R_train - psi_train) - psi_train * (w_train - mean_w_train)
        ) / denom
        return np.asarray(result)

    def _compute_quick_rsq(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Quick in-sample R² computation (no CV needed).

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            R-squared value between 0 and 1
        """
        var_y = np.var(y_true, ddof=1)
        if var_y <= 0:
            return 0.0
        var_resid = np.var(y_true - y_pred, ddof=1)
        return float(max(0, 1 - var_resid / var_y))

    def estimate(self) -> EstimationResult:
        """Compute estimates for all target policies with diagnostics."""
        self._validate_fitted()

        estimates = []
        standard_errors = []
        n_samples_used = {}
        influence_functions = {}
        df_info = {}  # Track degrees of freedom per policy

        # Compute estimates for each policy
        for policy in self.sampler.target_policies:
            if policy in self._no_overlap_policies:
                # No overlap - return NaN
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                logger.warning(f"Policy '{policy}' has no overlap - returning NaN")
                continue

            # Get calibrated weights
            weights = self._weights_cache.get(policy)
            if weights is None:
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            # Get rewards
            data = self.sampler.get_data_for_policy(policy)
            if data is None or len(data) == 0:
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            # Extract rewards from the list of dictionaries
            rewards = np.array([d["reward"] for d in data])
            n = len(rewards)
            n_samples_used[policy] = n

            # SAFETY CHECK: Refuse to provide unreliable estimates
            # Fail fast and clearly

            # Check effective sample size
            ess = np.sum(weights) ** 2 / np.sum(weights**2) / n

            # Check weight concentration: What fraction of total weight is on top 5% of samples?
            sorted_weights = np.sort(weights)[::-1]
            top_5pct_count = max(1, int(0.05 * n))
            top_5pct_weight = np.sum(sorted_weights[:top_5pct_count]) / np.sum(weights)

            # Check raw weights for hidden problems (calibration can mask issues)
            raw_weights = self.get_raw_weights(policy)
            raw_near_zero = 0.0
            if raw_weights is not None:
                raw_near_zero = float(np.sum(raw_weights < 1e-10) / len(raw_weights))

            # Coefficient of variation as additional check
            cv_weights = (
                np.std(weights) / np.mean(weights)
                if np.mean(weights) > 0
                else float("inf")
            )

            # Refuse if multiple indicators suggest unreliability
            # We use percentage-based gates because they measure distribution overlap quality,
            # not just statistical power. Poor overlap means the estimate is dominated by
            # a small subset of data, making it practically unreliable even if statistically valid.
            refuse = False
            reasons = []

            if ess < 0.30:  # Less than 30% effective sample size
                refuse = True
                reasons.append(f"ESS={ess:.1%}")

            if raw_near_zero > 0.85:  # More than 85% of raw weights near zero
                refuse = True
                reasons.append(f"raw_near_zero={raw_near_zero:.1%}")

            if (
                top_5pct_weight > 0.30 and cv_weights > 2.0
            ):  # High concentration AND high variability
                refuse = True
                reasons.append(f"top_5%={top_5pct_weight:.1%} with CV={cv_weights:.1f}")

            if refuse:
                # Build warning message that clarifies calibrated vs raw overlap issues
                if raw_near_zero > 0.85 and ess > 0.30:
                    # Calibration helped but raw overlap is poor
                    warning_msg = (
                        f"Policy '{policy}' has poor raw overlap ({raw_near_zero:.1%} of raw weights near-zero) "
                        f"despite calibration improving ESS to {ess:.1%}. "
                        f"Estimates may be dominated by a small subset of samples. "
                    )
                else:
                    # Low ESS or other issues
                    warning_msg = (
                        f"Policy '{policy}' has poor overlap: ESS fraction = {ess:.1%}. "
                        f"Estimates may be dominated by a small subset of samples. "
                    )

                warning_msg += f"Reasons: {', '.join(reasons)}. "

                # Provide context-appropriate solutions
                if self.suppress_overlap_warnings:
                    # Being used internally by DR - no warning needed
                    pass
                else:
                    # Standalone IPS usage - suggest DR
                    warning_msg += (
                        "Solutions: (1) Collect fresh generations from target policy and use DR methods, "
                        "(2) Use policies with better overlap, "
                        "(3) Collect data from more diverse base policies."
                    )

                if self.refuse_unreliable:
                    logger.error(f"Cannot reliably estimate {warning_msg}")
                    estimates.append(np.nan)
                    standard_errors.append(np.nan)
                    influence_functions[policy] = np.full(n, np.nan)
                    continue
                elif not self.suppress_overlap_warnings:
                    # Provide estimate with strong warning (unless suppressed)
                    logger.warning(f"⚠️ UNRELIABLE ESTIMATE: {warning_msg}")

            # ---------- Point estimate (use in-fold rewards) ----------
            # Split estimator into Hajek ratio term + sample-mean augmentation
            # ψ̂_w = (sum w_i R_i) / (sum w_i)  (equal to mean(wR) if weights have mean 1)
            mean_w = float(np.mean(weights))
            psi_w = (
                float(np.sum(weights * rewards) / np.sum(weights))
                if mean_w > 0
                else float("nan")
            )

            # No augmentation - OUA jackknife handles oracle uncertainty via variance
            estimate = psi_w
            estimates.append(estimate)

            # ---------- Influence function (OOF rewards + ratio IF + OOF augmentation) ----------
            # Build fold IDs deterministically (use dataset folds if present)
            try:
                from ..data.folds import get_fold

                n_folds = self.sampler.dataset.metadata.get("n_folds", 5)
                seed = self.sampler.dataset.metadata.get("fold_seed", 42)
                pids = [
                    str(d.get("prompt_id", f"sample_{i}")) for i, d in enumerate(data)
                ]
                fold_ids = np.array(
                    [get_fold(pid, n_folds, seed) for pid in pids], dtype=int
                )
            except Exception:
                fold_ids = self._build_default_folds(n, seed=42)

            # OOF rewards for IF path (prefer index-based)
            R_oof = rewards.copy()
            if self.reward_calibrator is not None:
                try:
                    if hasattr(self.reward_calibrator, "predict_oof_by_index"):
                        # Build mapping checking for duplicates
                        ds_index_by_pid = {}
                        dups = set()
                        for i, s in enumerate(self.sampler.dataset.samples):
                            pid = str(s.prompt_id)
                            if pid in ds_index_by_pid:
                                dups.add(pid)
                            ds_index_by_pid[pid] = i

                        pids = [str(d.get("prompt_id")) for d in data]

                        # Skip index-based if duplicates present
                        if not dups.intersection(pids):
                            ds_idx = np.asarray(
                                [ds_index_by_pid.get(pid, -1) for pid in pids],
                                dtype=int,
                            )
                        else:
                            ds_idx = np.array([-1])  # Force fallback

                        if np.all(ds_idx >= 0):
                            tmp = self.reward_calibrator.predict_oof_by_index(ds_idx)
                            if tmp is not None and len(tmp) > 0:
                                R_oof = np.asarray(tmp, dtype=float)
                    elif hasattr(self.reward_calibrator, "predict_oof"):
                        S_vec = np.asarray(
                            [d.get("judge_score", np.nan) for d in data], dtype=float
                        )
                        tmp = self.reward_calibrator.predict_oof(S_vec, fold_ids)
                        if tmp is not None and len(tmp) > 0:
                            R_oof = np.asarray(tmp, dtype=float)
                except Exception as e:
                    logger.debug(f"OOF reward prediction failed for IF path: {e}")

            # Ensure R_oof has correct shape after calibration attempts
            if len(R_oof) != len(rewards):
                logger.debug(
                    f"R_oof has wrong shape ({len(R_oof)} vs {len(rewards)}), using uncalibrated rewards"
                )
                R_oof = rewards.copy()

            # Check if we should use honest IFs with outer CV
            calib_info = self._calibration_info.get(policy, {})
            if self.use_outer_cv and "outer_fold_ids" in calib_info:
                # Honest influence functions using outer CV structure
                influence = np.zeros(n)
                outer_fold_ids = np.array(
                    calib_info["outer_fold_ids"]
                )  # Convert back from list
                per_fold_train_estimates = calib_info["per_fold_train_estimates"]

                for v in range(self.n_outer_folds):
                    test_mask = outer_fold_ids == v
                    if not np.any(test_mask):
                        continue

                    # Use training estimate and mean weight for this fold
                    train_estimate_dict = per_fold_train_estimates.get(v, {})
                    train_estimate = float(
                        train_estimate_dict.get("psi", psi_w)
                    )  # Fallback to global
                    mean_w_train = float(
                        train_estimate_dict.get("mean_w", mean_w)
                    )  # Use training mean weight

                    # Compute IF for test fold using training estimate and training mean weight
                    w_test = weights[test_mask]
                    r_oof_test = R_oof[test_mask]

                    # Hajek ratio IF with training mean weight (not test mean)
                    denom_train = mean_w_train if mean_w_train > 0 else 1.0

                    influence[test_mask] = (
                        w_test * (r_oof_test - train_estimate)
                        - train_estimate
                        * (w_test - mean_w_train)  # Use train mean weight
                    ) / denom_train
            else:
                # Standard single-pass influence functions
                # Ratio IF for Hajek term (use the same weights used in ψ̂_w)
                # φ^H_i = [w_i (R_oof_i - ψ̂_w) - ψ̂_w (w_i - mean_w)] / mean_w
                # Normalization by mean_w is critical for raw weights; harmless for Hájek (mean≈1)
                denom = mean_w if mean_w > 0 else 1.0
                ratio_if = (
                    weights * (R_oof - psi_w) - psi_w * (weights - mean_w)
                ) / denom

                # No augmentation - OUA jackknife handles oracle uncertainty via variance
                influence = ratio_if

            # IIC removed - use influence functions directly

            # Optional: Center influence functions for numerical stability
            influence = influence - np.mean(influence)

            # Compute standard error from the (possibly residualized) influence functions
            # Use cluster-robust SE when outer CV is used
            se = float(np.std(influence, ddof=1) / np.sqrt(n))  # fallback

            try:
                calib_info = self._calibration_info.get(policy, {})
                if self.use_outer_cv and "outer_fold_ids" in calib_info:
                    from ..diagnostics.robust_inference import cluster_robust_se

                    outer_fold_ids = np.asarray(calib_info["outer_fold_ids"], dtype=int)

                    # Use cluster-robust SE with t-based CI
                    res = cluster_robust_se(
                        data=influence,
                        cluster_ids=outer_fold_ids,
                        statistic_fn=lambda x: np.mean(x),
                        influence_fn=lambda x: x,  # IF already provided and centered
                        alpha=0.05,
                    )
                    se = res["se"]

                    # Store degrees of freedom for this policy
                    df_cluster = res.get("df", n - 1)

                    # If OUA was applied, get oracle DF and take minimum
                    df_final = df_cluster
                    if self.oua_jackknife and self.reward_calibrator is not None:
                        try:
                            if hasattr(
                                self.reward_calibrator, "get_fold_models_for_oua"
                            ):
                                fold_models = (
                                    self.reward_calibrator.get_fold_models_for_oua()
                                )
                                if fold_models:
                                    K = len(fold_models)
                                    df_oracle = K - 1
                                    df_final = min(df_cluster, df_oracle)
                        except Exception as e:
                            logger.debug(f"Could not get oracle DF for {policy}: {e}")

                    # Ensure DF is at least 1
                    df_final = max(df_final, 1)

                    # Store DF info
                    from scipy import stats

                    t_crit = stats.t.ppf(1 - 0.05 / 2, df_final)
                    df_info[policy] = {
                        "df": int(df_final),
                        "t_critical": float(t_crit),
                        "n_clusters": int(res.get("n_clusters", n)),
                    }

                    logger.debug(
                        f"Using cluster-robust SE for {policy}: "
                        f"naive={np.std(influence, ddof=1) / np.sqrt(n):.6f}, "
                        f"robust={se:.6f}, "
                        f"n_clusters={res['n_clusters']}, df={df_final}"
                    )
            except Exception as e:
                logger.debug(f"Cluster-robust SE failed for {policy}: {e}")

            standard_errors.append(se)

            # Oracle augmentation removed - diagnostics no longer needed

            # Store influence functions (always needed for proper inference)
            influence_functions[policy] = influence

        # Store influence functions for later use
        self._influence_functions = influence_functions

        # Create result with clean separation of concerns
        result = EstimationResult(
            estimates=np.array(estimates),
            standard_errors=np.array(standard_errors),
            n_samples_used=n_samples_used,
            method="calibrated_ips" if self.calibrate_weights else "raw_ips",
            influence_functions=influence_functions,
            diagnostics=None,  # Will be set below
            metadata={
                "target_policies": list(self.sampler.target_policies),
                "calibration_method": "simcal" if self.calibrate_weights else None,
                "ess_floor": self.ess_floor,
                "var_cap": self.var_cap,
                "calibration_info": self._calibration_info,  # TODO: Move to diagnostics
                "degrees_of_freedom": (
                    df_info if df_info else None
                ),  # Store DF per policy
            },
        )

        # Add calibration-floor metrics (logged only) per policy
        try:
            cal_info = getattr(self.sampler.dataset, "metadata", {}).get(
                "calibration_info", {}
            )
            f_min = float(cal_info.get("f_min", float("nan")))
            eps = 1e-6
            floor_meta: Dict[str, Dict[str, float]] = {}
            for policy in self.sampler.target_policies:
                data = self.sampler.get_data_for_policy(policy)
                if not data:
                    continue
                rewards = np.array([d["reward"] for d in data], dtype=float)
                if np.isfinite(f_min):
                    floor_mass_logged = float(np.mean(np.abs(rewards - f_min) <= eps))
                else:
                    floor_mass_logged = float("nan")
                floor_meta[policy] = {
                    "f_min": f_min,
                    "floor_mass_logged": floor_mass_logged,
                }
            # Attach to metadata
            if isinstance(result.metadata, dict):
                result.metadata["calibration_floor"] = floor_meta
        except Exception:
            pass

        # Optionally add oracle-uncertainty jackknife variance
        # Apply OUA jackknife using base class method
        self._apply_oua_jackknife(result)

        # Build and attach diagnostics directly
        diagnostics = self._build_diagnostics(result)
        result.diagnostics = diagnostics
        self._diagnostics = diagnostics

        # Attach compact core summary for empirical analysis (no UX change)
        try:
            core_summary: Dict[str, Dict[str, Any]] = {}
            ess = diagnostics.ess_per_policy if diagnostics else {}
            tails = getattr(diagnostics, "tail_indices", None) or {}
            hell_all = getattr(diagnostics, "hellinger_affinity", None)
            hell_per = getattr(diagnostics, "hellinger_per_policy", None) or {}
            cal_floor = (
                result.metadata.get("calibration_floor", {})
                if isinstance(result.metadata, dict)
                else {}
            )
            for policy in self.sampler.target_policies:
                core_summary[policy] = {
                    "ess_fraction": float(ess.get(policy, 0.0)) if ess else None,
                    "tail_index": (
                        float(tails[policy])
                        if policy in tails and tails[policy] is not None
                        else None
                    ),
                    "hellinger_affinity": (
                        float(hell_per[policy])
                        if policy in hell_per and hell_per[policy] is not None
                        else (float(hell_all) if hell_all is not None else None)
                    ),
                }
                if policy in cal_floor:
                    core_summary[policy].update(cal_floor[policy])
            if isinstance(result.metadata, dict):
                result.metadata["core_summary"] = core_summary
        except Exception:
            pass

        # Store for later access
        self._results = result

        return result

    def _build_default_folds(self, n: int, seed: int = 42) -> np.ndarray:
        """Create deterministic K-fold IDs when dataset folds are absent.

        Args:
            n: Number of samples
            seed: Random seed for fold assignment

        Returns:
            Array of fold IDs (0 to K-1)
        """
        try:
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            fold_ids = np.zeros(n, dtype=int)
            for i, (_, te) in enumerate(kf.split(np.arange(n))):
                fold_ids[te] = i
            return fold_ids
        except Exception:
            # Fallback to simple modulo if sklearn not available
            return np.arange(n) % 5

    def get_oracle_jackknife(self, policy: str) -> Optional[np.ndarray]:
        """Leave-one-oracle-fold jackknife estimates for IPS.

        For each reward_calibrator fold model f^(−k), recompute the IPS estimate using
        rewards R^(−k) = f^(−k)(S) and the same calibrated weights, and include
        oracle augmentation with the updated residuals.

        Returns an array of K estimates, or None if not applicable.
        """
        try:
            if self.reward_calibrator is None:
                return None

            # Use the unified method to get fold models
            if not hasattr(self.reward_calibrator, "get_fold_models_for_oua"):
                if self.oua_jackknife:
                    raise ValueError(
                        "OUA jackknife is enabled but reward calibrator doesn't support it. "
                        "Ensure calibrate_dataset() uses enable_cross_fit=True."
                    )
                return None

            fold_models = self.reward_calibrator.get_fold_models_for_oua()

            if not fold_models:
                if self.oua_jackknife:
                    logger.warning(
                        "OUA jackknife is enabled but no fold models available. "
                        "This may happen if calibration mode doesn't support cross-fitting."
                    )
                return None

            # Get required data
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                return None
            weights = self.get_weights(policy)
            if weights is None:
                return None

            judge_scores = np.array([d.get("judge_score") for d in data], dtype=float)

            # Sanity check alignment
            if len(judge_scores) != len(weights):
                return None

            jack: List[float] = []
            for fold_id, fold_model in fold_models.items():
                # Recompute rewards under leave-one-fold reward_calibrator
                # For FlexibleCalibrator two-stage, fold_model is already IsotonicRegression
                # For FlexibleCalibrator monotone or standard isotonic, it's IsotonicRegression
                rewards_loo = np.clip(fold_model.predict(judge_scores), 0.0, 1.0)

                # OUA jackknife: only recompute with different calibrator, no bias augmentation
                contrib = weights * rewards_loo
                jack.append(float(np.mean(contrib)))

            return np.asarray(jack, dtype=float) if jack else None
        except Exception as e:
            logger.debug(f"get_oracle_jackknife failed for {policy}: {e}")
            return None

    def get_calibration_info(self, target_policy: str) -> Optional[Dict]:
        """Get calibration information for a policy.

        Args:
            target_policy: Name of target policy

        Returns:
            Dictionary with calibration details or None
        """
        return self._calibration_info.get(target_policy)

    def _build_diagnostics(self, result: EstimationResult) -> IPSDiagnostics:
        """Build simplified diagnostics for this estimation.

        Args:
            result: The estimation result

        Returns:
            IPSDiagnostics object
        """
        # Get dataset info
        dataset = getattr(self.sampler, "dataset", None) or getattr(
            self.sampler, "_dataset", None
        )
        n_total = 0
        if dataset:
            n_total = (
                dataset.n_samples
                if hasattr(dataset, "n_samples")
                else len(dataset.samples)
            )

        # Build estimates dict
        estimates_dict = {}
        se_dict = {}
        policies = list(self.sampler.target_policies)
        for i, policy in enumerate(policies):
            if i < len(result.estimates):
                estimates_dict[policy] = float(result.estimates[i])
                se_dict[policy] = float(result.standard_errors[i])

        # Compute weight diagnostics
        ess_per_policy = {}
        max_weight_per_policy = {}
        tail_indices = {}
        status_per_policy = {}
        hellinger_per_policy = {}  # New: Hellinger affinity per policy
        overall_ess = 0.0
        total_n = 0

        for policy in policies:
            weights = self.get_weights(policy)
            if weights is not None and len(weights) > 0:
                w_diag = compute_weight_diagnostics(weights, policy, compute_hill=True)
                ess_per_policy[policy] = w_diag["ess_fraction"]
                max_weight_per_policy[policy] = w_diag["max_weight"]
                status_per_policy[policy] = w_diag["status"]  # Store per-policy status

                # Hill tail index is now computed in compute_weight_diagnostics
                if "tail_index" in w_diag:
                    tail_indices[policy] = w_diag["tail_index"]
                else:
                    tail_indices[policy] = None

                # Compute Hellinger affinity for this policy (use raw weights)
                raw_weights = self.get_raw_weights(policy)
                if raw_weights is not None and len(raw_weights) > 0:
                    from ..diagnostics.overlap import hellinger_affinity

                    hellinger_per_policy[policy] = hellinger_affinity(raw_weights)

                # Track overall
                n = len(weights)
                overall_ess += w_diag["ess_fraction"] * n
                total_n += n

        # Compute overall weight ESS
        weight_ess = overall_ess / total_n if total_n > 0 else 0.0

        # Compute overall Hellinger affinity (average across policies)
        overall_hellinger = None
        overlap_quality = None
        if hellinger_per_policy:
            overall_hellinger = float(np.mean(list(hellinger_per_policy.values())))
            # Determine overlap quality based on Hellinger
            if overall_hellinger < 0.20:
                overlap_quality = "catastrophic"
            elif overall_hellinger < 0.35:
                overlap_quality = "poor"
            elif overall_hellinger < 0.50:
                overlap_quality = "marginal"
            else:
                overlap_quality = "good"

        # Determine status based on ESS, Hellinger, and tail indices
        worst_tail_idx = min(
            (idx for idx in tail_indices.values() if idx is not None),
            default=float("inf"),
        )

        # Include Hellinger in status determination
        if overlap_quality == "catastrophic" or weight_ess < 0.01:
            weight_status = Status.CRITICAL
        elif worst_tail_idx < 1.5:  # Very heavy tails
            weight_status = Status.CRITICAL
        elif overlap_quality == "poor" or weight_ess < 0.1:
            weight_status = Status.WARNING
        elif worst_tail_idx < 2.0:  # Heavy tails (infinite variance)
            weight_status = Status.WARNING
        else:
            weight_status = Status.GOOD

        # Get calibration info if available
        calibration_rmse = None
        calibration_r2 = None
        n_oracle_labels = None

        # If dataset has calibration info in metadata
        if dataset and hasattr(dataset, "metadata"):
            cal_info = dataset.metadata.get("calibration_info", {})
            calibration_rmse = cal_info.get("rmse")
            calibration_r2 = cal_info.get("r2")  # May be None if not computed
            n_oracle_labels = cal_info.get("n_oracle")

        # Store tail indices in result metadata
        if tail_indices:
            result.metadata["tail_indices"] = tail_indices

        # Create IPSDiagnostics with new overlap metrics
        diagnostics = IPSDiagnostics(
            estimator_type="CalibratedIPS",
            method="calibrated_ips" if self.calibrate_weights else "raw_ips",
            n_samples_total=n_total,
            n_samples_valid=self.sampler.n_valid_samples,
            n_policies=len(policies),
            policies=policies,
            estimates=estimates_dict,
            standard_errors=se_dict,
            n_samples_used=result.n_samples_used,
            weight_ess=weight_ess,
            weight_status=weight_status,
            ess_per_policy=ess_per_policy,
            max_weight_per_policy=max_weight_per_policy,
            status_per_policy=status_per_policy,
            tail_indices=tail_indices,  # Use Hill indices instead of tail ratios
            # New overlap metrics
            hellinger_affinity=overall_hellinger,
            hellinger_per_policy=hellinger_per_policy if hellinger_per_policy else None,
            overlap_quality=overlap_quality,
            # Calibration fields
            calibration_rmse=calibration_rmse,
            calibration_r2=calibration_r2,
            n_oracle_labels=n_oracle_labels,
        )

        return diagnostics

    def get_diagnostics(self) -> Optional[IPSDiagnostics]:
        """Get the diagnostics object."""
        return self._diagnostics
