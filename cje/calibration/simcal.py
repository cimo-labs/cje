"""Stacked Surrogate-Indexed Monotone Calibration (SIMCal) for importance weights.

This module implements stacked SIMCal, which combines {increasing, decreasing}
isotonic candidates via convex optimization to minimize out-of-fold (OOF)
influence function variance.

The stacking approach:
1. Builds candidate weight vectors (isotonic increasing/decreasing, optionally baseline)
2. Computes OOF influence functions for each candidate
3. Solves a quadratic program on the simplex to find optimal mixture
4. Applies uniform blending to satisfy ESS/variance constraints
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List, cast
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
import warnings


@dataclass
class SimcalConfig:
    """Configuration for stacked SIMCal calibration.

    Stacked SIMCal combines multiple candidate weight vectors (baseline,
    increasing, decreasing) to minimize OOF influence function variance,
    then applies uniform blending to meet ESS/variance constraints.

    Args:
        ess_floor: Minimum ESS as fraction of n (e.g., 0.2 => ESS >= 0.2 * n)
        var_cap: Maximum allowed variance of calibrated weights (default 1.0 = no variance increase)
        epsilon: Small constant for numerical stability
        include_baseline: Whether to include raw weights in the stack (default True)
        ridge_lambda: Ridge regularization for covariance matrix (default 1e-8)
        n_folds: Number of folds for OOF if fold_ids not provided (default 5)
        baseline_shrink: Shrinkage toward baseline for stability (default 0.05)
        random_seed: Random seed for KFold when fold_ids not provided (default 42)
    """

    ess_floor: Optional[float] = 0.2
    var_cap: Optional[float] = 1.0
    epsilon: float = 1e-9
    include_baseline: bool = False  # Default OFF - isotonic usually sufficient
    ridge_lambda: float = 1e-8
    n_folds: int = 5
    baseline_shrink: float = 0.0
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.ess_floor is not None and not (0 < self.ess_floor <= 1):
            raise ValueError(f"ess_floor must be in (0, 1], got {self.ess_floor}")
        if self.var_cap is not None and self.var_cap <= 0:
            raise ValueError(f"var_cap must be positive, got {self.var_cap}")
        if self.baseline_shrink < 0 or self.baseline_shrink > 1:
            raise ValueError(
                f"baseline_shrink must be in [0, 1], got {self.baseline_shrink}"
            )

        # Validate consistency between ess_floor and var_cap
        if self.ess_floor is not None and self.var_cap is not None:
            # ESS = n/(1 + Var) implies Var <= 1/ess_floor - 1
            implied_var_cap = (1.0 / self.ess_floor) - 1.0
            if self.var_cap > implied_var_cap:
                warnings.warn(
                    f"var_cap={self.var_cap:.3f} is looser than ESS-implied cap "
                    f"{implied_var_cap:.3f} from ess_floor={self.ess_floor}. "
                    f"The ESS constraint will dominate.",
                    UserWarning,
                )


class SIMCalibrator:
    """Stacked Score-Indexed Monotone Calibrator.

    Combines {increasing, decreasing} candidates (optionally baseline) to minimize
    OOF influence function variance, then applies uniform blending to
    meet ESS/variance constraints.
    """

    def __init__(self, config: SimcalConfig):
        """Initialize SIMCalibrator with configuration.

        Args:
            config: SimcalConfig with calibration parameters
        """
        self.cfg = config

        # Learned state for fit/predict
        self._fitted = False
        self._isotonic_models: Dict[str, Any] = {}  # Stores learned isotonic models
        self._mixture_weights: Optional[np.ndarray] = None
        self._score_range: Optional[Tuple[float, float]] = None
        self._gamma: float = 0.0
        self._training_info: Dict[str, Any] = {}

    @staticmethod
    def implied_var_cap(ess_floor: float) -> float:
        """Compute the implied variance cap from an ESS floor constraint.

        Since ESS = n/(1 + Var), requiring ESS >= ess_floor * n
        implies Var <= 1/ess_floor - 1.

        Args:
            ess_floor: Minimum ESS as fraction of n (must be in (0, 1])

        Returns:
            Maximum allowed variance to satisfy the ESS constraint
        """
        if not (0 < ess_floor <= 1):
            raise ValueError(f"ess_floor must be in (0, 1], got {ess_floor}")
        return (1.0 / ess_floor) - 1.0

    def fit_transform(
        self,
        w: np.ndarray,
        s: np.ndarray,
        *,
        rewards: Optional[np.ndarray] = None,
        residuals: Optional[np.ndarray] = None,
        fold_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fit and transform in one step (original behavior).

        This is more efficient than fit() + predict() when you don't need
        to apply the calibration to new data.

        Args:
            w: Raw importance weights
            s: Score index for ordering
            rewards: Optional rewards for IPS IF
            residuals: Optional residuals for DR IF
            fold_ids: Optional fold assignments

        Returns:
            Tuple of (calibrated weights, info dict)
        """
        # Fit the model
        self.fit(w, s, rewards=rewards, residuals=residuals, fold_ids=fold_ids)

        # Apply to same data
        w_final = self.predict(w, s)

        # Build info dict (for backward compatibility)
        # Check for kinks
        boundary_alpha = (
            any(alpha < 1e-6 for alpha in self._mixture_weights)
            if self._mixture_weights is not None
            else False
        )
        cap_active = self._gamma > 0

        info = {
            "mixture_weights": (
                self._mixture_weights.tolist()
                if self._mixture_weights is not None
                else []
            ),
            "candidates": self._training_info["candidates"],
            "gamma": self._gamma,
            "var_before": float(np.var(w / w.mean())),
            "var_after": float(np.var(w_final)),
            "ess_before": len(w) / (1 + np.var(w / w.mean())),
            "ess_after": len(w) / (1 + np.var(w_final)),
            "oof_variance_reduction": float(1.0),  # Placeholder
            "if_type": self._training_info["if_type"],
            "n_folds": self.cfg.n_folds,
            "baseline_shrink": self.cfg.baseline_shrink,
            # Kink diagnostics
            "boundary_alpha": boundary_alpha,
            "cap_active": cap_active,
            "kinky": boundary_alpha or cap_active,
        }

        return w_final, info

    def transform(
        self,
        w: np.ndarray,
        s: np.ndarray,
        *,
        rewards: Optional[np.ndarray] = None,
        residuals: Optional[np.ndarray] = None,
        fold_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calibrate weights using stacked surrogate-indexed monotone projection.

        This method maintains backward compatibility by delegating to fit_transform().
        For new code, consider using fit() and predict() separately if you need to
        apply the calibration to multiple datasets.

        Algorithm:
        1. Build candidate weight vectors: {increasing, decreasing, baseline?}
        2. Compute OOF influence functions for each candidate
        3. Solve quadratic program to find optimal mixture on simplex
        4. Apply single γ-blend toward uniform for constraints
        5. Optional: Apply baseline shrinkage for stability

        Args:
            w: Raw importance weights (must be positive, will be normalized to mean 1)
            s: Score index (e.g., judge scores) for ordering
            rewards: Rewards for IPS influence functions (optional, uses weights only if None)
            residuals: DR residuals (R - g_oof(S)) for DR influence functions
            fold_ids: Pre-assigned fold IDs for OOF computation (optional)

        Returns:
            Tuple of (calibrated_weights, info_dict) where info_dict contains:
                - mixture_weights: Optimal convex combination weights
                - candidates: Names of candidate weight vectors
                - gamma: Uniform blending parameter for constraints
                - var_before: Variance of input weights
                - var_after: Final variance after all adjustments
                - ess_before: ESS of input weights
                - ess_after: Final ESS after all adjustments
                - oof_variance_reduction: Ratio of stacked to best single candidate

        Raises:
            ValueError: If weights contain non-positive, NaN, or infinite values
        """
        # Delegate to fit_transform for backward compatibility
        return self.fit_transform(
            w, s, rewards=rewards, residuals=residuals, fold_ids=fold_ids
        )

    def _solve_simplex_qp(self, Sigma: np.ndarray) -> np.ndarray:
        """Solve quadratic program on simplex using active set method.

        Minimize π^T Σ π subject to π ≥ 0, 1^T π = 1

        Args:
            Sigma: K x K positive semi-definite covariance matrix

        Returns:
            Optimal mixture weights on simplex
        """
        K = Sigma.shape[0]

        # Start with uniform weights
        active_set = set(range(K))

        max_iterations = 20
        for _ in range(max_iterations):
            # Solve on current active set
            if len(active_set) == 0:
                # Degenerate case - return uniform
                return np.ones(K) / K

            # Build reduced system
            active_idx = sorted(active_set)
            Sigma_active = Sigma[np.ix_(active_idx, active_idx)]

            # Solve equality-constrained QP: min π^T Σ π s.t. 1^T π = 1
            # Solution: π = Σ^{-1} 1 / (1^T Σ^{-1} 1)
            try:
                ones = np.ones(len(active_idx))
                Sigma_inv_ones = np.linalg.solve(Sigma_active, ones)
                denom = np.dot(ones, Sigma_inv_ones)

                if abs(denom) < 1e-10:
                    # Near-singular - use uniform on active set
                    pi_active = ones / len(active_idx)
                else:
                    pi_active = Sigma_inv_ones / denom
            except np.linalg.LinAlgError:
                # Singular - use uniform on active set
                pi_active = np.ones(len(active_idx)) / len(active_idx)

            # Check for negative weights
            if np.all(pi_active >= -1e-10):
                # Feasible - construct full solution
                pi_full = np.zeros(K)
                for i, idx in enumerate(active_idx):
                    pi_full[idx] = max(0, pi_active[i])

                # Renormalize to ensure exact sum to 1
                pi_full = pi_full / pi_full.sum()
                return cast(np.ndarray, pi_full)

            # Remove most negative from active set
            min_idx = np.argmin(pi_active)
            active_set.remove(active_idx[min_idx])

        # Fallback to uniform if no convergence
        return np.ones(K) / K

    def _apply_constraints(self, w: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply ESS/variance constraints via uniform blending.

        Args:
            w: Mean-one weight vector

        Returns:
            Tuple of (constrained_weights, gamma) where gamma is the blending parameter
        """
        n = len(w)
        var_w = np.var(w)
        gamma = 0.0

        if var_w > 0:
            # Check ESS constraint
            if self.cfg.ess_floor is not None:
                var_max_ess = (1.0 / self.cfg.ess_floor) - 1.0
                if var_w > var_max_ess:
                    gamma = max(gamma, 1.0 - np.sqrt(var_max_ess / var_w))

            # Check variance cap
            if self.cfg.var_cap is not None and var_w > self.cfg.var_cap:
                gamma = max(gamma, 1.0 - np.sqrt(self.cfg.var_cap / var_w))

        gamma = float(np.clip(gamma, 0.0, 1.0))

        # Apply blending: w ← 1 + (1-γ)(w-1)
        w_constrained = 1.0 + (1.0 - gamma) * (w - 1.0)
        w_constrained = np.maximum(w_constrained, self.cfg.epsilon)
        w_constrained = w_constrained / w_constrained.mean()

        return w_constrained, gamma

    def fit(
        self,
        w: np.ndarray,
        s: np.ndarray,
        *,
        rewards: Optional[np.ndarray] = None,
        residuals: Optional[np.ndarray] = None,
        fold_ids: Optional[np.ndarray] = None,
    ) -> "SIMCalibrator":
        """Fit SIMCal on training data.

        Learns isotonic regression models and mixture weights.

        Args:
            w: Raw importance weights (will be normalized to mean 1)
            s: Score index for ordering
            rewards: Rewards for IPS influence functions
            residuals: DR residuals for DR influence functions
            fold_ids: Fold assignments for cross-fitting

        Returns:
            self (for chaining)
        """
        # Input validation
        w = np.asarray(w, dtype=float)
        s = np.asarray(s, dtype=float)

        if len(w) != len(s):
            raise ValueError(f"Length mismatch: weights={len(w)}, scores={len(s)}")

        if not np.all(np.isfinite(w)) or not np.all(np.isfinite(s)):
            raise ValueError("NaNs or infinities in inputs")

        if np.any(w <= 0):
            raise ValueError("Weights must be positive")

        # Normalize weights
        w = w / w.mean()
        n = len(w)

        # Store score range for extrapolation handling
        self._score_range = (float(np.min(s)), float(np.max(s)))

        # Build candidates and store isotonic models
        self._isotonic_models = {}
        candidates = []
        candidate_names = []

        # 1. Baseline (if enabled)
        if self.cfg.include_baseline:
            candidates.append(w.copy())
            candidate_names.append("baseline")
            self._isotonic_models["baseline"] = None  # No model, just raw weights

        # 2. Isotonic increasing
        iso_inc = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso_inc.fit(s, w)
        w_inc = iso_inc.predict(s)
        w_inc = np.maximum(w_inc, self.cfg.epsilon)
        w_inc = w_inc / w_inc.mean()
        candidates.append(w_inc)
        candidate_names.append("increasing")
        self._isotonic_models["increasing"] = iso_inc

        # 3. Isotonic decreasing
        iso_dec = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso_dec.fit(s, w)
        w_dec = iso_dec.predict(s)
        w_dec = np.maximum(w_dec, self.cfg.epsilon)
        w_dec = w_dec / w_dec.mean()
        candidates.append(w_dec)
        candidate_names.append("decreasing")
        self._isotonic_models["decreasing"] = iso_dec

        K = len(candidates)

        # Determine influence function targets
        if residuals is not None:
            if_targets = residuals
            if_type = "dr"
        elif rewards is not None:
            if_targets = rewards
            if_type = "ips"
        else:
            if_targets = np.ones(n)
            if_type = "weight"

        # Generate fold IDs if not provided
        if fold_ids is None:
            from sklearn.model_selection import KFold

            kf = KFold(
                n_splits=self.cfg.n_folds,
                shuffle=True,
                random_state=self.cfg.random_seed,
            )
            fold_ids = np.zeros(n, dtype=int)
            for fold_idx, (_, test_idx) in enumerate(kf.split(np.arange(n))):
                fold_ids[test_idx] = fold_idx

        # Compute OOF influence matrix using Hájek ratio form
        IF_matrix = np.zeros((n, K))
        for k, w_cand in enumerate(candidates):
            for fold_id in range(self.cfg.n_folds):
                test_mask = fold_ids == fold_id
                train_mask = ~test_mask

                if np.sum(train_mask) < 2 or not np.any(test_mask):
                    continue

                if residuals is not None:
                    # DR path: residuals should be computed honestly; psi_tr ~ 0
                    psi_tr = 0.0
                    mean_w_tr = np.mean(w_cand[train_mask])
                    denom = max(mean_w_tr, 1e-12)
                    IF_matrix[test_mask, k] = (
                        w_cand[test_mask] * residuals[test_mask]
                        - psi_tr * (w_cand[test_mask] - mean_w_tr)
                    ) / denom
                else:
                    # IPS Hájek path
                    mean_w_tr = np.mean(w_cand[train_mask])
                    psi_tr = np.sum(w_cand[train_mask] * if_targets[train_mask]) / max(
                        np.sum(w_cand[train_mask]), 1e-12
                    )
                    denom = max(mean_w_tr, 1e-12)
                    IF_matrix[test_mask, k] = (
                        w_cand[test_mask] * (if_targets[test_mask] - psi_tr)
                        - psi_tr * (w_cand[test_mask] - mean_w_tr)
                    ) / denom

        # Compute covariance and solve QP
        Sigma = np.cov(IF_matrix.T)

        if self.cfg.ridge_lambda > 0:
            reg_amount = self.cfg.ridge_lambda * np.trace(Sigma) / K
            Sigma = Sigma + reg_amount * np.eye(K)

        self._mixture_weights = self._solve_simplex_qp(Sigma)

        # Compute stacked weights for constraint checking
        w_stacked = np.zeros(n)
        for k, pi_k in enumerate(self._mixture_weights):
            w_stacked += pi_k * candidates[k]

        # Determine gamma for constraints
        _, self._gamma = self._apply_constraints(w_stacked)

        # Store training info
        self._training_info = {
            "candidates": candidate_names,
            "n_train": n,
            "if_type": if_type,
            "score_mean": float(np.mean(s)),
            "score_std": float(np.std(s)),
        }

        self._fitted = True
        return self

    def predict(
        self,
        w: np.ndarray,
        s: np.ndarray,
    ) -> np.ndarray:
        """Apply learned calibration to new data.

        Args:
            w: Raw importance weights to calibrate
            s: Score index for ordering

        Returns:
            Calibrated weights (mean-one normalized)
        """
        if not self._fitted:
            raise ValueError("Must call fit() before predict()")

        # Input validation
        w = np.asarray(w, dtype=float)
        s = np.asarray(s, dtype=float)

        if len(w) != len(s):
            raise ValueError(f"Length mismatch: weights={len(w)}, scores={len(s)}")

        # Normalize input weights
        w = w / w.mean()
        n = len(w)

        # Clip scores to training range and track clip rate
        s_original = s.copy()
        if self._score_range is not None:
            s_clipped = np.clip(s, self._score_range[0], self._score_range[1])
            clip_rate = float(np.mean(s != s_clipped))
            n_clipped_low = int(np.sum(s < self._score_range[0]))
            n_clipped_high = int(np.sum(s > self._score_range[1]))
        else:
            s_clipped = s
            clip_rate = 0.0
            n_clipped_low = 0
            n_clipped_high = 0

        # Store diagnostic info
        self._last_predict_info = {
            "clip_rate": clip_rate,
            "n_clipped_low": n_clipped_low,
            "n_clipped_high": n_clipped_high,
        }

        # Build candidates using learned models
        candidates = []

        for name in self._training_info["candidates"]:
            if name == "baseline":
                # Raw weights
                candidates.append(w.copy())
            elif name in self._isotonic_models:
                # Apply isotonic model
                iso_model = self._isotonic_models[name]
                w_cand = iso_model.predict(s_clipped)
                w_cand = np.maximum(w_cand, self.cfg.epsilon)
                w_cand = w_cand / w_cand.mean()
                candidates.append(w_cand)

        # Apply learned mixture weights
        w_stacked = np.zeros(n)
        if self._mixture_weights is not None:
            for k, pi_k in enumerate(self._mixture_weights):
                w_stacked += pi_k * candidates[k]
        else:
            # Fallback to uniform if no weights learned
            w_stacked = w.copy()

        # Apply constraints using learned gamma
        if self._gamma > 0:
            w_final = 1.0 + (1.0 - self._gamma) * (w_stacked - 1.0)
            w_final = np.maximum(w_final, self.cfg.epsilon)
            w_final = w_final / w_final.mean()
        else:
            w_final = w_stacked

        # Optional baseline shrinkage
        if self.cfg.baseline_shrink > 0:
            w_final = (
                1 - self.cfg.baseline_shrink
            ) * w_final + self.cfg.baseline_shrink * w
            w_final = w_final / w_final.mean()

        return w_final
