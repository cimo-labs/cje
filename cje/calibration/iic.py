"""
Isotonic Influence Control (IIC) for variance reduction.

IIC residualizes influence functions against judge scores to reduce variance
without changing the target estimand. This is a pure variance reduction
technique that operates on the influence function layer.

Theory:
    For any asymptotically linear estimator with IF φ:
    - Fit isotonic regression: ĥ(s) ≈ E[φ|S=s]
    - Residualize: φ̃ = φ - ĥ(S)
    - Properties:
        * E[φ̃] = E[φ] = 0 (unbiased)
        * Var(φ̃) ≤ Var(φ) (variance reduction)
        * Same asymptotic distribution, tighter CIs

This is a key contribution from the CJE paper that provides "free" variance
reduction by exploiting the judge score structure.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
import logging

logger = logging.getLogger(__name__)


@dataclass
class IICConfig:
    """Configuration for Isotonic Influence Control.

    Args:
        enable: Whether to apply IIC (default True - it's free variance reduction!)
        use_cross_fit: Whether to use fold-honest fitting (recommended)
        min_samples_for_iic: Minimum samples to attempt IIC
        compute_diagnostics: Whether to compute R² and other diagnostics
        store_components: Whether to store E[φ|S] for visualization
        use_splines: Whether to use spline regression instead of isotonic (default True)
        n_knots: Number of knots for spline regression
        spline_degree: Degree of spline polynomials
    """

    enable: bool = True  # On by default - no reason not to use it
    use_cross_fit: bool = True
    min_samples_for_iic: int = 50  # Need enough data for regression
    compute_diagnostics: bool = True
    store_components: bool = False  # For debugging/visualization
    use_splines: bool = True  # Use flexible splines by default
    n_knots: int = 8  # Number of spline knots
    spline_degree: int = 3  # Cubic splines


class IsotonicInfluenceControl:
    """Reduce influence function variance via isotonic residualization.

    This implements the IIC component from the CJE paper, providing
    variance reduction by exploiting the relationship between influence
    functions and judge scores.

    The key insight: influence functions often correlate with judge scores
    (since both relate to outcome quality). By removing the predictable
    component E[φ|S], we keep the same estimand but reduce variance.
    """

    def __init__(self, config: Optional[IICConfig] = None):
        """Initialize IIC.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or IICConfig()
        self._diagnostics: Dict[str, Dict[str, Any]] = {}
        self._fitted_components: Dict[str, np.ndarray] = {}

    def residualize(
        self,
        influence: np.ndarray,
        judge_scores: np.ndarray,
        policy: str,
        fold_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Residualize influence function against judge scores.

        This is the main entry point for IIC. It fits E[φ|S] using isotonic
        regression and returns the residuals φ - Ê[φ|S], along with the
        adjustment needed for the point estimate.

        The direction (increasing/decreasing) is chosen automatically based on
        which gives better fit, measured by Spearman correlation.

        Args:
            influence: Raw influence function values (per-sample contributions)
            judge_scores: Judge scores S (ordering index)
            policy: Policy name (for diagnostics)
            fold_ids: Optional fold assignments for cross-fitting
                     (use same folds as reward calibration for consistency)

        Returns:
            residualized_if: φ̃ = φ - Ê[φ|S] (same mean, less variance)
            diagnostics: Dict with R², variance reduction, point_estimate_adjustment, etc.
        """
        if not self.config.enable:
            return influence, {"applied": False, "reason": "disabled"}

        n = len(influence)
        if n < self.config.min_samples_for_iic:
            logger.debug(f"Too few samples ({n}) for IIC")
            return influence, {"applied": False, "reason": "insufficient_samples"}

        # Validate inputs
        if len(judge_scores) != n:
            logger.warning(
                f"Length mismatch: {n} influences, {len(judge_scores)} scores"
            )
            return influence, {"applied": False, "reason": "length_mismatch"}

        # Check for valid values
        if not np.all(np.isfinite(influence)):
            logger.warning(f"Non-finite influence values for {policy}")
            return influence, {"applied": False, "reason": "non_finite_influence"}

        valid_scores = np.isfinite(judge_scores)
        if not np.all(valid_scores):
            # Handle missing judge scores gracefully
            if valid_scores.sum() < self.config.min_samples_for_iic:
                return influence, {
                    "applied": False,
                    "reason": "insufficient_valid_scores",
                }
            # We'll only fit on valid scores

        # Fit E[φ|S] using regression
        if self.config.use_cross_fit and fold_ids is not None:
            fitted_values = self._fit_cross_fitted(
                influence, judge_scores, fold_ids, valid_scores
            )
        else:
            fitted_values = self._fit_global(influence, judge_scores, valid_scores)

        # CRITICAL: Center the fitted values to preserve the mean
        # E[φ̃] = E[φ - (Ê[φ|S] - E[Ê[φ|S]])] = E[φ] - E[Ê[φ|S]] + E[Ê[φ|S]] = E[φ]
        fitted_mean = np.mean(fitted_values)
        fitted_values_centered = fitted_values - fitted_mean

        # Compute residuals: φ̃ = φ - centered(Ê[φ|S])
        residuals = influence - fitted_values_centered

        # CRITICAL: IIC must be variance-only
        # We residualize the influence functions but preserve the mean
        # Therefore, point_estimate_adjustment must always be 0
        point_estimate_adjustment = 0.0

        # Store fitted component if requested (for visualization)
        if self.config.store_components:
            self._fitted_components[policy] = fitted_values_centered

        # Compute diagnostics
        diagnostics = {
            "applied": True,
            "point_estimate_adjustment": point_estimate_adjustment,
        }
        if self.config.compute_diagnostics:
            diagnostics.update(
                self._compute_diagnostics(
                    influence, fitted_values_centered, residuals, policy
                )
            )

        self._diagnostics[policy] = diagnostics

        logger.debug(
            f"IIC for {policy}: R²={diagnostics.get('r_squared', 0):.3f}, "
            f"variance reduction={diagnostics.get('var_reduction', 0):.1%}"
        )

        return residuals, diagnostics

    def _fit_global(
        self, influence: np.ndarray, judge_scores: np.ndarray, valid_mask: np.ndarray
    ) -> np.ndarray:
        """Global fit (simpler but may overfit).

        This fits a single regression model on all data.
        Uses splines by default for flexibility, or isotonic as fallback.
        """
        fitted = np.zeros_like(influence)

        if valid_mask.sum() < 2:
            # Not enough valid data
            return fitted

        if self.config.use_splines:
            # Use spline regression for flexible fit
            n_samples = valid_mask.sum()
            # Adaptive number of knots based on sample size
            n_knots = min(
                self.config.n_knots,
                max(4, n_samples // 20),  # At least 4 knots, at most n_samples/20
            )

            try:
                # Fit spline model
                spline = SplineTransformer(
                    n_knots=n_knots,
                    degree=self.config.spline_degree,
                    include_bias=False,
                )
                ridge = RidgeCV(alphas=np.logspace(-3, 3, 13), store_cv_results=False)
                model = make_pipeline(spline, ridge)

                # Reshape for sklearn
                X = judge_scores[valid_mask].reshape(-1, 1)
                y = influence[valid_mask]

                model.fit(X, y)
                fitted[valid_mask] = model.predict(X)

                # For invalid scores, use mean of influence (no reduction)
                fitted[~valid_mask] = influence[valid_mask].mean()

                # Store model type in diagnostics
                self._last_model_type = "spline"

            except Exception as e:
                logger.warning(f"Spline fitting failed: {e}. Falling back to isotonic.")
                # Fall back to isotonic
                return self._fit_isotonic_fallback(influence, judge_scores, valid_mask)
        else:
            # Use isotonic regression (original behavior)
            return self._fit_isotonic_fallback(influence, judge_scores, valid_mask)

        return fitted

    def _fit_isotonic_fallback(
        self, influence: np.ndarray, judge_scores: np.ndarray, valid_mask: np.ndarray
    ) -> np.ndarray:
        """Fallback to isotonic regression (original behavior)."""
        fitted = np.zeros_like(influence)

        # Determine direction based on Spearman correlation
        from scipy.stats import spearmanr

        corr, _ = spearmanr(judge_scores[valid_mask], influence[valid_mask])

        # Choose direction: increasing if positive correlation, decreasing if negative
        increasing = corr >= 0

        # Fit isotonic regression on valid data with chosen direction
        iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
        iso.fit(judge_scores[valid_mask], influence[valid_mask])

        # Predict for all data
        fitted[valid_mask] = iso.predict(judge_scores[valid_mask])

        # For invalid scores, use mean of influence (no reduction)
        fitted[~valid_mask] = influence[valid_mask].mean()

        # Store direction in diagnostics
        self._last_direction = "increasing" if increasing else "decreasing"
        self._last_correlation = float(corr)
        self._last_model_type = "isotonic"

        return fitted

    def _fit_cross_fitted(
        self,
        influence: np.ndarray,
        judge_scores: np.ndarray,
        fold_ids: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """Fold-honest fit (recommended).

        This prevents overfitting by using out-of-fold predictions.
        Each sample's E[φ|S] is predicted using a model trained on other folds.
        """
        fitted = np.zeros_like(influence)
        unique_folds = np.unique(fold_ids[fold_ids >= 0])

        if len(unique_folds) < 2:
            # Not enough folds for cross-fitting
            logger.debug("Insufficient folds for cross-fitting, using global fit")
            return self._fit_global(influence, judge_scores, valid_mask)

        if self.config.use_splines:
            # Spline-based cross-fitting
            model_types = []

            for fold in unique_folds:
                # Define train and test sets
                train_mask = (fold_ids >= 0) & (fold_ids != fold) & valid_mask
                test_mask = (fold_ids == fold) & valid_mask

                if train_mask.sum() < 10 or test_mask.sum() == 0:
                    # Not enough data in this fold
                    if test_mask.sum() > 0:
                        fitted[test_mask] = (
                            influence[train_mask].mean() if train_mask.sum() > 0 else 0
                        )
                    continue

                n_train = train_mask.sum()
                # Adaptive number of knots based on training size
                n_knots = min(
                    self.config.n_knots,
                    max(4, n_train // 20),  # At least 4 knots, at most n_train/20
                )

                try:
                    # Fit spline model on training folds
                    spline = SplineTransformer(
                        n_knots=n_knots,
                        degree=self.config.spline_degree,
                        include_bias=False,
                    )
                    ridge = RidgeCV(
                        alphas=np.logspace(-3, 3, 13), store_cv_results=False
                    )
                    model = make_pipeline(spline, ridge)

                    X_train = judge_scores[train_mask].reshape(-1, 1)
                    y_train = influence[train_mask]
                    X_test = judge_scores[test_mask].reshape(-1, 1)

                    model.fit(X_train, y_train)
                    fitted[test_mask] = model.predict(X_test)
                    model_types.append("spline")

                except Exception as e:
                    logger.debug(f"Spline fitting failed for fold {fold}: {e}")
                    # Fall back to isotonic for this fold
                    fitted[test_mask] = self._fit_isotonic_single_fold(
                        judge_scores[train_mask],
                        influence[train_mask],
                        judge_scores[test_mask],
                    )
                    model_types.append("isotonic")

            # Store model info for diagnostics
            if model_types:
                from collections import Counter

                model_counts = Counter(model_types)
                self._last_model_type = model_counts.most_common(1)[0][0]
                self._fold_model_types = model_types

        else:
            # Original isotonic cross-fitting
            return self._fit_isotonic_cross_fitted(
                influence, judge_scores, fold_ids, valid_mask
            )

        # Handle samples not in any fold
        unfitted = (fold_ids < 0) | ~valid_mask
        if unfitted.sum() > 0:
            fitted[unfitted] = (
                influence[valid_mask].mean() if valid_mask.sum() > 0 else 0
            )

        return fitted

    def _fit_isotonic_single_fold(
        self,
        train_scores: np.ndarray,
        train_influence: np.ndarray,
        test_scores: np.ndarray,
    ) -> np.ndarray:
        """Fit isotonic regression for a single fold."""
        from scipy.stats import spearmanr

        corr, _ = spearmanr(train_scores, train_influence)
        increasing = corr >= 0

        iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
        iso.fit(train_scores, train_influence)
        return iso.predict(test_scores)

    def _fit_isotonic_cross_fitted(
        self,
        influence: np.ndarray,
        judge_scores: np.ndarray,
        fold_ids: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """Original isotonic cross-fitting implementation."""
        fitted = np.zeros_like(influence)
        unique_folds = np.unique(fold_ids[fold_ids >= 0])

        # Track directions used across folds for diagnostics
        directions_used = []
        correlations = []

        from scipy.stats import spearmanr

        for fold in unique_folds:
            # Define train and test sets
            train_mask = (fold_ids >= 0) & (fold_ids != fold) & valid_mask
            test_mask = (fold_ids == fold) & valid_mask

            if train_mask.sum() < 10 or test_mask.sum() == 0:
                # Not enough data in this fold
                if test_mask.sum() > 0:
                    fitted[test_mask] = (
                        influence[train_mask].mean() if train_mask.sum() > 0 else 0
                    )
                continue

            # Determine direction based on training data correlation
            corr, _ = spearmanr(judge_scores[train_mask], influence[train_mask])
            increasing = corr >= 0
            directions_used.append("increasing" if increasing else "decreasing")
            correlations.append(float(corr))

            # Fit isotonic regression on training folds with chosen direction
            iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
            iso.fit(judge_scores[train_mask], influence[train_mask])

            # Predict on test fold
            fitted[test_mask] = iso.predict(judge_scores[test_mask])

        # Handle samples not in any fold
        unfitted = (fold_ids < 0) | ~valid_mask
        if unfitted.sum() > 0:
            fitted[unfitted] = (
                influence[valid_mask].mean() if valid_mask.sum() > 0 else 0
            )

        # Store aggregated direction info for diagnostics
        if directions_used:
            # Most common direction across folds
            from collections import Counter

            direction_counts = Counter(directions_used)
            self._last_direction = direction_counts.most_common(1)[0][0]
            self._last_correlation = (
                float(np.mean(correlations)) if correlations else 0.0
            )
            self._fold_directions = directions_used  # Store for detailed diagnostics
            self._last_model_type = "isotonic"

        return fitted

    def _compute_diagnostics(
        self,
        original: np.ndarray,
        fitted: np.ndarray,
        residual: np.ndarray,
        policy: str,
    ) -> Dict[str, Any]:
        """Compute IIC diagnostics for reporting and visualization.

        Key metrics:
        - R²: How much of the IF variance is explained by judge scores
        - Variance reduction: How much we reduced the IF variance
        - SE reduction: How much we reduced the standard error
        """
        var_original = np.var(original, ddof=1)
        var_residual = np.var(residual, ddof=1)
        var_fitted = np.var(fitted, ddof=1)

        # R² of the isotonic fit (how well S predicts φ)
        ss_tot = np.sum((original - original.mean()) ** 2)
        ss_res = np.sum((original - fitted) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Variance reduction (main benefit)
        var_reduction = 1 - (var_residual / var_original) if var_original > 0 else 0

        # Standard error reduction (what users care about)
        se_reduction = (
            1 - np.sqrt(max(0, var_residual) / var_original) if var_original > 0 else 0
        )

        # Effective sample size gain (interpretable)
        # ESS ∝ 1/Var, so ESS gain = var_original/var_residual
        ess_gain = var_original / var_residual if var_residual > 0 else 1.0

        diagnostics = {
            "policy": policy,
            "r_squared": float(max(0, r_squared)),  # Can be negative if fit is terrible
            "var_original": float(var_original),
            "var_residual": float(var_residual),
            "var_fitted": float(var_fitted),
            "var_reduction": float(max(0, var_reduction)),
            "se_reduction": float(max(0, se_reduction)),
            "ess_gain": float(ess_gain),
            "n_samples": len(original),
            # Mean of residualized IF should equal mean of original minus mean of fitted
            "residual_mean": float(residual.mean()),
            "expected_residual_mean": float(original.mean() - fitted.mean()),
            "mean_check_passed": abs(
                residual.mean() - (original.mean() - fitted.mean())
            )
            < 1e-10,
        }

        # Add model type information
        if hasattr(self, "_last_model_type"):
            diagnostics["model_type"] = self._last_model_type

        # Add direction information if available (isotonic only)
        if hasattr(self, "_last_direction"):
            diagnostics["direction"] = self._last_direction
            diagnostics["correlation"] = self._last_correlation

        # Add per-fold model types if cross-fitting was used with splines
        if hasattr(self, "_fold_model_types"):
            from collections import Counter

            model_counts = Counter(self._fold_model_types)
            diagnostics["fold_model_types"] = dict(model_counts)

        # Add per-fold directions if cross-fitting was used with isotonic
        if hasattr(self, "_fold_directions"):
            from collections import Counter

            direction_counts = Counter(self._fold_directions)
            diagnostics["fold_directions"] = dict(direction_counts)

        return diagnostics

    def get_diagnostics(self, policy: Optional[str] = None) -> Dict[str, Any]:
        """Get IIC diagnostics for reporting.

        Args:
            policy: Specific policy (None for all)

        Returns:
            Dictionary of diagnostics
        """
        if policy is not None:
            return self._diagnostics.get(policy, {})
        return self._diagnostics.copy()

    def get_fitted_component(self, policy: str) -> Optional[np.ndarray]:
        """Get fitted E[φ|S] component for visualization.

        Only available if store_components=True in config.
        """
        return self._fitted_components.get(policy)

    def summary(self) -> str:
        """Generate summary of IIC performance across policies."""
        if not self._diagnostics:
            return "No IIC diagnostics available"

        lines = ["IIC Performance Summary:"]
        for policy, diag in self._diagnostics.items():
            if diag.get("applied", False):
                lines.append(
                    f"  {policy}: R²={diag.get('r_squared', 0):.3f}, "
                    f"SE reduction={diag.get('se_reduction', 0):.1%}, "
                    f"ESS gain={diag.get('ess_gain', 1):.2f}x"
                )
            else:
                lines.append(
                    f"  {policy}: Not applied ({diag.get('reason', 'unknown')})"
                )

        return "\n".join(lines)
