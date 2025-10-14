"""Flexible calibration modes for non-monotone relationships.

This module extends judge calibration to handle non-monotone relationships
through flexible shape fitting while maintaining cross-fitting support.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Literal, Callable, List
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def _fit_ecdf(x: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Fit empirical CDF for consistent ranking.

    Args:
        x: Training data to build ECDF from

    Returns:
        Function that maps values to their empirical CDF ranks in (0,1)
    """
    xs = np.sort(x)
    n = xs.size

    def F(z: np.ndarray) -> np.ndarray:
        # Right-continuous empirical CDF, mapped to mid-ranks in (0,1)
        idx = np.searchsorted(xs, z, side="right")
        return (idx - 0.5) / n

    return F


class FlexibleCalibrator:
    """Flexible calibration supporting monotone and non-monotone relationships.

    Modes:
    - 'monotone': Standard isotonic regression (current default)
    - 'two_stage': Learn smooth g(S) then isotonic on g(S)
    - 'auto': Automatically select based on cross-validation
    """

    def __init__(
        self,
        mode: Literal["monotone", "two_stage", "auto"] = "monotone",
        n_splines: int = 8,
        random_seed: int = 42,
        covariate_names: Optional[List[str]] = None,
    ):
        """Initialize flexible calibrator.

        Args:
            mode: Calibration mode
            n_splines: Number of splines for two_stage mode
            random_seed: Random seed for reproducibility
            covariate_names: Optional list of covariate names to use in two_stage mode
                (e.g., ["response_length", "domain"]). Covariates help model confounding
                where judge scores at fixed S have different oracle outcomes based on
                observable features.
        """
        self.mode = mode
        self.n_splines = n_splines
        self.random_seed = random_seed
        self.covariate_names = covariate_names or []
        self.selected_mode: Optional[Literal["monotone", "two_stage"]] = (
            None  # For auto mode
        )

        # Validate: covariates only work with two_stage
        if self.covariate_names and mode == "monotone":
            raise ValueError(
                "Covariates are only supported in 'two_stage' or 'auto' mode. "
                "Monotone isotonic regression is univariate and cannot incorporate covariates. "
                f"Got mode='{mode}' with covariates={covariate_names}"
            )

        # Storage for fitted models
        self._monotone_models: Dict[int, Any] = {}
        self._g_models: Dict[int, Any] = {}
        self._iso_models: Dict[int, Any] = {}
        self._ecdf_models: Dict[int, Callable] = {}  # Per-fold ECDFs

        # Full models for inference (no folds)
        self._full_monotone_model: Optional[Any] = None
        self._full_g_model: Optional[Any] = None
        self._full_iso_model: Optional[Any] = None
        self._full_ecdf: Optional[Callable] = None

    def fit(
        self,
        S: np.ndarray,
        Y: np.ndarray,
        folds: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> "FlexibleCalibrator":
        """Fit the calibrator with cross-fitting.

        Args:
            S: Judge scores (n_samples,)
            Y: Oracle labels (n_samples,)
            folds: Fold assignments for cross-fitting (n_samples,)
            covariates: Optional covariate matrix (n_samples, n_covariates)
                Only used in two_stage mode. Each column corresponds to a covariate
                specified in covariate_names.

        Returns:
            Self for chaining
        """
        unique_folds = np.unique(folds)
        n_samples = len(S)

        # Validate covariates if provided
        if covariates is not None:
            if len(covariates) != n_samples:
                raise ValueError(
                    f"Covariate matrix length {len(covariates)} doesn't match samples {n_samples}"
                )
            if self.covariate_names and covariates.shape[1] != len(
                self.covariate_names
            ):
                raise ValueError(
                    f"Covariate matrix has {covariates.shape[1]} columns but "
                    f"{len(self.covariate_names)} covariate names were specified"
                )
            if not self.covariate_names:
                logger.warning(
                    "Covariates provided but no covariate_names specified. "
                    "Covariates will be used but not labeled."
                )

        logger.debug(
            f"FlexibleCalibrator.fit: {n_samples} samples, {len(unique_folds)} folds, "
            f"mode={self.mode}, covariates={covariates.shape if covariates is not None else None}"
        )

        if self.mode == "auto":
            # If covariates provided, force two_stage
            if covariates is not None:
                logger.info(
                    "Auto mode with covariates: forcing two_stage mode "
                    "(covariates not supported in monotone)"
                )
                self._fit_two_stage(S, Y, folds, covariates)
                self.selected_mode = "two_stage"
            else:
                logger.debug("Auto mode: evaluating monotone fit first")
                # Always fit monotone (it's fast)
                self._fit_monotone(S, Y, folds)

                # Quick check: if monotone fit is very good, skip two-stage
                pred_mono = self._predict_monotone(S, folds)
                rmse_mono = np.sqrt(np.mean((Y - pred_mono) ** 2))

                # Check for clear non-monotonicity via regional performance
                sort_idx = np.argsort(S)
                n_third = len(S) // 3
                low_mask = sort_idx[:n_third]
                mid_mask = sort_idx[n_third : 2 * n_third]
                high_mask = sort_idx[2 * n_third :]

                rmse_low = np.sqrt(np.mean((Y[low_mask] - pred_mono[low_mask]) ** 2))
                rmse_mid = np.sqrt(np.mean((Y[mid_mask] - pred_mono[mid_mask]) ** 2))
                rmse_high = np.sqrt(np.mean((Y[high_mask] - pred_mono[high_mask]) ** 2))

                # Always fit two-stage and use _select_best_mode for auto mode
                # (ensures consistent application of 1-SE rule)
                max_regional_diff = max(rmse_low, rmse_mid, rmse_high) - min(
                    rmse_low, rmse_mid, rmse_high
                )
                logger.debug(
                    f"Regional RMSE differences: {max_regional_diff:.3f}, fitting two-stage for comparison"
                )
                self._fit_two_stage(S, Y, folds, covariates)
                self.selected_mode = self._select_best_mode(S, Y, folds, covariates)
        elif self.mode == "monotone":
            if covariates is not None:
                raise ValueError(
                    "Covariates provided but mode='monotone'. "
                    "Use mode='two_stage' or 'auto' for covariate support."
                )
            logger.debug("Fitting monotone calibration only")
            self._fit_monotone(S, Y, folds)
            self.selected_mode = "monotone"
        elif self.mode == "two_stage":
            logger.debug("Fitting two-stage calibration only")
            self._fit_two_stage(S, Y, folds, covariates)
            self.selected_mode = "two_stage"
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Also fit full models for inference (no folds)
        logger.debug("Fitting full models for inference")
        self._fit_full_models(S, Y, covariates)

        return self

    def _fit_monotone(self, S: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> None:
        """Fit standard monotone isotonic regression."""
        for k in np.unique(folds):
            train_mask = folds != k
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(S[train_mask], Y[train_mask])
            self._monotone_models[k] = iso

    def _fit_two_stage(
        self,
        S: np.ndarray,
        Y: np.ndarray,
        folds: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> None:
        """Fit two-stage calibrator: g(S, X_cov) -> isotonic.

        Args:
            S: Judge scores
            Y: Oracle labels
            folds: Fold assignments
            covariates: Optional covariate matrix (n_samples, n_covariates)
        """
        unique_folds = np.unique(folds)

        # Step 1: Fit smooth g(S, X_cov) and ECDF for each fold
        for k in unique_folds:
            train_mask = folds != k
            S_train = S[train_mask]
            Y_train = Y[train_mask]

            # Skip if too few training samples
            if len(S_train) < 20:
                # Fallback to monotone for small folds
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(S_train, Y_train)
                self._g_models[k] = None
                self._iso_models[k] = iso
                self._ecdf_models[k] = _fit_ecdf(
                    S_train
                )  # Still fit ECDF for consistency
                continue

            # Build feature matrix: [S, covariates]
            if covariates is not None:
                X_train = np.column_stack([S_train, covariates[train_mask]])
            else:
                X_train = S_train.reshape(-1, 1)

            # Fit spline + ridge for smooth transformation
            n_knots = min(max(5, self.n_splines), len(S_train) // 4)  # Minimum 5 knots
            spline = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 13), store_cv_results=False)
            g_model = make_pipeline(spline, ridge)

            # Fit g(S, X_cov) to predict Y
            g_model.fit(X_train, Y_train)
            self._g_models[k] = g_model

            # Fit ECDF on g(S, X_cov) predictions for this fold's training data
            g_train = g_model.predict(X_train)
            self._ecdf_models[k] = _fit_ecdf(g_train)

        # Step 2: Fit isotonic on rank-transformed space for each fold
        for k in unique_folds:
            train_mask = folds != k

            if self._g_models.get(k) is not None:
                # Build feature matrix for this fold
                if covariates is not None:
                    X_train = np.column_stack([S[train_mask], covariates[train_mask]])
                else:
                    X_train = S[train_mask].reshape(-1, 1)

                # Transform training data through g and ECDF
                g_train = self._g_models[k].predict(X_train)
                T_ranked_train = self._ecdf_models[k](g_train)
            else:
                # Fallback: use ECDF on original scores
                T_ranked_train = self._ecdf_models[k](S[train_mask])

            # Fit isotonic on ranked space
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(T_ranked_train, Y[train_mask])
            self._iso_models[k] = iso

    def _fit_full_models(
        self, S: np.ndarray, Y: np.ndarray, covariates: Optional[np.ndarray] = None
    ) -> None:
        """Fit full models on all data for inference.

        Args:
            S: Judge scores
            Y: Oracle labels
            covariates: Optional covariate matrix (n_samples, n_covariates)
        """
        if self.selected_mode == "monotone" or self.mode == "monotone":
            # Fit full monotone model
            self._full_monotone_model = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self._full_monotone_model.fit(S, Y)

        if (
            self.selected_mode == "two_stage"
            or self.mode == "two_stage"
            or self.mode == "auto"
        ):
            # Fit full two-stage model
            if len(S) >= 20:
                # Build feature matrix: [S, covariates]
                if covariates is not None:
                    X_full = np.column_stack([S, covariates])
                else:
                    X_full = S.reshape(-1, 1)

                # Fit g(S, X_cov)
                n_knots = min(max(5, self.n_splines), len(S) // 4)
                spline = SplineTransformer(
                    n_knots=n_knots, degree=3, include_bias=False
                )
                ridge = RidgeCV(alphas=np.logspace(-3, 3, 13), store_cv_results=False)
                self._full_g_model = make_pipeline(spline, ridge)
                self._full_g_model.fit(X_full, Y)

                # Fit ECDF on g(S, X_cov)
                g_full = self._full_g_model.predict(X_full)
                self._full_ecdf = _fit_ecdf(g_full)

                # Fit isotonic on ranked space
                T_ranked = self._full_ecdf(g_full)
                self._full_iso_model = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds="clip"
                )
                self._full_iso_model.fit(T_ranked, Y)
            else:
                # Fallback to monotone
                self._full_monotone_model = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds="clip"
                )
                self._full_monotone_model.fit(S, Y)

    def predict(
        self,
        S: np.ndarray,
        folds: Optional[np.ndarray] = None,
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict calibrated values.

        Args:
            S: Judge scores to calibrate
            folds: Optional fold assignments for OOF prediction
            covariates: Optional covariate matrix (n_samples, n_covariates)

        Returns:
            Calibrated predictions
        """
        mode = self.selected_mode or self.mode

        if mode == "monotone":
            if covariates is not None:
                raise ValueError("Covariates not supported in monotone mode")
            return self._predict_monotone(S, folds)
        elif mode == "two_stage":
            return self._predict_two_stage(S, folds, covariates)
        else:
            raise ValueError(f"No fitted models for mode: {mode}")

    def _predict_monotone(
        self, S: np.ndarray, folds: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict using monotone models."""
        if folds is None:
            # Use full model for inference
            if self._full_monotone_model is not None:
                return np.asarray(self._full_monotone_model.predict(S))
            else:
                # Fallback to ensemble average if full model not fitted
                preds = []
                for model in self._monotone_models.values():
                    preds.append(model.predict(S))
                return np.asarray(np.mean(preds, axis=0))
        else:
            # OOF prediction
            Y_hat = np.zeros_like(S)
            for k in np.unique(folds):
                mask = folds == k
                if k in self._monotone_models:
                    Y_hat[mask] = self._monotone_models[k].predict(S[mask])
                else:
                    # Fallback to full model if available
                    if self._full_monotone_model is not None:
                        Y_hat[mask] = self._full_monotone_model.predict(S[mask])
                    else:
                        # Last resort: ensemble average
                        preds = []
                        for model in self._monotone_models.values():
                            preds.append(model.predict(S[mask]))
                        Y_hat[mask] = np.mean(preds, axis=0)
            return Y_hat

    def _predict_two_stage(
        self,
        S: np.ndarray,
        folds: Optional[np.ndarray] = None,
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict using two-stage models.

        Args:
            S: Judge scores
            folds: Optional fold assignments for OOF prediction
            covariates: Optional covariate matrix (n_samples, n_covariates)

        Returns:
            Calibrated predictions
        """
        if folds is None:
            # Use full model for inference
            if (
                self._full_g_model is not None
                and self._full_iso_model is not None
                and self._full_ecdf is not None
            ):
                # Build feature matrix: [S, covariates]
                if covariates is not None:
                    X = np.column_stack([S, covariates])
                else:
                    X = S.reshape(-1, 1)

                g_pred = self._full_g_model.predict(X)
                T_ranked = self._full_ecdf(g_pred)
                return np.asarray(self._full_iso_model.predict(T_ranked))
            elif self._full_monotone_model is not None:
                # Fallback to monotone if two-stage wasn't fitted
                return np.asarray(self._full_monotone_model.predict(S))
            else:
                # Last resort: ensemble average
                preds = []
                for k in self._g_models.keys():
                    if k in self._ecdf_models and k in self._iso_models:
                        g_model = self._g_models[k]
                        iso_model = self._iso_models[k]
                        if g_model is not None:
                            # Build feature matrix for ensemble
                            if covariates is not None:
                                X = np.column_stack([S, covariates])
                            else:
                                X = S.reshape(-1, 1)
                            g_pred = g_model.predict(X)
                            T_ranked = self._ecdf_models[k](g_pred)
                        else:
                            T_ranked = self._ecdf_models[k](S)
                        preds.append(iso_model.predict(T_ranked))
                if preds:
                    return np.asarray(np.mean(preds, axis=0))
                else:
                    # Ultimate fallback: return mean of training labels
                    return np.full_like(S, 0.5)
        else:
            # OOF prediction
            Y_hat = np.zeros_like(S)
            for k in np.unique(folds):
                mask = folds == k
                if (
                    k in self._g_models
                    and k in self._iso_models
                    and k in self._ecdf_models
                ):
                    if self._g_models[k] is not None:
                        # Build feature matrix for this fold
                        if covariates is not None:
                            X_fold = np.column_stack([S[mask], covariates[mask]])
                        else:
                            X_fold = S[mask].reshape(-1, 1)
                        g_pred = self._g_models[k].predict(X_fold)
                        T_ranked = self._ecdf_models[k](g_pred)
                    else:
                        T_ranked = self._ecdf_models[k](S[mask])
                    Y_hat[mask] = self._iso_models[k].predict(T_ranked)
                else:
                    # Fallback to full model if available
                    if (
                        self._full_g_model is not None
                        and self._full_iso_model is not None
                        and self._full_ecdf is not None
                    ):
                        # Build feature matrix for fallback
                        if covariates is not None:
                            X_fallback = np.column_stack([S[mask], covariates[mask]])
                        else:
                            X_fallback = S[mask].reshape(-1, 1)
                        g_pred = self._full_g_model.predict(X_fallback)
                        T_ranked = self._full_ecdf(g_pred)
                        Y_hat[mask] = self._full_iso_model.predict(T_ranked)
                    elif self._full_monotone_model is not None:
                        Y_hat[mask] = self._full_monotone_model.predict(S[mask])
                    else:
                        # Ultimate fallback
                        Y_hat[mask] = np.mean(S[mask])
            return Y_hat

    def _select_best_mode(
        self,
        S: np.ndarray,
        Y: np.ndarray,
        folds: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> Literal["monotone", "two_stage"]:
        """Select best mode based on OOF RMSE.

        Args:
            S: Judge scores
            Y: Oracle labels
            folds: Fold assignments
            covariates: Optional covariate matrix (n_samples, n_covariates)

        Returns:
            Selected mode ('monotone' or 'two_stage')
        """
        # Get OOF predictions for each mode
        pred_mono = self._predict_monotone(S, folds)
        pred_two_stage = self._predict_two_stage(S, folds, covariates)

        # Calculate RMSEs
        rmse_mono = np.sqrt(np.mean((Y - pred_mono) ** 2))
        rmse_two_stage = np.sqrt(np.mean((Y - pred_two_stage) ** 2))

        # Check for non-monotonicity by comparing performance in different regions
        # Sort by judge scores
        sort_idx = np.argsort(S)

        # Split into thirds and check local performance
        n_third = len(S) // 3
        low_mask = sort_idx[:n_third]
        mid_mask = sort_idx[n_third : 2 * n_third]
        high_mask = sort_idx[2 * n_third :]

        rmse_mono_low = np.sqrt(np.mean((Y[low_mask] - pred_mono[low_mask]) ** 2))
        rmse_flex_low = np.sqrt(np.mean((Y[low_mask] - pred_two_stage[low_mask]) ** 2))

        rmse_mono_mid = np.sqrt(np.mean((Y[mid_mask] - pred_mono[mid_mask]) ** 2))
        rmse_flex_mid = np.sqrt(np.mean((Y[mid_mask] - pred_two_stage[mid_mask]) ** 2))

        rmse_mono_high = np.sqrt(np.mean((Y[high_mask] - pred_mono[high_mask]) ** 2))
        rmse_flex_high = np.sqrt(
            np.mean((Y[high_mask] - pred_two_stage[high_mask]) ** 2)
        )

        # Count regions where two-stage is better
        better_count = 0
        if rmse_flex_low < rmse_mono_low:
            better_count += 1
        if rmse_flex_mid < rmse_mono_mid:
            better_count += 1
        if rmse_flex_high < rmse_mono_high:
            better_count += 1

        # Apply 1-SE rule: prefer simpler model unless complex is significantly better
        # Standard error of RMSE estimate using delta method
        residuals_mono = Y - pred_mono
        n = len(S)
        se_mse = np.std(residuals_mono**2, ddof=1) / np.sqrt(n) if n > 1 else 0.0
        se_rmse = se_mse / (2.0 * max(rmse_mono, 1e-12))

        logger.info(f"Calibration mode selection:")
        logger.info(
            f"  Overall RMSE - Monotone: {rmse_mono:.4f}, Two-stage: {rmse_two_stage:.4f}"
        )
        logger.info(
            f"  Regional performance - Two-stage better in {better_count}/3 regions"
        )
        logger.debug(f"    Low S: Mono={rmse_mono_low:.4f}, Flex={rmse_flex_low:.4f}")
        logger.debug(f"    Mid S: Mono={rmse_mono_mid:.4f}, Flex={rmse_flex_mid:.4f}")
        logger.debug(
            f"    High S: Mono={rmse_mono_high:.4f}, Flex={rmse_flex_high:.4f}"
        )

        # Select two-stage if:
        # 1. It's significantly better overall (1-SE rule), OR
        # 2. It's better in at least 2/3 regions (indicates non-monotonicity)
        if rmse_two_stage < rmse_mono - se_rmse or better_count >= 2:
            logger.info(f"  → Selected: two_stage (better in {better_count}/3 regions)")
            return "two_stage"
        else:
            logger.info(f"  → Selected: monotone (simpler model preferred)")
            return "monotone"

    def index(
        self,
        S: np.ndarray,
        folds: Optional[np.ndarray] = None,
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return the index used for isotonic regression.

        For monotone mode: Returns S (or normalized S)
        For two-stage mode: Returns rank_transform(g(S, X_cov))

        Args:
            S: Judge scores
            folds: Optional fold assignments for OOF transformation
            covariates: Optional covariate matrix (n_samples, n_covariates)

        Returns:
            Index values for isotonic regression
        """
        mode = self.selected_mode or self.mode

        if mode == "monotone":
            if covariates is not None:
                raise ValueError("Covariates not supported in monotone mode")
            # In monotone mode, the index is just S
            return S

        elif mode == "two_stage":
            if folds is None:
                # Use full model for inference
                if self._full_g_model is not None:
                    # Build feature matrix
                    if covariates is not None:
                        X = np.column_stack([S, covariates])
                    else:
                        X = S.reshape(-1, 1)

                    g_pred = self._full_g_model.predict(X)
                    if self._full_ecdf is not None:
                        return np.asarray(self._full_ecdf(g_pred))
                    else:
                        # Fallback to rank transform if ECDF not available
                        return self._rank_transform(g_pred)
                else:
                    # Fallback: ensemble average
                    g_ensemble = np.zeros_like(S)
                    count = 0
                    for g_model in self._g_models.values():
                        if g_model is not None:
                            # Build feature matrix
                            if covariates is not None:
                                X = np.column_stack([S, covariates])
                            else:
                                X = S.reshape(-1, 1)
                            g_ensemble += g_model.predict(X)
                            count += 1
                    if count > 0:
                        g_ensemble /= count
                        return self._rank_transform(g_ensemble)
                    else:
                        return S  # Ultimate fallback
            else:
                # OOF transformation for training
                T = np.zeros_like(S)
                for k in np.unique(folds):
                    mask = folds == k
                    if k in self._g_models and k in self._ecdf_models:
                        if self._g_models[k] is not None:
                            # Build feature matrix for this fold
                            if covariates is not None:
                                X_fold = np.column_stack([S[mask], covariates[mask]])
                            else:
                                X_fold = S[mask].reshape(-1, 1)
                            g_pred = self._g_models[k].predict(X_fold)
                            T[mask] = self._ecdf_models[k](g_pred)
                        else:
                            # Fallback to ECDF on raw scores
                            T[mask] = self._ecdf_models[k](S[mask])
                    else:
                        # Ultimate fallback
                        T[mask] = S[mask]
                return T
        else:
            # Fallback for unknown modes
            return S

    def _rank_transform(self, x: np.ndarray) -> np.ndarray:
        """Simple rank transformation to [0, 1]."""
        from scipy.stats import rankdata

        ranks = rankdata(x, method="average")
        return np.asarray((ranks - 0.5) / len(x))

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics about the fitted calibrator."""
        return {
            "mode": self.selected_mode or self.mode,
            "n_folds": len(self._monotone_models or self._iso_models),
            "has_two_stage": bool(self._g_models),
            "has_monotone": bool(self._monotone_models),
        }

    @property
    def iso_reg(self) -> Optional[Any]:
        """Get the isotonic regression model for compatibility."""
        mode = self.selected_mode or self.mode
        if mode == "monotone":
            return self._full_monotone_model
        else:
            # For two-stage, return the final isotonic model
            return self._full_iso_model
