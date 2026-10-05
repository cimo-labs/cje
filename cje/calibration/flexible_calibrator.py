"""Flexible calibration modes for non-monotone relationships.

This module extends judge calibration to handle non-monotone relationships
through flexible shape fitting while maintaining cross-fitting support.
"""

import numpy as np
from typing import Optional, Dict, Any, Literal, Callable, List
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline, make_pipeline
import logging
import warnings

logger = logging.getLogger(__name__)

# Fewest labelled rows a two-stage fit (the full model, or one fold's training
# complement) needs to fit the spline index g(S, X). Below it the fit falls
# back to a judge-score-only monotone map that ignores any covariates.
TWO_STAGE_MIN_ROWS = 20


def validate_oracle_labels(labels: np.ndarray) -> None:
    """Raise ``ValueError`` unless every oracle label is finite and in [0, 1].

    The isotonic fits are bounded to [0, 1] and clip their outputs, so labels
    in original units (a 0-500 amount, a 1-5 rating) or centred on 0 would be
    silently collapsed instead of calibrated.
    """
    y = np.asarray(labels, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("oracle_labels must be finite (found NaN or inf).")
    if np.any((y < 0.0) | (y > 1.0)):
        raise ValueError(
            f"oracle_labels must lie in [0, 1] (calibrated rewards are clipped "
            f"to that range); got values in [{y.min():g}, {y.max():g}]. "
            "Use analyze_dataset, which normalises arbitrary bounded scales, or "
            "rescale with (y - lo) / (hi - lo) and map results back."
        )


def _predict_spline_ridge(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Evaluate the fitted smooth map in a fixed order for every row.

    Ridge's matrix product can round differently for different batch shapes.
    A last-bit difference can cross an ECDF knot and become a large change in
    calibrated reward. Accumulate the same spline coefficients in the same
    order for training ranks, full predictions and held-out fold predictions.
    This changes no fitted coefficients and introduces no tolerance-based ties.
    """
    spline, ridge = model.steps[0][1], model.steps[1][1]
    features = np.asarray(spline.transform(X), dtype=float)
    coefficients = np.asarray(ridge.coef_, dtype=float)
    predictions = np.zeros(features.shape[0], dtype=float)
    for column, coefficient in enumerate(coefficients):
        predictions += features[:, column] * coefficient
    predictions += float(ridge.intercept_)
    return predictions


def _fit_ecdf(
    x: np.ndarray, sample_weight: Optional[np.ndarray] = None
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit empirical CDF for consistent ranking.

    Args:
        x: Training data to build ECDF from

    Returns:
        Function that maps values to their empirical CDF ranks in (0,1)
    """
    order = np.argsort(x, kind="mergesort")
    xs = np.asarray(x, dtype=float)[order]
    weights = (
        np.ones(xs.size, dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)[order]
    )
    unique_xs, tie_starts = np.unique(xs, return_index=True)
    tie_weights = np.add.reduceat(weights, tie_starts)
    cumulative = np.cumsum(tie_weights)
    total_weight = float(cumulative[-1])
    midranks = (cumulative - 0.5 * tie_weights) / total_weight

    def F(z: np.ndarray) -> np.ndarray:
        # Equal values share the midpoint of their combined probability mass.
        idx = np.searchsorted(unique_xs, z, side="right")
        out = np.zeros_like(np.asarray(z, dtype=float), dtype=float)
        positive = idx > 0
        right = idx[positive] - 1
        out[positive] = midranks[right]
        return out

    return F


def _rows_needed_hint(n_folds: int) -> str:
    """How many labelled rows let every fold use the covariates."""
    needed = -(-TWO_STAGE_MIN_ROWS * n_folds // max(n_folds - 1, 1))
    return (
        f"Every fold uses the covariates once each fold's training complement "
        f"has {TWO_STAGE_MIN_ROWS} labelled rows (about {needed} labelled rows "
        f"with {n_folds} evenly sized folds)."
    )


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
        # Set by fit(). covariates_used: the full model uses the supplied
        # covariates. covariates_dropped: covariates were supplied but the
        # full model had too few labelled rows to use them, so the fit is
        # judge-score-only monotone. n_folds_without_covariates: folds whose
        # model ignores supplied covariates (0 when none were supplied).
        self.covariates_used: Optional[bool] = None
        self.covariates_dropped: bool = False
        self.n_folds_without_covariates: Optional[int] = None

        # Validate: covariates only work with two_stage
        if self.covariate_names and mode == "monotone":
            raise ValueError(
                "Covariates are only supported in 'two_stage' or 'auto' mode. "
                "Monotone isotonic regression is univariate and cannot incorporate covariates. "
                f"Got mode='{mode}' with covariates={covariate_names}"
            )

        self._log_level: int = logging.INFO

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
        sample_weight: Optional[np.ndarray] = None,
        log_level: int = logging.INFO,
    ) -> "FlexibleCalibrator":
        """Fit the calibrator with cross-fitting.

        Args:
            S: Judge scores (n_samples,)
            Y: Oracle labels (n_samples,), finite and in [0, 1]
            folds: Fold assignments for cross-fitting (n_samples,)
            covariates: Optional covariate matrix (n_samples, n_covariates)
                Only used in two_stage mode. Each column corresponds to a covariate
                specified in covariate_names. With fewer than
                TWO_STAGE_MIN_ROWS labelled rows the fit falls back to
                judge-score-only monotone calibration (selected_mode
                "monotone"), and folds whose training complement is that
                small ignore the covariates; both emit a UserWarning.
            sample_weight: Optional positive per-sample fit weights.
            log_level: Level for routine progress messages (mode forcing and
                mode selection). ``JudgeCalibrator.fit_cv(quiet=True)`` passes
                DEBUG so per-replicate refits stay silent.

        Returns:
            Self for chaining

        Raises:
            ValueError: If any label is non-finite or outside [0, 1].
        """
        validate_oracle_labels(Y)
        self._log_level = log_level
        unique_folds = np.unique(folds)
        n_samples = len(S)
        weights: Optional[np.ndarray] = None
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.shape != (n_samples,):
                raise ValueError(
                    f"sample_weight must have shape ({n_samples},), got "
                    f"{weights.shape}"
                )
            if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
                raise ValueError("sample_weight values must be finite and positive")

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
                logger.debug(
                    "Covariates provided but no covariate_names specified. "
                    "Covariates will be used but not labeled."
                )

        logger.debug(
            f"FlexibleCalibrator.fit: {n_samples} samples, {len(unique_folds)} folds, "
            f"mode={self.mode}, covariates={covariates.shape if covariates is not None else None}"
        )

        self.covariates_used = False
        self.covariates_dropped = False
        self.n_folds_without_covariates = 0
        if (
            covariates is not None
            and self.mode in ("auto", "two_stage")
            and n_samples < TWO_STAGE_MIN_ROWS
        ):
            # Too few rows for the spline index: the full model and every
            # fold (each training complement is smaller still) fall back to
            # judge-score-only isotonic fits, exactly as before. The fit is
            # unchanged; it is now reported as monotone, not two_stage.
            self._fit_two_stage(S, Y, folds, None, weights)
            self.selected_mode = "monotone"
            self.covariates_dropped = True
            self.n_folds_without_covariates = int(len(unique_folds))
            warnings.warn(
                f"Covariates were supplied but only {n_samples} labelled rows "
                f"are available; two-stage calibration needs at least "
                f"{TWO_STAGE_MIN_ROWS} to use them. Falling back to "
                "judge-score-only monotone calibration (selected_mode "
                "'monotone'): the covariates are ignored by the full model and "
                f"by all {len(unique_folds)} cross-fitting folds. "
                f"{_rows_needed_hint(len(unique_folds))}",
                UserWarning,
                stacklevel=2,
            )
        elif self.mode == "auto":
            # If covariates provided, force two_stage
            if covariates is not None:
                logger.log(
                    log_level,
                    "Auto mode with covariates: forcing two_stage mode "
                    "(covariates not supported in monotone)",
                )
                self._fit_two_stage(S, Y, folds, covariates, weights)
                self.selected_mode = "two_stage"
            else:
                logger.debug("Auto mode: evaluating monotone fit first")
                # Always fit monotone (it's fast)
                self._fit_monotone(S, Y, folds, weights)

                # Quick check: if monotone fit is very good, skip two-stage
                pred_mono = self._predict_monotone(S, folds)
                rmse_mono = np.sqrt(np.average((Y - pred_mono) ** 2, weights=weights))

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
                self._fit_two_stage(S, Y, folds, covariates, weights)
                self.selected_mode = self._select_best_mode(
                    S, Y, folds, covariates, weights
                )
        elif self.mode == "monotone":
            if covariates is not None:
                raise ValueError(
                    "Covariates provided but mode='monotone'. "
                    "Use mode='two_stage' or 'auto' for covariate support."
                )
            logger.debug("Fitting monotone calibration only")
            self._fit_monotone(S, Y, folds, weights)
            self.selected_mode = "monotone"
        elif self.mode == "two_stage":
            logger.debug("Fitting two-stage calibration only")
            self._fit_two_stage(S, Y, folds, covariates, weights)
            self.selected_mode = "two_stage"
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if covariates is not None and self.selected_mode == "two_stage":
            # The full model has enough rows to use the covariates, but a fold
            # whose training complement is too small fell back to a
            # judge-score-only fit. Those fold models produce the out-of-fold
            # predictions behind the residual correction and the jackknife.
            self.covariates_used = True
            self.n_folds_without_covariates = int(
                sum(self._g_models.get(k) is None for k in unique_folds)
            )
            if self.n_folds_without_covariates > 0:
                warnings.warn(
                    f"Covariates were supplied but {self.n_folds_without_covariates} "
                    f"of {len(unique_folds)} cross-fitting folds have fewer than "
                    f"{TWO_STAGE_MIN_ROWS} labelled training rows; those folds fall "
                    "back to a judge-score-only fit that ignores the covariates. "
                    "Their out-of-fold predictions feed the residual correction "
                    "and the oracle jackknife; the full calibrator still uses "
                    f"the covariates. {_rows_needed_hint(len(unique_folds))}",
                    UserWarning,
                    stacklevel=2,
                )

        # Also fit full models for inference (no folds)
        logger.debug("Fitting full models for inference")
        self._fit_full_models(S, Y, covariates, weights)

        return self

    def _fit_monotone(
        self,
        S: np.ndarray,
        Y: np.ndarray,
        folds: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Fit standard monotone isotonic regression."""
        for k in np.unique(folds):
            train_mask = folds != k
            if not np.any(train_mask):
                raise ValueError(
                    f"Fold {k} has no held-out complement; cross-fitted "
                    "calibration requires at least two nonempty folds."
                )
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(
                S[train_mask],
                Y[train_mask],
                sample_weight=(
                    sample_weight[train_mask] if sample_weight is not None else None
                ),
            )
            self._monotone_models[k] = iso

    def _fit_two_stage(
        self,
        S: np.ndarray,
        Y: np.ndarray,
        folds: np.ndarray,
        covariates: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
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
            if not np.any(train_mask):
                raise ValueError(
                    f"Fold {k} has no held-out complement; cross-fitted "
                    "calibration requires at least two nonempty folds."
                )
            S_train = S[train_mask]
            Y_train = Y[train_mask]
            weight_train = (
                sample_weight[train_mask] if sample_weight is not None else None
            )

            # Skip if too few training samples
            if len(S_train) < TWO_STAGE_MIN_ROWS:
                # Fallback to monotone for small folds
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(S_train, Y_train, sample_weight=weight_train)
                self._g_models[k] = None
                self._iso_models[k] = iso
                self._ecdf_models[k] = _fit_ecdf(S_train, weight_train)
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
            fit_kwargs = (
                {"ridgecv__sample_weight": weight_train}
                if weight_train is not None
                else {}
            )
            g_model.fit(X_train, Y_train, **fit_kwargs)
            self._g_models[k] = g_model

            # Fit ECDF on g(S, X_cov) predictions for this fold's training data
            g_train = _predict_spline_ridge(g_model, X_train)
            self._ecdf_models[k] = _fit_ecdf(g_train, weight_train)

        # Step 2: Fit isotonic on rank-transformed space for each fold
        for k in unique_folds:
            train_mask = folds != k
            if not np.any(train_mask):
                raise ValueError(
                    f"Fold {k} has no held-out complement; cross-fitted "
                    "calibration requires at least two nonempty folds."
                )

            if self._g_models.get(k) is not None:
                # Build feature matrix for this fold
                if covariates is not None:
                    X_train = np.column_stack([S[train_mask], covariates[train_mask]])
                else:
                    X_train = S[train_mask].reshape(-1, 1)

                # Transform training data through g and ECDF
                g_train = _predict_spline_ridge(self._g_models[k], X_train)
                T_ranked_train = self._ecdf_models[k](g_train)
            else:
                # Fallback: use ECDF on original scores
                T_ranked_train = self._ecdf_models[k](S[train_mask])

            # Fit isotonic on ranked space
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(
                T_ranked_train,
                Y[train_mask],
                sample_weight=(
                    sample_weight[train_mask] if sample_weight is not None else None
                ),
            )
            self._iso_models[k] = iso

    def _fit_full_models(
        self,
        S: np.ndarray,
        Y: np.ndarray,
        covariates: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
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
            self._full_monotone_model.fit(S, Y, sample_weight=sample_weight)

        if (
            self.selected_mode == "two_stage"
            or self.mode == "two_stage"
            or self.mode == "auto"
        ):
            # Fit full two-stage model
            if len(S) >= TWO_STAGE_MIN_ROWS:
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
                fit_kwargs = (
                    {"ridgecv__sample_weight": sample_weight}
                    if sample_weight is not None
                    else {}
                )
                self._full_g_model.fit(X_full, Y, **fit_kwargs)

                # Fit ECDF on g(S, X_cov)
                g_full = _predict_spline_ridge(self._full_g_model, X_full)
                self._full_ecdf = _fit_ecdf(g_full, sample_weight)

                # Fit isotonic on ranked space
                T_ranked = self._full_ecdf(g_full)
                self._full_iso_model = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds="clip"
                )
                self._full_iso_model.fit(T_ranked, Y, sample_weight=sample_weight)
            else:
                # Fallback to monotone
                self._full_monotone_model = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds="clip"
                )
                self._full_monotone_model.fit(S, Y, sample_weight=sample_weight)

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
            if getattr(self, "covariates_dropped", False):
                # Covariates were supplied with too few labelled rows to use
                # them: the models are the judge-score-only two-stage
                # fallbacks. Accept and ignore the covariates, as the fit did.
                return self._predict_two_stage(S, folds, None)
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
            Y_hat = np.zeros(np.asarray(S).shape, dtype=float)
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

                g_pred = _predict_spline_ridge(self._full_g_model, X)
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
                            g_pred = _predict_spline_ridge(g_model, X)
                            T_ranked = self._ecdf_models[k](g_pred)
                        else:
                            T_ranked = self._ecdf_models[k](S)
                        preds.append(iso_model.predict(T_ranked))
                if preds:
                    return np.asarray(np.mean(preds, axis=0))
                else:
                    # No fitted models at all: fail loudly rather than
                    # fabricate a constant reward.
                    raise RuntimeError(
                        "FlexibleCalibrator has no fitted two-stage models — "
                        "call fit() before predict()."
                    )
        else:
            # OOF prediction
            Y_hat = np.zeros(np.asarray(S).shape, dtype=float)
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
                        g_pred = _predict_spline_ridge(self._g_models[k], X_fold)
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
                        g_pred = _predict_spline_ridge(self._full_g_model, X_fallback)
                        T_ranked = self._full_ecdf(g_pred)
                        Y_hat[mask] = self._full_iso_model.predict(T_ranked)
                    elif self._full_monotone_model is not None:
                        Y_hat[mask] = self._full_monotone_model.predict(S[mask])
                    else:
                        # No model for this fold and no full-model fallback:
                        # fail loudly rather than fabricate rewards from the
                        # mean of raw judge scores.
                        raise RuntimeError(
                            f"FlexibleCalibrator has no fitted model for fold "
                            f"{k} and no full model to fall back on — call "
                            f"fit() before predict(). Fitted folds: "
                            f"{sorted(self._iso_models.keys())}."
                        )
            return Y_hat

    def _select_best_mode(
        self,
        S: np.ndarray,
        Y: np.ndarray,
        folds: np.ndarray,
        covariates: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
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

        def weighted_rmse(
            truth: np.ndarray,
            prediction: np.ndarray,
            indices: Optional[np.ndarray] = None,
        ) -> float:
            if indices is None:
                residual = truth - prediction
                weights = sample_weight
            else:
                residual = truth[indices] - prediction[indices]
                weights = sample_weight[indices] if sample_weight is not None else None
            return float(np.sqrt(np.average(residual**2, weights=weights)))

        # Calculate weighted RMSEs.  In a positive-weight bootstrap these are
        # the losses in that bootstrap world, so auto mode selection is part
        # of the replicated estimator.
        rmse_mono = weighted_rmse(Y, pred_mono)
        rmse_two_stage = weighted_rmse(Y, pred_two_stage)

        # Check for non-monotonicity by comparing performance in different regions
        # Sort by judge scores
        sort_idx = np.argsort(S)

        # Split into thirds and check local performance
        n_third = len(S) // 3
        low_mask = sort_idx[:n_third]
        mid_mask = sort_idx[n_third : 2 * n_third]
        high_mask = sort_idx[2 * n_third :]

        rmse_mono_low = weighted_rmse(Y, pred_mono, low_mask)
        rmse_flex_low = weighted_rmse(Y, pred_two_stage, low_mask)

        rmse_mono_mid = weighted_rmse(Y, pred_mono, mid_mask)
        rmse_flex_mid = weighted_rmse(Y, pred_two_stage, mid_mask)

        rmse_mono_high = weighted_rmse(Y, pred_mono, high_mask)
        rmse_flex_high = weighted_rmse(Y, pred_two_stage, high_mask)

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
        if sample_weight is None:
            se_mse = np.std(residuals_mono**2, ddof=1) / np.sqrt(n) if n > 1 else 0.0
        else:
            normalized = sample_weight / np.sum(sample_weight)
            mse = float(np.sum(normalized * residuals_mono**2))
            weighted_var = float(np.sum(normalized * (residuals_mono**2 - mse) ** 2))
            effective_n = 1.0 / float(np.sum(normalized**2))
            se_mse = np.sqrt(weighted_var / max(effective_n, 1.0))
        se_rmse = se_mse / (2.0 * max(rmse_mono, 1e-12))

        level = self._log_level
        logger.log(level, "Calibration mode selection:")
        logger.log(
            level,
            f"  Overall RMSE - Monotone: {rmse_mono:.4f}, Two-stage: {rmse_two_stage:.4f}",
        )
        logger.log(
            level,
            f"  Regional performance - Two-stage better in {better_count}/3 regions",
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
            logger.log(
                level, f"  → Selected: two_stage (better in {better_count}/3 regions)"
            )
            return "two_stage"
        else:
            logger.log(level, "  → Selected: monotone (simpler model preferred)")
            return "monotone"

    def fold_models(self) -> Dict[int, Any]:
        """Per-fold isotonic models for the selected mode.

        For two-stage mode these are the final isotonic stages and expect the
        RANK INDEX, not raw judge scores — route predictions through
        `JudgeCalibrator.predict_oof`, which applies the full transform.

        Returns:
            Dict of fold_id -> fitted isotonic model (may be empty pre-fit).
        """
        if (self.selected_mode or self.mode) == "two_stage" or getattr(
            self, "covariates_dropped", False
        ):
            return dict(self._iso_models)
        return dict(self._monotone_models)
