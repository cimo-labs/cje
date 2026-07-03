"""Base class for CJE estimators."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
import logging

from ..data.models import EstimationResult

logger = logging.getLogger(__name__)


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


class BaseCJEEstimator(ABC):
    """Abstract base class for CJE estimators.

    All estimators must implement:
    - fit(): Prepare the estimator (e.g., calibrate rewards)
    - estimate(): Compute estimates and diagnostics

    The estimate() method must populate EstimationResult.diagnostics.
    """

    def __init__(
        self,
        target_policies: List[str],
        run_diagnostics: bool = True,
        diagnostic_config: Optional[Dict[str, Any]] = None,
        reward_calibrator: Optional[Any] = None,
        oua_jackknife: bool = True,  # Default to True for calibration-aware inference
    ):
        """Initialize estimator.

        Args:
            target_policies: Names of the policies to evaluate
            run_diagnostics: Whether to compute diagnostics (default True)
            diagnostic_config: Optional configuration dict (for future use)
            reward_calibrator: Optional reward calibrator for oracle-jackknife inference
            oua_jackknife: Whether to include calibration uncertainty via the
                oracle jackknife (default True)
        """
        self.target_policies = list(target_policies)
        self.run_diagnostics = run_diagnostics
        self.diagnostic_config = diagnostic_config
        self._fitted = False
        self._influence_functions: Dict[str, np.ndarray] = {}
        self._results: Optional[EstimationResult] = None

        # Configure oracle-jackknife inference for calibration uncertainty
        self.reward_calibrator = reward_calibrator
        self.oua_jackknife = oua_jackknife

    @abstractmethod
    def fit(self) -> None:
        """Fit the estimator (e.g., calibrate rewards)."""
        pass

    @abstractmethod
    def estimate(self) -> EstimationResult:
        """Compute estimates for all target policies."""
        pass

    def fit_and_estimate(self) -> EstimationResult:
        """Convenience method to fit and estimate in one call."""
        self.fit()
        result = self.estimate()

        # All estimators now create diagnostics directly in estimate()
        # The DiagnosticSuite system has been removed for simplicity
        # Following principles: YAGNI, Do One Thing Well

        # Verify diagnostics were created
        if self.run_diagnostics and result is not None:
            if not hasattr(result, "diagnostics") or result.diagnostics is None:
                # This shouldn't happen anymore, but log a warning
                logger.warning(
                    f"{self.__class__.__name__} did not create diagnostics. "
                    "All estimators should create diagnostics directly in estimate()."
                )

        return result

    def get_influence_functions(self, policy: Optional[str] = None) -> Optional[Any]:
        """Get influence functions for a policy or all policies.

        Influence functions capture the per-sample contribution to the estimate
        and are essential for statistical inference (standard errors, confidence
        intervals, hypothesis tests).

        Args:
            policy: Specific policy name, or None for all policies

        Returns:
            If policy specified: array of influence functions for that policy
            If policy is None: dict of all influence functions by policy
            Returns None if not yet estimated
        """
        if not self._influence_functions:
            return None

        if policy is not None:
            return self._influence_functions.get(policy)

        return self._influence_functions

    @property
    def is_fitted(self) -> bool:
        """Check if estimator has been fitted."""
        return self._fitted

    def _validate_fitted(self) -> None:
        """Ensure estimator is fitted before making predictions."""
        if not self._fitted:
            raise RuntimeError("Estimator must be fitted before calling estimate()")

    def get_diagnostics(self) -> Optional[Any]:
        """Get the diagnostics from the last estimation.

        Returns:
            Diagnostics if estimate() has been called, None otherwise
        """
        if self._results and self._results.diagnostics:
            return self._results.diagnostics
        return None

    def _apply_oua_jackknife(self, result: EstimationResult) -> None:
        """Apply the oracle jackknife for calibration-aware inference.

        This method adds oracle uncertainty to standard_errors in-place, accounting
        for finite-sample uncertainty in the learned reward calibrator f̂(S).

        Args:
            result: EstimationResult with standard_errors to augment
        """
        if not (self.oua_jackknife and self.reward_calibrator is not None):
            return

        # Skip oracle-jackknife augmentation at 100% oracle coverage
        # (the calibrator knows its own coverage; at 100% there is no
        # calibration uncertainty to add). Estimators can override
        # _oracle_coverage_for_oua when the calibrator's coverage
        # describes a different dataset than the evaluation data.
        try:
            coverage = self._oracle_coverage_for_oua()

            if coverage is not None and coverage >= 1.0:
                if isinstance(result.metadata, dict):
                    result.metadata.setdefault("se_components", {})
                    result.metadata["se_components"][
                        "oracle_uncertainty_skipped"
                    ] = "100% oracle coverage"
                return
        except Exception:
            pass  # Continue with the default path if we can't check coverage

        # Check if oracle variance is already included (e.g., by the
        # bootstrap inference path, which refits the calibrator per replicate)
        if isinstance(result.metadata, dict) and result.metadata.get(
            "se_components", {}
        ).get("includes_oracle_uncertainty"):
            # Oracle variance already included in standard_errors
            return

        try:
            var_oracle_map: Dict[str, float] = {}
            jk_counts: Dict[str, int] = {}

            for i, policy in enumerate(self.target_policies):
                var_orc = 0.0
                K = 0
                jack = self.get_oracle_jackknife(policy)
                if (
                    jack is not None
                    and len(jack) >= 2
                    and i < len(result.standard_errors)
                ):
                    K = len(jack)
                    var_orc = oracle_jackknife_variance(jack)

                var_oracle_map[policy] = var_orc
                jk_counts[policy] = K

                # Update standard_errors in place (add oracle variance)
                if i < len(result.standard_errors):
                    se_base = float(result.standard_errors[i])
                    result.standard_errors[i] = float(np.sqrt(se_base**2 + var_orc))

            # Record that oracle uncertainty has been added
            if isinstance(result.metadata, dict):
                result.metadata.setdefault("se_components", {})
                result.metadata["se_components"].update(
                    {
                        "includes_oracle_uncertainty": True,
                        "oracle_variance_per_policy": var_oracle_map,
                        "oracle_jackknife_counts": jk_counts,
                    }
                )
        except Exception as e:
            logger.debug(f"Calibration-aware oracle jackknife failed: {e}")

    def _oracle_coverage_for_oua(self) -> Optional[float]:
        """Oracle coverage consulted by the 100%-coverage OUA skip.

        Defaults to the calibrator's own coverage (fraction of its
        training data with oracle labels). Estimators whose evaluation
        data is disjoint from the calibration data (e.g. Direct mode with
        a separate calibration source) override this: the calibrator's
        coverage then describes a different dataset, and returning None
        keeps the jackknife active.
        """
        coverage = getattr(self.reward_calibrator, "oracle_coverage", None)
        return float(coverage) if coverage is not None else None

    def get_oracle_jackknife(self, policy: str) -> Optional[np.ndarray]:
        """Compute leave-one-oracle-fold jackknife estimates.

        This method should be overridden by estimators that support
        calibration-aware oracle jackknife paths. The default implementation
        returns None.

        Args:
            policy: Policy name to compute jackknife estimates for

        Returns:
            Array of K jackknife estimates (one per fold), or None if not supported
        """
        return None
