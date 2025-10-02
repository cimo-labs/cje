"""Core CJE estimators and types.

This module contains:
- Estimators: CalibratedIPS and base classes
- Data models: Pydantic models for type safety
- Types: Data structures for results and error handling
"""

from .base_estimator import BaseCJEEstimator
from .calibrated_ips import CalibratedIPS
from .direct_method import CalibratedDirectEstimator
from .orthogonalized_ips import OrthogonalizedCalibratedIPS
from .orthogonalized_calibrated_dr import OrthogonalizedCalibratedDRCPO
from .stacking import StackedDREstimator
from ..data.models import (
    Sample,
    Dataset,
    EstimationResult,
    LogProbResult,
    LogProbStatus,
)

# Import DR estimators for convenience
try:
    from .dr_base import DRCPOEstimator
    from .tmle import TMLEEstimator
    from .mrdr import MRDREstimator
    from .tr_cpo import TRCPOEstimator

    _dr_available = True
except ImportError:
    _dr_available = False

__all__ = [
    # Estimators
    "BaseCJEEstimator",
    "CalibratedIPS",
    "CalibratedDirectEstimator",
    "OrthogonalizedCalibratedIPS",
    "OrthogonalizedCalibratedDRCPO",
    "StackedDREstimator",
    # Data models
    "Sample",
    "Dataset",
    "EstimationResult",
    # Types
    "LogProbResult",
    "LogProbStatus",
]

if _dr_available:
    __all__.extend(
        ["DRCPOEstimator", "TMLEEstimator", "MRDREstimator", "TRCPOEstimator"]
    )
