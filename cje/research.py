"""Research API - Experimental and internal features.

WARNING: This module is for research and experimentation.
APIs here may change without notice. Use at your own risk.

This module exposes:
- Internal calibration algorithms
- Experimental estimators
- Low-level weight manipulation
- Research diagnostics
- Teacher forcing utilities

Example:
    from cje.research import (
        calibrate_to_target_mean,
        compute_teacher_forced_logprob,
        IsotonicOutcomeModel,
    )
"""

# Low-level calibration
from .calibration import (
    calibrate_to_target_mean,
)

# Outcome models for DR
try:
    from .estimators.outcome_models import (
        BaseOutcomeModel,
        IsotonicOutcomeModel,
        LinearOutcomeModel,
    )

    _outcome_models_available = True
except ImportError:
    _outcome_models_available = False

# Teacher forcing
from .teacher_forcing import (
    compute_teacher_forced_logprob,
    ChatTemplateConfig,
    Llama3TemplateConfig,
    HuggingFaceTemplateConfig,
    compute_chat_logprob,
    convert_chat_to_completions,
)

# Experimental estimators (currently none)
_experimental_available = False

# Fresh draw utilities
try:
    from .data.fresh_draws import (
        FreshDrawSample,
        validate_fresh_draws,
        create_synthetic_fresh_draws,
        save_fresh_draws_to_jsonl,
    )

    _fresh_draws_available = True
except ImportError:
    _fresh_draws_available = False

# Data validation
from .data.validation import (
    validate_cje_data,
    validate_for_precomputed_sampler,
)

# Low-level data components
from .data import (
    DatasetLoader,
    JsonlDataSource,
    InMemoryDataSource,
    LogProbStatus,
    LogProbResult,
    add_rewards_to_existing_data,
)

__all__ = [
    # Low-level calibration
    "calibrate_to_target_mean",
    # Teacher forcing
    "compute_teacher_forced_logprob",
    "ChatTemplateConfig",
    "Llama3TemplateConfig",
    "HuggingFaceTemplateConfig",
    "compute_chat_logprob",
    "convert_chat_to_completions",
    # Data validation
    "validate_cje_data",
    "validate_for_precomputed_sampler",
    # Low-level data
    "DatasetLoader",
    "JsonlDataSource",
    "InMemoryDataSource",
    "LogProbStatus",
    "LogProbResult",
    "add_rewards_to_existing_data",
]

if _outcome_models_available:
    __all__.extend(
        [
            "BaseOutcomeModel",
            "IsotonicOutcomeModel",
            "LinearOutcomeModel",
        ]
    )

# No experimental estimators currently

if _fresh_draws_available:
    __all__.extend(
        [
            "FreshDrawSample",
            "validate_fresh_draws",
            "create_synthetic_fresh_draws",
            "save_fresh_draws_to_jsonl",
        ]
    )
