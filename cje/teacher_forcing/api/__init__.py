"""API implementations for teacher forcing.

Note: Fireworks API support requires optional 'teacher-forcing' dependencies.
Install with: pip install cje-eval[teacher-forcing]
"""

try:
    from .fireworks import compute_teacher_forced_logprob

    __all__ = [
        "compute_teacher_forced_logprob",
    ]
except ImportError:
    import warnings

    warnings.warn(
        "Fireworks API support requires optional dependencies. "
        "Install with: pip install cje-eval[teacher-forcing]",
        ImportWarning,
    )

    __all__ = []
