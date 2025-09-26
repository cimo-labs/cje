"""API implementations for teacher forcing."""

from .fireworks import compute_teacher_forced_logprob

__all__ = [
    "compute_teacher_forced_logprob",
]
