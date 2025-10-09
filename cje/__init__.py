"""CJE: Causal Judge Evaluation - Unbiased LLM Policy Evaluation.

Simple API for off-policy evaluation with judge scores.

Example:
    from cje import analyze_dataset

    results = analyze_dataset(
        "data.jsonl",
        estimator="calibrated-ips",
    )
    print(results.summary())
"""

__version__ = "0.2.1"

# Simple API - what 90% of users need
from .interface import analyze_dataset

# Core data structures
from .data import Dataset, Sample, EstimationResult

# Simple data loading
from .data import load_dataset_from_jsonl

__all__ = [
    # Simple API
    "analyze_dataset",
    # Core data structures
    "Dataset",
    "Sample",
    "EstimationResult",
    # Data loading
    "load_dataset_from_jsonl",
]
