"""
Utilities for exporting CJE results to various formats.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..data.models import EstimationResult


def export_results_json(
    results: EstimationResult,
    path: str,
    include_diagnostics: bool = True,
    include_metadata: bool = True,
    indent: int = 2,
) -> None:
    """
    Export estimation results to JSON format.

    Args:
        results: EstimationResult object to export
        path: Output file path
        include_diagnostics: Whether to include diagnostic information
        include_metadata: Whether to include metadata
        indent: JSON indentation level (None for compact)
    """
    # Prepare the export data
    export_data: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "method": results.method,
        "estimates": _serialize_array(results.estimates),
        "standard_errors": _serialize_array(results.standard_errors),
        "n_samples_used": results.n_samples_used,
    }

    # Add confidence intervals
    ci_lower, ci_upper = results.confidence_interval(alpha=0.05)
    export_data["confidence_intervals"] = {
        "alpha": 0.05,
        "lower": _serialize_array(ci_lower),
        "upper": _serialize_array(ci_upper),
    }

    # Add metadata if requested
    if include_metadata and results.metadata:
        export_data["metadata"] = _serialize_metadata(results.metadata)

    # Add diagnostics if requested
    if include_diagnostics:
        # Prefer the diagnostics object on the result, if present
        if getattr(results, "diagnostics", None) is not None:
            diag_obj = getattr(results, "diagnostics")
            # Try to_dict, else fallback to attribute dict
            if hasattr(diag_obj, "to_dict"):
                export_data["diagnostics"] = diag_obj.to_dict()
            else:
                try:
                    export_data["diagnostics"] = _serialize_diagnostics(
                        diag_obj.__dict__
                    )
                except Exception:
                    pass
        # Also allow diagnostics embedded in metadata for legacy paths
        elif "diagnostics" in results.metadata:
            export_data["diagnostics"] = _serialize_diagnostics(
                results.metadata.get("diagnostics", {})
            )

    # Add target policies if available
    if "target_policies" in results.metadata:
        export_data["target_policies"] = results.metadata["target_policies"]

        # Create policy-specific results for easier access
        export_data["per_policy_results"] = {}
        for i, policy in enumerate(results.metadata["target_policies"]):
            export_data["per_policy_results"][policy] = {
                "estimate": float(results.estimates[i]),
                "standard_error": float(results.standard_errors[i]),
                "ci_lower": float(ci_lower[i]),
                "ci_upper": float(ci_upper[i]),
                "n_samples": results.n_samples_used.get(policy, 0),
            }

    # Write to file
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w") as f:
        json.dump(export_data, f, indent=indent)


def export_results_csv(
    results: EstimationResult,
    path: str,
    include_ci: bool = True,
) -> None:
    """
    Export estimation results to CSV format.

    Args:
        results: EstimationResult object to export
        path: Output file path
        include_ci: Whether to include confidence intervals
    """
    import csv

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Prepare rows
    rows = []
    headers = ["policy", "estimate", "standard_error"]

    if include_ci:
        headers.extend(["ci_lower", "ci_upper"])
        ci_lower, ci_upper = results.confidence_interval(alpha=0.05)

    headers.extend(["n_samples", "method"])

    # Add data for each policy
    target_policies = results.metadata.get("target_policies", [])
    for i, policy in enumerate(target_policies):
        row = {
            "policy": policy,
            "estimate": results.estimates[i],
            "standard_error": results.standard_errors[i],
            "n_samples": results.n_samples_used.get(policy, 0),
            "method": results.method,
        }

        if include_ci:
            row["ci_lower"] = ci_lower[i]
            row["ci_upper"] = ci_upper[i]

        rows.append(row)

    # Write CSV
    with open(path_obj, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _serialize_array(arr: np.ndarray) -> list:
    """Convert numpy array to JSON-serializable list."""
    if isinstance(arr, np.ndarray):
        return [float(x) if not np.isnan(x) else None for x in arr]
    return list(arr)


def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize metadata, handling special types."""
    serialized: Dict[str, Any] = {}

    for key, value in metadata.items():
        # Skip complex objects that aren't easily serializable
        if key in ["diagnostics", "dr_diagnostics", "dr_calibration_data"]:
            continue

        if isinstance(value, np.ndarray):
            serialized[key] = _serialize_array(value)
        elif isinstance(value, (np.integer, np.floating)):
            serialized[key] = float(value)
        elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
            serialized[key] = value
        else:
            # Try to convert to string for other types
            try:
                serialized[key] = str(value)
            except:
                pass  # Skip if can't serialize

    return serialized


def _serialize_diagnostics(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize diagnostics, handling nested structures."""
    serialized: Dict[str, Any] = {}

    for key, value in diagnostics.items():
        if isinstance(value, dict):
            # Recursively serialize nested dicts
            serialized[key] = _serialize_diagnostics(value)
        elif isinstance(value, np.ndarray):
            serialized[key] = _serialize_array(value)
        elif isinstance(value, (np.integer, np.floating)):
            serialized[key] = float(value)
        elif hasattr(value, "__dict__"):
            # Handle diagnostic objects with attributes
            serialized[key] = {
                k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                for k, v in value.__dict__.items()
                if not k.startswith("_")
            }
        else:
            serialized[key] = value

    return serialized
