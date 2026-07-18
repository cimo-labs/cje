"""
Utilities for exporting CJE results to various formats.
"""

import json
from pathlib import Path
from datetime import datetime

from ..data.models import EstimationResult


def export_results_json(
    results: EstimationResult,
    path: str,
    include_diagnostics: bool = True,
    include_metadata: bool = True,
    indent: int = 2,
    detail: str = "portable",
) -> None:
    """
    Export estimation results to JSON format.

    Args:
        results: EstimationResult object to export
        path: Output file path
        include_diagnostics: Whether to include diagnostic information
        include_metadata: Whether to include metadata
        indent: JSON indentation level (None for compact)
        detail: Serialization detail: ``"summary"``, ``"portable"``
            (default), or ``"full"``. Full detail includes bootstrap and
            influence arrays when present.
    """
    export_data = results.to_dict(detail=detail)
    export_data["exported_at"] = datetime.now().isoformat()
    if not include_metadata:
        export_data.pop("metadata", None)
    if not include_diagnostics:
        export_data.pop("diagnostics", None)

    # Write to file
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w") as f:
        json.dump(export_data, f, indent=indent, allow_nan=False)
