"""Execute the public example and verify fit/label roles and corrected inference."""

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "audit_correction.py"


def test_audit_correction_example(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("audit_correction_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before, after, counts = module.run_example(tmp_path)
    assert counts == {
        "calibration_labels": 40,
        "correction_labels": 120,
        "total_human_labels": 160,
    }
    assert before.metadata["oracle_sources"]["total_oracle"] == 40
    assert after.metadata["oracle_sources"] == before.metadata["oracle_sources"]
    assert (
        after.metadata["data_provenance"]["calibration_fit_rows"]
        == before.metadata["data_provenance"]["calibration_fit_rows"]
    )
    scores = np.linspace(0.2, 0.6, 113)
    np.testing.assert_array_equal(
        before.calibrator.predict(scores), after.calibrator.predict(scores)
    )
    assert set(before.metadata["point_estimator"]["routes"]) == {"plug_in"}
    assert set(after.metadata["point_estimator"]["routes"]) == {"augmented"}
    assert before.metadata["transport_audits"]["candidate"]["status"] == "FAIL"
    assert after.metadata["transport_audits"]["candidate"]["status"] == "NOT_CHECKED"
    original = module.summarize(before)["candidate_minus_baseline"]
    corrected = module.summarize(after)["candidate_minus_baseline"]
    assert abs(original["estimate"]) < 1e-12
    assert abs(corrected["estimate"] - 0.15) < 0.025
    assert corrected["method"] == "paired_if_oua"
    assert corrected["se"] > 0 and corrected["se"] != original["se"]
    assert corrected["ci"][0] < corrected["estimate"] < corrected["ci"][1]
