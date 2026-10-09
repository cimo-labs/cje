"""0.5.2 legacy-line markers: the import-time stderr banner and the first
line of EstimationResult.summary()."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from cje.data.models import EstimationResult

BANNER = (
    "cje-eval 0.5.x is the legacy Python 3.9 line. Current releases (0.9+) need "
    "Python 3.10-3.13: pip install -U 'cje-eval>=0.9.2'. This version's API and "
    "outputs differ from the current docs."
)
LEGACY_LINE = "LEGACY cje-eval 0.5.x (Python 3.9): current docs describe 0.9+."


def _import_cje_stderr(extra: str = "") -> str:
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", extra + "import cje\n"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert BANNER not in result.stdout
    return result.stderr


def test_import_prints_banner_once_to_stderr() -> None:
    assert _import_cje_stderr().count(BANNER) == 1


def test_banner_survives_silenced_logging_and_warnings() -> None:
    silence = (
        "import logging, warnings\n"
        "logging.disable(logging.CRITICAL)\n"
        "warnings.simplefilter('ignore')\n"
    )
    assert _import_cje_stderr(silence).count(BANNER) == 1


def test_summary_first_line_is_legacy_marker() -> None:
    result = EstimationResult(
        estimates=np.array([0.5, 0.7]),
        standard_errors=np.array([0.02, 0.03]),
        n_samples_used={"a": 100, "b": 100},
        method="calibrated_direct",
        metadata={"target_policies": ["a", "b"]},
    )
    lines = result.summary().splitlines()
    assert lines[0] == LEGACY_LINE
    assert lines[1].startswith("CJE Estimation Results")


def test_summary_without_policies_also_starts_with_legacy_marker() -> None:
    result = EstimationResult(
        estimates=np.array([0.5]),
        standard_errors=np.array([0.02]),
        n_samples_used={},
        method="calibrated_direct",
        metadata={},
    )
    assert result.summary().splitlines()[0] == LEGACY_LINE
