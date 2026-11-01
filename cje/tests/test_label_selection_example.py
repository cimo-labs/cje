"""Exercise the bundled label-selection example's output contract."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "examples" / "label_selection" / "run.py"


def test_label_selection_example_writes_expected_structure(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"

    proc = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--repetitions",
            "2",
            "--output",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]

    runs = json.loads((output_dir / "runs.json").read_text())
    summary = json.loads((output_dir / "summary.json").read_text())
    with (output_dir / "summary.csv").open(newline="") as stream:
        summary_rows = list(csv.DictReader(stream))

    assert len(runs) == 12
    assert len(summary["summary"]) == 18
    assert len(summary_rows) == 18
    assert all(count == 60 for run in runs for count in run["label_counts"].values())
