"""Verify an installed core-only wheel from outside the source checkout."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import cje
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()
    assert cje.__version__ == args.expected_version
    assert Path(cje.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert importlib.util.find_spec("matplotlib") is None, "Use a core-only venv"
    draws = {
        policy: [
            {
                "prompt_id": f"q{i}",
                "judge_score": (i + 1) / 81,
                "oracle_label": (
                    (i + 1) / 81 if policy == "base" and i % 2 == 0 else None
                ),
            }
            for i in range(80)
        ]
        for policy in ("base", "candidate")
    }
    result = cje.analyze_dataset(fresh_draws_data=draws)
    assert result.method == "calibrated_direct"
    assert np.all(np.isfinite(result.estimates))
    assert np.all(result.standard_errors > 0)
    with tempfile.TemporaryDirectory(prefix="cje-wheel-") as directory:
        data = Path(directory) / "fresh.jsonl"
        data.write_text(
            "".join(
                json.dumps({**row, "target_policy": policy}, allow_nan=False) + "\n"
                for policy, rows in draws.items()
                for row in rows
            )
        )
        # Check the installed console entrypoint as well as the module CLI.
        subprocess.run(
            [str(Path(sys.executable).with_name("cje")), "--help"],
            check=True,
            cwd=directory,
        )
        for command in ("validate", "analyze"):
            subprocess.run(
                [sys.executable, "-m", "cje", command, str(data)],
                check=True,
                cwd=directory,
            )
    print(f"Installed core-only cje-eval {cje.__version__}: API and CLI smoke passed")


if __name__ == "__main__":
    main()
