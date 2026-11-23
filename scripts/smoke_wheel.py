"""Verify an installed core-only wheel from outside the source checkout."""

import argparse
import importlib.util
import json
from pathlib import Path
import re
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
    historical = [
        {
            "prompt_id": f"historical-{i}",
            "judge_score": i / 39,
            "oracle_label": i / 39,
            "fit_weight": 1 + i % 2,
        }
        for i in range(40)
    ]
    external = cje.analyze_dataset(
        fresh_draws_data=draws,
        calibration_data=historical,
        calibration_weight_field="fit_weight",
        combine_oracle_sources=False,
    )
    assert external.metadata["calibration_weighting"]["n_labeled_rows"] == 40
    plan = cje.plan_binary_reviews(objective="slope", judge_fail_share=0.1, budget=20)
    assert (plan.fail_reviews, plan.pass_reviews) == (10, 10)
    audit = cje.weighted_paired_audit(
        cluster_ids=["a", "b"],
        traffic_selected=[True, True],
        label_selected=[True, True],
        traffic_inclusion=[1.0, 1.0],
        label_inclusion=[1.0, 1.0],
        predictions=[0.5, 0.5],
        outcomes=[0.0, 1.0],
        outcome_bounds=(0.0, 1.0),
        target_weights=[1, 3],
    )
    assert audit.estimate == 0.75 and audit.ci == (0.75, 0.75)
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
        # The README's agent handoff runs `cje skill`: the bundled skill and
        # reference must ship in the wheel.
        for extra, marker in (
            ([], "## Hard rules"),
            (["--reference"], "# CJE reference"),
        ):
            shown = subprocess.run(
                [sys.executable, "-m", "cje", "skill", *extra],
                check=True,
                cwd=directory,
                capture_output=True,
                text=True,
            ).stdout
            assert marker in shown, f"cje skill {extra} printed no {marker!r}"
            if extra == ["--reference"]:
                section = shown.split("\n## Sampling designs\n", 1)[1].split(
                    "\n## ", 1
                )[0]
                recipes = re.findall(
                    r"^```python\n(.*?)^```$", section, re.MULTILINE | re.DOTALL
                )
                assert (
                    len(recipes) >= 2
                ), "Missing offline review planning or audit example"
                for recipe in recipes:
                    subprocess.run(
                        [sys.executable, "-c", recipe], check=True, cwd=directory
                    )
    print(f"Installed core-only cje-eval {cje.__version__}: API and CLI smoke passed")


if __name__ == "__main__":
    main()
