"""Real-data integration check of the documented, replayable planning example."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cje import analyze_dataset

pytestmark = [pytest.mark.e2e, pytest.mark.uses_arena_sample]


def test_planning_example_replays_heldout_comparison(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "skills/cje/scripts/planning_example.py"
    subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path)],
        check=True,
        cwd=root,
        capture_output=True,
        text=True,
    )
    design = json.loads((tmp_path / "design.json").read_text())
    selection = json.loads((tmp_path / "selection.json").read_text())
    inputs = json.loads((tmp_path / "analysis_input.json").read_text())
    audit = json.loads((tmp_path / "audit.json").read_text())
    plan = audit["planning"]["budget_plan"]
    assert not set(design["pilot_ids"]) & set(selection["eval_ids"])
    assert set(selection["eval_ids"]) <= set(design["heldout_pool_ids"])
    assert set(selection["label_ids"]) <= set(selection["eval_ids"])
    assert len(selection["eval_ids"]) == plan["n_samples"]
    assert len(selection["label_ids"]) == plan["m_oracle"]
    assert sum("oracle_label" in r for r in inputs["base"]) == plan["m_oracle"]
    assert all("oracle_label" not in r for r in inputs["parallel_universe_prompt"])
    assert (
        len(inputs["base"])
        == len(inputs["parallel_universe_prompt"])
        == plan["n_samples"]
    )
    assert plan["total_cost"] <= design["config"]["budget"]
    assert plan["total_cost"] == pytest.approx(
        0.02 * plan["n_samples"] + plan["m_oracle"]
    )
    for source in design["sources"].values():
        assert (
            hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest()
            == source["sha256"]
        )
    assert design["versions"]["cje-eval"] and (tmp_path / "rerun.sh").exists()
    assert all(
        (tmp_path / f).exists() for f in ["run.log", "console.log", "warnings.json"]
    )
    replay = analyze_dataset(fresh_draws_data=inputs, **design["analysis_config"])
    comparison = replay.compare_policies(
        replay.target_policies.index("parallel_universe_prompt"),
        replay.target_policies.index("base"),
        alpha=design["config"]["alpha"],
    )
    for key in ["difference", "se_difference", "ci_lower", "ci_upper", "p_value"]:
        assert comparison[key] == pytest.approx(audit["comparison"][key], abs=1e-12)
    assert comparison["method"] == audit["comparison"]["method"] == "paired_if_oua"
    assert (
        audit["transport_audits"]["parallel_universe_prompt"]["status"] == "NOT_CHECKED"
    )
    assert audit["power_validated"] is False and audit["coverage_validated"] is False
