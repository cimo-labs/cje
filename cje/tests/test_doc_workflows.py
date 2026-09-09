"""Execute documented calls with explicit synthetic inputs for their placeholders.

Bodies are executed unchanged. The separate snippet suite verifies their imports;
this suite covers the work those imports enable, including multi-step estimators.
"""

import ast
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cje import analyze_dataset, transport_audit
from cje.calibration import calibrate_dataset
from cje.data import Dataset, FreshDrawDataset, Sample, fresh_draws_from_dict
from cje.estimators import CalibratedDirectEstimator
from cje.tests.test_doc_snippets import REPO_ROOT, SNIPPETS


def _is_illustrative(path: str, code: str) -> bool:
    """Keep signature sketches and intentionally removed APIs in syntax tests."""
    return (
        path == "cje/tests/README.md"
        or path == "MIGRATING-0.6.md"
        or "fresh_draws_data=None" in code
        or code.startswith("def ")
    )


def _workflow_cases() -> List[Any]:
    cases = []
    for path, line, code in SNIPPETS:
        if _is_illustrative(path, code):
            continue
        # Keep the documented 2,000 refits; put expensive workflows in slow CI.
        bootstrap = any(
            isinstance(node, ast.Constant) and node.value == "bootstrap"
            for node in ast.walk(ast.parse(code))
        )
        cases.append(
            pytest.param(
                path,
                line,
                code,
                id=f"{path}:{line}",
                marks=[pytest.mark.slow] if bootstrap else [],
            )
        )
    return cases


@pytest.fixture(scope="module")
def doc_inputs(tmp_path_factory: pytest.TempPathFactory) -> Tuple[Path, Dict[str, Any]]:
    root = tmp_path_factory.mktemp("doc-workflows")
    # The two Arena examples intentionally exercise the shipped sample files.
    (root / "examples").symlink_to(REPO_ROOT / "examples", target_is_directory=True)
    rng = np.random.default_rng(124)
    scores = rng.uniform(0.01, 0.99, 600)
    truth = np.clip(0.15 + 0.65 * scores + rng.normal(0, 0.10, 600), 0, 1)
    labels = np.full(600, np.nan)
    indices = rng.choice(600, 300, replace=False)
    labels[indices] = truth[indices]
    rows: List[Dict[str, Any]] = [
        {
            "prompt_id": f"p{i}",
            "judge_score": float(score),
            "oracle_label": float(labels[i]) if np.isfinite(labels[i]) else None,
            "response": "example response " * (10 + i % 3),
            "metadata": {"domain": i % 3, "difficulty": i % 4},
        }
        for i, score in enumerate(scores)
    ]
    probe_scores = rng.uniform(0.1, 0.9, 80)
    probe_labels = np.clip(0.15 + 0.65 * probe_scores + rng.normal(0, 0.03, 80), 0, 1)
    probes = [
        {
            "prompt_id": f"probe{i}",
            "judge_score": float(score),
            "oracle_label": float(label),
            "response": "independent probe",
            "metadata": {"domain": i % 3, "difficulty": i % 4},
        }
        for i, (score, label) in enumerate(zip(probe_scores, probe_labels))
    ]
    assert {r["prompt_id"] for r in rows}.isdisjoint(r["prompt_id"] for r in probes)

    def save_rows(path: Path, records: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in records)
        )

    for directory in ("responses", "responses/current_batch", "responses/pilot"):
        for policy in ("base", "model_a", "model_b"):
            save_rows(root / directory / f"{policy}_responses.jsonl", rows)
    for filename in (
        "calibration.jsonl",
        "human_labels.jsonl",
        "probes/policy_gpt56mini.jsonl",
        "probes/new_policy.jsonl",
        "gpt56_mini_probe.jsonl",
    ):
        save_rows(root / filename, probes)
    datasets, _ = fresh_draws_from_dict(
        {policy: rows for policy in ("base", "policy_a", "policy_b")}
    )
    dataset = Dataset(
        samples=[Sample(prompt="", **row) for row in rows], target_policies=[]
    )
    _, cal_result = calibrate_dataset(dataset)
    draws = {
        policy: rows for policy in ("base", "candidate", "fable-5", "gpt-5.6-mini")
    }
    results = analyze_dataset(fresh_draws_data=draws)
    namespace: Dict[str, Any] = {
        # These imports are introduced by earlier snippets in the same section.
        "np": np,
        "analyze_dataset": analyze_dataset,
        "transport_audit": transport_audit,
        "CalibratedDirectEstimator": CalibratedDirectEstimator,
        "results": results,
        "result": results,
        "draws": draws,
        "dataset": dataset,
        "cal_result": cal_result,
        "fresh_draws": datasets["base"],
        "pilot_data": datasets["base"],
        "base_pilot_data": datasets["base"],
        "fresh_draws_a": datasets["policy_a"],
        "fresh_draws_b": datasets["policy_b"],
        "judge_scores": scores,
        "scores": scores,
        "labels": labels,
        "oracle_labels": labels,
        "oracle_mask": np.isfinite(labels),
        "held_out_probe_rows": probes,
        "probe_rows": probes,
        "probe_scores": probe_scores,
        "probe_labels": probe_labels,
        "calibrator": results.calibrator,
        "prompt_ids": [f"probe{i}" for i in range(len(probes))],
        "n_groups": 4,
        "pilot_rows": rows,
        "probe": probes,
        "held_out_idx": np.arange(100),
    }
    return root, namespace


@pytest.mark.parametrize("path,line,code", _workflow_cases())
def test_documented_workflow_executes(
    path: str,
    line: int,
    code: str,
    doc_inputs: Tuple[Path, Dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, inputs = doc_inputs
    monkeypatch.chdir(root)
    namespace: Dict[str, Any] = {
        key: (
            copy.deepcopy(value)
            if isinstance(value, (np.ndarray, list, dict, Dataset, FreshDrawDataset))
            else value
        )
        for key, value in inputs.items()
    }
    if path == "cje/calibration/README.md" and "calibrator.fit_cv(" in code:
        # The lower-level fit_cv API takes compact labels with an explicit mask.
        namespace["oracle_labels"] = inputs["oracle_labels"][inputs["oracle_mask"]]
    try:
        exec(compile(code, f"{path}:{line}", "exec"), namespace)
        if path == "cje/estimators/README.md" and "fit_and_estimate()" in code:
            result = namespace["result"]
            assert result.metadata["target_policies"] == ["policy_a", "policy_b"]
            assert np.all(np.isfinite(result.estimates))
            assert np.all(result.standard_errors > 0)
            np.testing.assert_allclose(result.estimates[0], result.estimates[1])
    finally:
        plt.close("all")
