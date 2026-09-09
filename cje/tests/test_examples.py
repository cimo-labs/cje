"""Execute the real notebooks and protect their data-separation/setup contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import urllib.request

import nbformat
import numpy as np
import pytest
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbconvert.preprocessors import ExecutePreprocessor

from cje import analyze_dataset


pytestmark = [pytest.mark.e2e, pytest.mark.uses_arena_sample]
REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = REPO_ROOT / "examples" / "arena_sample"


def _read_notebook(name: str) -> Any:
    return nbformat.read(REPO_ROOT / "examples" / name, as_version=4)


def _prepare_notebook_for_local_repo_execution(nb: Any) -> Any:
    """Replace only installation; require this checkout and an offline data cache."""
    bootstrap = (
        "import sys, urllib.request\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "import cje\n"
        f"assert cje.__file__.startswith({str(REPO_ROOT)!r})\n"
        f"assert sys.executable == {sys.executable!r}\n"
        "def _unexpected_download(*args, **kwargs):\n"
        "    raise AssertionError('Notebook execution must use bundled sample data')\n"
        "urllib.request.urlretrieve = _unexpected_download\n"
    )
    replacements = 0
    for cell in nb.cells:
        if cell.cell_type == "code" and "pip install" in cell.source:
            assert "%pip install" in cell.source
            cell.source = bootstrap
            replacements += 1
    assert replacements == 1
    return nb


def _execute_notebook(nb: Any, run_dir: Path, name: str) -> None:
    """Use the test interpreter, never a user/system kernelspec or live network."""
    shutil.copytree(SAMPLE_DIR, run_dir / "arena_sample")
    kernels = run_dir / "kernels"
    kernel_dir = kernels / "cje-tests"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "CJE test interpreter",
                "language": "python",
                "env": {
                    "PYTHONPATH": str(REPO_ROOT),
                    "IPYTHONDIR": str(run_dir / "ipython"),
                    "MPLCONFIGDIR": str(run_dir / "matplotlib"),
                    "MPLBACKEND": "module://matplotlib_inline.backend_inline",
                },
            }
        )
    )
    manager = KernelManager(
        kernel_name="cje-tests",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]),
        connection_file=str(run_dir / "connection.json"),
    )
    executor = ExecutePreprocessor(timeout=600, allow_errors=False)
    try:
        executor.preprocess(nb, {"metadata": {"path": str(run_dir)}}, km=manager)
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
        destination = Path(os.environ.get("CJE_NOTEBOOK_OUTPUT_DIR", str(run_dir)))
        destination.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, destination / f"{name}.executed.ipynb")


class TestNotebookWalkthrough:
    def test_direct_mode_first(self, arena_fresh_draws: Any) -> None:
        """Exercise the documented calibrated API rather than averaging raw scores."""
        results = analyze_dataset(fresh_draws_dir=str(SAMPLE_DIR / "fresh_draws"))
        assert len(results.estimates) == len(arena_fresh_draws)
        assert np.all(np.isfinite(results.estimates))
        assert np.all(results.standard_errors > 0)
        lower, upper = results.confidence_interval()
        assert np.all(lower <= results.estimates)
        assert np.all(upper >= results.estimates)


class TestAdvancedNotebookStub:
    def test_advanced_notebook_is_a_markdown_stub(self) -> None:
        nb = _read_notebook("cje_advanced.ipynb")
        assert all(cell.cell_type == "markdown" for cell in nb.cells)
        text = "\n".join(cell.source for cell in nb.cells)
        assert 'pip install "cje-eval==0.3.*"' in text
        assert "v0.3.0" in text


@pytest.mark.slow
class TestNotebookExecution:
    @pytest.mark.parametrize(
        "name,production,pilot",
        [
            ("cje_core_demo", False, False),
            ("cje_planning", False, False),
            ("cje_planning", False, True),
            ("cje_planning", True, True),
        ],
        ids=["core", "planning-default", "planning-fast-pilot", "planning-production"],
    )
    def test_notebook(
        self, tmp_path: Path, name: str, production: bool, pilot: bool
    ) -> None:
        nb = _prepare_notebook_for_local_repo_execution(_read_notebook(f"{name}.ipynb"))
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            if production:
                cell.source = cell.source.replace(
                    "FAST_NOTEBOOK_MODE = True", "FAST_NOTEBOOK_MODE = False"
                )
            if pilot:
                cell.source = cell.source.replace(
                    "RUN_OPTIONAL_PILOT_REFINEMENT = False",
                    "RUN_OPTIONAL_PILOT_REFINEMENT = True",
                )

        if name == "cje_core_demo":
            checks = """
training_ids = {row['prompt_id'] for rows in evaluation_data.values()
                for row in rows if row.get('oracle_label') is not None}
audit_ids = {row['prompt_id'] for rows in probe_data.values() for row in rows}
weekly_ids = [set(row['prompt_id'] for row in rows) for rows in weeks.values()]
assert training_ids.isdisjoint(audit_ids)
assert len(set.union(*weekly_ids)) == sum(map(len, weekly_ids))
assert all(ids.isdisjoint(training_ids | audit_ids) for ids in weekly_ids)
assert all(a.family_size == 3 for a in [*audits.values(), *weekly_audits.values()])
assert all(len(rows) >= 20 for rows in [*probe_data.values(), *weeks.values()])
"""
        else:
            checks = (
                f"assert result.scenario_fingerprint['n_replicates'] == {50 if production else 2}\n"
                f"assert (pilot_model is not None) == {pilot!r}\n"
                f"assert RUN_FULL_R2_SWEEP == {production!r}\n"
            )
        nb.cells.append(nbformat.v4.new_code_cell(checks))
        variant = f"{name}-{'production' if production else 'fast'}-pilot-{pilot}"
        _execute_notebook(nb, tmp_path, variant)
        figures = [
            output
            for cell in nb.cells
            for output in cell.get("outputs", [])
            if "image/png" in output.get("data", {})
        ]
        assert len(figures) >= (4 if name == "cje_core_demo" else 1)


def _download_cell(name: str) -> str:
    return str(
        next(
            cell.source
            for cell in _read_notebook(name).cells
            if cell.cell_type == "code" and "def download_sample_files" in cell.source
        )
    )


@pytest.mark.parametrize("name", ["cje_core_demo.ipynb", "cje_planning.ipynb"])
def test_downloader_recovers_partial_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """A cached base file must not suppress other files; failed writes stay atomic."""
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "arena_sample" / "fresh_draws" / "base_responses.jsonl"
    base.parent.mkdir(parents=True)
    shutil.copyfile(SAMPLE_DIR / "fresh_draws" / base.name, base)
    calls = []
    fail_once = True

    def download(url: str, destination: Path) -> None:
        nonlocal fail_once
        assert "/9d6878e91393d47e802e4d4dc26378cff018344e/" in url
        relative = url.split("/examples/arena_sample/")[1]
        calls.append(relative)
        if fail_once:
            fail_once = False
            Path(destination).write_bytes(b"interrupted transfer")
            raise OSError("simulated interrupted download")
        shutil.copyfile(SAMPLE_DIR / relative, destination)

    monkeypatch.setattr(urllib.request, "urlretrieve", download)
    namespace: dict[str, Any] = {"RUN_OPTIONAL_PILOT_REFINEMENT": True}
    code = _download_cell(name)
    with pytest.raises(OSError, match="simulated interrupted"):
        exec(code, namespace)
    assert not list(tmp_path.rglob("*.part"))
    assert not (base.parent / "clone_responses.jsonl").exists()
    exec(code, namespace)
    for relative in namespace["required_files"]:
        assert (tmp_path / "arena_sample" / relative).read_bytes() == (
            SAMPLE_DIR / relative
        ).read_bytes()
    assert "fresh_draws/base_responses.jsonl" not in calls
    before = len(calls)
    exec(code, namespace)
    assert len(calls) == before


def test_core_downloader_after_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduce the original planning-then-core setup path with controlled URLs."""
    monkeypatch.chdir(tmp_path)
    calls = []

    def download(url: str, destination: Path) -> None:
        relative = url.split("/examples/arena_sample/")[1]
        calls.append(relative)
        shutil.copyfile(SAMPLE_DIR / relative, destination)

    monkeypatch.setattr(urllib.request, "urlretrieve", download)
    exec(_download_cell("cje_planning.ipynb"), {"RUN_OPTIONAL_PILOT_REFINEMENT": True})
    assert len(calls) == 4
    exec(_download_cell("cje_core_demo.ipynb"), {})
    assert len(calls) == 7
    assert all(path.startswith("probe_slice/") for path in calls[4:])
