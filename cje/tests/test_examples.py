"""End-to-end tests for example notebooks and tutorials.

These tests validate the complete walkthrough from the notebook examples,
ensuring the documented usage patterns work correctly.
"""

import pytest
import numpy as np
from typing import Any
from pathlib import Path


# Mark all tests in this file as E2E tests
pytestmark = [pytest.mark.e2e, pytest.mark.uses_arena_sample]


def _prepare_notebook_for_local_repo_execution(nb: Any) -> Any:
    """Rewrite install cells so notebook tests exercise the local checkout."""
    repo_root = Path(__file__).parent.parent.parent
    bootstrap_local_repo = (
        "import sys\n"
        f"repo_root = {str(repo_root)!r}\n"
        "if repo_root not in sys.path:\n"
        "    sys.path.insert(0, repo_root)\n"
        "print(f'Using local CJE checkout: {repo_root}')\n"
    )

    for cell in nb.cells:
        if (
            cell.cell_type == "code"
            and "pip install" in cell.source
            and "cje-eval" in cell.source
        ):
            cell.source = bootstrap_local_repo

    return nb


class TestNotebookWalkthrough:
    """Test the notebook walkthrough (Direct mode)."""

    def test_direct_mode_first(self, arena_fresh_draws: Any) -> None:
        """Test Direct mode (Step 3 in notebook) - simplest mode with fresh draws only."""
        # Direct mode: Fresh draws only (no logged data needed)

        # Get policies from fresh draws
        policies = list(arena_fresh_draws.keys())

        # Simple direct estimation: average judge scores per policy
        estimates = []
        std_errors = []

        for policy in policies:
            dataset = arena_fresh_draws[policy]
            judge_scores = [
                s.judge_score for s in dataset.samples if s.judge_score is not None
            ]

            if judge_scores:
                est = np.mean(judge_scores)
                se = np.std(judge_scores) / np.sqrt(len(judge_scores))
                estimates.append(est)
                std_errors.append(se)
            else:
                estimates.append(np.nan)
                std_errors.append(np.nan)

        # Validate Direct mode results
        valid_estimates = [e for e in estimates if not np.isnan(e)]
        assert len(valid_estimates) >= 2, "Should have estimates for multiple policies"
        assert all(0 <= e <= 1 for e in valid_estimates), "Estimates should be in [0,1]"
        assert all(
            se > 0 for se, e in zip(std_errors, estimates) if not np.isnan(e)
        ), "Should have positive SEs"

        print(f"✓ Direct mode: {len(valid_estimates)} policies estimated")


class TestAdvancedNotebookStub:
    """cje_advanced.ipynb is a stub since 0.4.0 (OPE moved to the 0.3.x line)."""

    def test_advanced_notebook_is_a_markdown_stub(self) -> None:
        pytest.importorskip("nbformat")
        import nbformat

        notebook_path = (
            Path(__file__).parent.parent.parent / "examples" / "cje_advanced.ipynb"
        )
        assert notebook_path.exists(), f"Notebook not found at {notebook_path}"

        nb = nbformat.read(notebook_path, as_version=4)

        # No executable cells: the OPE walkthrough lives at the v0.3.0 tag.
        assert all(cell.cell_type == "markdown" for cell in nb.cells)
        text = "\n".join("".join(cell.source) for cell in nb.cells)
        assert 'pip install "cje-eval==0.3.*"' in text
        assert "v0.3.0" in text


@pytest.mark.slow
class TestNotebookExecution:
    """Test that the actual notebooks execute without errors."""

    def test_tutorial_notebook(self) -> None:
        """Execute the quick start tutorial notebook (Direct mode only).

        This test catches issues that the API tests miss, such as:
        - KeyError from direct dict access without .get()
        - Missing imports in cells
        - Broken cell execution order
        - Invalid markdown or formatting

        Uses nbconvert to execute all cells in order, simulating Colab execution.
        """
        pytest.importorskip("nbformat")
        pytest.importorskip("nbconvert")

        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        from pathlib import Path

        # Find the notebook
        notebook_path = (
            Path(__file__).parent.parent.parent / "examples" / "cje_core_demo.ipynb"
        )
        assert notebook_path.exists(), f"Notebook not found at {notebook_path}"

        # Read the notebook
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        nb = _prepare_notebook_for_local_repo_execution(nb)

        # Execute all cells
        # Note: This will download data, install packages, etc. - mark as slow test
        ep = ExecutePreprocessor(
            timeout=600,  # 10 minutes max
            kernel_name="python3",
            allow_errors=False,  # Fail on any cell error
        )

        try:
            # Execute in a temporary directory to avoid polluting the repo
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                ep.preprocess(nb, {"metadata": {"path": tmpdir}})
        except Exception as e:
            pytest.fail(f"Tutorial notebook execution failed: {e}")

        print("✓ Tutorial notebook executed successfully")

    def test_planning_notebook(self) -> None:
        """Execute the planning notebook (budget optimization).

        Tests the planning tutorial covering:
        - Variance model fitting
        - Budget-constrained planning
        - MDE-constrained planning
        - Planning dashboard visualization
        """
        pytest.importorskip("nbformat")
        pytest.importorskip("nbconvert")

        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        from pathlib import Path

        # Find the notebook
        notebook_path = (
            Path(__file__).parent.parent.parent / "examples" / "cje_planning.ipynb"
        )
        assert notebook_path.exists(), f"Notebook not found at {notebook_path}"

        # Read the notebook
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        nb = _prepare_notebook_for_local_repo_execution(nb)

        # Execute all cells
        ep = ExecutePreprocessor(
            timeout=600,  # 10 minutes max
            kernel_name="python3",
            allow_errors=False,  # Fail on any cell error
        )

        try:
            # Execute in a temporary directory to avoid polluting the repo
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                ep.preprocess(nb, {"metadata": {"path": tmpdir}})
        except Exception as e:
            pytest.fail(f"Planning notebook execution failed: {e}")

        print("✓ Planning notebook executed successfully")
