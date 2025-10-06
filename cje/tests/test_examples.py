"""Test that all example scripts in examples/ run without errors.

This ensures examples stay in sync with the library as it evolves.
"""

import subprocess
from pathlib import Path
import pytest


pytestmark = [pytest.mark.integration, pytest.mark.uses_arena_sample]


def _get_example_scripts() -> list[Path]:
    """Find all Python scripts in examples/ directory."""
    examples_dir = Path(__file__).parent.parent.parent / "examples"

    if not examples_dir.exists():
        pytest.skip(f"Examples directory not found: {examples_dir}")

    # Get all .py files, excluding __pycache__ and other non-example files
    scripts = [
        script
        for script in examples_dir.glob("*.py")
        if script.is_file() and not script.name.startswith("_")
    ]

    scripts.sort()  # Ensure consistent order
    return scripts


@pytest.mark.parametrize("script_path", _get_example_scripts(), ids=lambda p: p.name)
def test_example_script_runs(script_path: Path) -> None:
    """Test that an example script runs without errors.

    Args:
        script_path: Path to the example script to run
    """
    # Run the script in a subprocess
    result = subprocess.run(
        ["python", str(script_path)],
        capture_output=True,
        text=True,
        timeout=60,  # 60 second timeout
    )

    # Check it exited successfully
    if result.returncode != 0:
        error_msg = (
            f"Example script {script_path.name} failed with exit code {result.returncode}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )
        pytest.fail(error_msg)

    # Script ran successfully
    assert result.returncode == 0


def test_all_examples_discovered() -> None:
    """Verify we found the expected example scripts."""
    scripts = _get_example_scripts()
    script_names = {script.name for script in scripts}

    # Expected examples (update this list when adding new examples)
    expected = {
        "01_understanding_modes.py",
        "02_minimal.py",
        "03_with_fresh_draws.py",
        "04_comparing_policies.py",
        "05_checking_reliability.py",
    }

    assert script_names == expected, (
        f"Unexpected example scripts found.\n"
        f"Expected: {sorted(expected)}\n"
        f"Found: {sorted(script_names)}\n"
        f"Missing: {sorted(expected - script_names)}\n"
        f"Extra: {sorted(script_names - expected)}"
    )
