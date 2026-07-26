"""The plot_* names must raise an actionable ImportError on a no-viz install.

0.3.0 warned at `import cje` time (ImportWarning every non-viz user saw)
and then raised a bare AttributeError on plot_* access. 0.4.0 imports
silently and resolves plot_* lazily: with matplotlib installed the names
work as before; without it, access raises ImportError with the
`pip install "cje-eval[viz]"` hint.
"""

import importlib
import sys

import pytest

pytestmark = pytest.mark.unit

VIZ_NAMES = (
    "plot_policy_estimates",
    "plot_planning_dashboard",
)


def test_viz_names_resolve_when_matplotlib_installed() -> None:
    pytest.importorskip("matplotlib")
    import cje

    for name in VIZ_NAMES:
        assert callable(getattr(cje, name))
        assert name in cje.__all__


def test_advanced_viz_name_resolves_lazily() -> None:
    pytest.importorskip("matplotlib")
    import cje.advanced as advanced
    from cje.visualization import plot_policy_estimates

    assert advanced.plot_policy_estimates is plot_policy_estimates


def test_transport_comparison_reexport_with_matplotlib() -> None:
    """The plot moved to cje.visualization.transport in 0.5.0, but the
    cje.diagnostics import path the demo notebook uses must keep working."""
    pytest.importorskip("matplotlib")
    from cje.diagnostics import plot_transport_comparison
    from cje.visualization.transport import (
        plot_transport_comparison as viz_plot_transport_comparison,
    )

    assert plot_transport_comparison is viz_plot_transport_comparison


def test_import_cje_advanced_is_matplotlib_free() -> None:
    """`import cje.advanced` must not load matplotlib (D8): the 0.4.x eager
    try/except viz import block is gone; plot_* resolves lazily."""
    import os
    import subprocess
    from pathlib import Path

    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    code = (
        "import cje.advanced, sys\n"
        "assert 'matplotlib' not in sys.modules, 'matplotlib was imported'\n"
        "print('matplotlib' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_unknown_attribute_still_raises_attribute_error() -> None:
    import cje

    with pytest.raises(AttributeError, match="no attribute"):
        _ = cje.not_a_real_name


def test_viz_hint_when_matplotlib_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a no-viz install: block matplotlib/seaborn and re-import
    cje.visualization so its import-time probe fails."""
    import cje

    original_viz_attr = getattr(cje, "visualization", None)

    for module in list(sys.modules):
        if module.startswith(("matplotlib", "seaborn", "cje.visualization")):
            monkeypatch.delitem(sys.modules, module, raising=False)
    # None in sys.modules makes `import matplotlib` raise ImportError
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "seaborn", None)

    try:
        viz = importlib.import_module("cje.visualization")

        # Importing must be silent (the 0.3.0 ImportWarning is gone) and
        # attribute access must raise the actionable hint...
        with pytest.raises(
            ImportError,
            match=r"plot_policy_estimates requires the viz extra. "
            r'Install with: pip install "cje-eval\[viz\]"',
        ):
            _ = viz.plot_policy_estimates

        # ...including through the top-level lazy attribute
        with pytest.raises(ImportError, match=r"cje-eval\[viz\]"):
            _ = cje.plot_planning_dashboard

        # The lazily re-exported transport plot stays importable without
        # matplotlib (the demo notebook imports it eagerly) and raises the
        # hint only when CALLED
        from cje.diagnostics import plot_transport_comparison

        assert callable(plot_transport_comparison)
        with pytest.raises(ImportError, match=r"cje-eval\[viz\]"):
            plot_transport_comparison({})

        # Unknown names still raise AttributeError, not the hint
        with pytest.raises(AttributeError):
            _ = viz.not_a_plot
    finally:
        # importlib.import_module rebinds cje.visualization to the broken
        # module object; restore the original binding for later tests
        # (monkeypatch only restores sys.modules).
        if original_viz_attr is not None:
            cje.visualization = original_viz_attr
        elif hasattr(cje, "visualization"):
            del cje.visualization


def test_import_cje_emits_no_import_warning() -> None:
    """`import cje` must not warn even without viz extras: the eager
    visualization import (and its ImportWarning) is gone."""
    import os
    import subprocess
    from pathlib import Path

    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    code = (
        "import sys, warnings\n"
        "warnings.simplefilter('error', ImportWarning)\n"
        "import cje\n"
        "assert 'cje.visualization' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
