"""The agent skill ships in the package and `cje skill` prints it.

skills/cje/ is the canonical copy (fetched by URL, installed into agent skill
directories). The wheel carries byte-identical copies at cje/.agents/skills/cje/
(library-skills convention) so an installed cje-eval can hand its own,
version-matched skill to a coding agent: `cje skill` prints SKILL.md,
`--reference` prints reference.md, `--path` prints the folder. Refresh the
bundled copies with `make sync-skill`.
"""

import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import cje
from cje.interface import cli

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DIR = REPO_ROOT / "skills" / "cje"
BUNDLED_DIR = REPO_ROOT / "cje" / ".agents" / "skills" / "cje"
SKILL_FILES = ("SKILL.md", "reference.md")


def _entry_point() -> str:
    """The `cje` console-script target declared in pyproject.toml."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    match = re.search(r'^cje\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert match, "pyproject.toml declares no `cje` console script"
    return match.group(1)


def _run_entry_point(*argv: str) -> int:
    module_name, _, attr = _entry_point().partition(":")
    status = getattr(importlib.import_module(module_name), attr)(list(argv))
    assert isinstance(status, int)
    return status


@pytest.mark.parametrize("name", SKILL_FILES)
def test_bundled_skill_is_byte_identical_to_canonical(name: str) -> None:
    canonical = CANONICAL_DIR / name
    bundled = BUNDLED_DIR / name
    assert canonical.is_file(), canonical
    assert bundled.is_file(), f"{bundled} is missing; run `make sync-skill`"
    assert bundled.read_bytes() == canonical.read_bytes(), (
        f"cje/.agents/skills/cje/{name} has drifted from skills/cje/{name}; "
        "run `make sync-skill`"
    )


def test_cli_resolves_the_bundled_folder() -> None:
    assert cli.skill_dir() == BUNDLED_DIR.resolve()
    assert cli.SKILL_FILES == SKILL_FILES


def test_entry_point_targets_cli_main() -> None:
    assert _entry_point() == "cje.interface.cli:main"


@pytest.mark.parametrize(
    "flags,expected",
    [((), BUNDLED_DIR / "SKILL.md"), (("--reference",), BUNDLED_DIR / "reference.md")],
    ids=["skill", "reference"],
)
def test_cje_skill_prints_the_exact_file(
    capsysbinary: pytest.CaptureFixture[bytes], flags: tuple, expected: Path
) -> None:
    assert _run_entry_point("skill", *flags) == 0
    captured = capsysbinary.readouterr()
    assert captured.out == expected.read_bytes()
    assert captured.err == b""


def test_cje_skill_path_prints_the_folder(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run_entry_point("skill", "--path") == 0
    printed = Path(capsys.readouterr().out.strip())
    assert printed == BUNDLED_DIR.resolve()
    for name in SKILL_FILES:
        assert (printed / name).is_file()


def test_reference_and_path_are_mutually_exclusive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run_entry_point("skill", "--reference", "--path")
    assert excinfo.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_missing_bundle_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    monkeypatch.setattr(cli, "skill_dir", lambda: tmp_path)
    for flags in ((), ("--reference",), ("--path",)):
        assert _run_entry_point("skill", *flags) == 1
        captured = capsysbinary.readouterr()
        assert captured.out == b""
        err = captured.err.decode("utf-8")
        assert "not bundled" in err
        assert "SKILL.md" in err and "reference.md" in err
        assert cli.SKILL_SOURCE_URL in err


def test_python_dash_m_cje_skill() -> None:
    """`python -m cje skill` reaches the same command (no console script needed)."""
    env = {**os.environ, "PYTHONIOENCODING": "ascii"}  # bytes must not re-encode
    skill = subprocess.run(
        [sys.executable, "-m", "cje", "skill"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        timeout=120,
    )
    assert skill.returncode == 0, skill.stderr.decode("utf-8", "replace")
    assert skill.stdout == (BUNDLED_DIR / "SKILL.md").read_bytes()

    path = subprocess.run(
        [sys.executable, "-m", "cje", "skill", "--path"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert path.returncode == 0, path.stderr
    assert Path(path.stdout.strip()) == BUNDLED_DIR.resolve()


def test_top_level_help_points_agents_at_the_skill(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run_entry_point("--help")
    assert excinfo.value.code == 0
    out = " ".join(capsys.readouterr().out.split())  # argparse rewraps text
    assert "skill" in out
    assert "Coding agents: run `cje skill`" in out


def test_package_docstring_opens_with_the_agent_rules() -> None:
    doc = cje.__doc__ or ""
    assert doc.startswith("Coding agents: run `cje skill` before using this API.")
    flat = " ".join(doc.split())
    for rule in (
        "Never report a raw judge-score mean as quality",
        "attach oracle labels only from random samples",
        "for a policy without labels of its own, name no winner and no lean "
        "until it is labeled or its transport audit PASSes",
    ):
        assert rule in flat


def test_cli_version_flag(capsys: pytest.CaptureFixture) -> None:
    from cje.interface.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"cje-eval {cje.__version__}"
