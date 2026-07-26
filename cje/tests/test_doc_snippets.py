"""Keep documentation code snippets honest.

Extracts every ```python block from the repo READMEs, the top-level guides
(MIGRATING-0.6.md, PLAYBOOK.md), and the agent-skill files (skills/cje/) and
checks it at two levels:

1. SYNTAX: every snippet must compile() — catches broken example code.
2. IMPORTS: every `import`/`from` line in every snippet is executed — catches
   references to modules/functions that do not exist (the review found several
   snippets importing from the wrong module or using deleted APIs).

Snippets are not executed end-to-end (most need data files or API keys), but
attribute chains on cje modules referenced by the imports are verified, which
is where docs historically drifted.
"""

import ast
import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

README_PATHS = sorted(
    p
    for p in [
        REPO_ROOT / "README.md",
        REPO_ROOT / "MIGRATING-0.6.md",
        REPO_ROOT / "PLAYBOOK.md",
        *(REPO_ROOT / "cje").glob("*/README.md"),
        *(REPO_ROOT / "skills").glob("*/*.md"),
    ]
    if p.exists()
)

_FENCE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _snippets() -> Iterator[Tuple[str, int, str]]:
    for path in README_PATHS:
        text = path.read_text()
        for match in _FENCE_RE.finditer(text):
            line = text[: match.start()].count("\n") + 2
            yield (str(path.relative_to(REPO_ROOT)), line, match.group(1))


SNIPPETS: List[Tuple[str, int, str]] = list(_snippets())
IDS = [f"{path}:{line}" for path, line, _ in SNIPPETS]


def test_readmes_found() -> None:
    assert len(README_PATHS) >= 5, README_PATHS
    assert len(SNIPPETS) >= 10


@pytest.mark.parametrize("path,line,code", SNIPPETS, ids=IDS)
def test_snippet_compiles(path: str, line: int, code: str) -> None:
    try:
        compile(code, f"<doc-snippet {path}:{line}>", "exec")
    except SyntaxError as e:  # pragma: no cover - failure path
        pytest.fail(f"{path}:{line} snippet has a syntax error: {e}")


def test_skill_quickstart_matches_readme() -> None:
    """The multi-policy example in the agent skill must be a byte-exact copy
    of the README quickstart — one canonical example, no drift. (Each copy
    passes compile+import on its own, so only this equality check catches
    divergence.)"""
    skill_text = (REPO_ROOT / "skills" / "cje" / "SKILL.md").read_text()
    readme_text = (REPO_ROOT / "README.md").read_text()
    quickstart = [
        code
        for code in _FENCE_RE.findall(skill_text)
        if "draws = {" in code and "analyze_dataset(fresh_draws_data=draws)" in code
    ]
    assert len(quickstart) == 1, "expected exactly one canonical example in SKILL.md"
    assert quickstart[0] in readme_text, (
        "SKILL.md quickstart has drifted from the README quickstart; "
        "re-paste it byte-exact from README.md"
    )


@pytest.mark.parametrize("path,line,code", SNIPPETS, ids=IDS)
def test_snippet_imports_resolve(path: str, line: int, code: str) -> None:
    """Execute the import statements of each snippet, one at a time.

    MIGRATING-0.6.md shows removed 0.5.x imports next to their replacements.
    An import statement that any line of the snippet annotates with an
    ``ImportError`` comment is documented-to-fail: the test asserts it
    actually raises ImportError (pinning the removal) instead of failing on
    it, and still verifies the remaining imports.
    """
    tree = ast.parse(code)
    import_nodes: List[ast.stmt] = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    if not import_nodes:
        pytest.skip("snippet has no imports")
    lines = code.splitlines()
    documented_import_errors = {
        stmt.strip()
        for stmt, _, comment in (raw.partition("#") for raw in lines)
        if "ImportError" in comment
    }
    namespace: Dict[str, object] = {}
    for node in import_nodes:
        compiled = compile(
            ast.Module(body=[node], type_ignores=[]),
            f"<doc-snippet {path}:{line}>",
            "exec",
        )
        statement = lines[node.lineno - 1].partition("#")[0].strip()
        if statement in documented_import_errors:
            with pytest.raises(ImportError):
                exec(compiled, namespace)
            continue
        try:
            exec(compiled, namespace)
        except ImportError as e:  # pragma: no cover - failure path
            # Optional-extra dependencies are allowed to be absent in the dev
            # env; docs for those modules state the required extra explicitly.
            msg = str(e)
            optional_markers = (
                "matplotlib",
                "seaborn",
                "cje-eval[viz]",  # the lazy plot_* hint on no-viz installs
            )
            if any(marker in msg for marker in optional_markers):
                pytest.skip(f"optional extra not installed: {msg}")
            pytest.fail(f"{path}:{line} snippet import fails: {e}")
