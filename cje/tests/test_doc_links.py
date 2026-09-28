"""Links must work on PyPI and in the documented two-file skill install."""

import re
import shutil
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

from cje.tests.test_doc_snippets import REPO_ROOT


def _links(path: Path) -> list[str]:
    # These documents use inline Markdown links (including linked badges).
    return re.findall(r"\]\(([^\s)]+)\)", path.read_text())


def test_readme_links_survive_pypi_rendering() -> None:
    for target in _links(REPO_ROOT / "README.md"):
        if target.startswith("#"):
            continue
        resolved = urlsplit(urljoin("https://pypi.org/project/cje-eval/", target))
        assert resolved.scheme == "https", target
        assert urlsplit(
            target
        ).netloc, f"Relative README link resolves under PyPI: {resolved.geturl()}"
        prefix = "/cimo-labs/cje/blob/main/"
        if resolved.netloc == "github.com" and resolved.path.startswith(prefix):
            assert (REPO_ROOT / unquote(resolved.path[len(prefix) :])).is_file()


def test_two_file_skill_install_has_no_missing_local_links(tmp_path: Path) -> None:
    for name in ("SKILL.md", "reference.md"):
        shutil.copy2(REPO_ROOT / "skills" / "cje" / name, tmp_path / name)
    for path in tmp_path.iterdir():
        for target in _links(path):
            parsed = urlsplit(target)
            if not parsed.scheme and parsed.path:
                assert (path.parent / unquote(parsed.path)).is_file(), target
