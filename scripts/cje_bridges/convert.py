#!/usr/bin/env python3
"""Convenience wrapper: convert common eval tool outputs into CJE fresh_draws_data.

This repo currently ships standalone converters:
- scripts/promptfoo_cje/promptfoo_to_cje.py
- scripts/trulens_cje/trulens_to_cje.py
- scripts/langsmith_cje/langsmith_to_cje.py

This wrapper provides a single entrypoint:
  python3 scripts/cje_bridges/convert.py <tool> [args...]

Example:
  python3 scripts/cje_bridges/convert.py promptfoo results.json --out cje.json

Notes:
- This wrapper intentionally forwards args verbatim to the underlying script.
- For full help on each converter, run the underlying script with --help.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _run_script(path: Path, argv: list[str]) -> int:
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(path)] + argv
        runpy.run_path(str(path), run_name="__main__")
        return 0
    except SystemExit as e:
        # Underlying scripts call SystemExit(main())
        code = e.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    finally:
        sys.argv = old_argv


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(
            "Usage: python3 scripts/cje_bridges/convert.py <tool> [args...]\n\n"
            "Tools:\n"
            "  promptfoo   Convert Promptfoo results JSON to CJE fresh_draws_data\n"
            "  trulens     Convert TruLens records+feedback to CJE fresh_draws_data\n"
            "  langsmith   Convert LangSmith runs+feedback to CJE fresh_draws_data\n\n"
            "Examples:\n"
            "  python3 scripts/cje_bridges/convert.py promptfoo results.json --out cje.json\n"
            "  python3 scripts/cje_bridges/convert.py trulens --database-url sqlite:///default.sqlite --judge-col 'Answer Relevance'\n"
            "  python3 scripts/cje_bridges/convert.py langsmith --project my_project --feedback-key correctness --out cje.json\n\n"
            "For full help on a tool:\n"
            "  python3 scripts/promptfoo_cje/promptfoo_to_cje.py --help\n"
            "  python3 scripts/trulens_cje/trulens_to_cje.py --help\n"
            "  python3 scripts/langsmith_cje/langsmith_to_cje.py --help\n"
        )
        return 0

    tool = sys.argv[1]
    rest = sys.argv[2:]

    scripts_dir = Path(__file__).resolve().parents[1]
    promptfoo_script = scripts_dir / "promptfoo_cje" / "promptfoo_to_cje.py"
    trulens_script = scripts_dir / "trulens_cje" / "trulens_to_cje.py"
    langsmith_script = scripts_dir / "langsmith_cje" / "langsmith_to_cje.py"

    if tool == "promptfoo":
        if not promptfoo_script.exists():
            print(f"Missing script: {promptfoo_script}", file=sys.stderr)
            return 2
        return _run_script(promptfoo_script, rest)

    if tool == "trulens":
        if not trulens_script.exists():
            print(f"Missing script: {trulens_script}", file=sys.stderr)
            return 2
        return _run_script(trulens_script, rest)

    if tool == "langsmith":
        if not langsmith_script.exists():
            print(f"Missing script: {langsmith_script}", file=sys.stderr)
            return 2
        return _run_script(langsmith_script, rest)

    print(
        f"Unknown tool {tool!r}. Expected one of: promptfoo, trulens, langsmith.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
