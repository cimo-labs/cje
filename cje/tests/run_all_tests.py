#!/usr/bin/env python
"""Run all tests."""

import subprocess
import sys
from pathlib import Path

tests = [
    "test_simple.py",
    "test_pipeline.py",
    "test_edge_cases.py",
    "test_integration.py",
]


def main() -> int:
    """Run all test files."""
    print("Running all CJE tests...\n")

    failed = []
    for test in tests:
        print(f"{'='*50}")
        print(f"Running {test}")
        print(f"{'='*50}")

        result = subprocess.run(
            [sys.executable, f"tests/{test}"], capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"❌ {test} FAILED")
            print(result.stdout)
            print(result.stderr)
            failed.append(test)
        else:
            print(result.stdout)
            print(f"✅ {test} passed\n")

    print(f"\n{'='*50}")
    if failed:
        print(f"❌ {len(failed)} tests failed: {failed}")
        return 1
    else:
        print("✅ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
