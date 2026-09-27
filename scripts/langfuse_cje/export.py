"""Export, validate, and summarize two Langfuse runs; resume from saved reads."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence
from urllib.parse import urlsplit

# This companion command is run from a source checkout. Core conversion is also
# available in the installed wheel, without the HTTP dependency.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cje.bridges.langfuse import digest, prepare
from transport import collect, collect_dataset


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return value


def save(path: Path, value: dict) -> None:
    """Atomically create private evidence; an identical repeat is a no-op."""
    data = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"{path.name} differs; use a new output directory")
    finally:
        os.unlink(temporary)


def run(
    config: dict,
    destination: Path,
    *,
    snapshot: dict | None = None,
    oracle_manifest: dict | None = None,
    dataset_manifest: dict | None = None,
    client: Any = None,
) -> dict:
    """Save complete reads before validation; resume without a network client."""
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    save(destination / "config.json", config)
    snapshot_path = destination / "snapshot.json"
    if snapshot is None:
        if snapshot_path.exists():
            snapshot = read(snapshot_path)
        else:
            if client is None:
                raise ValueError(
                    "A saved snapshot or an authenticated client is required"
                )
            snapshot = collect(client, config)
    save(snapshot_path, snapshot)
    # Save the raw experiment export first, even if fetching the dataset fails.
    dataset_path = destination / "dataset-manifest.json"
    if dataset_manifest is None and dataset_path.exists():
        dataset_manifest = read(dataset_path)
    if dataset_manifest is None and config.get("dataset_version") is not None:
        if client is None:
            raise ValueError(
                "Frozen dataset read is missing; provide --dataset-manifest or resume online"
            )
        dataset_manifest = collect_dataset(client, config)
    if dataset_manifest is not None:
        save(dataset_path, dataset_manifest)
    oracle_path = destination / "oracle-manifest.json"
    if oracle_manifest is None and oracle_path.exists():
        oracle_manifest = read(oracle_path)
    # A missing manifest may be supplied on a subsequent offline invocation.
    prepared = prepare(
        snapshot,
        config,
        oracle_manifest=oracle_manifest,
        dataset_manifest=dataset_manifest,
    )
    if oracle_manifest is not None:
        save(oracle_path, oracle_manifest)
    save(destination / "fresh-draws.json", prepared["fresh_draws_data"])
    report = {
        "schema": 1,
        "status": "PREPARED",
        "provenance": prepared["provenance"],
        "scales": {role: config[role]["scale"] for role in ("judge", "oracle")},
        "response_design": "one observation per dataset item per policy",
        "sampling_design": "UNKNOWN: confirm how the human-labeled slice was selected",
        "independence": "NOT_VERIFIED: shared prompt IDs support pairing; dataset items may still be dependent",
        "audit_allocation": "NOT_DECLARED: hold out audit labels before fitting",
        "config_sha256": digest(config),
        "fresh_draws_sha256": digest(prepared["fresh_draws_data"]),
    }
    save(destination / "readiness.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--snapshot", type=Path, help="Complete raw export; no network needed"
    )
    parser.add_argument("--oracle-manifest", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument(
        "--online", action="store_true", help="Allow GET requests for missing reads"
    )
    args = parser.parse_args(argv)
    try:
        config = read(args.config)
        options = {
            "snapshot": read(args.snapshot) if args.snapshot else None,
            "oracle_manifest": (
                read(args.oracle_manifest) if args.oracle_manifest else None
            ),
            "dataset_manifest": (
                read(args.dataset_manifest) if args.dataset_manifest else None
            ),
        }
        if args.online:
            import httpx

            url = os.environ["LANGFUSE_BASE_URL"].rstrip("/") + "/"
            parsed = urlsplit(url)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "LANGFUSE_BASE_URL must be an HTTPS instance URL without credentials, query, or fragment"
                )
            with httpx.Client(
                base_url=url,
                auth=(
                    os.environ["LANGFUSE_PUBLIC_KEY"],
                    os.environ["LANGFUSE_SECRET_KEY"],
                ),
                timeout=60,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                report = run(config, args.out_dir, client=client, **options)
        else:
            report = run(config, args.out_dir, **options)
    except Exception as error:
        # Do not print HTTP headers, credential values, or arbitrary server bodies.
        if type(error).__module__.startswith("httpx"):
            print(
                "Langfuse read failed. Complete saved reads remain available; retry the same command.",
                file=sys.stderr,
            )
        else:
            print(f"Import stopped: {error}", file=sys.stderr)
        return 1
    print(
        f"Prepared {sum(r['responses'] for r in report['provenance']['counts'].values())} responses. See {args.out_dir / 'readiness.json'}; estimation has not run."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
