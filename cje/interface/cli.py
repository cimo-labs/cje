#!/usr/bin/env python3
"""
CJE Command Line Interface.

Simple CLI for common CJE analysis tasks.
"""

import sys
import argparse
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Simple format for CLI
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for CJE CLI."""
    parser = argparse.ArgumentParser(
        prog="cje",
        description=(
            "Causal Judge Evaluation - calibrated Direct-mode evaluation of "
            "LLM policies from judge-scored fresh draws"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run Direct-mode CJE analysis on fresh draws",
        description=(
            "Estimate per-policy values from judge-scored fresh draws "
            "(Direct mode). PATH is a directory of per-policy response files "
            "(e.g. <policy>_responses.jsonl) or a single JSONL file whose "
            "records carry a target_policy field."
        ),
    )

    analyze_parser.add_argument(
        "path",
        nargs="?",
        help=(
            "Fresh draws: directory of per-policy response files, or a "
            "single JSONL file with a target_policy field per record"
        ),
    )

    # Names are validated by the interface factory so that removed 0.3.x OPE
    # estimators surface the full migration error instead of an argparse
    # choices message.
    analyze_parser.add_argument(
        "--estimator",
        default="calibrated-direct",
        help=(
            "Estimation method: 'calibrated-direct' (default) or 'direct'. "
            "OPE names removed in 0.4.0 (calibrated-ips, dr-cpo, ...) raise "
            "a migration error pointing at the 0.3.x line."
        ),
    )

    analyze_parser.add_argument(
        "--output",
        "-o",
        help="Path to save results JSON (optional)",
    )

    analyze_parser.add_argument(
        "--fresh-draws-dir",
        help="Alias for PATH (kept for 0.3.x compatibility)",
    )

    analyze_parser.add_argument(
        "--calibration-data",
        help=(
            "JSONL file with judge_score + oracle_label pairs used to learn "
            "the judge->oracle calibration (your old logged data works here)"
        ),
    )

    # Note: We intentionally do not expose oracle_coverage here.
    # Production uses all available oracle labels for calibration.

    analyze_parser.add_argument(
        "--estimator-config",
        type=json.loads,
        help="JSON config for estimator (e.g., '{\"n_bootstrap\": 4000}')",
    )

    analyze_parser.add_argument(
        "--judge-field",
        default="judge_score",
        help="Metadata field containing judge scores (default: judge_score)",
    )

    analyze_parser.add_argument(
        "--oracle-field",
        default="oracle_label",
        help="Metadata field containing oracle labels (default: oracle_label)",
    )

    analyze_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    analyze_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-essential output",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate Direct-mode data (fresh draws or calibration file)",
        description=(
            "Check that data is ready for 'cje analyze'. PATH is a fresh-draws "
            "directory of per-policy response files, a single fresh-draws JSONL "
            "file with a target_policy field per record, or a calibration JSONL "
            "file with judge_score + oracle_label pairs."
        ),
    )

    validate_parser.add_argument(
        "dataset",
        help="Path to a fresh-draws directory or a JSONL file",
    )

    validate_parser.add_argument(
        "--judge-field",
        default="judge_score",
        help="Field containing judge scores (default: judge_score)",
    )

    validate_parser.add_argument(
        "--oracle-field",
        default="oracle_label",
        help="Field containing oracle labels (default: oracle_label)",
    )

    validate_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed validation results",
    )

    return parser


def _unreliable_policies(results: object) -> set:
    """Policies flagged unreliable by the refusal gates or CRITICAL status."""
    flagged = set()

    metadata = getattr(results, "metadata", None)
    if isinstance(metadata, dict):
        gates = metadata.get("reliability_gates") or {}
        for policy, info in gates.items():
            if isinstance(info, dict) and info.get("flagged"):
                flagged.add(policy)

    diagnostics = getattr(results, "diagnostics", None)
    status_per_policy = getattr(diagnostics, "status_per_policy", None)
    if status_per_policy:
        for policy, status in status_per_policy.items():
            value = getattr(status, "value", status)
            if value == "critical":
                flagged.add(policy)

    return flagged


def best_policy_lines(results: object) -> list:
    """Build the best-policy announcement, demoting unreliable argmaxes.

    The raw argmax previously earned the trophy line even when the winning
    policy had been flagged UNRELIABLE by the refusal gates (verified on
    the bundled arena sample, where the adversarial 'unhelpful' policy
    won). The trophy now goes only to a policy that passed the gates; a
    flagged argmax is demoted to a warning line.
    """
    import numpy as np

    estimates = getattr(results, "estimates", None)
    metadata = getattr(results, "metadata", None)
    target_policies = metadata.get("target_policies", []) if metadata else []
    if estimates is None or len(estimates) == 0 or not target_policies:
        return []

    estimates = np.asarray(estimates, dtype=float)
    if np.all(np.isnan(estimates)):
        return [
            "⚠️ No usable estimates: every policy was refused as unreliable "
            "(see diagnostics)."
        ]

    best_idx = int(np.nanargmax(estimates))
    best_policy = target_policies[best_idx]
    flagged = _unreliable_policies(results)

    if best_policy not in flagged:
        return [f"🏆 Best policy: {best_policy}"]

    lines = [
        f"⚠️ Best by point estimate: {best_policy} " f"(UNRELIABLE — see diagnostics)"
    ]
    reliable = [
        (float(estimates[i]), policy)
        for i, policy in enumerate(target_policies)
        if policy not in flagged and not np.isnan(estimates[i])
    ]
    if reliable:
        _, best_reliable = max(reliable)
        lines.append(f"🏆 Best reliable policy: {best_reliable}")
    else:
        lines.append(
            "No policy passed the reliability gates; do not pick a winner "
            "from this run."
        )
    return lines


# Logged-data logprob fields. In fresh draws they are ignored (Direct mode
# needs none); a file of records WITHOUT target_policy but WITH these fields
# is an 0.3.x logged dataset and gets the migration error.
_LOGPROB_FIELDS = (
    "base_policy_logprob",
    "target_policy_logprobs",
    "logprob",
    "logprobs",
    "total_logprob",
    "token_logprobs",
)

# Cap per-file scanning for logprob detection (uniform record shapes make
# the first few records representative).
_LOGPROB_SCAN_LIMIT = 100


def _records_have_logprob_fields(records: list) -> bool:
    return any(
        field in record
        for record in records[:_LOGPROB_SCAN_LIMIT]
        for field in _LOGPROB_FIELDS
    )


def _dir_has_logprob_fields(directory: Path) -> bool:
    """Scan the leading records of each JSONL file in a fresh-draws dir."""
    for file_path in sorted(directory.rglob("*.jsonl")):
        with open(file_path) as f:
            for i, line in enumerate(f):
                if i >= _LOGPROB_SCAN_LIMIT:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    break  # Leave loud parsing errors to the real loader
                if isinstance(record, dict) and any(
                    field in record for field in _LOGPROB_FIELDS
                ):
                    return True
    return False


def run_analysis(args: argparse.Namespace) -> int:
    """Run the analysis command."""
    from .analysis import LOGGED_DATA_PATH_REMOVED_MESSAGE, analyze_dataset

    # Set logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Resolve the fresh-draws path (positional, or the 0.3.x-compatible
        # --fresh-draws-dir alias). Both together is the removed 0.3.x OPE
        # invocation (logged positional + fresh draws): migration error.
        if args.path and args.fresh_draws_dir:
            raise ValueError(LOGGED_DATA_PATH_REMOVED_MESSAGE)
        path = args.path or args.fresh_draws_dir
        if not path:
            raise ValueError(
                "Provide fresh draws: 'cje analyze PATH' (directory or "
                "JSONL file) or --fresh-draws-dir PATH."
            )

        # Prepare kwargs
        estimator_choice = args.estimator
        if estimator_choice in (None, "auto"):
            estimator_choice = "calibrated-direct"

        kwargs = {
            "estimator": estimator_choice,
            "judge_field": args.judge_field,
            "oracle_field": args.oracle_field,
        }

        if args.estimator_config:
            kwargs["estimator_config"] = args.estimator_config

        if args.calibration_data:
            kwargs["calibration_data_path"] = args.calibration_data

        path_obj = Path(path)
        if path_obj.is_dir():
            if _dir_has_logprob_fields(path_obj):
                logger.info("logprob fields present and ignored (Direct mode)")
            kwargs["fresh_draws_dir"] = str(path_obj)
        elif path_obj.is_file():
            from ..data.ingest import fresh_draws_data_from_file

            fresh_draws_data = fresh_draws_data_from_file(path_obj)
            all_records = [r for recs in fresh_draws_data.values() for r in recs]
            if _records_have_logprob_fields(all_records):
                logger.info("logprob fields present and ignored (Direct mode)")
            kwargs["fresh_draws_data"] = fresh_draws_data
        else:
            raise FileNotFoundError(path)

        # Run analysis
        if not args.quiet:
            print(f"Running CJE analysis on {path}")
            print("=" * 50)

        results = analyze_dataset(**kwargs)

        # Display results
        if not args.quiet:
            print("\nResults:")
            print("-" * 40)

            # Display estimates
            target_policies = results.metadata.get("target_policies", [])
            for i, policy in enumerate(target_policies):
                estimate = results.estimates[i]
                se = results.standard_errors[i]
                print(
                    f"  {policy}: {estimate:.3f} ± {se:.3f} (1 SE; 95% CI ≈ ±1.96·SE)"
                )

            # Best policy (reliability-aware: an argmax that failed the
            # refusal gates is demoted instead of crowned)
            lines = best_policy_lines(results)
            if lines:
                print()
                for line in lines:
                    print(line)

        # Save results if requested
        if args.output:
            from ..utils.export import export_results_json

            export_results_json(results, args.output)
            if not args.quiet:
                print(f"\n✓ Results saved to: {args.output}")

        return 0

    except FileNotFoundError as e:
        print(f"❌ Error: Fresh draws path not found: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def _resolve_policy_file(directory: Path, policy: str) -> Path:
    """Locate the per-policy fresh-draws file for a discovered policy.

    Delegates to the canonical POLICY_FILE_PATTERNS resolver so `cje
    validate` accepts exactly the layouts `cje analyze` loads.
    """
    from ..data.ingest import POLICY_FILE_PATTERNS, resolve_policy_file

    candidate = resolve_policy_file(directory, policy)
    if candidate is not None:
        return candidate
    patterns = ", ".join(p.format(policy=policy) for p in POLICY_FILE_PATTERNS)
    raise FileNotFoundError(
        f"No fresh-draws file found for policy '{policy}' in {directory} "
        f"(expected one of: {patterns})"
    )


def _read_fresh_draws_dir(directory: Path) -> list:
    """Read every per-policy file in a fresh-draws directory.

    Records are tagged with the policy from the filename (an explicit
    target_policy field in a record wins), so validation can run its
    per-policy checks and report per-policy counts.
    """
    from ..data.fresh_draws import discover_policies_from_fresh_draws
    from ..data.ingest import read_jsonl_records

    records = []
    for policy in discover_policies_from_fresh_draws(directory):
        file_path = _resolve_policy_file(directory, policy)
        for record in read_jsonl_records(file_path):
            record.setdefault("target_policy", policy)
            records.append(record)
    return records


def validate_data(args: argparse.Namespace) -> int:
    """Run the validate command on the RAW parsed JSONL records.

    Validates the file(s) exactly as parsed — no Dataset round-trip. The
    0.2.x-era implementation converted records to a Dataset first, which
    regenerated prompt_ids and relocated judge/oracle fields, so `cje
    validate` false-failed ALL valid data.
    """
    from ..data.ingest import read_jsonl_records
    from ..data.validation import NOTE_PREFIX, read_record_field, validate_direct_data

    path = Path(args.dataset)

    try:
        print(f"Validating {path}...")

        if path.is_dir():
            records = _read_fresh_draws_dir(path)
        elif path.is_file():
            records = read_jsonl_records(path)
        else:
            raise FileNotFoundError(args.dataset)

        is_valid, findings = validate_direct_data(
            records,
            judge_field=args.judge_field,
            oracle_field=args.oracle_field,
        )
        issues = [f for f in findings if not f.startswith(NOTE_PREFIX)]
        notes = [f for f in findings if f.startswith(NOTE_PREFIX)]

        def _oracle_count(recs: list) -> int:
            return sum(
                1
                for r in recs
                if isinstance(read_record_field(r, args.oracle_field), (int, float))
            )

        print(f"✓ Loaded {len(records)} records")

        # Per-policy sample + oracle counts (fresh draws); a file without
        # target_policy is a single pool (calibration source)
        by_policy: dict = {}
        for record in records:
            policy = record.get("target_policy")
            if policy is not None:
                by_policy.setdefault(str(policy), []).append(record)

        if by_policy:
            print(
                f"✓ Target policies ({len(by_policy)}): {', '.join(sorted(by_policy))}"
            )
            for policy in sorted(by_policy):
                recs = by_policy[policy]
                print(
                    f"  {policy}: {len(recs)} samples, "
                    f"{_oracle_count(recs)} oracle labels"
                )

        n_oracle_total = _oracle_count(records)
        print(f"✓ Oracle labels: {n_oracle_total}/{len(records)} records")

        if issues:
            print("\n⚠️  Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        for note in notes:
            print(f"  {note}")

        if is_valid:
            if by_policy:
                print("\n✓ Data is valid and ready for analysis")
                print(f"  Next: cje analyze {path}")
            else:
                # Logged-style / calibration file: judge + oracle pairs
                # without target_policy — valid as a calibration source
                print(
                    "\n✓ Data is valid as a calibration source "
                    "(judge + oracle pairs; no target_policy field) and "
                    "ready for use with --calibration-data"
                )
                print(f"  Next: cje analyze FRESH_DRAWS --calibration-data {path}")

        if args.verbose:
            import numpy as np

            print("\nDetailed Statistics:")
            print("-" * 40)

            judge_scores = [
                float(v)
                for r in records
                if isinstance(
                    (v := read_record_field(r, args.judge_field)), (int, float)
                )
            ]
            oracle_labels = [
                float(v)
                for r in records
                if isinstance(
                    (v := read_record_field(r, args.oracle_field)), (int, float)
                )
            ]

            if judge_scores:
                print(f"Judge scores: {len(judge_scores)} samples")
                print(f"  Range: [{min(judge_scores):.3f}, {max(judge_scores):.3f}]")
                print(f"  Mean: {np.mean(judge_scores):.3f}")

            if oracle_labels:
                print(f"Oracle labels: {len(oracle_labels)} samples")
                print(f"  Range: [{min(oracle_labels):.3f}, {max(oracle_labels):.3f}]")
                print(f"  Mean: {np.mean(oracle_labels):.3f}")

        return 0 if is_valid else 1

    except FileNotFoundError as e:
        print(f"❌ Error: Dataset path not found: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"❌ Error validating dataset: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Error validating dataset: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "analyze":
        return run_analysis(args)
    elif args.command == "validate":
        return validate_data(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
