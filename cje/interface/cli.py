#!/usr/bin/env python3
"""
CJE Command Line Interface.

Simple CLI for common CJE analysis tasks.
"""

import sys
import argparse
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from ..data.models import EstimationResult

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

    # Names are validated by the analysis pipeline (interface/_removed.py) so
    # that removed 0.3.x OPE estimators surface the full migration error
    # instead of an argparse choices message.
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
        "--fresh-judge-scale",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Declared judge-score scale for fresh draws",
    )
    analyze_parser.add_argument(
        "--fresh-oracle-scale",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Declared oracle-label scale for fresh draws",
    )
    analyze_parser.add_argument(
        "--calibration-judge-scale",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Declared judge-score scale for --calibration-data",
    )
    analyze_parser.add_argument(
        "--calibration-oracle-scale",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Declared oracle-label scale for --calibration-data",
    )
    analyze_parser.add_argument(
        "--output-scale",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Scale for reported estimates and uncertainty",
    )
    analyze_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on an invalid record instead of dropping it",
    )

    analyze_parser.add_argument(
        "--transport-probe",
        action="append",
        default=[],
        metavar="POLICY=PATH",
        help=(
            "Held-out oracle-probe JSONL for a policy; repeat for multiple " "policies"
        ),
    )
    analyze_parser.add_argument(
        "--transport-margin",
        action="append",
        default=[],
        metavar="POLICY=DELTA",
        help=(
            "Practical absolute residual margin in oracle/output units; "
            "repeat per audited policy"
        ),
    )
    analyze_parser.add_argument(
        "--transport-family-size",
        type=int,
        help="Predeclared number of simultaneous policy/group audits",
    )
    analyze_parser.add_argument(
        "--transport-alpha",
        type=float,
        help="Family-wise error rate for transport audits (default: 0.05)",
    )
    analyze_parser.add_argument(
        "--transport-min-clusters",
        type=float,
        help="Minimum effective probe clusters for grading (default: 20)",
    )
    analyze_parser.add_argument(
        "--transport-bins",
        type=int,
        help="Score-quantile bins for display diagnostics (default: 10)",
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
        "--judge-scale",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Declared minimum and maximum judge-score values",
    )

    validate_parser.add_argument(
        "--oracle-scale",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Declared minimum and maximum oracle-label values",
    )

    validate_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed validation results",
    )

    return parser


def best_policy_lines(results: "EstimationResult") -> list:
    """Build a point-estimate winner announcement with limitations.

    Thin renderer over EstimationResult.best_policy() (which owns the
    gate/CRITICAL derivation). Diagnostics qualify the numerical winner but
    never silently replace it with a different policy.
    """
    import numpy as np

    estimates = getattr(results, "estimates", None)
    metadata = getattr(results, "metadata", None)
    target_policies = metadata.get("target_policies", []) if metadata else []
    if estimates is None or len(estimates) == 0 or not target_policies:
        return []

    if np.all(np.isnan(np.asarray(estimates, dtype=float))):
        return ["No usable estimates: every point estimate is NaN (see diagnostics)."]

    verdict = results.best_policy()
    limitations = []
    if verdict.flagged:
        limitations.append(
            "no policy passed the reliability gates"
            if verdict.all_flagged
            else "reliability gates flagged this policy"
        )
    if metadata and metadata.get("calibration_status") == "UNCALIBRATED":
        limitations.append("UNCALIBRATED raw judge-score mean")
    transport_audits = metadata.get("transport_audits", {}) if metadata else {}
    winner_audit = transport_audits.get(verdict.name, {})
    transport_status = winner_audit.get("status", "NOT_CHECKED")
    if transport_status != "PASS":
        limitations.append(f"residual transport {transport_status}")

    lines = [f"Best by point estimate: {verdict.name}"]
    if limitations:
        lines.append("Limitations: " + "; ".join(limitations))
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

        # Prepare kwargs. The estimator name is passed through unmapped so
        # the CLI and the API record the same resolved name in
        # metadata["estimator"] for the same run.
        kwargs = {
            "estimator": args.estimator,
            "judge_field": args.judge_field,
            "oracle_field": args.oracle_field,
            "strict": args.strict,
        }

        for argument, keyword in (
            (args.fresh_judge_scale, "fresh_judge_scale"),
            (args.fresh_oracle_scale, "fresh_oracle_scale"),
            (args.calibration_judge_scale, "calibration_judge_scale"),
            (args.calibration_oracle_scale, "calibration_oracle_scale"),
            (args.output_scale, "output_scale"),
        ):
            if argument is not None:
                kwargs[keyword] = tuple(argument)

        if args.estimator_config:
            kwargs["estimator_config"] = args.estimator_config

        if args.calibration_data:
            kwargs["calibration_data_path"] = args.calibration_data

        transport_requested = bool(
            args.transport_probe
            or args.transport_margin
            or args.transport_family_size is not None
            or args.transport_alpha is not None
            or args.transport_min_clusters is not None
            or args.transport_bins is not None
        )
        if transport_requested:
            from ..data.ingest import read_jsonl_records
            from ..diagnostics.transport import TransportAuditConfig

            probes = {}
            for assignment in args.transport_probe:
                policy, separator, raw_path = assignment.partition("=")
                if not separator or not policy or not raw_path:
                    raise ValueError("--transport-probe must use POLICY=PATH syntax")
                if policy in probes:
                    raise ValueError(
                        f"Duplicate --transport-probe for policy {policy!r}"
                    )
                probe_path = Path(raw_path)
                if not probe_path.is_file():
                    raise FileNotFoundError(probe_path)
                records = read_jsonl_records(probe_path, on_invalid="error")
                source_id = str(probe_path.resolve())
                for record in records:
                    line_num = record.get("_cje_line_num")
                    record["_cje_source_id"] = source_id
                    record["_cje_row_id"] = f"{source_id}:line:{line_num}"
                probes[policy] = records

            margins = {}
            for assignment in args.transport_margin:
                policy, separator, raw_margin = assignment.partition("=")
                if not separator or not policy or not raw_margin:
                    raise ValueError("--transport-margin must use POLICY=DELTA syntax")
                if policy in margins:
                    raise ValueError(
                        f"Duplicate --transport-margin for policy {policy!r}"
                    )
                try:
                    margins[policy] = float(raw_margin)
                except ValueError as exc:
                    raise ValueError(
                        f"Transport margin for policy {policy!r} must be numeric"
                    ) from exc

            kwargs["transport"] = TransportAuditConfig(
                probes_by_policy=probes,
                delta_max_by_policy=margins,
                bins=(args.transport_bins if args.transport_bins is not None else 10),
                alpha=(
                    args.transport_alpha if args.transport_alpha is not None else 0.05
                ),
                family_size=args.transport_family_size,
                min_effective_clusters=(
                    args.transport_min_clusters
                    if args.transport_min_clusters is not None
                    else 20.0
                ),
            )

        path_obj = Path(path)
        if path_obj.is_dir():
            if _dir_has_logprob_fields(path_obj):
                logger.info("logprob fields present and ignored (Direct mode)")
            kwargs["fresh_draws_dir"] = str(path_obj)
        elif path_obj.is_file():
            from ..data.ingest import fresh_draws_data_from_file

            fresh_draws_data = fresh_draws_data_from_file(
                path_obj, on_invalid="error" if args.strict else "drop"
            )
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
                audit = results.metadata.get("transport_audits", {}).get(policy, {})
                print(
                    "    residual transport: " f"{audit.get('status', 'NOT_CHECKED')}"
                )

            # Best policy (reliability-aware: an argmax that failed the
            # refusal-gate limitations are printed beside the point winner)
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
            judge_scale=args.judge_scale,
            oracle_scale=args.oracle_scale,
        )
        issues = [f for f in findings if not f.startswith(NOTE_PREFIX)]
        notes = [f for f in findings if f.startswith(NOTE_PREFIX)]

        def _numeric_values(recs: list, field: str) -> list:
            values = []
            for record in recs:
                try:
                    value = read_record_field(record, field)
                except ValueError:
                    # validate_direct_data already reports alias conflicts as
                    # findings; summary statistics must not mask that report.
                    continue
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ):
                    values.append(float(value))
            return values

        def _oracle_count(recs: list) -> int:
            return len(_numeric_values(recs, args.oracle_field))

        def _scale_cli_args(flag: str, values: Optional[Sequence[float]]) -> str:
            if values is None:
                return ""
            minimum, maximum = values
            return f" {flag} {minimum:g} {maximum:g}"

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

        if not by_policy and n_oracle_total == 0:
            issues.append(
                "This file has neither target_policy values nor oracle labels. "
                "As written it is neither single-file fresh draws nor a useful "
                "calibration source: add target_policy to evaluate raw judge-score "
                "means, or add oracle labels for --calibration-data."
            )
            is_valid = False

        if issues:
            print("\n⚠️  Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        for note in notes:
            print(f"  {note}")

        if is_valid:
            if by_policy:
                print("\n✓ Data is valid and ready for analysis")
                scale_args = _scale_cli_args(
                    "--fresh-judge-scale", args.judge_scale
                ) + _scale_cli_args("--fresh-oracle-scale", args.oracle_scale)
                print(f"  Next: cje analyze {path}{scale_args}")
            else:
                # Logged-style / calibration file: judge + oracle pairs
                # without target_policy — valid as a calibration source
                print(
                    "\n✓ Data is valid as a calibration source "
                    "(judge + oracle pairs; no target_policy field) and "
                    "ready for use with --calibration-data"
                )
                scale_args = _scale_cli_args(
                    "--calibration-judge-scale", args.judge_scale
                ) + _scale_cli_args("--calibration-oracle-scale", args.oracle_scale)
                print(
                    f"  Next: cje analyze FRESH_DRAWS --calibration-data "
                    f"{path}{scale_args}"
                )

        if args.verbose:
            import numpy as np

            print("\nDetailed Statistics:")
            print("-" * 40)

            judge_scores = _numeric_values(records, args.judge_field)
            oracle_labels = _numeric_values(records, args.oracle_field)

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
