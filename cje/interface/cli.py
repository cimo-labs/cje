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
from typing import Optional, Dict, Any

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
        description="Causal Judge Evaluation - Unbiased LLM evaluation using causal inference",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run CJE analysis on a dataset",
        description="Analyze a dataset using CJE estimation methods",
    )

    analyze_parser.add_argument(
        "dataset",
        help="Path to JSONL dataset file",
    )

    from .factory import get_estimator_names

    analyze_parser.add_argument(
        "--estimator",
        choices=list(get_estimator_names()),
        default="stacked-dr",
        help=(
            "Estimation method. Default: stacked-dr (robust ensemble). "
            "Use calibrated-ips for speed over robustness or if you don't have fresh draws."
        ),
    )

    analyze_parser.add_argument(
        "--output",
        "-o",
        help="Path to save results JSON (optional)",
    )

    analyze_parser.add_argument(
        "--fresh-draws-dir",
        help="Directory containing fresh draw response files (for DR estimators)",
    )

    # Note: We intentionally do not expose oracle_coverage here.
    # Production uses all available oracle labels for calibration.

    analyze_parser.add_argument(
        "--estimator-config",
        type=json.loads,
        help="JSON config for estimator (e.g., '{\"n_folds\": 10}')",
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
        help="Validate a CJE dataset",
        description="Check dataset format and completeness",
    )

    validate_parser.add_argument(
        "dataset",
        help="Path to JSONL dataset file",
    )

    validate_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed validation results",
    )

    return parser


def run_analysis(args: argparse.Namespace) -> int:
    """Run the analysis command."""
    from .analysis import analyze_dataset  # Same module, this is fine

    # Set logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Prepare kwargs
        # Determine estimator default based on presence of fresh draws
        estimator_choice = args.estimator
        if estimator_choice in (None, "auto"):
            estimator_choice = (
                "stacked-dr" if args.fresh_draws_dir else "calibrated-ips"
            )

        kwargs = {
            "estimator": estimator_choice,
            "judge_field": args.judge_field,
            "oracle_field": args.oracle_field,
        }

        if args.estimator_config:
            kwargs["estimator_config"] = args.estimator_config

        if args.fresh_draws_dir:
            kwargs["fresh_draws_dir"] = args.fresh_draws_dir

        # Run analysis
        if not args.quiet:
            print(f"Running CJE analysis on {args.dataset}")
            print("=" * 50)

        results = analyze_dataset(logged_data_path=args.dataset, **kwargs)

        # Display results
        if not args.quiet:
            print("\nResults:")
            print("-" * 40)

            # Display estimates
            target_policies = results.metadata.get("target_policies", [])
            for i, policy in enumerate(target_policies):
                estimate = results.estimates[i]
                se = results.standard_errors[i]
                print(f"  {policy}: {estimate:.3f} ± {se:.3f}")

            # Best policy
            if len(results.estimates) > 0:
                best_idx = results.estimates.argmax()
                best_policy = target_policies[best_idx]
                print(f"\n🏆 Best policy: {best_policy}")

        # Save results if requested
        if args.output:
            from ..utils.export import export_results_json

            export_results_json(results, args.output)
            if not args.quiet:
                print(f"\n✓ Results saved to: {args.output}")

        return 0

    except FileNotFoundError as e:
        print(f"❌ Error: Dataset file not found: {e}", file=sys.stderr)
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


def validate_data(args: argparse.Namespace) -> int:
    """Run the validate command using existing validation utilities."""
    from .. import load_dataset_from_jsonl
    from ..data.validation import validate_cje_data

    try:
        if not args.verbose:
            print(f"Validating {args.dataset}...")

        # Load dataset
        dataset = load_dataset_from_jsonl(args.dataset)

        # Convert to list of dicts for validation (backward compatibility)
        data_list = []
        for sample in dataset.samples:
            record = {
                "prompt": sample.prompt,
                "response": sample.response,
                "base_policy_logprob": sample.base_policy_logprob,
                "target_policy_logprobs": sample.target_policy_logprobs,
                "metadata": sample.metadata,
            }
            if sample.reward is not None:
                record["reward"] = sample.reward
            data_list.append(record)

        # Use existing validation function
        is_valid, issues = validate_cje_data(
            data_list,
            reward_field="reward",
            judge_field="judge_score",
            oracle_field="oracle_label",
        )

        # Display results
        print(f"✓ Loaded {dataset.n_samples} samples")
        print(f"✓ Target policies: {', '.join(dataset.target_policies)}")

        # Check rewards
        n_with_rewards = sum(1 for s in dataset.samples if s.reward is not None)
        if n_with_rewards > 0:
            print(f"✓ Rewards: {n_with_rewards}/{dataset.n_samples} samples")

        if issues:
            print("\n⚠️  Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n✓ Dataset is valid and ready for analysis")

        if args.verbose:
            # Detailed statistics
            print("\nDetailed Statistics:")
            print("-" * 40)

            # Judge scores and oracle labels
            judge_scores = []
            oracle_labels = []
            for s in dataset.samples:
                if s.judge_score is not None:
                    judge_scores.append(s.judge_score)
                if s.oracle_label is not None:
                    oracle_labels.append(s.oracle_label)

            if judge_scores:
                import numpy as np

                print(f"Judge scores: {len(judge_scores)} samples")
                print(f"  Range: [{min(judge_scores):.3f}, {max(judge_scores):.3f}]")
                print(f"  Mean: {np.mean(judge_scores):.3f}")

            if oracle_labels:
                print(f"Oracle labels: {len(oracle_labels)} samples")
                print(f"  Range: [{min(oracle_labels):.3f}, {max(oracle_labels):.3f}]")
                print(f"  Mean: {np.mean(oracle_labels):.3f}")

            # Valid samples per policy
            print(f"\nValid samples per policy:")
            for policy in dataset.target_policies:
                n_valid = sum(
                    1
                    for s in dataset.samples
                    if s.base_policy_logprob is not None
                    and s.target_policy_logprobs.get(policy) is not None
                )
                print(f"  {policy}: {n_valid}/{dataset.n_samples}")

        return 0 if is_valid else 1

    except FileNotFoundError as e:
        print(f"❌ Error: Dataset file not found: {e}", file=sys.stderr)
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
