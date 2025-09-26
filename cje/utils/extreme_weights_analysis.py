"""Analyze and log extreme importance weights for debugging."""

import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging

# Import compute_ess from diagnostics module
from ..diagnostics import compute_ess

logger = logging.getLogger(__name__)


def analyze_extreme_weights(
    dataset: Any,  # Dataset object
    sampler: Any,  # PrecomputedSampler object
    raw_weights_dict: Dict[str, np.ndarray],
    calibrated_weights_dict: Dict[str, np.ndarray],
    n_extreme: int = 5,
    output_dir: Optional[Path] = None,
    near_zero_threshold: float = 1e-10,
) -> Tuple[Dict[str, Any], str]:
    """Analyze extreme weights and generate detailed report.

    Args:
        dataset: Dataset with samples
        sampler: PrecomputedSampler with log probabilities
        raw_weights_dict: Raw importance weights by policy
        calibrated_weights_dict: Calibrated weights by policy
        n_extreme: Number of extreme samples to analyze
        output_dir: Directory to save reports (optional)
        near_zero_threshold: Threshold for near-zero weights (default: 1e-10)

    Returns:
        Tuple of (json_report, text_report)
    """

    # Initialize report structure
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_samples": dataset.n_samples,
            "n_valid_samples": sampler.n_valid_samples,
            "policies_analyzed": list(raw_weights_dict.keys()),
            "n_extreme_analyzed": n_extreme,
        },
        "per_policy_analysis": {},
        "cross_policy_insights": {
            "consistently_extreme": [],
            "high_variance_samples": [],
        },
    }

    # Track extreme samples across policies
    all_extreme_high: Dict[str, List[str]] = {}  # sample_id -> list of policies
    all_extreme_low: Dict[str, List[str]] = {}  # sample_id -> list of policies

    # Analyze each policy
    for policy_name in raw_weights_dict.keys():
        raw_weights = raw_weights_dict[policy_name]
        cal_weights = calibrated_weights_dict.get(policy_name, raw_weights)

        # Get valid data for this policy
        policy_data = sampler.get_data_for_policy(policy_name)
        if not policy_data:
            continue

        # Calculate statistics
        finite_raw = raw_weights[np.isfinite(raw_weights) & (raw_weights > 0)]

        policy_analysis: Dict[str, Any] = {
            "statistics": {
                "raw_weight_range": (
                    [float(np.min(finite_raw)), float(np.max(finite_raw))]
                    if len(finite_raw) > 0
                    else [0, 0]
                ),
                "calibrated_weight_range": [
                    float(np.min(cal_weights)),
                    float(np.max(cal_weights)),
                ],
                "n_clipped_high": int(
                    np.sum(raw_weights >= 100.0)
                ),  # Assuming clip at 100
                "n_near_zero": int(np.sum(raw_weights < near_zero_threshold)),
                "ess_raw": f"{compute_ess(raw_weights) / len(raw_weights) * 100:.1f}%",
                "ess_calibrated": f"{compute_ess(cal_weights) / len(cal_weights) * 100:.1f}%",
            },
            "extreme_samples": {
                "highest_weights": [],
                "lowest_weights": [],
            },
        }

        # Find extreme samples
        sorted_indices = np.argsort(raw_weights)
        highest_indices = sorted_indices[-n_extreme:][::-1]  # Reverse for descending
        lowest_indices = sorted_indices[:n_extreme]

        # Analyze highest weight samples
        for rank, idx in enumerate(highest_indices, 1):
            if idx >= len(policy_data):
                continue

            sample_data = policy_data[idx]
            sample_id = sample_data.get("prompt_id", f"sample_{idx}")

            # Track for cross-policy analysis
            if sample_id not in all_extreme_high:
                all_extreme_high[sample_id] = []
            all_extreme_high[sample_id].append(policy_name)

            # Get prompt and response (truncate for display)
            prompt = sample_data.get("prompt", "")[:100]
            response = sample_data.get("response", "")[:100]

            # Get log probabilities
            base_logp = sample_data.get("base_policy_logprob", 0)
            target_logp = sample_data.get("target_policy_logprobs", {}).get(
                policy_name, 0
            )
            log_ratio = target_logp - base_logp

            extreme_sample = {
                "rank": rank,
                "sample_id": sample_id,
                "prompt": prompt
                + ("..." if len(sample_data.get("prompt", "")) > 100 else ""),
                "response_preview": response
                + ("..." if len(sample_data.get("response", "")) > 100 else ""),
                "base_logprob": float(base_logp),
                "target_logprob": float(target_logp),
                "log_ratio": float(log_ratio),
                "raw_weight": float(raw_weights[idx]),
                "calibrated_weight": float(cal_weights[idx]),
                "reward": float(sample_data.get("reward", 0)),
                "explanation": _explain_weight(log_ratio),
            }

            policy_analysis["extreme_samples"]["highest_weights"].append(extreme_sample)

        # Analyze lowest weight samples
        for rank, idx in enumerate(lowest_indices, 1):
            if idx >= len(policy_data):
                continue

            sample_data = policy_data[idx]
            sample_id = sample_data.get("prompt_id", f"sample_{idx}")

            # Track for cross-policy analysis
            if sample_id not in all_extreme_low:
                all_extreme_low[sample_id] = []
            all_extreme_low[sample_id].append(policy_name)

            # Get prompt and response (truncate for display)
            prompt = sample_data.get("prompt", "")[:100]
            response = sample_data.get("response", "")[:100]

            # Get log probabilities
            base_logp = sample_data.get("base_policy_logprob", 0)
            target_logp = sample_data.get("target_policy_logprobs", {}).get(
                policy_name, 0
            )
            log_ratio = target_logp - base_logp

            extreme_sample = {
                "rank": rank,
                "sample_id": sample_id,
                "prompt": prompt
                + ("..." if len(sample_data.get("prompt", "")) > 100 else ""),
                "response_preview": response
                + ("..." if len(sample_data.get("response", "")) > 100 else ""),
                "base_logprob": float(base_logp),
                "target_logprob": float(target_logp),
                "log_ratio": float(log_ratio),
                "raw_weight": float(raw_weights[idx]),
                "calibrated_weight": float(cal_weights[idx]),
                "reward": float(sample_data.get("reward", 0)),
                "explanation": _explain_weight(log_ratio),
            }

            policy_analysis["extreme_samples"]["lowest_weights"].append(extreme_sample)

        report["per_policy_analysis"][policy_name] = policy_analysis

    # Cross-policy insights
    for sample_id, policies in all_extreme_high.items():
        if len(policies) > 1:
            report["cross_policy_insights"]["consistently_extreme"].append(
                {
                    "sample_id": sample_id,
                    "policies_extreme_high": policies,
                    "interpretation": "High weight across multiple policies suggests base policy underestimates this response",
                }
            )

    for sample_id, policies in all_extreme_low.items():
        if len(policies) > 1:
            report["cross_policy_insights"]["consistently_extreme"].append(
                {
                    "sample_id": sample_id,
                    "policies_extreme_low": policies,
                    "interpretation": "Low weight across multiple policies suggests base policy overestimates this response",
                }
            )

    # Generate text report
    text_report = _format_text_report(report)

    # Save reports if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        json_path = output_dir / "extreme_weights_analysis.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved JSON report to {json_path}")

        # Save text report
        text_path = output_dir / "extreme_weights_analysis.txt"
        with open(text_path, "w") as f:
            f.write(text_report)
        logger.info(f"Saved text report to {text_path}")

    return report, text_report


def _explain_weight(log_ratio: float) -> str:
    """Generate human-readable explanation for weight."""
    if log_ratio > 20:
        return f"Target policy MUCH more likely (Δ={log_ratio:.1f} nats)"
    elif log_ratio > 5:
        return f"Target policy more likely (Δ={log_ratio:.1f} nats)"
    elif log_ratio > -5:
        return f"Similar likelihood (Δ={log_ratio:.1f} nats)"
    elif log_ratio > -20:
        return f"Target policy less likely (Δ={log_ratio:.1f} nats)"
    else:
        return f"Target policy MUCH less likely (Δ={log_ratio:.1f} nats)"


def _format_text_report(report: Dict[str, Any]) -> str:
    """Format report as human-readable text."""
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("EXTREME WEIGHTS ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {report['metadata']['timestamp']}")
    lines.append(
        f"Samples: {report['metadata']['n_valid_samples']} valid / {report['metadata']['n_samples']} total"
    )
    lines.append(f"Policies: {', '.join(report['metadata']['policies_analyzed'])}")
    lines.append("=" * 80)
    lines.append("")

    # Per-policy analysis
    for policy, analysis in report["per_policy_analysis"].items():
        lines.append(f"POLICY: {policy}")
        lines.append("-" * 40)

        stats = analysis["statistics"]
        lines.append(
            f"Raw weight range: [{stats['raw_weight_range'][0]:.2e}, {stats['raw_weight_range'][1]:.2e}]"
        )
        lines.append(
            f"Calibrated range: [{stats['calibrated_weight_range'][0]:.3f}, {stats['calibrated_weight_range'][1]:.3f}]"
        )
        lines.append(f"ESS improvement: {stats['ess_raw']} → {stats['ess_calibrated']}")
        lines.append(
            f"Extreme weights: {stats['n_clipped_high']} very high (≥100), {stats['n_near_zero']} near-zero (<1e-10)"
        )
        lines.append("")

        # Highest weights
        lines.append(
            f"TOP {len(analysis['extreme_samples']['highest_weights'])} HIGHEST WEIGHTS:"
        )
        for sample in analysis["extreme_samples"]["highest_weights"]:
            lines.append(
                f"{sample['rank']}. Sample {sample['sample_id']} (raw: {sample['raw_weight']:.2e} → cal: {sample['calibrated_weight']:.3f})"
            )
            lines.append(f"   Prompt: {sample['prompt']}")
            lines.append(f"   Response: {sample['response_preview']}")
            lines.append(
                f"   Base logp: {sample['base_logprob']:.2f}, Target logp: {sample['target_logprob']:.2f} (Δ={sample['log_ratio']:+.2f})"
            )
            lines.append(f"   → {sample['explanation']}")
            lines.append("")

        # Lowest weights
        lines.append(
            f"BOTTOM {len(analysis['extreme_samples']['lowest_weights'])} LOWEST WEIGHTS:"
        )
        for sample in analysis["extreme_samples"]["lowest_weights"]:
            lines.append(
                f"{sample['rank']}. Sample {sample['sample_id']} (raw: {sample['raw_weight']:.2e} → cal: {sample['calibrated_weight']:.3f})"
            )
            lines.append(f"   Prompt: {sample['prompt']}")
            lines.append(f"   Response: {sample['response_preview']}")
            lines.append(
                f"   Base logp: {sample['base_logprob']:.2f}, Target logp: {sample['target_logprob']:.2f} (Δ={sample['log_ratio']:+.2f})"
            )
            lines.append(f"   → {sample['explanation']}")
            lines.append("")

        lines.append("")

    # Cross-policy patterns
    if report["cross_policy_insights"]["consistently_extreme"]:
        lines.append("=" * 80)
        lines.append("CROSS-POLICY PATTERNS")
        lines.append("=" * 80)
        lines.append("Samples consistently extreme across policies:")
        for pattern in report["cross_policy_insights"]["consistently_extreme"]:
            if "policies_extreme_high" in pattern:
                lines.append(
                    f"  • {pattern['sample_id']}: High in {', '.join(pattern['policies_extreme_high'])}"
                )
            else:
                lines.append(
                    f"  • {pattern['sample_id']}: Low in {', '.join(pattern['policies_extreme_low'])}"
                )
            lines.append(f"    {pattern['interpretation']}")
        lines.append("")

    return "\n".join(lines)
