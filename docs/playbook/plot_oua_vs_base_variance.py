#!/usr/bin/env python
"""Plot OUA uncertainty vs. base uncertainty for DM+cov estimator.

Shows how uncertainty (standard error) components change with oracle coverage.
Standard error directly determines confidence interval width.
Fixed sample size of 1000.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Load ablation results
results_path = Path(__file__).parent.parent.parent / "cje/experiments/arena_10k_simplified/ablations/results/all_experiments.jsonl"

print(f"Loading results from {results_path}...")

# Extract DM+cov results for n=1000 across different oracle coverages
dm_cov_results = []

with open(results_path) as f:
    for line in f:
        result = json.loads(line)
        spec = result.get("spec", {})

        # Filter for DM+cov with n=1000
        if (spec.get("estimator") == "direct" and
            spec.get("extra", {}).get("use_covariates") == True and
            spec.get("sample_size") == 1000):

            dm_cov_results.append({
                "oracle_coverage": spec.get("oracle_coverage"),
                "estimates": result.get("estimates", {}),
                "standard_errors": result.get("standard_errors", {}),
                "oracle_truths": result.get("oracle_truths", {}),
            })

print(f"Found {len(dm_cov_results)} DM+cov results with n=1000")

# Group by oracle coverage
coverage_groups = defaultdict(list)
for r in dm_cov_results:
    cov = r["oracle_coverage"]
    if cov is not None:
        coverage_groups[cov].append(r)

# Sort coverages
coverages = sorted(coverage_groups.keys())
print(f"Oracle coverages: {coverages}")

# For each coverage, compute average SE (uncertainty) across policies and seeds
# At 100% oracle coverage, OUA = 0, so SE = base SE only
# At lower coverages, total uncertainty combines base uncertainty + OUA uncertainty

# We'll use the average SE across all policies as our metric
avg_ses = []

for cov in coverages:
    results = coverage_groups[cov]

    # Collect all SEs across all policies and seeds
    all_ses = []
    for r in results:
        ses = r["standard_errors"]
        if isinstance(ses, dict):
            all_ses.extend(ses.values())
        elif isinstance(ses, list):
            all_ses.extend(ses)

    if all_ses:
        avg_se = np.mean(all_ses)
        avg_ses.append(avg_se)
        print(f"Coverage {cov:.0%}: avg SE (uncertainty) = {avg_se:.6f}")
    else:
        avg_ses.append(np.nan)

# Convert to arrays
coverages = np.array(coverages)
avg_ses = np.array(avg_ses)

# Estimate base SE from 100% coverage (where OUA = 0)
if 1.0 in coverages:
    idx_full = np.where(coverages == 1.0)[0][0]
    base_se = avg_ses[idx_full]
    print(f"\nBase uncertainty (at 100% oracle): {base_se:.6f}")

    # Compute OUA SE component using variance decomposition
    # SE_total^2 = SE_base^2 + SE_oua^2
    oua_ses = np.sqrt(np.maximum(avg_ses**2 - base_se**2, 0))

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert coverage to percentage for x-axis
    coverage_pct = coverages * 100

    # Plot stacked area (using SE = uncertainty)
    ax.fill_between(
        coverage_pct,
        0,
        base_se * np.ones_like(coverage_pct),
        alpha=0.7,
        color="#FED976",
        label="Base uncertainty (sampling)",
        zorder=5,
    )
    ax.fill_between(
        coverage_pct,
        base_se * np.ones_like(coverage_pct),
        avg_ses,
        alpha=0.7,
        color="#FC4E2A",
        label="OUA uncertainty (oracle learning)",
        zorder=6,
    )

    # Plot total uncertainty line with white-edged markers (matching two-stage style)
    ax.plot(
        coverage_pct,
        avg_ses,
        "-",
        color="black",
        linewidth=3.5,
        alpha=1.0,
        zorder=10,
    )
    ax.scatter(
        coverage_pct,
        avg_ses,
        s=150,
        alpha=1.0,
        color="black",
        edgecolor="white",
        linewidth=3,
        label="Total uncertainty",
        zorder=11,
    )

    ax.set_xlabel("Oracle Coverage (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Uncertainty (Standard Error)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Uncertainty Decomposition: Sampling vs. Oracle Learning (DM+cov, n=1000)",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    ax.legend(fontsize=12, loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Set x-axis limits to make clear we start at 5%, not 0%
    ax.set_xlim([0, 105])
    ax.set_xticks([0, 5, 10, 25, 50, 100])

    # Add vertical line at x=0 to emphasize we don't have data there
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    plt.tight_layout()

    # Save plot
    output_path = Path(__file__).parent / "oua_vs_base_variance.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"\n✓ Plot saved to {output_path}")

    # Print summary table
    print("\nUncertainty Decomposition Summary:")
    print("=" * 85)
    print(f"{'Coverage':<12} {'Total SE':<15} {'Base SE':<15} {'OUA SE':<15} {'OUA %':<10}")
    print("=" * 85)
    for cov, tot_se, oua_se in zip(coverages, avg_ses, oua_ses):
        oua_pct = 100 * (oua_se**2) / (tot_se**2) if tot_se > 0 else 0
        print(f"{cov:>10.0%}  {tot_se:<15.6f} {base_se:<15.6f} {oua_se:<15.6f} {oua_pct:<10.1f}%")
    print("=" * 85)
    print("\nNote: Standard Error (SE) determines confidence interval width.")
    print("      Lower oracle coverage → higher uncertainty → wider confidence intervals.")

else:
    print("\nError: No 100% oracle coverage results found to compute base variance")
