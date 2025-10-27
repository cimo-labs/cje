#!/usr/bin/env python
"""Plot OUA uncertainty vs. base uncertainty for DM+cov estimator.

Shows how uncertainty (standard error) components change with sample size.
Standard error directly determines confidence interval width.
Fixed oracle coverage of 25%.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Load ablation results
results_path = Path(__file__).parent.parent.parent / "cje/experiments/arena_10k_simplified/ablations/results/all_experiments.jsonl"

print(f"Loading results from {results_path}...")

# Extract DM+cov results for oracle_coverage=0.25 across different sample sizes
dm_cov_results = []

with open(results_path) as f:
    for line in f:
        result = json.loads(line)
        spec = result.get("spec", {})

        # Filter for DM+cov with oracle_coverage=0.25
        if (spec.get("estimator") == "direct" and
            spec.get("extra", {}).get("use_covariates") == True and
            spec.get("oracle_coverage") == 0.25):

            dm_cov_results.append({
                "sample_size": spec.get("sample_size"),
                "estimates": result.get("estimates", {}),
                "standard_errors": result.get("standard_errors", {}),
                "oracle_truths": result.get("oracle_truths", {}),
            })

print(f"Found {len(dm_cov_results)} DM+cov results with oracle_coverage=25%")

# Group by sample size
size_groups = defaultdict(list)
for r in dm_cov_results:
    n = r["sample_size"]
    if n is not None:
        size_groups[n].append(r)

# Sort sample sizes
sample_sizes = sorted(size_groups.keys())
print(f"Sample sizes: {sample_sizes}")

# For each sample size, compute average SE (uncertainty) across policies and seeds
avg_ses = []

for n in sample_sizes:
    results = size_groups[n]

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
        print(f"Sample size {n}: avg SE (uncertainty) = {avg_se:.6f}")
    else:
        avg_ses.append(np.nan)

# Convert to arrays
sample_sizes = np.array(sample_sizes)
avg_ses = np.array(avg_ses)

# Estimate base SE from largest sample size (where OUA contribution is smallest relative to sampling error)
# For a more accurate decomposition, we'll use the theoretical √n scaling
# SE_base ~ 1/√n, so we can estimate base uncertainty contribution

# Get the largest sample size result
if len(sample_sizes) > 0:
    # For uncertainty decomposition, we'll use the fact that:
    # - Base (sampling) SE scales as 1/√n
    # - OUA SE is relatively constant with n (it's about oracle learning, not sample size)

    # Estimate OUA component from smallest n (where it dominates)
    # and base scaling from largest n (where sampling dominates)

    # We'll estimate by fitting: SE² = a/n + b
    # where a = base variance scaling, b = OUA variance (approximately constant)

    # Using the fact that Var(SE) = a/n + b
    var_ses = avg_ses**2

    # Fit using least squares
    # var = a/n + b
    # Rearrange: var*n = a + b*n
    # This is linear in n

    from scipy.optimize import curve_fit

    def model(n, a, b):
        return a / n + b

    # Fit the model
    try:
        popt, _ = curve_fit(model, sample_sizes, var_ses, p0=[0.001, 0.0001])
        a_est, b_est = popt

        # Base SE component (sampling): sqrt(a/n)
        base_ses = np.sqrt(a_est / sample_sizes)

        # OUA SE component: sqrt(b)
        oua_se_const = np.sqrt(max(b_est, 0))
        oua_ses = oua_se_const * np.ones_like(sample_sizes)

        print(f"\nFitted model parameters:")
        print(f"  a (base variance scaling) = {a_est:.8f}")
        print(f"  b (OUA variance) = {b_est:.8f}")
        print(f"  OUA SE (constant) = {oua_se_const:.6f}")

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot stacked area (using SE = uncertainty)
        ax.fill_between(
            sample_sizes,
            0,
            base_ses,
            alpha=0.7,
            color="#FED976",
            label="Base uncertainty (sampling)",
            zorder=5,
        )
        ax.fill_between(
            sample_sizes,
            base_ses,
            avg_ses,
            alpha=0.7,
            color="#FC4E2A",
            label="OUA uncertainty (oracle learning)",
            zorder=6,
        )

        # Plot total uncertainty line with white-edged markers (matching two-stage style)
        ax.plot(
            sample_sizes,
            avg_ses,
            "-",
            color="black",
            linewidth=3.5,
            alpha=1.0,
            zorder=10,
        )
        ax.scatter(
            sample_sizes,
            avg_ses,
            s=150,
            alpha=1.0,
            color="black",
            edgecolor="white",
            linewidth=3,
            label="Total uncertainty",
            zorder=11,
        )

        ax.set_xlabel("Sample Size (n)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Uncertainty (Standard Error)", fontsize=14, fontweight="bold")
        ax.set_title(
            "Uncertainty Decomposition: Sampling vs. Oracle Learning (DM+cov, 25% oracle)",
            fontsize=15,
            fontweight="bold",
            pad=15,
        )
        ax.legend(fontsize=12, loc="upper right", framealpha=0.95)
        ax.grid(True, alpha=0.3)

        # Set x-axis to use nice round numbers
        ax.set_xlim([0, max(sample_sizes) * 1.05])

        plt.tight_layout()

        # Save plot
        output_path = Path(__file__).parent / "oua_vs_base_sample_size.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

        print(f"\n✓ Plot saved to {output_path}")

        # Print summary table
        print("\nUncertainty Decomposition Summary:")
        print("=" * 95)
        print(f"{'Sample Size':<15} {'Total SE':<15} {'Base SE':<15} {'OUA SE':<15} {'OUA %':<10}")
        print("=" * 95)
        for n, tot_se, base_se, oua_se in zip(sample_sizes, avg_ses, base_ses, oua_ses):
            oua_pct = 100 * (oua_se**2) / (tot_se**2) if tot_se > 0 else 0
            print(f"{n:<15d} {tot_se:<15.6f} {base_se:<15.6f} {oua_se:<15.6f} {oua_pct:<10.1f}%")
        print("=" * 95)
        print("\nNote: Standard Error (SE) determines confidence interval width.")
        print("      Larger sample size → lower base uncertainty.")
        print("      OUA uncertainty depends on oracle coverage (fixed at 25% here), not sample size.")

    except Exception as e:
        print(f"\nError fitting model: {e}")
        print("Could not decompose uncertainty components")

else:
    print("\nError: No results found for oracle_coverage=25%")
