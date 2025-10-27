#!/usr/bin/env python3
"""Generate a forest plot from direct mode analysis showing estimates vs oracle truth."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from cje import analyze_dataset

# Paths to arena_10k_simplified data (full dataset with all 5 policies)
DATA_DIR = Path(__file__).parent.parent / "cje" / "experiments" / "arena_10k_simplified" / "data"
FRESH_DRAWS = DATA_DIR / "responses"

print("="*70)
print("DIRECT MODE FOREST PLOT EXAMPLE")
print("="*70)

# Run direct mode analysis (fresh draws only, no logged data)
print("\nRunning direct mode analysis...")
results = analyze_dataset(
    fresh_draws_dir=str(FRESH_DRAWS),
    estimator="direct",
    include_response_length=True,  # Add response_length as covariate
    verbose=True,
)

print("\n" + "="*70)
print("RESULTS")
print("="*70)

# Extract results
policies = results.metadata["target_policies"]
estimates = results.estimates
standard_errors = results.standard_errors
ci_lower, ci_upper = results.confidence_interval(alpha=0.05)

# Get oracle truths from the fresh draws
from cje.data.fresh_draws import load_fresh_draws_auto

oracle_truths = {}
for policy in policies:
    fd = load_fresh_draws_auto(FRESH_DRAWS, policy, verbose=False)
    oracle_labels = [s.oracle_label for s in fd.samples if s.oracle_label is not None]
    if oracle_labels:
        oracle_truths[policy] = np.mean(oracle_labels)
    else:
        oracle_truths[policy] = None

# Print results
print(f"\n{'Policy':<30} {'Estimate':>10} {'SE':>10} {'95% CI':>20} {'Oracle':>10}")
print("-" * 90)
for i, policy in enumerate(policies):
    est = estimates[i]
    se = standard_errors[i]
    ci_l = ci_lower[i]
    ci_u = ci_upper[i]
    oracle = oracle_truths.get(policy)

    oracle_str = f"{oracle:.4f}" if oracle is not None else "N/A"
    print(f"{policy:<30} {est:>10.4f} {se:>10.4f}  [{ci_l:>7.4f}, {ci_u:>7.4f}]  {oracle_str:>10}")

# Create forest plot
print("\nCreating forest plot...")

fig, ax = plt.subplots(figsize=(10, max(4, len(policies) * 0.35)))

# Sort policies alphabetically for consistent display
policy_indices = sorted(range(len(policies)), key=lambda i: policies[i])
sorted_policies = [policies[i] for i in policy_indices]
sorted_estimates = [estimates[i] for i in policy_indices]
sorted_ci_lower = [ci_lower[i] for i in policy_indices]
sorted_ci_upper = [ci_upper[i] for i in policy_indices]
sorted_oracles = [oracle_truths.get(policies[i]) for i in policy_indices]

# Y positions (reverse so first is at top)
y_pos = np.arange(len(sorted_policies))[::-1]

# Calculate error bars
yerr_lower = [est - ci_l for est, ci_l in zip(sorted_estimates, sorted_ci_lower)]
yerr_upper = [ci_u - est for est, ci_u in zip(sorted_estimates, sorted_ci_upper)]

# Plot oracle truth (if available)
oracle_y = []
oracle_x = []
for i, (y, oracle) in enumerate(zip(y_pos, sorted_oracles)):
    if oracle is not None:
        oracle_y.append(y)
        oracle_x.append(oracle)

if oracle_x:
    ax.scatter(oracle_x, oracle_y, color='red', marker='d', s=100,
              label='Oracle truth', zorder=3, alpha=0.8, edgecolors='darkred', linewidths=1.5)

# Plot estimates with error bars
ax.errorbar(sorted_estimates, y_pos,
            xerr=[yerr_lower, yerr_upper],
            fmt='o', markersize=8, capsize=5, capthick=2,
            color='steelblue', ecolor='steelblue',
            label='Estimate ± 95% CI', zorder=2)

# Formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_policies, fontsize=11)
ax.set_xlabel('Oracle Score', fontsize=12)
ax.set_ylabel('Policy', fontsize=12)
ax.set_title('Forest Plot: direct+cov\n(Arena 10k Simplified Data)',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='best', fontsize=10)
ax.grid(axis='x', alpha=0.3, linestyle=':')

plt.tight_layout()

# Save figure
output_path = Path(__file__).parent / "forest_plot_direct_mode.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved forest plot to {output_path}")

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
