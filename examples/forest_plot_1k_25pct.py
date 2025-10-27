#!/usr/bin/env python3
"""Generate forest plot using analyze_dataset() with n=1k, 25% oracle coverage."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json
import tempfile
import shutil
from cje import analyze_dataset
from cje.data.fresh_draws import load_fresh_draws_auto

# Paths to arena_10k_simplified data
DATA_DIR = Path(__file__).parent.parent / "cje" / "experiments" / "arena_10k_simplified" / "data"
FRESH_DRAWS = DATA_DIR / "responses"

# Configuration to match ablation scenario
SAMPLE_SIZE = 1000
ORACLE_COVERAGE = 0.25
SEED = 123

print("="*70)
print("FOREST PLOT: DIRECT MODE (n=1k, oracle=25%)")
print("="*70)
print(f"\nConfiguration:")
print(f"  Sample size: {SAMPLE_SIZE}")
print(f"  Oracle coverage: {int(ORACLE_COVERAGE*100)}%")
print(f"  Seed: {SEED}")

# Create temporary directory with subsampled data
print("\nPreparing subsampled data...")
temp_dir = Path(tempfile.mkdtemp())
temp_fresh_draws = temp_dir / "fresh_draws"
temp_fresh_draws.mkdir()

np.random.seed(SEED)

oracle_truths = {}  # Store full oracle truth for comparison

# Subsample each policy's responses
for policy_file in FRESH_DRAWS.glob("*_responses.jsonl"):
    policy = policy_file.stem.replace("_responses", "")

    # Load all responses
    with open(policy_file) as f:
        all_responses = [json.loads(line) for line in f]

    # Compute oracle truth from all data (for evaluation)
    all_oracle_labels = [r.get("metadata", {}).get("oracle_label")
                         for r in all_responses
                         if r.get("metadata", {}).get("oracle_label") is not None]
    if all_oracle_labels:
        oracle_truths[policy] = np.mean(all_oracle_labels)

    # Subsample to SAMPLE_SIZE
    indices = np.random.choice(len(all_responses), SAMPLE_SIZE, replace=False)
    sampled_responses = [all_responses[i] for i in sorted(indices)]

    # Determine which samples get oracle labels (ORACLE_COVERAGE fraction)
    n_oracle = int(SAMPLE_SIZE * ORACLE_COVERAGE)
    oracle_indices = set(np.random.choice(len(sampled_responses), n_oracle, replace=False))

    # Remove oracle labels from non-oracle samples
    for i, resp in enumerate(sampled_responses):
        if i not in oracle_indices:
            if "metadata" in resp and "oracle_label" in resp["metadata"]:
                resp["metadata"]["oracle_label"] = None

    # Write subsampled data
    output_file = temp_fresh_draws / f"{policy}_responses.jsonl"
    with open(output_file, "w") as f:
        for resp in sampled_responses:
            f.write(json.dumps(resp) + "\n")

    n_with_oracle = sum(1 for r in sampled_responses
                        if r.get("metadata", {}).get("oracle_label") is not None)
    print(f"  {policy}: {len(sampled_responses)} samples, {n_with_oracle} with oracle labels")

# Run direct mode analysis
print("\nRunning direct mode analysis...")
results = analyze_dataset(
    fresh_draws_dir=str(temp_fresh_draws),
    estimator="direct",
    include_response_length=True,  # Add response_length as covariate
    verbose=False,
)

# Clean up temp directory
shutil.rmtree(temp_dir)

print("\n" + "="*70)
print("RESULTS")
print("="*70)

# Extract results
policies = results.metadata["target_policies"]
estimates = results.estimates
standard_errors = results.standard_errors
ci_lower, ci_upper = results.confidence_interval(alpha=0.05)

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
ax.set_title(f'Policy Performance Estimates vs. Ground Truth\n(n={SAMPLE_SIZE}, oracle={int(ORACLE_COVERAGE*100)}%)',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='best', fontsize=10)
ax.grid(axis='x', alpha=0.3, linestyle=':')

plt.tight_layout()

# Save figure
output_path = Path(__file__).parent / f"forest_plot_n{SAMPLE_SIZE}_oracle{int(ORACLE_COVERAGE*100)}.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved forest plot to {output_path}")

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
