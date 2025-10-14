#!/usr/bin/env env python
"""Generate reward calibration plot for playbook using Arena experiment data."""

import json
import numpy as np
from pathlib import Path
from cje.visualization.calibration import plot_calibration_comparison
from cje.calibration import JudgeCalibrator

# Load arena data
data_path = (
    Path(__file__).parent.parent.parent
    / "cje/experiments/arena_10k_simplified/data/cje_dataset.jsonl"
)

print(f"Loading data from {data_path}...")
samples = []
with open(data_path) as f:
    for line in f:
        samples.append(json.loads(line))

print(f"Loaded {len(samples)} samples")

# Extract judge scores and oracle labels
judge_scores = []
oracle_labels = []

for sample in samples:
    # Get judge score and oracle label from metadata
    metadata = sample.get("metadata", {})
    judge_score = metadata.get("judge_score")
    oracle_label = metadata.get("oracle_label")

    if judge_score is not None and oracle_label is not None:
        judge_scores.append(judge_score)
        oracle_labels.append(oracle_label)

judge_scores = np.array(judge_scores)
oracle_labels = np.array(oracle_labels)

print(f"Found {len(judge_scores)} samples with both judge scores and oracle labels")
print(f"Judge score range: [{judge_scores.min():.3f}, {judge_scores.max():.3f}]")
print(f"Oracle label range: [{oracle_labels.min():.3f}, {oracle_labels.max():.3f}]")

# Fit judge calibrator
print("\nFitting judge calibrator...")
calibrator = JudgeCalibrator()
result = calibrator.fit_transform(judge_scores, oracle_labels)

# Get calibrated scores
calibrated_scores = result.calibrated_scores

print(
    f"Calibrated score range: [{calibrated_scores.min():.3f}, {calibrated_scores.max():.3f}]"
)

# Generate plot
print("\nGenerating calibration plot...")
output_path = Path(__file__).parent / "calibration_plot.png"

fig = plot_calibration_comparison(
    judge_scores=judge_scores,
    oracle_labels=oracle_labels,
    calibrated_scores=calibrated_scores,
    calibrator=calibrator,
    save_path=output_path,
    figsize=(10, 8),
)

print(f"\n✓ Plot saved to {output_path}")
print("\nPlot shows:")
print("  - Density heatmap of judge scores vs oracle labels")
print("  - Empirical mean E[Oracle|Judge] (dark blue curve)")
print("  - Fitted calibration function f(Judge) (red curve)")
print("  - Perfect calibration reference (gray diagonal)")
