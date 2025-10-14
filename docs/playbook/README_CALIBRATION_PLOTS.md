# Calibration Visualization Scripts

This directory contains scripts to generate calibration visualizations using Arena experiment data.

## Generated Plots

### 1. calibration_plot.png
Standard monotone calibration visualization showing:
- Density heatmap of judge scores vs oracle labels
- Empirical mean E[Oracle|Judge] curve
- Fitted isotonic calibration function
- Calibration statistics (ECE, RMSE)

**Generate:** `python generate_calibration_plot.py`

### 2. two_stage_comparison.png
Side-by-side comparison of monotone vs two-stage calibration:
- Left: Standard monotone isotonic regression
- Right: Two-stage flexible calibration (spline → rank → isotonic)

**Status:** Moved to root directory and incorporated into main README

**Generate:** `python generate_two_stage_comparison.py`

### 3. two_stage_detailed.png
Four-panel pipeline breakdown showing:
1. Raw data with empirical means
2. Stage 1: Smooth spline transformation g(S)
3. Stage 2: Rank transformation for monotonicity
4. Final calibration function

**Generate:** `python generate_two_stage_detailed.py`

## Data Source

All visualizations use data from:
```
../../cje/experiments/arena_10k_simplified/data/cje_dataset.jsonl
```

- 4,989 samples with full oracle coverage
- Judge scores and oracle labels both in [0, 1]

## When Two-Stage Helps

Two-stage calibration provides benefits when:
- **Regional miscalibration**: Monotone fits well at extremes but poorly in middle
- **Length bias**: Judge scores don't account for response length effects on quality
- **Non-monotone patterns**: Empirical E[Oracle|Judge] has local non-monotonicity

On the Arena dataset, both methods perform similarly (RMSE ~0.197) because the relationship is already fairly monotone. Two-stage would show larger improvements on datasets with clear non-monotone patterns.

## Regenerating Plots

All scripts are standalone and can be run anytime:

```bash
cd docs/playbook

# Generate all plots
python generate_calibration_plot.py
python generate_two_stage_comparison.py
python generate_two_stage_detailed.py
```

Each script:
- Loads Arena data automatically
- Fits the appropriate calibrator(s)
- Saves high-resolution PNG (150 DPI)
- Prints summary statistics
