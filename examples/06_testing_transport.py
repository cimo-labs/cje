#!/usr/bin/env python3
"""
Example 6: Testing Calibrator Transportability

Before reusing a calibrator across policies or time periods, test if it transports.
The probe protocol uses 40-60 labeled samples to detect miscalibration.
"""

from pathlib import Path
from cje.data import load_dataset_from_jsonl
from cje.calibration import calibrate_dataset
from cje.diagnostics.transport import audit_transportability
from cje.visualization.transport import plot_transport_audit
from copy import deepcopy

DATA_PATH = Path(__file__).parent / "arena_sample" / "logged_data.jsonl"

# Load dataset and calibrate on first portion
dataset = load_dataset_from_jsonl(str(DATA_PATH))
train_dataset = deepcopy(dataset)
train_dataset.samples = dataset.samples[:100]  # Use first 100 for calibration

calibrated, cal_result = calibrate_dataset(
    train_dataset, judge_field="judge_score", oracle_field="oracle_label"
)
calibrator = cal_result.calibrator

# Get probe samples from held-out portion (with oracle labels)
probe = [s for s in dataset.samples[100:200] if s.oracle_label is not None][:50]

# Run transport audit
diag = audit_transportability(calibrator, probe, group_label="arena_holdout")

print(diag.summary())
print(f"\nStatus: {diag.status}")
print(f"Action: {diag.recommended_action}")

# Visualize if available
try:
    plot_transport_audit(diag, save_path=Path("transport_audit.png"))
except ImportError:
    print("(Install matplotlib to visualize)")
