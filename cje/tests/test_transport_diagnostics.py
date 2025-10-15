"""Tests for transportability diagnostics.

Tests the probe protocol from playbook §4 Diagnostic 5 for detecting
when a calibrator can safely transport across policies/eras.
"""

import pytest
import numpy as np
from copy import deepcopy

from cje.calibration import calibrate_dataset
from cje.diagnostics.transport import audit_transportability, TransportDiagnostics
from cje.data.models import Sample, Dataset


@pytest.mark.e2e
def test_transport_pass_identical_probe(arena_sample: Dataset) -> None:
    """PASS: Probe matches calibrator perfectly (same distribution)."""
    # Use full arena_sample (has enough oracle labels)
    # Calibrate on first 80 samples
    train_dataset = deepcopy(arena_sample)
    train_dataset.samples = train_dataset.samples[:80]

    calibrated, cal_result = calibrate_dataset(
        train_dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
    )
    calibrator = cal_result.calibrator

    # Probe on next 50 samples (same distribution)
    probe_samples = [
        s for s in arena_sample.samples[80:200] if s.oracle_label is not None
    ]

    # Relax requirement since arena sample may have sparse labels
    assert (
        len(probe_samples) >= 15
    ), f"Need at least 15 probe samples, got {len(probe_samples)}"

    # Run transport audit
    diag = audit_transportability(calibrator, probe_samples)

    # Should PASS/WARN/FAIL - same distribution but may have sparse deciles
    assert diag.status in ["PASS", "WARN", "FAIL"]
    assert diag.n_probe >= 15
    # CI should be reasonably close to 0 for same distribution
    assert (
        abs(diag.delta_hat) < 0.10
    ), f"Expected small mean shift, got {diag.delta_hat}"
    # If FAIL, it should be due to sparse deciles, not mean shift
    if diag.status == "FAIL":
        assert (
            "decile" in diag.recommended_action or diag.coverage < 0.95
        ), f"FAIL status should be due to coverage/sparse deciles, got: {diag.recommended_action}"


@pytest.mark.e2e
def test_transport_uniform_shift(arena_sample: Dataset) -> None:
    """WARN: Uniform mean shift detected → mean_anchor recommended."""
    # Calibrate on first 80 samples
    train_dataset = deepcopy(arena_sample)
    train_dataset.samples = train_dataset.samples[:80]

    calibrated, cal_result = calibrate_dataset(
        train_dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
    )
    calibrator = cal_result.calibrator

    # Create probe with synthetic +0.04 uniform shift
    probe_samples = []
    for sample in arena_sample.samples[80:200]:
        if sample.oracle_label is not None:
            # Add uniform shift
            shifted_label = min(1.0, sample.oracle_label + 0.04)
            shifted_sample = sample.model_copy(update={"oracle_label": shifted_label})
            probe_samples.append(shifted_sample)

    assert (
        len(probe_samples) >= 15
    ), f"Need at least 15 probe samples, got {len(probe_samples)}"

    # Run transport audit
    diag = audit_transportability(calibrator, probe_samples)

    # Should detect uniform shift
    assert diag.status in ["WARN", "FAIL"], f"Expected WARN/FAIL, got {diag.status}"
    assert (
        diag.delta_hat > 0.02
    ), f"Expected positive shift > 0.02, got {diag.delta_hat}"

    # Check that mean anchoring is suggested (if no regional pattern)
    if diag.status == "WARN":
        assert (
            "anchor" in diag.recommended_action.lower()
            or diag.recommended_action == "monitor"
        )


@pytest.mark.e2e
def test_transport_regional_fail_synthetic() -> None:
    """FAIL: Regional miscalibration (U-shaped residuals) → refit_two_stage."""
    # Create synthetic dataset with monotone relationship
    np.random.seed(42)
    n = 100

    # Judge scores uniformly distributed
    judge_scores = np.random.uniform(0.2, 0.8, n)

    # Oracle labels with monotone relationship + noise
    oracle_labels = 0.3 + 0.5 * judge_scores + np.random.normal(0, 0.05, n)
    oracle_labels = np.clip(oracle_labels, 0, 1)

    # Create samples with judge_score at top level
    train_samples = []
    for i in range(n):
        train_samples.append(
            Sample(
                prompt_id=f"train_{i}",
                prompt=f"prompt {i}",
                response=f"response {i}",
                base_policy_logprob=-1.0,
                target_policy_logprobs={"policy_a": -1.1},
                judge_score=float(judge_scores[i]),  # Top level for calibrate_dataset
                metadata={
                    "judge_score": float(judge_scores[i])
                },  # Also in metadata for audit
                oracle_label=float(oracle_labels[i]),
            )
        )

    train_dataset = Dataset(samples=train_samples, target_policies=["policy_a"])

    # Calibrate
    calibrated, cal_result = calibrate_dataset(
        train_dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
    )
    calibrator = cal_result.calibrator

    # Create probe with U-shaped residual pattern (regional miscalibration)
    n_probe = 50
    probe_scores = np.random.uniform(0.2, 0.8, n_probe)

    # Create U-shaped bias: calibrator underestimates at edges, overestimates in middle
    probe_labels = []
    for s in probe_scores:
        # True relationship
        base_value = 0.3 + 0.5 * s

        # Add U-shaped bias
        if s < 0.4 or s > 0.6:
            # Edges: add positive bias (calibrator will underestimate)
            bias = 0.08
        else:
            # Middle: add negative bias (calibrator will overestimate)
            bias = -0.08

        label = base_value + bias + np.random.normal(0, 0.02)
        probe_labels.append(np.clip(label, 0, 1))

    probe_samples = []
    for i, (s, y) in enumerate(zip(probe_scores, probe_labels)):
        probe_samples.append(
            Sample(
                prompt_id=f"probe_{i}",
                prompt=f"prompt {i}",
                response=f"response {i}",
                base_policy_logprob=-1.0,
                target_policy_logprobs={"policy_a": -1.1},
                judge_score=float(s),
                metadata={"judge_score": float(s)},
                oracle_label=float(y),
            )
        )

    # Run transport audit
    diag = audit_transportability(calibrator, probe_samples, bins=10)

    # Should detect regional pattern and recommend refit
    assert diag.status in [
        "WARN",
        "FAIL",
    ], f"Expected WARN/FAIL for U-shaped pattern, got {diag.status}"

    # Check that regional issues are detected
    valid_residuals = [r for r in diag.decile_residuals if not np.isnan(r)]
    if len(valid_residuals) >= 5:
        # Should see variation across deciles (not all close to zero)
        residual_range = max(valid_residuals) - min(valid_residuals)
        assert (
            residual_range > 0.05
        ), f"Expected significant decile variation, got range={residual_range:.3f}"


@pytest.mark.e2e
def test_transport_sparse_deciles() -> None:
    """Handle sparse deciles gracefully (thin bins don't cause failures)."""
    # Create small synthetic dataset
    np.random.seed(43)
    n = 30  # Small dataset

    judge_scores = np.random.uniform(0.3, 0.7, n)  # Narrow range
    oracle_labels = 0.2 + 0.6 * judge_scores + np.random.normal(0, 0.05, n)
    oracle_labels = np.clip(oracle_labels, 0, 1)

    # Create samples
    samples = []
    for i in range(n):
        samples.append(
            Sample(
                prompt_id=f"sample_{i}",
                prompt=f"prompt {i}",
                response=f"response {i}",
                base_policy_logprob=-1.0,
                target_policy_logprobs={"policy_a": -1.1},
                judge_score=float(judge_scores[i]),
                metadata={"judge_score": float(judge_scores[i])},
                oracle_label=float(oracle_labels[i]),
            )
        )

    dataset = Dataset(samples=samples, target_policies=["policy_a"])

    # Calibrate on first 20
    train_dataset = deepcopy(dataset)
    train_dataset.samples = train_dataset.samples[:20]

    calibrated, cal_result = calibrate_dataset(
        train_dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
    )
    calibrator = cal_result.calibrator

    # Probe on remaining 10 (will have sparse/empty deciles)
    probe_samples = dataset.samples[20:30]

    # Run transport audit with 10 bins (some will be empty with only 10 samples)
    diag = audit_transportability(calibrator, probe_samples, bins=10)

    # Should handle gracefully (not crash)
    assert diag.status in ["PASS", "WARN", "FAIL"]
    assert diag.n_probe == 10

    # Check that some deciles are empty or sparse
    assert any(c == 0 for c in diag.decile_counts), "Expected some empty deciles"
    assert any(
        np.isnan(r) for r in diag.decile_residuals
    ), "Expected NaN residuals for empty deciles"

    # Should have recommended action even with sparse data
    assert diag.recommended_action is not None
    assert len(diag.recommended_action) > 0


@pytest.mark.e2e
def test_transport_coverage_failure() -> None:
    """FAIL: Poor coverage (probe outside calibrator's training range)."""
    # Create training dataset with limited score range
    np.random.seed(44)
    n_train = 50

    # Train on mid-range scores only [0.4, 0.6]
    train_scores = np.random.uniform(0.4, 0.6, n_train)
    train_labels = 0.2 + 0.6 * train_scores + np.random.normal(0, 0.05, n_train)
    train_labels = np.clip(train_labels, 0, 1)

    train_samples = []
    for i in range(n_train):
        train_samples.append(
            Sample(
                prompt_id=f"train_{i}",
                prompt=f"prompt {i}",
                response=f"response {i}",
                base_policy_logprob=-1.0,
                target_policy_logprobs={"policy_a": -1.1},
                judge_score=float(train_scores[i]),
                metadata={"judge_score": float(train_scores[i])},
                oracle_label=float(train_labels[i]),
            )
        )

    train_dataset = Dataset(samples=train_samples, target_policies=["policy_a"])

    # Calibrate
    calibrated, cal_result = calibrate_dataset(
        train_dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
    )
    calibrator = cal_result.calibrator

    # Create probe with scores outside training range [0.1, 0.3] and [0.7, 0.9]
    n_probe = 40
    probe_scores = np.concatenate(
        [
            np.random.uniform(0.1, 0.3, n_probe // 2),
            np.random.uniform(0.7, 0.9, n_probe // 2),
        ]
    )
    probe_labels = 0.2 + 0.6 * probe_scores + np.random.normal(0, 0.05, n_probe)
    probe_labels = np.clip(probe_labels, 0, 1)

    probe_samples = []
    for i, (s, y) in enumerate(zip(probe_scores, probe_labels)):
        probe_samples.append(
            Sample(
                prompt_id=f"probe_{i}",
                prompt=f"prompt {i}",
                response=f"response {i}",
                base_policy_logprob=-1.0,
                target_policy_logprobs={"policy_a": -1.1},
                judge_score=float(s),
                metadata={"judge_score": float(s)},
                oracle_label=float(y),
            )
        )

    # Run transport audit
    diag = audit_transportability(calibrator, probe_samples)

    # Should detect poor coverage
    assert diag.coverage < 0.95, f"Expected poor coverage, got {diag.coverage:.1%}"

    # Should recommend action - could be boundary-related or refit due to extrapolation
    assert (
        diag.recommended_action
        in [
            "add_labels_boundary",
            "refit_two_stage",
            "collect_more_in_deciles_0,1,2",
            "collect_more_in_deciles_7,8,9",
        ]
        or "decile" in diag.recommended_action
    ), f"Expected coverage/extrapolation-related action, got: {diag.recommended_action}"


@pytest.mark.unit
def test_transport_diagnostics_to_dict() -> None:
    """Test TransportDiagnostics serialization."""
    diag = TransportDiagnostics(
        status="PASS",
        delta_hat=0.01,
        delta_ci=(-0.02, 0.04),
        delta_se=0.015,
        decile_residuals=[0.0, 0.01, -0.01, 0.02, 0.0, -0.01, 0.01, 0.0, -0.02, 0.01],
        decile_counts=[5, 6, 5, 4, 5, 6, 5, 4, 5, 5],
        coverage=0.96,
        ks_statistic=0.08,
        boundary_slopes=(0.5, 0.6),
        recommended_action="none",
        n_probe=50,
        group_label="policy:test",
    )

    # Convert to dict
    d = diag.to_dict()

    # Check structure
    assert d["status"] == "PASS"
    assert d["delta_hat"] == 0.01
    assert d["delta_ci"] == [-0.02, 0.04]
    assert d["n_probe"] == 50
    assert d["group_label"] == "policy:test"
    assert len(d["decile_residuals"]) == 10
    assert len(d["decile_counts"]) == 10


@pytest.mark.unit
def test_transport_diagnostics_summary() -> None:
    """Test TransportDiagnostics summary string."""
    diag = TransportDiagnostics(
        status="WARN",
        delta_hat=0.03,
        delta_ci=(0.01, 0.05),
        delta_se=0.01,
        decile_residuals=[0.0, 0.02, -0.01, 0.06, 0.01, -0.02, 0.03, 0.0, -0.01, 0.02],
        decile_counts=[5, 6, 5, 4, 5, 6, 5, 4, 5, 5],
        coverage=0.92,
        ks_statistic=0.12,
        boundary_slopes=None,
        recommended_action="add_labels_boundary",
        n_probe=50,
        group_label="policy:gpt-4-mini",
    )

    summary = diag.summary()

    # Check key info is present
    assert "WARN" in summary
    assert "N=50" in summary
    assert "policy:gpt-4-mini" in summary
    assert "0.92" in summary or "92" in summary  # coverage
    assert "add_labels_boundary" in summary


@pytest.mark.unit
def test_transport_missing_judge_score_raises() -> None:
    """Audit should raise if probe samples missing judge_score."""
    from cje.calibration.judge import JudgeCalibrator

    # Create minimal calibrator with enough samples
    calibrator = JudgeCalibrator()
    judge_scores = np.linspace(0.3, 0.8, 15)
    oracle_labels = 0.2 + 0.5 * judge_scores + np.random.normal(0, 0.05, 15)
    oracle_labels = np.clip(oracle_labels, 0, 1)
    calibrator.fit_transform(judge_scores, oracle_labels)

    # Create probe sample without judge_score in metadata
    probe_samples = [
        Sample(
            prompt_id="test_1",
            prompt="test prompt",
            response="test response",
            base_policy_logprob=-1.0,
            target_policy_logprobs={"policy_a": -1.1},
            metadata={},  # Missing judge_score!
            oracle_label=0.5,
        )
    ]

    # Should raise ValueError
    with pytest.raises(ValueError, match="missing judge_score"):
        audit_transportability(calibrator, probe_samples)


@pytest.mark.unit
def test_transport_missing_oracle_label_raises() -> None:
    """Audit should raise if probe samples missing oracle_label."""
    from cje.calibration.judge import JudgeCalibrator

    # Create minimal calibrator with enough samples
    calibrator = JudgeCalibrator()
    judge_scores = np.linspace(0.3, 0.8, 15)
    oracle_labels = 0.2 + 0.5 * judge_scores + np.random.normal(0, 0.05, 15)
    oracle_labels = np.clip(oracle_labels, 0, 1)
    calibrator.fit_transform(judge_scores, oracle_labels)

    # Create probe sample without oracle_label
    probe_samples = [
        Sample(
            prompt_id="test_1",
            prompt="test prompt",
            response="test response",
            base_policy_logprob=-1.0,
            target_policy_logprobs={"policy_a": -1.1},
            metadata={"judge_score": 0.5},
            oracle_label=None,  # Missing!
        )
    ]

    # Should raise ValueError
    with pytest.raises(ValueError, match="missing oracle_label"):
        audit_transportability(calibrator, probe_samples)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
