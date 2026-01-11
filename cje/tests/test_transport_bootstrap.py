"""Test transport-aware bootstrap with calibration_policy_idx.

This test verifies that the bootstrap correctly separates:
- Calibration oracle: used for fitting the calibrator (single policy)
- Residual oracle: used for θ̂_aug residual corrections (all policies)

When calibration doesn't transport to target policies, the residual correction
should capture the transport bias and correct the estimates.
"""

import numpy as np
import pytest
from typing import Dict, List

from cje.diagnostics.robust_inference import (
    cluster_bootstrap_direct_with_refit,
    build_direct_eval_table,
    make_calibrator_factory,
    DirectEvalTable,
)
from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample


def create_transport_test_data(
    n_samples: int = 200,
    transport_bias: float = 0.15,
    oracle_coverage: float = 0.5,
    seed: int = 42,
) -> Dict[str, FreshDrawDataset]:
    """Create test data with known transport bias.

    Base policy: judge_score correlates with oracle_label (calibration works)
    Target policy: same judge_score but oracle is shifted by transport_bias

    Args:
        n_samples: Number of samples per policy
        transport_bias: Offset between base and target oracle labels
        oracle_coverage: Fraction of samples with oracle labels
        seed: Random seed

    Returns:
        Dict mapping policy names to FreshDrawDataset
    """
    np.random.seed(seed)

    # Generate base policy data
    # Judge scores and oracle labels are correlated (calibration should work)
    base_judge = np.clip(0.5 + 0.2 * np.random.randn(n_samples), 0.1, 0.9)
    base_oracle = base_judge + 0.1 * np.random.randn(n_samples)
    base_oracle = np.clip(base_oracle, 0.0, 1.0)

    # Target policy: same judge scores, but oracle is shifted
    target_judge = base_judge.copy()
    target_oracle = base_oracle + transport_bias
    target_oracle = np.clip(target_oracle, 0.0, 1.0)

    # Randomly select which samples have oracle labels
    n_oracle = int(n_samples * oracle_coverage)
    oracle_indices = np.random.choice(n_samples, n_oracle, replace=False)
    oracle_mask = np.zeros(n_samples, dtype=bool)
    oracle_mask[oracle_indices] = True

    # Create base policy samples
    base_samples = []
    for i in range(n_samples):
        sample = FreshDrawSample(
            prompt_id=f"prompt_{i}",
            target_policy="base",
            judge_score=float(base_judge[i]),
            oracle_label=float(base_oracle[i]) if oracle_mask[i] else None,
            draw_idx=0,
            response="",
            fold_id=None,
        )
        base_samples.append(sample)

    # Create target policy samples
    target_samples = []
    for i in range(n_samples):
        sample = FreshDrawSample(
            prompt_id=f"prompt_{i}",
            target_policy="target",
            judge_score=float(target_judge[i]),
            oracle_label=float(target_oracle[i]) if oracle_mask[i] else None,
            draw_idx=0,
            response="",
            fold_id=None,
        )
        target_samples.append(sample)

    return {
        "base": FreshDrawDataset(
            samples=base_samples,
            target_policy="base",
            draws_per_prompt=1,
        ),
        "target": FreshDrawDataset(
            samples=target_samples,
            target_policy="target",
            draws_per_prompt=1,
        ),
    }


class TestCalibrationPolicyIdx:
    """Test calibration_policy_idx parameter in bootstrap."""

    def test_parameter_accepted(self) -> None:
        """Test that calibration_policy_idx parameter is accepted."""
        fresh_draws = create_transport_test_data(n_samples=100)
        eval_table = build_direct_eval_table(fresh_draws)
        calibrator_factory = make_calibrator_factory("monotone")

        # Should not raise
        result = cluster_bootstrap_direct_with_refit(
            eval_table=eval_table,
            calibrator_factory=calibrator_factory,
            n_bootstrap=10,  # Small for fast test
            min_oracle_per_replicate=5,
            calibration_policy_idx=0,  # Base policy
        )

        assert "calibration_policy_idx" in result
        assert result["calibration_policy_idx"] == 0

    def test_none_calibration_policy_idx(self) -> None:
        """Test default behavior when calibration_policy_idx is None."""
        fresh_draws = create_transport_test_data(n_samples=100)
        eval_table = build_direct_eval_table(fresh_draws)
        calibrator_factory = make_calibrator_factory("monotone")

        result = cluster_bootstrap_direct_with_refit(
            eval_table=eval_table,
            calibrator_factory=calibrator_factory,
            n_bootstrap=10,
            min_oracle_per_replicate=5,
            calibration_policy_idx=None,  # Default
        )

        assert result["calibration_policy_idx"] is None

    def test_residual_corrections_with_transport_bias(self) -> None:
        """Test that residual corrections capture transport bias.

        When calibration_policy_idx is set:
        - Base policy should have small residual correction (calibrator fits well)
        - Target policy should have large residual correction (transport bias)
        """
        transport_bias = 0.15
        fresh_draws = create_transport_test_data(
            n_samples=200,
            transport_bias=transport_bias,
            oracle_coverage=0.5,
        )
        eval_table = build_direct_eval_table(fresh_draws)
        calibrator_factory = make_calibrator_factory("monotone")

        # Run with calibration_policy_idx=0 (base only)
        result = cluster_bootstrap_direct_with_refit(
            eval_table=eval_table,
            calibrator_factory=calibrator_factory,
            n_bootstrap=50,
            min_oracle_per_replicate=10,
            calibration_policy_idx=0,  # Base policy
            use_augmented_estimator=True,
        )

        aug_diag = result.get("augmentation_diagnostics", {})
        residual_corrections = aug_diag.get("residual_corrections", [])

        assert len(residual_corrections) == 2, "Should have 2 policies"

        base_idx = result["policy_names"].index("base")
        target_idx = result["policy_names"].index("target")

        base_residual = residual_corrections[base_idx]
        target_residual = residual_corrections[target_idx]

        # Base policy: calibrator fits well, small residual
        assert (
            abs(base_residual) < 0.05
        ), f"Base residual {base_residual:.3f} should be small"

        # Target policy: transport bias should appear in residual
        # Residual = mean(Y - f(S)), where f was fit on base
        # Since target Y is shifted by transport_bias, residual should be ~transport_bias
        assert (
            abs(target_residual - transport_bias) < 0.05
        ), f"Target residual {target_residual:.3f} should be ~{transport_bias:.3f}"

    def test_estimates_corrected_by_residuals(self) -> None:
        """Test that estimates are corrected for transport bias."""
        transport_bias = 0.15
        fresh_draws = create_transport_test_data(
            n_samples=200,
            transport_bias=transport_bias,
            oracle_coverage=0.5,
            seed=123,
        )
        eval_table = build_direct_eval_table(fresh_draws)
        calibrator_factory = make_calibrator_factory("monotone")

        # Compute true oracle means
        base_oracle_mean = np.mean(
            [
                s.oracle_label
                for s in fresh_draws["base"].samples
                if s.oracle_label is not None
            ]
        )
        target_oracle_mean = np.mean(
            [
                s.oracle_label
                for s in fresh_draws["target"].samples
                if s.oracle_label is not None
            ]
        )

        # Run bootstrap with calibration_policy_idx
        result = cluster_bootstrap_direct_with_refit(
            eval_table=eval_table,
            calibrator_factory=calibrator_factory,
            n_bootstrap=100,
            min_oracle_per_replicate=10,
            calibration_policy_idx=0,  # Base policy
            use_augmented_estimator=True,
        )

        estimates = result["estimates"]
        base_idx = result["policy_names"].index("base")
        target_idx = result["policy_names"].index("target")

        base_estimate = estimates[base_idx]
        target_estimate = estimates[target_idx]

        # Both estimates should be close to their true oracle means
        assert (
            abs(base_estimate - base_oracle_mean) < 0.05
        ), f"Base estimate {base_estimate:.3f} should be ~{base_oracle_mean:.3f}"
        assert (
            abs(target_estimate - target_oracle_mean) < 0.05
        ), f"Target estimate {target_estimate:.3f} should be ~{target_oracle_mean:.3f}"

    def test_comparison_with_vs_without_calibration_policy_idx(self) -> None:
        """Compare results with and without calibration_policy_idx.

        Without calibration_policy_idx: calibrator fits on all oracle
        With calibration_policy_idx: calibrator fits only on base oracle

        The target policy estimate should be different (and more accurate
        with calibration_policy_idx when there's transport bias).
        """
        transport_bias = 0.20
        fresh_draws = create_transport_test_data(
            n_samples=200,
            transport_bias=transport_bias,
            oracle_coverage=0.5,
            seed=456,
        )
        eval_table = build_direct_eval_table(fresh_draws)
        calibrator_factory = make_calibrator_factory("monotone")

        # True target oracle mean
        target_oracle_mean = np.mean(
            [
                s.oracle_label
                for s in fresh_draws["target"].samples
                if s.oracle_label is not None
            ]
        )

        # Run WITHOUT calibration_policy_idx (fits on all oracle)
        result_all = cluster_bootstrap_direct_with_refit(
            eval_table=eval_table,
            calibrator_factory=calibrator_factory,
            n_bootstrap=50,
            min_oracle_per_replicate=10,
            calibration_policy_idx=None,
            use_augmented_estimator=True,
            seed=42,
        )

        # Run WITH calibration_policy_idx=0 (fits on base only)
        result_base = cluster_bootstrap_direct_with_refit(
            eval_table=eval_table,
            calibrator_factory=calibrator_factory,
            n_bootstrap=50,
            min_oracle_per_replicate=10,
            calibration_policy_idx=0,
            use_augmented_estimator=True,
            seed=42,
        )

        target_idx = result_all["policy_names"].index("target")

        target_estimate_all = result_all["estimates"][target_idx]
        target_estimate_base = result_base["estimates"][target_idx]

        # The "base only" version should be closer to true target oracle
        error_all = abs(target_estimate_all - target_oracle_mean)
        error_base = abs(target_estimate_base - target_oracle_mean)

        print(f"Target oracle mean: {target_oracle_mean:.3f}")
        print(
            f"Estimate (all oracle): {target_estimate_all:.3f}, error: {error_all:.3f}"
        )
        print(
            f"Estimate (base only): {target_estimate_base:.3f}, error: {error_base:.3f}"
        )

        # With transport bias, base-only calibration + residual correction
        # should be more accurate
        assert error_base < error_all + 0.02, (
            f"Base-only calibration should be at least as good: "
            f"error_base={error_base:.3f} vs error_all={error_all:.3f}"
        )


class TestBootstrapWithCalibrationPolicyIdx:
    """Test bootstrap loop behavior with calibration_policy_idx."""

    def test_bootstrap_uses_correct_oracle_count(self) -> None:
        """Test that bootstrap checks calibration oracle count, not total."""
        # Create data where base has few oracle labels
        np.random.seed(789)
        n_samples = 100

        # Base: only 20% oracle coverage
        base_samples = []
        for i in range(n_samples):
            has_oracle = i < 20  # First 20 have oracle
            base_samples.append(
                FreshDrawSample(
                    prompt_id=f"p{i}",
                    target_policy="base",
                    judge_score=np.random.uniform(0.3, 0.7),
                    oracle_label=np.random.uniform(0.3, 0.7) if has_oracle else None,
                    draw_idx=0,
                    response="",
                    fold_id=None,
                )
            )

        # Target: 80% oracle coverage
        target_samples = []
        for i in range(n_samples):
            has_oracle = i >= 20  # Last 80 have oracle
            target_samples.append(
                FreshDrawSample(
                    prompt_id=f"p{i}",
                    target_policy="target",
                    judge_score=np.random.uniform(0.3, 0.7),
                    oracle_label=np.random.uniform(0.3, 0.7) if has_oracle else None,
                    draw_idx=0,
                    response="",
                    fold_id=None,
                )
            )

        fresh_draws = {
            "base": FreshDrawDataset(
                samples=base_samples, target_policy="base", draws_per_prompt=1
            ),
            "target": FreshDrawDataset(
                samples=target_samples, target_policy="target", draws_per_prompt=1
            ),
        }

        eval_table = build_direct_eval_table(fresh_draws)
        calibrator_factory = make_calibrator_factory("monotone")

        # With calibration_policy_idx=0, only base oracle (20 samples) matters
        # min_oracle_per_replicate=15 should work
        result = cluster_bootstrap_direct_with_refit(
            eval_table=eval_table,
            calibrator_factory=calibrator_factory,
            n_bootstrap=20,
            min_oracle_per_replicate=15,  # Less than base's 20
            calibration_policy_idx=0,
        )

        assert result["n_valid_replicates"] > 0, "Should have valid replicates"

    def test_oof_predictions_for_non_calibration_oracle(self) -> None:
        """Test that non-calibration oracle gets full-model predictions."""
        fresh_draws = create_transport_test_data(n_samples=100, oracle_coverage=1.0)
        eval_table = build_direct_eval_table(fresh_draws)
        calibrator_factory = make_calibrator_factory("monotone")

        result = cluster_bootstrap_direct_with_refit(
            eval_table=eval_table,
            calibrator_factory=calibrator_factory,
            n_bootstrap=10,
            min_oracle_per_replicate=5,
            calibration_policy_idx=0,
            use_augmented_estimator=True,
        )

        # Both policies should have estimates (target uses full-model for OOF)
        assert not np.isnan(result["estimates"][0]), "Base estimate should exist"
        assert not np.isnan(result["estimates"][1]), "Target estimate should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
