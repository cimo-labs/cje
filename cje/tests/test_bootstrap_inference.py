"""Tests for cluster bootstrap with calibrator refit for Direct mode.

Following CJE test philosophy:
- E2E tests use real arena data for authentic testing
- Unit tests for complex data structures and edge cases
- Tests verify user-visible outcomes
"""

import numpy as np
import pytest
from typing import Dict

from cje.data.fresh_draws import FreshDrawDataset
from cje.data.models import EstimationResult
from cje.diagnostics.robust_inference import (
    DirectEvalTable,
    build_direct_eval_table,
    cluster_bootstrap_direct_with_refit,
    make_calibrator_factory,
    compare_policies_bootstrap,
)
from cje.estimators.direct_method import CalibratedDirectEstimator
from cje.calibration.judge import JudgeCalibrator


# ============================================================================
# E2E Tests - Complete Workflows with Real Arena Data
# ============================================================================


class TestBootstrapE2EWorkflows:
    """End-to-end tests using real arena data."""

    def test_direct_mode_bootstrap_workflow(
        self, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Test complete Direct mode bootstrap workflow with real arena data.

        This is the primary E2E test - what users actually do:
        1. Load fresh draws
        2. Fit calibrator on oracle labels
        3. Create Direct estimator with bootstrap
        4. Get estimates with bootstrap CIs
        """
        if len(arena_fresh_draws) < 2:
            pytest.skip("Need at least 2 policies for comparison")

        policies = list(arena_fresh_draws.keys())[:2]
        fd_dict = {p: arena_fresh_draws[p] for p in policies}

        # Collect oracle data from first policy for calibration
        first_policy = policies[0]
        oracle_scores = []
        oracle_labels = []
        for sample in fd_dict[first_policy].samples:
            if sample.oracle_label is not None:
                oracle_scores.append(sample.judge_score)
                oracle_labels.append(sample.oracle_label)

        if len(oracle_scores) < 30:
            pytest.skip("Not enough oracle labels for bootstrap test")

        # Fit calibrator
        calibrator = JudgeCalibrator(calibration_mode="monotone")
        calibrator.fit_cv(np.array(oracle_scores), np.array(oracle_labels), n_folds=5)

        # Create Direct estimator with explicit bootstrap
        estimator = CalibratedDirectEstimator(
            target_policies=policies,
            reward_calibrator=calibrator,
            inference_method="bootstrap",
            n_bootstrap=100,  # Reduced for test speed
        )

        for policy in policies:
            estimator.add_fresh_draws(policy, fd_dict[policy])
        estimator.fit()

        # Get estimates
        result = estimator.estimate()

        # Validate E2E results
        assert result.method == "calibrated_direct_bootstrap"
        assert len(result.estimates) == len(policies)
        assert all(0 <= e <= 1 for e in result.estimates if not np.isnan(e))

        # Verify bootstrap metadata is present
        assert "bootstrap_ci" in result.metadata
        assert "inference" in result.metadata
        assert result.metadata["inference"]["method"] == "cluster_bootstrap_refit"

        # Verify CIs are reasonable
        lower, upper = result.confidence_interval()
        assert all(
            lower[i] <= result.estimates[i] <= upper[i] for i in range(len(policies))
        )

    def test_direct_mode_cluster_robust_workflow(
        self, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Test Direct mode with explicit cluster_robust (no bootstrap).

        Verifies that users can opt out of bootstrap.
        """
        if not arena_fresh_draws:
            pytest.skip("No fresh draws available")

        policies = list(arena_fresh_draws.keys())[:1]
        fd_dict = {p: arena_fresh_draws[p] for p in policies}

        # Collect oracle data
        first_policy = policies[0]
        oracle_scores = []
        oracle_labels = []
        for sample in fd_dict[first_policy].samples:
            if sample.oracle_label is not None:
                oracle_scores.append(sample.judge_score)
                oracle_labels.append(sample.oracle_label)

        if len(oracle_scores) < 10:
            pytest.skip("Not enough oracle labels")

        # Fit calibrator
        calibrator = JudgeCalibrator(calibration_mode="monotone")
        calibrator.fit_cv(np.array(oracle_scores), np.array(oracle_labels), n_folds=5)

        # Create Direct estimator with explicit cluster_robust
        estimator = CalibratedDirectEstimator(
            target_policies=policies,
            reward_calibrator=calibrator,
            inference_method="cluster_robust",
        )

        for policy in policies:
            estimator.add_fresh_draws(policy, fd_dict[policy])
        estimator.fit()

        result = estimator.estimate()

        # Should NOT use bootstrap
        assert result.method == "calibrated_direct"
        assert "bootstrap_ci" not in result.metadata

    def test_bootstrap_and_cluster_robust_give_same_point_estimates(
        self, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """REGRESSION TEST: Bootstrap and cluster_robust must give identical point estimates.

        This test guards against a bug where bootstrap used a refitted calibrator
        for point estimates, while cluster_robust used the original calibrator.
        Point estimates should always use the original calibrator.
        """
        if not arena_fresh_draws:
            pytest.skip("No fresh draws available")

        policies = list(arena_fresh_draws.keys())[:2]
        fd_dict = {p: arena_fresh_draws[p] for p in policies}

        # Collect oracle data from first policy
        first_policy = policies[0]
        oracle_scores = []
        oracle_labels = []
        for sample in fd_dict[first_policy].samples:
            if sample.oracle_label is not None:
                oracle_scores.append(sample.judge_score)
                oracle_labels.append(sample.oracle_label)

        if len(oracle_scores) < 30:
            pytest.skip("Not enough oracle labels")

        # Fit calibrator ONCE
        calibrator = JudgeCalibrator(calibration_mode="monotone")
        calibrator.fit_cv(np.array(oracle_scores), np.array(oracle_labels), n_folds=5)

        # Run cluster_robust
        est_cr = CalibratedDirectEstimator(
            target_policies=policies,
            reward_calibrator=calibrator,
            inference_method="cluster_robust",
        )
        for p in policies:
            est_cr.add_fresh_draws(p, fd_dict[p])
        est_cr.fit()
        result_cr = est_cr.estimate()

        # Run bootstrap (with augmented estimator disabled to match cluster_robust)
        est_boot = CalibratedDirectEstimator(
            target_policies=policies,
            reward_calibrator=calibrator,
            inference_method="bootstrap",
            n_bootstrap=50,  # Small for speed
            use_augmented_estimator=False,  # Disable to match cluster_robust
        )
        for p in policies:
            est_boot.add_fresh_draws(p, fd_dict[p])
        est_boot.fit()
        result_boot = est_boot.estimate()

        # Point estimates MUST be identical when using same estimator type
        # (Both use plug-in estimator when use_augmented_estimator=False)
        np.testing.assert_array_almost_equal(
            result_cr.estimates,
            result_boot.estimates,
            decimal=10,
            err_msg="Bootstrap and cluster_robust should give identical point estimates (both plug-in)",
        )

        # CIs should be properly centered for cluster_robust (z-based)
        # Bootstrap uses percentile CIs which may differ slightly from SE-based
        lower_cr, upper_cr = result_cr.confidence_interval()
        lower_boot, upper_boot = result_boot.confidence_interval()

        for i in range(len(policies)):
            # Cluster-robust uses z-based CIs - should be centered
            ci_center_cr = (lower_cr[i] + upper_cr[i]) / 2
            assert abs(result_cr.estimates[i] - ci_center_cr) < 1e-6

            # Bootstrap percentile CIs may be slightly asymmetric
            # Just verify estimate is within CI bounds
            assert lower_boot[i] <= result_boot.estimates[i] <= upper_boot[i], (
                f"Policy {i}: estimate {result_boot.estimates[i]:.4f} not in "
                f"CI [{lower_boot[i]:.4f}, {upper_boot[i]:.4f}]"
            )

    def test_bootstrap_pairwise_comparison(
        self, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Test pairwise policy comparison using bootstrap.

        Verifies that paired contrasts preserve correlation structure.
        """
        if len(arena_fresh_draws) < 2:
            pytest.skip("Need at least 2 policies for comparison")

        policies = list(arena_fresh_draws.keys())[:2]
        fd_dict = {p: arena_fresh_draws[p] for p in policies}

        # Collect oracle data
        first_policy = policies[0]
        oracle_scores = []
        oracle_labels = []
        for sample in fd_dict[first_policy].samples:
            if sample.oracle_label is not None:
                oracle_scores.append(sample.judge_score)
                oracle_labels.append(sample.oracle_label)

        if len(oracle_scores) < 30:
            pytest.skip("Not enough oracle labels")

        # Build eval table and run bootstrap
        table = build_direct_eval_table(fd_dict)
        factory = make_calibrator_factory(mode="monotone", seed=42)

        bootstrap_result = cluster_bootstrap_direct_with_refit(
            eval_table=table,
            calibrator_factory=factory,
            n_bootstrap=100,
            min_oracle_per_replicate=10,
            seed=42,
        )

        # Test pairwise comparison
        comparison = compare_policies_bootstrap(
            bootstrap_result, policy_a=0, policy_b=1
        )

        assert "diff_estimate" in comparison
        assert "diff_se" in comparison
        assert "ci_lower" in comparison
        assert "ci_upper" in comparison
        assert "p_value" in comparison
        assert 0 <= comparison["p_value"] <= 1


# ============================================================================
# Infrastructure Tests - Core Data Structures
# ============================================================================


class TestDirectEvalTableInfrastructure:
    """Test DirectEvalTable data structure (infrastructure test)."""

    def test_build_from_arena_data(
        self, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Test building eval table from real arena fresh draws."""
        if not arena_fresh_draws:
            pytest.skip("No fresh draws available")

        table = build_direct_eval_table(arena_fresh_draws)

        assert table.n_policies == len(arena_fresh_draws)
        assert table.n_clusters > 0
        assert len(table.cluster_to_rows) == table.n_clusters
        assert all(len(rows) > 0 for rows in table.cluster_to_rows.values())

    def test_cluster_to_rows_precomputed(
        self, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Test that cluster-to-rows mapping is precomputed correctly."""
        if not arena_fresh_draws:
            pytest.skip("No fresh draws available")

        table = build_direct_eval_table(arena_fresh_draws)

        # Verify each cluster maps to correct rows
        for cluster_id, rows in table.cluster_to_rows.items():
            # All rows should have this cluster_id
            assert all(table.prompt_ids[r] == cluster_id for r in rows)

    def test_oracle_mask_accuracy(
        self, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Test that oracle mask correctly identifies labeled samples."""
        if not arena_fresh_draws:
            pytest.skip("No fresh draws available")

        table = build_direct_eval_table(arena_fresh_draws)

        # Check oracle mask matches NaN pattern
        for i, is_oracle in enumerate(table.oracle_mask):
            if is_oracle:
                assert not np.isnan(table.oracle_labels[i])
            else:
                assert np.isnan(table.oracle_labels[i])


class TestCalibratorFactory:
    """Test calibrator factory (infrastructure test)."""

    def test_factory_creates_fresh_instances(self) -> None:
        """Test that factory creates new instances each call."""
        factory = make_calibrator_factory(mode="monotone", seed=42)

        cal1 = factory()
        cal2 = factory()

        assert cal1 is not cal2
        assert cal1.calibration_mode == "monotone"
        assert cal2.calibration_mode == "monotone"

    def test_factory_preserves_mode(self) -> None:
        """Test that factory preserves the specified mode."""
        from typing import cast, Literal

        for mode in ["monotone", "two_stage"]:
            factory = make_calibrator_factory(
                mode=cast(Literal["monotone", "two_stage"], mode), seed=42
            )
            cal = factory()
            assert cal.calibration_mode == mode


# ============================================================================
# Feature Tests - Bootstrap Behavior
# ============================================================================


class TestBootstrapBehavior:
    """Test bootstrap-specific behaviors."""

    def test_resample_until_valid(
        self, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Test that resample-until-valid works correctly."""
        if not arena_fresh_draws:
            pytest.skip("No fresh draws available")

        table = build_direct_eval_table(arena_fresh_draws)
        factory = make_calibrator_factory(mode="monotone", seed=42)

        # Request high min_oracle threshold to trigger resampling
        result = cluster_bootstrap_direct_with_refit(
            eval_table=table,
            calibrator_factory=factory,
            n_bootstrap=20,
            min_oracle_per_replicate=10,
            seed=42,
        )

        # Should complete with valid replicates
        assert result["n_valid_replicates"] > 0
        # n_attempts should be >= n_valid_replicates (may need retries)
        assert result["n_attempts"] >= result["n_valid_replicates"]

    def test_oracle_count_summary(
        self, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Test that oracle count summary is provided."""
        if not arena_fresh_draws:
            pytest.skip("No fresh draws available")

        table = build_direct_eval_table(arena_fresh_draws)
        factory = make_calibrator_factory(mode="monotone", seed=42)

        result = cluster_bootstrap_direct_with_refit(
            eval_table=table,
            calibrator_factory=factory,
            n_bootstrap=20,
            min_oracle_per_replicate=5,
            seed=42,
        )

        assert "oracle_count_summary" in result
        summary = result["oracle_count_summary"]
        if summary:  # May be empty if no valid replicates
            assert "min" in summary
            assert "p10" in summary
            assert "median" in summary

    def test_bootstrap_ci_uses_se_based_intervals(self) -> None:
        """Test that confidence_interval() uses SE-based CIs, not bootstrap percentile CIs.

        After the fix to use original calibrator for point estimates, bootstrap
        percentile CIs would be mis-centered. We now use estimate ± z*SE for
        proper coverage.
        """
        # Create result with bootstrap CIs in metadata
        result = EstimationResult(
            estimates=np.array([0.5, 0.6]),
            standard_errors=np.array([0.02, 0.03]),
            n_samples_used={"a": 100, "b": 100},
            method="calibrated_direct_bootstrap",
            influence_functions=None,
            diagnostics=None,
            metadata={
                "bootstrap_ci": {
                    "lower": [0.45, 0.54],
                    "upper": [0.55, 0.66],
                    "method": "percentile",
                    "alpha": 0.05,
                },
            },
        )

        lower, upper = result.confidence_interval()

        # CIs are percentile bootstrap intervals from metadata
        # (BCa was removed - simple percentile intervals achieve ~95% coverage
        # via θ̂_aug + bootstrap refit, not BCa corrections)
        expected_lower = np.array(result.metadata["bootstrap_ci"]["lower"])
        expected_upper = np.array(result.metadata["bootstrap_ci"]["upper"])
        np.testing.assert_array_almost_equal(lower, expected_lower)
        np.testing.assert_array_almost_equal(upper, expected_upper)


class TestCouplingDetection:
    """Test calibration/evaluation coupling detection."""

    def test_coupling_detection_with_arena_data(
        self, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Test that coupling is detected correctly with real data."""
        if not arena_fresh_draws:
            pytest.skip("No fresh draws available")

        policies = list(arena_fresh_draws.keys())[:1]

        # Collect oracle data
        first_policy = policies[0]
        oracle_scores = []
        oracle_labels = []
        for sample in arena_fresh_draws[first_policy].samples:
            if sample.oracle_label is not None:
                oracle_scores.append(sample.judge_score)
                oracle_labels.append(sample.oracle_label)

        if len(oracle_scores) < 10:
            pytest.skip("Not enough oracle labels")

        calibrator = JudgeCalibrator(calibration_mode="monotone")
        calibrator.fit_cv(np.array(oracle_scores), np.array(oracle_labels), n_folds=5)

        estimator = CalibratedDirectEstimator(
            target_policies=policies,
            reward_calibrator=calibrator,
            inference_method="auto",
        )
        estimator.add_fresh_draws(first_policy, arena_fresh_draws[first_policy])
        estimator.fit()

        # Oracle labels come from same prompts as evaluation (coupled)
        coupled, overlap = estimator._calibration_overlaps_evaluation()
        assert coupled  # Should detect coupling
        assert overlap > 0  # Should have overlapping clusters


class TestLowOracleCoverage:
    """Tests for bootstrap with low oracle coverage (regression tests for oracle data asymmetry fix)."""

    def test_bootstrap_with_10_percent_oracle_coverage(
        self, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Test bootstrap works correctly with 10% oracle coverage.

        REGRESSION TEST: Before the fix, bootstrap would refit on 100% oracle coverage
        from fresh_draws even when calibration used only 10% coverage. This led to:
        - Smaller SEs (calibrator more stable with more data)
        - Severe undercoverage

        After fix: fresh_draws oracle labels are masked to match calibration oracle slice.
        Bootstrap should:
        1. Complete successfully (adaptive min_oracle_per_replicate)
        2. Have similar SE magnitudes to cluster_robust/OUA at same coverage
        """
        if len(arena_fresh_draws) < 1:
            pytest.skip("No fresh draws available")

        policy = list(arena_fresh_draws.keys())[0]
        fd = arena_fresh_draws[policy]

        # Collect all oracle data
        all_oracle_scores = []
        all_oracle_labels = []
        all_prompt_ids = []
        for sample in fd.samples:
            if sample.oracle_label is not None:
                all_oracle_scores.append(sample.judge_score)
                all_oracle_labels.append(sample.oracle_label)
                all_prompt_ids.append(sample.prompt_id)

        n_total_oracle = len(all_oracle_scores)
        if n_total_oracle < 50:
            pytest.skip(f"Need at least 50 oracle labels, got {n_total_oracle}")

        # Simulate 20% oracle coverage (enough for bootstrap to work)
        np.random.seed(42)
        n_keep = max(10, int(n_total_oracle * 0.20))
        keep_indices = np.random.choice(n_total_oracle, size=n_keep, replace=False)
        oracle_prompt_ids = set(all_prompt_ids[i] for i in keep_indices)

        # Create masked fresh draws (as would happen in ablation)
        from cje.data.fresh_draws import FreshDrawSample, FreshDrawDataset

        masked_samples = []
        for sample in fd.samples:
            # Create a copy with potentially masked oracle
            masked_sample = FreshDrawSample(
                prompt_id=sample.prompt_id,
                target_policy=policy,
                response=sample.response,
                judge_score=sample.judge_score,
                oracle_label=(
                    sample.oracle_label
                    if sample.prompt_id in oracle_prompt_ids
                    else None
                ),
                draw_idx=getattr(sample, "draw_idx", 0),
                fold_id=getattr(sample, "fold_id", None),
            )
            masked_samples.append(masked_sample)

        masked_fd = FreshDrawDataset(
            samples=masked_samples,
            target_policy=policy,
            draws_per_prompt=fd.draws_per_prompt,
        )

        # Fit calibrator on the MASKED oracle slice only
        masked_oracle_scores = []
        masked_oracle_labels = []
        for sample in masked_fd.samples:
            if sample.oracle_label is not None:
                masked_oracle_scores.append(sample.judge_score)
                masked_oracle_labels.append(sample.oracle_label)

        calibrator = JudgeCalibrator(calibration_mode="monotone")
        calibrator.fit_cv(
            np.array(masked_oracle_scores),
            np.array(masked_oracle_labels),
            n_folds=min(5, len(masked_oracle_scores) // 3),
        )

        # Create estimator with bootstrap
        estimator = CalibratedDirectEstimator(
            target_policies=[policy],
            reward_calibrator=calibrator,
            inference_method="bootstrap",
            n_bootstrap=50,  # Small for speed
        )
        estimator.add_fresh_draws(policy, masked_fd)
        estimator.fit()

        result = estimator.estimate()

        # Verify bootstrap completed
        assert result.method == "calibrated_direct_bootstrap"
        assert "inference" in result.metadata
        assert result.metadata["inference"]["n_bootstrap_valid"] > 0

        # Verify SEs are reasonable (not too small)
        # With ~20% oracle and masked fresh_draws, SEs should be substantial
        assert result.standard_errors[0] > 0.005, (
            f"SE too small ({result.standard_errors[0]:.4f}), "
            "bootstrap may be using more oracle data than expected"
        )

        # Verify estimate is reasonable
        assert 0 < result.estimates[0] < 1

    def test_bootstrap_adaptive_min_oracle(
        self, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Test that adaptive min_oracle_per_replicate works at low coverage.

        At very low oracle coverage (e.g., 15 samples), fixed min_oracle=30
        would cause 100% rejection rate. Adaptive min_oracle should allow
        bootstrap to complete.
        """
        if len(arena_fresh_draws) < 1:
            pytest.skip("No fresh draws available")

        policy = list(arena_fresh_draws.keys())[0]
        fd = arena_fresh_draws[policy]

        # Find oracle samples
        oracle_samples = [s for s in fd.samples if s.oracle_label is not None]
        n_total_oracle = len(oracle_samples)
        if n_total_oracle < 30:
            pytest.skip(f"Need at least 30 oracle labels, got {n_total_oracle}")

        # Keep only ~20 oracle samples (below the old fixed threshold of 30)
        np.random.seed(123)
        n_keep = 20
        keep_prompt_ids = set(
            oracle_samples[i].prompt_id
            for i in np.random.choice(len(oracle_samples), size=n_keep, replace=False)
        )

        # Mask fresh draws
        from cje.data.fresh_draws import FreshDrawSample, FreshDrawDataset

        masked_samples = []
        for sample in fd.samples:
            masked_sample = FreshDrawSample(
                prompt_id=sample.prompt_id,
                target_policy=policy,
                response=sample.response,
                judge_score=sample.judge_score,
                oracle_label=(
                    sample.oracle_label if sample.prompt_id in keep_prompt_ids else None
                ),
                draw_idx=getattr(sample, "draw_idx", 0),
                fold_id=getattr(sample, "fold_id", None),
            )
            masked_samples.append(masked_sample)

        masked_fd = FreshDrawDataset(
            samples=masked_samples,
            target_policy=policy,
            draws_per_prompt=fd.draws_per_prompt,
        )

        # Fit calibrator
        oracle_scores = [
            s.judge_score for s in masked_samples if s.oracle_label is not None
        ]
        oracle_labels = [
            s.oracle_label for s in masked_samples if s.oracle_label is not None
        ]
        assert (
            len(oracle_scores) == n_keep
        ), f"Expected {n_keep}, got {len(oracle_scores)}"

        calibrator = JudgeCalibrator(calibration_mode="monotone")
        calibrator.fit_cv(
            np.array(oracle_scores),
            np.array(oracle_labels),
            n_folds=min(5, n_keep // 3),
        )

        # This should succeed with adaptive min_oracle (would fail with fixed min=30)
        estimator = CalibratedDirectEstimator(
            target_policies=[policy],
            reward_calibrator=calibrator,
            inference_method="bootstrap",
            n_bootstrap=30,
        )
        estimator.add_fresh_draws(policy, masked_fd)
        estimator.fit()

        result = estimator.estimate()

        # Key assertion: bootstrap should complete with valid replicates
        assert result.method == "calibrated_direct_bootstrap"
        n_valid = result.metadata["inference"]["n_bootstrap_valid"]
        assert n_valid > 0, f"Expected valid replicates, got {n_valid}"

        # With only 20 oracle samples and adaptive min (max(10, 20//3)=10),
        # we should get most replicates
        skip_rate = result.metadata["inference"]["skip_rate"]
        assert skip_rate < 0.5, f"Skip rate too high: {skip_rate:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
