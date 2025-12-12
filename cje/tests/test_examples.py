"""End-to-end tests for example notebooks and tutorials.

These tests validate the complete walkthrough from the notebook examples,
ensuring the documented usage patterns work correctly.
"""

import pytest
import numpy as np
from typing import Any

from cje import analyze_dataset

# Mark all tests in this file as E2E tests
pytestmark = [pytest.mark.e2e, pytest.mark.uses_arena_sample]


class TestNotebookWalkthrough:
    """Test the complete notebook walkthrough: Direct → IPS → DR modes."""

    def test_direct_mode_first(self, arena_fresh_draws: Any) -> None:
        """Test Direct mode (Step 3 in notebook) - simplest mode with fresh draws only."""
        # Direct mode: Fresh draws only (no logged data needed)

        # Get policies from fresh draws
        policies = list(arena_fresh_draws.keys())

        # Simple direct estimation: average judge scores per policy
        estimates = []
        std_errors = []

        for policy in policies:
            dataset = arena_fresh_draws[policy]
            judge_scores = [
                s.judge_score for s in dataset.samples if s.judge_score is not None
            ]

            if judge_scores:
                est = np.mean(judge_scores)
                se = np.std(judge_scores) / np.sqrt(len(judge_scores))
                estimates.append(est)
                std_errors.append(se)
            else:
                estimates.append(np.nan)
                std_errors.append(np.nan)

        # Validate Direct mode results
        valid_estimates = [e for e in estimates if not np.isnan(e)]
        assert len(valid_estimates) >= 2, "Should have estimates for multiple policies"
        assert all(0 <= e <= 1 for e in valid_estimates), "Estimates should be in [0,1]"
        assert all(
            se > 0 for se, e in zip(std_errors, estimates) if not np.isnan(e)
        ), "Should have positive SEs"

        print(f"✓ Direct mode: {len(valid_estimates)} policies estimated")

    def test_ips_mode_second(self, arena_sample: Any) -> None:
        """Test IPS mode (Step 4 in notebook) - logged data with importance sampling."""
        # IPS mode: Logged data only (uses lower-level API like other E2E tests)
        from cje.calibration import calibrate_dataset
        from cje.data.precomputed_sampler import PrecomputedSampler
        from cje.estimators import CalibratedIPS

        # Calibrate dataset
        calibrated, cal_result = calibrate_dataset(
            arena_sample,
            judge_field="judge_score",
            oracle_field="oracle_label",
        )

        # Create sampler
        sampler = PrecomputedSampler(calibrated)

        # Create IPS estimator
        estimator = CalibratedIPS(sampler)
        results_ips = estimator.fit_and_estimate()

        # Validate IPS results
        policies = sampler.target_policies
        assert len(results_ips.estimates) == len(policies)
        assert all(0 <= e <= 1 for e in results_ips.estimates), "Estimates in [0,1]"
        assert all(se > 0 for se in results_ips.standard_errors), "Positive SEs"
        assert results_ips.method == "calibrated_ips"

        # Check ESS diagnostics
        assert results_ips.diagnostics is not None

        print(f"✓ IPS mode: {len(policies)} policies estimated")

    def test_dr_mode_third(self, arena_sample: Any, arena_fresh_draws: Any) -> None:
        """Test DR mode (Step 5 in notebook) - most accurate with both data sources."""
        # DR mode: Logged data + fresh draws
        from cje.calibration import calibrate_dataset
        from cje.data.precomputed_sampler import PrecomputedSampler
        from cje.estimators import DRCPOEstimator

        # Calibrate dataset
        calibrated, cal_result = calibrate_dataset(
            arena_sample,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,
            n_folds=5,
        )

        # Create sampler
        sampler = PrecomputedSampler(calibrated)

        # Create DR estimator
        dr_estimator = DRCPOEstimator(
            sampler,
            reward_calibrator=cal_result.calibrator,
            n_folds=5,
        )

        # Add fresh draws
        for policy, fresh_dataset in arena_fresh_draws.items():
            if policy in sampler.target_policies:
                dr_estimator.add_fresh_draws(policy, fresh_dataset)

        # Run estimation
        results_dr = dr_estimator.fit_and_estimate()

        # Validate DR results
        policies = sampler.target_policies
        assert len(results_dr.estimates) == len(policies)
        assert all(0 <= e <= 1 for e in results_dr.estimates), "Estimates in [0,1]"
        assert all(se > 0 for se in results_dr.standard_errors), "Positive SEs"
        assert results_dr.method == "dr_cpo"

        # Check DR diagnostics
        assert results_dr.diagnostics is not None

        print(f"✓ DR mode: {len(policies)} policies with doubly robust estimation")

    def test_full_notebook_flow(
        self, arena_sample: Any, arena_fresh_draws: Any
    ) -> None:
        """Test complete notebook flow: Direct → IPS → DR with comparisons."""
        # 1. Direct mode
        policies = list(arena_fresh_draws.keys())

        direct_estimates = []
        for policy in policies:
            dataset = arena_fresh_draws[policy]
            judge_scores = [
                s.judge_score for s in dataset.samples if s.judge_score is not None
            ]
            if judge_scores:
                direct_estimates.append(np.mean(judge_scores))
            else:
                direct_estimates.append(np.nan)

        # 2. IPS mode
        from cje.calibration import calibrate_dataset
        from cje.data.precomputed_sampler import PrecomputedSampler
        from cje.estimators import CalibratedIPS

        calibrated_ips, cal_result_ips = calibrate_dataset(
            arena_sample,
            judge_field="judge_score",
            oracle_field="oracle_label",
        )
        sampler_ips = PrecomputedSampler(calibrated_ips)
        estimator_ips = CalibratedIPS(sampler_ips)
        results_ips = estimator_ips.fit_and_estimate()

        # 3. DR mode
        from cje.calibration import calibrate_dataset
        from cje.data.precomputed_sampler import PrecomputedSampler
        from cje.estimators import DRCPOEstimator

        calibrated, cal_result = calibrate_dataset(
            arena_sample,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,
            n_folds=5,
        )

        sampler = PrecomputedSampler(calibrated)
        dr_estimator = DRCPOEstimator(
            sampler, reward_calibrator=cal_result.calibrator, n_folds=5
        )

        for policy, fresh_dataset in arena_fresh_draws.items():
            if policy in sampler.target_policies:
                dr_estimator.add_fresh_draws(policy, fresh_dataset)

        results_dr = dr_estimator.fit_and_estimate()

        # 4. Validate consistency
        valid_direct = [e for e in direct_estimates if not np.isnan(e)]
        assert len(valid_direct) >= 2, "Direct mode should estimate multiple policies"
        assert (
            len(results_ips.estimates) >= 2
        ), "IPS mode should estimate multiple policies"
        assert (
            len(results_dr.estimates) >= 2
        ), "DR mode should estimate multiple policies"

        # Estimates should be broadly consistent (within reasonable range)
        for i, policy in enumerate(policies):
            estimates_for_policy = []

            if not np.isnan(direct_estimates[i]):
                estimates_for_policy.append(direct_estimates[i])

            if policy in results_ips.metadata["target_policies"]:
                ips_idx = results_ips.metadata["target_policies"].index(policy)
                estimates_for_policy.append(results_ips.estimates[ips_idx])

            if policy in sampler.target_policies:
                dr_idx = sampler.target_policies.index(policy)
                estimates_for_policy.append(results_dr.estimates[dr_idx])

            if len(estimates_for_policy) >= 2:
                # Check range is reasonable (not wildly different)
                # Note: Policies with poor overlap (like "unhelpful") may legitimately
                # vary more between methods
                estimate_range = max(estimates_for_policy) - min(estimates_for_policy)
                assert (
                    estimate_range < 0.6
                ), f"{policy}: estimates vary by {estimate_range:.3f}"

        print("✓ Full notebook flow complete - all modes give consistent estimates")


class TestAutoModeSelection:
    """Test automatic mode selection (internal)."""

    def test_detect_ips_mode(self, arena_sample: Any) -> None:
        """detect_analysis_mode should return IPS when no fresh draws available."""
        from cje.interface.mode_detection import detect_analysis_mode

        mode, explanation, coverage = detect_analysis_mode(
            dataset=arena_sample,
            fresh_draws_dir=None,
        )

        assert mode == "ips"
        assert coverage > 0  # Arena sample has logprobs
        assert "IPS mode" in explanation

    def test_detect_dr_mode(self, arena_sample: Any, tmp_path: Any) -> None:
        """detect_analysis_mode should return DR when fresh draws directory exists."""
        from cje.interface.mode_detection import detect_analysis_mode

        # Create a temporary fresh draws directory
        fresh_dir = tmp_path / "fresh_draws"
        fresh_dir.mkdir()

        mode, explanation, coverage = detect_analysis_mode(
            dataset=arena_sample,
            fresh_draws_dir=str(fresh_dir),
        )

        assert mode == "dr"
        assert coverage > 0  # Arena sample has logprobs
        assert "DR mode" in explanation


@pytest.mark.slow
class TestNotebookExecution:
    """Test that the actual notebooks execute without errors."""

    def test_tutorial_notebook(self) -> None:
        """Execute the quick start tutorial notebook (Direct mode only).

        This test catches issues that the API tests miss, such as:
        - KeyError from direct dict access without .get()
        - Missing imports in cells
        - Broken cell execution order
        - Invalid markdown or formatting

        Uses nbconvert to execute all cells in order, simulating Colab execution.
        """
        pytest.importorskip("nbformat")
        pytest.importorskip("nbconvert")

        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        from pathlib import Path

        # Find the notebook
        notebook_path = (
            Path(__file__).parent.parent.parent / "examples" / "cje_core_demo.ipynb"
        )
        assert notebook_path.exists(), f"Notebook not found at {notebook_path}"

        # Read the notebook
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        # Execute all cells
        # Note: This will download data, install packages, etc. - mark as slow test
        ep = ExecutePreprocessor(
            timeout=600,  # 10 minutes max
            kernel_name="python3",
            allow_errors=False,  # Fail on any cell error
        )

        try:
            # Execute in a temporary directory to avoid polluting the repo
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                ep.preprocess(nb, {"metadata": {"path": tmpdir}})
        except Exception as e:
            pytest.fail(f"Tutorial notebook execution failed: {e}")

        print("✓ Tutorial notebook executed successfully")

    def test_advanced_notebook(self) -> None:
        """Execute the advanced tutorial notebook (IPS and DR modes).

        Tests the advanced off-policy evaluation tutorial covering:
        - IPS mode with logged data
        - DR mode with logged data + fresh draws
        - ESS diagnostics
        - Mode comparisons
        """
        pytest.importorskip("nbformat")
        pytest.importorskip("nbconvert")

        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        from pathlib import Path

        # Find the notebook
        notebook_path = (
            Path(__file__).parent.parent.parent / "examples" / "cje_advanced.ipynb"
        )
        assert notebook_path.exists(), f"Notebook not found at {notebook_path}"

        # Read the notebook
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        # Execute all cells
        ep = ExecutePreprocessor(
            timeout=600,  # 10 minutes max
            kernel_name="python3",
            allow_errors=False,  # Fail on any cell error
        )

        try:
            # Execute in a temporary directory to avoid polluting the repo
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                ep.preprocess(nb, {"metadata": {"path": tmpdir}})
        except Exception as e:
            pytest.fail(f"Advanced notebook execution failed: {e}")

        print("✓ Advanced notebook executed successfully")
