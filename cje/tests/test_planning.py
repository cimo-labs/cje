"""End-to-end tests for budget optimization utilities.

Tests the Square Root Allocation Law implementation from CJE paper Appendix F
using real arena data and complete workflows via analyze_dataset().
"""

import pytest
import numpy as np
from pathlib import Path

from cje import analyze_dataset

from cje.diagnostics.planning import (
    CostModel,
    compute_optimal_allocation,
    estimate_variance_components,
    diagnose_allocation_efficiency,
    compute_mde_contours,
)

# Mark all tests in this file as E2E tests
pytestmark = [pytest.mark.e2e, pytest.mark.uses_arena_sample]


# Path to arena sample data (shared with tutorials)
ARENA_SAMPLE_DIR = Path(__file__).parent.parent.parent / "examples" / "arena_sample"


class TestPlanningWorkflow:
    """Test complete planning workflows with real data via analyze_dataset()."""

    def test_plan_allocation_convenience_method(self) -> None:
        """Test EstimationResult.plan_allocation() convenience method.

        This is the simplest user workflow - one line after pilot run.
        """
        fresh_draws_dir = ARENA_SAMPLE_DIR / "fresh_draws"
        if not fresh_draws_dir.exists():
            pytest.skip(f"Fresh draws not found at {fresh_draws_dir}")

        # Run pilot
        result = analyze_dataset(
            fresh_draws_dir=str(fresh_draws_dir),
            verbose=False,
        )

        # Use convenience method (the cleanest API)
        allocation = result.plan_allocation(budget=5000)

        # Validate output
        assert allocation.n_samples > 0
        assert allocation.m_oracle > 0
        assert allocation.expected_se > 0
        assert "Optimal allocation" in allocation.summary()

        # Test with custom cost model (import from top-level to verify export)
        from cje import CostModel as TopLevelCostModel

        allocation_custom = result.plan_allocation(
            budget=5000,
            cost_model=TopLevelCostModel(oracle_cost=32.0),  # More expensive oracle
        )
        # More expensive oracle → fewer oracle labels
        assert allocation_custom.m_oracle <= allocation.m_oracle

    def test_direct_mode_pilot_to_production(self) -> None:
        """Test: pilot analysis → extract variances → plan production allocation.

        This is the primary user workflow for budget optimization.
        Uses analyze_dataset() which is the high-level API users interact with.
        """
        fresh_draws_dir = ARENA_SAMPLE_DIR / "fresh_draws"
        if not fresh_draws_dir.exists():
            pytest.skip(f"Fresh draws not found at {fresh_draws_dir}")

        # Step 1: Run pilot analysis using the high-level API (Direct mode)
        pilot_result = analyze_dataset(
            fresh_draws_dir=str(fresh_draws_dir),
            verbose=False,
        )

        # Verify pilot ran successfully
        assert len(pilot_result.estimates) > 0
        assert all(0 <= e <= 1 for e in pilot_result.estimates)

        # Step 2: Extract variance components from pilot
        sigma2_eval, sigma2_cal = estimate_variance_components(pilot_result)

        # Variances should be positive
        assert sigma2_eval > 0, "Evaluation variance should be positive"
        assert sigma2_cal >= 0, "Calibration variance should be non-negative"

        # Step 3: Plan optimal allocation for production
        cost_model = CostModel(oracle_cost=16.0)  # Paper's 16× cost ratio
        allocation = compute_optimal_allocation(
            budget=5000.0,
            cost_model=cost_model,
            sigma2_eval=sigma2_eval,
            sigma2_cal=sigma2_cal,
        )

        # Validate allocation makes sense
        assert allocation.n_samples > 0
        assert allocation.m_oracle > 0
        assert allocation.m_oracle <= allocation.n_samples
        assert 0 < allocation.oracle_fraction <= 1
        assert allocation.expected_se > 0
        assert allocation.total_cost <= 5000.0 * 1.01  # Budget respected

        # Summary should be readable
        summary = allocation.summary()
        assert "Optimal allocation" in summary
        assert "Expected SE" in summary

    def test_dr_mode_pilot_to_production(self) -> None:
        """Test planning workflow with DR mode pilot (most accurate variance estimates)."""
        logged_data_path = ARENA_SAMPLE_DIR / "logged_data.jsonl"
        fresh_draws_dir = ARENA_SAMPLE_DIR / "fresh_draws"

        if not logged_data_path.exists():
            pytest.skip(f"Logged data not found at {logged_data_path}")
        if not fresh_draws_dir.exists():
            pytest.skip(f"Fresh draws not found at {fresh_draws_dir}")

        # Run DR mode pilot (both logged data and fresh draws → auto-selects DR)
        pilot_result = analyze_dataset(
            logged_data_path=str(logged_data_path),
            fresh_draws_dir=str(fresh_draws_dir),
            verbose=False,
        )

        # Verify DR mode was selected
        mode_selection = pilot_result.metadata.get("mode_selection", {})
        assert (
            mode_selection.get("mode") == "dr"
        ), f"Expected DR mode, got {mode_selection}"

        # Extract variance components
        sigma2_eval, sigma2_cal = estimate_variance_components(pilot_result)

        # Plan production run
        allocation = compute_optimal_allocation(
            budget=10000.0,
            cost_model=CostModel(oracle_cost=16.0),
            sigma2_eval=sigma2_eval,
            sigma2_cal=sigma2_cal,
        )

        # Should produce sensible allocation
        assert allocation.n_samples > 0
        assert allocation.m_oracle > 0
        assert allocation.expected_se > 0

    def test_efficiency_diagnostic_workflow(self) -> None:
        """Test: run analysis → diagnose allocation efficiency → get recommendation."""
        fresh_draws_dir = ARENA_SAMPLE_DIR / "fresh_draws"
        if not fresh_draws_dir.exists():
            pytest.skip(f"Fresh draws not found at {fresh_draws_dir}")

        # Run analysis
        result = analyze_dataset(
            fresh_draws_dir=str(fresh_draws_dir),
            verbose=False,
        )

        # Diagnose allocation efficiency
        cost_model = CostModel(oracle_cost=16.0)
        diag = diagnose_allocation_efficiency(result, cost_model)

        # Should return valid diagnostic
        assert "status" in diag
        assert diag["status"] in ["UNDER_LABELED", "OVER_LABELED", "BALANCED"]
        assert "recommendation" in diag
        assert "imbalance" in diag
        assert diag["imbalance"] is not None

        # Diagnostic values should be floats
        assert isinstance(diag["calibration_uncertainty_share"], float)
        assert isinstance(diag["oracle_spend_fraction"], float)


class TestMDEPlanning:
    """Test MDE contour computation for sample size planning."""

    def test_mde_contours_from_pilot(self) -> None:
        """Test: extract variances from pilot → compute MDE grid for planning."""
        fresh_draws_dir = ARENA_SAMPLE_DIR / "fresh_draws"
        if not fresh_draws_dir.exists():
            pytest.skip(f"Fresh draws not found at {fresh_draws_dir}")

        # Run pilot via high-level API
        pilot_result = analyze_dataset(
            fresh_draws_dir=str(fresh_draws_dir),
            verbose=False,
        )

        # Extract variances
        sigma2_eval, sigma2_cal = estimate_variance_components(pilot_result)

        # Compute MDE grid for sample size planning
        n_range = [500, 1000, 2000, 5000]
        oracle_fractions = [0.05, 0.10, 0.25]

        mde_grid = compute_mde_contours(
            n_range=n_range,
            oracle_fractions=oracle_fractions,
            sigma2_eval=sigma2_eval,
            sigma2_cal=sigma2_cal,
            power=0.8,
            alpha=0.05,
        )

        # Validate grid shape
        assert mde_grid.shape == (len(n_range), len(oracle_fractions))

        # MDE should decrease with larger n
        for j in range(len(oracle_fractions)):
            for i in range(1, len(n_range)):
                assert mde_grid[i, j] < mde_grid[i - 1, j], (
                    f"MDE should decrease with n: "
                    f"MDE[n={n_range[i]}] >= MDE[n={n_range[i-1]}]"
                )

        # All MDE values should be positive and reasonable
        assert np.all(mde_grid > 0)
        assert np.all(mde_grid < 10)  # Sanity check


class TestSquareRootLawProperties:
    """Test mathematical properties of the Square Root Allocation Law."""

    def test_budget_constraint_respected(self) -> None:
        """Verify allocation respects budget constraint."""
        fresh_draws_dir = ARENA_SAMPLE_DIR / "fresh_draws"
        if not fresh_draws_dir.exists():
            pytest.skip(f"Fresh draws not found at {fresh_draws_dir}")

        # Get variances from real data
        result = analyze_dataset(
            fresh_draws_dir=str(fresh_draws_dir),
            verbose=False,
        )
        sigma2_eval, sigma2_cal = estimate_variance_components(result)

        # Test with various budgets (start at 500 to avoid edge cases
        # where minimum m_oracle=1 dominates the budget)
        for budget in [500, 1000, 10000]:
            allocation = compute_optimal_allocation(
                budget=budget,
                cost_model=CostModel(oracle_cost=16.0),
                sigma2_eval=sigma2_eval,
                sigma2_cal=sigma2_cal,
            )
            # Allow small tolerance for integer rounding
            assert allocation.total_cost <= budget * 1.05

    def test_m_leq_n_constraint(self) -> None:
        """Oracle samples cannot exceed total samples (m ≤ n)."""
        fresh_draws_dir = ARENA_SAMPLE_DIR / "fresh_draws"
        if not fresh_draws_dir.exists():
            pytest.skip(f"Fresh draws not found at {fresh_draws_dir}")

        # Get variances from real data
        result = analyze_dataset(
            fresh_draws_dir=str(fresh_draws_dir),
            verbose=False,
        )
        sigma2_eval, sigma2_cal = estimate_variance_components(result)

        # Test with cheap oracle (would want m > n without constraint)
        allocation = compute_optimal_allocation(
            budget=1000,
            cost_model=CostModel(surrogate_cost=1.0, oracle_cost=0.1),
            sigma2_eval=sigma2_eval,
            sigma2_cal=max(sigma2_cal, sigma2_eval * 10),  # Force high cal variance
        )
        assert allocation.m_oracle <= allocation.n_samples


class TestCostModelIntegration:
    """Test CostModel with real workflows."""

    def test_cost_ratio_affects_allocation(self) -> None:
        """Higher oracle cost → lower oracle fraction."""
        fresh_draws_dir = ARENA_SAMPLE_DIR / "fresh_draws"
        if not fresh_draws_dir.exists():
            pytest.skip(f"Fresh draws not found at {fresh_draws_dir}")

        # Get variances from real data
        result = analyze_dataset(
            fresh_draws_dir=str(fresh_draws_dir),
            verbose=False,
        )
        sigma2_eval, sigma2_cal = estimate_variance_components(result)

        # Compare allocations with different cost ratios
        alloc_cheap = compute_optimal_allocation(
            budget=1000,
            cost_model=CostModel(oracle_cost=2.0),  # Cheap oracle
            sigma2_eval=sigma2_eval,
            sigma2_cal=sigma2_cal,
        )

        alloc_expensive = compute_optimal_allocation(
            budget=1000,
            cost_model=CostModel(oracle_cost=32.0),  # Expensive oracle
            sigma2_eval=sigma2_eval,
            sigma2_cal=sigma2_cal,
        )

        # More expensive oracle → lower oracle fraction
        assert alloc_expensive.oracle_fraction <= alloc_cheap.oracle_fraction

    def test_to_dict_serialization(self) -> None:
        """Allocation can be serialized for logging/storage."""
        import json

        fresh_draws_dir = ARENA_SAMPLE_DIR / "fresh_draws"
        if not fresh_draws_dir.exists():
            pytest.skip(f"Fresh draws not found at {fresh_draws_dir}")

        # Get variances from real data
        result = analyze_dataset(
            fresh_draws_dir=str(fresh_draws_dir),
            verbose=False,
        )
        sigma2_eval, sigma2_cal = estimate_variance_components(result)

        allocation = compute_optimal_allocation(
            budget=1000,
            cost_model=CostModel(),
            sigma2_eval=sigma2_eval,
            sigma2_cal=sigma2_cal,
        )

        # Should be JSON-serializable
        d = allocation.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0

        # Should contain key fields
        assert "n_samples" in d
        assert "m_oracle" in d
        assert "oracle_fraction" in d
        assert "cost_model" in d


class TestCostModelUnit:
    """Unit tests for CostModel dataclass (no external data needed)."""

    def test_default_values(self) -> None:
        """Default values match paper (16× oracle cost)."""
        cost = CostModel()
        assert cost.surrogate_cost == 1.0
        assert cost.oracle_cost == 16.0

    def test_cost_ratio(self) -> None:
        """Cost ratio computed correctly."""
        cost = CostModel(surrogate_cost=1.0, oracle_cost=16.0)
        assert cost.cost_ratio == pytest.approx(1 / 16)
