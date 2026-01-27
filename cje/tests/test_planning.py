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
    FittedVarianceModel,
    EvaluationPlan,
    fit_variance_model,
    plan_evaluation,
    plan_for_mde,
    _measure_variance_direct,
)

# Mark all tests in this file as E2E tests
pytestmark = [pytest.mark.e2e, pytest.mark.uses_arena_sample]


# Path to arena sample data (shared with tutorials)
ARENA_SAMPLE_DIR = Path(__file__).parent.parent.parent / "examples" / "arena_sample"


class TestPlanningWorkflow:
    """Test complete planning workflows with real data via analyze_dataset()."""

    @pytest.mark.slow
    def test_plan_allocation_convenience_method(self) -> None:
        """Test EstimationResult.plan_allocation() convenience method.

        This is the user workflow: run pilot → load fresh draws → plan allocation.
        """
        from cje.data.fresh_draws import (
            load_fresh_draws_auto,
            discover_policies_from_fresh_draws,
        )

        fresh_draws_dir = ARENA_SAMPLE_DIR / "fresh_draws"
        if not fresh_draws_dir.exists():
            pytest.skip(f"Fresh draws not found at {fresh_draws_dir}")

        # Run pilot
        result = analyze_dataset(
            fresh_draws_dir=str(fresh_draws_dir),
            verbose=False,
        )

        # Load fresh draws for variance model fitting (use base policy)
        base_data = load_fresh_draws_auto(fresh_draws_dir, "base")

        # Use convenience method (cost_model is required)
        from cje import CostModel as TopLevelCostModel

        allocation = result.plan_allocation(
            budget=5000,
            cost_model=TopLevelCostModel(oracle_cost=16.0),
            fresh_draws=base_data,
            verbose=False,
        )

        # Validate output
        assert allocation.n_samples > 0
        assert allocation.m_oracle > 0
        assert allocation.se_level > 0
        assert "Evaluation Plan" in allocation.summary()

        # Verify the allocation is valid (oracle can't exceed total samples)
        assert allocation.m_oracle <= allocation.n_samples
        # Note: High oracle fraction (even 100%) can be optimal when calibration
        # variance dominates and cost ratio is moderate. The arena data has a
        # highly predictive judge, so σ²_eval ≈ 0 is expected.

        # Test with different cost model
        allocation_custom = result.plan_allocation(
            budget=5000,
            cost_model=TopLevelCostModel(oracle_cost=32.0),  # More expensive oracle
            fresh_draws=base_data,
            verbose=False,
        )
        # More expensive oracle → fewer oracle labels
        assert allocation_custom.m_oracle <= allocation.m_oracle

    @pytest.mark.slow
    def test_empirical_variance_model_returns_positive_components(self) -> None:
        """Empirically-fitted variance model should have positive components.

        Tests that fit_variance_model produces a model with
        non-negative variance components that sum to a positive total.
        """
        from cje.data.fresh_draws import (
            load_fresh_draws_auto,
            discover_policies_from_fresh_draws,
        )

        fresh_draws_dir = ARENA_SAMPLE_DIR / "fresh_draws"
        if not fresh_draws_dir.exists():
            pytest.skip(f"Fresh draws not found at {fresh_draws_dir}")

        # Load fresh draws (use base policy for variance model fitting)
        base_data = load_fresh_draws_auto(fresh_draws_dir, "base")

        # Fit variance model (small grid for speed)
        model = fit_variance_model(
            base_data,
            n_grid=[100, 200],
            oracle_fraction_grid=[0.25, 0.50],
            n_replicates=50,
            verbose=False,
        )

        # Both components should be non-negative and sum to positive
        assert (
            model.sigma2_eval >= 0
        ), f"sigma2_eval should be non-negative, got {model.sigma2_eval}"
        assert (
            model.sigma2_cal >= 0
        ), f"sigma2_cal should be non-negative, got {model.sigma2_cal}"
        assert (
            model.sigma2_eval + model.sigma2_cal > 0
        ), "Total variance should be positive"

    @pytest.mark.slow
    def test_direct_mode_pilot_to_production(self) -> None:
        """Test: pilot analysis → fit variance model → plan production allocation.

        This is the primary user workflow for budget optimization.
        Uses analyze_dataset() + fit_variance_model().
        """
        from cje.data.fresh_draws import (
            load_fresh_draws_auto,
            discover_policies_from_fresh_draws,
        )

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

        # Step 2: Load fresh draws and fit variance model (use base policy)
        base_data = load_fresh_draws_auto(fresh_draws_dir, "base")

        variance_model = fit_variance_model(
            base_data,
            n_grid=[100, 200],
            oracle_fraction_grid=[0.25, 0.50],
            n_replicates=50,
            verbose=False,
        )

        # Variances should be non-negative
        assert (
            variance_model.sigma2_eval >= 0
        ), "Evaluation variance should be non-negative"
        assert (
            variance_model.sigma2_cal >= 0
        ), "Calibration variance should be non-negative"

        # Step 3: Plan optimal allocation for production
        cost_model = CostModel(oracle_cost=16.0)  # Paper's 16× cost ratio
        plan = plan_evaluation(
            budget=5000.0,
            cost_model=cost_model,
            variance_model=variance_model,
        )

        # Validate allocation makes sense
        assert plan.n_samples > 0
        assert plan.m_oracle > 0
        assert plan.m_oracle <= plan.n_samples
        assert 0 < plan.oracle_fraction <= 1
        assert plan.se_level > 0
        # Allow 10% tolerance for rounding when variance components are very small
        assert plan.total_cost <= 5000.0 * 1.10  # Budget respected

        # Summary should be readable
        summary = plan.summary()
        assert "Evaluation Plan" in summary
        assert "MDE" in summary

    @pytest.mark.slow
    def test_dr_mode_pilot_to_production(self) -> None:
        """Test planning workflow with DR mode pilot."""
        from cje.data.fresh_draws import (
            load_fresh_draws_auto,
            discover_policies_from_fresh_draws,
        )

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

        # Load fresh draws and fit variance model (use base policy)
        base_data = load_fresh_draws_auto(fresh_draws_dir, "base")

        variance_model = fit_variance_model(
            base_data,
            n_grid=[100, 200],
            oracle_fraction_grid=[0.25, 0.50],
            n_replicates=50,
            verbose=False,
        )

        # Plan production run
        plan = plan_evaluation(
            budget=10000.0,
            cost_model=CostModel(oracle_cost=16.0),
            variance_model=variance_model,
        )

        # Should produce sensible allocation
        assert plan.n_samples > 0
        assert plan.m_oracle > 0
        assert plan.se_level > 0


class TestSquareRootLawProperties:
    """Test mathematical properties of the Square Root Allocation Law."""

    def test_budget_constraint_respected(self) -> None:
        """Verify allocation respects budget constraint."""
        # Use synthetic variance model for mathematical property tests
        model = FittedVarianceModel(
            sigma2_eval=1.0,
            sigma2_cal=0.5,
            r_squared=0.95,
            n_measurements=9,
        )

        # Test with various budgets (start at 500 to avoid edge cases
        # where minimum m_oracle=1 dominates the budget)
        for budget in [500, 1000, 10000]:
            plan = plan_evaluation(
                budget=budget,
                cost_model=CostModel(oracle_cost=16.0),
                variance_model=model,
            )
            # Allow small tolerance for integer rounding
            assert plan.total_cost <= budget * 1.05

    def test_m_leq_n_constraint(self) -> None:
        """Oracle samples cannot exceed total samples (m ≤ n)."""
        # Use synthetic variance with high calibration variance
        # (would want m > n without constraint)
        model = FittedVarianceModel(
            sigma2_eval=1.0,
            sigma2_cal=10.0,  # High calibration variance
            r_squared=0.95,
            n_measurements=9,
        )

        # Test with cheap oracle
        plan = plan_evaluation(
            budget=1000,
            cost_model=CostModel(surrogate_cost=1.0, oracle_cost=0.1),
            variance_model=model,
        )
        assert plan.m_oracle <= plan.n_samples


class TestCostModelIntegration:
    """Test CostModel with real workflows."""

    def test_cost_ratio_affects_allocation(self) -> None:
        """Higher oracle cost → lower oracle fraction."""
        # Use synthetic variance model
        model = FittedVarianceModel(
            sigma2_eval=1.0,
            sigma2_cal=0.5,
            r_squared=0.95,
            n_measurements=9,
        )

        # Compare allocations with different cost ratios
        # Use higher budget to avoid m_min constraint dominating
        plan_cheap = plan_evaluation(
            budget=10000,
            cost_model=CostModel(oracle_cost=2.0),  # Cheap oracle
            variance_model=model,
        )

        plan_expensive = plan_evaluation(
            budget=10000,
            cost_model=CostModel(oracle_cost=32.0),  # Expensive oracle
            variance_model=model,
        )

        # More expensive oracle → lower oracle fraction
        assert plan_expensive.oracle_fraction <= plan_cheap.oracle_fraction

    def test_to_dict_serialization(self) -> None:
        """Allocation can be serialized for logging/storage."""
        import json

        # Use synthetic variance model
        model = FittedVarianceModel(
            sigma2_eval=1.0,
            sigma2_cal=0.5,
            r_squared=0.95,
            n_measurements=9,
        )

        plan = plan_evaluation(
            budget=1000,
            cost_model=CostModel(),
            variance_model=model,
        )

        # Should be JSON-serializable
        d = plan.to_dict()
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

    def test_validates_positive_surrogate_cost(self) -> None:
        """CostModel rejects non-positive surrogate_cost."""
        with pytest.raises(ValueError, match="surrogate_cost must be positive"):
            CostModel(surrogate_cost=0.0, oracle_cost=16.0)

        with pytest.raises(ValueError, match="surrogate_cost must be positive"):
            CostModel(surrogate_cost=-1.0, oracle_cost=16.0)

    def test_validates_positive_oracle_cost(self) -> None:
        """CostModel rejects non-positive oracle_cost."""
        with pytest.raises(ValueError, match="oracle_cost must be positive"):
            CostModel(surrogate_cost=1.0, oracle_cost=0.0)

        with pytest.raises(ValueError, match="oracle_cost must be positive"):
            CostModel(surrogate_cost=1.0, oracle_cost=-5.0)


class TestEmpiricalVarianceMeasurement:
    """Tests for empirical variance measurement utilities."""

    def test_measure_variance_direct_basic(self) -> None:
        """Basic test that _measure_variance_direct runs without error."""
        from cje.data.fresh_draws import load_fresh_draws_auto

        fresh_draws_dir = ARENA_SAMPLE_DIR / "fresh_draws"
        if not fresh_draws_dir.exists():
            pytest.skip(f"Fresh draws not found at {fresh_draws_dir}")

        # Load fresh draws (use base policy)
        base_data = load_fresh_draws_auto(fresh_draws_dir, "base")

        # Measure variance at a single allocation
        result = _measure_variance_direct(
            base_data,
            n_prompts=100,
            m_oracle=20,
            n_replicates=3,  # Fewer replicates for test speed
        )

        # Should return valid results
        assert "se" in result
        assert "variance" in result
        assert result["n_actual"] == 100
        assert result["m_actual"] == 20

        # SE should be positive and finite
        if not np.isnan(result["se"]):
            assert result["se"] > 0
            assert result["variance"] > 0


class TestFittedVarianceModel:
    """Tests for the empirically-fitted variance model."""

    def test_fit_variance_model_synthetic(self) -> None:
        """Fit model to synthetic data with known parameters."""
        from cje.diagnostics.planning import (
            _fit_variance_model_from_measurements,
        )

        # Generate synthetic measurements from known model
        true_sigma2_eval = 1.0
        true_sigma2_cal = 0.5

        np.random.seed(42)
        measurements = []
        for n in [100, 200, 400]:
            for m in [25, 50, 100]:
                true_var = true_sigma2_eval / n + true_sigma2_cal / m
                # Add small noise (5%)
                noisy_var = true_var * (1 + np.random.normal(0, 0.05))
                measurements.append((n, m, noisy_var))

        model = _fit_variance_model_from_measurements(measurements)

        # Fitted parameters should be close to true values
        assert (
            abs(model.sigma2_eval - true_sigma2_eval) < 0.3
        ), f"sigma2_eval={model.sigma2_eval}, expected ~{true_sigma2_eval}"
        assert (
            abs(model.sigma2_cal - true_sigma2_cal) < 0.2
        ), f"sigma2_cal={model.sigma2_cal}, expected ~{true_sigma2_cal}"
        assert model.r_squared > 0.90, f"R²={model.r_squared}, expected > 0.90"

    def test_fit_variance_model_prediction(self) -> None:
        """Model should predict variance correctly."""
        from cje.diagnostics.planning import (
            _fit_variance_model_from_measurements,
        )

        # Simple exact measurements (no noise)
        measurements = [
            (100, 50, 1.0 / 100 + 0.5 / 50),  # 0.02
            (200, 100, 1.0 / 200 + 0.5 / 100),  # 0.01
            (400, 200, 1.0 / 400 + 0.5 / 200),  # 0.005
        ]

        model = _fit_variance_model_from_measurements(measurements)

        # Predictions should match the formula
        predicted = model.predict_variance(100, 50)
        expected = model.sigma2_eval / 100 + model.sigma2_cal / 50
        assert abs(predicted - expected) < 1e-10

        # SE should be sqrt of variance
        assert abs(model.predict_se(100, 50) - np.sqrt(predicted)) < 1e-10

    def test_fit_variance_model_extrapolation(self) -> None:
        """Model should extrapolate to larger n than training data."""
        from cje.diagnostics.planning import (
            _fit_variance_model_from_measurements,
        )

        # Fit on small n values
        measurements = [
            (100, 25, 0.030),
            (100, 50, 0.020),
            (200, 50, 0.015),
            (200, 100, 0.010),
        ]

        model = _fit_variance_model_from_measurements(measurements)

        # Extrapolate to much larger n
        var_1000 = model.predict_variance(1000, 250)
        var_2000 = model.predict_variance(2000, 500)

        # Variance should decrease with larger n
        assert var_1000 < model.predict_variance(200, 50)
        assert var_2000 < var_1000

        # Variance should be positive
        assert var_1000 > 0
        assert var_2000 > 0

    def test_fit_variance_model_requires_min_measurements(self) -> None:
        """Should raise error with fewer than 3 measurements."""
        from cje.diagnostics.planning import (
            _fit_variance_model_from_measurements,
        )

        with pytest.raises(ValueError, match="at least 3"):
            _fit_variance_model_from_measurements([(100, 25, 0.01), (200, 50, 0.005)])

    def test_plan_evaluation_with_model(self) -> None:
        """plan_evaluation should work with FittedVarianceModel."""
        model = FittedVarianceModel(
            sigma2_eval=1.0,
            sigma2_cal=0.5,
            r_squared=0.95,
            n_measurements=9,
        )

        plan = plan_evaluation(
            budget=1000.0,
            cost_model=CostModel(oracle_cost=16.0),
            variance_model=model,
        )

        # Should return valid allocation
        assert plan.n_samples > 0
        assert plan.m_oracle > 0
        assert plan.m_oracle <= plan.n_samples
        assert plan.total_cost <= 1000.0 * 1.05  # Allow small rounding

    @pytest.mark.slow
    def test_fit_variance_model_real_data(self) -> None:
        """Fit model from real pilot data."""
        from cje.data.fresh_draws import load_fresh_draws_auto

        fresh_draws_dir = ARENA_SAMPLE_DIR / "fresh_draws"
        if not fresh_draws_dir.exists():
            pytest.skip(f"Fresh draws not found at {fresh_draws_dir}")

        # Load fresh draws (use base policy)
        base_data = load_fresh_draws_auto(fresh_draws_dir, "base")

        # Fit model with small grid for speed
        model = fit_variance_model(
            base_data,
            n_grid=[100, 200],
            oracle_fraction_grid=[0.25, 0.50],
            n_replicates=50,
            verbose=False,
        )

        # Model should have non-negative variance components
        # With a small grid (4 points), NNLS may put all variance into one component
        assert model.sigma2_eval >= 0
        assert model.sigma2_cal >= 0
        assert (
            model.sigma2_eval + model.sigma2_cal > 0
        ), "Total variance should be positive"
        assert model.n_measurements >= 3

        # R² can vary with few noisy measurements - just check it's computed
        # For proper validation, use larger grid with more replicates
        assert not np.isnan(model.r_squared)


class TestEvaluationPlanAPI:
    """Tests for the new MDE-centric planning API (plan_evaluation, plan_for_mde)."""

    def test_plan_evaluation_basic(self) -> None:
        """plan_evaluation returns valid EvaluationPlan with MDE."""
        model = FittedVarianceModel(
            sigma2_eval=0.008,
            sigma2_cal=0.004,
            r_squared=0.95,
            n_measurements=12,
        )
        cost_model = CostModel(oracle_cost=16.0)

        plan = plan_evaluation(budget=5000, variance_model=model, cost_model=cost_model)

        # Should return EvaluationPlan
        assert isinstance(plan, EvaluationPlan)

        # Core fields should be populated
        assert plan.n_samples > 0
        assert plan.m_oracle > 0
        assert plan.m_oracle <= plan.n_samples
        assert plan.total_cost <= 5000 * 1.01  # Allow small rounding

        # MDE should be positive and reasonable
        assert plan.mde > 0
        assert plan.mde < 1.0  # Should be less than 100%

        # SE fields
        assert plan.se_level > 0
        assert plan.se_comparison == pytest.approx(np.sqrt(2) * plan.se_level)

        # Settings preserved
        assert plan.power == 0.8
        assert plan.alpha == 0.05

    def test_plan_evaluation_custom_power(self) -> None:
        """plan_evaluation respects power and alpha parameters."""
        model = FittedVarianceModel(
            sigma2_eval=0.008, sigma2_cal=0.004, r_squared=0.95, n_measurements=12
        )
        cost_model = CostModel(oracle_cost=16.0)

        plan_80 = plan_evaluation(
            budget=5000, variance_model=model, cost_model=cost_model, power=0.8
        )
        plan_90 = plan_evaluation(
            budget=5000, variance_model=model, cost_model=cost_model, power=0.9
        )

        # Same allocation (budget determines allocation, not power)
        assert plan_80.n_samples == plan_90.n_samples
        assert plan_80.m_oracle == plan_90.m_oracle

        # Higher power → higher MDE (harder to achieve)
        assert plan_90.mde > plan_80.mde

    def test_plan_for_mde_basic(self) -> None:
        """plan_for_mde finds budget needed for target MDE."""
        model = FittedVarianceModel(
            sigma2_eval=0.008, sigma2_cal=0.004, r_squared=0.95, n_measurements=12
        )
        cost_model = CostModel(oracle_cost=16.0)

        # Target 2% MDE
        plan = plan_for_mde(
            target_mde=0.02, variance_model=model, cost_model=cost_model
        )

        # Should achieve close to target MDE
        assert plan.mde <= 0.021  # Allow small tolerance

        # Should return valid allocation
        assert plan.n_samples > 0
        assert plan.m_oracle > 0
        assert plan.total_cost > 0

    def test_plan_for_mde_smaller_target_needs_more_budget(self) -> None:
        """Smaller MDE target requires larger budget."""
        model = FittedVarianceModel(
            sigma2_eval=0.008, sigma2_cal=0.004, r_squared=0.95, n_measurements=12
        )
        cost_model = CostModel(oracle_cost=16.0)

        plan_5pct = plan_for_mde(
            target_mde=0.05, variance_model=model, cost_model=cost_model
        )
        plan_2pct = plan_for_mde(
            target_mde=0.02, variance_model=model, cost_model=cost_model
        )
        plan_1pct = plan_for_mde(
            target_mde=0.01, variance_model=model, cost_model=cost_model
        )

        # Smaller MDE → more samples → higher cost
        assert plan_1pct.total_cost > plan_2pct.total_cost
        assert plan_2pct.total_cost > plan_5pct.total_cost

    def test_evaluation_plan_mde_at_power(self) -> None:
        """EvaluationPlan.mde_at_power computes MDE at different power levels."""
        model = FittedVarianceModel(
            sigma2_eval=0.008, sigma2_cal=0.004, r_squared=0.95, n_measurements=12
        )
        cost_model = CostModel(oracle_cost=16.0)

        plan = plan_evaluation(
            budget=5000, variance_model=model, cost_model=cost_model, power=0.8
        )

        # mde_at_power(0.8) should match plan.mde
        assert plan.mde_at_power(0.8) == pytest.approx(plan.mde, rel=1e-6)

        # Higher power → higher MDE
        mde_50 = plan.mde_at_power(0.5)
        mde_80 = plan.mde_at_power(0.8)
        mde_90 = plan.mde_at_power(0.9)
        mde_95 = plan.mde_at_power(0.95)

        assert mde_50 < mde_80 < mde_90 < mde_95

    def test_evaluation_plan_power_to_detect(self) -> None:
        """EvaluationPlan.power_to_detect computes power for effect sizes."""
        model = FittedVarianceModel(
            sigma2_eval=0.008, sigma2_cal=0.004, r_squared=0.95, n_measurements=12
        )
        cost_model = CostModel(oracle_cost=16.0)

        plan = plan_evaluation(
            budget=5000, variance_model=model, cost_model=cost_model, power=0.8
        )

        # At the MDE, power should be ~0.8
        power_at_mde = plan.power_to_detect(plan.mde)
        assert power_at_mde == pytest.approx(0.8, abs=0.01)

        # Larger effect → higher power
        power_2x = plan.power_to_detect(plan.mde * 2)
        assert power_2x > power_at_mde

        # Smaller effect → lower power
        power_half = plan.power_to_detect(plan.mde * 0.5)
        assert power_half < power_at_mde

        # Power should be in [0, 1]
        assert 0 <= power_half <= 1
        assert 0 <= power_2x <= 1

    def test_evaluation_plan_summary(self) -> None:
        """EvaluationPlan.summary() returns readable string."""
        model = FittedVarianceModel(
            sigma2_eval=0.008, sigma2_cal=0.004, r_squared=0.95, n_measurements=12
        )
        cost_model = CostModel(oracle_cost=16.0)

        plan = plan_evaluation(budget=5000, variance_model=model, cost_model=cost_model)
        summary = plan.summary()

        # Should contain key information
        assert "Evaluation Plan" in summary
        assert "Allocation" in summary
        assert "MDE" in summary
        assert "80%" in summary  # power
        assert str(plan.n_samples) in summary.replace(",", "")

    def test_evaluation_plan_to_dict(self) -> None:
        """EvaluationPlan.to_dict() returns serializable dict."""
        import json

        model = FittedVarianceModel(
            sigma2_eval=0.008, sigma2_cal=0.004, r_squared=0.95, n_measurements=12
        )
        cost_model = CostModel(oracle_cost=16.0)

        plan = plan_evaluation(budget=5000, variance_model=model, cost_model=cost_model)
        d = plan.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0

        # Should contain key fields
        assert "n_samples" in d
        assert "m_oracle" in d
        assert "mde" in d
        assert "se_level" in d
        assert "cost_model" in d

    def test_mde_formula_correctness(self) -> None:
        """Verify MDE = (z_alpha + z_beta) * sqrt(2) * SE."""
        from scipy import stats

        model = FittedVarianceModel(
            sigma2_eval=0.01, sigma2_cal=0.005, r_squared=0.95, n_measurements=12
        )
        cost_model = CostModel(oracle_cost=16.0)

        plan = plan_evaluation(
            budget=10000,
            variance_model=model,
            cost_model=cost_model,
            power=0.8,
            alpha=0.05,
        )

        # Manual calculation
        z_alpha = stats.norm.ppf(1 - 0.05 / 2)  # ~1.96
        z_power = stats.norm.ppf(0.8)  # ~0.84
        se_comparison = np.sqrt(2) * plan.se_level
        expected_mde = (z_alpha + z_power) * se_comparison

        assert plan.mde == pytest.approx(expected_mde, rel=1e-6)

    def test_top_level_imports(self) -> None:
        """New API is accessible from top-level cje package."""
        from cje import (
            plan_evaluation,
            FittedVarianceModel,
            EvaluationPlan,
            CostModel,
        )

        # Should be callable
        model = FittedVarianceModel(
            sigma2_eval=0.01, sigma2_cal=0.005, r_squared=0.95, n_measurements=12
        )
        cost_model = CostModel(oracle_cost=16.0)
        plan = plan_evaluation(budget=5000, variance_model=model, cost_model=cost_model)
        assert isinstance(plan, EvaluationPlan)
