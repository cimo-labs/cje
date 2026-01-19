"""E2E and smoke tests for planning visualization functions.

E2E tests verify complete workflows with real arena data.
Smoke tests verify functions run without error with synthetic data.

Following CJE testing philosophy:
- E2E tests with real data (marked @pytest.mark.slow)
- Fast smoke tests with synthetic data for CI

IMPORTANT: All visualization functions require explicit cost_model.
There are no meaningful defaults - allocation depends on actual costs.
"""

from __future__ import annotations

import pytest
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
from typing import Dict

from cje.data.fresh_draws import FreshDrawDataset
from cje.diagnostics.planning import (
    FittedVarianceModel,
    CostModel,
    fit_variance_model_from_pilot,
)


# Mark all tests as using arena sample
pytestmark = [pytest.mark.uses_arena_sample]


# ============================================================================
# E2E Tests: Complete Workflows with Real Data
# ============================================================================


class TestPlanningVisualizationE2E:
    """E2E tests using real arena data.

    These tests verify the complete user workflow:
    1. Load fresh draws from pilot
    2. Fit variance model
    3. Generate planning visualizations with explicit cost model
    """

    @pytest.mark.slow
    def test_fit_and_visualize_workflow(
        self, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Complete workflow: fit variance model → generate all plots.

        This is what users actually do:
        1. Load fresh draws from pilot
        2. Fit variance model
        3. Specify their actual cost model
        4. Generate planning visualizations
        """
        from cje.visualization.planning import generate_canonical_planning_figures

        # Step 1: Fit variance model from real data (small grid for speed)
        variance_model = fit_variance_model_from_pilot(
            arena_fresh_draws,
            n_grid=[100, 200],
            oracle_fraction_grid=[0.25, 0.50],
            n_replicates=5,
            n_bootstrap=50,
            verbose=False,
        )

        # Validate model was fit
        assert variance_model.sigma2_eval >= 0
        assert variance_model.sigma2_cal >= 0
        assert variance_model.n_measurements >= 3

        # Step 2: Specify cost model (user must think about their costs)
        cost_model = CostModel(oracle_cost=16.0)  # 16× ratio from paper

        # Step 3: Generate measurements for fit plot (from model fitting)
        measurements = _generate_measurements_from_model(variance_model)

        # Step 4: Generate all canonical figures with explicit cost model
        figs = generate_canonical_planning_figures(
            variance_model, cost_model, measurements
        )

        # Validate outputs
        assert isinstance(figs, dict)
        assert len(figs) == 4

        for name, fig in figs.items():
            assert isinstance(fig, plt.Figure), f"{name} is not a Figure"
            assert len(fig.axes) > 0, f"{name} has no axes"
            plt.close(fig)

    @pytest.mark.slow
    def test_dashboard_with_real_variance_model(
        self, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Dashboard plot shows sensible values with real data."""
        from cje.visualization.planning import plot_planning_dashboard

        # Fit real variance model
        variance_model = fit_variance_model_from_pilot(
            arena_fresh_draws,
            n_grid=[100, 200],
            oracle_fraction_grid=[0.25, 0.50],
            n_replicates=5,
            n_bootstrap=50,
            verbose=False,
        )

        # Explicit cost model required
        cost_model = CostModel(oracle_cost=16.0)

        # Generate dashboard - budget range computed adaptively
        fig = plot_planning_dashboard(variance_model, cost_model)

        # Validate structure
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3  # 3 panels

        plt.close(fig)

    @pytest.mark.slow
    def test_variance_fit_plot_with_real_data(
        self, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Variance fit plot shows R² with real measurements."""
        from cje.visualization.planning import plot_variance_model_fit

        # Fit real variance model
        variance_model = fit_variance_model_from_pilot(
            arena_fresh_draws,
            n_grid=[100, 200],
            oracle_fraction_grid=[0.25, 0.50],
            n_replicates=5,
            n_bootstrap=50,
            verbose=False,
        )

        # Explicit cost model required
        cost_model = CostModel(oracle_cost=16.0)

        # Generate measurements
        measurements = _generate_measurements_from_model(variance_model)

        # Generate plot
        fig = plot_variance_model_fit(measurements, variance_model, cost_model)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2  # 2 panels (+ colorbar)

        plt.close(fig)


# ============================================================================
# Smoke Tests: Quick Validation with Synthetic Data
# ============================================================================


class TestPlanningVisualizationSmoke:
    """Fast smoke tests with synthetic data for CI.

    These tests verify that plotting functions don't crash and return
    valid Figure objects. They use synthetic data to run quickly.

    IMPORTANT: All tests must pass explicit cost_model.
    """

    @pytest.fixture
    def synthetic_model(self) -> FittedVarianceModel:
        """Synthetic variance model for fast tests."""
        return FittedVarianceModel(
            sigma2_eval=0.008,
            sigma2_cal=0.004,
            r_squared=0.95,
            n_measurements=12,
        )

    @pytest.fixture
    def cost_model(self) -> CostModel:
        """Explicit cost model for all tests."""
        return CostModel(oracle_cost=16.0)

    @pytest.fixture
    def synthetic_measurements(
        self, synthetic_model: FittedVarianceModel
    ) -> list[tuple[int, int, float]]:
        """Synthetic measurements matching the model."""
        return [
            (200, 40, synthetic_model.predict_variance(200, 40)),
            (200, 80, synthetic_model.predict_variance(200, 80)),
            (400, 80, synthetic_model.predict_variance(400, 80)),
            (400, 160, synthetic_model.predict_variance(400, 160)),
        ]

    def test_all_plots_run_without_error(
        self,
        synthetic_model: FittedVarianceModel,
        cost_model: CostModel,
        synthetic_measurements: list[tuple[int, int, float]],
    ) -> None:
        """All plot functions execute without raising."""
        from cje.visualization.planning import (
            plot_planning_dashboard,
            plot_variance_model_fit,
            plot_oracle_sensitivity,
            plot_optimality_proof,
        )

        # Each should return a Figure (cost_model is required)
        fig1 = plot_planning_dashboard(synthetic_model, cost_model)
        fig2 = plot_variance_model_fit(
            synthetic_measurements, synthetic_model, cost_model
        )
        fig3 = plot_oracle_sensitivity(synthetic_model, cost_model)
        fig4 = plot_optimality_proof(synthetic_model, cost_model)

        for fig in [fig1, fig2, fig3, fig4]:
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_dashboard_panel_count(
        self, synthetic_model: FittedVarianceModel, cost_model: CostModel
    ) -> None:
        """Dashboard has exactly 3 panels."""
        from cje.visualization.planning import plot_planning_dashboard

        fig = plot_planning_dashboard(synthetic_model, cost_model)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_optimality_proof_panel_count(
        self, synthetic_model: FittedVarianceModel, cost_model: CostModel
    ) -> None:
        """Optimality proof has exactly 2 panels."""
        from cje.visualization.planning import plot_optimality_proof

        fig = plot_optimality_proof(synthetic_model, cost_model)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_custom_budget_range(
        self, synthetic_model: FittedVarianceModel, cost_model: CostModel
    ) -> None:
        """Plot functions accept custom budget range."""
        from cje.visualization.planning import plot_planning_dashboard

        # Custom budget range
        fig = plot_planning_dashboard(
            synthetic_model,
            cost_model,
            budget_range=(2000, 15000),
            highlight_budgets=[3000, 7000],
            figsize=(12, 4),
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_different_cost_models(self, synthetic_model: FittedVarianceModel) -> None:
        """Visualizations work with different cost ratios."""
        from cje.visualization.planning import plot_planning_dashboard

        # Cheap oracle (4×)
        fig1 = plot_planning_dashboard(synthetic_model, CostModel(oracle_cost=4.0))
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        # Expensive oracle (64×)
        fig2 = plot_planning_dashboard(synthetic_model, CostModel(oracle_cost=64.0))
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

    def test_generate_canonical_figures_returns_dict(
        self,
        synthetic_model: FittedVarianceModel,
        cost_model: CostModel,
        synthetic_measurements: list[tuple[int, int, float]],
    ) -> None:
        """generate_canonical_planning_figures returns dict with all figures."""
        from cje.visualization.planning import generate_canonical_planning_figures

        figs = generate_canonical_planning_figures(
            synthetic_model, cost_model, synthetic_measurements
        )

        assert isinstance(figs, dict)
        assert "dashboard" in figs
        assert "variance_fit" in figs
        assert "oracle_sensitivity" in figs
        assert "optimality" in figs

        for name, fig in figs.items():
            assert isinstance(fig, plt.Figure), f"{name} is not a Figure"
            plt.close(fig)

    def test_cost_model_is_required(self, synthetic_model: FittedVarianceModel) -> None:
        """Visualization functions require explicit cost_model (no defaults)."""
        from cje.visualization.planning import plot_planning_dashboard

        # This should raise TypeError because cost_model is required
        with pytest.raises(TypeError):
            plot_planning_dashboard(synthetic_model)  # type: ignore[call-arg]


# ============================================================================
# Import Tests
# ============================================================================


class TestPlanningVizImports:
    """Verify functions are properly exported."""

    def test_import_from_visualization_module(self) -> None:
        """Functions importable from cje.visualization."""
        from cje.visualization import (
            plot_planning_dashboard,
            plot_variance_model_fit,
            plot_oracle_sensitivity,
            plot_optimality_proof,
            generate_canonical_planning_figures,
        )

        assert callable(plot_planning_dashboard)
        assert callable(plot_variance_model_fit)
        assert callable(plot_oracle_sensitivity)
        assert callable(plot_optimality_proof)
        assert callable(generate_canonical_planning_figures)


# ============================================================================
# Helper Functions
# ============================================================================


def _generate_measurements_from_model(
    model: FittedVarianceModel,
    n_grid: list[int] = [100, 200, 400],
    oracle_fractions: list[float] = [0.20, 0.35, 0.50],
) -> list[tuple[int, int, float]]:
    """Generate synthetic measurements consistent with fitted model.

    Used to create measurement data for the variance fit plot when
    actual measurements aren't available from the fit process.
    """
    measurements = []
    for n in n_grid:
        for frac in oracle_fractions:
            m = max(int(n * frac), 1)
            var = model.predict_variance(n, m)
            measurements.append((n, m, var))
    return measurements
