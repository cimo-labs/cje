"""Tests for planning dashboard visualization.

Tests verify the planning dashboard runs without error and returns valid figures.

IMPORTANT: cost_model is REQUIRED - there are no meaningful defaults.
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
    """E2E tests using real arena data."""

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

        # Skip if variance model is degenerate (rare edge case with sparse data)
        if variance_model.sigma2_eval == 0 or variance_model.sigma2_cal == 0:
            pytest.skip("Degenerate variance model - skipping visualization test")

        # Explicit cost model required
        cost_model = CostModel(oracle_cost=16.0)

        # Generate dashboard - budget range computed adaptively
        fig = plot_planning_dashboard(variance_model, cost_model)

        # Validate structure
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3  # 3 panels

        plt.close(fig)


# ============================================================================
# Smoke Tests: Quick Validation with Synthetic Data
# ============================================================================


class TestPlanningVisualizationSmoke:
    """Fast smoke tests with synthetic data for CI."""

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

    def test_dashboard_runs_without_error(
        self,
        synthetic_model: FittedVarianceModel,
        cost_model: CostModel,
    ) -> None:
        """Dashboard executes without raising."""
        from cje.visualization.planning import plot_planning_dashboard

        fig = plot_planning_dashboard(synthetic_model, cost_model)
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

    def test_custom_budget_range(
        self, synthetic_model: FittedVarianceModel, cost_model: CostModel
    ) -> None:
        """Dashboard accepts custom budget range."""
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
        """Dashboard works with different cost ratios."""
        from cje.visualization.planning import plot_planning_dashboard

        # Cheap oracle (4×)
        fig1 = plot_planning_dashboard(synthetic_model, CostModel(oracle_cost=4.0))
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        # Expensive oracle (64×)
        fig2 = plot_planning_dashboard(synthetic_model, CostModel(oracle_cost=64.0))
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

    def test_cost_model_is_required(self, synthetic_model: FittedVarianceModel) -> None:
        """Dashboard requires explicit cost_model (no defaults)."""
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
        """plot_planning_dashboard importable from cje.visualization."""
        from cje.visualization import plot_planning_dashboard

        assert callable(plot_planning_dashboard)
