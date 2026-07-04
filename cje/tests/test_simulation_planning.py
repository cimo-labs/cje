"""Tests for simulation-based label planning interface.

Note: Tests that run actual simulations are marked with @pytest.mark.slow
since each simulation takes ~30-60 seconds. Run with `pytest -m "not slow"`
to skip them during normal development.
"""

import pytest
import numpy as np
from types import SimpleNamespace
from typing import Any, List, Tuple, cast

from cje import (
    simulate_variance_model,
    simulate_planning,
    simulate_planning_sweep,
    correlation_to_r2,
    SimulationPlanningResult,
    CostModel,
    FittedVarianceModel,
    plan_evaluation,
    plan_for_mde,
)
from cje.diagnostics.simulation_planning import (
    _generate_synthetic_data,
)


def _fake_analyze_dataset_factory(
    capture: List[Tuple[int, int]],
) -> Any:
    """Fake analyze_dataset reporting SE from the sizes it actually receives.

    With SE^2 = 0.05/n + 0.02/m, NNLS recovers (0.05, 0.02) exactly ONLY if
    the recorded measurement grid matches what analyze_dataset received.
    """

    def fake_analyze_dataset(**kwargs: Any) -> Any:
        records = next(iter(kwargs["fresh_draws_data"].values()))
        n_recv = len(records)
        m_recv = sum(1 for r in records if "oracle_label" in r)
        capture.append((n_recv, m_recv))
        se = float(np.sqrt(0.05 / n_recv + 0.02 / max(m_recv, 1)))
        return SimpleNamespace(standard_errors=[se])

    return fake_analyze_dataset


class TestCorrelationToR2:
    """Tests for correlation_to_r2 helper function."""

    def test_linear_relationship(self) -> None:
        """Linear relationship: R² = r²."""
        assert correlation_to_r2(0.7) == pytest.approx(0.49)
        assert correlation_to_r2(0.5) == pytest.approx(0.25)
        assert correlation_to_r2(1.0) == pytest.approx(1.0)
        assert correlation_to_r2(0.0) == pytest.approx(0.0)

    def test_negative_correlation(self) -> None:
        """Negative correlation: R² = r² (same as positive)."""
        assert correlation_to_r2(-0.7) == pytest.approx(0.49)
        assert correlation_to_r2(-1.0) == pytest.approx(1.0)

    def test_monotone_relationship_higher_r2(self) -> None:
        """Monotone nonlinear relationships have higher isotonic R²."""
        r2_linear = correlation_to_r2(0.7, "linear")
        r2_monotone = correlation_to_r2(0.7, "monotone")
        assert r2_monotone > r2_linear

    def test_monotone_capped_at_1(self) -> None:
        """Monotone R² is capped at 1.0."""
        r2 = correlation_to_r2(1.0, "monotone")
        assert r2 == 1.0

    def test_invalid_correlation_raises(self) -> None:
        """Correlation outside [-1, 1] raises ValueError."""
        with pytest.raises(ValueError, match="Correlation must be in"):
            correlation_to_r2(1.5)
        with pytest.raises(ValueError, match="Correlation must be in"):
            correlation_to_r2(-1.1)

    def test_invalid_relationship_raises(self) -> None:
        """Invalid relationship type raises ValueError."""
        with pytest.raises(ValueError, match="relationship must be"):
            correlation_to_r2(0.7, "quadratic")


class TestGenerateSyntheticData:
    """Tests for synthetic data generation."""

    def test_correct_size(self) -> None:
        """Generates correct number of samples."""
        data = _generate_synthetic_data(
            n_total=100, oracle_fraction=0.3, r2=0.7, seed=42
        )
        assert len(data) == 100

    def test_correct_oracle_fraction(self) -> None:
        """Correct fraction of samples have oracle labels."""
        data = _generate_synthetic_data(
            n_total=1000, oracle_fraction=0.4, r2=0.7, seed=42
        )
        n_with_oracle = sum(1 for r in data if "oracle_label" in r)
        assert n_with_oracle == 400

    def test_judge_scores_in_range(self) -> None:
        """Judge scores are in [0, 1]."""
        data = _generate_synthetic_data(
            n_total=100, oracle_fraction=0.3, r2=0.7, seed=42
        )
        scores = [r["judge_score"] for r in data]
        assert all(0 <= s <= 1 for s in scores)

    def test_oracle_labels_binary(self) -> None:
        """Oracle labels are binary (0 or 1)."""
        data = _generate_synthetic_data(
            n_total=100, oracle_fraction=0.5, r2=0.7, seed=42
        )
        labels = [r["oracle_label"] for r in data if "oracle_label" in r]
        assert all(label in [0, 1] for label in labels)

    def test_unique_prompt_ids(self) -> None:
        """All prompt_ids are unique."""
        data = _generate_synthetic_data(
            n_total=100, oracle_fraction=0.3, r2=0.7, seed=42
        )
        prompt_ids = [r["prompt_id"] for r in data]
        assert len(prompt_ids) == len(set(prompt_ids))

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces same data."""
        data1 = _generate_synthetic_data(
            n_total=100, oracle_fraction=0.3, r2=0.7, seed=42
        )
        data2 = _generate_synthetic_data(
            n_total=100, oracle_fraction=0.3, r2=0.7, seed=42
        )
        assert data1 == data2

    def test_different_seeds_different_data(self) -> None:
        """Different seeds produce different data."""
        data1 = _generate_synthetic_data(
            n_total=100, oracle_fraction=0.3, r2=0.7, seed=42
        )
        data2 = _generate_synthetic_data(
            n_total=100, oracle_fraction=0.3, r2=0.7, seed=43
        )
        scores1 = [r["judge_score"] for r in data1]
        scores2 = [r["judge_score"] for r in data2]
        assert scores1 != scores2


class TestSyntheticDataR2Calibration:
    """FIX 7 regression tests: the generator delivers the requested isotonic R².

    The naive mixing weight targeted the continuous oracle while labels are
    binarized at 0.5, attenuating realized isotonic R² 15-25% below the
    request (e.g. request 0.5 -> realized 0.39). The mixing weight is now
    calibrated numerically against the binarized labels.
    """

    @pytest.mark.parametrize("r2", [0.5, 0.7])
    def test_realized_isotonic_r2_matches_request(self, r2: float) -> None:
        """Realized isotonic R² of binarized labels vs judge is within ±0.05."""
        from sklearn.isotonic import IsotonicRegression

        data = _generate_synthetic_data(
            n_total=20000, oracle_fraction=1.0, r2=r2, seed=123
        )
        judge = np.array([d["judge_score"] for d in data])
        labels = np.array([float(d["oracle_label"]) for d in data])

        iso = IsotonicRegression(out_of_bounds="clip")
        pred = iso.fit(judge, labels).predict(judge)
        ss_res = float(np.sum((labels - pred) ** 2))
        ss_tot = float(np.sum((labels - labels.mean()) ** 2))
        realized = 1 - ss_res / ss_tot

        assert (
            abs(realized - r2) <= 0.05
        ), f"requested r2={r2}, realized isotonic R2={realized:.3f}"


class TestSimulateGridAndFitterUnification:
    """FIX 6 regression tests: shared grid derivation + shared NNLS fitter."""

    def _run_simulate_with_fake_analyze(
        self, oracle_fraction: float, capture: List[Tuple[int, int]]
    ) -> FittedVarianceModel:
        import cje.interface.analysis as analysis_module

        fake_analyze_dataset = _fake_analyze_dataset_factory(capture)
        original = analysis_module.analyze_dataset
        setattr(analysis_module, "analyze_dataset", fake_analyze_dataset)
        try:
            return simulate_variance_model(
                r2=0.7,
                n_total=1000,
                oracle_fraction=oracle_fraction,
                n_replicates=2,
                seed=42,
                verbose=False,
            )
        finally:
            setattr(analysis_module, "analyze_dataset", original)

    def test_full_oracle_fraction_grid_is_actually_delivered(self) -> None:
        """Regression: at oracle_fraction=1.0 the old pool-based grid recorded
        (n, m) pairs that analyze_dataset never actually received (there was
        no unlabeled pool to supply the n - m judge-only records)."""
        capture: List[Tuple[int, int]] = []
        model = self._run_simulate_with_fake_analyze(1.0, capture)

        received = sorted(set(capture))
        unique_n = sorted(set(n for n, m in received))
        unique_m = sorted(set(m for n, m in received))

        # A valid orthogonal grid was actually delivered to the pipeline
        assert len(unique_n) >= 2, f"Need >=2 unique n, got {unique_n}"
        assert len(unique_m) >= 2, f"Need >=2 unique m, got {unique_m}"
        assert all(m < n for n, m in received), f"All m must be < n: {received}"

        # Exact recovery of the generating components proves the recorded
        # grid matches what analyze_dataset received.
        assert model.sigma2_eval == pytest.approx(0.05, rel=1e-3)
        assert model.sigma2_cal == pytest.approx(0.02, rel=1e-3)

    def test_simulate_uses_shared_fitter(self) -> None:
        """The duplicated NNLS fitter is gone; simulate produces the identical
        FittedVarianceModel as calling the shared fitter on the same
        measurements."""
        import inspect
        import cje.diagnostics.simulation_planning as sim_module
        from cje.diagnostics.planning import _fit_variance_model_from_measurements

        # The copy-pasted fitter (inline nnls) is deleted
        src = inspect.getsource(sim_module.simulate_variance_model)
        assert "nnls" not in src

        captured_measurements: List[List[Tuple[int, int, float]]] = []
        real_fitter = _fit_variance_model_from_measurements

        def spy_fitter(
            measurements: List[Tuple[int, int, float]],
        ) -> FittedVarianceModel:
            captured_measurements.append(list(measurements))
            return real_fitter(measurements)

        original_fitter = sim_module._fit_variance_model_from_measurements
        setattr(sim_module, "_fit_variance_model_from_measurements", spy_fitter)
        try:
            capture: List[Tuple[int, int]] = []
            model = self._run_simulate_with_fake_analyze(0.4, capture)
        finally:
            setattr(
                sim_module, "_fit_variance_model_from_measurements", original_fitter
            )

        assert len(captured_measurements) == 1
        assert model == real_fitter(captured_measurements[0])


class TestSimulateVarianceModel:
    """Tests for simulate_variance_model (core primitive).

    Note: These tests run actual simulations and are slow (~30-60s each).
    """

    def test_invalid_r2_raises(self) -> None:
        """R² outside [0, 1] raises ValueError (no simulation needed)."""
        with pytest.raises(ValueError, match="r2 must be in"):
            simulate_variance_model(r2=-0.1)
        with pytest.raises(ValueError, match="r2 must be in"):
            simulate_variance_model(r2=1.5)

    @pytest.mark.slow
    def test_returns_fitted_variance_model(self) -> None:
        """Returns a FittedVarianceModel with positive variance components."""
        model = simulate_variance_model(r2=0.7, verbose=False)
        assert isinstance(model, FittedVarianceModel)
        assert model.sigma2_eval >= 0
        assert model.sigma2_cal >= 0
        assert model.sigma2_eval + model.sigma2_cal > 0
        assert model.n_measurements > 0  # Simulation produces real measurements

    @pytest.mark.slow
    def test_compatible_with_plan_evaluation(self) -> None:
        """Variance model works with plan_evaluation."""
        cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
        model = simulate_variance_model(r2=0.7, verbose=False)

        plan = plan_evaluation(budget=5000, variance_model=model, cost_model=cost)

        assert plan.n_samples > 0
        assert plan.m_oracle > 0
        assert plan.mde > 0

    @pytest.mark.slow
    def test_compatible_with_plan_for_mde(self) -> None:
        """Variance model works with plan_for_mde."""
        cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
        model = simulate_variance_model(r2=0.7, verbose=False)

        plan = plan_for_mde(target_mde=0.05, variance_model=model, cost_model=cost)

        assert plan.n_samples > 0
        assert plan.m_oracle > 0

    @pytest.mark.slow
    def test_matches_fit_variance_model_interface(self) -> None:
        """Has same interface as fit_variance_model output."""
        model = simulate_variance_model(r2=0.7, verbose=False)

        # Should have all the same attributes
        assert hasattr(model, "sigma2_eval")
        assert hasattr(model, "sigma2_cal")
        assert hasattr(model, "r_squared")
        assert hasattr(model, "n_measurements")
        assert hasattr(model, "predict_variance")
        assert hasattr(model, "predict_se")

        # predict_variance should work
        var = model.predict_variance(n=1000, m=100)
        assert var > 0


class TestSimulatePlanning:
    """Tests for simulate_planning function.

    Note: These tests run actual simulations and are slow (~30-60s each).
    """

    def test_invalid_r2_raises(self) -> None:
        """R² outside [0, 1] raises ValueError (no simulation needed)."""
        cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)

        with pytest.raises(ValueError, match="r2 must be in"):
            simulate_planning(r2=-0.1, budget=5000, cost_model=cost)

        with pytest.raises(ValueError, match="r2 must be in"):
            simulate_planning(r2=1.5, budget=5000, cost_model=cost)

    @pytest.mark.slow
    def test_returns_valid_result(self) -> None:
        """Returns valid SimulationPlanningResult."""
        cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
        result = simulate_planning(r2=0.7, budget=5000, cost_model=cost, verbose=False)

        assert isinstance(result, SimulationPlanningResult)
        assert result.r2 == 0.7
        assert result.plan.n_samples > 0
        assert result.plan.m_oracle > 0
        assert result.plan.mde > 0

    @pytest.mark.slow
    def test_variance_fractions_sum_to_one(self) -> None:
        """Variance fractions sum to 1.0."""
        cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
        result = simulate_planning(r2=0.7, budget=5000, cost_model=cost, verbose=False)

        total = result.eval_variance_fraction + result.cal_variance_fraction
        assert total == pytest.approx(1.0)

    @pytest.mark.slow
    def test_summary_contains_key_info(self) -> None:
        """Summary string contains key information."""
        cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
        result = simulate_planning(r2=0.7, budget=5000, cost_model=cost, verbose=False)
        summary = result.summary()

        assert "R² = 0.70" in summary
        assert "Evaluation Plan" in summary
        assert "MDE" in summary

    @pytest.mark.slow
    def test_explain_provides_educational_content(self) -> None:
        """Explain method provides educational content."""
        cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
        result = simulate_planning(r2=0.7, budget=5000, cost_model=cost, verbose=False)
        explanation = result.explain()

        assert "good" in explanation  # R²=0.7 is "good" quality
        assert "variance" in explanation.lower()
        assert "oracle" in explanation.lower()


class TestSimulatePlanningSweep:
    """Tests for simulate_planning_sweep function.

    Note: Sweeps run multiple simulations and are very slow.
    """

    def test_empty_r2_list(self) -> None:
        """Empty R² list returns empty results (no simulation needed)."""
        cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
        results = simulate_planning_sweep([], budget=5000, cost_model=cost)
        assert results == []

    @pytest.mark.slow
    def test_returns_correct_number_of_results(self) -> None:
        """Returns one result per R² value."""
        cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
        r2_values = [0.5, 0.7]  # Just 2 values to keep test reasonable

        results = simulate_planning_sweep(
            r2_values, budget=5000, cost_model=cost, verbose=False
        )

        assert len(results) == len(r2_values)

    @pytest.mark.slow
    def test_results_correspond_to_r2_values(self) -> None:
        """Each result has correct R² value."""
        cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
        r2_values = [0.5, 0.7]

        results = simulate_planning_sweep(
            r2_values, budget=5000, cost_model=cost, verbose=False
        )

        for r2, result in zip(r2_values, results):
            assert result.r2 == r2


class TestTopLevelImports:
    """Test that new API is accessible from top-level cje package."""

    def test_import_simulate_variance_model(self) -> None:
        """simulate_variance_model is importable from cje."""
        from cje import simulate_variance_model

        assert callable(simulate_variance_model)

    def test_import_simulate_planning(self) -> None:
        """simulate_planning is importable from cje."""
        from cje import simulate_planning

        assert callable(simulate_planning)

    def test_import_simulate_planning_sweep(self) -> None:
        """simulate_planning_sweep is importable from cje."""
        from cje import simulate_planning_sweep

        assert callable(simulate_planning_sweep)

    def test_import_correlation_to_r2(self) -> None:
        """correlation_to_r2 is importable from cje."""
        from cje import correlation_to_r2

        assert callable(correlation_to_r2)

    def test_import_simulation_planning_result(self) -> None:
        """SimulationPlanningResult is importable from cje."""
        from cje import SimulationPlanningResult

        assert SimulationPlanningResult is not None


class TestIntegrationWithExistingPlanning:
    """Test integration with existing planning infrastructure."""

    @pytest.mark.slow
    def test_simulate_variance_model_with_plan_evaluation(self) -> None:
        """simulate_variance_model + plan_evaluation = composable workflow."""
        cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)

        # Core primitive workflow (parallels pilot-based workflow)
        variance_model = simulate_variance_model(r2=0.7, verbose=False)
        plan = plan_evaluation(
            budget=5000, variance_model=variance_model, cost_model=cost
        )

        assert plan.n_samples > 0
        assert plan.m_oracle > 0
        assert plan.mde > 0

    @pytest.mark.slow
    def test_simulate_variance_model_with_plan_for_mde(self) -> None:
        """simulate_variance_model + plan_for_mde = composable workflow."""
        cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)

        variance_model = simulate_variance_model(r2=0.7, verbose=False)
        plan = plan_for_mde(
            target_mde=0.05, variance_model=variance_model, cost_model=cost
        )

        assert plan.n_samples > 0

    @pytest.mark.slow
    def test_plan_has_correct_variance_components(self) -> None:
        """Plan includes correct variance components from simulation."""
        cost = CostModel(surrogate_cost=0.01, oracle_cost=0.16)
        result = simulate_planning(r2=0.7, budget=5000, cost_model=cost, verbose=False)

        # Plan should have variance components
        assert result.plan.sigma2_eval >= 0
        assert result.plan.sigma2_cal >= 0

        # Should match variance model
        assert result.plan.sigma2_eval == result.variance_model.sigma2_eval
        assert result.plan.sigma2_cal == result.variance_model.sigma2_cal
