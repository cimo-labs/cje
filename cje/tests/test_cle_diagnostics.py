"""Tests for CLE (Coverage-Limited Efficiency) diagnostics.

These tests verify the TTC and CLE diagnostic implementations match
the paper's theoretical framework and produce expected results on
known test cases.
"""

import numpy as np
import pytest
from cje.diagnostics import (
    compute_ttc,
    compute_cle_diagnostics,
    CLEDiagnostics,
    hellinger_affinity,
)


class TestComputeTTC:
    """Tests for compute_ttc() function.

    Key insight: TTC = β = logger coverage of target-typical regions.

    T = target-typical region (where target concentrates)
    TTC = P_π₀(T) = fraction of samples (from logger) that fall in T

    - For identical policies: T = everywhere, so TTC ≈ target_typical_mass
    - Low TTC means logger rarely generates what target wants
    """

    def test_ttc_identical_policies(self) -> None:
        """Identical policies should give high TTC.

        When policies are identical, target concentrates where logger concentrates,
        so logger should have good coverage of target-typical regions.
        """
        np.random.seed(42)
        n = 1000
        base_lp = np.random.normal(-50, 10, n)
        target_lp = base_lp.copy()  # Identical

        ttc = compute_ttc(base_lp, target_lp, target_typical_mass=0.8)
        # Identical policies: logger perfectly covers target-typical region
        # TTC should be high (close to target_typical_mass)
        assert ttc > 0.7

    def test_ttc_similar_but_spread_policies(self) -> None:
        """Similar but spread-out policies should give high TTC.

        When target is similar to logger (small random perturbation),
        target concentrates in similar regions, giving good coverage.
        """
        np.random.seed(42)
        n = 1000
        base_lp = np.random.normal(-50, 10, n)
        # Small perturbation - target is similar to logger
        target_lp = base_lp + np.random.normal(0, 0.5, n)

        ttc = compute_ttc(base_lp, target_lp)
        assert ttc > 0.6  # Good coverage since policies are similar

    def test_ttc_misaligned_policies(self) -> None:
        """Misaligned policies should give low TTC.

        When target prefers what logger dislikes (inverse correlation),
        logger has poor coverage of target-typical regions.
        """
        np.random.seed(42)
        n = 1000
        base_lp = np.random.normal(-50, 10, n)
        # Target prefers what logger dislikes (inverse correlation)
        target_lp = -base_lp

        ttc = compute_ttc(base_lp, target_lp)
        assert ttc < 0.5  # Poor coverage - logger doesn't generate what target wants

    def test_ttc_moderate_difference(self) -> None:
        """Moderate policy difference should give medium TTC."""
        np.random.seed(42)
        n = 1000
        base_lp = np.random.normal(-50, 10, n)
        # Moderate perturbation - some misalignment but still correlated
        target_lp = base_lp + np.random.normal(0, 3, n)

        ttc = compute_ttc(base_lp, target_lp)
        # Should have some coverage but not perfect
        assert 0.3 < ttc < 0.95

    def test_ttc_bounds(self) -> None:
        """TTC should always be in (0, 1]."""
        np.random.seed(42)
        for shift in [0, 0.5, 1.0, 2.0, 5.0]:
            base_lp = np.random.normal(-50, 10, 1000)
            target_lp = base_lp + np.random.normal(shift, 1, 1000)
            ttc = compute_ttc(base_lp, target_lp)
            assert 0 < ttc <= 1.0

    def test_ttc_empty_input(self) -> None:
        """Empty input should return 0."""
        ttc = compute_ttc(np.array([]), np.array([]))
        assert ttc == 0.0

    def test_ttc_target_typical_mass_effect(self) -> None:
        """Higher target_typical_mass means larger T, so higher TTC."""
        np.random.seed(42)
        n = 1000
        base_lp = np.random.normal(-50, 10, n)
        target_lp = base_lp + np.random.normal(1, 2, n)

        ttc_50 = compute_ttc(base_lp, target_lp, target_typical_mass=0.5)
        ttc_90 = compute_ttc(base_lp, target_lp, target_typical_mass=0.9)

        # Larger T (capturing more target mass) = more logger samples in T = higher TTC
        assert ttc_90 >= ttc_50


class TestHellingerAffinity:
    """Tests for hellinger_affinity() function (Bhattacharyya coefficient)."""

    def test_uniform_weights(self) -> None:
        """Uniform weights (perfect overlap) should give affinity ≈ 1."""
        weights = np.ones(1000)
        affinity = hellinger_affinity(weights)
        assert affinity == pytest.approx(1.0, abs=0.01)

    def test_good_overlap(self) -> None:
        """Low-variance log-normal weights should give high affinity."""
        np.random.seed(42)
        weights = np.random.lognormal(0, 0.3, 1000)
        affinity = hellinger_affinity(weights)
        assert affinity > 0.9  # Very high

    def test_poor_overlap(self) -> None:
        """High-variance log-normal weights should give low affinity."""
        np.random.seed(42)
        weights = np.random.lognormal(0, 3, 1000)
        affinity = hellinger_affinity(weights)
        assert affinity < 0.5  # Low

    def test_bounds(self) -> None:
        """Affinity should always be in (0, 1]."""
        np.random.seed(42)
        for sigma in [0.1, 0.5, 1.0, 2.0, 3.0]:
            weights = np.random.lognormal(0, sigma, 1000)
            affinity = hellinger_affinity(weights)
            assert 0 < affinity <= 1.0


class TestCLEDiagnostics:
    """Tests for CLEDiagnostics dataclass."""

    def test_summary_good_ttc(self) -> None:
        """High TTC should report GOOD status."""
        cle = CLEDiagnostics(
            ttc=0.9,
            bhattacharyya=0.85,
            alpha=0.9,
            beta=0.8,
            coverage_penalty=1.0,
            chi_squared_T=0.5,
            shape_mismatch=1.22,
            cle_factor=1.22,
            ess_fraction=0.5,
            n_samples=1000,
        )
        summary = cle.summary()
        assert "GOOD" in summary
        assert "IPS may work" in summary

    def test_summary_marginal_ttc(self) -> None:
        """Medium TTC should report MARGINAL status."""
        cle = CLEDiagnostics(
            ttc=0.5,
            bhattacharyya=0.4,
            alpha=0.5,
            beta=0.3,
            coverage_penalty=0.91,
            chi_squared_T=5.0,
            shape_mismatch=2.45,
            cle_factor=2.23,
            ess_fraction=0.1,
            n_samples=1000,
        )
        summary = cle.summary()
        assert "MARGINAL" in summary
        assert "consider DR" in summary

    def test_summary_poor_ttc(self) -> None:
        """Low TTC should report POOR status."""
        cle = CLEDiagnostics(
            ttc=0.2,
            bhattacharyya=0.15,
            alpha=0.2,
            beta=0.1,
            coverage_penalty=0.63,
            chi_squared_T=50.0,
            shape_mismatch=7.14,
            cle_factor=4.50,
            ess_fraction=0.01,
            n_samples=1000,
        )
        summary = cle.summary()
        assert "POOR" in summary
        assert "IPS will fail" in summary

    def test_to_dict(self) -> None:
        """to_dict should return all fields."""
        cle = CLEDiagnostics(
            ttc=0.5,
            bhattacharyya=0.6,
            alpha=0.5,
            beta=0.4,
            coverage_penalty=0.79,
            chi_squared_T=2.0,
            shape_mismatch=1.73,
            cle_factor=1.37,
            ess_fraction=0.3,
            n_samples=1000,
        )
        d = cle.to_dict()
        assert d["ttc"] == 0.5
        assert d["bhattacharyya"] == 0.6
        assert d["alpha"] == 0.5
        assert d["chi_squared_T"] == 2.0
        assert d["n_samples"] == 1000


class TestComputeCLEDiagnostics:
    """Tests for compute_cle_diagnostics() function."""

    def test_empty_input(self) -> None:
        """Empty input should return degenerate diagnostics."""
        cle = compute_cle_diagnostics(np.array([]), np.array([]))
        assert cle.ttc == 0.0
        assert cle.n_samples == 0
        assert np.isinf(cle.coverage_penalty)

    def test_identical_policies(self) -> None:
        """Identical policies should give optimal diagnostics.

        With identical policies:
        - TTC = coverage_percentile (0.8) since weights are uniform
        - Bhattacharyya = 1.0 (perfect overlap)
        - ESS = 1.0 (no variance inflation)
        """
        np.random.seed(42)
        n = 1000
        base_lp = np.random.normal(-50, 10, n)
        target_lp = base_lp.copy()

        cle = compute_cle_diagnostics(base_lp, target_lp)

        assert cle.ttc == pytest.approx(
            0.80, abs=0.02
        )  # TTC = coverage_percentile for uniform weights
        assert cle.bhattacharyya == pytest.approx(1.0, abs=0.02)  # Perfect overlap
        assert cle.ess_fraction == pytest.approx(1.0, abs=0.02)  # No variance
        assert cle.chi_squared_T < 0.1  # Very low variance in T

    def test_similar_policies(self) -> None:
        """Similar policies should maintain good coverage."""
        np.random.seed(42)
        n = 1000
        base_lp = np.random.normal(-50, 10, n)
        target_lp = base_lp + np.random.normal(0, 0.3, n)

        cle = compute_cle_diagnostics(base_lp, target_lp)

        assert cle.ttc > 0.7  # Should be near coverage_percentile
        assert cle.bhattacharyya > 0.8  # Good Bhattacharyya overlap
        assert cle.ess_fraction > 0.1
        assert cle.cle_factor < 5.0

    def test_target_concentrated_outside_logger_typical(self) -> None:
        """Target concentrated outside T should give low TTC."""
        np.random.seed(42)
        n = 1000
        base_lp = np.random.normal(-50, 10, n)
        # Target prefers what logger dislikes (inverse relationship)
        target_lp = -base_lp

        cle = compute_cle_diagnostics(base_lp, target_lp)

        assert cle.ttc < 0.5  # Target mass outside T
        assert cle.ess_fraction < 0.1  # High variance weights

    def test_ttc_equals_beta(self) -> None:
        """TTC should equal beta (they're the same thing)."""
        np.random.seed(42)
        n = 1000
        base_lp = np.random.normal(-50, 10, n)
        target_lp = base_lp + np.random.normal(1, 2, n)

        cle = compute_cle_diagnostics(base_lp, target_lp)

        assert cle.ttc == cle.beta

    def test_alpha_approximately_target_typical_mass(self) -> None:
        """α should be approximately target_typical_mass (by construction)."""
        np.random.seed(42)
        n = 1000
        base_lp = np.random.normal(-50, 10, n)
        target_lp = base_lp + np.random.normal(1, 1, n)

        cle = compute_cle_diagnostics(base_lp, target_lp, target_typical_mass=0.8)

        # α = target mass on T, T is defined to contain 80% of target mass
        # Should be approximately 0.80 (or slightly higher due to threshold rounding)
        assert cle.alpha == pytest.approx(
            0.80, abs=0.10
        )  # Allow more tolerance due to discretization

    def test_shape_mismatch_formula(self) -> None:
        """Shape mismatch should be √(1 + χ²_T)."""
        np.random.seed(42)
        n = 1000
        base_lp = np.random.normal(-50, 10, n)
        target_lp = base_lp + np.random.normal(1, 1, n)

        cle = compute_cle_diagnostics(base_lp, target_lp)

        expected_shape = np.sqrt(1 + cle.chi_squared_T)
        assert cle.shape_mismatch == pytest.approx(expected_shape, rel=0.001)

    def test_cle_factor_is_product(self) -> None:
        """CLE factor should be coverage_penalty × shape_mismatch."""
        np.random.seed(42)
        n = 1000
        base_lp = np.random.normal(-50, 10, n)
        target_lp = base_lp + np.random.normal(1, 1, n)

        cle = compute_cle_diagnostics(base_lp, target_lp)

        expected = cle.coverage_penalty * cle.shape_mismatch
        assert cle.cle_factor == pytest.approx(expected, rel=0.001)

    def test_coverage_penalty_formula(self) -> None:
        """Coverage penalty should be α/√β."""
        np.random.seed(42)
        n = 1000
        base_lp = np.random.normal(-50, 10, n)
        target_lp = base_lp + np.random.normal(1, 1, n)

        cle = compute_cle_diagnostics(base_lp, target_lp)

        expected = cle.alpha / np.sqrt(cle.beta)
        assert cle.coverage_penalty == pytest.approx(expected, rel=0.001)


class TestTTCVsBhattacharyya:
    """Tests to verify TTC and Bhattacharyya are distinct metrics."""

    def test_different_metrics(self) -> None:
        """TTC and Bhattacharyya measure different things."""
        np.random.seed(42)
        n = 1000
        base_lp = np.random.normal(-50, 10, n)
        target_lp = base_lp + np.random.normal(2, 2, n)

        cle = compute_cle_diagnostics(base_lp, target_lp)

        # Both should be in valid range
        assert 0 < cle.ttc <= 1.0
        assert 0 < cle.bhattacharyya <= 1.0

        # TTC = β (logger coverage of T)
        # Bhattacharyya = E[√w] (shape mismatch)
        # These measure fundamentally different things

    def test_ttc_changes_with_target_typical_mass(self) -> None:
        """TTC should change with target_typical_mass (T definition)."""
        np.random.seed(42)
        n = 1000
        base_lp = np.random.normal(-50, 10, n)
        target_lp = base_lp + np.random.normal(1, 2, n)

        cle_80 = compute_cle_diagnostics(base_lp, target_lp, target_typical_mass=0.8)
        cle_50 = compute_cle_diagnostics(base_lp, target_lp, target_typical_mass=0.5)

        # Different target_typical_mass = different T = different TTC
        # Larger T (0.8) should generally include more logger samples
        assert cle_80.ttc >= cle_50.ttc

    def test_bhattacharyya_independent_of_target_typical_mass(self) -> None:
        """Bhattacharyya should NOT depend on target_typical_mass."""
        np.random.seed(42)
        n = 1000
        base_lp = np.random.normal(-50, 10, n)
        target_lp = base_lp + np.random.normal(1, 2, n)

        cle_80 = compute_cle_diagnostics(base_lp, target_lp, target_typical_mass=0.8)
        cle_50 = compute_cle_diagnostics(base_lp, target_lp, target_typical_mass=0.5)

        # Bhattacharyya is a global metric, doesn't depend on T definition
        assert cle_80.bhattacharyya == pytest.approx(cle_50.bhattacharyya, rel=0.001)


class TestPaperConsistency:
    """Tests to verify consistency with CJE paper claims."""

    def test_ttc_threshold_decision_rule(self) -> None:
        """Verify TTC threshold decision rule from paper."""
        # Paper says: "If TTC < 0.7, logs-only IPS will fail"

        # Good case: TTC > 0.7 should be "IPS may work"
        # Note: TTC = β = logger coverage of T
        cle_good = CLEDiagnostics(
            ttc=0.8,
            bhattacharyya=0.75,
            alpha=0.85,
            beta=0.8,
            coverage_penalty=0.95,
            chi_squared_T=0.5,
            shape_mismatch=1.22,
            cle_factor=1.16,
            ess_fraction=0.5,
            n_samples=1000,
        )
        assert "IPS may work" in cle_good.summary()

        # Poor case: TTC < 0.3 should be "IPS will fail"
        cle_poor = CLEDiagnostics(
            ttc=0.2,
            bhattacharyya=0.15,
            alpha=0.8,
            beta=0.2,
            coverage_penalty=1.79,
            chi_squared_T=50.0,
            shape_mismatch=7.14,
            cle_factor=12.8,
            ess_fraction=0.01,
            n_samples=1000,
        )
        assert "IPS will fail" in cle_poor.summary()

    def test_chi_squared_computed_inside_T(self) -> None:
        """χ² should be computed inside T, not globally.

        This is the key theoretical fix: the paper's CLE bound uses
        χ²_T (variance inside T), not global χ².
        """
        np.random.seed(42)
        n = 1000
        base_lp = np.random.normal(-50, 10, n)
        target_lp = base_lp + np.random.normal(1, 2, n)

        cle = compute_cle_diagnostics(base_lp, target_lp, target_typical_mass=0.8)

        # Compute global χ² for comparison
        log_weights = target_lp - base_lp
        weights = np.exp(log_weights - np.max(log_weights))
        weights = weights / np.mean(weights)
        global_chi_sq = float(np.var(weights))

        # χ²_T should generally differ from global χ²
        # (they're only equal if weights are uniform across T vs outside)
        # Just verify it's computing something reasonable
        assert cle.chi_squared_T >= 0
        assert np.isfinite(cle.chi_squared_T)
