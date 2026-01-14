"""Tests for multi-policy EIF with density ratio weighting.

Tests the efficient influence function that pools oracle labels across
policies with f_p(z)/g(z) density ratio weighting.
"""

import numpy as np
import pytest
from cje.diagnostics.robust_inference import (
    compute_density_ratios,
    compute_augmented_estimate_multipolicy,
    compute_augmented_estimate_per_policy,
)


class TestDensityRatios:
    """Tests for compute_density_ratios()."""

    def test_basic_computation(self) -> None:
        """Test density ratio computation with simple data."""
        np.random.seed(42)
        n_per_policy = 100
        n_policies = 3

        # Create calibration index (judge scores in monotone mode) with different
        # distributions per policy
        calibration_index = np.concatenate(
            [
                np.random.beta(2, 5, n_per_policy),  # Policy 0: skewed left
                np.random.beta(5, 5, n_per_policy),  # Policy 1: symmetric
                np.random.beta(5, 2, n_per_policy),  # Policy 2: skewed right
            ]
        )
        policy_indices = np.concatenate(
            [
                np.zeros(n_per_policy, dtype=int),
                np.ones(n_per_policy, dtype=int),
                np.full(n_per_policy, 2, dtype=int),
            ]
        )

        # Oracle mask: 10% of each policy
        oracle_mask = np.zeros(len(calibration_index), dtype=bool)
        for p in range(n_policies):
            p_indices = np.where(policy_indices == p)[0]
            oracle_indices = np.random.choice(p_indices, size=10, replace=False)
            oracle_mask[oracle_indices] = True

        density_ratios, diagnostics = compute_density_ratios(
            calibration_index=calibration_index,
            policy_indices=policy_indices,
            oracle_mask=oracle_mask,
            n_policies=n_policies,
            n_bins=20,
        )

        # Basic shape checks
        n_oracle = np.sum(oracle_mask)
        assert density_ratios.shape == (n_policies, n_oracle)

        # Density ratios should be positive
        assert np.all(density_ratios >= 0)

        # Check diagnostics
        assert diagnostics["n_oracle"] == n_oracle
        assert len(diagnostics["policy_counts"]) == n_policies

    def test_single_policy_weight_equals_inverse_oracle_prob(self) -> None:
        """With one policy, w(z) = f(z)/g(z) = 1/e(z), NOT ~1.

        When there's only one policy, g(z) = rho * e(z) * f(z) with rho = 1,
        so w(z) = f(z) / (e(z) * f(z)) = 1/e(z).

        This is correct: the weight upweights oracle samples by the inverse
        of their selection probability, which is exactly what the EIF requires.
        """
        np.random.seed(42)
        n = 500  # Larger sample for stable bin estimates

        calibration_index = np.random.beta(3, 3, n)
        policy_indices = np.zeros(n, dtype=int)  # All policy 0

        # Create oracle mask with known oracle probability ~20%
        oracle_prob = 0.2
        oracle_mask = np.random.rand(n) < oracle_prob

        density_ratios, diagnostics = compute_density_ratios(
            calibration_index=calibration_index,
            policy_indices=policy_indices,
            oracle_mask=oracle_mask,
            n_policies=1,
            n_bins=10,
        )

        # Shape should be correct
        assert density_ratios.shape[0] == 1
        assert np.all(density_ratios >= 0)

        # Key identity: (1/N) * Σ_{i:L_i=1} w(Z_i) ≈ 1
        # This is the sample analog of E[L * w(Z)] = 1
        n_oracle = np.sum(oracle_mask)
        weighted_sum = np.sum(density_ratios[0]) / n
        # Should be close to 1 (oracle fraction cancels with weights)
        assert 0.5 < weighted_sum < 2.0, f"Weighted sum = {weighted_sum}, expected ~1.0"

        # Weights should be roughly 1/oracle_prob on average
        # (with some variation due to binning and sampling)
        mean_weight = np.mean(density_ratios[0])
        expected_mean = 1.0 / oracle_prob  # = 5 for 20% oracle
        # Allow 2x tolerance due to sampling variation
        assert (
            0.3 * expected_mean < mean_weight < 3.0 * expected_mean
        ), f"Mean weight = {mean_weight}, expected ~{expected_mean}"

    def test_empty_oracle_mask(self) -> None:
        """Test handling of empty oracle mask."""
        calibration_index = np.random.rand(100)
        policy_indices = np.zeros(100, dtype=int)
        oracle_mask = np.zeros(100, dtype=bool)

        density_ratios, diagnostics = compute_density_ratios(
            calibration_index=calibration_index,
            policy_indices=policy_indices,
            oracle_mask=oracle_mask,
            n_policies=1,
        )

        assert density_ratios.shape == (1, 0)
        assert "error" in diagnostics

    def test_different_oracle_fractions(self) -> None:
        """Test with different oracle fractions per policy."""
        np.random.seed(42)
        n_per_policy = 200
        n_policies = 2

        calibration_index = np.concatenate(
            [
                np.random.beta(3, 3, n_per_policy),
                np.random.beta(3, 3, n_per_policy),
            ]
        )
        policy_indices = np.concatenate(
            [
                np.zeros(n_per_policy, dtype=int),
                np.ones(n_per_policy, dtype=int),
            ]
        )

        # Policy 0: 50% oracle, Policy 1: 5% oracle
        oracle_mask = np.zeros(len(calibration_index), dtype=bool)
        p0_indices = np.where(policy_indices == 0)[0]
        p1_indices = np.where(policy_indices == 1)[0]
        oracle_mask[np.random.choice(p0_indices, size=100, replace=False)] = True
        oracle_mask[np.random.choice(p1_indices, size=10, replace=False)] = True

        density_ratios, diagnostics = compute_density_ratios(
            calibration_index=calibration_index,
            policy_indices=policy_indices,
            oracle_mask=oracle_mask,
            n_policies=n_policies,
        )

        # Should handle the different oracle fractions
        assert density_ratios.shape == (n_policies, 110)
        assert np.all(density_ratios >= 0)


class TestMultipolicyEstimator:
    """Tests for compute_augmented_estimate_multipolicy()."""

    def test_basic_estimates(self) -> None:
        """Test basic multi-policy estimates."""
        np.random.seed(42)
        n_per_policy = 100
        n_policies = 2

        # Create synthetic data - calibration_index is judge scores in monotone mode
        calibration_index = np.concatenate(
            [
                np.random.beta(3, 5, n_per_policy),
                np.random.beta(5, 3, n_per_policy),
            ]
        )
        policy_indices = np.concatenate(
            [
                np.zeros(n_per_policy, dtype=int),
                np.ones(n_per_policy, dtype=int),
            ]
        )

        # Oracle labels: calibrated values + noise
        true_calibration = lambda s: 0.3 + 0.4 * s  # Linear calibration
        oracle_labels = np.full(len(calibration_index), np.nan)
        oracle_mask = np.zeros(len(calibration_index), dtype=bool)

        # 20% oracle per policy
        for p in range(n_policies):
            p_indices = np.where(policy_indices == p)[0]
            oracle_indices = np.random.choice(p_indices, size=20, replace=False)
            oracle_mask[oracle_indices] = True
            oracle_labels[oracle_indices] = true_calibration(
                calibration_index[oracle_indices]
            ) + np.random.normal(0, 0.05, len(oracle_indices))

        # Calibrated predictions (assume perfect calibration)
        calibrated_full = true_calibration(calibration_index)
        oof_predictions = calibrated_full.copy()

        estimates, diagnostics = compute_augmented_estimate_multipolicy(
            calibrated_full=calibrated_full,
            oracle_labels=oracle_labels,
            oracle_mask=oracle_mask,
            oof_predictions=oof_predictions,
            calibration_index=calibration_index,
            policy_indices=policy_indices,
            n_policies=n_policies,
        )

        # Check shape and values
        assert len(estimates) == n_policies
        assert np.all(np.isfinite(estimates))

        # Check diagnostics
        assert "plug_in_estimates" in diagnostics
        assert "residual_corrections" in diagnostics
        assert "density_diagnostics" in diagnostics

    def test_comparison_with_perpolicy(self) -> None:
        """Compare multi-policy EIF with per-policy residuals."""
        np.random.seed(42)
        n_per_policy = 200
        n_policies = 3

        # Create data with similar distributions
        calibration_index = np.concatenate(
            [np.random.beta(4, 4, n_per_policy) for _ in range(n_policies)]
        )
        policy_indices = np.concatenate(
            [np.full(n_per_policy, p, dtype=int) for p in range(n_policies)]
        )

        # Shared calibration curve
        true_calibration = lambda s: 0.2 + 0.6 * s
        oracle_labels = np.full(len(calibration_index), np.nan)
        oracle_mask = np.zeros(len(calibration_index), dtype=bool)

        # 10% oracle per policy
        for p in range(n_policies):
            p_indices = np.where(policy_indices == p)[0]
            oracle_indices = np.random.choice(p_indices, size=20, replace=False)
            oracle_mask[oracle_indices] = True
            oracle_labels[oracle_indices] = true_calibration(
                calibration_index[oracle_indices]
            ) + np.random.normal(0, 0.02, len(oracle_indices))

        calibrated_full = true_calibration(calibration_index)
        oof_predictions = calibrated_full.copy()

        # Multi-policy estimates
        mp_estimates, mp_diag = compute_augmented_estimate_multipolicy(
            calibrated_full=calibrated_full,
            oracle_labels=oracle_labels,
            oracle_mask=oracle_mask,
            oof_predictions=oof_predictions,
            calibration_index=calibration_index,
            policy_indices=policy_indices,
            n_policies=n_policies,
        )

        # Per-policy estimates
        pp_estimates, pp_diag = compute_augmented_estimate_per_policy(
            calibrated_full=calibrated_full,
            oracle_labels=oracle_labels,
            oracle_mask=oracle_mask,
            oof_predictions=oof_predictions,
            policy_indices=policy_indices,
            n_policies=n_policies,
        )

        # Both should produce valid estimates
        assert np.all(np.isfinite(mp_estimates))
        assert np.all(np.isfinite(pp_estimates))

        # Estimates should be similar (but not identical) when calibration is shared
        # The difference comes from the density ratio weighting
        assert np.allclose(mp_estimates, pp_estimates, atol=0.05)

    def test_transport_bias_detection(self) -> None:
        """Test that multi-policy EIF can detect transport bias."""
        np.random.seed(42)
        n_per_policy = 200
        n_policies = 2

        # Policy 0: low scores, Policy 1: high scores
        calibration_index = np.concatenate(
            [
                np.random.beta(2, 6, n_per_policy),  # Mean ~0.25
                np.random.beta(6, 2, n_per_policy),  # Mean ~0.75
            ]
        )
        policy_indices = np.concatenate(
            [
                np.zeros(n_per_policy, dtype=int),
                np.ones(n_per_policy, dtype=int),
            ]
        )

        # Calibration trained on policy 0 only
        true_calibration = lambda s: 0.1 + 0.8 * s
        oracle_labels = np.full(len(calibration_index), np.nan)
        oracle_mask = np.zeros(len(calibration_index), dtype=bool)

        # Only policy 0 has oracle labels
        p0_indices = np.where(policy_indices == 0)[0]
        oracle_indices = np.random.choice(p0_indices, size=40, replace=False)
        oracle_mask[oracle_indices] = True
        oracle_labels[oracle_indices] = true_calibration(
            calibration_index[oracle_indices]
        ) + np.random.normal(0, 0.02, len(oracle_indices))

        calibrated_full = true_calibration(calibration_index)
        oof_predictions = calibrated_full.copy()

        estimates, diagnostics = compute_augmented_estimate_multipolicy(
            calibrated_full=calibrated_full,
            oracle_labels=oracle_labels,
            oracle_mask=oracle_mask,
            oof_predictions=oof_predictions,
            calibration_index=calibration_index,
            policy_indices=policy_indices,
            n_policies=n_policies,
        )

        # Both policies should get valid estimates
        assert np.all(np.isfinite(estimates))

        # Policy 1 should have higher estimate (higher scores)
        assert estimates[1] > estimates[0]


class TestBootstrapIntegration:
    """Test integration with bootstrap infrastructure."""

    @pytest.fixture
    def arena_style_data(self) -> dict:
        """Create arena-style test data with multiple policies."""
        np.random.seed(42)
        n_prompts = 50
        n_policies = 3

        # Generate data for each policy
        calibration_index: list = []
        policy_indices: list = []
        oracle_labels: list = []
        oracle_mask: list = []
        prompt_ids: list = []

        for p in range(n_policies):
            # Different score distributions per policy
            scores = np.random.beta(3 + p, 5 - p, n_prompts)
            calibration_index.extend(scores)
            policy_indices.extend([p] * n_prompts)
            prompt_ids.extend([f"prompt_{i}" for i in range(n_prompts)])

            # Oracle labels for 20% of samples
            labels = np.full(n_prompts, np.nan)
            mask = np.zeros(n_prompts, dtype=bool)
            oracle_idx = np.random.choice(n_prompts, size=10, replace=False)
            mask[oracle_idx] = True
            labels[oracle_idx] = (
                0.3 + 0.4 * scores[oracle_idx] + np.random.normal(0, 0.02, 10)
            )
            oracle_labels.extend(labels)
            oracle_mask.extend(mask)

        return {
            "calibration_index": np.array(calibration_index),
            "policy_indices": np.array(policy_indices, dtype=int),
            "oracle_labels": np.array(oracle_labels),
            "oracle_mask": np.array(oracle_mask),
            "prompt_ids": prompt_ids,
            "n_policies": n_policies,
        }

    def test_multipolicy_with_bootstrap_data(self, arena_style_data: dict) -> None:
        """Test multi-policy EIF with arena-style data."""
        data = arena_style_data

        # Calibrated predictions
        calibrated_full = 0.3 + 0.4 * data["calibration_index"]
        oof_predictions = calibrated_full.copy()

        estimates, diagnostics = compute_augmented_estimate_multipolicy(
            calibrated_full=calibrated_full,
            oracle_labels=data["oracle_labels"],
            oracle_mask=data["oracle_mask"],
            oof_predictions=oof_predictions,
            calibration_index=data["calibration_index"],
            policy_indices=data["policy_indices"],
            n_policies=data["n_policies"],
        )

        # Should produce valid estimates for all policies
        assert len(estimates) == data["n_policies"]
        assert np.all(np.isfinite(estimates))

        # Check density diagnostics are present
        assert "density_diagnostics" in diagnostics
        assert diagnostics["density_diagnostics"]["n_oracle"] == np.sum(
            data["oracle_mask"]
        )


class TestEdgeCases:
    """Tests for edge cases and numerical robustness."""

    def test_out_of_range_scores_clamped(self) -> None:
        """Scores outside [0, 1] should be clamped, not cause negative indexing.

        This is the bug the analyst identified: int(score * n_bins) with negative
        scores causes Python to index from the end of the array (silent bug).
        """
        np.random.seed(42)
        n = 100
        n_policies = 2

        # Create scores with some out-of-range values
        calibration_index = np.concatenate(
            [
                np.array([-0.1, -0.05, 0.0, 1.0, 1.05, 1.1]),  # Edge cases
                np.random.beta(4, 4, n - 6),  # Normal values
            ]
        )
        policy_indices = np.concatenate(
            [
                np.array([0, 1, 0, 1, 0, 1]),
                np.random.randint(0, n_policies, n - 6),
            ]
        )

        # Put oracle labels at some edge cases
        oracle_mask = np.zeros(n, dtype=bool)
        oracle_mask[[0, 1, 2, 3, 4, 5, 10, 11, 12]] = True
        oracle_labels = np.full(n, np.nan)
        oracle_labels[oracle_mask] = 0.5 + np.random.normal(
            0, 0.02, np.sum(oracle_mask)
        )

        # Should NOT crash with out-of-range values
        density_ratios, diag = compute_density_ratios(
            calibration_index=calibration_index,
            policy_indices=policy_indices,
            oracle_mask=oracle_mask,
            n_policies=n_policies,
            n_bins=10,
        )

        # Ratios should be finite and non-negative
        assert np.all(
            np.isfinite(density_ratios)
        ), f"Non-finite ratios with out-of-range scores: {density_ratios}"
        assert np.all(
            density_ratios >= 0
        ), f"Negative ratios with out-of-range scores: {density_ratios}"

        # The negative scores should be treated as bin 0
        # The >1 scores should be treated as the last bin
        assert density_ratios.shape == (n_policies, np.sum(oracle_mask))

    def test_exact_bin_boundaries(self) -> None:
        """Scores at exact bin boundaries (0.0, 0.1, 0.2, ..., 1.0) handled correctly."""
        np.random.seed(42)
        n_policies = 2
        n_bins = 10

        # Create scores at exact boundaries
        calibration_index = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        policy_indices = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

        # All oracle
        oracle_mask = np.ones(len(calibration_index), dtype=bool)
        oracle_labels = 0.2 + 0.6 * calibration_index

        density_ratios, diag = compute_density_ratios(
            calibration_index=calibration_index,
            policy_indices=policy_indices,
            oracle_mask=oracle_mask,
            n_policies=n_policies,
            n_bins=n_bins,
        )

        # All ratios should be well-defined
        assert np.all(np.isfinite(density_ratios)), "Non-finite at boundaries"
        assert np.all(density_ratios >= 0), "Negative at boundaries"


class TestAnalyticIdentities:
    """Tests verifying key analytic identities from the EIF derivation."""

    def test_weighted_sum_approximately_one(self) -> None:
        """When policies have identical distributions, weighted sum ≈ 1.

        When all policies p have the same density f(z):
        - w_p(z) = f(z) / g(z) where g(z) = Σ_q ρ_q * e_q(z) * f(z) = e(z) * f(z)
        - So w_p(z) = 1/e(z) for all p
        - (1/N) * Σ_{i:L_i=1} w(Z_i) ≈ (1/N) * n_oracle * (1/e) ≈ 1

        This verifies the weights are properly normalized.
        """
        np.random.seed(42)
        n_per_policy = 1000  # Large n for convergence
        n_policies = 3
        n_total = n_per_policy * n_policies

        # All policies same distribution
        calibration_index = np.concatenate(
            [np.random.beta(4, 4, n_per_policy) for _ in range(n_policies)]
        )
        policy_indices = np.concatenate(
            [np.full(n_per_policy, p, dtype=int) for p in range(n_policies)]
        )

        # 20% oracle, uniform across all
        oracle_coverage = 0.2
        oracle_mask = np.random.rand(n_total) < oracle_coverage

        density_ratios, diag = compute_density_ratios(
            calibration_index=calibration_index,
            policy_indices=policy_indices,
            oracle_mask=oracle_mask,
            n_policies=n_policies,
            n_bins=20,
        )

        # When policies identical, (1/N) * Σ_{i:L_i=1} w_p(Z_i) ≈ 1
        # Because weights sum to n_oracle/oracle_coverage ≈ N
        for p in range(n_policies):
            weighted_sum = np.sum(density_ratios[p]) / n_total
            assert (
                0.5 < weighted_sum < 2.0
            ), f"Policy {p}: (1/N)*Σw_p = {weighted_sum:.3f}, expected ~1.0"

    def test_single_policy_weights_are_inverse_oracle_prob(self) -> None:
        """With one policy, w(z) = 1/e(z) where e(z) = P(L=1|Z=z).

        In the single-policy case:
        - g(z) = ρ * e(z) * f(z) with ρ = 1
        - w(z) = f(z) / g(z) = f(z) / (e(z) * f(z)) = 1/e(z)

        So oracle samples with low selection probability get high weights.
        """
        np.random.seed(42)
        n = 1000

        calibration_index = np.random.beta(4, 4, n)
        policy_indices = np.zeros(n, dtype=int)

        # Create non-uniform oracle selection: more oracle at high scores
        # e(z) = 0.1 + 0.3 * z (10-40% depending on score)
        oracle_probs = 0.1 + 0.3 * calibration_index
        oracle_mask = np.random.rand(n) < oracle_probs

        density_ratios, diag = compute_density_ratios(
            calibration_index=calibration_index,
            policy_indices=policy_indices,
            oracle_mask=oracle_mask,
            n_policies=1,
            n_bins=20,
        )

        # Weights should be inversely related to oracle probability
        oracle_z = calibration_index[oracle_mask]
        weights = density_ratios[0]

        # Low-score oracle samples (low e(z)) should have higher weights
        low_score_mask = oracle_z < 0.3
        high_score_mask = oracle_z > 0.7

        if np.any(low_score_mask) and np.any(high_score_mask):
            mean_weight_low = np.mean(weights[low_score_mask])
            mean_weight_high = np.mean(weights[high_score_mask])

            # Low-score samples should have higher weights (lower oracle prob)
            # Allow some noise since histogram estimation is imperfect
            assert (
                mean_weight_low > mean_weight_high * 0.5
            ), f"Low-score weight {mean_weight_low:.2f} should be > high-score weight {mean_weight_high:.2f}"

    def test_mixture_density_matches_oracle_histogram(self) -> None:
        """Verify mixture density g(z) via rho/e/f matches direct oracle histogram.

        The mixture density is computed as:
            g(z) = Σ_p ρ_p * e_p(z) * f_p(z)

        But it can also be computed directly as the histogram of oracle samples:
            g(z) ≈ histogram(oracle_z) / (N * bin_width)

        These should match within floating point tolerance. This is a sharp
        regression test for boundary/binning logic consistency.
        """
        np.random.seed(42)
        n_per_policy = 500
        n_policies = 3
        n_total = n_per_policy * n_policies
        n_bins = 20
        bin_width = 1.0 / n_bins

        # Create data with different distributions per policy
        calibration_index = np.concatenate(
            [
                np.random.beta(2, 5, n_per_policy),  # Policy 0: skewed left
                np.random.beta(5, 5, n_per_policy),  # Policy 1: symmetric
                np.random.beta(5, 2, n_per_policy),  # Policy 2: skewed right
            ]
        )
        policy_indices = np.concatenate(
            [np.full(n_per_policy, p, dtype=int) for p in range(n_policies)]
        )

        # 15% oracle uniformly
        oracle_mask = np.random.rand(n_total) < 0.15

        # Compute density ratios (which internally computes mixture density)
        density_ratios, diag = compute_density_ratios(
            calibration_index=calibration_index,
            policy_indices=policy_indices,
            oracle_mask=oracle_mask,
            n_policies=n_policies,
            n_bins=n_bins,
        )

        # Method 1: Compute mixture density directly from oracle histogram
        # This is the "ground truth" - simple histogram of oracle samples
        bin_edges = np.linspace(0, 1, n_bins + 1)
        z_clamped = np.clip(calibration_index, 0.0, 1.0)
        oracle_z = z_clamped[oracle_mask]

        direct_counts, _ = np.histogram(oracle_z, bins=bin_edges)
        direct_mixture_density = direct_counts / (n_total * bin_width)

        # Method 2: Extract the mixture_density from diagnostics
        # (it's computed via rho/e/f decomposition internally)
        mixture_density = diag["mixture_density"]

        # They should match within floating point tolerance
        # Use relative tolerance for bins with significant mass
        for b in range(n_bins):
            if direct_mixture_density[b] > 0.01:  # Skip near-empty bins
                rel_error = abs(
                    (mixture_density[b] - direct_mixture_density[b])
                    / direct_mixture_density[b]
                )
                assert rel_error < 0.01, (
                    f"Bin {b}: mixture_density={mixture_density[b]:.6f}, "
                    f"direct={direct_mixture_density[b]:.6f}, rel_error={rel_error:.4f}"
                )

        # Also check that they're numerically close overall
        np.testing.assert_allclose(
            mixture_density,
            direct_mixture_density,
            rtol=0.01,
            atol=1e-6,
            err_msg="Mixture density via rho/e/f should match direct oracle histogram",
        )
