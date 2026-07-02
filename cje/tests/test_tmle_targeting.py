"""Regression tests: TMLE targeting must not collapse to the plug-in.

A prior implementation solved the fluctuation on the logged data and then
deliberately did NOT shift the fresh-draw predictions. Because the solved
score equation makes sum(w * (R - g_logged_star)) ~= 0, the IPS correction
vanished and the "TMLE" point estimate equalled the untargeted direct-method
plug-in to machine precision - no double robustness. The targeting step now
propagates the fluctuation to fresh draws through a score-indexed clever
covariate, and these tests pin that behavior on the exact functions the
estimator calls.
"""

from typing import List

import numpy as np
import pytest

from cje.estimators.tmle import (
    apply_targeted_fluctuation,
    eval_fold_weight_map,
    fit_fold_weight_maps,
)


def test_weight_map_recovers_score_indexed_weights() -> None:
    """For S-calibrated weights (a function of S per fold), the map is exact."""
    rng = np.random.default_rng(3)
    n = 500
    scores = rng.uniform(0, 1, size=n)
    fold_ids = rng.integers(0, 5, size=n)
    # A different deterministic monotone weight function per fold
    weights = 0.5 + (fold_ids + 1) * 0.2 * scores

    maps = fit_fold_weight_maps(scores, weights, fold_ids)
    for k in range(5):
        mask = fold_ids == k
        recovered = eval_fold_weight_map(maps, k, scores[mask])
        np.testing.assert_allclose(recovered, weights[mask], rtol=1e-12)


def _simulate(n: int = 20000, bias: float = -0.10, seed: int = 11) -> dict:
    """DGP with known truth and a deliberately biased outcome model.

    Logged scores S ~ U(0,1); true outcome mu(S) = 0.2 + 0.6*S plus noise.
    Target policy tilts toward high S with mean-one weights w(S) = 0.4 + 1.2*S.
    Fresh draws are sampled from the tilted score distribution, so
    V(pi') = E[w(S) * mu(S)] under the logging distribution.
    """
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0, 1, size=n)
    mu = 0.2 + 0.6 * scores
    rewards = np.clip(mu + rng.normal(0, 0.05, size=n), 0, 1)
    weights = 0.4 + 1.2 * scores  # mean one under U(0,1)

    # Truth under the tilted (target) distribution
    truth = float(np.mean(weights * mu))

    # Fresh draws: one per logged prompt, S' ~ tilted density via rejection
    fresh_scores: List[np.ndarray] = []
    for _ in range(n):
        while True:
            s = rng.uniform(0, 1)
            if rng.uniform(0, 1.6) <= 0.4 + 1.2 * s:
                fresh_scores.append(np.array([s]))
                break

    fold_ids = np.arange(n) % 5
    g0 = lambda s: np.clip(0.2 + 0.6 * np.asarray(s) + bias, 1e-7, 1 - 1e-7)

    return dict(
        scores=scores,
        rewards=rewards,
        weights=weights,
        fold_ids=fold_ids,
        fresh_scores=fresh_scores,
        g0=g0,
        truth=truth,
    )


def test_targeting_restores_double_robustness() -> None:
    """Biased outcome model + correct weights => TMLE ~ truth, plug-in ~ truth+bias."""
    sim = _simulate()
    g0 = sim["g0"]
    g_logged0 = g0(sim["scores"])
    g_fresh_draws = [g0(s) for s in sim["fresh_scores"]]

    maps = fit_fold_weight_maps(sim["scores"], sim["weights"], sim["fold_ids"])
    h_logged = np.empty_like(sim["scores"])
    for k in range(5):
        mask = sim["fold_ids"] == k
        h_logged[mask] = eval_fold_weight_map(maps, k, sim["scores"][mask])
    h_fresh = [
        eval_fold_weight_map(maps, sim["fold_ids"][i], s)
        for i, s in enumerate(sim["fresh_scores"])
    ]

    for link in ("logit", "identity"):
        g_logged_star, g_fresh_star, eps, info = apply_targeted_fluctuation(
            g_logged0,
            sim["rewards"],
            h_logged,
            g_fresh_draws,
            h_fresh,
            link=link,
        )

        psi_dm = float(np.mean([g.mean() for g in g_fresh_draws]))
        ips_corr = float(np.mean(sim["weights"] * (sim["rewards"] - g_logged_star)))
        psi_tmle = float(np.mean(g_fresh_star)) + ips_corr
        psi_drcpo = psi_dm + float(
            np.mean(sim["weights"] * (sim["rewards"] - g_logged0))
        )
        truth = sim["truth"]

        # The plug-in inherits the outcome-model bias (~0.1)
        assert abs(psi_dm - truth) > 0.07, f"{link}: DM should be visibly biased"
        # Targeting must remove (nearly all of) it
        assert (
            abs(psi_tmle - truth) < 0.02
        ), f"{link}: TMLE {psi_tmle:.4f} should be near truth {truth:.4f}"
        # And must NOT equal the plug-in (the old collapse)
        assert abs(psi_tmle - psi_dm) > 0.05, f"{link}: TMLE collapsed to plug-in"
        # First-order agreement with DR-CPO on the same nuisances
        assert (
            abs(psi_tmle - psi_drcpo) < 0.02
        ), f"{link}: TMLE {psi_tmle:.4f} vs DR-CPO {psi_drcpo:.4f}"
        # Targeted fresh predictions actually moved
        assert info["epsilon"] != 0.0


def test_well_specified_model_barely_moves() -> None:
    """With an unbiased outcome model, targeting should be a small perturbation."""
    sim = _simulate(bias=0.0, seed=5)
    g0 = sim["g0"]
    g_logged0 = g0(sim["scores"])
    g_fresh_draws = [g0(s) for s in sim["fresh_scores"]]

    maps = fit_fold_weight_maps(sim["scores"], sim["weights"], sim["fold_ids"])
    h_logged = np.empty_like(sim["scores"])
    for k in range(5):
        mask = sim["fold_ids"] == k
        h_logged[mask] = eval_fold_weight_map(maps, k, sim["scores"][mask])
    h_fresh = [
        eval_fold_weight_map(maps, sim["fold_ids"][i], s)
        for i, s in enumerate(sim["fresh_scores"])
    ]

    g_logged_star, g_fresh_star, eps, info = apply_targeted_fluctuation(
        g_logged0, sim["rewards"], h_logged, g_fresh_draws, h_fresh, link="logit"
    )
    psi_dm = float(np.mean([g.mean() for g in g_fresh_draws]))
    psi_tmle = float(np.mean(g_fresh_star)) + float(
        np.mean(sim["weights"] * (sim["rewards"] - g_logged_star))
    )
    assert abs(psi_tmle - sim["truth"]) < 0.01
    assert abs(psi_tmle - psi_dm) < 0.01


def test_empty_fresh_draws_yield_nan_not_crash() -> None:
    g_logged0 = np.array([0.4, 0.6, 0.5])
    rewards = np.array([0.5, 0.7, 0.4])
    h = np.array([1.0, 1.0, 1.0])
    g_fresh = [np.array([0.5]), np.array([]), np.array([0.6, 0.4])]
    h_fresh = [np.array([1.0]), np.array([]), np.array([1.0, 1.0])]
    _, g_fresh_star, _, _ = apply_targeted_fluctuation(
        g_logged0, rewards, h, g_fresh, h_fresh, link="logit"
    )
    assert np.isnan(g_fresh_star[1])
    assert np.isfinite(g_fresh_star[0]) and np.isfinite(g_fresh_star[2])
