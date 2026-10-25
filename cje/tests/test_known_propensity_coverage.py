"""Coverage and width of the known-propensity interval under two label designs.

``label_design="known_propensity"`` corrects the calibrated mean with the
uncentred Horvitz-Thompson residual 1[labelled] * (Y - f(S)) / p and takes a
cluster-robust variance of that pseudo-outcome. The variance uses only each
row's inclusion probability p, so it matches independent (Bernoulli/Poisson)
inclusion whatever the strata, and over-states the variance when the number
of labels per stratum is fixed: slightly with judge-score strata, heavily with
strata that predict the calibration residual.

Each test checks coverage and SE/RMSE, the mean reported standard error over
the replicates divided by the estimate's actual root-mean-square error about
the population mean. SE/RMSE is the sharp check. Coverage alone cannot tell a
correct variance from a conservative one, and it misses mistakes that only
show with residual-predictive strata: centring the residual within strata, for
example, agrees with the correct variance under judge-score strata, but under
independent inclusion with residual strata it gives SE/RMSE 0.68 and coverage
0.82. The documented per-stratum recipe for fixed-count designs is checked on
the same replicates.

DGP (one policy, a fresh pool of N_ROWS prompts per replicate):

- judge score S ~ U(0, 1);
- outcome mean mu(S) = 0.15 + 0.7 * S, so the population mean is exactly 0.5
  (for the clipped Gaussian outcome too: mu is symmetric about 0.5 under
  S -> 1 - S and the noise is symmetric);
- label rates 0.1 and 0.4 in two strata: below and above the judge-score
  median, or a negative and positive residual (an extreme stand-in for a
  pre-labelling signal that predicts the residual).

Coverage is checked against 0.5 and against each pool's own mean (the
finite-pool mean), which the interval covers at least as often because it also
carries the prompt-sampling variance.

Tolerances. With R replicates the Monte Carlo SE of a coverage near 0.95 is
sqrt(0.95 * 0.05 / R): 0.0069 at R = 1000 and 0.0089 at R = 600. The relative
Monte Carlo SE of SE/RMSE is about 1 / sqrt(2R): 2.2% and 2.9%.

- Independent inclusion: coverage of 0.5 is about 94%, not 95% (18 runs of
  1000 replicates over both strata and several seeds: 92.4% to 94.9%; a small
  finite-sample shortfall with about 100 labels), so the coverage band,
  [0.915, 0.975], is set around that figure. SE/RMSE was 0.98 to 1.04 over 8
  runs (1.038 and 1.026 on the seeds here) and must lie in [0.95, 1.15]:
  SEs 10% smaller give 0.93 and 0.92, 25% larger 1.30 and 1.28, and the
  centred variance 0.67 with residual strata.
- Fixed counts: coverage must reach 0.925. SE/RMSE is 1.04 for judge-score
  strata (band [0.92, 1.2]) and 1.59 for residual strata (must exceed 1.3;
  the centred variance gives 1.03). The per-stratum recipe on the residual
  strata covers 95.8% with SE/RMSE 1.05 (band [0.9, 1.2]).

Dropping the oracle jackknife term (a few percent of the variance here)
passes all of these checks. Runtime is 20 to 35 ms per replicate, 3,200
replicates in all.
"""

import logging
import warnings
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pytest

from cje import analyze_dataset
from cje.array_api import calibrated_mean_ci

N_ROWS = 400
LOW_RATE, HIGH_RATE = 0.1, 0.4
POPULATION_MEAN = 0.5
VALIDITY_FLOOR = 0.925


def _mu(scores: np.ndarray) -> np.ndarray:
    return np.asarray(0.15 + 0.7 * scores)


@pytest.fixture(autouse=True)
def _quiet_cje_logging() -> Iterator[None]:
    """REFUSE-LEVEL and similar warnings repeat every replicate; drop them."""
    logger = logging.getLogger("cje")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous)


def _pool(rng: np.random.Generator, outcome: str) -> Tuple[np.ndarray, np.ndarray]:
    scores = rng.uniform(size=N_ROWS)
    if outcome == "binary":
        outcomes = (rng.uniform(size=N_ROWS) < _mu(scores)).astype(float)
    else:
        outcomes = np.clip(_mu(scores) + rng.normal(0, 0.15, N_ROWS), 0.0, 1.0)
    return scores, outcomes


def _known_propensity(
    scores: np.ndarray,
    outcomes: np.ndarray,
    labelled: np.ndarray,
    propensities: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Estimate, SE and interval from analyze_dataset's known-propensity route."""
    records: List[Dict[str, Any]] = [
        {
            "prompt_id": f"p{i}",
            "judge_score": float(scores[i]),
            "oracle_label": float(outcomes[i]) if labelled[i] else None,
        }
        for i in range(len(scores))
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = analyze_dataset(
            fresh_draws_data={"policy": records},
            label_design="known_propensity",
            label_propensities={"policy": propensities},
            fresh_judge_scale=(0, 1),
            fresh_oracle_scale=(0, 1),
        )
    assert results.metadata["point_estimator"]["routes"] == ["augmented"]
    lower, upper = results.ci()[0]
    return (
        float(results.estimates[0]),
        float(results.standard_errors[0]),
        float(lower),
        float(upper),
    )


def _per_stratum(
    scores: np.ndarray,
    outcomes: np.ndarray,
    labelled: np.ndarray,
    stratum: np.ndarray,
) -> Tuple[float, float]:
    """The per-stratum recipe in reference.md (Array API), shares from the rows.

    Estimate sum_h W_h mu_h with W_h = N_h / N; variance
    sum_h W_h^2 SE_h^2 + sum_h W_h (mu_h - mu)^2 / N, the second term being
    the sampling variance of the shares.
    """
    estimates, ses, shares = [], [], []
    for members in (~stratum, stratum):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calibrated_mean_ci(
                scores[members], np.where(labelled[members], outcomes[members], np.nan)
            )
        estimates.append(result.estimate)
        ses.append(result.se)
        shares.append(members.mean())
    mu_h, se_h, w_h = (np.asarray(v, dtype=float) for v in (estimates, ses, shares))
    estimate = float(w_h @ mu_h)
    variance = np.sum(w_h**2 * se_h**2) + np.sum(w_h * (mu_h - estimate) ** 2) / N_ROWS
    return estimate, float(np.sqrt(variance))


def _summary(
    estimates: List[float],
    ses: List[float],
    intervals: List[Tuple[float, float]],
    pool_means: List[float],
) -> Dict[str, float]:
    lower, upper = np.asarray(intervals).T
    truths = np.asarray(pool_means)
    errors = np.asarray(estimates) - POPULATION_MEAN
    return {
        "population": float(
            np.mean((lower <= POPULATION_MEAN) & (POPULATION_MEAN <= upper))
        ),
        "finite_pool": float(np.mean((lower <= truths) & (truths <= upper))),
        "se_over_rmse": float(np.mean(ses) / np.sqrt(np.mean(errors**2))),
    }


def _simulate(
    design: str, strata: str, outcome: str, seed: int, replicates: int
) -> Dict[str, Dict[str, float]]:
    """Run the replicates; summaries for known_propensity and the recipe."""
    rng = np.random.default_rng(seed)
    kp: Dict[str, List[Any]] = {"est": [], "se": [], "ci": [], "pool": []}
    ps: Dict[str, List[Any]] = {"est": [], "se": [], "ci": []}
    for _ in range(replicates):
        scores, outcomes = _pool(rng, outcome)
        stratum = scores >= 0.5 if strata == "judge_score" else outcomes > _mu(scores)
        if design == "independent":
            propensities = np.where(stratum, HIGH_RATE, LOW_RATE)
            labelled = rng.uniform(size=N_ROWS) < propensities
        else:
            labelled = np.zeros(N_ROWS, dtype=bool)
            propensities = np.empty(N_ROWS)
            for members, rate in ((~stratum, LOW_RATE), (stratum, HIGH_RATE)):
                rows = np.flatnonzero(members)
                n_labels = int(round(rate * rows.size))
                labelled[rng.choice(rows, n_labels, replace=False)] = True
                propensities[rows] = n_labels / rows.size
        estimate, se, lower, upper = _known_propensity(
            scores, outcomes, labelled, propensities
        )
        kp["est"].append(estimate)
        kp["se"].append(se)
        kp["ci"].append((lower, upper))
        kp["pool"].append(outcomes.mean())
        if design == "fixed":
            estimate, se = _per_stratum(scores, outcomes, labelled, stratum)
            half_width = 1.959964 * se
            ps["est"].append(estimate)
            ps["se"].append(se)
            ps["ci"].append((estimate - half_width, estimate + half_width))
    out = {"known_propensity": _summary(kp["est"], kp["se"], kp["ci"], kp["pool"])}
    if design == "fixed":
        out["per_stratum"] = _summary(ps["est"], ps["se"], ps["ci"], kp["pool"])
    return out


@pytest.mark.slow
@pytest.mark.parametrize(
    "strata, outcome, seed",
    [("judge_score", "binary", 20261024), ("residual", "continuous", 20261027)],
)
def test_known_propensity_interval_matches_independent_inclusion(
    strata: str, outcome: str, seed: int
) -> None:
    """Bernoulli inclusion at unequal known rates: the SE matches the error.

    The residual-strata case is the one where a variance centred within
    strata would go wrong (SE/RMSE 0.68).
    """
    replicates = 1000
    summary = _simulate("independent", strata, outcome, seed, replicates)
    kp = summary["known_propensity"]
    context = f"{strata} strata, independent inclusion, {replicates} replicates"
    assert (
        0.95 <= kp["se_over_rmse"] <= 1.15
    ), f"{context}: SE/RMSE {kp['se_over_rmse']:.3f} outside [0.95, 1.15]"
    assert 0.915 <= kp["population"] <= 0.975, (
        f"{context}: population-mean coverage {kp['population']:.3f} outside "
        "[0.915, 0.975] (long-run coverage is about 0.939)"
    )
    assert kp["finite_pool"] >= VALIDITY_FLOOR, (
        f"{context}: finite-pool coverage {kp['finite_pool']:.3f} below "
        f"{VALIDITY_FLOOR}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("strata", ["judge_score", "residual"])
def test_known_propensity_interval_with_fixed_counts(strata: str) -> None:
    """A fixed number of labels per stratum: valid and conservative.

    Judge-score strata over-state the variance slightly. Residual strata
    over-state it heavily (coverage near 100%); there the per-stratum recipe
    sizes the interval to the error.
    """
    replicates = 600
    seed = 20261025 if strata == "judge_score" else 20261026
    summary = _simulate("fixed", strata, "continuous", seed, replicates)
    kp = summary["known_propensity"]
    context = f"{strata} strata, fixed counts, {replicates} replicates"
    for name in ("population", "finite_pool"):
        assert (
            kp[name] >= VALIDITY_FLOOR
        ), f"{context}: {name} coverage {kp[name]:.3f} below {VALIDITY_FLOOR}"
    if strata == "judge_score":
        assert (
            0.92 <= kp["se_over_rmse"] <= 1.2
        ), f"{context}: SE/RMSE {kp['se_over_rmse']:.3f} outside [0.92, 1.2]"
        return
    assert kp["se_over_rmse"] >= 1.3, (
        f"{context}: SE/RMSE {kp['se_over_rmse']:.3f} below 1.3; the documented "
        "over-statement for residual strata no longer holds"
    )
    recipe = summary["per_stratum"]
    assert recipe["population"] >= VALIDITY_FLOOR, (
        f"{context}: per-stratum recipe coverage {recipe['population']:.3f} "
        f"below {VALIDITY_FLOOR}"
    )
    assert 0.9 <= recipe["se_over_rmse"] <= 1.2, (
        f"{context}: per-stratum recipe SE/RMSE {recipe['se_over_rmse']:.3f} "
        "outside [0.9, 1.2]"
    )
