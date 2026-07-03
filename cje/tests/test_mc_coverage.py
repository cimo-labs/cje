"""Monte Carlo ground-truth coverage harness for CJE estimators.

Synthetic DGP with a KNOWN policy value:

- judge scores S ~ U(0,1) on logged data;
- true outcome mu(S) = 0.25 + 0.5*S, oracle labels Y = clip(mu + noise, 0, 1)
  observed on a 25% random slice;
- fresh draws S' ~ a mean-one tilted score density w(S) = 0.4 + 1.2*S via
  rejection sampling.

True policy value: V(pi') = E[w(S) * mu(S)] under U(0,1)
                 = 0.4*0.25 + (0.4*0.5 + 1.2*0.25)*1/2 + 1.2*0.5*1/3 = 0.55.

The fresh draws are sampled from the tilted density directly, so the Direct
estimand E[mu(S')] over the draw distribution equals the same TRUE_VALUE.

Two layers:

- FAST (runs in CI, deterministic seeds): R=40 replicates of
  CalibratedDirectEstimator (cluster_robust inference). Asserts (i) point
  estimates unbiased within MC tolerance, (ii) mean reported SE within
  [0.7, 1.3]x of the empirical SD across replicates at 100% oracle coverage
  (the IF-only regime), (iii) reported SE not understated (ratio >= 0.7) at
  25% oracle coverage (the OUA /K bug produced ~0.45 here), and (iv) a
  deterministic boundary-card regression: cards appear in results and fire
  on out-of-range judge scores.

- SLOW (@pytest.mark.slow, excluded from CI): R=300 replicates; asserts 95%
  CI coverage >= 88% for the direct estimator at 25% oracle coverage.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from cje.calibration import calibrate_dataset
from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample
from cje.data.models import Dataset, Sample
from cje.estimators.direct_method import CalibratedDirectEstimator

POLICY = "target"
DEFAULT_ORACLE_FRAC = 0.25
OUTCOME_NOISE = 0.08

# E[w(S) * mu(S)] with w(S)=0.4+1.2S, mu(S)=0.25+0.5S, S~U(0,1)
TRUE_VALUE = 0.4 * 0.25 + (0.4 * 0.5 + 1.2 * 0.25) * 0.5 + 1.2 * 0.5 / 3.0


def _mu(s: np.ndarray) -> np.ndarray:
    return np.asarray(0.25 + 0.5 * s)


def _tilt(s: float) -> float:
    return 0.4 + 1.2 * s  # mean-one under U(0,1)


def _sample_tilted_score(rng: np.random.Generator) -> float:
    """Rejection-sample S' from the tilted density w(s) on [0,1]."""
    while True:
        s = float(rng.uniform())
        if rng.uniform(0, 1.6) <= _tilt(s):
            return s


def _simulate(
    n: int,
    m_draws: int,
    rng: np.random.Generator,
    oracle_frac: float = DEFAULT_ORACLE_FRAC,
) -> Tuple[Dataset, FreshDrawDataset]:
    """One replicate of logged data + fresh draws from the DGP."""
    samples = []
    for i in range(n):
        s = float(rng.uniform())
        y = float(np.clip(_mu(np.array(s)) + rng.normal(0, OUTCOME_NOISE), 0, 1))
        samples.append(
            Sample(
                prompt_id=f"p{i}",
                prompt=f"question {i}",
                response=f"answer {i}",
                reward=None,
                base_policy_logprob=-10.0,
                target_policy_logprobs={POLICY: -10.0 + float(np.log(_tilt(s)))},
                judge_score=s,
                oracle_label=y if rng.uniform() < oracle_frac else None,
            )
        )
    dataset = Dataset(samples=samples, target_policies=[POLICY])

    fresh_samples = []
    for i in range(n):
        for d in range(m_draws):
            fresh_samples.append(
                FreshDrawSample(
                    prompt_id=f"p{i}",
                    judge_score=_sample_tilted_score(rng),
                    oracle_label=None,
                    response=None,
                    fold_id=None,
                    target_policy=POLICY,
                    draw_idx=d,
                )
            )
    fresh = FreshDrawDataset(
        samples=fresh_samples, target_policy=POLICY, draws_per_prompt=m_draws
    )
    return dataset, fresh


def _run_replicate(
    seed: int,
    n: int,
    m_draws: int,
    estimator_names: List[str],
    oracle_frac: float = DEFAULT_ORACLE_FRAC,
) -> Dict[str, Dict[str, float]]:
    """Run the requested estimators on one simulated dataset.

    Returns per-estimator {estimate, se, ci_lo, ci_hi}.
    """
    rng = np.random.default_rng(seed)
    dataset, fresh = _simulate(n, m_draws, rng, oracle_frac=oracle_frac)
    calibrated, cal_result = calibrate_dataset(
        dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
        enable_cross_fit=True,
        n_folds=5,
    )
    calibrator = cal_result.calibrator

    out: Dict[str, Dict[str, float]] = {}
    for name in estimator_names:
        est: Any
        if name == "direct":
            est = CalibratedDirectEstimator(
                target_policies=[POLICY],
                reward_calibrator=calibrator,
                inference_method="cluster_robust",
                oua_jackknife=True,
            )
            est.add_fresh_draws(POLICY, fresh)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown estimator '{name}'")

        result = est.fit_and_estimate()
        lo, hi = result.confidence_interval(alpha=0.05)
        out[name] = {
            "estimate": float(result.estimates[0]),
            "se": float(result.standard_errors[0]),
            "ci_lo": float(lo[0]),
            "ci_hi": float(hi[0]),
        }
    return out


def _collect(
    estimator_names: List[str],
    n_reps: int,
    n: int,
    m_draws: int,
    seed0: int,
    oracle_frac: float = DEFAULT_ORACLE_FRAC,
) -> Dict[str, Dict[str, np.ndarray]]:
    rows: Dict[str, Dict[str, List[float]]] = {
        name: {"estimate": [], "se": [], "ci_lo": [], "ci_hi": []}
        for name in estimator_names
    }
    for r in range(n_reps):
        rep = _run_replicate(
            seed0 + r, n, m_draws, estimator_names, oracle_frac=oracle_frac
        )
        for name in estimator_names:
            for key in rows[name]:
                rows[name][key].append(rep[name][key])
    return {
        name: {key: np.asarray(vals) for key, vals in cols.items()}
        for name, cols in rows.items()
    }


# ---------------------------------------------------------------------------
# FAST layer: unbiasedness + SE calibration for the direct estimator
# ---------------------------------------------------------------------------

N_FAST = 800
R_FAST = 40


@pytest.fixture(scope="module")
def fast_direct_replicates() -> Dict[str, Dict[str, np.ndarray]]:
    """R_FAST replicates at 25% oracle coverage.

    Total uncertainty = cluster-robust IF variance + oracle (calibration)
    jackknife variance.
    """
    return _collect(
        ["direct"],
        n_reps=R_FAST,
        n=N_FAST,
        m_draws=1,
        seed0=20260701,
    )


@pytest.fixture(scope="module")
def fast_direct_full_oracle_replicates() -> Dict[str, Dict[str, np.ndarray]]:
    """R_FAST replicates at 100% oracle coverage.

    With every logged sample labeled, the OUA jackknife is skipped by
    design, so the reported SE is the cluster-robust IF component alone and
    the empirical SD across replicates is the matching truth. This isolates
    IF-SE correctness from oracle-jackknife behavior.
    """
    return _collect(
        ["direct"],
        n_reps=R_FAST,
        n=N_FAST,
        m_draws=1,
        seed0=41_000_000,
        oracle_frac=1.0,
    )


def test_fast_direct_point_estimate_unbiased(
    fast_direct_replicates: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """The calibrated plug-in mean over tilted draws must hit TRUE_VALUE.

    The DGP encodes the truth in the draw distribution itself: fresh draws
    are rejection-sampled from w(s), so E[f*(S')] = E[w(S) mu(S)] = 0.55.
    Tolerance is MC noise plus a small isotonic-boundary allowance.
    """
    estimates = fast_direct_replicates["direct"]["estimate"]
    mc_se_of_mean = float(np.std(estimates, ddof=1) / np.sqrt(len(estimates)))
    bias = float(np.mean(estimates) - TRUE_VALUE)
    tol = 4 * mc_se_of_mean + 0.005
    assert abs(bias) < tol, (
        f"direct: mean estimate {np.mean(estimates):.4f} deviates from truth "
        f"{TRUE_VALUE:.4f} by {bias:.4f} (tol {tol:.4f})"
    )


def test_fast_direct_se_matches_empirical_sd_full_oracle(
    fast_direct_full_oracle_replicates: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """At 100% oracle coverage the reported SE is the IF component alone.

    This is the sharp test for the SE composition: a broken cluster-robust
    computation or a re-added variance term would miss in either direction.
    """
    estimates = fast_direct_full_oracle_replicates["direct"]["estimate"]
    ses = fast_direct_full_oracle_replicates["direct"]["se"]
    empirical_sd = float(np.std(estimates, ddof=1))
    ratio = float(np.mean(ses)) / empirical_sd
    assert 0.7 <= ratio <= 1.3, (
        f"direct (100% oracle): mean reported SE {np.mean(ses):.5f} vs "
        f"empirical SD {empirical_sd:.5f} (ratio {ratio:.2f}) outside [0.7, 1.3]"
    )


def test_fast_direct_reported_se_not_understated(
    fast_direct_replicates: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """With 25% oracle coverage, total SE must not understate reality.

    The old OUA jackknife (/K) bug produced ratios ~0.45 here — the exact
    failure mode OUA exists to prevent. No upper bound: the K=5
    delete-one-fold jackknife on isotonic calibrators is noisy and
    right-skewed, so the MEAN reported SE runs conservative in
    oracle-dominated regimes; the slow coverage test guards the
    under-coverage direction.
    """
    estimates = fast_direct_replicates["direct"]["estimate"]
    ses = fast_direct_replicates["direct"]["se"]
    empirical_sd = float(np.std(estimates, ddof=1))
    ratio = float(np.mean(ses)) / empirical_sd
    assert ratio >= 0.7, (
        f"direct: mean reported SE {np.mean(ses):.5f} understates empirical SD "
        f"{empirical_sd:.5f} (ratio {ratio:.2f} < 0.7)"
    )


def _run_direct_with_fresh_scores(
    logged_scores: np.ndarray,
    fresh_scores: np.ndarray,
    rng: np.random.Generator,
) -> Any:
    """Fit calibration on logged data and run Direct on injected draws."""
    samples = []
    for i, s in enumerate(logged_scores):
        y = float(np.clip(_mu(np.array(s)) + rng.normal(0, OUTCOME_NOISE), 0, 1))
        samples.append(
            Sample(
                prompt_id=f"p{i}",
                prompt=f"question {i}",
                response=f"answer {i}",
                reward=None,
                base_policy_logprob=-10.0,
                target_policy_logprobs={POLICY: -10.0},
                judge_score=float(s),
                oracle_label=y if rng.uniform() < 0.5 else None,
            )
        )
    dataset = Dataset(samples=samples, target_policies=[POLICY])
    _, cal_result = calibrate_dataset(
        dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
        enable_cross_fit=True,
        n_folds=5,
    )

    fresh_samples = [
        FreshDrawSample(
            prompt_id=f"p{i % len(logged_scores)}",
            judge_score=float(s),
            oracle_label=None,
            response=None,
            fold_id=None,
            target_policy=POLICY,
            draw_idx=0,
        )
        for i, s in enumerate(fresh_scores)
    ]
    fresh = FreshDrawDataset(
        samples=fresh_samples, target_policy=POLICY, draws_per_prompt=1
    )

    est = CalibratedDirectEstimator(
        target_policies=[POLICY],
        reward_calibrator=cal_result.calibrator,
        inference_method="cluster_robust",
        oua_jackknife=True,
    )
    est.add_fresh_draws(POLICY, fresh)
    return est.fit_and_estimate()


def test_fast_direct_boundary_cards_regression() -> None:
    """Deterministic pin: boundary cards appear and fire on out-of-range scores.

    Oracle calibration support is S in [0, 0.6]; 10% of the injected fresh
    draws sit above it -> REFUSE-LEVEL (>= 5% threshold), CRITICAL status,
    and the reliability-gate flag the CLI trophy logic consumes. An
    in-range run stays OK and unflagged.
    """
    rng = np.random.default_rng(20260702)
    logged_scores = rng.uniform(0.0, 0.6, 300)

    # Out-of-range run: 10% of fresh-draw judge mass above the oracle range.
    # The in-range component stays strictly inside the REALIZED oracle
    # support (labels cover a random half of [0, 0.6]), so only the
    # injected 10% is out of range.
    fresh_bad = np.concatenate(
        [rng.uniform(0.05, 0.55, 270), rng.uniform(0.75, 0.95, 30)]
    )
    result = _run_direct_with_fresh_scores(logged_scores, fresh_bad, rng)
    card = result.diagnostics.boundary_cards[POLICY]
    assert card["status"] == "REFUSE-LEVEL"
    assert card["out_of_range"] == pytest.approx(0.10, abs=0.01)
    assert result.metadata["boundary_cards"][POLICY] == card
    assert result.diagnostics.status_per_policy[POLICY].value == "critical"
    gate = result.metadata["reliability_gates"][POLICY]
    assert gate["flagged"] is True
    assert gate["refuse_level_claims"] is True

    # In-range control: card computed, OK, unflagged
    rng = np.random.default_rng(20260702)
    logged_scores = rng.uniform(0.0, 0.6, 300)
    fresh_ok = rng.uniform(0.05, 0.55, 300)
    result = _run_direct_with_fresh_scores(logged_scores, fresh_ok, rng)
    card = result.diagnostics.boundary_cards[POLICY]
    assert card["status"] == "OK"
    assert result.diagnostics.status_per_policy[POLICY].value == "good"
    assert result.metadata["reliability_gates"][POLICY]["flagged"] is False


# ---------------------------------------------------------------------------
# SLOW layer: 95% CI coverage for the direct estimator
# ---------------------------------------------------------------------------

N_SLOW = 500
R_SLOW = 300
SLOW_ESTIMATORS = ["direct"]


@pytest.fixture(scope="module")
def slow_replicates() -> Dict[str, Dict[str, np.ndarray]]:
    return _collect(
        SLOW_ESTIMATORS,
        n_reps=R_SLOW,
        n=N_SLOW,
        m_draws=2,
        seed0=8_000_000,
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", SLOW_ESTIMATORS)
def test_slow_ci_coverage(
    slow_replicates: Dict[str, Dict[str, np.ndarray]], name: str
) -> None:
    lo = slow_replicates[name]["ci_lo"]
    hi = slow_replicates[name]["ci_hi"]
    covered = float(np.mean((lo <= TRUE_VALUE) & (TRUE_VALUE <= hi)))
    # Lower bound is the real guard: under-coverage is the failure mode this
    # harness exists to catch (the OUA /K bug produced ~50-80% here). No upper
    # bound: the K=5 delete-one-fold jackknife on isotonic calibrators is
    # conservative in oracle-dominated regimes (~99% observed), which is the
    # honest direction.
    assert covered >= 0.88, (
        f"{name}: 95% CI coverage {covered:.1%} over {len(lo)} replicates " f"below 88%"
    )
