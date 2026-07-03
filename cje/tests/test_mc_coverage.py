"""Monte Carlo ground-truth coverage harness for CJE estimators.

Synthetic DGP with a KNOWN policy value:

- judge scores S ~ U(0,1) on logged data;
- true outcome mu(S) = 0.25 + 0.5*S, oracle labels Y = clip(mu + noise, 0, 1)
  observed on a 25% random slice;
- fresh draws S' ~ a mean-one tilted score density w(S) = 0.4 + 1.2*S via
  rejection sampling.

True policy value: V(pi') = E[w(S) * mu(S)] under U(0,1)
                 = 0.4*0.25 + (0.4*0.5 + 1.2*0.25)*1/2 + 1.2*0.5*1/3 = 0.55.

NOTE(WP3): the fast layer (unbiasedness + SE-calibration assertions that ran
in CI) covered only the removed OPE estimators; WP3 adds a direct-mode fast
layer so 0.4.0 ships with an inference regression guard. Until then only the
SLOW coverage test below exercises this harness.

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
