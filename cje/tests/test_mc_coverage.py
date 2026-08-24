"""Monte Carlo ground-truth coverage harness for CJE estimators.

Synthetic DGP with a KNOWN policy value:

- judge scores S ~ U(0,1) on logged data;
- true outcome mu(S) = 0.25 + 0.5*S, oracle labels Y = mu + bounded noise
  observed on a 25% random slice;
- fresh draws S' follow the mean-one tilted score density
  w(S) = 0.4 + 1.2*S. Logged and fresh scores share a prompt-level uniform
  latent with controlled probability; otherwise a target draw gets fresh
  within-prompt variation. This preserves the exact tilted marginal while
  matching a coupled "same prompts, stochastic policy response" design.

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
  CI coverage in [88%, 99%] for the direct estimator at 25% oracle coverage.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from cje.array_api import calibrated_mean_ci
from cje.calibration import calibrate_dataset
from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample
from cje.data.models import Dataset, Sample
from cje.diagnostics.robust_inference import CalibrationProvenance
from cje.estimators.direct_method import CalibratedDirectEstimator

POLICY = "target"
DEFAULT_ORACLE_FRAC = 0.25
OUTCOME_NOISE = 0.08
PROMPT_COUPLING = 0.70

# E[w(S) * mu(S)] with w(S)=0.4+1.2S, mu(S)=0.25+0.5S, S~U(0,1)
TRUE_VALUE = 0.4 * 0.25 + (0.4 * 0.5 + 1.2 * 0.25) * 0.5 + 1.2 * 0.5 / 3.0


def _mu(s: np.ndarray) -> np.ndarray:
    return np.asarray(0.25 + 0.5 * s)


def _tilted_quantile(u: float) -> float:
    """Map U(0,1) to density w(s)=0.4+1.2s via its inverse CDF."""
    return float((-0.4 + np.sqrt(0.16 + 2.4 * u)) / 1.2)


def _simulate(
    n: int,
    m_draws: int,
    rng: np.random.Generator,
    oracle_frac: float = DEFAULT_ORACLE_FRAC,
) -> Tuple[Dataset, FreshDrawDataset]:
    """One replicate of logged data + fresh draws from the DGP."""
    samples = []
    prompt_latents = []
    for i in range(n):
        s = float(rng.uniform())
        prompt_latents.append(s)
        y = float(_mu(np.array(s)) + rng.uniform(-OUTCOME_NOISE, OUTCOME_NOISE))
        samples.append(
            Sample(
                prompt_id=f"p{i}",
                prompt=f"question {i}",
                response=f"answer {i}",
                reward=None,
                judge_score=s,
                oracle_label=y if rng.uniform() < oracle_frac else None,
            )
        )
    dataset = Dataset(samples=samples, target_policies=[POLICY])

    fresh_samples = []
    for i, prompt_latent in enumerate(prompt_latents):
        for d in range(m_draws):
            target_quantile = (
                prompt_latent
                if rng.uniform() < PROMPT_COUPLING
                else float(rng.uniform())
            )
            target_score = _tilted_quantile(target_quantile)
            fresh_samples.append(
                FreshDrawSample(
                    prompt_id=f"p{i}",
                    judge_score=target_score,
                    oracle_label=(
                        float(
                            _mu(np.asarray(target_score))
                            + rng.uniform(-OUTCOME_NOISE, OUTCOME_NOISE)
                        )
                        if oracle_frac >= 1.0
                        else None
                    ),
                    response=None,
                    target_policy=POLICY,
                    draw_idx=d,
                )
            )
    fresh = FreshDrawDataset(samples=fresh_samples, target_policy=POLICY)
    return dataset, fresh


def test_simulation_preserves_stochastic_within_prompt_draws() -> None:
    """The coupled DGP must not collapse all draws for a prompt to one score."""
    _, fresh = _simulate(100, 3, np.random.default_rng(20260824))
    by_prompt: Dict[str, set[float]] = {}
    for sample in fresh.samples:
        by_prompt.setdefault(sample.prompt_id, set()).add(float(sample.judge_score))
    assert any(len(scores) > 1 for scores in by_prompt.values())


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
    oracle_rows = [
        sample for sample in dataset.samples if sample.oracle_label is not None
    ]
    provenance = CalibrationProvenance(
        judge_scores=np.asarray([sample.judge_score for sample in oracle_rows]),
        oracle_labels=np.asarray([sample.oracle_label for sample in oracle_rows]),
        prompt_ids=[sample.prompt_id for sample in oracle_rows],
    )

    out: Dict[str, Dict[str, float]] = {}
    for name in estimator_names:
        est: Any
        if name == "direct":
            est = CalibratedDirectEstimator(
                target_policies=[POLICY],
                reward_calibrator=calibrator,
                calibration_provenance=provenance,
            )
            est.add_fresh_draws(POLICY, fresh)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown estimator '{name}'")

        result = est.fit_and_estimate()
        if oracle_frac < 1.0:
            assert result.metadata["inference"]["coupled"] is True
            assert result.metadata["inference"]["coupling_overlap"] == len(oracle_rows)
        else:
            assert result.metadata["inference"]["coupled"] is False
            assert result.metadata["inference"]["coupling_overlap"] == 0
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

    With every evaluation draw labeled, the estimator routes directly to the
    oracle mean and skips the OUA jackknife. The reported SE is therefore the
    cluster-robust IF component alone, isolating its calibration from the
    oracle-jackknife behavior.
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

    The DGP encodes the truth in the draw distribution itself: the inverse-CDF
    transform gives fresh scores density w(s), so
    E[f*(S')] = E[w(S) mu(S)] = 0.55.
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
            target_policy=POLICY,
            draw_idx=0,
        )
        for i, s in enumerate(fresh_scores)
    ]
    fresh = FreshDrawDataset(samples=fresh_samples, target_policy=POLICY)

    est = CalibratedDirectEstimator(
        target_policies=[POLICY],
        reward_calibrator=cal_result.calibrator,
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
    """Coupled-data coverage evidence for the default analytic path.

    ``_simulate`` gives the oracle slice and fresh draws the same prompt IDs,
    so calibration and evaluation overlap by construction. The slow check
    therefore exercises the default additive cluster-robust + oracle-jackknife
    interval in the coupled setting, rather than an artificially disjoint one.
    """
    return _collect(
        SLOW_ESTIMATORS,
        n_reps=R_SLOW,
        n=N_SLOW,
        m_draws=2,
        seed0=8_000_000,
    )


@pytest.fixture(scope="module")
def slow_array_replicates() -> Dict[str, np.ndarray]:
    """Same-row partial-label coverage for the array API's analytic default."""
    rows: Dict[str, List[float]] = {
        "estimate": [],
        "ci_lo": [],
        "ci_hi": [],
    }
    for replicate in range(R_SLOW):
        rng = np.random.default_rng(18_000_000 + replicate)
        scores = rng.uniform(size=N_SLOW)
        full_labels = np.clip(
            _mu(scores) + rng.normal(0, OUTCOME_NOISE, size=N_SLOW), 0, 1
        )
        oracle_indices = rng.choice(
            N_SLOW, size=int(DEFAULT_ORACLE_FRAC * N_SLOW), replace=False
        )
        observed_labels = np.full(N_SLOW, np.nan)
        observed_labels[oracle_indices] = full_labels[oracle_indices]

        result = calibrated_mean_ci(
            scores,
            observed_labels,
            cluster_ids=[f"p{i}" for i in range(N_SLOW)],
        )
        assert result.method == "cluster_robust"
        assert result.diagnostics["inference_reason"] == (
            "cluster_robust requested/default"
        )
        rows["estimate"].append(result.estimate)
        rows["ci_lo"].append(result.ci[0])
        rows["ci_hi"].append(result.ci[1])
    return {key: np.asarray(values) for key, values in rows.items()}


@pytest.mark.slow
@pytest.mark.parametrize("name", SLOW_ESTIMATORS)
def test_slow_ci_coverage(
    slow_replicates: Dict[str, Dict[str, np.ndarray]], name: str
) -> None:
    lo = slow_replicates[name]["ci_lo"]
    hi = slow_replicates[name]["ci_hi"]
    covered = float(np.mean((lo <= TRUE_VALUE) & (TRUE_VALUE <= hi)))
    assert 0.88 <= covered <= 0.99, (
        f"{name}: 95% CI coverage {covered:.1%} over {len(lo)} replicates "
        "outside [88%, 99%]"
    )


@pytest.mark.slow
def test_slow_array_api_same_row_ci_coverage(
    slow_array_replicates: Dict[str, np.ndarray],
) -> None:
    """The default interval covers a same-row partially labeled mean."""
    lo = slow_array_replicates["ci_lo"]
    hi = slow_array_replicates["ci_hi"]
    covered = float(np.mean((lo <= 0.5) & (0.5 <= hi)))
    assert 0.88 <= covered <= 0.99, (
        f"array API: 95% CI coverage {covered:.1%} over {len(lo)} replicates "
        "outside [88%, 99%]"
    )
