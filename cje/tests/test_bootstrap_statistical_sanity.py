"""Statistical sanity bounds for the positive-weight cluster bootstrap.

The statistical-core remediation replaced the bootstrap's sampling
distribution entirely (multinomial cluster resampling with
resample-until-valid -> positive exponential mean-one cluster weights with a
calibrator refit in every world).  The structural tests elsewhere pin counts,
routes, and NaN honesty but no property of the bootstrap DISTRIBUTION — a
mis-scaled weight draw (wrong exponential scale, or per-row instead of
per-cluster weights) that inflates or deflates every bootstrap SE would pass
them.  These tests bound the scheme against known statistical targets:

- FAST: on one seeded clustered dataset per case, the bootstrap SE must land
  within [0.7, 1.3]x of the analytic cluster-robust SE — for the raw-oracle
  route, the calibrator-refit (augmented) route with partial oracle labels,
  and the paired two-policy difference (paired bootstrap vs analytic paired
  SE).  The clusters carry strong within-prompt correlation, so per-row
  weighting would deflate the bootstrap SE below the band.
- SLOW (@pytest.mark.slow, excluded from CI): ~100-replication Monte Carlo
  harness with a known truth asserting >= 88% empirical coverage of the
  default bootstrap path's 95% CIs, mirroring the loose bounds of
  test_mc_coverage.py (which only exercises inference_method =
  "cluster_robust").
"""

from typing import Optional

import numpy as np
import pytest

from cje.calibration.judge import JudgeCalibrator
from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample
from cje.data.models import EstimationResult
from cje.estimators.direct_method import CalibratedDirectEstimator

SE_RATIO_BAND = (0.7, 1.3)


def _clustered_draws(
    policy: str,
    n_prompts: int,
    draws_per_prompt: int,
    rng: np.random.Generator,
    labeled_frac: float = 1.0,
    policy_shift: float = 0.0,
    prompt_effects: Optional[np.ndarray] = None,
) -> FreshDrawDataset:
    """Fresh draws with prompt-level (cluster) correlation and partial labels.

    Judge scores are uniform per row; oracle labels follow the affine curve
    0.2 + 0.6*S plus a shared prompt effect, so rows within a prompt are
    correlated and prompt is the honest sampling unit.  Labels are observed
    for a prompt-level subset (the "representative" design).
    """
    if prompt_effects is None:
        prompt_effects = rng.normal(0.0, 0.05, n_prompts)
    labeled_prompts = rng.uniform(size=n_prompts) < labeled_frac
    samples = []
    for i in range(n_prompts):
        for draw in range(draws_per_prompt):
            score = float(rng.uniform())
            label = float(
                np.clip(
                    0.2
                    + 0.6 * score
                    + policy_shift
                    + prompt_effects[i]
                    + rng.normal(0.0, 0.05),
                    0.0,
                    1.0,
                )
            )
            samples.append(
                FreshDrawSample(
                    prompt_id=f"p{i}",
                    target_policy=policy,
                    judge_score=score,
                    oracle_label=(label if labeled_prompts[i] else None),
                    response=None,
                    draw_idx=draw,
                )
            )
    return FreshDrawDataset(target_policy=policy, samples=samples)


def _assert_se_ratio_in_band(
    bootstrap_se: float, analytic_se: float, label: str
) -> None:
    ratio = bootstrap_se / analytic_se
    lo, hi = SE_RATIO_BAND
    assert lo <= ratio <= hi, (
        f"{label}: bootstrap SE {bootstrap_se:.5f} vs analytic SE "
        f"{analytic_se:.5f} (ratio {ratio:.2f}) outside [{lo}, {hi}] — the "
        f"bootstrap sampling distribution is mis-scaled"
    )


# ---------------------------------------------------------------------------
# FAST layer: bootstrap SE vs analytic cluster-robust SE on one seeded world
# ---------------------------------------------------------------------------


def test_bootstrap_se_matches_analytic_se_full_oracle() -> None:
    """Raw-oracle route: exponential-weight SE vs CRV1 cluster-robust SE.

    Full oracle coverage routes every policy to the weighted oracle mean, so
    no calibrator runs and the comparison isolates the weight scheme itself.
    Prompt means are drawn from U(0.2, 0.8) with small row noise — the
    design effect of per-row weighting across the 3 draws per prompt would
    deflate the bootstrap SE by ~1/sqrt(3), outside the band.
    """
    rng = np.random.default_rng(101)
    n_prompts, m = 80, 3
    samples = []
    for i, mu in enumerate(rng.uniform(0.2, 0.8, n_prompts)):
        for draw in range(m):
            label = float(np.clip(mu + rng.normal(0.0, 0.05), 0.0, 1.0))
            samples.append(
                FreshDrawSample(
                    prompt_id=f"p{i}",
                    target_policy="policy",
                    judge_score=label,
                    oracle_label=label,
                    response=None,
                    draw_idx=draw,
                )
            )
    draws = FreshDrawDataset(target_policy="policy", samples=samples)

    analytic = CalibratedDirectEstimator(
        ["policy"], None, inference_method="cluster_robust"
    )
    analytic.add_fresh_draws("policy", draws)
    result_analytic = analytic.fit_and_estimate()

    bootstrap = CalibratedDirectEstimator(
        ["policy"], None, inference_method="bootstrap", n_bootstrap=400
    )
    bootstrap.add_fresh_draws("policy", draws)
    result_bootstrap = bootstrap.fit_and_estimate()

    assert (
        result_bootstrap.metadata["inference"]["bootstrap_scheme"]
        == "positive_exponential_cluster_weights"
    )
    np.testing.assert_allclose(result_analytic.estimates, result_bootstrap.estimates)
    _assert_se_ratio_in_band(
        float(result_bootstrap.standard_errors[0]),
        float(result_analytic.standard_errors[0]),
        "full-oracle direct route",
    )


def test_bootstrap_se_matches_analytic_se_with_calibrator_refit() -> None:
    """Calibrator-refit route: default bootstrap SE vs cluster-robust + OUA SE.

    Partial prompt-level oracle labels exercise the full default path — the
    per-replicate calibrator refit on exponentially reweighted calibration
    rows, the evaluation-linked weight coupling, and the weighted-ratio
    residual augmentation.  The analytic comparator combines the
    cluster-robust IF SE with the oracle jackknife; both target the same
    total uncertainty, so the SEs must agree within the MC band.
    """
    rng = np.random.default_rng(202)
    draws = _clustered_draws(
        "policy", n_prompts=150, draws_per_prompt=2, rng=rng, labeled_frac=0.5
    )
    labeled = [s for s in draws.samples if s.oracle_label is not None]
    scores = np.asarray([s.judge_score for s in labeled])
    labels = np.asarray([s.oracle_label for s in labeled])
    prompts = [s.prompt_id for s in labeled]

    def run(inference_method: Optional[str]) -> EstimationResult:
        calibrator = JudgeCalibrator(calibration_mode="monotone")
        calibrator.fit_cv(scores, labels, prompt_ids=prompts)
        estimator = CalibratedDirectEstimator(
            ["policy"],
            calibrator,
            inference_method=inference_method,
            n_bootstrap=300,
        )
        estimator.add_fresh_draws("policy", draws)
        return estimator.fit_and_estimate()

    result_analytic = run("cluster_robust")
    # inference_method=None pins the DEFAULT: bootstrap whenever a reward
    # calibrator is supplied.
    result_bootstrap = run(None)

    inference = result_bootstrap.metadata["inference"]
    assert inference["method"] == "cluster_bootstrap_refit"
    assert inference["n_bootstrap_valid"] == 300
    assert inference["effective_estimator_routes"] == ["augmented"]
    np.testing.assert_allclose(
        result_analytic.estimates, result_bootstrap.estimates, atol=1e-12
    )
    _assert_se_ratio_in_band(
        float(result_bootstrap.standard_errors[0]),
        float(result_analytic.standard_errors[0]),
        "calibrator-refit augmented route",
    )


def test_paired_bootstrap_difference_se_matches_analytic_paired_se() -> None:
    """Paired two-policy case: replicate-delta SE vs analytic paired SE.

    Both policies share prompts with a strong common prompt effect
    (U(0.2, 0.7) prompt means), so the honest paired difference SE is far
    below the independent combination.  The paired bootstrap must (a) agree
    with the analytic paired SE within the band and (b) stay well below the
    independent combination — a bootstrap that decoupled the shared prompt
    weights across policies would lose the pairing and inflate to the
    independent level.
    """
    rng = np.random.default_rng(303)
    n_prompts, m = 100, 2
    prompt_means = rng.uniform(0.2, 0.7, n_prompts)

    def draws_for(policy: str, shift: float) -> FreshDrawDataset:
        samples = []
        for i, mu in enumerate(prompt_means):
            for draw in range(m):
                label = float(np.clip(mu + shift + rng.normal(0.0, 0.04), 0.0, 1.0))
                samples.append(
                    FreshDrawSample(
                        prompt_id=f"p{i}",
                        target_policy=policy,
                        judge_score=label,
                        oracle_label=label,
                        response=None,
                        draw_idx=draw,
                    )
                )
        return FreshDrawDataset(target_policy=policy, samples=samples)

    draws_a = draws_for("a", 0.0)
    draws_b = draws_for("b", 0.05)

    analytic = CalibratedDirectEstimator(
        ["a", "b"], None, inference_method="cluster_robust"
    )
    bootstrap = CalibratedDirectEstimator(
        ["a", "b"], None, inference_method="bootstrap", n_bootstrap=300
    )
    for estimator in (analytic, bootstrap):
        estimator.add_fresh_draws("a", draws_a)
        estimator.add_fresh_draws("b", draws_b)
    comparison_analytic = analytic.fit_and_estimate().compare_policies(0, 1)
    result_bootstrap = bootstrap.fit_and_estimate()
    comparison_bootstrap = result_bootstrap.compare_policies(0, 1)

    assert comparison_analytic["method"] == "paired_if_oua"
    assert comparison_bootstrap["method"] == "paired_bootstrap"
    _assert_se_ratio_in_band(
        float(comparison_bootstrap["se_difference"]),
        float(comparison_analytic["se_difference"]),
        "paired two-policy difference",
    )
    independent_se = float(
        np.sqrt(
            result_bootstrap.standard_errors[0] ** 2
            + result_bootstrap.standard_errors[1] ** 2
        )
    )
    assert comparison_bootstrap["se_difference"] < 0.5 * independent_se, (
        f"paired bootstrap SE {comparison_bootstrap['se_difference']:.5f} is not "
        f"below the independent combination {independent_se:.5f} — the joint "
        f"prompt-weight coupling across policies is broken"
    )


# ---------------------------------------------------------------------------
# SLOW layer: 95% CI coverage of the default bootstrap path
# ---------------------------------------------------------------------------

# E[0.2 + 0.6*S] with S ~ U(0, 1); prompt and row noise are mean-zero and the
# clip at [0, 1] is >= 4 sigma away, so its truncation bias is negligible.
BOOTSTRAP_TRUE_VALUE = 0.5
R_SLOW_BOOTSTRAP = 100


@pytest.mark.slow
def test_slow_bootstrap_ci_coverage() -> None:
    """Empirical 95% CI coverage of the default (bootstrap) inference path.

    100 seeded replications of the coupled partial-oracle design: labels on
    half the prompts fit the calibrator, the default estimator runs the
    positive-weight bootstrap with per-replicate refit, and the percentile
    CI must cover the known truth in >= 88% of replications (same loose
    bound as test_mc_coverage.py's slow cell for the analytic path; a
    2x-deflated bootstrap SE collapses coverage far below it).
    """
    covered = 0
    for replication in range(R_SLOW_BOOTSTRAP):
        rng = np.random.default_rng(71_000_000 + replication)
        draws = _clustered_draws(
            "target", n_prompts=150, draws_per_prompt=2, rng=rng, labeled_frac=0.5
        )
        labeled = [s for s in draws.samples if s.oracle_label is not None]
        calibrator = JudgeCalibrator(calibration_mode="monotone")
        calibrator.fit_cv(
            np.asarray([s.judge_score for s in labeled]),
            np.asarray([s.oracle_label for s in labeled]),
            prompt_ids=[s.prompt_id for s in labeled],
        )
        estimator = CalibratedDirectEstimator(
            target_policies=["target"],
            reward_calibrator=calibrator,
            n_bootstrap=200,
        )
        estimator.add_fresh_draws("target", draws)
        result = estimator.fit_and_estimate()
        assert result.metadata["inference"]["method"] == "cluster_bootstrap_refit"
        lower, upper = result.confidence_interval(alpha=0.05)
        if lower[0] <= BOOTSTRAP_TRUE_VALUE <= upper[0]:
            covered += 1

    coverage = covered / R_SLOW_BOOTSTRAP
    assert coverage >= 0.88, (
        f"default bootstrap path: 95% CI coverage {coverage:.1%} over "
        f"{R_SLOW_BOOTSTRAP} replications below 88%"
    )
