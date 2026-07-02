"""Regression tests: unified DR-family standard errors.

Four prior defects are pinned here:

1. DR-CPO double-counted fresh-draw Monte Carlo variance: the influence
   functions are built from per-prompt MEANS of the fresh-draw predictions, so
   the IF variance already contains the within-prompt MC noise; mc_var was
   nevertheless re-added in quadrature (up to ~sqrt(2)x SE inflation at M=1).
   MC numbers are now diagnostics-only metadata.

2. TMLE and MRDR computed naive IID SEs (np.std/sqrt(n)) with no cluster-robust
   correction and no df metadata (z-based CIs), while DR-CPO used cluster-robust
   SEs with t-based CIs. All three now share DREstimator._compute_policy_se.

3. StackedDR's MC-aware objective penalized only components that *reported* MC
   diagnostics (only dr-cpo), tilting weights for reasons unrelated to
   efficiency, and re-added mc_var to the stacked SE. The MC term is gone from
   both; the kwargs are accepted with a DeprecationWarning.

4. compute_robust_inference returned bootstrap/cluster CIs centered at ~0
   (mean-centered IFs, unshifted) and the raw-data path used a constant
   placeholder statistic (SE ~1e-16, p=0). CIs are now re-centered on the
   point estimate and the data-only path raises ValueError.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pytest
from scipy import stats

from cje.calibration import calibrate_dataset
from cje.data.folds import get_fold
from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample
from cje.data.models import Dataset, EstimationResult, Sample
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.diagnostics.robust_inference import (
    cluster_robust_se,
    compute_robust_inference,
)
from cje.estimators.dr_base import DRCPOEstimator, DREstimator
from cje.estimators.mrdr import MRDREstimator
from cje.estimators.stacking import StackedDREstimator
from cje.estimators.tmle import TMLEEstimator

POLICY = "target"


def _make_dr_setup(
    n: int = 500,
    m_draws: int = 1,
    oracle_frac: float = 0.25,
    seed: int = 11,
) -> Tuple[PrecomputedSampler, Any, FreshDrawDataset]:
    """Synthetic logged data + fresh draws with a mean-one tilted target policy.

    Judge scores S ~ U(0,1); true outcome mu(S) = 0.2 + 0.6*S; oracle labels on
    a random subset; target policy tilts toward high S. Fresh draws are sampled
    from the tilted score density w(S) = 0.4 + 1.2*S via rejection sampling.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        s = float(rng.uniform())
        mu = 0.2 + 0.6 * s
        oracle = (
            float(np.clip(mu + rng.normal(0, 0.1), 0, 1))
            if rng.uniform() < oracle_frac
            else None
        )
        samples.append(
            Sample(
                prompt_id=f"p{i}",
                prompt=f"question {i}",
                response=f"answer {i}",
                reward=None,
                base_policy_logprob=-10.0,
                target_policy_logprobs={POLICY: -10.0 + 2.0 * (s - 0.5)},
                judge_score=s,
                oracle_label=oracle,
            )
        )

    dataset = Dataset(samples=samples, target_policies=[POLICY])
    calibrated, cal_result = calibrate_dataset(
        dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
        enable_cross_fit=True,
        n_folds=5,
    )
    sampler = PrecomputedSampler(calibrated)

    fresh_samples = []
    for i in range(n):
        for d in range(m_draws):
            while True:
                sp = float(rng.uniform())
                if rng.uniform(0, 1.6) <= 0.4 + 1.2 * sp:
                    break
            fresh_samples.append(
                FreshDrawSample(
                    prompt_id=f"p{i}",
                    judge_score=sp,
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
    return sampler, cal_result.calibrator, fresh


# ---------------------------------------------------------------------------
# 1. MC variance is no longer composed into DR SEs
# ---------------------------------------------------------------------------


def test_dr_cpo_se_excludes_separate_mc_term_at_m1() -> None:
    """At M=1 the old composition re-added the full DM variance as mc_var."""
    sampler, calibrator, fresh = _make_dr_setup(m_draws=1)
    est = DRCPOEstimator(sampler, reward_calibrator=calibrator, n_folds=5)
    est.add_fresh_draws(POLICY, fresh)
    result = est.fit_and_estimate()

    comps = result.metadata["se_components"]
    assert comps["includes_oracle_uncertainty"] is True
    assert comps["includes_mc_variance"] is False
    assert comps["mc_variance_in_if"] is True

    se = float(result.standard_errors[0])
    se_if = comps["se_if_per_policy"][POLICY]
    var_oracle = comps["oracle_variance_per_policy"][POLICY]

    # SE is exactly IF (cluster-robust) + oracle in quadrature — nothing else
    assert se == pytest.approx(np.sqrt(se_if**2 + var_oracle), rel=1e-9)

    # MC diagnostics still exist, flagged as excluded from the SE
    mc = result.metadata["mc_variance_diagnostics"][POLICY]
    assert mc["mc_var"] > 0
    assert mc["included_in_se"] is False
    assert mc["fallback_used"] is True  # all M=1
    assert mc["min_draws_per_prompt"] == 1

    # The old composition sqrt(se_if^2 + var_oracle + mc_var) was materially
    # inflated: at M=1 mc_var duplicates the DM-part of the IF variance.
    assert np.sqrt(se_if**2 + mc["mc_var"]) > 1.2 * se_if


def test_dr_cpo_mc_diagnostics_exact_at_m2() -> None:
    sampler, calibrator, fresh = _make_dr_setup(m_draws=2, seed=13)
    est = DRCPOEstimator(sampler, reward_calibrator=calibrator, n_folds=5)
    est.add_fresh_draws(POLICY, fresh)
    result = est.fit_and_estimate()

    mc = result.metadata["mc_variance_diagnostics"][POLICY]
    assert mc["fallback_used"] is False
    assert mc["fallback_method"] == "exact"
    assert mc["included_in_se"] is False

    # Still no separate MC term in the SE
    comps = result.metadata["se_components"]
    se_if = comps["se_if_per_policy"][POLICY]
    var_oracle = comps["oracle_variance_per_policy"][POLICY]
    assert float(result.standard_errors[0]) == pytest.approx(
        np.sqrt(se_if**2 + var_oracle), rel=1e-9
    )


# ---------------------------------------------------------------------------
# 2. TMLE / MRDR / DR-CPO share the same SE construction
# ---------------------------------------------------------------------------


def test_dr_family_ses_use_shared_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """All three DR estimators must route SEs through _compute_policy_se and
    report cluster-robust + oracle SEs with t-based df metadata."""
    sampler, calibrator, fresh = _make_dr_setup(m_draws=2, seed=7)

    helper_calls: Dict[str, float] = {}
    orig = DREstimator._compute_policy_se

    def spy(
        self: DREstimator,
        policy: str,
        if_contributions: np.ndarray,
        fold_ids: Any,
        alpha: float = 0.05,
    ) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
        out = orig(self, policy, if_contributions, fold_ids, alpha)
        helper_calls[type(self).__name__] = float(out[0])
        return out

    monkeypatch.setattr(DREstimator, "_compute_policy_se", spy)

    estimators = {
        "DRCPOEstimator": DRCPOEstimator(
            sampler, reward_calibrator=calibrator, n_folds=5
        ),
        "TMLEEstimator": TMLEEstimator(
            sampler, reward_calibrator=calibrator, n_folds=5
        ),
        "MRDREstimator": MRDREstimator(
            sampler, reward_calibrator=calibrator, n_folds=5
        ),
    }

    for cls_name, est in estimators.items():
        est.add_fresh_draws(POLICY, fresh)
        result = est.fit_and_estimate()

        # The helper was used, and its output IS the reported SE
        assert cls_name in helper_calls, f"{cls_name} bypassed _compute_policy_se"
        assert float(result.standard_errors[0]) == pytest.approx(
            helper_calls[cls_name], rel=1e-12
        ), f"{cls_name} reported an SE different from the shared helper's"

        # df metadata present -> confidence_interval() uses t, not z
        df_entry = result.metadata["degrees_of_freedom"][POLICY]
        for key in ("df", "t_critical", "n_clusters"):
            assert key in df_entry, f"{cls_name} missing df metadata '{key}'"
        assert df_entry["df"] >= 1

        lo, hi = result.confidence_interval(alpha=0.05)
        t_crit = stats.t.ppf(0.975, df_entry["df"])
        se = float(result.standard_errors[0])
        assert hi[0] - lo[0] == pytest.approx(
            2 * t_crit * se, rel=1e-9
        ), f"{cls_name} CI is not t-based with the stored df"

        # se_components decomposition present with oracle variance included once
        comps = result.metadata["se_components"]
        assert comps["includes_oracle_uncertainty"] is True
        assert comps["includes_mc_variance"] is False
        se_if = comps["se_if_per_policy"][POLICY]
        var_oracle = comps["oracle_variance_per_policy"][POLICY]
        assert var_oracle > 0, f"{cls_name} lost the oracle-jackknife variance"
        assert se == pytest.approx(np.sqrt(se_if**2 + var_oracle), rel=1e-9)

        # MC diagnostics populated for every DR estimator now
        assert POLICY in result.metadata["mc_variance_diagnostics"], cls_name


def test_dr_cpo_and_tmle_se_is_cluster_robust_not_naive() -> None:
    """Recompute the cluster-robust IF SE directly and match se_if exactly.

    DR-CPO and TMLE cluster on the unified prompt-id-hash folds
    (get_fold(prompt_id, n_folds, seed=42)), so the expected value can be
    reconstructed from the influence functions alone.
    """
    sampler, calibrator, fresh = _make_dr_setup(m_draws=2, seed=5)

    for est in (
        DRCPOEstimator(sampler, reward_calibrator=calibrator, n_folds=5),
        TMLEEstimator(sampler, reward_calibrator=calibrator, n_folds=5),
    ):
        est.add_fresh_draws(POLICY, fresh)
        result = est.fit_and_estimate()

        assert result.influence_functions is not None
        ifs = result.influence_functions[POLICY]
        prompt_ids = result.metadata["if_sample_indices"][POLICY]
        fold_ids = np.array([get_fold(str(pid), 5, 42) for pid in prompt_ids])

        expected = cluster_robust_se(
            data=ifs,
            cluster_ids=fold_ids,
            statistic_fn=lambda x: float(np.mean(x)),
            influence_fn=lambda x: x,
            alpha=0.05,
        )
        se_if = result.metadata["se_components"]["se_if_per_policy"][POLICY]
        assert se_if == pytest.approx(
            float(expected["se"]), rel=1e-9
        ), f"{type(est).__name__} se_if is not the cluster-robust IF SE"
        # And it is NOT the naive IID SE plugged into the total
        naive = float(np.std(ifs, ddof=1) / np.sqrt(len(ifs)))
        var_oracle = result.metadata["se_components"]["oracle_variance_per_policy"][
            POLICY
        ]
        naive_total = float(np.sqrt(naive**2 + var_oracle))
        reported = float(result.standard_errors[0])
        cluster_total = float(np.sqrt(float(expected["se"]) ** 2 + var_oracle))
        assert reported == pytest.approx(cluster_total, rel=1e-9)
        if abs(naive - float(expected["se"])) > 1e-12:
            assert reported != pytest.approx(naive_total, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. Stacking: MC asymmetry removed
# ---------------------------------------------------------------------------


class _DummySampler:
    target_policies = [POLICY]


def _fake_component_result(method: str, metadata: Dict[str, Any]) -> EstimationResult:
    return EstimationResult(
        estimates=np.array([0.5]),
        standard_errors=np.array([0.01]),
        n_samples_used={POLICY: 500},
        method=method,
        influence_functions=None,
        diagnostics=None,
        metadata=metadata,
    )


def test_stacking_weights_ignore_reported_mc_diagnostics() -> None:
    """Identical IFs must get ~equal weights even if only one component
    reports MC diagnostics (the old rank-1 penalty hit only that one)."""
    est = StackedDREstimator(sampler=_DummySampler())  # type: ignore[arg-type]

    rng = np.random.default_rng(3)
    base_if = rng.normal(0.0, 1.0, size=500)
    base_if -= base_if.mean()
    IF_matrix = np.column_stack([base_if, base_if, base_if])

    # Only dr-cpo reports MC diagnostics — exactly the old asymmetry
    est.component_results = {
        "dr-cpo": _fake_component_result(
            "dr_cpo",
            {"mc_variance_diagnostics": {POLICY: {"mc_var": 0.05}}},
        ),
        "tmle": _fake_component_result("tmle", {}),
        "mrdr": _fake_component_result("mrdr", {}),
    }

    weights, diagnostics = est._compute_optimal_weights(
        IF_matrix, ["dr-cpo", "tmle", "mrdr"], POLICY
    )
    np.testing.assert_allclose(weights, np.ones(3) / 3, atol=1e-6)
    assert "mc_aware" not in diagnostics


def test_stacking_mc_kwargs_deprecated_but_accepted() -> None:
    with pytest.warns(DeprecationWarning, match="include_mc_in_objective"):
        StackedDREstimator(
            sampler=_DummySampler(), include_mc_in_objective=True  # type: ignore[arg-type]
        )
    with pytest.warns(DeprecationWarning, match="mc_lambda"):
        StackedDREstimator(sampler=_DummySampler(), mc_lambda=2.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. compute_robust_inference CIs
# ---------------------------------------------------------------------------


def _centered_ifs(seed: int = 0, n: int = 300) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ifs = np.column_stack([rng.normal(0, 1.0, size=n), rng.normal(0, 0.5, size=n)])
    return np.asarray(ifs - ifs.mean(axis=0, keepdims=True))


@pytest.mark.parametrize(
    "method", ["stationary_bootstrap", "moving_block", "cluster", "classical"]
)
def test_robust_inference_cis_contain_their_estimates(method: str) -> None:
    np.random.seed(0)  # bootstrap paths use the global RNG
    ifs = _centered_ifs()
    estimates = np.array([0.7, 0.4])
    cluster_ids = np.arange(len(ifs)) % 5

    res = compute_robust_inference(
        estimates,
        influence_functions=ifs,
        method=method,
        cluster_ids=cluster_ids,
        n_bootstrap=500,
    )

    assert len(res["robust_cis"]) == 2
    for i, (lo, hi) in enumerate(res["robust_cis"]):
        assert lo < estimates[i] < hi, (
            f"{method}: CI ({lo:.3f}, {hi:.3f}) does not contain "
            f"estimate {estimates[i]:.3f}"
        )
        # The interval is a local neighborhood of the estimate, not of zero
        assert abs((lo + hi) / 2 - estimates[i]) < 0.1
        assert res["robust_ses"][i] > 1e-6  # no degenerate zero-width CI
    # p-values remain finite and in range
    for p in res["p_values"]:
        assert 0.0 <= p <= 1.0


def test_robust_inference_data_path_raises() -> None:
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="influence_functions"):
        compute_robust_inference(
            np.array([0.5]),
            data=rng.normal(0.5, 0.1, size=100),
            method="stationary_bootstrap",
        )
    with pytest.raises(ValueError, match="influence_functions"):
        compute_robust_inference(np.array([0.5]), method="cluster")
