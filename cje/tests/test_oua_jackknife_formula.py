"""Regression tests for the oracle (OUA) jackknife variance formula.

The delete-one-fold jackknife variance is

    Var_cal = (K-1)/K * SUM_k (psi^(-k) - psi_bar)^2

(sum over folds, paper Alg. 6). A prior implementation divided by K (used the
mean over folds), understating the oracle-uncertainty variance by exactly a
factor of K and hence CI width contributions by ~sqrt(K). These tests pin the
correct formula.
"""

import numpy as np
import pytest

from cje.data.models import EstimationResult
from cje.diagnostics.robust_inference import (
    combine_cluster_and_oracle,
    oracle_jackknife_variance,
)
from cje.estimators.direct_method import CalibratedDirectEstimator


class _JackknifeStub(CalibratedDirectEstimator):
    """Minimal estimator exposing a fixed jackknife array."""

    def __init__(self, jack: np.ndarray):
        super().__init__(
            target_policies=["policy_a"],
            reward_calibrator=object(),  # non-None so OUA path runs
            oua_jackknife=True,
        )
        self._jack = np.asarray(jack, dtype=float)

    def get_oracle_jackknife(self, policy: str) -> np.ndarray:
        return self._jack


def test_helper_matches_sum_formula() -> None:
    jack = np.array([0.50, 0.60, 0.55, 0.52, 0.58])
    K = len(jack)
    expected = (K - 1) / K * float(np.sum((jack - jack.mean()) ** 2))
    assert oracle_jackknife_variance(jack) == pytest.approx(expected, rel=1e-12)


def test_helper_is_k_times_the_old_mean_form() -> None:
    """The buggy mean-over-folds form understated variance by exactly K."""
    rng = np.random.default_rng(7)
    for K in (2, 3, 5, 10):
        jack = rng.normal(0.5, 0.1, size=K)
        mean_form = (K - 1) / K * float(np.mean((jack - jack.mean()) ** 2))
        assert oracle_jackknife_variance(jack) == pytest.approx(
            K * mean_form, rel=1e-12
        )


def test_helper_degenerate_inputs() -> None:
    assert oracle_jackknife_variance(np.array([])) == 0.0
    assert oracle_jackknife_variance(np.array([0.5])) == 0.0
    assert oracle_jackknife_variance(np.array([0.5, 0.5, 0.5])) == 0.0


def test_helper_unbiased_monte_carlo() -> None:
    """Delete-one-fold jackknife should estimate Var(theta_hat) unbiasedly.

    DGP: theta_hat = mean of n iid draws, K equal folds. The mean-form would
    come out at ~1/K of the truth here.
    """
    rng = np.random.default_rng(0)
    n, K, reps = 200, 5, 2000
    folds = np.arange(n) % K
    estimates = np.empty(reps)
    jack_vars = np.empty(reps)
    for r in range(reps):
        y = rng.normal(0.5, 0.3, size=n)
        estimates[r] = y.mean()
        jack = np.array([y[folds != k].mean() for k in range(K)])
        jack_vars[r] = oracle_jackknife_variance(jack)
    true_var = estimates.var()
    ratio = jack_vars.mean() / true_var
    # Unbiased up to MC noise; the old mean-form would give ratio ~= 0.2.
    assert 0.85 < ratio < 1.15


def test_apply_oua_jackknife_adds_sum_form_variance() -> None:
    jack = np.array([0.42, 0.47, 0.44, 0.50, 0.45])
    se_base = 0.03
    est = _JackknifeStub(jack)
    result = EstimationResult(
        estimates=np.array([0.45]),
        standard_errors=np.array([se_base]),
        n_samples_used={"policy_a": 100},
        method="stub",
        influence_functions=None,
        diagnostics=None,
    )
    est._apply_oua_jackknife(result)

    var_orc = oracle_jackknife_variance(jack)
    expected_se = np.sqrt(se_base**2 + var_orc)
    assert result.standard_errors[0] == pytest.approx(expected_se, rel=1e-12)
    comps = result.metadata["se_components"]
    assert comps["includes_oracle_uncertainty"] is True
    assert comps["oracle_variance_per_policy"]["policy_a"] == pytest.approx(
        var_orc, rel=1e-12
    )
    assert comps["oracle_jackknife_counts"]["policy_a"] == len(jack)


def test_welch_satterthwaite_combines_variance_components() -> None:
    se_cluster = 0.03
    var_oracle = 0.0004
    df_cluster = 39
    folds = 5
    se, df = combine_cluster_and_oracle(se_cluster, df_cluster, var_oracle, folds)
    expected_df = (se_cluster**2 + var_oracle) ** 2 / (
        se_cluster**4 / df_cluster + var_oracle**2 / (folds - 1)
    )
    assert se == pytest.approx(np.sqrt(se_cluster**2 + var_oracle))
    assert df == pytest.approx(expected_df)
    assert not float(df).is_integer()


@pytest.mark.parametrize(
    ("se_cluster", "df_cluster", "var_oracle", "folds", "expected_se", "expected_df"),
    [
        (0.03, 39, 0.0, 5, 0.03, 39.0),
        (0.0, 39, 0.0004, 5, 0.02, 4.0),
        (0.0, 39, 0.0, 0, 0.0, 39.0),
        (0.03, 0, 0.0, 0, 0.03, 1.0),
        (0.03, 39, -0.1, 5, 0.03, 39.0),
    ],
)
def test_welch_satterthwaite_component_limits(
    se_cluster: float,
    df_cluster: int,
    var_oracle: float,
    folds: int,
    expected_se: float,
    expected_df: float,
) -> None:
    se, df = combine_cluster_and_oracle(se_cluster, df_cluster, var_oracle, folds)
    assert se == pytest.approx(expected_se)
    assert df == pytest.approx(expected_df)


def test_welch_satterthwaite_nonfinite_component_is_unavailable() -> None:
    se, df = combine_cluster_and_oracle(float("nan"), 19, 0.0, 0)
    assert np.isnan(se)
    assert df == 19.0

    se, df = combine_cluster_and_oracle(0.03, 19, float("nan"), 5)
    assert np.isnan(se)
    assert df == 19.0


def test_positive_oracle_variance_requires_two_folds() -> None:
    with pytest.raises(ValueError, match="at least two jackknife folds"):
        combine_cluster_and_oracle(0.03, 19, 0.0001, 1)


def test_store_df_info_uses_actual_variance_shares() -> None:
    jack = np.array([0.42, 0.47, 0.44, 0.50, 0.45])
    se_cluster = 0.03
    est = _JackknifeStub(jack)
    est._df_cluster = {"policy_a": 39}
    est._se_methods = {"policy_a": "cluster_robust"}
    est._n_clusters = {"policy_a": 40}
    result = EstimationResult(
        estimates=np.array([0.45]),
        standard_errors=np.array([se_cluster]),
        n_samples_used={"policy_a": 100},
        method="stub",
        influence_functions=None,
        diagnostics=None,
    )
    est._apply_oua_jackknife(result)
    est._store_df_info(result, sampling_ses=[se_cluster])

    var_oracle = oracle_jackknife_variance(jack)
    _, expected_df = combine_cluster_and_oracle(se_cluster, 39, var_oracle, len(jack))
    stored = result.metadata["degrees_of_freedom"]["policy_a"]
    assert stored["df"] == pytest.approx(expected_df)
    assert stored["df"] != len(jack) - 1
    assert stored["df_method"] == "welch_satterthwaite"
