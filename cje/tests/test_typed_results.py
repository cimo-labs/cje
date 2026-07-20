"""Typed accessors on EstimationResult (0.5.0).

Covers the typed layer added over the metadata mirrors (which remain the
serialized source of truth and keep being written):

- target_policies property (reads metadata["target_policies"])
- gates property (typed view of metadata["reliability_gates"])
- best_policy() -> PolicyVerdict (gate-aware; replaced the naive int argmax)
- summary() compact text report
- ci_info typed CI record + the D9 alpha-mismatch warning (both the ci_info
  path and the legacy metadata-sniffing path)
- compare_policies (0.5.1 four-path dispatch: paired_bootstrap >
  paired_if_oua > paired_if_legacy > independent_conservative) and
  compare_all_policies (+ Benjamini-Hochberg adjustment)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from scipy import stats

from cje import analyze_dataset
from cje.data.models import (
    CIInfo,
    EstimationResult,
    GateResult,
    PolicyVerdict,
)

pytestmark = pytest.mark.unit


def _make_result(
    estimates: List[float],
    policies: List[str],
    gates: Optional[Dict[str, Dict[str, Any]]] = None,
    standard_errors: Optional[List[float]] = None,
    influence_functions: Optional[Dict[str, np.ndarray]] = None,
    metadata_extra: Optional[Dict[str, Any]] = None,
    ci_info: Optional[CIInfo] = None,
    bootstrap_samples: Optional[np.ndarray] = None,
) -> EstimationResult:
    metadata: Dict[str, Any] = {"target_policies": policies}
    if gates is not None:
        metadata["reliability_gates"] = gates
    if metadata_extra:
        metadata.update(metadata_extra)
    return EstimationResult(
        estimates=np.array(estimates, dtype=float),
        standard_errors=np.array(
            standard_errors if standard_errors is not None else [0.05] * len(estimates)
        ),
        n_samples_used={p: 100 for p in policies},
        method="calibrated_direct",
        influence_functions=influence_functions,
        diagnostics=None,
        metadata=metadata,
        ci_info=ci_info,
        bootstrap_samples=bootstrap_samples,
    )


def _gate(flagged: bool, reasons: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "flagged": flagged,
        "refused": False,
        "refuse_level_claims": flagged,
        "reasons": reasons or ([] if not flagged else ["boundary: 12.0%"]),
    }


def _labeled_draws(n: int = 40) -> Dict[str, List[Dict[str, Any]]]:
    records = []
    for i in range(n):
        s = 0.2 + 0.6 * i / (n - 1)
        record: Dict[str, Any] = {"prompt_id": f"p{i:03d}", "judge_score": s}
        if i % 2 == 0:
            record["oracle_label"] = 0.25 + 0.5 * s
        records.append(record)
    return {"pi": records}


# ---------------------------------------------------------------------------
# target_policies + gates properties
# ---------------------------------------------------------------------------


class TestTargetPoliciesProperty:
    def test_reads_metadata(self) -> None:
        result = _make_result([0.5, 0.7], ["a", "b"])
        assert result.target_policies == ["a", "b"]

    def test_absent_metadata_returns_empty_list(self) -> None:
        result = _make_result([0.5], ["a"])
        result.metadata.pop("target_policies")
        assert result.target_policies == []


class TestGatesProperty:
    def test_typed_view_of_reliability_gates(self) -> None:
        result = _make_result(
            [0.5, 0.7],
            ["a", "b"],
            gates={"a": _gate(False), "b": _gate(True, ["boundary: 12.0%"])},
        )
        gates = result.gates
        assert set(gates) == {"a", "b"}
        assert isinstance(gates["b"], GateResult)
        assert gates["b"].policy == "b"
        assert gates["b"].flagged is True
        assert gates["b"].refuse_level_claims is True
        assert gates["b"].reasons == ["boundary: 12.0%"]
        assert gates["a"].flagged is False

    def test_no_gates_metadata_returns_empty(self) -> None:
        result = _make_result([0.5], ["a"])
        assert result.gates == {}

    def test_end_to_end_gates_mirror_metadata(self) -> None:
        result = analyze_dataset(
            fresh_draws_data=_labeled_draws(),
            estimator_config={"inference_method": "cluster_robust"},
        )
        # Metadata mirror is still written; typed view matches it
        raw = result.metadata["reliability_gates"]
        gates = result.gates
        assert set(gates) == set(raw)
        for policy, info in raw.items():
            assert gates[policy].flagged == info["flagged"]


# ---------------------------------------------------------------------------
# best_policy() -> PolicyVerdict
# ---------------------------------------------------------------------------


class TestBestPolicyVerdict:
    def test_unflagged_argmax_wins(self) -> None:
        result = _make_result(
            [0.5, 0.7],
            ["base", "good"],
            gates={"base": _gate(False), "good": _gate(False)},
        )
        verdict = result.best_policy()
        assert isinstance(verdict, PolicyVerdict)
        assert verdict.name == "good"
        assert verdict.index == 1
        assert verdict.estimate == pytest.approx(0.7)
        assert verdict.flagged is False
        assert verdict.all_flagged is False
        assert verdict.runner_up is None

    def test_flagged_argmax_is_demoted_by_default(self) -> None:
        result = _make_result(
            [0.771, 0.756, 0.763],
            ["unhelpful", "base", "clone"],
            gates={
                "unhelpful": _gate(True),
                "base": _gate(False),
                "clone": _gate(False),
            },
        )
        verdict = result.best_policy()
        assert verdict.name == "clone"
        assert verdict.index == 2
        assert verdict.estimate == pytest.approx(0.763)
        assert verdict.flagged is False
        assert verdict.all_flagged is False
        # Loud divergence: the demoted raw argmax and why it was flagged
        # travel with the verdict.
        assert verdict.runner_up == "unhelpful"
        assert verdict.runner_up_reasons == ["boundary: 12.0%"]

    def test_reliable_only_false_returns_raw_argmax(self) -> None:
        result = _make_result(
            [0.771, 0.756],
            ["unhelpful", "base"],
            gates={"unhelpful": _gate(True), "base": _gate(False)},
        )
        verdict = result.best_policy(reliable_only=False)
        assert verdict.name == "unhelpful"
        assert verdict.flagged is True
        assert verdict.all_flagged is False
        assert verdict.runner_up is None
        assert verdict.runner_up_reasons is None

    def test_all_flagged_returns_argmax_marked(self) -> None:
        result = _make_result(
            [0.9, 0.6], ["a", "b"], gates={"a": _gate(True), "b": _gate(True)}
        )
        verdict = result.best_policy()
        assert verdict.name == "a"
        assert verdict.flagged is True
        assert verdict.all_flagged is True
        assert verdict.runner_up is None

    def test_critical_status_demotes_point_winner(self) -> None:
        from cje.diagnostics import DirectDiagnostics, Status

        diag = DirectDiagnostics(
            estimator_type="Direct",
            method="calibrated_direct",
            n_samples_total=100,
            n_samples_valid=100,
            policies=["bad", "ok"],
            estimates={"bad": 0.9, "ok": 0.6},
            standard_errors={"bad": 0.1, "ok": 0.05},
            n_samples_used={"bad": 100, "ok": 100},
            status_per_policy={"bad": Status.CRITICAL, "ok": Status.GOOD},
        )
        result = _make_result([0.9, 0.6], ["bad", "ok"])
        result.diagnostics = diag
        verdict = result.best_policy()
        assert verdict.name == "ok"
        assert verdict.flagged is False
        assert verdict.runner_up == "bad"
        assert verdict.runner_up_reasons == ["diagnostics status CRITICAL"]

    def test_demotion_warning_logged_once_per_result(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The demotion warning fires on the first best_policy() call only:
        summary() and renderers re-derive the verdict without re-warning."""
        result = _make_result(
            [0.771, 0.756],
            ["unhelpful", "base"],
            gates={"unhelpful": _gate(True), "base": _gate(False)},
        )
        with caplog.at_level(logging.WARNING, logger="cje.data.models"):
            first = result.best_policy()
            result.summary()  # re-derives the verdict internally
            second = result.best_policy()
        warnings_logged = [
            record
            for record in caplog.records
            if "raw argmax 'unhelpful' was flagged" in record.message
        ]
        assert len(warnings_logged) == 1
        # The verdict itself stays loud and identical on repeat calls.
        assert first == second
        assert second.runner_up == "unhelpful"

    def test_all_nan_raises(self) -> None:
        result = _make_result([float("nan"), float("nan")], ["a", "b"])
        with pytest.raises(ValueError, match="No usable estimates"):
            result.best_policy()

    def test_no_policies_raises(self) -> None:
        result = _make_result([0.5], ["a"])
        result.metadata.pop("target_policies")
        with pytest.raises(ValueError, match="target_policies"):
            result.best_policy()


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_lists_policies_cis_and_best(self) -> None:
        result = _make_result(
            [0.5, 0.7],
            ["base", "good"],
            gates={"base": _gate(False), "good": _gate(False)},
        )
        text = result.summary()
        assert isinstance(text, str)
        assert "calibrated_direct" in text
        assert "base" in text and "good" in text
        assert "95% CI [" in text
        assert "Best by point estimate: good" in text

    def test_summary_shows_flagged_winner_and_reliable_fallback(self) -> None:
        result = _make_result(
            [0.771, 0.756],
            ["unhelpful", "base"],
            gates={"unhelpful": _gate(True), "base": _gate(False)},
        )
        text = result.summary()
        # The flagged raw winner stays visible with its limitations...
        assert "[gate: FLAGGED]" in text
        assert "Best by point estimate: unhelpful" in text
        assert "flagged by the reliability gates" in text
        # ...and the divergence from the returned reliable winner is loud.
        assert "Best reliable policy: base" in text
        assert "raw argmax unhelpful was flagged (boundary: 12.0%)" in text
        assert "reliable_only=False" in text

    def test_docstring_example_works_end_to_end(self) -> None:
        """The cje/__init__.py docstring example: print(results.summary())."""
        results = analyze_dataset(
            fresh_draws_data=_labeled_draws(),
            estimator_config={"inference_method": "cluster_robust"},
        )
        text = results.summary()
        assert "CJE Estimation Results" in text
        assert "pi" in text


# ---------------------------------------------------------------------------
# ci_info + D9 alpha warning
# ---------------------------------------------------------------------------


class TestCIInfo:
    def test_bootstrap_path_writes_percentile_ci_info(self) -> None:
        result = analyze_dataset(
            fresh_draws_data=_labeled_draws(),
            estimator_config={"n_bootstrap": 100},
        )
        assert result.ci_info is not None
        assert result.ci_info.method == "percentile"
        assert result.ci_info.alpha == pytest.approx(0.05)
        # ci_info mirrors the metadata mirror exactly
        boot_ci = result.metadata["bootstrap_ci"]
        assert result.ci_info.lower == pytest.approx(boot_ci["lower"])
        assert result.ci_info.upper == pytest.approx(boot_ci["upper"])
        lower, upper = result.confidence_interval()
        np.testing.assert_allclose(lower, np.array(boot_ci["lower"]))
        np.testing.assert_allclose(upper, np.array(boot_ci["upper"]))

    def test_analytic_path_writes_t_ci_info(self) -> None:
        result = analyze_dataset(
            fresh_draws_data=_labeled_draws(),
            estimator_config={"inference_method": "cluster_robust"},
        )
        assert result.ci_info is not None
        assert result.ci_info.method == "t"
        assert result.ci_info.df_per_policy == result.metadata["degrees_of_freedom"]
        # ci_info path and the legacy metadata-sniffing path agree
        lower, upper = result.confidence_interval()
        result_no_ci = result.model_copy(update={"ci_info": None})
        legacy_lower, legacy_upper = result_no_ci.confidence_interval()
        np.testing.assert_allclose(lower, legacy_lower)
        np.testing.assert_allclose(upper, legacy_upper)

    def test_alpha_mismatch_warns_on_ci_info_path(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        result = _make_result(
            [0.5],
            ["a"],
            ci_info=CIInfo(method="percentile", alpha=0.05, lower=[0.4], upper=[0.6]),
        )
        with caplog.at_level(logging.WARNING):
            lower, upper = result.confidence_interval(alpha=0.10)
        assert any("alpha" in r.message for r in caplog.records)
        np.testing.assert_allclose(lower, [0.4])
        np.testing.assert_allclose(upper, [0.6])

    def test_alpha_mismatch_warns_on_legacy_metadata_path(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        result = _make_result(
            [0.5],
            ["a"],
            metadata_extra={
                "bootstrap_ci": {
                    "method": "percentile",
                    "alpha": 0.05,
                    "lower": [0.4],
                    "upper": [0.6],
                }
            },
        )
        assert result.ci_info is None
        with caplog.at_level(logging.WARNING):
            lower, upper = result.confidence_interval(alpha=0.10)
        assert any("alpha" in r.message for r in caplog.records)
        np.testing.assert_allclose(lower, [0.4])
        np.testing.assert_allclose(upper, [0.6])

    def test_matching_alpha_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        result = _make_result(
            [0.5],
            ["a"],
            ci_info=CIInfo(method="percentile", alpha=0.05, lower=[0.4], upper=[0.6]),
        )
        with caplog.at_level(logging.WARNING):
            result.confidence_interval(alpha=0.05)
        assert not any("alpha" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# compare_policies (previously untested)
# ---------------------------------------------------------------------------


class TestComparePolicies:
    def test_influence_function_z_test_is_deterministic(self) -> None:
        if_a = np.array([0.1, -0.1, 0.2, -0.2, 0.0])
        if_b = np.array([0.05, -0.05, 0.1, -0.1, 0.0])
        result = _make_result(
            [0.7, 0.6],
            ["a", "b"],
            standard_errors=[0.05, 0.04],
            influence_functions={"a": if_a, "b": if_b},
        )

        comparison = result.compare_policies(0, 1)

        diff_if = if_a - if_b
        expected_se = float(np.std(diff_if, ddof=1) / np.sqrt(len(diff_if)))
        expected_z = 0.1 / expected_se
        expected_p = 2 * (1 - stats.norm.cdf(abs(expected_z)))

        assert comparison["difference"] == pytest.approx(0.1)
        assert comparison["se_difference"] == pytest.approx(expected_se)
        assert comparison["z_score"] == pytest.approx(expected_z)
        assert comparison["p_value"] == pytest.approx(expected_p)
        assert comparison["significant"] == (expected_p < 0.05)
        assert comparison["used_influence"] is True
        assert comparison["method"] == "paired_if_legacy"

    def test_falls_back_to_conservative_se_without_ifs(self) -> None:
        result = _make_result([0.7, 0.6], ["a", "b"], standard_errors=[0.05, 0.04])
        comparison = result.compare_policies(0, 1)
        assert comparison["se_difference"] == pytest.approx(np.sqrt(0.05**2 + 0.04**2))
        assert comparison["used_influence"] is False
        assert comparison["method"] == "independent_conservative"


# ---------------------------------------------------------------------------
# compare_policies: paired bootstrap path (0.5.1)
# ---------------------------------------------------------------------------


def _paired_matrix(deltas: np.ndarray, base: float = 0.5) -> np.ndarray:
    """(B, 2) replicate matrix whose column-0 minus column-1 equals deltas."""
    return np.column_stack([base + deltas, np.full(len(deltas), base)])


class TestComparePoliciesPairedBootstrap:
    def test_paired_bootstrap_deterministic_hand_computed(self) -> None:
        """Fixed 200x2 matrix with closed-form std / percentiles / sign counts."""
        deltas = np.linspace(-0.5, 1.5, 200)
        result = _make_result(
            [0.72, 0.60], ["a", "b"], bootstrap_samples=_paired_matrix(deltas)
        )
        comparison = result.compare_policies(0, 1)

        assert comparison["method"] == "paired_bootstrap"
        # difference stays the point-estimate difference, NOT the delta mean (0.5)
        assert comparison["difference"] == pytest.approx(0.12)
        # sample std of a linspace: h * sqrt(n(n+1)/12) with h = 2/199
        expected_se = (2 / 199) * np.sqrt(200 * 201 / 12)
        assert comparison["se_difference"] == pytest.approx(expected_se)
        # linear-interpolation percentiles land exactly on the grid:
        # 2.5%: position 0.025*199 = 4.975 -> deltas[4] + 0.975*h = -0.45
        # 97.5%: position 194.025 -> deltas[194] + 0.025*h = 1.45
        assert comparison["ci_lower"] == pytest.approx(-0.45)
        assert comparison["ci_upper"] == pytest.approx(1.45)
        # 50 deltas <= 0 (k <= 49.75), 150 >= 0, no exact zeros:
        # p = min(1, 2*min(1+50, 1+150)/201) = 102/201
        assert comparison["p_value"] == pytest.approx(102 / 201)
        assert comparison["significant"] is False
        assert comparison["z_score"] == pytest.approx(0.12 / expected_se)
        assert comparison["n_replicates"] == 200
        assert comparison["alpha"] == pytest.approx(0.05)
        assert comparison["used_influence"] is False

    def test_p_value_floor_is_2_over_b_plus_1(self) -> None:
        deltas = np.linspace(0.01, 0.02, 150)  # every delta positive
        result = _make_result(
            [0.7, 0.6], ["a", "b"], bootstrap_samples=_paired_matrix(deltas)
        )
        comparison = result.compare_policies(0, 1)
        assert comparison["p_value"] == pytest.approx(2 / 151)
        assert comparison["significant"] is True

    def test_nan_replicates_are_filtered(self) -> None:
        deltas = np.concatenate([np.linspace(-0.1, 0.3, 150), np.full(60, np.nan)])
        result = _make_result(
            [0.7, 0.6], ["a", "b"], bootstrap_samples=_paired_matrix(deltas)
        )
        comparison = result.compare_policies(0, 1)
        assert comparison["method"] == "paired_bootstrap"
        assert comparison["n_replicates"] == 150

    def test_reversed_indices_flip_sign_only(self) -> None:
        deltas = np.linspace(-0.1, 0.3, 200)
        result = _make_result(
            [0.7, 0.6], ["a", "b"], bootstrap_samples=_paired_matrix(deltas)
        )
        c01 = result.compare_policies(0, 1)
        c10 = result.compare_policies(1, 0)
        assert c10["difference"] == pytest.approx(-c01["difference"])
        assert c10["se_difference"] == pytest.approx(c01["se_difference"])
        assert c10["p_value"] == pytest.approx(c01["p_value"])
        assert c10["ci_lower"] == pytest.approx(-c01["ci_upper"])
        assert c10["ci_upper"] == pytest.approx(-c01["ci_lower"])

    def test_under_100_deltas_warns_and_falls_through(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # 50 replicates: strongly one-sided, would be "significant" if used —
        # but below the floor the bootstrap path must decline and fall
        # through to the legacy IF basis, with a warning.
        deltas = np.linspace(0.01, 0.02, 50)
        if_a = np.array([0.1, -0.1, 0.2, -0.2, 0.0])
        if_b = np.array([0.05, -0.05, 0.1, -0.1, 0.0])
        result = _make_result(
            [0.7, 0.6],
            ["a", "b"],
            influence_functions={"a": if_a, "b": if_b},
            bootstrap_samples=_paired_matrix(deltas),
        )
        with caplog.at_level(logging.WARNING):
            comparison = result.compare_policies(0, 1)
        assert any("50 valid paired bootstrap" in r.message for r in caplog.records)
        assert comparison["method"] == "paired_if_legacy"

    def test_near_tie_with_shared_noise_is_not_significant(self) -> None:
        """The ADJUDICATION defect scenario, in miniature.

        Two near-tie policies whose bootstrap replicates share calibrator
        noise (per-replicate deltas are tiny), plus per-sample influence
        functions whose z-test SE carries none of that replicate noise.
        The legacy IF basis declares a confident difference (36-47%
        wrong-sign in the pre-registered experiment); the paired bootstrap
        basis correctly declines to call it.
        """
        rng = np.random.default_rng(7)
        n_reps = 500
        shared = rng.normal(0.0, 0.05, n_reps)  # calibrator noise, common
        idio_a = rng.normal(0.0, 0.002, n_reps)
        idio_b = rng.normal(0.0, 0.002, n_reps)
        offset = 0.0005  # tiny true gap
        matrix = np.column_stack(
            [0.70 + offset + shared + idio_a, 0.70 + shared + idio_b]
        )
        # IFs engineered so the legacy z-test is confidently wrong: paired
        # IF differences with std ~0.001 over 400 samples -> SE ~5e-5
        base_if = rng.normal(0.0, 0.5, 400)
        if_a = base_if
        if_b = base_if - rng.normal(0.0, 0.001, 400)

        result = _make_result(
            [0.7005, 0.7000],
            ["a", "b"],
            influence_functions={"a": if_a, "b": if_b},
            bootstrap_samples=matrix,
        )

        paired = result.compare_policies(0, 1)
        assert paired["method"] == "paired_bootstrap"
        assert paired["significant"] is False
        assert paired["p_value"] > 0.05
        assert paired["ci_lower"] < 0.0 < paired["ci_upper"]

        # Contrast: the pre-0.5.1 basis on the same result is anti-conservative
        result.bootstrap_samples = None
        legacy = result.compare_policies(0, 1)
        assert legacy["method"] == "paired_if_legacy"
        assert legacy["significant"] is True


# ---------------------------------------------------------------------------
# compare_policies: dispatch precedence (0.5.1)
# ---------------------------------------------------------------------------


class TestComparePoliciesDispatch:
    def _full_stack_result(self) -> EstimationResult:
        """Result carrying every inference basis at once."""
        if_a = np.array([0.1, -0.1, 0.2, -0.2, 0.0])
        if_b = np.array([0.05, -0.05, 0.1, -0.1, 0.0])
        deltas = np.linspace(-0.01, 0.05, 150)
        return _make_result(
            [0.7, 0.6],
            ["a", "b"],
            standard_errors=[0.05, 0.04],
            influence_functions={"a": if_a, "b": if_b},
            metadata_extra={
                "pairwise_inference": {
                    "0-1": {
                        "policy1": "a",
                        "policy2": "b",
                        "se": 0.02,
                        "df": 4,
                        "basis": "index_paired",
                        "se_sampling": 0.015,
                        "var_oua_diff": 0.000175,
                        "n_pairs": 5,
                        "oua_folds": 5,
                    }
                }
            },
            bootstrap_samples=_paired_matrix(deltas),
        )

    def test_bootstrap_beats_pairwise_inference(self) -> None:
        result = self._full_stack_result()
        assert result.compare_policies(0, 1)["method"] == "paired_bootstrap"

    def test_pairwise_inference_beats_legacy(self) -> None:
        result = self._full_stack_result()
        result.bootstrap_samples = None
        comparison = result.compare_policies(0, 1)
        assert comparison["method"] == "paired_if_oua"
        assert comparison["basis"] == "index_paired"
        assert comparison["used_influence"] is True
        # t-based numerics from the stored se/df
        expected_t = 0.1 / 0.02
        assert comparison["z_score"] == pytest.approx(expected_t)
        expected_p = 2 * (1 - stats.t.cdf(abs(expected_t), 4))
        assert comparison["p_value"] == pytest.approx(expected_p)
        t_crit = float(stats.t.ppf(0.975, 4))
        assert comparison["ci_lower"] == pytest.approx(0.1 - t_crit * 0.02)
        assert comparison["ci_upper"] == pytest.approx(0.1 + t_crit * 0.02)
        assert comparison["df"] == 4

    def test_pairwise_inference_reversed_pair_shares_entry(self) -> None:
        result = self._full_stack_result()
        result.bootstrap_samples = None
        c01 = result.compare_policies(0, 1)
        c10 = result.compare_policies(1, 0)
        assert c10["method"] == "paired_if_oua"
        assert c10["difference"] == pytest.approx(-c01["difference"])
        assert c10["se_difference"] == pytest.approx(c01["se_difference"])
        assert c10["p_value"] == pytest.approx(c01["p_value"])

    def test_legacy_beats_independent(self) -> None:
        result = self._full_stack_result()
        result.bootstrap_samples = None
        result.metadata.pop("pairwise_inference")
        assert result.compare_policies(0, 1)["method"] == "paired_if_legacy"

    def test_independent_when_nothing_else(self) -> None:
        result = self._full_stack_result()
        result.bootstrap_samples = None
        result.metadata.pop("pairwise_inference")
        result.influence_functions = None
        comparison = result.compare_policies(0, 1)
        assert comparison["method"] == "independent_conservative"
        assert comparison["se_difference"] == pytest.approx(np.sqrt(0.05**2 + 0.04**2))

    def test_unusable_stored_se_falls_through(self) -> None:
        result = self._full_stack_result()
        result.bootstrap_samples = None
        result.metadata["pairwise_inference"]["0-1"]["se"] = 0.0
        assert result.compare_policies(0, 1)["method"] == "paired_if_legacy"


# ---------------------------------------------------------------------------
# compare_all_policies + Benjamini-Hochberg
# ---------------------------------------------------------------------------


class TestCompareAllPolicies:
    def test_all_pairs_carry_names_in_index_order(self) -> None:
        result = _make_result([0.0, 0.1, 0.2], ["a", "b", "c"])
        comparisons = result.compare_all_policies()
        assert [(c["policy1"], c["policy2"]) for c in comparisons] == [
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
        ]
        assert all("p_adjusted" not in c for c in comparisons)
        assert all("method" in c for c in comparisons)

    def test_bh_adjustment_hand_computed(self) -> None:
        # 4 policies on the independent path with se_diff = 0.05 for every
        # pair: per-pair p = 2*(1 - Phi(|diff|/0.05)), hand-checkable.
        se = 0.05 / np.sqrt(2)
        result = _make_result(
            [0.0, 0.01, 0.25, 0.45],
            ["p0", "p1", "p2", "p3"],
            standard_errors=[se, se, se, se],
        )
        comparisons = result.compare_all_policies(alpha=0.05, adjust="bh")
        assert len(comparisons) == 6
        by_pair = {(c["policy1"], c["policy2"]): c for c in comparisons}

        def p_of(diff: float) -> float:
            return float(2 * (1 - stats.norm.cdf(abs(diff) / 0.05)))

        # Ascending p: (p0,p3) z=9, (p1,p3) z=8.8, (p0,p2) z=5,
        # (p1,p2) z=4.8, (p2,p3) z=4, (p0,p1) z=0.2.
        # Hand-computed BH threshold at alpha=0.05, m=6: the largest k with
        # p_(k) <= k/6*0.05 is k=5 (p_(5)=6.33e-5 <= 0.0417; p_(6)=0.84 > 0.05)
        # -> reject exactly the 5 smallest.
        assert by_pair[("p0", "p1")]["significant_adjusted"] is False
        for pair in [
            ("p0", "p2"),
            ("p0", "p3"),
            ("p1", "p2"),
            ("p1", "p3"),
            ("p2", "p3"),
        ]:
            assert by_pair[pair]["significant_adjusted"] is True

        # Hand-computed adjusted p-values (p_adj_(k) = min_{l>=k} p_(l)*m/l):
        assert by_pair[("p0", "p1")]["p_adjusted"] == pytest.approx(p_of(0.01))
        assert by_pair[("p2", "p3")]["p_adjusted"] == pytest.approx(p_of(0.20) * 6 / 5)
        assert by_pair[("p1", "p2")]["p_adjusted"] == pytest.approx(p_of(0.24) * 6 / 4)
        assert by_pair[("p0", "p2")]["p_adjusted"] == pytest.approx(
            min(p_of(0.25) * 6 / 3, p_of(0.24) * 6 / 4)
        )
        # Raw p-values and per-pair significance are untouched
        assert by_pair[("p0", "p1")]["p_value"] == pytest.approx(p_of(0.01))
        assert by_pair[("p0", "p1")]["significant"] is False

    def test_invalid_adjust_raises(self) -> None:
        result = _make_result([0.0, 0.1], ["a", "b"])
        with pytest.raises(ValueError, match="adjust"):
            result.compare_all_policies(adjust="bonferroni")

    def test_nan_pair_keeps_nan_adjusted_p(self) -> None:
        result = _make_result([0.0, float("nan"), 0.2], ["a", "b", "c"])
        comparisons = result.compare_all_policies(adjust="bh")
        by_pair = {(c["policy1"], c["policy2"]): c for c in comparisons}
        assert np.isnan(by_pair[("a", "b")]["p_adjusted"])
        assert by_pair[("a", "b")]["significant_adjusted"] is False
        assert np.isfinite(by_pair[("a", "c")]["p_adjusted"])


# ---------------------------------------------------------------------------
# compare_policies end-to-end: bootstrap matrix, analytic pairwise metadata,
# serialization, denormalization
# ---------------------------------------------------------------------------


def _two_policy_draws(
    n: int = 50, gap: float = 0.03, scale: float = 1.0, seed: int = 11
) -> Dict[str, List[Dict[str, Any]]]:
    """Two policies on shared prompts; judge/oracle span the full [0, scale]
    range so auto-normalization detects exactly (0, scale)."""
    rng = np.random.default_rng(seed)
    out: Dict[str, List[Dict[str, Any]]] = {"a": [], "b": []}
    for i in range(n):
        s = i / (n - 1)
        rec_a: Dict[str, Any] = {"prompt_id": f"p{i:03d}", "judge_score": s * scale}
        if i % 2 == 0 or i == n - 1:
            y = min(max(s + float(rng.normal(0.0, 0.03)), 0.0), 1.0)
            if i == 0:
                y = 0.0
            if i == n - 1:
                y = 1.0
            rec_a["oracle_label"] = y * scale
        out["a"].append(rec_a)
        s_b = min(max(s + gap, 0.0), 1.0)
        out["b"].append({"prompt_id": f"p{i:03d}", "judge_score": s_b * scale})
    return out


class TestComparePoliciesGateFlags:
    def test_flagged_policy_annotated_and_warned(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        result = _make_result(
            [0.7, 0.4],
            ["a", "b"],
            gates={
                "a": {"flagged": False, "refuse_level_claims": False, "reasons": []},
                "b": {
                    "flagged": True,
                    "refuse_level_claims": True,
                    "reasons": ["transport audit FAIL"],
                },
            },
        )
        with caplog.at_level(logging.WARNING, logger="cje.data.models"):
            cmp_ = result.compare_policies(0, 1)
        assert cmp_["gate_flagged"] == ["b"]
        assert any("gates discipline" in rec.message for rec in caplog.records)

    def test_unflagged_pair_has_empty_gate_flagged(self) -> None:
        result = _make_result(
            [0.7, 0.4],
            ["a", "b"],
            gates={
                "a": {"flagged": False, "refuse_level_claims": False, "reasons": []},
                "b": {"flagged": False, "refuse_level_claims": False, "reasons": []},
            },
        )
        cmp_ = result.compare_policies(0, 1)
        assert cmp_["gate_flagged"] == []

    def test_no_gates_metadata_yields_empty_list(self) -> None:
        result = _make_result([0.7, 0.4], ["a", "b"])
        assert result.compare_policies(0, 1)["gate_flagged"] == []


class TestComparePoliciesEndToEnd:
    def test_bootstrap_run_attaches_matrix_and_dispatches(self, tmp_path: Any) -> None:
        result = analyze_dataset(
            fresh_draws_data=_two_policy_draws(),
            estimator_config={"n_bootstrap": 120},
        )
        assert result.bootstrap_samples is not None
        n_valid = result.metadata["inference"]["n_bootstrap_valid"]
        assert result.bootstrap_samples.shape == (n_valid, 2)
        assert n_valid >= 100

        comparison = result.compare_policies(0, 1)
        assert comparison["method"] == "paired_bootstrap"
        assert comparison["n_replicates"] == n_valid
        assert comparison["difference"] == pytest.approx(
            float(result.estimates[0] - result.estimates[1])
        )
        # p-value can never be exactly 0 at finite B
        assert comparison["p_value"] >= 2 / (n_valid + 1)

        # Serialization must not crash with the matrix present, and must
        # not embed the raw matrix
        as_dict = result.to_dict()
        assert "bootstrap_samples" not in as_dict
        assert as_dict["bootstrap_samples_summary"]["n_replicates"] == n_valid

        from cje.utils.export import export_results_json

        out_path = tmp_path / "results.json"
        export_results_json(result, str(out_path))
        assert out_path.exists() and out_path.stat().st_size > 0

    def test_analytic_run_stores_pairwise_inference(self) -> None:
        result = analyze_dataset(
            fresh_draws_data=_two_policy_draws(),
            estimator_config={"inference_method": "cluster_robust"},
        )
        pairwise = result.metadata["pairwise_inference"]
        entry = pairwise["0-1"]
        assert entry["basis"] == "prompt_cluster_paired"
        assert entry["se"] > 0
        # additive decomposition: se^2 = se_sampling^2 + var_oua_diff
        assert entry["se"] ** 2 == pytest.approx(
            entry["se_sampling"] ** 2 + entry["var_oua_diff"]
        )

        comparison = result.compare_policies(0, 1)
        assert comparison["method"] == "paired_if_oua"
        assert comparison["basis"] == "prompt_cluster_paired"
        assert comparison["se_difference"] == pytest.approx(entry["se"])

    def test_analytic_prompt_paired_basis_when_order_differs(self) -> None:
        # Same prompts as multisets, different record order: rows no longer
        # align one-to-one, so the pair must aggregate per prompt
        draws = _two_policy_draws()
        draws["b"] = list(reversed(draws["b"]))
        result = analyze_dataset(
            fresh_draws_data=draws,
            estimator_config={"inference_method": "cluster_robust"},
        )
        entry = result.metadata["pairwise_inference"]["0-1"]
        assert entry["basis"] == "prompt_cluster_paired"
        assert entry["n_pairs"] == 50
        comparison = result.compare_policies(0, 1)
        assert comparison["method"] == "paired_if_oua"
        assert comparison["basis"] == "prompt_cluster_paired"
        assert comparison["used_influence"] is True

    def test_analytic_independent_basis_when_prompts_differ(self) -> None:
        draws = _two_policy_draws()
        for rec in draws["b"]:
            rec["prompt_id"] = "x" + str(rec["prompt_id"])
        result = analyze_dataset(
            fresh_draws_data=draws,
            estimator_config={"inference_method": "cluster_robust"},
        )
        entry = result.metadata["pairwise_inference"]["0-1"]
        assert entry["basis"] == "prompt_cluster_disjoint"
        comparison = result.compare_policies(0, 1)
        assert comparison["method"] == "paired_if_oua"
        assert comparison["basis"] == "prompt_cluster_disjoint"
        # independent basis does not exploit influence-function pairing
        assert comparison["used_influence"] is False

    def test_denormalization_returns_oracle_scale_numbers(self) -> None:
        """0-100-scale run: compare_policies output is on the oracle scale.

        The scaled inputs are exactly 100x the unit-scale inputs and span
        the full range, so normalization reproduces the unit-scale pipeline
        bit-for-bit and every reported quantity must be exactly 100x.
        """
        config = {"n_bootstrap": 120}
        result_unit = analyze_dataset(
            fresh_draws_data=_two_policy_draws(scale=1.0),
            estimator_config=dict(config),
        )
        result_scaled = analyze_dataset(
            fresh_draws_data=_two_policy_draws(scale=100.0),
            estimator_config=dict(config),
        )
        oracle_range = result_scaled.metadata["normalization"]["oracle_label"][
            "original_range"
        ]
        assert tuple(oracle_range) == (0.0, 100.0)

        # The replicate matrix itself is denormalized
        assert result_scaled.bootstrap_samples is not None
        assert result_unit.bootstrap_samples is not None
        np.testing.assert_allclose(
            result_scaled.bootstrap_samples,
            100.0 * result_unit.bootstrap_samples,
            rtol=1e-10,
        )

        c_unit = result_unit.compare_policies(0, 1)
        c_scaled = result_scaled.compare_policies(0, 1)
        assert c_scaled["method"] == c_unit["method"] == "paired_bootstrap"
        for key in ("difference", "se_difference", "ci_lower", "ci_upper"):
            assert c_scaled[key] == pytest.approx(100.0 * c_unit[key], rel=1e-9)
        # sign-based p-value is scale-invariant
        assert c_scaled["p_value"] == pytest.approx(c_unit["p_value"])

    def test_denormalization_scales_pairwise_inference(self) -> None:
        config = {"inference_method": "cluster_robust"}
        result_unit = analyze_dataset(
            fresh_draws_data=_two_policy_draws(scale=1.0),
            estimator_config=dict(config),
        )
        result_scaled = analyze_dataset(
            fresh_draws_data=_two_policy_draws(scale=100.0),
            estimator_config=dict(config),
        )
        entry_unit = result_unit.metadata["pairwise_inference"]["0-1"]
        entry_scaled = result_scaled.metadata["pairwise_inference"]["0-1"]
        assert entry_scaled["se"] == pytest.approx(100.0 * entry_unit["se"], rel=1e-9)
        assert entry_scaled["se_sampling"] == pytest.approx(
            100.0 * entry_unit["se_sampling"], rel=1e-9
        )
        assert entry_scaled["var_oua_diff"] == pytest.approx(
            100.0**2 * entry_unit["var_oua_diff"], rel=1e-9
        )
        assert entry_scaled["df"] == entry_unit["df"]

        c_scaled = result_scaled.compare_policies(0, 1)
        c_unit = result_unit.compare_policies(0, 1)
        assert c_scaled["method"] == "paired_if_oua"
        assert c_scaled["difference"] == pytest.approx(
            100.0 * c_unit["difference"], rel=1e-9
        )
        # t-statistic and p-value are scale-invariant
        assert c_scaled["p_value"] == pytest.approx(c_unit["p_value"], rel=1e-9)
