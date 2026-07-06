"""Typed accessors on EstimationResult (0.5.0).

Covers the typed layer added over the metadata mirrors (which remain the
serialized source of truth and keep being written):

- target_policies property (reads metadata["target_policies"])
- gates property (typed view of metadata["reliability_gates"])
- best_policy() -> PolicyVerdict (gate-aware; replaced the naive int argmax)
- summary() compact text report
- ci_info typed CI record + the D9 alpha-mismatch warning (both the ci_info
  path and the legacy metadata-sniffing path)
- compare_policies (deterministic influence-function z-test)
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

    def test_flagged_argmax_is_demoted(self) -> None:
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
        assert verdict.name == "clone"  # best reliable
        assert verdict.index == 2
        assert verdict.estimate == pytest.approx(0.763)
        assert verdict.flagged is False
        assert verdict.all_flagged is False
        assert verdict.runner_up == "unhelpful"  # demoted point-estimate winner

    def test_reliable_only_false_returns_raw_argmax(self) -> None:
        result = _make_result(
            [0.771, 0.756],
            ["unhelpful", "base"],
            gates={"unhelpful": _gate(True), "base": _gate(False)},
        )
        verdict = result.best_policy(reliable_only=False)
        assert verdict.name == "unhelpful"
        assert verdict.flagged is True
        assert verdict.runner_up is None

    def test_all_flagged_returns_argmax_marked(self) -> None:
        result = _make_result(
            [0.9, 0.6], ["a", "b"], gates={"a": _gate(True), "b": _gate(True)}
        )
        verdict = result.best_policy()
        assert verdict.name == "a"
        assert verdict.flagged is True
        assert verdict.all_flagged is True
        assert verdict.runner_up is None

    def test_critical_status_also_demotes(self) -> None:
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
        assert verdict.runner_up == "bad"

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
        assert "Best policy: good" in text

    def test_summary_marks_flagged_and_demotes(self) -> None:
        result = _make_result(
            [0.771, 0.756],
            ["unhelpful", "base"],
            gates={"unhelpful": _gate(True), "base": _gate(False)},
        )
        text = result.summary()
        assert "[gate: FLAGGED]" in text
        assert "Best reliable policy: base" in text
        assert "unhelpful" in text

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

    def test_falls_back_to_conservative_se_without_ifs(self) -> None:
        result = _make_result([0.7, 0.6], ["a", "b"], standard_errors=[0.05, 0.04])
        comparison = result.compare_policies(0, 1)
        assert comparison["se_difference"] == pytest.approx(np.sqrt(0.05**2 + 0.04**2))
        assert comparison["used_influence"] is False
