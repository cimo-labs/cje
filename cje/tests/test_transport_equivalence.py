"""Regression tests for residual-transport equivalence grading."""

import logging
import warnings

import numpy as np
import pytest
from typing import Any, Dict, List, Optional

from cje import TransportAuditConfig, analyze_dataset
from cje.diagnostics import TRANSPORT_STATUS_TO_STATUS, Status
from cje.diagnostics.transport import audit_transportability


class _IdentityCalibrator:
    covariate_names: List[str] = []

    def predict(self, scores: Any, covariates: Optional[Any] = None) -> np.ndarray:
        result = np.asarray(scores, dtype=float)
        if covariates is not None:
            result = result + np.asarray(covariates, dtype=float)[:, 0]
        return result


def _probe(residuals: Any) -> List[Dict[str, Any]]:
    return [
        {
            "prompt_id": f"p{i}",
            "judge_score": 0.5,
            "oracle_label": 0.5 + float(residual),
        }
        for i, residual in enumerate(residuals)
    ]


def test_no_margin_is_not_graded_and_warns_about_vocabulary_change() -> None:
    with pytest.warns(FutureWarning, match="NOT_GRADED and can never PASS or FAIL"):
        diag = audit_transportability(_IdentityCalibrator(), _probe(np.zeros(30)))
    assert diag.status == "NOT_GRADED"
    assert diag.reason_code == "margin_not_declared"


def test_declared_margin_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        diag = audit_transportability(
            _IdentityCalibrator(), _probe(np.zeros(30)), delta_max=0.05
        )
    assert diag.status == "PASS"


def test_narrow_ci_inside_margin_passes() -> None:
    residuals = np.tile([-0.01, 0.01], 20)
    diag = audit_transportability(
        _IdentityCalibrator(), _probe(residuals), delta_max=0.05
    )
    assert diag.status == "PASS"
    assert diag.delta_ci[0] >= -0.05
    assert diag.delta_ci[1] <= 0.05


def test_ci_disjoint_from_margin_fails() -> None:
    residuals = np.tile([0.09, 0.11], 20)
    diag = audit_transportability(
        _IdentityCalibrator(), _probe(residuals), delta_max=0.05
    )
    assert diag.status == "FAIL"
    assert diag.delta_ci[0] > 0.05


def test_ci_crossing_margin_is_inconclusive() -> None:
    residuals = np.tile([-0.04, 0.16], 20)
    diag = audit_transportability(
        _IdentityCalibrator(), _probe(residuals), delta_max=0.05
    )
    assert diag.status == "INCONCLUSIVE"
    assert diag.delta_ci[0] < 0.05 < diag.delta_ci[1]


def test_fewer_than_twenty_effective_clusters_is_inconclusive() -> None:
    diag = audit_transportability(
        _IdentityCalibrator(), _probe(np.zeros(19)), delta_max=0.05
    )
    assert diag.status == "INCONCLUSIVE"
    assert diag.reason_code == "insufficient_effective_clusters"


def test_ci_wholly_outside_margin_fails_below_cluster_floor() -> None:
    # 15 clusters with a mean residual of 0.40 against a 0.05 margin: the
    # whole CI sits far outside the margin, so the effective-cluster floor
    # must not launder decisive bias into a non-blocking INCONCLUSIVE.
    residuals = np.linspace(0.35, 0.45, 15)
    diag = audit_transportability(
        _IdentityCalibrator(), _probe(residuals), delta_max=0.05
    )
    assert diag.effective_clusters < 20
    assert diag.delta_ci[0] > 0.05
    assert diag.status == "FAIL"
    assert diag.reason_code == "unacceptable_mean_residual"


def test_repeated_rows_do_not_create_independent_clusters() -> None:
    residuals = np.tile([-0.01, 0.01], 20)
    records = _probe(residuals)
    cluster_ids = np.repeat([f"c{i}" for i in range(10)], 4)
    diag = audit_transportability(
        _IdentityCalibrator(),
        records,
        delta_max=0.05,
        cluster_ids=cluster_ids,
    )
    assert diag.n_probe == 40
    assert diag.n_clusters == 10
    assert diag.status == "INCONCLUSIVE"


def test_family_adjustment_widens_interval() -> None:
    residuals = np.tile([-0.03, 0.03], 30)
    single = audit_transportability(
        _IdentityCalibrator(), _probe(residuals), delta_max=0.05
    )
    family = audit_transportability(
        _IdentityCalibrator(),
        _probe(residuals),
        delta_max=0.05,
        family_size=5,
    )
    assert family.ci_half_width > single.ci_half_width
    # Bonferroni: family-wise coverage stays 1 - alpha; each of the K
    # intervals is at the wider per-audit level 1 - alpha/K.
    assert family.simultaneous_confidence_level == pytest.approx(0.95)
    assert family.per_audit_confidence_level == pytest.approx(0.99)
    assert single.simultaneous_confidence_level == pytest.approx(0.95)
    assert single.per_audit_confidence_level == pytest.approx(0.95)


def test_weights_and_covariates_are_used() -> None:
    records = _probe(np.zeros(30))
    covariates = np.full((30, 1), 0.02)
    # Labels are 0.02 above the score, exactly matching the covariate-aware fit.
    for row in records:
        row["oracle_label"] += 0.02
    diag = audit_transportability(
        _IdentityCalibrator(),
        records,
        delta_max=0.05,
        covariates=covariates,
        sample_weights=np.linspace(1.0, 2.0, 30),
    )
    assert diag.status == "PASS"
    assert diag.weighted is True
    assert diag.delta_hat == pytest.approx(0.0)


@pytest.mark.parametrize("bad_margin", [0.0, -0.1, np.inf])
def test_margin_must_be_finite_and_positive(bad_margin: float) -> None:
    with pytest.raises(ValueError, match="delta_max"):
        audit_transportability(
            _IdentityCalibrator(), _probe(np.zeros(30)), delta_max=bad_margin
        )


def test_status_ladder_not_checked_is_informational() -> None:
    # Absence of a probe is not a defect: NOT_CHECKED must not drag
    # worst-status rollups to WARNING on every probe-less analysis.
    assert TRANSPORT_STATUS_TO_STATUS["NOT_CHECKED"] is Status.GOOD
    assert TRANSPORT_STATUS_TO_STATUS["NOT_GRADED"] is Status.WARNING
    assert TRANSPORT_STATUS_TO_STATUS["INCONCLUSIVE"] is Status.WARNING
    assert TRANSPORT_STATUS_TO_STATUS["PASS"] is Status.GOOD
    assert TRANSPORT_STATUS_TO_STATUS["FAIL"] is Status.CRITICAL


def test_residuals_and_margin_remain_in_public_oracle_units() -> None:
    class PublicUnitCalibrator:
        def predict(self, scores: np.ndarray) -> np.ndarray:
            # Public judge units are 0..10; public oracle units are 0..100.
            return np.asarray(scores, dtype=float) * 10.0

    probes = [
        {
            "prompt_id": f"p{i}",
            "judge_score": float(i % 10),
            "oracle_label": float((i % 10) * 10 + 1),
        }
        for i in range(25)
    ]

    diagnostic = audit_transportability(PublicUnitCalibrator(), probes, delta_max=2.0)

    assert diagnostic.delta_hat == pytest.approx(1.0)
    assert diagnostic.delta_ci == pytest.approx((1.0, 1.0))
    assert diagnostic.delta_max == 2.0
    assert diagnostic.status == "PASS"


def _labeled_policy(policy: str) -> List[Dict[str, Any]]:
    return [
        {
            "prompt_id": f"p{i}",
            "judge_score": 0.2 + 0.4 * i / 29,
            "oracle_label": 0.2 + 0.4 * i / 29,
        }
        for i in range(30)
    ]


def _policy_probe(policy: str, n: int, shift: float = 0.0) -> List[Dict[str, Any]]:
    return [
        {
            "prompt_id": f"probe-{policy}-{i}",
            "judge_score": 0.2 + 0.4 * (i % 20) / 19,
            "oracle_label": 0.2 + 0.4 * (i % 20) / 19 + shift,
        }
        for i in range(n)
    ]


def test_high_level_transport_records_all_graded_states() -> None:
    policies = ["pass", "fail", "low_power", "descriptive", "missing"]
    draws = {policy: _labeled_policy(policy) for policy in policies}
    for rows in draws.values():
        for row in rows[20:]:
            row.pop("oracle_label")
    config = TransportAuditConfig(
        probes_by_policy={
            "pass": _policy_probe("pass", 30),
            "fail": _policy_probe("fail", 30, shift=0.2),
            "low_power": _policy_probe("low", 10),
            "descriptive": _policy_probe("description", 30),
        },
        delta_max_by_policy={
            "pass": 0.05,
            "fail": 0.05,
            "low_power": 0.05,
        },
    )

    result = analyze_dataset(
        fresh_draws_data=draws,
        estimator_config={"inference_method": "cluster_robust"},
        transport=config,
    )

    audits = result.metadata["transport_audits"]
    assert audits["pass"]["status"] == "PASS"
    assert audits["fail"]["status"] == "FAIL"
    assert audits["low_power"]["status"] == "INCONCLUSIVE"
    assert audits["descriptive"]["status"] == "NOT_GRADED"
    assert audits["missing"]["status"] == "NOT_CHECKED"
    assert audits["pass"]["performed"] is True
    assert audits["descriptive"]["graded"] is False
    assert audits["missing"]["performed"] is False
    assert audits["fail"]["applies_to_current_estimate"] is True
    assert result.metadata["transport_status"] == "FAIL"
    assert result.gates["fail"].flagged is True
    assert np.all(np.isfinite(result.estimates))
    assert result.diagnostics is not None
    assert result.diagnostics.transport_status_per_policy == {
        policy: audits[policy]["status"] for policy in policies
    }
    restored = type(result).from_dict(result.to_dict(detail="portable"))
    assert restored.metadata["transport_audits"]["fail"]["status"] == "FAIL"
    assert restored.diagnostics is not None
    assert restored.diagnostics.transport_status_per_policy == (
        result.diagnostics.transport_status_per_policy
    )


def test_high_level_unmargined_probe_gets_info_not_future_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit TransportAuditConfig callers are new-API users, not 0.5.x
    migration cases: an unmargined probe records NOT_GRADED with a concise
    info note, without the module-level FutureWarning. Direct
    audit_transportability calls keep the FutureWarning (pinned above)."""
    config = TransportAuditConfig(
        probes_by_policy={"policy": _policy_probe("policy", 30)}
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(logging.INFO, logger="cje.interface.analysis"):
            result = analyze_dataset(
                fresh_draws_data={"policy": _labeled_policy("policy")},
                estimator_config={"inference_method": "cluster_robust"},
                transport=config,
            )

    assert result.metadata["transport_audits"]["policy"]["status"] == "NOT_GRADED"
    assert not any(
        issubclass(warning.category, FutureWarning)
        and "NOT_GRADED and can never PASS or FAIL" in str(warning.message)
        for warning in caught
    )
    assert any(
        "descriptive-only (NOT_GRADED)" in record.getMessage()
        for record in caplog.records
    )


def test_transport_fail_marks_only_failing_policy_when_statuses_absent() -> None:
    """A transport FAIL must not fabricate GOOD statuses for policies that
    were never assessed: only the failing policy gets a (CRITICAL) entry."""
    from cje.data.models import EstimationResult
    from cje.data.normalization import unit_scale
    from cje.diagnostics import DirectDiagnostics
    from cje.interface.analysis import _attach_transport_audits

    policies = ["fail", "other"]
    diagnostics = DirectDiagnostics(
        estimator_type="Direct",
        method="calibrated_direct",
        n_samples_total=60,
        n_samples_valid=60,
        policies=policies,
        estimates={"fail": 0.4, "other": 0.4},
        standard_errors={"fail": 0.01, "other": 0.01},
        n_samples_used={policy: 30 for policy in policies},
    )
    assert diagnostics.status_per_policy is None  # never assessed
    result = EstimationResult(
        estimates=np.array([0.4, 0.4]),
        standard_errors=np.array([0.01, 0.01]),
        n_samples_used={policy: 30 for policy in policies},
        method="calibrated_direct",
        influence_functions=None,
        diagnostics=diagnostics,
        calibrator=_IdentityCalibrator(),
        metadata={"target_policies": policies},
    )
    config = TransportAuditConfig(
        probes_by_policy={"fail": _policy_probe("fail", 30, shift=0.2)},
        delta_max_by_policy={"fail": 0.05},
    )

    _attach_transport_audits(
        result,
        config=config,
        target_policies=policies,
        calibration_dataset=None,
        judge_field="judge_score",
        oracle_field="oracle_label",
        oracle_input_scale=unit_scale(),
        output_scale=unit_scale(),
    )

    assert result.metadata["transport_audits"]["fail"]["status"] == "FAIL"
    assert result.diagnostics is not None
    # Only the audited-and-failed policy carries a status; "other" stays
    # absent instead of being recorded as GOOD.
    assert result.diagnostics.status_per_policy == {"fail": Status.CRITICAL}


def test_high_level_transport_transforms_probe_labels_to_output_scale() -> None:
    draws = _labeled_policy("policy")
    for row in draws:
        row["oracle_label"] = 1.0 + 4.0 * row["oracle_label"]
    probes = [
        {
            "prompt_id": f"probe-{i}",
            "judge_score": row["judge_score"],
            "oracle_label": row["oracle_label"],
        }
        for i, row in enumerate(draws)
    ]

    result = analyze_dataset(
        fresh_draws_data={"policy": draws},
        fresh_oracle_scale=(1, 5),
        output_scale=(0, 100),
        estimator_config={"inference_method": "cluster_robust"},
        transport=TransportAuditConfig(
            probes_by_policy={"policy": probes},
            delta_max_by_policy={"policy": 1.0},
        ),
    )

    audit = result.metadata["transport_audits"]["policy"]
    assert result.estimates[0] == pytest.approx(40.0)
    assert audit["delta_hat"] == pytest.approx(0.0, abs=1e-10)
    assert audit["status"] == "PASS"


def test_high_level_transport_computes_response_length_for_probes() -> None:
    draws = _labeled_policy("policy")
    for i, row in enumerate(draws):
        row["response"] = "word " * (i % 4 + 1)
    probes = [
        {
            "prompt_id": f"probe-{i}",
            "judge_score": row["judge_score"],
            "oracle_label": row["oracle_label"],
            "response": row["response"],
        }
        for i, row in enumerate(draws)
    ]

    result = analyze_dataset(
        fresh_draws_data={"policy": draws},
        include_response_length=True,
        estimator_config={"inference_method": "cluster_robust"},
        transport=TransportAuditConfig(
            probes_by_policy={"policy": probes},
            delta_max_by_policy={"policy": 0.05},
        ),
    )

    audit = result.metadata["transport_audits"]["policy"]
    assert audit["status"] == "PASS"
    assert audit["covariate_names"] == ["response_length"]
    assert audit["n_probe"] == len(probes)


def test_high_level_without_probes_is_explicitly_not_checked() -> None:
    result = analyze_dataset(
        fresh_draws_data={"policy": _labeled_policy("policy")},
        estimator_config={"inference_method": "cluster_robust"},
    )
    audit = result.metadata["transport_audits"]["policy"]
    assert audit["status"] == "NOT_CHECKED"
    assert audit["reason_code"] == "probe_not_provided"
    assert result.best_policy().name == "policy"
    assert "residual transport NOT_CHECKED" in result.summary()


def test_high_level_rejects_probe_reused_for_calibrator_fit() -> None:
    draws = {"policy": _labeled_policy("policy")}
    draws["policy"][0]["source_id"] = "shared-source"
    draws["policy"][0]["row_id"] = "shared-row"
    probe = _policy_probe("policy", 30)
    probe[0]["source_id"] = "shared-source"
    probe[0]["row_id"] = "shared-row"
    config = TransportAuditConfig(
        probes_by_policy={"policy": probe},
        delta_max_by_policy={"policy": 0.05},
    )
    with pytest.raises(ValueError, match="used to fit the calibrator"):
        analyze_dataset(
            fresh_draws_data=draws,
            estimator_config={"inference_method": "cluster_robust"},
            transport=config,
        )


def test_transport_config_rejects_understated_family_and_empty_probe() -> None:
    with pytest.raises(ValueError, match="family_size"):
        TransportAuditConfig(
            probes_by_policy={"a": _probe(np.zeros(20)), "b": _probe(np.zeros(20))},
            family_size=1,
        )

    config = TransportAuditConfig(probes_by_policy={"policy": []})
    with pytest.raises(ValueError, match="collection.*empty"):
        analyze_dataset(
            fresh_draws_data={"policy": _labeled_policy("policy")},
            estimator_config={"inference_method": "cluster_robust"},
            transport=config,
        )
