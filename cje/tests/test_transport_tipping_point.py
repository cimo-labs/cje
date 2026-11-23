"""Sensitivity to fixed differential calibration error on the output scale."""

from typing import Any

import numpy as np
import pytest

from cje.data.models import EstimationResult, InferenceUnavailableError


@pytest.fixture
def borrowed_pair() -> EstimationResult:
    # Original label scale 0–100. B borrows A's map.
    return EstimationResult(
        estimates=np.array([68.0, 74.0]),
        standard_errors=np.array([2.0, 2.0]),
        n_samples_used={"A": 100, "B": 100},
        method="calibrated_direct",
        influence_functions=None,
        diagnostics=None,
        metadata={
            "target_policies": ["A", "B"],
            "own_oracle_labels_by_policy": {"A": 40, "B": 0},
            "transport_unverified": ["B"],
            "pairwise_inference": {
                "0-1": {"se": 1.0, "df": 39, "basis": "prompt_cluster_paired"}
            },
        },
    )


@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_signed_shifts_use_original_units_and_requested_interval(
    borrowed_pair: EstimationResult, alpha: float
) -> None:
    result = borrowed_pair
    first = result.compare_policies("B", "A", alpha, sensitivity_margin=1.0)
    reverse = result.compare_policies("A", "B", alpha, sensitivity_margin=-1.0)
    tip = first["transport_tipping_point"]
    assert first["transport_unverified"] == ["B"]
    assert tip == {
        "margin": 1.0,
        "point_shift": -5.0,
        "interval_shift": 1.0 - first["ci_lower"],
    }
    assert reverse["transport_tipping_point"]["point_shift"] == -tip["point_shift"]
    assert (
        reverse["transport_tipping_point"]["interval_shift"] == -tip["interval_shift"]
    )
    assert first["significant"] is True
    assert result.best_policy().decision_ready is False
    assert "A shift of +" in result.summary()


def test_zero_distance_and_both_arms_unverified(
    borrowed_pair: EstimationResult,
) -> None:
    result = borrowed_pair
    result.metadata["transport_unverified"] = ["A", "B"]
    first = result.compare_policies("B", "A")
    for margin in [first["ci_lower"], first["difference"], first["ci_upper"]]:
        comparison = result.compare_policies("B", "A", sensitivity_margin=margin)
        assert comparison["transport_unverified"] == ["B", "A"]
        assert comparison["transport_tipping_point"]["interval_shift"] == 0.0
        assert comparison["transport_tipping_point"]["point_shift"] == margin - 6.0
        assert comparison["significant"] == first["significant"]
    result.estimates[0] = result.estimates[1]
    tip = result.compare_all_policies()[0]["transport_tipping_point"]
    assert tip == {"margin": 0.0, "interval_shift": 0.0, "point_shift": 0.0}
    assert "The point difference is already zero" in result.summary()


def test_point_shift_rounding_preserves_a_small_nonzero_direction(
    borrowed_pair: EstimationResult,
) -> None:
    result = borrowed_pair
    result.estimates = np.array([0.5, 0.500001])
    assert "shift of +1e-06 label units" in result.summary()


def test_percentile_interval_can_lie_on_opposite_side_from_point(
    borrowed_pair: EstimationResult,
) -> None:
    # Each signed threshold is determined by its own statistic. Percentile
    # bounds need not straddle the observed point.
    result = borrowed_pair
    result.method = "calibrated_direct_bootstrap"
    series = np.arange(120) / 100
    result.bootstrap_samples = np.column_stack((60.0 + series, 59.0 - series))
    pair = result.compare_policies("B", "A")
    assert pair["difference"] == 6.0 and pair["ci_upper"] < 0
    assert pair["transport_tipping_point"]["point_shift"] == -6.0
    assert pair["transport_tipping_point"]["interval_shift"] == -pair["ci_upper"]


def test_fallback_without_paired_bounds_and_unavailable_inference(
    borrowed_pair: EstimationResult,
) -> None:
    result = borrowed_pair
    result.metadata.pop("pairwise_inference")
    pair = result.compare_policies("B", "A")
    assert pair["method"] == "independent_conservative"
    assert pair["transport_tipping_point"]["interval_shift"] is None
    assert "No paired interval threshold is available" in result.summary()
    result.metadata["inference_unavailable_policies"] = ["A"]
    with pytest.raises(InferenceUnavailableError):
        result.compare_policies("B", "A")


@pytest.mark.parametrize("route", ["unknown", "verified", "naive", "flagged", "failed"])
def test_gates_and_provenance_take_precedence(
    borrowed_pair: EstimationResult, route: str
) -> None:
    result = borrowed_pair
    if route == "unknown":
        result.metadata.pop("transport_unverified")
    elif route == "verified":
        result.metadata["transport_unverified"] = []
    elif route == "naive":
        result.method = "naive_direct"
    elif route == "flagged":
        result.metadata["reliability_gates"] = {"B": {"flagged": True}}
    else:
        result.metadata["transport_audits"] = {"B": {"status": "FAIL"}}
    assert "transport_tipping_point" not in result.compare_policies("B", "A")
    assert "A net calibration shift" not in result.summary()


def test_portable_recomputes_direction_and_margin(
    borrowed_pair: EstimationResult,
) -> None:
    result = borrowed_pair
    portable = result.to_dict()
    stored = portable["pairwise_inference_state"]["comparisons"]["0-1"]
    assert "transport_tipping_point" not in stored
    restored = EstimationResult.from_dict(portable)
    assert result.compare_policies("B", "A", sensitivity_margin=1.5) == (
        restored.compare_policies("B", "A", sensitivity_margin=1.5)
    )
    assert "transport_tipping_point" not in result.compare_policies("B", "B")


@pytest.mark.parametrize("margin", [True, "0.2", np.nan, np.inf, -np.inf, 10**400])
def test_invalid_margin(borrowed_pair: EstimationResult, margin: Any) -> None:
    with pytest.raises(ValueError, match="sensitivity_margin"):
        borrowed_pair.compare_policies("B", "A", sensitivity_margin=margin)
