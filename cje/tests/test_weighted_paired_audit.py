"""Finite-frame sampling controls for the paired, two-stage audit."""

import itertools
from typing import Any

import numpy as np
import pytest

from cje.weighted_audit import weighted_paired_audit


def frame(states: list[int]) -> dict[str, Any]:
    # 0 = omitted, 1 = only scored, 2 = paired human review.
    state = np.asarray(states)
    return dict(
        cluster_ids=["d0", "d1", "d2"],
        traffic_selected=state > 0,
        label_selected=state == 2,
        traffic_inclusion=[1.0, 0.6, 0.7],
        label_inclusion=[0.5, 1.0, 0.3],
        predictions=np.where(state > 0, [0.25, -0.5, 0.1], np.nan),
        outcomes=np.where(state == 2, [1.0, -1.0, -0.2], np.nan),
        target_weights=[1, 4, 2],
        outcome_bounds=(-1, 1),
    )


def test_exact_two_stage_design_expectation_and_diagnostics() -> None:
    mean = 0.0
    coverage = 0.0
    total_probability = 0.0
    true_contrast = (1.0 - 4.0 - 0.4) / 7
    for state in itertools.product(range(3), repeat=3):
        args = frame(list(state))
        q, p = (np.array(args[k]) for k in ("traffic_inclusion", "label_inclusion"))
        probability = np.prod(np.choose(state, [1 - q, q * (1 - p), q * p]))
        if not probability:
            continue
        result = weighted_paired_audit(**args)
        mean += probability * result.estimate
        total_probability += probability
        coverage += probability * (
            result.ci[0] <= true_contrast + 1e-12
            and result.ci[1] >= true_contrast - 1e-12
        )
        a, b = result.outcome_identification_bounds
        assert a <= true_contrast + 1e-12 and b >= true_contrast - 1e-12
        assert a <= result.ci[0] <= result.ci[1] <= b
    assert total_probability == pytest.approx(1)
    assert mean == pytest.approx(true_contrast)
    assert coverage >= 0.95
    args = frame([1, 2, 2])
    r = weighted_paired_audit(**args)
    w = np.array(args["target_weights"]) / 7
    traffic = w / np.array(args["traffic_inclusion"])
    joint = traffic / np.array(args["label_inclusion"])
    expected = traffic @ args["predictions"] + joint[1:] @ (
        args["outcomes"][1:] - args["predictions"][1:]
    )
    assert r.estimate == pytest.approx(expected)
    assert (r.n_frame, r.n_traffic, r.n_labeled) == (3, 3, 2)
    assert r.traffic_effective_n == pytest.approx(
        traffic.sum() ** 2 / (traffic @ traffic)
    )
    assert r.label_effective_n == pytest.approx(
        joint[1:].sum() ** 2 / (joint[1:] @ joint[1:])
    )
    assert r.min_joint_inclusion == pytest.approx(0.21)
    assert r.labeled_target_mass == pytest.approx(6 / 7)


def test_census_ignores_predictions_and_supports_nonunit_outcome_units() -> None:
    args = frame([2, 2, 2])
    args.update(
        traffic_inclusion=[1] * 3,
        label_inclusion=[1] * 3,
        target_weights=None,
        outcome_bounds=(-500, 500),
        predictions=[275, -375, 125],
        outcomes=[125, -125, -375],
    )
    r = weighted_paired_audit(**args)
    assert [r.estimate, *r.ci, *r.outcome_identification_bounds] == pytest.approx(
        [-125] * 5
    )


def test_empty_draw_retains_full_support_even_with_tiny_positive_inclusions() -> None:
    args = frame([0, 0, 0])
    args["traffic_inclusion"] = [0.1, 0.2, 1e-120]
    r = weighted_paired_audit(**args)
    assert r.estimate == 0.0
    assert r.ci == r.outcome_identification_bounds == (-1.0, 1.0)
    assert r.label_effective_n == r.traffic_effective_n == 0.0


def test_known_masses_can_reverse_order() -> None:
    args = frame([2, 2, 2])
    args.update(
        traffic_inclusion=[1] * 3,
        label_inclusion=[1] * 3,
        outcomes=[1, 1, -1],
        target_weights=None,
    )
    assert weighted_paired_audit(**args).estimate == pytest.approx(1 / 3)
    args["target_weights"] = [1, 1, 8]
    assert weighted_paired_audit(**args).estimate == pytest.approx(-0.6)


def test_large_independent_frame_ci_can_tighten_the_raw_outcome_bounds() -> None:
    n = 2000
    seen = np.arange(n) % 4 != 0
    rated = seen & (np.arange(n) % 2 == 1)
    result = weighted_paired_audit(
        cluster_ids=np.arange(n),
        traffic_selected=seen,
        label_selected=rated,
        traffic_inclusion=np.full(n, 0.75),
        label_inclusion=np.full(n, 2 / 3),
        predictions=np.where(seen, 0.1, np.nan),
        outcomes=np.where(rated, 0.1, np.nan),
        outcome_bounds=(0, 1),
    )
    assert result.ci[0] <= 0.1 <= result.ci[1]
    assert result.ci[1] < result.outcome_identification_bounds[1] - 0.1


def test_raw_statistic_can_fall_outside_observed_support() -> None:
    a = frame([2, 2, 2])
    a.update(
        traffic_inclusion=[0.25] * 3,
        label_inclusion=[1] * 3,
        outcomes=[1] * 3,
        target_weights=None,
    )
    r = weighted_paired_audit(**a)
    assert r.estimate == pytest.approx(4.0)
    assert r.ci == (1.0, 1.0)


@pytest.mark.parametrize(
    "key,value",
    [
        ("cluster_ids", []),
        ("cluster_ids", ["d", "d", "e"]),
        ("cluster_ids", [1, None, 2]),
        ("cluster_ids", [1, float("nan"), 2]),
        ("cluster_ids", [True, 1, 2]),
        ("cluster_ids", ["a", " ", "c"]),
        ("traffic_inclusion", [1, 1, 0]),
        ("traffic_inclusion", [1, -1, 1]),
        ("traffic_inclusion", [1, np.nan, 1]),
        ("traffic_inclusion", [1, 1]),
        ("label_inclusion", [1, 1, 1.1]),
        ("label_inclusion", [True] * 3),
        ("target_weights", [1, 2, 0]),
        ("target_weights", [1, 2, np.inf]),
        ("target_weights", [np.nextafter(0.0, 1.0), 1e300, 1]),
        ("traffic_selected", [1, 1, 1]),
        ("label_selected", [True] * 2),
        ("label_selected", [False, True, True]),
        ("label_selected", [False] * 3),
        ("traffic_selected", [False, True, False]),
        ("outcomes", np.ma.array([np.nan, 1, np.nan], mask=[False, True, False])),
        ("traffic_inclusion", np.ma.array([1, 0.5, 0.3], mask=[False, True, False])),
        ("cluster_ids", np.ma.array([1, 2, 3], mask=[False, True, False])),
        ("outcomes", [np.nan, 1, 0.0]),
        ("outcomes", [np.nan, np.nan, np.nan]),
        ("predictions", [0, 1, 0]),
        ("predictions", [1.1, 0, np.nan]),
        ("predictions", np.array([0, 1j, np.nan])),
        ("predictions", ["0.25", "-0.5", None]),
        ("outcomes", [None, "-1", None]),
        ("cluster_ids", ["d0", 1, "d2"]),
        ("outcome_bounds", (0,)),
        ("outcome_bounds", (0, 0)),
        ("outcome_bounds", (-np.inf, 1)),
        ("alpha", -0.1),
        ("alpha", True),
        ("alpha", np.nan),
    ],
)
def test_invalid_or_inconsistent_frame(key: str, value: Any) -> None:
    a = frame([1, 2, 0])
    a[key] = value
    with pytest.raises(ValueError):
        weighted_paired_audit(**a)


def test_unrepresentable_joint_probability_fails_explicitly() -> None:
    a = frame([1, 2, 0])
    a["traffic_inclusion"] = [1, 0.6, 1e-300]
    a["label_inclusion"] = [0.5, 1, 1e-300]
    with pytest.raises(ValueError, match="numerical range"):
        weighted_paired_audit(**a)


def test_sparse_wmt_document_replay() -> None:
    # Public WMT24 English→German, GPT-4 incumbent, Claude-3.5 challenger.
    # Each entry is (segments, segments with MetricX drop, new major errors,
    # number of documents). Published annotations condensed to the sufficient
    # statistics for this fixed-frame replay. Both arms are graded together.
    grouped = np.array(
        [
            [1, 0, 0, 65],
            [1, 0, 1, 6],
            [1, 1, 0, 36],
            [1, 1, 1, 5],
            [2, 1, 0, 2],
            [3, 0, 0, 1],
            [3, 1, 0, 2],
            [3, 1, 1, 1],
            [3, 2, 1, 1],
            [4, 1, 0, 3],
            [4, 2, 0, 3],
            [4, 4, 0, 1],
            [5, 0, 0, 1],
            [5, 1, 0, 1],
            [5, 2, 0, 4],
            [5, 3, 0, 1],
            [5, 4, 0, 1],
            [6, 0, 0, 1],
            [6, 2, 0, 1],
            [6, 3, 0, 1],
            [6, 3, 2, 1],
            [6, 4, 1, 2],
            [6, 5, 0, 1],
            [7, 1, 2, 1],
            [7, 2, 0, 3],
            [7, 3, 0, 1],
            [7, 4, 0, 2],
            [7, 5, 0, 3],
            [8, 2, 0, 1],
            [8, 3, 0, 1],
            [8, 4, 0, 4],
            [8, 4, 1, 1],
            [8, 5, 0, 1],
            [9, 4, 0, 2],
            [9, 4, 1, 1],
            [9, 7, 1, 1],
            [9, 8, 0, 1],
            [10, 1, 1, 1],
            [10, 5, 0, 2],
            [11, 7, 2, 1],
            [12, 3, 0, 1],
            [12, 7, 5, 1],
        ]
    )
    count, drops, failures = np.repeat(grouped[:, :3], grouped[:, 3], axis=0).T
    assert (len(count), count.sum(), failures.sum()) == (170, 485, 30)
    y, drop_share = failures / count, drops / count
    q = np.where(count <= 2, 0.85, 0.40)
    p = np.where(drop_share > 0.5, 0.70, 0.30)
    m = 0.04 + 0.12 * drop_share
    truth = y.mean()
    rng = np.random.default_rng(9298)
    estimates = []
    widths = []
    wald_hits = 0
    variance_contribution = (
        (1 - q) / q * y**2 + (1 - p) / (q * p) * (y - m) ** 2
    ) / len(q) ** 2
    for _ in range(500):
        seen = rng.random(len(q)) < q
        rated = seen & (rng.random(len(q)) < p)
        r = weighted_paired_audit(
            cluster_ids=np.arange(len(q)),
            traffic_selected=seen,
            label_selected=rated,
            traffic_inclusion=q,
            label_inclusion=p,
            predictions=np.where(seen, m, np.nan),
            outcomes=np.where(rated, y, np.nan),
            outcome_bounds=(0, 1),
        )
        assert r.ci[0] <= truth <= r.ci[1]
        estimates.append(r.estimate)
        widths.append(r.ci[1] - r.ci[0])
        variance = np.sum(variance_contribution[rated] / (q[rated] * p[rated]))
        wald_hits += abs(r.estimate - truth) <= 1.96 * np.sqrt(variance)
    assert truth == pytest.approx(0.08038091)
    assert abs(np.mean(estimates) - truth) < 0.005
    assert 0.47 < np.mean(widths) < 0.51
    assert wald_hits / len(estimates) < 0.94
