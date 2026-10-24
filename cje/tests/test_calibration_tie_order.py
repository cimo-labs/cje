"""Auto calibration-mode selection must not depend on the order of tied scores.

Auto mode compares the monotone and two-stage fits in the low, mid and high
judge-score thirds. Cutting an unstable argsort of S at n // 3 and 2 * (n // 3)
split blocks of tied scores (LLM judges emit few distinct values) by the sort's
tie order, so shuffling the rows could flip the selected mode and change the
estimate. Ties now go wholly to one third.
"""

import warnings

import numpy as np
import pytest

from cje.calibration.flexible_calibrator import FlexibleCalibrator

N_ROWS = 240
N_SHUFFLES = 20


def _tied_data(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eight judge-score levels (blocks of 17-45 rows) and a curved response."""
    rng = np.random.default_rng(seed)
    scores = rng.integers(0, 8, N_ROWS) / 7.0
    labels = np.clip(
        0.2 + 0.6 * scores + 0.15 * np.sin(6 * scores) + rng.normal(0, 0.2, N_ROWS),
        0.0,
        1.0,
    )
    folds = np.arange(N_ROWS) % 5
    return scores, labels, folds


def _fit(
    scores: np.ndarray, labels: np.ndarray, folds: np.ndarray
) -> tuple[str, np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrator = FlexibleCalibrator(mode="auto").fit(scores, labels, folds)
    assert calibrator.selected_mode is not None
    return (
        calibrator.selected_mode,
        calibrator.predict(scores, folds),
        calibrator.predict(scores),
    )


# One seed per mode, each decided by RMSE margins of at least 1e-3 (seeds 8
# and 30, used before, rested on 8e-5 and 6e-6). With the old split, 5 of the
# 20 shuffles below gave the other mode on each seed (numpy 2.2.6 and 2.4.6 on
# arm64; the count depends on the platform's unstable sort).
@pytest.mark.parametrize("seed, expected_mode", [(281, "monotone"), (299, "two_stage")])
def test_auto_mode_and_predictions_ignore_the_order_of_tied_rows(
    seed: int, expected_mode: str
) -> None:
    scores, labels, folds = _tied_data(seed)
    mode, oof, full = _fit(scores, labels, folds)
    assert mode == expected_mode
    shuffles = np.random.default_rng(2026)
    for _ in range(N_SHUFFLES):
        perm = shuffles.permutation(N_ROWS)
        inverse = np.argsort(perm)
        shuffled_mode, shuffled_oof, shuffled_full = _fit(
            scores[perm], labels[perm], folds[perm]
        )
        assert shuffled_mode == mode
        # Same fitted maps; the tolerance only absorbs last-bit differences a
        # BLAS could introduce in the ridge fit, far below any mode change.
        np.testing.assert_allclose(shuffled_oof[inverse], oof, rtol=0, atol=1e-12)
        np.testing.assert_allclose(shuffled_full[inverse], full, rtol=0, atol=1e-12)


def test_score_thirds_keep_ties_whole_and_match_the_plain_cut_without_ties() -> None:
    from cje.calibration.flexible_calibrator import _score_thirds

    rng = np.random.default_rng(3)
    # Distinct scores: exactly the previous split (argsort cut at n // 3).
    distinct = rng.permutation(100).astype(float)
    order = np.argsort(distinct)
    for got, want in zip(
        _score_thirds(distinct), (order[:33], order[33:66], order[66:])
    ):
        np.testing.assert_array_equal(got, want)

    # Tied scores: each score value lies in one third, and the membership of
    # every row is the same after a shuffle.
    tied = rng.integers(0, 6, 90).astype(float)
    thirds = _score_thirds(tied)
    assert sum(len(rows) for rows in thirds) == len(tied)
    assert all(
        set(tied[a]).isdisjoint(tied[b])
        for i, a in enumerate(thirds)
        for b in thirds[i + 1 :]
    )
    perm = rng.permutation(len(tied))
    for rows, shuffled_rows in zip(thirds, _score_thirds(tied[perm])):
        assert set(rows) == set(perm[shuffled_rows])


@pytest.mark.parametrize("rows_per_level", [30, 1])
def test_empty_thirds_are_skipped_in_mode_selection(rows_per_level: int) -> None:
    """Few distinct scores leave thirds empty; selection skips them.

    Two levels of 30 rows leave the mid third empty. Two rows leave the low
    and mid thirds empty; the old split computed RMSEs on those empty thirds
    and failed (RuntimeWarning, or ZeroDivisionError with weights).
    """
    from cje.calibration.flexible_calibrator import _score_thirds

    rng = np.random.default_rng(5)
    scores = np.repeat([0.25, 0.75], rows_per_level)
    labels = np.clip(scores + rng.normal(0, 0.1, scores.size), 0.0, 1.0)
    sizes = tuple(len(rows) for rows in _score_thirds(scores))
    assert sizes == ((30, 0, 30) if rows_per_level == 30 else (0, 0, 2))
    folds = np.arange(scores.size) % min(5, scores.size)
    weights = rng.uniform(0.5, 2.0, scores.size)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        calibrator = FlexibleCalibrator(mode="auto").fit(
            scores, labels, folds, sample_weight=weights
        )
    # Here both fits predict exactly the same value on each score level, so
    # two-stage wins no region and the 1-SE rule cannot fire.
    assert calibrator.selected_mode == "monotone"
