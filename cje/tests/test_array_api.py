"""Tests for the array-first API (cje/array_api.py).

Equivalence contract: single-policy `calibrated_mean_ci` must reproduce
`CalibratedDirectEstimator` when both see the same data and the same
calibrator construction — for BOTH inference methods. The bootstrap sides
share the seed and the eval-table layout, so they should agree essentially
bit-for-bit.
"""

from typing import Dict, List, Tuple

import numpy as np
import pytest

from cje import calibrated_mean_ci, transport_audit, CalibratedMeanResult
from cje.calibration.judge import JudgeCalibrator
from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample
from cje.data.models import Dataset
from cje.estimators.direct_method import CalibratedDirectEstimator

# Small but real bootstrap; both sides use the same seed so equivalence is
# exact regardless of B.
N_BOOT = 80
N_ARENA = 400  # subsample of the arena fresh draws (speed)

ArenaArrays = Tuple[str, FreshDrawDataset, np.ndarray, np.ndarray, List[str]]


@pytest.fixture
def arena_single_policy(
    arena_fresh_draws: Dict[str, FreshDrawDataset], arena_dataset: Dataset
) -> ArenaArrays:
    """One policy's arena fresh draws with oracle labels joined from logged data.

    The shipped fresh-draw files carry judge scores only; oracle labels are
    attached by prompt_id from the logged arena data (real labels, ~48%
    coverage), then truncated to N_ARENA rows for speed.
    """
    if not arena_fresh_draws:
        pytest.skip("No fresh draws available")
    policy = sorted(arena_fresh_draws.keys())[0]
    fd = arena_fresh_draws[policy]

    oracle_by_prompt = {
        s.prompt_id: s.oracle_label
        for s in arena_dataset.samples
        if s.oracle_label is not None
    }

    samples = []
    judge: List[float] = []
    labels: List[float] = []
    prompts: List[str] = []
    for sample in fd.samples[:N_ARENA]:
        label = oracle_by_prompt.get(sample.prompt_id)
        samples.append(
            FreshDrawSample(
                prompt_id=sample.prompt_id,
                target_policy=policy,
                judge_score=sample.judge_score,
                oracle_label=label,
                response=None,
                draw_idx=sample.draw_idx,
            )
        )
        judge.append(sample.judge_score)
        labels.append(label if label is not None else np.nan)
        prompts.append(sample.prompt_id)

    labels_arr = np.asarray(labels, dtype=float)
    if int(np.sum(~np.isnan(labels_arr))) < 30:
        pytest.skip("Not enough oracle labels after the join")

    fd_labeled = FreshDrawDataset(
        target_policy=policy,
        samples=samples,
    )
    return policy, fd_labeled, np.asarray(judge, dtype=float), labels_arr, prompts


def _fit_reference_calibrator(
    judge: np.ndarray, labels: np.ndarray, prompts: List[str]
) -> JudgeCalibrator:
    """Fit a calibrator exactly the way calibrated_mean_ci does internally."""
    mask = ~np.isnan(labels)
    calibrator = JudgeCalibrator(random_seed=42, calibration_mode="auto")
    calibrator.fit_cv(
        judge_scores=judge,
        oracle_labels=labels[mask],
        oracle_mask=mask,
        n_folds=5,
        prompt_ids=prompts,
    )
    return calibrator


# ============================================================================
# Equivalence with CalibratedDirectEstimator (arena data)
# ============================================================================


class TestEstimatorEquivalence:
    def test_cluster_robust_matches_direct_estimator(
        self, arena_single_policy: ArenaArrays
    ) -> None:
        policy, fd, judge, labels, prompts = arena_single_policy

        calibrator = _fit_reference_calibrator(judge, labels, prompts)
        estimator = CalibratedDirectEstimator(
            target_policies=[policy],
            reward_calibrator=calibrator,
            inference_method="cluster_robust",
        )
        estimator.add_fresh_draws(policy, fd)
        ref = estimator.fit_and_estimate()
        ref_lo, ref_hi = ref.confidence_interval(alpha=0.05)

        res = calibrated_mean_ci(
            judge,
            labels,
            cluster_ids=prompts,
            inference="cluster_robust",
            n_folds=5,
            seed=42,
        )

        # Point estimate: identical calibrator + identical mean -> exact.
        assert res.estimate == pytest.approx(float(ref.estimates[0]), abs=1e-12)
        # SE: CRV1 with singleton clusters == standard SE analytically; the
        # OUA jackknife term is computed with the same recipe on the same
        # calibrator. Only float-op ordering differs.
        assert res.se == pytest.approx(float(ref.standard_errors[0]), rel=1e-9)
        # CI: same t-critical value (df = min(G-1, K-1)) both sides.
        assert res.ci[0] == pytest.approx(float(ref_lo[0]), rel=1e-9)
        assert res.ci[1] == pytest.approx(float(ref_hi[0]), rel=1e-9)
        assert res.method == "cluster_robust"
        assert res.n == len(judge)
        assert res.n_oracle == int(np.sum(~np.isnan(labels)))

    def test_bootstrap_matches_direct_estimator(
        self, arena_single_policy: ArenaArrays
    ) -> None:
        policy, fd, judge, labels, prompts = arena_single_policy

        calibrator = _fit_reference_calibrator(judge, labels, prompts)
        estimator = CalibratedDirectEstimator(
            target_policies=[policy],
            reward_calibrator=calibrator,
            inference_method="bootstrap",
            n_bootstrap=N_BOOT,
            bootstrap_seed=42,
        )
        estimator.add_fresh_draws(policy, fd)
        ref = estimator.fit_and_estimate()
        ref_lo, ref_hi = ref.confidence_interval(alpha=0.05)

        res = calibrated_mean_ci(
            judge,
            labels,
            cluster_ids=prompts,
            inference="bootstrap",
            n_bootstrap=N_BOOT,
            seed=42,
            n_folds=5,
        )

        # Same seed, same eval-table layout, same refit mode and adaptive
        # min-oracle rule -> the bootstrap draws are identical.
        assert res.estimate == pytest.approx(float(ref.estimates[0]), abs=1e-12)
        assert res.se == pytest.approx(float(ref.standard_errors[0]), rel=1e-9)
        # Percentile CI on the identical bootstrap matrix.
        assert res.ci[0] == pytest.approx(float(ref_lo[0]), abs=1e-12)
        assert res.ci[1] == pytest.approx(float(ref_hi[0]), abs=1e-12)
        assert res.method == "bootstrap"
        assert (
            res.diagnostics["bootstrap"]["n_valid_replicates"]
            == ref.metadata["inference"]["n_bootstrap_valid"]
        )


# ============================================================================
# Mask semantics and determinism (synthetic)
# ============================================================================


def _synthetic(
    n: int = 300, n_labeled: int = 80, seed: int = 7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = rng.uniform(size=n)
    labels_full = np.clip(scores + rng.normal(0, 0.1, size=n), 0, 1)
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=n_labeled, replace=False)] = True
    return scores, labels_full, mask


class TestMaskSemantics:
    def test_mask_vs_nan_default_equivalence(self) -> None:
        """Explicit mask == NaN-derived default mask, and labels outside the
        mask are ignored entirely (no leak)."""
        scores, labels_full, mask = _synthetic()
        labels_nan = np.where(mask, labels_full, np.nan)

        # (a) NaN labels, default mask
        r_nan = calibrated_mean_ci(
            scores, labels_nan, inference="cluster_robust", seed=11
        )
        # (b) same labels, explicit mask
        r_mask_nan = calibrated_mean_ci(
            scores, labels_nan, oracle_mask=mask, inference="cluster_robust", seed=11
        )
        # (c) FULL labels everywhere, explicit mask — masked-out values must
        # not influence anything.
        r_mask_full = calibrated_mean_ci(
            scores, labels_full, oracle_mask=mask, inference="cluster_robust", seed=11
        )

        for other in (r_mask_nan, r_mask_full):
            assert other.estimate == r_nan.estimate
            assert other.se == r_nan.se
            assert other.ci == r_nan.ci
            assert other.n_oracle == r_nan.n_oracle

    def test_seed_determinism_bootstrap(self) -> None:
        scores, labels_full, mask = _synthetic()
        labels = np.where(mask, labels_full, np.nan)

        r1 = calibrated_mean_ci(
            scores, labels, inference="bootstrap", n_bootstrap=40, seed=123
        )
        r2 = calibrated_mean_ci(
            scores, labels, inference="bootstrap", n_bootstrap=40, seed=123
        )

        assert r1.estimate == r2.estimate
        assert r1.se == r2.se
        assert r1.ci == r2.ci
        assert r1.diagnostics["bootstrap"] == r2.diagnostics["bootstrap"]

    def test_auto_resolves_to_bootstrap(self) -> None:
        """The oracle slice lives inside the evaluation sample, so 'auto'
        follows the estimator's coupling rule and selects bootstrap."""
        scores, labels_full, mask = _synthetic()
        labels = np.where(mask, labels_full, np.nan)
        res = calibrated_mean_ci(scores, labels, n_bootstrap=30, seed=5)
        assert res.method == "bootstrap"
        assert "coupled" in res.diagnostics["inference_reason"]


# ============================================================================
# Input validation
# ============================================================================


class TestValidation:
    def test_mismatched_label_length_raises(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            calibrated_mean_ci(np.linspace(0, 1, 50), np.zeros(49))

    def test_mismatched_cluster_ids_raise(self) -> None:
        scores, labels_full, mask = _synthetic(n=60, n_labeled=30)
        labels = np.where(mask, labels_full, np.nan)
        with pytest.raises(ValueError, match="cluster_ids"):
            calibrated_mean_ci(scores, labels, cluster_ids=["a"] * 59)

    def test_mismatched_covariates_raise(self) -> None:
        scores, labels_full, mask = _synthetic(n=60, n_labeled=30)
        labels = np.where(mask, labels_full, np.nan)
        with pytest.raises(ValueError, match="covariates"):
            calibrated_mean_ci(scores, labels, covariates=np.ones((59, 2)))

    def test_all_nan_oracle_raises_clear_message(self) -> None:
        scores = np.linspace(0, 1, 50)
        with pytest.raises(ValueError, match="No oracle labels"):
            calibrated_mean_ci(scores, np.full(50, np.nan))

    def test_mask_selecting_nan_labels_raises(self) -> None:
        scores, labels_full, mask = _synthetic(n=60, n_labeled=30)
        labels = np.where(mask, labels_full, np.nan)
        bad_mask = np.ones(60, dtype=bool)  # selects NaN rows too
        with pytest.raises(ValueError, match="NaN"):
            calibrated_mean_ci(scores, labels, oracle_mask=bad_mask)

    def test_non_boolean_mask_raises(self) -> None:
        scores, labels_full, _ = _synthetic(n=60, n_labeled=30)
        with pytest.raises(ValueError, match="boolean"):
            calibrated_mean_ci(
                scores, labels_full, oracle_mask=np.arange(30)  # index array
            )

    def test_nan_judge_scores_raise(self) -> None:
        scores, labels_full, mask = _synthetic(n=60, n_labeled=30)
        labels = np.where(mask, labels_full, np.nan)
        scores = scores.copy()
        scores[3] = np.nan
        with pytest.raises(ValueError, match="judge_scores"):
            calibrated_mean_ci(scores, labels)

    def test_labels_outside_unit_interval_raise(self) -> None:
        scores, labels_full, mask = _synthetic(n=60, n_labeled=30)
        labels = np.where(mask, labels_full * 10.0, np.nan)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            calibrated_mean_ci(scores, labels)

    def test_invalid_inference_raises(self) -> None:
        scores, labels_full, mask = _synthetic(n=60, n_labeled=30)
        labels = np.where(mask, labels_full, np.nan)
        with pytest.raises(ValueError, match="Invalid inference"):
            calibrated_mean_ci(scores, labels, inference="jackknife")


# ============================================================================
# Boundary diagnostics (coverage badge)
# ============================================================================


def test_boundary_card_flags_out_of_range_scores() -> None:
    """Oracle labels confined to mid-range judge scores -> REFUSE-LEVEL badge."""
    rng = np.random.default_rng(3)
    scores = rng.uniform(size=300)
    labels = np.full(300, np.nan)
    mid = np.where((scores >= 0.4) & (scores <= 0.6))[0]
    labels[mid] = np.clip(scores[mid] + rng.normal(0, 0.05, size=len(mid)), 0, 1)
    if len(mid) < 10:
        pytest.skip("Unlucky draw: too few mid-range samples")

    res = calibrated_mean_ci(scores, labels, inference="cluster_robust", seed=3)
    card = res.diagnostics["boundary_card"]
    assert card["status"] == "REFUSE-LEVEL"
    assert card["out_of_range"] > 0.5
    assert len(card["oracle_s_range"]) == 2


# ============================================================================
# transport_audit
# ============================================================================


class TestTransportAudit:
    @staticmethod
    def _fitted_calibrator(seed: int = 0) -> JudgeCalibrator:
        rng = np.random.default_rng(seed)
        s_train = rng.uniform(size=300)
        y_train = np.clip(s_train + rng.normal(0, 0.05, size=300), 0, 1)
        calibrator = JudgeCalibrator(calibration_mode="monotone", random_seed=seed)
        calibrator.fit_cv(s_train, y_train, n_folds=5)
        return calibrator

    def test_pass_on_well_calibrated_probe(self) -> None:
        calibrator = self._fitted_calibrator()
        rng = np.random.default_rng(1)
        s_probe = rng.uniform(size=200)
        # Residuals alternate ±0.01 -> mean residual exactly 0 (deterministic PASS)
        noise = np.where(np.arange(200) % 2 == 0, 0.01, -0.01)
        y_probe = calibrator.predict(s_probe) + noise

        audit = transport_audit(s_probe, y_probe, calibrator)
        assert audit.status == "PASS"
        assert audit.delta_hat == pytest.approx(0.0, abs=1e-12)
        assert audit.n_probe == 200

    def test_fail_on_shifted_probe(self) -> None:
        calibrator = self._fitted_calibrator()
        rng = np.random.default_rng(2)
        s_probe = rng.uniform(size=200)
        noise = np.where(np.arange(200) % 2 == 0, 0.01, -0.01)
        y_probe = calibrator.predict(s_probe) - 0.2 + noise

        audit = transport_audit(
            s_probe, y_probe, calibrator, group_label="policy:shifted"
        )
        assert audit.status == "FAIL"
        assert audit.delta_hat == pytest.approx(-0.2, abs=1e-6)
        assert audit.group_label == "policy:shifted"

    def test_nan_probe_labels_are_dropped(self) -> None:
        calibrator = self._fitted_calibrator()
        rng = np.random.default_rng(4)
        s_probe = rng.uniform(size=100)
        y_probe = calibrator.predict(s_probe).astype(float)
        y_probe[::2] = np.nan  # half unlabeled

        audit = transport_audit(s_probe, y_probe, calibrator)
        assert audit.n_probe == 50

    def test_too_few_labeled_probes_raise(self) -> None:
        calibrator = self._fitted_calibrator()
        with pytest.raises(ValueError, match="at least 2"):
            transport_audit(
                np.array([0.1, 0.5, 0.9]),
                np.array([np.nan, 0.5, np.nan]),
                calibrator,
            )

    def test_composes_with_calibrated_mean_ci(self) -> None:
        """The calibrator returned by calibrated_mean_ci feeds transport_audit."""
        scores, labels_full, mask = _synthetic()
        labels = np.where(mask, labels_full, np.nan)
        res: CalibratedMeanResult = calibrated_mean_ci(
            scores, labels, inference="cluster_robust", seed=9
        )
        rng = np.random.default_rng(10)
        s_probe = rng.uniform(size=150)
        y_probe = np.clip(s_probe + rng.normal(0, 0.1, size=150), 0, 1)
        audit = transport_audit(s_probe, y_probe, res.calibrator, bins=5)
        assert audit.status in ("PASS", "WARN", "FAIL")
        assert len(audit.decile_counts) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
