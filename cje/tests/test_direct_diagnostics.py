"""Regression tests for Direct-mode diagnostics and the coverage badge.

Ports the surviving boundary-card cases from the retired 0.3.x
test_refusal_gates.py (findings #31/#20: coverage badge wired in; coverage
risk never silently "low") and pins the 0.4.0 wiring:

- the boundary_card primitive (REFUSE-LEVEL at >= 5% out-of-range mass);
- CalibratedDirectEstimator computes per-policy cards against the oracle
  S-range the calibrator stored at fit time, sets CRITICAL statuses, and
  writes metadata["boundary_cards"] + metadata["reliability_gates"];
- DirectDiagnostics surfaces the badge (validate/summary/overall_status);
- the CLI trophy demotion fires for a boundary-violating argmax;
- IPSDiagnostics is a deprecated alias of DirectDiagnostics (gone in 0.5.0).
"""

from typing import Tuple

import numpy as np
import pytest

from cje.calibration import calibrate_dataset
from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample
from cje.data.models import Dataset, EstimationResult, Sample
from cje.diagnostics import (
    BoundaryCard,
    DirectDiagnostics,
    IPSDiagnostics,
    Status,
    boundary_card,
)
from cje.estimators.direct_method import CalibratedDirectEstimator
from cje.interface.cli import best_policy_lines

pytestmark = pytest.mark.unit

POLICY = "target"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _calibration_dataset(
    rng: np.random.Generator,
    n: int = 300,
    s_max: float = 0.8,
    oracle_frac: float = 0.5,
) -> Dataset:
    """Logged data whose oracle labels only cover judge scores <= s_max."""
    samples = []
    for i in range(n):
        s = float(rng.uniform())
        y = float(np.clip(0.25 + 0.5 * s + rng.normal(0, 0.08), 0, 1))
        has_oracle = s <= s_max and rng.uniform() < oracle_frac
        samples.append(
            Sample(
                prompt_id=f"p{i}",
                prompt=f"question {i}",
                response=f"answer {i}",
                reward=None,
                base_policy_logprob=-10.0,
                target_policy_logprobs={POLICY: -10.0},
                judge_score=s,
                oracle_label=y if has_oracle else None,
            )
        )
    return Dataset(samples=samples, target_policies=[POLICY])


def _fresh(policy: str, scores: np.ndarray) -> FreshDrawDataset:
    samples = [
        FreshDrawSample(
            prompt_id=f"p{i}",
            judge_score=float(s),
            oracle_label=None,
            response=None,
            fold_id=None,
            target_policy=policy,
            draw_idx=0,
        )
        for i, s in enumerate(scores)
    ]
    return FreshDrawDataset(samples=samples, target_policy=policy, draws_per_prompt=1)


def _run_direct(
    dataset: Dataset, fresh_scores: np.ndarray
) -> Tuple[CalibratedDirectEstimator, EstimationResult]:
    _, cal_result = calibrate_dataset(
        dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
        enable_cross_fit=True,
        n_folds=5,
    )
    estimator = CalibratedDirectEstimator(
        target_policies=[POLICY],
        reward_calibrator=cal_result.calibrator,
        inference_method="cluster_robust",
    )
    estimator.add_fresh_draws(POLICY, _fresh(POLICY, fresh_scores))
    result = estimator.fit_and_estimate()
    return estimator, result


def _make_direct_diagnostics(**overrides: object) -> DirectDiagnostics:
    defaults: dict = dict(
        estimator_type="Direct",
        method="calibrated_direct",
        n_samples_total=100,
        n_samples_valid=100,
        policies=["pi"],
        estimates={"pi": 0.7},
        standard_errors={"pi": 0.02},
        n_samples_used={"pi": 100},
    )
    defaults.update(overrides)
    return DirectDiagnostics(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The boundary_card primitive (ported from 0.3.x TestBoundaryCard)
# ---------------------------------------------------------------------------


class TestBoundaryCardPrimitive:
    """Findings #31/#20: coverage badge fires at >= 5% out-of-range mass."""

    def test_refuse_level_fires_for_out_of_range_fresh_scores(self) -> None:
        rng = np.random.default_rng(4)
        S_oracle = rng.uniform(0.3, 0.7, 200)  # calibration support
        # 10% of the policy's (fresh) judge scores extrapolate above 0.7
        S_policy = np.concatenate(
            [rng.uniform(0.3, 0.7, 900), rng.uniform(0.75, 0.95, 100)]
        )
        R_policy = rng.uniform(0.35, 0.65, 1000)
        card = boundary_card(
            S_policy=S_policy,
            S_oracle=S_oracle,
            R_policy=R_policy,
            R_min=0.3,
            R_max=0.7,
        )
        assert isinstance(card, BoundaryCard)
        assert card.status == "REFUSE-LEVEL"
        assert card.out_of_range == pytest.approx(0.10, abs=0.01)

    def test_in_range_scores_are_ok(self) -> None:
        rng = np.random.default_rng(5)
        S = rng.uniform(0.3, 0.7, 500)
        card = boundary_card(
            S_policy=S,
            S_oracle=np.array([0.25, 0.75]),
            R_policy=rng.uniform(0.4, 0.6, 500),
            R_min=0.0,
            R_max=1.0,
        )
        assert card.status == "OK"


# ---------------------------------------------------------------------------
# Direct wiring: cards computed during estimation
# ---------------------------------------------------------------------------


class TestBoundaryCardWiredIntoDirect:
    """The 0.3.x wiring lived in CalibratedIPS; 0.4.0 computes the badge in
    CalibratedDirectEstimator against the calibrator's stored oracle
    S-range."""

    def test_refuse_level_fires_end_to_end(self) -> None:
        # Oracle labels exist only for S in [0.0, 0.8]; ~20% of the fresh
        # draws' judge scores sit above the oracle range -> REFUSE-LEVEL
        rng = np.random.default_rng(6)
        dataset = _calibration_dataset(rng, s_max=0.8)
        fresh_scores = np.concatenate(
            [rng.uniform(0.0, 0.8, 400), rng.uniform(0.85, 0.99, 100)]
        )
        estimator, result = _run_direct(dataset, fresh_scores)

        diag = estimator.get_diagnostics()
        assert diag is not None
        assert isinstance(diag, DirectDiagnostics)
        assert diag.boundary_cards is not None
        card = diag.boundary_cards[POLICY]
        assert card["status"] == "REFUSE-LEVEL"
        assert card["out_of_range"] >= 0.05
        # oracle_s_range is recorded alongside the card
        s_lo, s_hi = card["oracle_s_range"]
        assert 0.0 <= s_lo < s_hi <= 0.8

        # Metadata keys: boundary_cards + reliability_gates (CLI contract)
        assert result.metadata["boundary_cards"][POLICY] == card
        gate = result.metadata["reliability_gates"][POLICY]
        assert gate["flagged"] is True
        assert gate["refused"] is False  # estimates still reported
        assert gate["refuse_level_claims"] is True
        assert any("boundary" in reason for reason in gate["reasons"])

        # Status: the badge escalates the policy (and overall) to CRITICAL
        assert diag.status_per_policy is not None
        assert diag.status_per_policy[POLICY] == Status.CRITICAL
        assert diag.overall_status == Status.CRITICAL

        # validate() names the problem; summary() surfaces it
        assert any("REFUSE-LEVEL" in issue for issue in diag.validate())
        assert "REFUSE-LEVEL" in diag.summary()
        assert diag.refuse_level_policies == [POLICY]

    def test_in_range_run_is_ok_and_unflagged(self) -> None:
        rng = np.random.default_rng(7)
        dataset = _calibration_dataset(rng, s_max=0.8)
        fresh_scores = rng.uniform(0.05, 0.75, 500)
        estimator, result = _run_direct(dataset, fresh_scores)

        diag = estimator.get_diagnostics()
        assert diag is not None and diag.boundary_cards is not None
        assert diag.boundary_cards[POLICY]["status"] == "OK"
        assert diag.status_per_policy is not None
        assert diag.status_per_policy[POLICY] == Status.GOOD
        assert result.metadata["reliability_gates"][POLICY]["flagged"] is False
        assert not any("REFUSE-LEVEL" in issue for issue in diag.validate())

    def test_naive_direct_has_no_cards(self) -> None:
        # Without a reward calibrator there is no oracle range to check:
        # no boundary cards, no reliability gates in metadata
        rng = np.random.default_rng(8)
        estimator = CalibratedDirectEstimator(
            target_policies=[POLICY],
            reward_calibrator=None,
            inference_method="cluster_robust",
        )
        estimator.add_fresh_draws(POLICY, _fresh(POLICY, rng.uniform(0, 1, 100)))
        result = estimator.fit_and_estimate()
        assert result.diagnostics is not None
        assert result.diagnostics.boundary_cards is None
        assert "boundary_cards" not in result.metadata
        assert "reliability_gates" not in result.metadata

    def test_cli_trophy_demotes_boundary_violating_argmax(self) -> None:
        # End-to-end: the violating policy wins the raw argmax (isotonic
        # clipping maps its out-of-range scores to the top reward) but the
        # CLI demotes it and crowns the best RELIABLE policy instead.
        rng = np.random.default_rng(9)
        dataset = _calibration_dataset(rng, s_max=0.6)
        _, cal_result = calibrate_dataset(
            dataset,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,
            n_folds=5,
        )
        estimator = CalibratedDirectEstimator(
            target_policies=["safe", "violator"],
            reward_calibrator=cal_result.calibrator,
            inference_method="cluster_robust",
        )
        estimator.add_fresh_draws("safe", _fresh("safe", rng.uniform(0.05, 0.55, 200)))
        violator_scores = np.concatenate(
            [rng.uniform(0.3, 0.55, 120), rng.uniform(0.7, 0.95, 80)]
        )
        estimator.add_fresh_draws("violator", _fresh("violator", violator_scores))
        result = estimator.fit_and_estimate()

        # The violator wins the raw argmax...
        assert result.estimates[1] > result.estimates[0]
        # ...but only its gate is flagged
        gates = result.metadata["reliability_gates"]
        assert gates["violator"]["flagged"] is True
        assert gates["safe"]["flagged"] is False

        lines = best_policy_lines(result)
        assert any(
            "Best by point estimate: violator" in line and "UNRELIABLE" in line
            for line in lines
        )
        assert not any(line.startswith("🏆 Best policy:") for line in lines)
        assert any("Best reliable policy: safe" in line for line in lines)


# ---------------------------------------------------------------------------
# DirectDiagnostics surface (coverage-risk cases adapted from 0.3.x)
# ---------------------------------------------------------------------------


class TestDirectDiagnosticsSurface:
    """The badge is the coverage-risk carrier; no silent 'all clear'."""

    def test_validate_and_summary_report_refuse_level(self) -> None:
        diag = _make_direct_diagnostics(
            boundary_cards={"pi": {"status": "REFUSE-LEVEL", "out_of_range": 0.12}},
            status_per_policy={"pi": Status.CRITICAL},
        )
        issues = diag.validate()
        assert any("REFUSE-LEVEL" in issue for issue in issues)
        assert any("12.0%" in issue for issue in issues)
        assert "REFUSE-LEVEL: pi" in diag.summary()
        assert diag.refuse_level_policies == ["pi"]
        assert diag.overall_status == Status.CRITICAL

    def test_ok_cards_are_clean(self) -> None:
        diag = _make_direct_diagnostics(
            boundary_cards={"pi": {"status": "OK", "out_of_range": 0.0}},
            status_per_policy={"pi": Status.GOOD},
        )
        assert diag.validate() == []
        assert diag.refuse_level_policies == []
        assert diag.overall_status == Status.GOOD

    def test_no_cards_means_no_refuse_claims(self) -> None:
        diag = _make_direct_diagnostics()
        assert diag.boundary_cards is None
        assert diag.refuse_level_policies == []

    def test_overall_status_is_worst_per_policy(self) -> None:
        diag = _make_direct_diagnostics(
            policies=["a", "b"],
            estimates={"a": 0.5, "b": 0.6},
            standard_errors={"a": 0.02, "b": 0.02},
            n_samples_used={"a": 50, "b": 50},
            status_per_policy={"a": Status.GOOD, "b": Status.CRITICAL},
        )
        assert diag.overall_status == Status.CRITICAL

    def test_to_dict_round_trip(self) -> None:
        diag = _make_direct_diagnostics(
            boundary_cards={"pi": {"status": "OK", "out_of_range": 0.0}},
            status_per_policy={"pi": Status.GOOD},
        )
        d = diag.to_dict()
        assert d["status_per_policy"] == {"pi": "good"}
        assert d["overall_status"] == "good"
        rebuilt = DirectDiagnostics.from_dict(d)
        assert rebuilt.status_per_policy == {"pi": Status.GOOD}
        assert rebuilt.boundary_cards == diag.boundary_cards


# ---------------------------------------------------------------------------
# Deprecated alias
# ---------------------------------------------------------------------------


class TestDeprecatedAlias:
    def test_ips_diagnostics_is_direct_diagnostics(self) -> None:
        # 0.3.x name kept as an alias until 0.5.0 — same object, not a copy
        assert IPSDiagnostics is DirectDiagnostics

    def test_alias_importable_from_advanced(self) -> None:
        from cje.advanced import IPSDiagnostics as AdvancedAlias

        assert AdvancedAlias is DirectDiagnostics
