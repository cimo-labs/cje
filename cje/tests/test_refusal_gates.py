"""Regression tests for the paper's refusal gates (arXiv:2512.11150).

Covers:
- One canonical ESS ladder: a policy at ESS 0.25 gets the same verdict on
  every surface (gates helper, weight diagnostics, display fallback,
  overlap recommendations).
- hill_tail_index returning NaN (not +inf) on estimation failure, with
  downstream consumers treating NaN as "unknown" rather than the
  healthiest possible tail.
- TTC (Target-Typicality Coverage) computed with the paper's per-token
  construction (mass 0.9), populated into IPSDiagnostics, and folded
  into weight_status.
"""

from typing import Any, Callable, Optional, Tuple

import numpy as np
import pytest

from cje.calibration import calibrate_dataset
from cje.data.models import Dataset, EstimationResult, Sample
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.diagnostics import (
    IPSDiagnostics,
    Status,
    compute_weight_diagnostics,
    create_weight_summary_table,
    ess_status,
    hill_tail_index,
    tail_status,
    ttc_status,
    worst_status,
)
from cje.diagnostics import (
    BoundaryCard,
    CJEDiagnostics,
    bhattacharyya_judge_space,
    bhattacharyya_status,
    boundary_card,
)
from cje.diagnostics.gates import (
    ESS_GOOD_THRESHOLD,
    ESS_WARNING_THRESHOLD,
    format_tail_index,
)
from cje.diagnostics.overlap import (
    compute_cle_diagnostics,
    compute_overlap_metrics,
    compute_ttc,
)
from cje.estimators.calibrated_ips import CalibratedIPS

POLICY = "target"


def _tilted_dataset(
    n: int,
    rng: np.random.Generator,
    tilt: Callable[[float], float],
    oracle_frac: float = 0.5,
) -> Dataset:
    """DGP from test_mc_coverage: tilted logprobs encode w(S) exactly."""
    samples = []
    for i in range(n):
        s = float(rng.uniform())
        y = float(np.clip(0.25 + 0.5 * s + rng.normal(0, 0.08), 0, 1))
        samples.append(
            Sample(
                prompt_id=f"p{i}",
                prompt=f"question {i}",
                response=f"answer {i}",
                reward=None,
                base_policy_logprob=-10.0,
                target_policy_logprobs={POLICY: -10.0 + float(np.log(tilt(s)))},
                judge_score=s,
                oracle_label=y if rng.uniform() < oracle_frac else None,
            )
        )
    return Dataset(samples=samples, target_policies=[POLICY])


def _run_calibrated_ips(dataset: Dataset) -> Tuple[CalibratedIPS, EstimationResult]:
    calibrated, cal_result = calibrate_dataset(
        dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
        enable_cross_fit=True,
        n_folds=5,
    )
    sampler = PrecomputedSampler(calibrated)
    estimator = CalibratedIPS(sampler, reward_calibrator=cal_result.calibrator)
    result = estimator.fit_and_estimate()
    return estimator, result


def _weights_with_ess(ess_fraction: float, n: int = 1000) -> np.ndarray:
    """Two-point weights: a fraction `ess_fraction` carry all the mass.

    ESS/n = (mean w)^2 / mean(w^2) -> ess_fraction as the small weight -> 0.
    """
    k = int(round(n * ess_fraction))
    weights = np.full(n, 1e-9)
    weights[:k] = 1.0
    return weights


def _make_ips_diagnostics(**overrides: object) -> IPSDiagnostics:
    defaults: dict = dict(
        estimator_type="CalibratedIPS",
        method="calibrated_ips",
        n_samples_total=1000,
        n_samples_valid=1000,
        n_policies=1,
        policies=["pi"],
        estimates={"pi": 0.5},
        standard_errors={"pi": 0.05},
        n_samples_used={"pi": 1000},
        weight_ess=0.25,
        weight_status=Status.WARNING,
        ess_per_policy={"pi": 0.25},
        max_weight_per_policy={"pi": 4.0},
    )
    defaults.update(overrides)
    return IPSDiagnostics(**defaults)  # type: ignore[arg-type]


class TestCanonicalESSLadder:
    """Finding #7: ESS 0.25 must get the same verdict everywhere."""

    def test_gates_ladder_matches_paper(self) -> None:
        assert ESS_GOOD_THRESHOLD == 0.30  # paper ship-gate
        assert ESS_WARNING_THRESHOLD == 0.10
        assert ess_status(0.35) == Status.GOOD
        assert ess_status(0.30) == Status.GOOD
        assert ess_status(0.25) == Status.WARNING
        assert ess_status(0.10) == Status.WARNING
        assert ess_status(0.05) == Status.CRITICAL

    def test_compute_weight_diagnostics_agrees_at_025(self) -> None:
        weights = _weights_with_ess(0.25)
        diag = compute_weight_diagnostics(weights)
        assert diag["ess_fraction"] == pytest.approx(0.25, abs=0.01)
        assert diag["status"] == Status.WARNING

    def test_display_fallback_agrees_at_025(self) -> None:
        # No status_per_policy: the display fallback must use the same ladder
        diagnostics = _make_ips_diagnostics(status_per_policy=None)
        table = create_weight_summary_table(diagnostics)
        row = next(line for line in table.splitlines() if line.startswith("pi"))
        assert "warning" in row
        # Old thresholds (0.5/0.2) graded 0.25 as WARNING too, but graded
        # 0.15 as CRITICAL; the canonical ladder keeps it WARNING.
        diagnostics = _make_ips_diagnostics(
            status_per_policy=None,
            ess_per_policy={"pi": 0.15},
            weight_ess=0.15,
        )
        table = create_weight_summary_table(diagnostics)
        row = next(line for line in table.splitlines() if line.startswith("pi"))
        assert "warning" in row and "critical" not in row

    def test_overlap_recommendation_uses_ship_gate(self) -> None:
        # ESS 0.25 (below the 0.30 ship-gate): calibration recommended
        rec_below = compute_overlap_metrics(
            _weights_with_ess(0.25), compute_tail_index=False
        ).recommended_method
        assert rec_below == "calibrated-ips"
        # ESS 0.35 (above the gate): plain IPS is fine
        rec_above = compute_overlap_metrics(
            _weights_with_ess(0.35), compute_tail_index=False
        ).recommended_method
        assert rec_above == "ips"

    def test_worst_status_combines(self) -> None:
        assert worst_status(Status.GOOD, None, Status.WARNING) == Status.WARNING
        assert worst_status(Status.CRITICAL, Status.GOOD) == Status.CRITICAL
        assert worst_status() == Status.GOOD


class TestHillTailIndexFailure:
    """Finding #66: estimation failure must be NaN (unknown), not +inf."""

    def test_96pct_zeros_returns_nan(self) -> None:
        weights = np.zeros(1000)
        weights[:40] = 1.0  # 96% exact zeros: Hill cannot be estimated
        result = hill_tail_index(weights)
        assert np.isnan(result), f"expected NaN for degenerate weights, got {result}"

    def test_too_few_samples_returns_nan(self) -> None:
        assert np.isnan(hill_tail_index(np.ones(5)))

    def test_uniform_tail_keeps_inf(self) -> None:
        # Genuinely uniform tail: lightest possible, inf is CORRECT here
        assert np.isinf(hill_tail_index(np.ones(1000)))

    def test_downstream_status_treats_nan_as_unknown(self) -> None:
        # NaN tail + low ESS -> at least WARNING (previously: inf passed
        # every check and the ESS 4% case could read healthier than it is)
        assert tail_status(float("nan"), ess_fraction=0.25) == Status.WARNING
        # NaN tail with healthy ESS -> unknown, no independent escalation
        assert tail_status(float("nan"), ess_fraction=0.50) is None
        # inf (uniform tail) is genuinely light
        assert tail_status(float("inf"), ess_fraction=0.25) == Status.GOOD

    def test_compute_weight_diagnostics_degenerate_weights(self) -> None:
        weights = np.zeros(1000)
        weights[:40] = 1.0
        diag = compute_weight_diagnostics(weights)
        # ESS = 4% -> CRITICAL on the canonical ladder; tail is NaN not inf
        assert diag["status"] == Status.CRITICAL
        assert np.isnan(diag["tail_index"])

    def test_worst_tail_index_property_handles_nan(self) -> None:
        diag = _make_ips_diagnostics(tail_indices={"pi": float("nan")})
        worst = diag.worst_tail_index
        assert worst is not None and np.isnan(worst)
        # A finite index elsewhere wins over the failed one
        diag = _make_ips_diagnostics(
            policies=["pi", "rho"],
            tail_indices={"pi": float("nan"), "rho": 2.5},
        )
        assert diag.worst_tail_index == 2.5

    def test_validate_reports_failed_tail(self) -> None:
        diag = _make_ips_diagnostics(tail_indices={"pi": float("nan")})
        issues = diag.validate()
        assert any("estimation failed" in issue for issue in issues)

    def test_format_tail_index(self) -> None:
        assert format_tail_index(float("nan")) == "n/a"
        assert format_tail_index(None) == "n/a"
        assert format_tail_index(float("inf")) == "inf"
        assert format_tail_index(2.345) == "2.35"


class TestTTCStatistic:
    """Findings #27/#28: TTC must use the paper's per-token construction."""

    def test_default_mass_is_090(self) -> None:
        # Identical policies: TTC = target_typical_mass, which must default
        # to the paper's q_0.9 construction (was 0.8)
        rng = np.random.default_rng(7)
        base_lp = rng.normal(-50, 10, 2000)
        assert compute_ttc(base_lp, base_lp.copy()) == pytest.approx(0.90, abs=0.02)
        assert compute_cle_diagnostics(base_lp, base_lp.copy()).alpha == pytest.approx(
            0.90, abs=0.03
        )

    def test_per_token_normalization_removes_length_confound(self) -> None:
        # Two groups of logged samples:
        # - 900 long responses: total surprisal 100 over 1000 tokens
        #   (0.1/token, the MOST per-token-typical outputs)
        # - 100 short responses: total surprisal 20 over 10 tokens
        #   (2.0/token) carrying ~91% of the target's mass (w = 9.1)
        # Total-NLL ranks every long response as more surprising than every
        # short one, so T collapses to the short-response region and TTC
        # reads ~0.1 — a false alarm driven purely by length (the paper's
        # confound). The per-token statistic includes both groups (both are
        # within the q_0.9 per-token threshold under the target), TTC = 1.
        n_long, n_short = 900, 100
        target_lp = np.concatenate([np.full(n_long, -100.0), np.full(n_short, -20.0)])
        log_w = np.concatenate(
            [np.full(n_long, np.log(0.1)), np.full(n_short, np.log(9.1))]
        )
        base_lp = target_lp - log_w
        token_counts = np.concatenate([np.full(n_long, 1000.0), np.full(n_short, 10.0)])

        # Per-token (paper statistic): logger covers the target-typical set
        ttc_per_token = compute_ttc(base_lp, target_lp, token_counts=token_counts)
        assert ttc_per_token >= 0.99

        # Total surprisal (old statistic): T becomes "short responses only"
        ttc_total = compute_ttc(base_lp, target_lp)
        assert ttc_total <= 0.15

    def test_token_counts_validation(self) -> None:
        base_lp = np.full(10, -10.0)
        with pytest.raises(ValueError):
            compute_ttc(base_lp, base_lp, token_counts=np.ones(5))
        with pytest.raises(ValueError):
            compute_ttc(base_lp, base_lp, token_counts=np.zeros(10))


class TestTTCWiredIntoPipeline:
    """Finding #27: TTC must be computed and surfaced during estimation."""

    def test_ttc_populated_in_ips_diagnostics(self) -> None:
        rng = np.random.default_rng(42)
        # Mild mean-one tilt: w(S) = 0.4 + 1.2 S (same DGP as test_mc_coverage)
        dataset = _tilted_dataset(500, rng, lambda s: 0.4 + 1.2 * s)
        estimator, result = _run_calibrated_ips(dataset)

        diag = estimator.get_diagnostics()
        assert diag is not None
        assert diag.ttc_per_policy is not None
        assert POLICY in diag.ttc_per_policy
        ttc = diag.ttc_per_policy[POLICY]
        assert 0.0 < ttc <= 1.0
        # Mild tilt: coverage should be healthy
        assert ttc >= 0.7

        # The length-normalizer used must be recorded (char proxy here:
        # the synthetic samples have no response_token_count metadata)
        ttc_meta = result.metadata["ttc_diagnostics"][POLICY]
        assert ttc_meta["length_normalizer"] == "response_chars"
        assert ttc_meta["ttc"] == pytest.approx(ttc)

        # Surfaced in the human-readable summary
        assert "TTC" in diag.summary()

    def test_token_count_metadata_is_preferred(self) -> None:
        rng = np.random.default_rng(3)
        dataset = _tilted_dataset(300, rng, lambda s: 0.4 + 1.2 * s)
        for sample in dataset.samples:
            sample.metadata["response_token_count"] = 25
        estimator, result = _run_calibrated_ips(dataset)
        ttc_meta = result.metadata["ttc_diagnostics"][POLICY]
        assert ttc_meta["length_normalizer"] == "response_token_count"

    def test_bc_sigmaS_present_and_graded_after_run(self) -> None:
        rng = np.random.default_rng(5)
        dataset = _tilted_dataset(400, rng, lambda s: 0.4 + 1.2 * s)
        estimator, _ = _run_calibrated_ips(dataset)
        diag = estimator.get_diagnostics()
        assert diag is not None
        assert diag.bc_sigmaS_per_policy is not None
        bc = diag.bc_sigmaS_per_policy[POLICY]
        assert 0.0 <= bc <= 1.0
        # Mild tilt: judge marginals nearly coincide, gate should pass
        assert bc >= 0.85
        assert bhattacharyya_status(bc) == Status.GOOD

    def test_low_ttc_escalates_weight_status(self) -> None:
        rng = np.random.default_rng(11)
        # Concentrated tilt: target puts ~91% of its mass on the top decile
        # of S, so the logger covers the target-typical region on only ~10%
        # of samples -> TTC ~= 0.1 < 0.30 -> CRITICAL, even though ESS is
        # far above the 1% the old ladder needed for CRITICAL.
        dataset = _tilted_dataset(800, rng, lambda s: 9.1 if s > 0.9 else 0.1)
        estimator, _ = _run_calibrated_ips(dataset)

        diag = estimator.get_diagnostics()
        assert diag is not None
        assert diag.ttc_per_policy is not None
        ttc = diag.ttc_per_policy[POLICY]
        assert ttc < 0.30
        assert ttc_status(ttc) == Status.CRITICAL
        assert diag.weight_status == Status.CRITICAL
        assert diag.status_per_policy is not None
        assert diag.status_per_policy[POLICY] == Status.CRITICAL
        # And validate() names the problem
        assert any("TTC" in issue for issue in diag.validate())


class TestJudgeSpaceBhattacharyya:
    """Finding #29: the paper gates on judge-space A_B >= 0.85, not E[sqrt(w)]."""

    def test_identical_marginals_give_one(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 1, 5000)
        # Uniform weights: pi' marginal == pi0 marginal
        assert bhattacharyya_judge_space(scores, weights=np.ones(5000)) >= 0.99
        # Fresh-draw mode with an i.i.d. copy
        assert (
            bhattacharyya_judge_space(
                scores, judge_scores_target=rng.uniform(0, 1, 5000)
            )
            > 0.95
        )

    def test_disjoint_marginals_give_zero(self) -> None:
        rng = np.random.default_rng(1)
        logged = rng.uniform(0.0, 0.4, 2000)
        target = rng.uniform(0.6, 1.0, 2000)
        bc = bhattacharyya_judge_space(logged, judge_scores_target=target)
        assert bc <= 0.05
        assert bhattacharyya_status(bc) == Status.CRITICAL

    def test_gate_grading(self) -> None:
        assert bhattacharyya_status(0.90) == Status.GOOD
        assert bhattacharyya_status(0.85) == Status.GOOD
        assert bhattacharyya_status(0.70) == Status.WARNING
        assert bhattacharyya_status(0.40) == Status.CRITICAL

    def test_requires_weights_or_target_scores(self) -> None:
        with pytest.raises(ValueError):
            bhattacharyya_judge_space(np.linspace(0, 1, 10))

    def test_overlap_metrics_populates_sigmaS_fields(self) -> None:
        rng = np.random.default_rng(2)
        scores = rng.uniform(0, 1, 1000)
        weights = 0.4 + 1.2 * scores  # mean-one tilt in S
        metrics = compute_overlap_metrics(
            weights, compute_tail_index=False, judge_scores=scores
        )
        assert metrics.bc_sigmaS is not None
        assert 0.0 < metrics.bc_sigmaS <= 1.0
        assert metrics.aessf_sigmaS == pytest.approx(metrics.bc_sigmaS**2)

    def test_ab_upper_bounds_action_space_affinity(self) -> None:
        # Data-processing inequality: judge-space A_B >= E[sqrt(w)]
        from cje.diagnostics import hellinger_affinity

        rng = np.random.default_rng(3)
        scores = rng.uniform(0, 1, 4000)
        weights = np.exp(rng.normal(0, 1.0, 4000))  # weights independent of S
        bc = bhattacharyya_judge_space(scores, weights=weights)
        assert bc >= hellinger_affinity(weights) - 0.02


class TestBoundaryCard:
    """Findings #31/#20: coverage badge wired in; coverage_risk never a
    hardcoded 'low'."""

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

    def test_cje_diagnostics_coverage_risk_follows_cards(self) -> None:
        ips = _make_ips_diagnostics(
            boundary_cards={
                "pi": {"status": "REFUSE-LEVEL", "out_of_range": 0.12},
            }
        )
        unified = CJEDiagnostics.from_ips_diagnostics(ips)
        assert unified.coverage_risk == "critical"
        assert unified.refuse_level_claims is True
        assert unified.can_make_level_claims is False
        assert unified.extrapolation_rate == pytest.approx(0.12)
        assert "REFUSE LEVEL CLAIMS" in unified.summary()

    def test_cje_diagnostics_ok_cards_give_low(self) -> None:
        ips = _make_ips_diagnostics(
            boundary_cards={"pi": {"status": "OK", "out_of_range": 0.0}}
        )
        unified = CJEDiagnostics.from_ips_diagnostics(ips)
        assert unified.coverage_risk == "low"
        assert unified.refuse_level_claims is False

    def test_cje_diagnostics_unknown_without_cards(self) -> None:
        # No boundary cards computed: coverage must be UNKNOWN (not 'low'),
        # and level claims must not be certified
        ips = _make_ips_diagnostics()
        unified = CJEDiagnostics.from_ips_diagnostics(ips)
        assert unified.coverage_risk == "unknown"
        assert unified.refuse_level_claims is False
        assert unified.can_make_level_claims is False

    def test_boundary_card_wired_into_estimation(self) -> None:
        # Oracle labels exist only for S in [0.0, 0.8]: ~20% of logged judge
        # scores sit above the oracle calibration range -> REFUSE-LEVEL
        rng = np.random.default_rng(6)
        samples = []
        for i in range(500):
            s = float(rng.uniform())
            y = float(np.clip(0.25 + 0.5 * s + rng.normal(0, 0.08), 0, 1))
            has_oracle = s <= 0.8 and rng.uniform() < 0.5
            samples.append(
                Sample(
                    prompt_id=f"p{i}",
                    prompt=f"question {i}",
                    response=f"answer {i}",
                    reward=None,
                    base_policy_logprob=-10.0,
                    target_policy_logprobs={
                        POLICY: -10.0 + float(np.log(0.4 + 1.2 * s))
                    },
                    judge_score=s,
                    oracle_label=y if has_oracle else None,
                )
            )
        dataset = Dataset(samples=samples, target_policies=[POLICY])
        estimator, result = _run_calibrated_ips(dataset)

        diag = estimator.get_diagnostics()
        assert diag is not None
        assert diag.boundary_cards is not None
        card = diag.boundary_cards[POLICY]
        assert card["status"] == "REFUSE-LEVEL"
        assert card["out_of_range"] >= 0.05
        assert result.metadata["boundary_cards"][POLICY] == card

        # coverage_risk follows the badge; refuse_level_claims can now fire
        unified = CJEDiagnostics.from_ips_diagnostics(diag)
        assert unified.coverage_risk == "critical"
        assert unified.refuse_level_claims is True
        # validate() names the problem
        assert any("REFUSE-LEVEL" in issue for issue in diag.validate())


def _catastrophic_overlap_records(n: int = 200) -> list:
    """96% of samples have near-zero raw importance weight (log w = -50)."""
    rng = np.random.default_rng(8)
    records = []
    for i in range(n):
        s = float(rng.uniform())
        log_w = float(np.log(25.0)) if s > 0.96 else -50.0
        metadata = {"judge_score": s}
        if rng.uniform() < 0.25:
            metadata["oracle_label"] = float(
                np.clip(0.25 + 0.5 * s + rng.normal(0, 0.08), 0, 1)
            )
        records.append(
            {
                "prompt_id": f"p{i}",
                "prompt": f"question {i}",
                "response": f"answer {i}",
                "base_policy_logprob": -10.0,
                "target_policy_logprobs": {POLICY: -10.0 + log_w},
                "metadata": metadata,
            }
        )
    return records


class TestRefuseUnreliablePlumbing:
    """Finding #30: refuse_unreliable must be reachable from analyze_dataset
    and reach DR's internal ips_estimator."""

    def test_analyze_dataset_default_warns_but_estimates(self, tmp_path: Any) -> None:
        import json

        from cje import analyze_dataset

        path = tmp_path / "catastrophic.jsonl"
        with open(path, "w") as f:
            for record in _catastrophic_overlap_records():
                f.write(json.dumps(record) + "\n")

        results = analyze_dataset(
            logged_data_path=str(path), estimator="calibrated-ips"
        )
        # Default: numeric estimate returned, but the gate outcome is recorded
        assert not np.isnan(results.estimates[0])
        gates = results.metadata["reliability_gates"]
        assert gates[POLICY]["flagged"] is True
        assert gates[POLICY]["refused"] is False
        assert gates[POLICY]["reasons"]

    def test_analyze_dataset_refuse_unreliable_returns_nan(self, tmp_path: Any) -> None:
        import json

        from cje import analyze_dataset

        path = tmp_path / "catastrophic.jsonl"
        with open(path, "w") as f:
            for record in _catastrophic_overlap_records():
                f.write(json.dumps(record) + "\n")

        results = analyze_dataset(
            logged_data_path=str(path),
            estimator="calibrated-ips",
            estimator_config={"refuse_unreliable": True},
        )
        assert np.isnan(results.estimates[0])
        assert np.isnan(results.standard_errors[0])
        gates = results.metadata["reliability_gates"]
        assert gates[POLICY]["refused"] is True

    def test_factory_forwards_to_dr_internal_ips(self) -> None:
        from cje.interface.factory import create_estimator

        rng = np.random.default_rng(9)
        dataset = _tilted_dataset(120, rng, lambda s: 0.4 + 1.2 * s)
        calibrated, _ = calibrate_dataset(
            dataset, judge_field="judge_score", oracle_field="oracle_label"
        )
        sampler = PrecomputedSampler(calibrated)

        for name in ["dr-cpo", "mrdr", "tmle"]:
            estimator = create_estimator(
                name, sampler, {"refuse_unreliable": True}, None, False
            )
            assert estimator.ips_estimator.refuse_unreliable is True, name

        estimator = create_estimator(
            "calibrated-ips", sampler, {"refuse_unreliable": True}, None, False
        )
        assert estimator.refuse_unreliable is True

        estimator = create_estimator(
            "raw-ips", sampler, {"refuse_unreliable": True}, None, False
        )
        assert estimator.refuse_unreliable is True

        estimator = create_estimator(
            "stacked-dr", sampler, {"refuse_unreliable": True}, None, False
        )
        assert estimator.refuse_unreliable is True


class TestCLIBestPolicy:
    """Finding #8: the CLI must not crown a gate-flagged policy."""

    @staticmethod
    def _make_results(
        estimates: list,
        policies: list,
        gates: Optional[dict] = None,
        diagnostics: Optional[Any] = None,
    ) -> EstimationResult:
        metadata: dict = {"target_policies": policies}
        if gates is not None:
            metadata["reliability_gates"] = gates
        return EstimationResult(
            estimates=np.array(estimates, dtype=float),
            standard_errors=np.full(len(estimates), 0.05),
            n_samples_used={p: 100 for p in policies},
            method="calibrated_ips",
            influence_functions={},
            diagnostics=diagnostics,
            metadata=metadata,
        )

    def test_reliable_argmax_gets_trophy(self) -> None:
        from cje.interface.cli import best_policy_lines

        results = self._make_results(
            [0.5, 0.7],
            ["base", "good"],
            gates={
                "base": {"flagged": False, "refused": False, "reasons": []},
                "good": {"flagged": False, "refused": False, "reasons": []},
            },
        )
        lines = best_policy_lines(results)
        assert lines == ["🏆 Best policy: good"]

    def test_flagged_argmax_is_demoted(self) -> None:
        from cje.interface.cli import best_policy_lines

        # The verified repro: adversarial 'unhelpful' wins the raw argmax
        # while flagged UNRELIABLE by the refusal gates
        results = self._make_results(
            [0.771, 0.756, 0.763],
            ["unhelpful", "base", "clone"],
            gates={
                "unhelpful": {
                    "flagged": True,
                    "refused": False,
                    "reasons": ["raw_near_zero=90.2%"],
                },
                "base": {"flagged": False, "refused": False, "reasons": []},
                "clone": {"flagged": False, "refused": False, "reasons": []},
            },
        )
        lines = best_policy_lines(results)
        assert any(
            "Best by point estimate: unhelpful" in line and "UNRELIABLE" in line
            for line in lines
        )
        assert not any(line.startswith("🏆 Best policy:") for line in lines)
        # The best RELIABLE policy is named
        assert any("Best reliable policy: clone" in line for line in lines)

    def test_critical_status_also_demotes(self) -> None:
        from cje.interface.cli import best_policy_lines

        diag = _make_ips_diagnostics(
            policies=["bad", "ok"],
            estimates={"bad": 0.9, "ok": 0.6},
            standard_errors={"bad": 0.1, "ok": 0.05},
            ess_per_policy={"bad": 0.05, "ok": 0.6},
            status_per_policy={"bad": Status.CRITICAL, "ok": Status.GOOD},
        )
        results = self._make_results([0.9, 0.6], ["bad", "ok"], diagnostics=diag)
        lines = best_policy_lines(results)
        assert any("UNRELIABLE" in line for line in lines)
        assert any("Best reliable policy: ok" in line for line in lines)

    def test_all_flagged_no_winner(self) -> None:
        from cje.interface.cli import best_policy_lines

        results = self._make_results(
            [0.9, 0.6],
            ["a", "b"],
            gates={
                "a": {"flagged": True, "refused": False, "reasons": ["ESS=5%"]},
                "b": {"flagged": True, "refused": False, "reasons": ["ESS=8%"]},
            },
        )
        lines = best_policy_lines(results)
        assert any("do not pick a winner" in line for line in lines)

    def test_all_nan_estimates(self) -> None:
        from cje.interface.cli import best_policy_lines

        results = self._make_results([float("nan"), float("nan")], ["a", "b"])
        lines = best_policy_lines(results)
        assert any("every policy was refused" in line for line in lines)
