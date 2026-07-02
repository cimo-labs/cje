"""Regression tests for the paper's refusal gates (arXiv:2512.11150).

Covers:
- One canonical ESS ladder: a policy at ESS 0.25 gets the same verdict on
  every surface (gates helper, weight diagnostics, display fallback,
  overlap recommendations).
- hill_tail_index returning NaN (not +inf) on estimation failure, with
  downstream consumers treating NaN as "unknown" rather than the
  healthiest possible tail.
"""

import numpy as np
import pytest

from cje.diagnostics import (
    IPSDiagnostics,
    Status,
    compute_weight_diagnostics,
    create_weight_summary_table,
    ess_status,
    hill_tail_index,
    tail_status,
    worst_status,
)
from cje.diagnostics.gates import (
    ESS_GOOD_THRESHOLD,
    ESS_WARNING_THRESHOLD,
    format_tail_index,
)
from cje.diagnostics.overlap import compute_overlap_metrics


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
