"""Reliability gates respect residual correction.

A residual-transport FAIL and a REFUSE-LEVEL score-support badge describe the
calibration map. In 0.9.1 they demoted every policy whose route was not
``direct_oracle``, including augmented policies whose level is residual-
corrected and does not depend on that map. They now spare an augmented policy
whose correction design check passes (at least 20 effective labelled prompts,
a known-propensity design effective sample size of at least 20 with no
unlabelled row declared at propensity 1, non-constant labelled outcomes, and
judge-score balance under the declared design). Every other policy, including
an augmented one whose check fails, keeps 0.9.1 gating exactly.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pytest

from cje import TransportAuditConfig, analyze_dataset, calibrated_mean_ci
from cje.data.models import EstimationResult
from cje.data.normalization import unit_scale
from cje.diagnostics import DirectDiagnostics, Status
from cje.diagnostics.gates import (
    CORRECTION_DESIGN_BALANCE_T,
    CORRECTION_EXEMPT_MIN_LABELLED_PROMPTS,
    correction_caution,
    correction_design_check,
    level_gate_scope,
)
from cje.interface.cli import best_policy_lines

DELTA = 0.03
DESIGN_WARNING = "Residual correction for policy"
EXEMPT_SUFFIX = "  [corrected: map gates not applied]"
CORRECTED_TRANSPORT = (
    "residual transport FAIL (calibration map only; estimate residual-corrected)"
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _anchor(
    tmp_path: Path, *, seed: int = 0, lo: float = 0.2, hi: float = 0.6, n: int = 40
) -> Path:
    """External calibration file: y = 0.2 + 0.5 s on [lo, hi]."""
    rng = np.random.default_rng(seed)
    scores = np.r_[lo, rng.uniform(lo, hi, n - 2), hi]
    labels = 0.2 + 0.5 * scores + rng.normal(0, 0.025, n)
    path = tmp_path / f"anchor-{seed}-{lo}-{hi}.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "prompt_id": f"fit-{i}",
                    "judge_score": float(s),
                    "oracle_label": float(np.clip(y, 0, 1)),
                }
            )
            + "\n"
            for i, (s, y) in enumerate(zip(scores, labels))
        )
    )
    return path


def _external(path: Path, **overrides: Any) -> Dict[str, Any]:
    """One-call options with an external calibration file (guide workflow)."""
    options: Dict[str, Any] = dict(
        calibration_data_path=str(path),
        combine_oracle_sources=False,
        calibration_judge_scale=(0, 1),
        calibration_oracle_scale=(0, 1),
        fresh_judge_scale=(0, 1),
        fresh_oracle_scale=(0, 1),
        output_scale=(0, 1),
        estimator_config={"inference_method": "cluster_robust"},
    )
    options.update(overrides)
    return options


def _probes(
    rng: np.random.Generator,
    policy: str,
    *,
    shift: Any = 0.0,
    n: int = 60,
    lo: float = 0.2,
    hi: float = 0.6,
) -> List[Dict[str, Any]]:
    """Independent probes on new prompts; ``shift`` may be a function of s."""
    shift_fn = shift if callable(shift) else (lambda s: shift)
    scores = rng.uniform(lo, hi, n)
    return [
        {
            "prompt_id": f"probe-{policy}-{j}",
            "judge_score": float(s),
            "oracle_label": float(
                np.clip(0.2 + 0.5 * s + shift_fn(s) + rng.normal(0, 0.025), 0, 1)
            ),
        }
        for j, s in enumerate(scores)
    ]


def _transport(probes: Dict[str, List[Any]]) -> TransportAuditConfig:
    return TransportAuditConfig(
        probes_by_policy=probes,
        delta_max_by_policy={policy: DELTA for policy in probes},
    )


def _rows(
    scores: Any,
    labelled: Any,
    label_fn: Callable[[float], float],
    *,
    prompt_ids: Optional[Sequence[str]] = None,
    prefix: str = "e",
) -> List[Dict[str, Any]]:
    rows = []
    for i, score in enumerate(scores):
        row: Dict[str, Any] = {
            "prompt_id": prompt_ids[i] if prompt_ids is not None else f"{prefix}{i}",
            "judge_score": float(score),
        }
        if labelled[i]:
            row["oracle_label"] = float(np.clip(label_fn(float(score)), 0, 1))
        rows.append(row)
    return rows


def _even_rank_mask(scores: np.ndarray, k: int, margin: int = 5) -> np.ndarray:
    """Label k rows at evenly spaced score ranks (balanced by construction)."""
    order = np.argsort(scores)
    ranks = np.round(np.linspace(margin, len(scores) - 1 - margin, k)).astype(int)
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[ranks]] = True
    return mask


def _guide(tmp_path: Path, *, seed: int = 42, n_lab: int = 60) -> Dict[str, Any]:
    """The guide's audit-then-correct data: external map, 60 random labels.

    The candidate's outcomes sit 0.15 above the calibration map, so its
    independent probes FAIL; its 60 uniformly sampled labels correct it.
    """
    rng = np.random.default_rng(seed)
    path = _anchor(tmp_path, seed=seed)
    scores = rng.uniform(0.2, 0.6, 200)
    outcomes = {
        "baseline": 0.2 + 0.5 * scores + rng.normal(0, 0.025, 200),
        "candidate": 0.2 + 0.5 * scores + 0.15 + rng.normal(0, 0.025, 200),
    }
    selected = rng.choice(200, size=n_lab, replace=False)
    draws: Dict[str, List[Dict[str, Any]]] = {
        policy: [
            {
                "prompt_id": f"eval-{i}",
                "response_id": f"{policy}:eval-{i}",
                "judge_score": float(s),
            }
            for i, s in enumerate(scores)
        ]
        for policy in outcomes
    }
    for policy, rows in draws.items():
        for i in selected:
            rows[i]["oracle_label"] = float(np.clip(outcomes[policy][i], 0, 1))
    probes = {
        "baseline": _probes(rng, "baseline"),
        "candidate": _probes(rng, "candidate", shift=0.15),
    }
    truth = {policy: float(np.mean(values)) for policy, values in outcomes.items()}
    return {"path": path, "draws": draws, "probes": probes, "truth": truth}


def _s1(tmp_path: Path, *, seed: int = 7) -> Dict[str, Any]:
    """Convenience labels: the candidate's 40 lowest-scored rows (default design).

    The map overstates the candidate's high-score rows by 0.25, so the
    labels on its low-score rows see no residual and the corrected estimate
    is biased upward; independent probes FAIL.
    """
    rng = np.random.default_rng(seed)
    path = _anchor(tmp_path, seed=seed, hi=0.7, n=60)
    n = 300
    s_c = rng.uniform(0.25, 0.65, n)
    s_b = rng.uniform(0.2, 0.6, n)
    y_c = np.clip(0.2 + 0.5 * s_c - 0.25 * (s_c > 0.45) + rng.normal(0, 0.025, n), 0, 1)
    y_b = np.clip(0.2 + 0.5 * s_b + rng.normal(0, 0.025, n), 0, 1)
    lab_c = np.zeros(n, dtype=bool)
    lab_c[np.argsort(s_c)[:40]] = True
    lab_b = np.zeros(n, dtype=bool)
    lab_b[rng.choice(n, 40, replace=False)] = True
    draws = {
        "candidate": _rows(s_c, lab_c, lambda s: 0.0, prefix="c"),
        "baseline": _rows(s_b, lab_b, lambda s: 0.0, prefix="b"),
    }
    for rows, labelled, outcome in (
        (draws["candidate"], lab_c, y_c),
        (draws["baseline"], lab_b, y_b),
    ):
        for i, row in enumerate(rows):
            if labelled[i]:
                row["oracle_label"] = float(outcome[i])
    probes = {
        "candidate": _probes(
            rng, "candidate", shift=lambda s: -0.25 * (s > 0.45), lo=0.25, hi=0.65
        ),
        "baseline": _probes(rng, "baseline"),
    }
    truth = {"candidate": float(y_c.mean()), "baseline": float(y_b.mean())}
    return {"path": path, "draws": draws, "probes": probes, "truth": truth}


def _single_policy_fail(
    tmp_path: Path,
    rows: List[Dict[str, Any]],
    *,
    seed: int = 5,
    probe_shift: float = 0.1,
    probe_label: Optional[float] = None,
    **overrides: Any,
) -> EstimationResult:
    rng = np.random.default_rng(seed)
    probes = _probes(rng, "candidate", shift=probe_shift)
    if probe_label is not None:
        for probe in probes:
            probe["oracle_label"] = probe_label
    return analyze_dataset(
        fresh_draws_data={"candidate": rows},
        transport=_transport({"candidate": probes}),
        **_external(_anchor(tmp_path, seed=seed), **overrides),
    )


def _status(result: EstimationResult, policy: str) -> Status:
    assert result.diagnostics is not None
    assert result.diagnostics.status_per_policy is not None
    return result.diagnostics.status_per_policy[policy]


def _gate(result: EstimationResult, policy: str) -> Dict[str, Any]:
    """The raw gate record (``metadata["reliability_gates"][policy]``)."""
    gate: Dict[str, Any] = result.metadata["reliability_gates"][policy]
    return gate


def _fail_reason(audit: Dict[str, Any]) -> str:
    """0.9.1's transport FAIL reason for an audit record."""
    lo, hi = audit["delta_ci"]
    return (
        f"residual transport FAIL: simultaneous CI [{lo:+.3f}, {hi:+.3f}] "
        f"is outside margin +/-{audit['delta_max']:.3f}"
    )


def _messages(caplog: pytest.LogCaptureFixture, level: int) -> List[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == level]


# ---------------------------------------------------------------------------
# Corrected policies are not demoted by a calibration-map FAIL
# ---------------------------------------------------------------------------


def test_transport_fail_does_not_demote_corrected_policy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    data = _guide(tmp_path)
    with caplog.at_level(logging.INFO):
        result = analyze_dataset(
            fresh_draws_data=data["draws"],
            transport=_transport(data["probes"]),
            **_external(data["path"]),
        )

    checks = result.metadata["correction_checks"]
    assert checks["candidate"]["passed"] is True
    assert checks["baseline"]["passed"] is True
    assert checks["candidate"]["effective_labelled_prompts"] == 60

    audit = result.metadata["transport_audits"]["candidate"]
    assert audit["status"] == "FAIL"
    assert audit["applies_to_current_estimate"] is False
    assert audit["gate_exemption"] == "residual_corrected"
    # The map-level rollup still reports the FAIL.
    assert result.metadata["transport_status"] == "FAIL"

    gate = _gate(result, "candidate")
    assert gate["flagged"] is False
    assert gate["refuse_level_claims"] is False
    assert gate["reasons"] == []
    assert gate["exemption"] == "residual_corrected"
    assert len(gate["notes"]) == 1
    note = gate["notes"][0]
    assert note.startswith("residual transport FAIL (simultaneous CI [")
    assert "describes the calibration map, not this estimate" in note
    assert "60 effective labelled prompts (representative design" in note
    assert result.gates["candidate"].flagged is False

    assert result.diagnostics is not None
    assert _status(result, "candidate") == Status.GOOD
    assert result.diagnostics.overall_status != Status.CRITICAL

    lo, hi = result.confidence_interval()
    index = result.target_policies.index("candidate")
    assert lo[index] <= data["truth"]["candidate"] <= hi[index]
    verdict = result.best_policy()
    assert verdict.name == "candidate"
    assert verdict.runner_up is None
    comparison = result.compare_policies(index, 1 - index)
    assert comparison["gate_flagged"] == []

    text = result.summary()
    candidate_line = next(
        line for line in text.splitlines() if line.startswith("  candidate")
    )
    assert candidate_line.endswith(EXEMPT_SUFFIX)
    assert f"    note: {note}" in text.splitlines()
    assert f"Limitations: {CORRECTED_TRANSPORT}" in text
    assert best_policy_lines(result) == [
        "Best by point estimate: candidate",
        f"Limitations: {CORRECTED_TRANSPORT}",
    ]
    assert not any(
        "best_policy()" in message for message in _messages(caplog, logging.WARNING)
    )


def test_convenience_labels_keep_gate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """S1: labels on the candidate's lowest-scored rows under the default design.

    Exempting this policy would report 0.43 against a truth of 0.30 as
    unflagged and crown it; the design check keeps 0.9.1 gating and warns.
    """
    data = _s1(tmp_path)
    with caplog.at_level(logging.INFO):
        result = analyze_dataset(
            fresh_draws_data=data["draws"],
            transport=_transport(data["probes"]),
            **_external(data["path"]),
        )

    check = result.metadata["correction_checks"]["candidate"]
    assert check["failed"] == ["labelled_rows_unbalanced"]
    assert check["balance_t"]["judge_level"] < -CORRECTION_DESIGN_BALANCE_T
    assert result.metadata["correction_checks"]["baseline"]["passed"] is True

    audit = result.metadata["transport_audits"]["candidate"]
    assert audit["status"] == "FAIL"
    assert audit["applies_to_current_estimate"] is True
    assert "gate_exemption" not in audit
    # 0.9.1's gate record, byte for byte (no notes or exemption keys).
    assert _gate(result, "candidate") == {
        "flagged": True,
        "refused": False,
        "refuse_level_claims": True,
        "reasons": [_fail_reason(audit)],
    }
    assert result.diagnostics is not None
    assert _status(result, "candidate") == Status.CRITICAL
    assert "    note: " not in result.summary()

    verdict = result.best_policy()
    assert verdict.name == "baseline"
    assert verdict.runner_up == "candidate"
    design_warnings = [
        m for m in _messages(caplog, logging.WARNING) if m.startswith(DESIGN_WARNING)
    ]
    assert len(design_warnings) == 1
    assert design_warnings[0].startswith(
        "Residual correction for policy 'candidate': its labelled rows do not look "
        "like a representative sample of its evaluation rows (judge_level t=-"
    )
    assert design_warnings[0].endswith(
        "Calibration-map gates still apply to this policy."
    )


def test_design_check_warns_without_changing_outputs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed check without any map finding only logs a warning."""
    data = _s1(tmp_path)
    with caplog.at_level(logging.INFO):
        result = analyze_dataset(
            fresh_draws_data={"candidate": data["draws"]["candidate"]},
            **_external(data["path"]),
        )

    assert result.metadata["correction_checks"]["candidate"]["failed"] == [
        "labelled_rows_unbalanced"
    ]
    assert _gate(result, "candidate") == {
        "flagged": False,
        "refused": False,
        "refuse_level_claims": False,
        "reasons": [],
    }
    assert result.diagnostics is not None
    assert _status(result, "candidate") == Status.GOOD
    assert result.best_policy().name == "candidate"
    assert (
        sum(m.startswith(DESIGN_WARNING) for m in _messages(caplog, logging.WARNING))
        == 1
    )

    text = result.summary()
    assert "note:" not in text
    assert EXEMPT_SUFFIX not in text
    assert best_policy_lines(result) == [
        "Best by point estimate: candidate",
        "Limitations: residual transport NOT_CHECKED",
    ]


# ---------------------------------------------------------------------------
# What the design check requires
# ---------------------------------------------------------------------------


def _floor_rows(rng: np.random.Generator, layout: str) -> List[Dict[str, Any]]:
    """Rows with 0.1 above the map and labels balanced on the score."""

    def outcome(s: float) -> float:
        return float(0.2 + 0.5 * s + 0.1 + rng.normal(0, 0.02))

    if layout in ("19", "20"):
        scores = rng.uniform(0.2, 0.6, 300)
        return _rows(scores, _even_rank_mask(scores, int(layout)), outcome)
    if layout == "20_rows_10_prompts":
        prompts = [f"p{j}" for j in range(150) for _ in range(2)]
        scores = rng.uniform(0.2, 0.6, 300)
        labelled = np.zeros(300, dtype=bool)
        for j in np.round(np.linspace(3, 146, 10)).astype(int):
            labelled[2 * j : 2 * j + 2] = True
        return _rows(scores, labelled, outcome, prompt_ids=prompts)
    assert layout == "one_prompt_30_draws"
    prompts = ["big"] * 30 + [f"p{j}" for j in range(270)]
    scores = rng.uniform(0.2, 0.6, 300)
    labelled = np.zeros(300, dtype=bool)
    labelled[:30] = True
    labelled[30 + np.round(np.linspace(3, 266, 19)).astype(int)] = True
    return _rows(scores, labelled, outcome, prompt_ids=prompts)


@pytest.mark.parametrize(
    "layout, exempt, effective",
    [
        ("19", False, 19.0),
        ("20", True, 20.0),
        ("20_rows_10_prompts", False, 10.0),
        ("one_prompt_30_draws", False, 2.6),
    ],
)
def test_below_effective_floor_keeps_gate(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    layout: str,
    exempt: bool,
    effective: float,
) -> None:
    rows = _floor_rows(np.random.default_rng(13), layout)
    with caplog.at_level(logging.WARNING):
        result = _single_policy_fail(tmp_path, rows)

    check = result.metadata["correction_checks"]["candidate"]
    assert check["effective_labelled_prompts"] == pytest.approx(effective, abs=0.05)
    audit = result.metadata["transport_audits"]["candidate"]
    assert audit["status"] == "FAIL"
    assert result.diagnostics is not None
    status = _status(result, "candidate")
    if exempt:
        assert check["passed"] is True
        assert _gate(result, "candidate")["flagged"] is False
        assert _gate(result, "candidate")["exemption"] == "residual_corrected"
        assert status == Status.GOOD
        return
    assert check["failed"] == ["too_few_effective_labelled_prompts"]
    assert _gate(result, "candidate") == {
        "flagged": True,
        "refused": False,
        "refuse_level_claims": True,
        "reasons": [_fail_reason(audit)],
    }
    assert status == Status.CRITICAL
    # Too few labels shows in the interval: no design warning.
    assert not any(
        m.startswith(DESIGN_WARNING) for m in _messages(caplog, logging.WARNING)
    )


def test_constant_labelled_outcomes_keep_gate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A binary oracle whose labelled outcomes are all 1 cannot lift the gate."""
    scores = np.random.default_rng(17).uniform(0.2, 0.6, 300)
    rows = _rows(scores, _even_rank_mask(scores, 40), lambda s: 1.0)
    with caplog.at_level(logging.INFO):
        result = _single_policy_fail(tmp_path, rows, probe_label=1.0)

    check = result.metadata["correction_checks"]["candidate"]
    assert check["failed"] == ["labelled_outcomes_constant"]
    gate = _gate(result, "candidate")
    assert gate["flagged"] is True
    assert "exemption" not in gate
    # The existing provisional-interval warning covers this case.
    warnings_logged = _messages(caplog, logging.WARNING)
    assert any("Every labelled outcome is identical" in m for m in warnings_logged)
    assert not any(m.startswith(DESIGN_WARNING) for m in warnings_logged)


# ---------------------------------------------------------------------------
# Every other route keeps 0.9.1 behaviour exactly
# ---------------------------------------------------------------------------


def _plug_in_data(tmp_path: Path, *, labelled: bool) -> Dict[str, Any]:
    rng = np.random.default_rng(23)
    n = 200
    s_b = rng.uniform(0.2, 0.5, n)
    s_c = rng.uniform(0.3, 0.6, n)
    mask = np.zeros(n, dtype=bool)
    if labelled:
        mask[rng.choice(n, 60, replace=False)] = True
    draws = {
        "baseline": _rows(s_b, mask, lambda s: 0.2 + 0.5 * s, prefix="b"),
        "candidate": _rows(s_c, mask, lambda s: 0.2 + 0.5 * s + 0.1, prefix="c"),
    }
    probes = {"candidate": _probes(rng, "candidate", shift=0.1)}
    return {"path": _anchor(tmp_path, seed=23), "draws": draws, "probes": probes}


@pytest.mark.parametrize(
    "case, route",
    [
        ("no_labels", "plug_in"),
        ("augmentation_off", "plug_in"),
        ("targeted_unknown", "plug_in_targeted_unknown"),
    ],
)
def test_plug_in_routes_unchanged(tmp_path: Path, case: str, route: str) -> None:
    data = _plug_in_data(tmp_path, labelled=case != "no_labels")
    overrides: Dict[str, Any] = {}
    if case == "augmentation_off":
        overrides["estimator_config"] = {
            "inference_method": "cluster_robust",
            "use_augmented_estimator": False,
        }
    if case == "targeted_unknown":
        overrides["label_design"] = "targeted_unknown"
    result = analyze_dataset(
        fresh_draws_data=data["draws"],
        transport=_transport(data["probes"]),
        **_external(data["path"], **overrides),
    )

    assert result.metadata["point_estimator"]["routes"] == [route, route]
    assert "correction_checks" not in result.metadata
    audit = result.metadata["transport_audits"]["candidate"]
    assert audit["status"] == "FAIL"
    assert audit["applies_to_current_estimate"] is True
    assert "gate_exemption" not in audit
    assert all(
        "gate_exemption" not in card
        for card in result.metadata["boundary_cards"].values()
    )
    fail_reason = _fail_reason(audit)
    # 0.9.1 gate dict, byte for byte (no notes or exemption keys).
    assert _gate(result, "candidate") == {
        "flagged": True,
        "refused": False,
        "refuse_level_claims": True,
        "reasons": [fail_reason],
    }
    assert _gate(result, "baseline") == {
        "flagged": False,
        "refused": False,
        "refuse_level_claims": False,
        "reasons": [],
    }
    assert result.diagnostics is not None
    assert _status(result, "candidate") == Status.CRITICAL
    verdict = result.best_policy()
    assert (verdict.name, verdict.runner_up) == ("baseline", "candidate")
    comparison = result.compare_policies(0, 1)
    assert comparison["gate_flagged"] == ["candidate"]

    lo, hi = result.confidence_interval()
    est = result.estimates
    demoted = (
        "Best reliable policy: baseline — raw argmax candidate was flagged "
        f"({fail_reason}; diagnostics status CRITICAL); pass reliable_only=False "
        "for the raw argmax"
    )
    assert result.summary() == "\n".join(
        [
            f"CJE Estimation Results (method: {result.method})",
            f"  baseline   {est[0]:.3f}  95% CI [{lo[0]:.3f}, {hi[0]:.3f}]",
            f"  candidate  {est[1]:.3f}  95% CI [{lo[1]:.3f}, {hi[1]:.3f}]"
            "  [gate: FLAGGED]",
            "Best by point estimate: candidate",
            "Limitations: flagged by the reliability gates; residual transport FAIL",
            demoted,
            "Status: critical",
        ]
    )
    assert best_policy_lines(result) == [
        "Best by point estimate: candidate",
        "Limitations: reliability gates flagged this policy; residual transport FAIL",
        demoted,
    ]


def test_transport_fail_does_not_gate_direct_oracle(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A complete-oracle mean ignores a map FAIL, exactly as in 0.9.1."""
    rng = np.random.default_rng(79)
    n = 200
    scores = rng.uniform(0.2, 0.6, n)
    draws = {
        "oracle": _rows(
            scores,
            np.ones(n, dtype=bool),
            lambda s: 0.2 + 0.5 * s + 0.1 + rng.normal(0, 0.02),
            prefix="o",
        ),
        "baseline": _rows(
            rng.uniform(0.2, 0.6, n), np.zeros(n, dtype=bool), lambda s: 0.0
        ),
    }
    probes = {"oracle": _probes(rng, "oracle", shift=0.1)}
    with caplog.at_level(logging.WARNING):
        result = analyze_dataset(
            fresh_draws_data=draws,
            transport=_transport(probes),
            **_external(_anchor(tmp_path, seed=79)),
        )

    index = result.target_policies.index("oracle")
    assert result.metadata["point_estimator"]["routes"][index] == "direct_oracle"
    assert "correction_checks" not in result.metadata
    audit = result.metadata["transport_audits"]["oracle"]
    assert audit["status"] == "FAIL"
    assert audit["applies_to_current_estimate"] is False
    assert "gate_exemption" not in audit
    gates = result.metadata.get("reliability_gates") or {}
    assert not (gates.get("oracle") or {}).get("flagged", False)
    assert "exemption" not in (gates.get("oracle") or {})
    assert "oracle" not in result._unreliable_policies()
    assert _status(result, "oracle") == Status.GOOD
    verdict = result.best_policy()
    assert verdict.name == "oracle"
    assert verdict.runner_up is None
    assert not any(
        "best_policy()" in message for message in _messages(caplog, logging.WARNING)
    )


# ---------------------------------------------------------------------------
# Known-propensity designs
# ---------------------------------------------------------------------------


def test_known_propensity_correction_is_exempt(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 300
    scores = rng.uniform(0.2, 0.6, n)
    propensities = 0.2 + 0.2 * rng.uniform(size=n)
    observed = rng.uniform(size=n) < propensities
    assert observed.sum() >= 60
    rows = _rows(scores, observed, lambda s: 0.2 + 0.5 * s + 0.1 + rng.normal(0, 0.02))
    result = _single_policy_fail(
        tmp_path,
        rows,
        label_design="known_propensity",
        label_propensities={"candidate": propensities.tolist()},
    )

    check = result.metadata["correction_checks"]["candidate"]
    assert check["design"] == "known_propensity"
    assert check["passed"] is True
    assert check["design_effective_n"] >= CORRECTION_EXEMPT_MIN_LABELLED_PROMPTS
    assert set(check["balance_t"]) == {"label_count", "judge_level", "judge_spread"}
    assert result.metadata["transport_audits"]["candidate"]["status"] == "FAIL"
    gate = _gate(result, "candidate")
    assert gate["flagged"] is False
    assert gate["exemption"] == "residual_corrected"
    assert "(known-propensity design;" in gate["notes"][0]


def _rare_stratum_rows(rng: np.random.Generator) -> Dict[str, Any]:
    """kp_rare layout: a 10% stratum at p=0.005 (unsampled), p=0.08 elsewhere."""
    n = 400
    scores = rng.uniform(0.2, 0.6, n)
    propensities = np.where(np.arange(n) < 40, 0.005, 0.08)
    observed = np.zeros(n, dtype=bool)
    rest = 40 + np.argsort(scores[40:])
    observed[rest[np.round(np.linspace(3, 356, 29)).astype(int)]] = True
    rows = _rows(scores, observed, lambda s: 0.2 + 0.5 * s + 0.1 + rng.normal(0, 0.02))
    return {"rows": rows, "propensities": propensities.tolist()}


def test_known_propensity_low_design_n_keeps_gate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    data = _rare_stratum_rows(np.random.default_rng(29))
    options: Dict[str, Any] = dict(
        label_design="known_propensity",
        label_propensities={"candidate": data["propensities"]},
    )
    result = _single_policy_fail(tmp_path, data["rows"], **options)

    check = result.metadata["correction_checks"]["candidate"]
    assert check["labelled_prompts"] == 29
    assert check["design_effective_n"] == pytest.approx(12.8)
    assert check["failed"] == ["low_design_effective_n"]
    audit = result.metadata["transport_audits"]["candidate"]
    assert _gate(result, "candidate") == {
        "flagged": True,
        "refused": False,
        "refuse_level_claims": True,
        "reasons": [_fail_reason(audit)],
    }

    # The design warning is logged even without any map finding.
    caplog.clear()
    with caplog.at_level(logging.INFO):
        no_audit = analyze_dataset(
            fresh_draws_data={"candidate": data["rows"]},
            **_external(_anchor(tmp_path, seed=5), **options),
        )
    assert no_audit.gates["candidate"].flagged is False
    warnings_logged = [
        m for m in _messages(caplog, logging.WARNING) if m.startswith(DESIGN_WARNING)
    ]
    assert len(warnings_logged) == 1
    assert "design effective sample size of 12.8 (< 20)" in warnings_logged[0]


def test_known_propensity_misdeclared_is_caught(tmp_path: Path) -> None:
    """Labels taken from the top of the score range, declared as p = 0.2."""
    rng = np.random.default_rng(31)
    n = 300
    scores = rng.uniform(0.2, 0.6, n)
    observed = np.zeros(n, dtype=bool)
    observed[np.argsort(scores)[-60:]] = True
    rows = _rows(scores, observed, lambda s: 0.2 + 0.5 * s + 0.1 + rng.normal(0, 0.02))
    result = _single_policy_fail(
        tmp_path,
        rows,
        label_design="known_propensity",
        label_propensities={"candidate": [0.2] * n},
    )

    check = result.metadata["correction_checks"]["candidate"]
    assert check["failed"] == ["labelled_rows_unbalanced"]
    assert check["balance_t"]["label_count"] == pytest.approx(0.0, abs=1e-9)
    assert check["balance_t"]["judge_level"] > CORRECTION_DESIGN_BALANCE_T
    assert result.gates["candidate"].flagged is True


def test_known_propensity_certain_unlabelled_rows_keep_gate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Propensity 1 on every row ("no reweighting") with 40 random labels.

    Propensity 1 means "always labelled", so the 260 unlabelled rows
    contradict the design, and the correction stays near the map's level.
    """
    rng = np.random.default_rng(67)
    n = 300
    scores = rng.uniform(0.2, 0.6, n)
    observed = np.zeros(n, dtype=bool)
    observed[rng.choice(n, 40, replace=False)] = True
    rows = _rows(scores, observed, lambda s: 0.2 + 0.5 * s + 0.1 + rng.normal(0, 0.02))
    with caplog.at_level(logging.INFO):
        result = _single_policy_fail(
            tmp_path,
            rows,
            label_design="known_propensity",
            label_propensities={"candidate": [1.0] * n},
        )

    check = result.metadata["correction_checks"]["candidate"]
    assert check["design_effective_n"] == n
    assert check["effective_labelled_prompts"] == 40
    assert check["unlabelled_certain_rows"] == 260
    assert check["failed"] == [
        "unlabelled_rows_declared_certain",
        "labelled_rows_unbalanced",
    ]
    assert check["balance_t"]["label_count"] < -CORRECTION_DESIGN_BALANCE_T

    audit = result.metadata["transport_audits"]["candidate"]
    assert audit["status"] == "FAIL"
    assert audit["applies_to_current_estimate"] is True
    assert "gate_exemption" not in audit
    assert _gate(result, "candidate") == {
        "flagged": True,
        "refused": False,
        "refuse_level_claims": True,
        "reasons": [_fail_reason(audit)],
    }
    assert _status(result, "candidate") == Status.CRITICAL
    design_warnings = [
        m for m in _messages(caplog, logging.WARNING) if m.startswith(DESIGN_WARNING)
    ]
    assert len(design_warnings) == 1
    assert design_warnings[0].startswith(
        "Residual correction for policy 'candidate': 260 unlabelled rows have a "
        "declared label propensity of 1 (certain to be labelled), which the "
        "declared design rules out"
    )

    # Propensity 1 on the labelled rows only is a valid declaration.
    unit = correction_design_check(
        scores,
        observed,
        np.arange(n),
        "known_propensity",
        propensities=np.where(observed, 1.0, 0.1),
    )
    assert unit["unlabelled_certain_rows"] == 0
    assert "unlabelled_rows_declared_certain" not in unit["failed"]


def _whole_prompt_rows(
    seed: int, *, draws: int = 4, propensity: float = 0.25, n_prompts: int = 150
) -> List[Dict[str, Any]]:
    """Known-propensity design that labels every draw of a sampled prompt."""
    rng = np.random.default_rng(seed)
    prompts = [f"q{j}" for j in range(n_prompts) for _ in range(draws)]
    scores = np.clip(
        np.repeat(rng.uniform(0.2, 0.6, n_prompts), draws)
        + rng.normal(0, 0.03, n_prompts * draws),
        0.2,
        0.6,
    )
    sampled = np.repeat(rng.uniform(size=n_prompts) < propensity, draws)
    rows = _rows(
        scores,
        sampled,
        lambda s: 0.2 + 0.5 * s + 0.1 + rng.normal(0, 0.02),
        prompt_ids=prompts,
    )
    for i, row in enumerate(rows):
        row["draw_idx"] = i % draws
    return rows


def test_known_propensity_whole_prompt_labels_are_exempt(tmp_path: Path) -> None:
    """Prompt-level sampling, 4 draws each, correctly declared p = 0.25.

    A row-level Poisson variance gave judge_level t = -3.09 here (34 effective
    labelled prompts) and kept the gate; the cluster variance does not.
    """
    rows = _whole_prompt_rows(9)
    rng = np.random.default_rng(9)
    probes = _probes(rng, "candidate", shift=0.1)
    result = analyze_dataset(
        fresh_draws_data={"candidate": rows},
        transport=_transport({"candidate": probes}),
        **_external(
            _anchor(tmp_path, seed=5),
            label_design="known_propensity",
            label_propensities={"candidate": [0.25] * len(rows)},
        ),
    )

    check = result.metadata["correction_checks"]["candidate"]
    assert check["effective_labelled_prompts"] == pytest.approx(34.0)
    assert check["passed"] is True
    assert max(abs(t) for t in check["balance_t"].values()) < 2.0
    assert result.metadata["transport_audits"]["candidate"]["status"] == "FAIL"
    assert _gate(result, "candidate")["flagged"] is False
    assert _gate(result, "candidate")["exemption"] == "residual_corrected"


@pytest.mark.parametrize("prompt_level", [True, False], ids=["prompts", "rows"])
@pytest.mark.parametrize("draws", [2, 4])
def test_known_propensity_balance_false_fail_rate(
    draws: int, prompt_level: bool
) -> None:
    """Correct known propensities rarely fail balance, whole prompts or rows.

    A row-level Poisson variance failed 7% (2 draws) and 27% (4 draws) of
    whole-prompt designs; the cluster variance stays near 1%.
    """
    n_prompts, propensity = 150, 0.25
    clusters = np.repeat(np.arange(n_prompts), draws)
    failures = 0
    for seed in range(200):
        rng = np.random.default_rng([seed, draws])
        scores = np.clip(
            0.7 * np.repeat(rng.uniform(size=n_prompts), draws)
            + 0.3 * rng.uniform(size=n_prompts * draws),
            0,
            1,
        )
        if prompt_level:
            observed = np.repeat(rng.uniform(size=n_prompts) < propensity, draws)
        else:
            observed = rng.uniform(size=n_prompts * draws) < propensity
        check = correction_design_check(
            scores,
            observed,
            clusters,
            "known_propensity",
            propensities=np.full(len(scores), propensity),
        )
        failures += "labelled_rows_unbalanced" in check["failed"]
    assert failures <= 6


@pytest.mark.slow
def test_design_check_false_positive_rate() -> None:
    """Random representative labels at the floor rarely fail balance."""
    failures = 0
    for seed in range(400):
        rng = np.random.default_rng(seed)
        scores = rng.uniform(size=200)
        observed = np.zeros(200, dtype=bool)
        observed[
            rng.choice(200, CORRECTION_EXEMPT_MIN_LABELLED_PROMPTS, replace=False)
        ] = True
        check = correction_design_check(
            scores, observed, np.arange(200), "representative"
        )
        failures += "labelled_rows_unbalanced" in check["failed"]
        assert check == correction_design_check(
            scores, observed, np.arange(200), "representative"
        )
    assert failures <= 6


def test_design_check_is_json_safe_and_deterministic() -> None:
    scores = np.linspace(0.2, 0.6, 30)
    observed = np.r_[np.ones(20), np.zeros(10)].astype(bool)
    check = correction_design_check(
        scores, observed, [f"p{i}" for i in range(30)], "representative"
    )
    assert json.loads(json.dumps(check)) == check
    assert check == correction_design_check(
        scores, observed, [f"p{i}" for i in range(30)], "representative"
    )
    # The existing transport fixture: labels on the 20 lowest of 30 rows.
    assert check["balance_t"]["judge_level"] == pytest.approx(-4.05, abs=0.01)
    assert check["failed"] == ["labelled_rows_unbalanced"]
    with pytest.raises(ValueError, match="label_design"):
        correction_design_check(scores, observed, np.arange(30), "targeted_unknown")


@pytest.mark.parametrize(
    "case", ["two_point_0.3_0.7", "two_point_0.1_0.9", "constant", "constant_kp"]
)
def test_design_check_float_noise_guard(case: str) -> None:
    """Centred values that are float noise give t == 0, not a noise ratio.

    A two-point symmetric judge has the same |s - median| on every row, and a
    constant judge has no deviation at all; without the guard their t values
    are ratios of rounding noise (judge_spread t = -0.77 for the first case).
    """
    rng = np.random.default_rng(71)
    n = 300
    if case.startswith("two_point"):
        lo, hi = (float(v) for v in case.split("_")[2:])
        scores = np.r_[np.full(n // 2, lo), np.full(n // 2, hi)]
        variables = ["judge_spread"]
    else:
        scores = np.full(n, 0.1)
        variables = ["judge_level", "judge_spread"]
    observed = np.zeros(n, dtype=bool)
    observed[rng.choice(n, 37, replace=False)] = True
    options: Dict[str, Any] = {}
    design = "representative"
    if case == "constant_kp":
        design = "known_propensity"
        options["propensities"] = np.full(n, 0.13)
    check = correction_design_check(scores, observed, np.arange(n), design, **options)
    for variable in variables:
        assert check["balance_t"][variable] == 0.0
    assert "labelled_rows_unbalanced" not in check["failed"]
    assert check["passed"] is True


def test_design_check_without_design_variance_fails_balance() -> None:
    """A deviation whose design variance overflows is unbalanced, t None."""
    rng = np.random.default_rng(73)
    n = 300
    scores = rng.uniform(0.2, 0.6, n)
    observed = np.zeros(n, dtype=bool)
    observed[rng.choice(n, 40, replace=False)] = True
    propensities = np.full(n, 0.2)
    propensities[np.flatnonzero(observed)[0]] = 1e-200
    check = correction_design_check(
        scores, observed, np.arange(n), "known_propensity", propensities=propensities
    )
    assert check["balance_t"]["label_count"] is None
    assert "labelled_rows_unbalanced" in check["failed"]
    assert check["effective_labelled_prompts"] == 1.0
    # Strict JSON (no NaN or Infinity).
    assert json.loads(json.dumps(check, allow_nan=False)) == check
    caution = correction_caution("candidate", check)
    assert caution is not None
    assert "label_count t undefined (no design variance)" in caution


def test_level_gate_scope() -> None:
    passed = {"passed": True, "failed": []}
    failed = {"passed": False, "failed": ["labelled_rows_unbalanced"]}
    assert level_gate_scope("direct_oracle", None) == (False, "direct_oracle")
    assert level_gate_scope("augmented", passed) == (False, "residual_corrected")
    assert level_gate_scope("augmented", failed) == (True, None)
    assert level_gate_scope("augmented", None) == (True, None)
    assert level_gate_scope("plug_in", passed) == (True, None)
    assert level_gate_scope("plug_in_targeted_unknown", passed) == (True, None)
    assert level_gate_scope("no_data", None) == (True, None)
    assert level_gate_scope(None, passed) == (True, None)


# ---------------------------------------------------------------------------
# Score-support badge (boundary card)
# ---------------------------------------------------------------------------


def _badge_data(labels_on: str) -> Dict[str, List[Dict[str, Any]]]:
    """Fresh-only: policy 'corrected' labels 30 of 300 rows, 'uncorrected' none."""
    rng = np.random.default_rng(37)
    n = 300
    s_a = rng.uniform(0, 1, n)
    s_b = rng.uniform(0, 1, n)
    if labels_on == "even":
        # Evenly spaced score ranks 10..289: >= 5% of rows fall outside the
        # labelled (calibration) range, and the labels are balanced.
        order = np.argsort(s_a)
        mask = np.zeros(n, dtype=bool)
        mask[order[np.round(np.linspace(10, 289, 30)).astype(int)]] = True
    else:
        mask = np.zeros(n, dtype=bool)
        mask[np.argsort(s_a)[-30:]] = True
    return {
        "corrected": _rows(
            s_a, mask, lambda s: 0.1 + 0.6 * s + rng.normal(0, 0.02), prefix="a"
        ),
        "uncorrected": _rows(s_b, np.zeros(n, dtype=bool), lambda s: 0.0, prefix="b"),
    }


def test_refuse_level_badge_exempt_for_corrected_policy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        result = analyze_dataset(
            fresh_draws_data=_badge_data("even"),
            estimator_config={"inference_method": "cluster_robust"},
        )
    diagnostics = result.diagnostics
    assert diagnostics is not None and diagnostics.boundary_cards is not None

    card = diagnostics.boundary_cards["corrected"]
    assert card["status"] == "REFUSE-LEVEL"
    assert card["out_of_range"] >= 0.05
    assert card["applies_to_current_estimate"] is False
    assert card["gate_exemption"] == "residual_corrected"
    gate = _gate(result, "corrected")
    assert gate["flagged"] is False
    assert gate["reasons"] == []
    assert gate["exemption"] == "residual_corrected"
    assert gate["notes"][0].startswith(
        f"boundary REFUSE-LEVEL ({card['out_of_range']:.1%} of judge scores outside "
        "the oracle calibration range) describes the calibration map"
    )
    assert _status(result, "corrected") == Status.GOOD
    assert "corrected" not in diagnostics.refuse_level_policies
    assert not any("for corrected:" in issue for issue in diagnostics.validate())
    warnings_logged = _messages(caplog, logging.WARNING)
    assert not any("REFUSE-LEVEL for policy 'corrected'" in m for m in warnings_logged)
    assert any(
        m.startswith("boundary REFUSE-LEVEL") for m in _messages(caplog, logging.INFO)
    )
    corrected_line = next(
        line for line in result.summary().splitlines() if line.startswith("  corrected")
    )
    assert corrected_line.endswith(EXEMPT_SUFFIX)

    # The plug-in policy keeps 0.9.1 behaviour.
    other = diagnostics.boundary_cards["uncorrected"]
    assert other["status"] == "REFUSE-LEVEL"
    assert other["applies_to_current_estimate"] is True
    assert "gate_exemption" not in other
    assert _gate(result, "uncorrected") == {
        "flagged": True,
        "refused": False,
        "refuse_level_claims": True,
        "reasons": [
            f"boundary: {other['out_of_range']:.1%} of judge scores outside the "
            "oracle calibration range"
        ],
    }
    assert _status(result, "uncorrected") == Status.CRITICAL
    assert diagnostics.refuse_level_policies == ["uncorrected"]
    assert any("REFUSE-LEVEL for policy 'uncorrected'" in m for m in warnings_logged)


def test_refuse_level_with_unbalanced_labels_keeps_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        result = analyze_dataset(
            fresh_draws_data=_badge_data("top"),
            estimator_config={"inference_method": "cluster_robust"},
        )
    diagnostics = result.diagnostics
    assert diagnostics is not None and diagnostics.boundary_cards is not None
    card = diagnostics.boundary_cards["corrected"]
    assert card["status"] == "REFUSE-LEVEL"
    assert card["applies_to_current_estimate"] is True
    assert "gate_exemption" not in card
    assert _status(result, "corrected") == Status.CRITICAL
    assert _gate(result, "corrected") == {
        "flagged": True,
        "refused": False,
        "refuse_level_claims": True,
        "reasons": [
            f"boundary: {card['out_of_range']:.1%} of judge scores outside the "
            "oracle calibration range"
        ],
    }
    assert "corrected" in diagnostics.refuse_level_policies
    warnings_logged = _messages(caplog, logging.WARNING)
    assert any("REFUSE-LEVEL for policy 'corrected'" in m for m in warnings_logged)
    assert any(
        m.startswith("Residual correction for policy 'corrected'")
        for m in warnings_logged
    )


def test_refuse_level_listing_skips_only_exempt_cards(tmp_path: Path) -> None:
    """Exempt cards leave the REFUSE-LEVEL listing; direct-oracle ones stay."""

    def diagnostics(card: Dict[str, Any]) -> DirectDiagnostics:
        return DirectDiagnostics(
            estimator_type="Direct",
            method="calibrated_direct",
            n_samples_total=10,
            n_samples_valid=10,
            policies=["pi"],
            estimates={"pi": 0.5},
            standard_errors={"pi": 0.01},
            n_samples_used={"pi": 10},
            boundary_cards={
                "pi": {"status": "REFUSE-LEVEL", "out_of_range": 0.12, **card}
            },
        )

    exempt = diagnostics(
        {"applies_to_current_estimate": False, "gate_exemption": "residual_corrected"}
    )
    assert exempt.refuse_level_policies == []
    assert not any("REFUSE-LEVEL" in issue for issue in exempt.validate())
    for card in ({"applies_to_current_estimate": False}, {}):
        unchanged = diagnostics(card)
        assert unchanged.refuse_level_policies == ["pi"]
        assert any("REFUSE-LEVEL for pi" in issue for issue in unchanged.validate())

    # End to end: a fully labelled policy outside a narrow external map keeps
    # 0.9.1's descriptive card (listed, not applied, no new keys).
    rng = np.random.default_rng(41)
    scores = rng.uniform(0.2, 0.6, 50)
    rows = _rows(scores, np.ones(50, dtype=bool), lambda s: 0.2 + 0.5 * s)
    result = analyze_dataset(
        fresh_draws_data={"oracle": rows},
        **_external(_anchor(tmp_path, seed=41, lo=0.4, hi=0.5)),
    )
    assert result.diagnostics is not None
    assert result.diagnostics.boundary_cards is not None
    card = result.diagnostics.boundary_cards["oracle"]
    assert card["status"] == "REFUSE-LEVEL"
    assert card["applies_to_current_estimate"] is False
    assert "gate_exemption" not in card
    assert result.diagnostics.refuse_level_policies == ["oracle"]
    assert "correction_checks" not in result.metadata
    assert _gate(result, "oracle") == {
        "flagged": False,
        "refused": False,
        "refuse_level_claims": False,
        "reasons": [],
    }


# ---------------------------------------------------------------------------
# calibrated_mean_ci
# ---------------------------------------------------------------------------


def _array_case(case: str) -> Dict[str, Any]:
    rng = np.random.default_rng(43)
    scores = rng.uniform(0, 1, 400)
    outcomes = np.clip(0.1 + 0.6 * scores + rng.normal(0, 0.03, 400), 0, 1)
    mask = np.zeros(400, dtype=bool)
    if case == "random_30":
        mask[rng.choice(400, 30, replace=False)] = True
    elif case == "top_30":
        mask[np.argsort(scores)[-30:]] = True
    else:
        mask[rng.choice(400, 12, replace=False)] = True
    return {"scores": scores, "labels": np.where(mask, outcomes, np.nan), "mask": mask}


@pytest.mark.parametrize(
    "case, applies", [("random_30", False), ("top_30", True), ("few_12", True)]
)
def test_calibrated_mean_ci_badge_scope(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    case: str,
    applies: bool,
) -> None:
    data = _array_case(case)
    with caplog.at_level(logging.INFO):
        result = calibrated_mean_ci(data["scores"], data["labels"], data["mask"])

    card = result.diagnostics["boundary_card"]
    check = result.diagnostics["correction_check"]
    assert card["applies_to_current_estimate"] is applies
    assert check["passed"] is (not applies)
    warnings_logged = _messages(caplog, logging.WARNING)
    refuse_warned = any("REFUSE-LEVEL" in m for m in warnings_logged)
    design_warned = any(m.startswith(DESIGN_WARNING) for m in warnings_logged)
    if case == "random_30":
        assert card["status"] == "REFUSE-LEVEL" and card["out_of_range"] >= 0.05
        assert card["gate_exemption"] == "residual_corrected"
        assert not refuse_warned and not design_warned
        assert any(
            m.startswith("boundary REFUSE-LEVEL")
            for m in _messages(caplog, logging.INFO)
        )
    elif case == "top_30":
        assert card["status"] == "REFUSE-LEVEL"
        assert "gate_exemption" not in card
        assert refuse_warned and design_warned
        assert "labelled_rows_unbalanced" in check["failed"]
    else:
        assert "gate_exemption" not in card
        assert check["failed"] == ["too_few_effective_labelled_prompts"]
        assert not design_warned

    # The check never moves the numbers: forcing the opposite verdict leaves
    # the estimate and interval intact.
    flipped = dict(check, passed=applies, failed=[] if applies else ["forced"])
    monkeypatch.setattr(
        "cje.array_api.correction_design_check", lambda *a, **k: flipped
    )
    forced = calibrated_mean_ci(data["scores"], data["labels"], data["mask"])
    assert forced.diagnostics["boundary_card"]["applies_to_current_estimate"] is (
        not applies
    )
    assert forced.estimate == result.estimate
    assert forced.ci == result.ci
    assert forced.se == result.se


# ---------------------------------------------------------------------------
# Probes, the private entry point, serialization and the CLI
# ---------------------------------------------------------------------------


def test_probe_reused_for_calibrator_fit_still_raises(tmp_path: Path) -> None:
    """0.9.1's held-out check applies to exempt policies too."""
    data = _guide(tmp_path)
    fit_rows = [json.loads(line) for line in data["path"].read_text().splitlines()]
    for i, row in enumerate(fit_rows):
        row["observation_id"] = f"fit-obs-{i}"
    data["path"].write_text("".join(json.dumps(row) + "\n" for row in fit_rows))
    probes = dict(data["probes"])
    probes["candidate"] = list(probes["candidate"]) + [dict(fit_rows[0])]
    with pytest.raises(ValueError, match="used to fit the calibrator"):
        analyze_dataset(
            fresh_draws_data=data["draws"],
            transport=_transport(probes),
            **_external(data["path"]),
        )


class _IdentityCalibrator:
    covariate_names: List[str] = []

    def predict(self, scores: Any, covariates: Optional[Any] = None) -> np.ndarray:
        return np.asarray(scores, dtype=float)


@pytest.mark.parametrize(
    "check, exempt",
    [
        (None, False),
        ({"passed": False, "failed": ["labelled_rows_unbalanced"]}, False),
        ({"passed": True, "failed": [], "effective_labelled_prompts": 40.0}, True),
    ],
    ids=["unchecked", "failed", "passed"],
)
def test_private_attach_scopes_fail_by_correction_check(
    check: Optional[Dict[str, Any]], exempt: bool
) -> None:
    from cje.interface.analysis import _attach_transport_audits

    policies = ["augmented", "plug_in"]
    metadata: Dict[str, Any] = {
        "target_policies": policies,
        "point_estimator": {"routes": ["augmented", "plug_in"]},
    }
    if check is not None:
        metadata["correction_checks"] = {"augmented": check}
    result = EstimationResult(
        estimates=np.array([0.4, 0.4]),
        standard_errors=np.array([0.01, 0.01]),
        n_samples_used={policy: 30 for policy in policies},
        method="calibrated_direct",
        influence_functions=None,
        diagnostics=None,
        calibrator=_IdentityCalibrator(),
        metadata=metadata,
    )
    rng = np.random.default_rng(59)
    config = _transport(
        {policy: _probes(rng, policy, shift=0.2) for policy in policies}
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

    audits = result.metadata["transport_audits"]
    for policy in policies:
        assert audits[policy]["status"] == "FAIL"
    plug_in = audits["plug_in"]
    assert plug_in["applies_to_current_estimate"] is True
    assert "gate_exemption" not in plug_in
    assert _gate(result, "plug_in") == {
        "flagged": True,
        "refused": False,
        "refuse_level_claims": True,
        "reasons": [_fail_reason(plug_in)],
    }
    audit = audits["augmented"]
    if exempt:
        assert audit["applies_to_current_estimate"] is False
        assert audit["gate_exemption"] == "residual_corrected"
        gate = _gate(result, "augmented")
        assert gate["flagged"] is False
        assert gate["exemption"] == "residual_corrected"
        assert "40 effective labelled prompts" in gate["notes"][0]
    else:
        assert audit["applies_to_current_estimate"] is True
        assert "gate_exemption" not in audit
        assert _gate(result, "augmented") == {
            "flagged": True,
            "refused": False,
            "refuse_level_claims": True,
            "reasons": [_fail_reason(audit)],
        }


@pytest.mark.parametrize("detail", ["portable", "full"])
def test_gate_notes_round_trip(tmp_path: Path, detail: str) -> None:
    data = _guide(tmp_path)
    result = analyze_dataset(
        fresh_draws_data=data["draws"],
        transport=_transport(data["probes"]),
        **_external(data["path"]),
    )
    restored = EstimationResult.from_dict(
        json.loads(json.dumps(result.to_dict(detail=detail)))
    )

    assert restored.gates == result.gates
    for key in ("reliability_gates", "correction_checks"):
        assert restored.metadata[key] == result.metadata[key]
    assert restored.metadata["reliability_gates"]["candidate"]["exemption"] == (
        "residual_corrected"
    )
    for policy, audit in result.metadata["transport_audits"].items():
        saved = restored.metadata["transport_audits"][policy]
        assert saved.get("gate_exemption") == audit.get("gate_exemption")
    assert restored.best_policy() == result.best_policy()
    assert restored.summary() == result.summary()


def test_cli_marks_exempt_audit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from cje.interface.cli import main

    rng = np.random.default_rng(61)
    n = 300
    s_a = rng.uniform(0.2, 0.6, n)
    s_b = rng.uniform(0.2, 0.6, n)
    top = np.zeros(n, dtype=bool)
    top[np.argsort(s_b)[-40:]] = True
    draws = {
        "corrected": _rows(
            s_a,
            _even_rank_mask(s_a, 40),
            lambda s: 0.3 + 0.5 * s + rng.normal(0, 0.02),
            prefix="a",
        ),
        "unbalanced": _rows(
            s_b, top, lambda s: 0.3 + 0.5 * s + rng.normal(0, 0.02), prefix="b"
        ),
    }
    directory = tmp_path / "responses"
    directory.mkdir()
    for policy, rows in draws.items():
        (directory / f"{policy}_responses.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    probe_path = tmp_path / "corrected_probe.jsonl"
    probe_path.write_text(
        "".join(json.dumps(row) + "\n" for row in _probes(rng, "corrected", shift=0.25))
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cje",
            "analyze",
            str(directory),
            "--transport-probe",
            f"corrected={probe_path}",
            "--transport-margin",
            "corrected=0.03",
            "--estimator-config",
            '{"inference_method": "cluster_robust"}',
        ],
    )
    assert main() == 0
    out = capsys.readouterr().out
    assert (
        "    residual transport: FAIL (calibration map only; estimate "
        "residual-corrected)\n" in out
    )
    assert "    residual transport: NOT_CHECKED\n" in out
    assert "design check" not in out
    assert "Best by point estimate: " in out
