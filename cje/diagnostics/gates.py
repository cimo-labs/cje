"""Canonical diagnostic gate thresholds and status helpers.

Single source of truth for the CJE paper's diagnostic gates
(arXiv:2512.11150). Every surface that grades diagnostics must import
thresholds from this module so that a given number receives the same
verdict everywhere.

Diagnostic summary:

- Scalar range support: >= 5% of target judge-score mass outside the labeled
  calibration range triggers REFUSE-LEVEL for absolute claims. This check does
  not establish residual transport or the validity of policy rankings.
- Scope of calibration-map gates: a REFUSE-LEVEL badge or a residual-transport
  FAIL gates a policy only when its reported level depends on the calibration
  map (``level_gate_scope``). Complete-oracle means never do; residual-
  corrected estimates do not when their correction design check passes
  (``correction_design_check``).
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .models import Status

# ---------------------------------------------------------------------------
# Coverage badge (boundary_card)
# ---------------------------------------------------------------------------
# Fraction of target judge-score mass outside the oracle calibration range
# at or above which level claims are refused (REFUSE-LEVEL).
OUT_OF_RANGE_REFUSE_THRESHOLD = 0.05

# Fraction of calibrated rewards near the oracle reward bounds at or above
# which the card downgrades to CAUTION (boundary effects likely) when the
# out-of-range mass is below the refuse threshold.
SATURATION_CAUTION_THRESHOLD = 0.20

# Canonical boundary-card status -> per-policy Status ladder: a CAUTION card
# is a real WARNING tier between GOOD and CRITICAL (it used to leave the
# policy GOOD, making CAUTION invisible in overall_status).
BOUNDARY_CARD_STATUS_TO_STATUS: Dict[str, Status] = {
    "OK": Status.GOOD,
    "CAUTION": Status.WARNING,
    "REFUSE-LEVEL": Status.CRITICAL,
    "INCONCLUSIVE": Status.WARNING,
}

# ---------------------------------------------------------------------------
# Transport audit (audit_transportability)
# ---------------------------------------------------------------------------
# Deprecated compatibility constant. Residual audits are graded against an
# explicit, user-declared delta_max; this value is not used by the classifier.
TRANSPORT_FAIL_DELTA_THRESHOLD = 0.05

# Canonical transport status -> Status ladder for consumers that grade
# TransportDiagnostics alongside other diagnostics.
TRANSPORT_STATUS_TO_STATUS: Dict[str, Status] = {
    "PASS": Status.GOOD,
    "FAIL": Status.CRITICAL,
    "INCONCLUSIVE": Status.WARNING,
    "NOT_GRADED": Status.WARNING,
    # NOT_CHECKED means no probe was supplied at all — informational, not a
    # defect: it maps to GOOD so probe-less runs do not warn in worst-status
    # rollups. The per-policy audit record still says NOT_CHECKED explicitly.
    "NOT_CHECKED": Status.GOOD,
    # Compatibility for previously serialized diagnostics only.
    "WARN": Status.WARNING,
}


_STATUS_ORDER = {Status.GOOD: 0, Status.WARNING: 1, Status.CRITICAL: 2}


def worst_status(*statuses: Optional[Status]) -> Status:
    """Combine statuses, returning the worst. None entries are ignored.

    Returns Status.GOOD when no non-None status is provided.
    """
    worst = Status.GOOD
    for status in statuses:
        if status is not None and _STATUS_ORDER[status] > _STATUS_ORDER[worst]:
            worst = status
    return worst


# ---------------------------------------------------------------------------
# Which estimates the calibration-map gates apply to
# ---------------------------------------------------------------------------
# A transport FAIL and a REFUSE-LEVEL badge are statements about the
# calibration map. They gate a policy's level only when that level depends on
# the map. A complete-oracle mean ("direct_oracle") never does. A residual-
# corrected estimate ("augmented") does not either, provided its labelled rows
# really are the declared design's sample of the evaluation rows; then the
# exemption swaps the hard gate for the corrected interval, so it needs enough
# labels for that interval to be trusted. Evidence for the two constants:
#
# * Floor (deep dive E1, ``cov_i``, design A: one labelled row per prompt at
#   equal propensity): on the 0.9.0 analytic interval, weight-one coverage of
#   a nominal 95% interval was 0.895-0.927 at 10 labelled prompts and
#   0.932-0.942 at 20 (0.9.1 takes the interval's degrees of freedom from the
#   labelled prompts, which never narrows it). The floor counts *effective*
#   labelled prompts (Kish count over prompt clusters of the inverse-propensity
#   weights), which equals the distinct-prompt count in E1's design and is
#   smaller when a few prompts carry most of the labels or the weight.
# * Known propensities also need a design effective sample size
#   N^2 / sum(1 / p) of at least the floor. In ``kp_rare`` (400 rows, a 10%
#   stratum at p=0.005, p=0.08 elsewhere) labelled prompts had median 29 but
#   the design effective n was 12.8 and the interval covered 0.167 of the time.
# * Balance threshold (design-check simulations): under random representative
#   labels (discrete and continuous judges, 1-2 draws per prompt, 20/30/60
#   labelled prompts) P(max |t| >= 3) was 0.1-0.65%. Under correctly declared
#   known propensities (constant or score-dependent; each row or each whole
#   prompt sampled; 1/2/4 draws per prompt; 20/30/60 expected labelled
#   prompts; 2,000 seeds per cell) it was 0.1-2.0%, median 0.55%, highest at
#   20 labelled prompts with 4 draws each; a row-level Poisson variance alone
#   gave 7% at 2 and 27% at 4 draws when whole prompts were sampled (p=0.25,
#   150 prompts, 400 seeds). Labels on the top, bottom, middle band or tails
#   by score were caught in every run at 20+ labelled prompts under both
#   designs. The check is a falsification test only: it cannot see selection
#   unrelated to the judge score, and a mild exp(0.5 z) score tilt is caught
#   only 11-14/28-34/74-77% of the time at 20/30/60 labels.
CORRECTION_EXEMPT_MIN_LABELLED_PROMPTS = 20
CORRECTION_DESIGN_BALANCE_T = 3.0


def level_gate_scope(
    route: Optional[str], check: Optional[Mapping[str, Any]]
) -> Tuple[bool, Optional[str]]:
    """Whether calibration-map gates apply to a policy's reported level.

    Returns ``(applies, exemption)``. A complete-oracle mean
    (``"direct_oracle"``) never depends on the map, as before. An
    ``"augmented"`` (residual-corrected) level does not either when its
    correction design check passed: ``(False, "residual_corrected")``. Every
    other policy, including an augmented one whose check failed or was not
    recorded, keeps the gates exactly as in 0.9.1: ``(True, None)``.
    """
    if route == "direct_oracle":
        return False, "direct_oracle"
    if route == "augmented" and check is not None and check.get("passed") is True:
        return False, "residual_corrected"
    return True, None


def correction_design_check(
    judge_scores: Any,
    observed: Any,
    cluster_ids: Any,
    label_design: str,
    propensities: Optional[Any] = None,
    labelled_outcomes_constant: bool = False,
) -> Dict[str, Any]:
    """Check that a residual correction's labels can lift map gates.

    Inputs are one policy's evaluation rows. The check requires at least
    ``CORRECTION_EXEMPT_MIN_LABELLED_PROMPTS`` effective labelled prompts
    (Kish count over prompt clusters of the inverse-propensity weights), a
    known-propensity design effective sample size ``N^2 / sum(1/p)`` of at
    least the same floor, no unlabelled row declared certain to be labelled
    (known propensity 1), non-constant labelled outcomes, and judge-score
    balance: the Horvitz-Thompson weighted labelled rows must reproduce the
    mean and spread of the judge score over all rows (and, under known
    propensities, the row count) within ``CORRECTION_DESIGN_BALANCE_T``
    design standard errors. The standard errors are prompt-cluster ones (under
    known propensities, at least the declared row-level Poisson one), so
    labels drawn by prompt (every draw of a sampled prompt) are graded
    correctly.

    It is a falsification check, not proof: it catches labels selected by
    judge score, but not selection unrelated to the score, and it does not
    balance calibrator covariates.

    Returns a JSON-safe dict with ``design``, ``labelled_prompts``,
    ``effective_labelled_prompts``, ``design_effective_n`` and
    ``unlabelled_certain_rows`` (both None unless known-propensity),
    ``balance_t`` ({variable: t}; t is None when a deviation has no design
    variance, which fails balance), ``passed`` and ``failed`` (failure codes,
    in order).
    """
    if label_design not in ("representative", "known_propensity"):
        raise ValueError(
            "correction_design_check needs label_design 'representative' or "
            f"'known_propensity', got {label_design!r}"
        )
    scores = np.asarray(judge_scores, dtype=float).reshape(-1)
    mask = np.asarray(observed, dtype=bool).reshape(-1)
    n_rows = len(scores)
    clusters = np.asarray(cluster_ids)
    if mask.shape != (n_rows,) or clusters.shape[:1] != (n_rows,):
        raise ValueError("judge_scores, observed and cluster_ids must be row-aligned")
    if n_rows == 0:
        raise ValueError("correction_design_check needs at least one row")
    if not np.all(np.isfinite(scores)):
        raise ValueError("judge_scores must be finite")
    _, cluster_index = np.unique(clusters, return_inverse=True)
    cluster_index = np.asarray(cluster_index).reshape(-1)
    n_labelled = int(np.sum(mask))

    known = label_design == "known_propensity"
    if known:
        if propensities is None:
            raise ValueError("known_propensity design check needs propensities")
        inclusion = np.asarray(propensities, dtype=float).reshape(-1)
        if inclusion.shape != (n_rows,):
            raise ValueError("propensities must align to the policy's rows")
        if not np.all(np.isfinite(inclusion)) or np.any(
            (inclusion <= 0) | (inclusion > 1)
        ):
            raise ValueError("propensities must be finite in (0, 1]")
    else:
        inclusion = np.full(n_rows, n_labelled / n_rows, dtype=float)

    weights = np.zeros(n_rows, dtype=float)
    weights[mask] = 1.0 / inclusion[mask]
    labelled_clusters = np.unique(cluster_index[mask])
    if n_labelled:
        cluster_weights = np.bincount(cluster_index, weights=weights)[labelled_clusters]
        # Scaled by the largest weight so tiny propensities cannot overflow.
        cluster_weights = cluster_weights / np.max(cluster_weights)
        effective = float(np.sum(cluster_weights) ** 2 / np.sum(cluster_weights**2))
    else:
        effective = 0.0
    # Round away float noise so the floor is inclusive (20 equal clusters
    # must give exactly 20, not 19.999999999999996).
    effective = round(effective, 9)
    design_effective_n = (
        round(float(n_rows**2 / np.sum(1.0 / inclusion)), 9) if known else None
    )

    # Under known propensities a row declared certain (p = 1) is always
    # labelled; an unlabelled one contradicts the declared design ("1 = no
    # reweighting" is the usual slip), and the Horvitz-Thompson correction
    # then leaves the level close to the map's.
    certain_unlabelled = int(np.sum((inclusion >= 1.0) & ~mask)) if known else None

    spread_raw = np.abs(scores - np.median(scores))
    balance_variables: Dict[str, np.ndarray] = {}
    if known:
        balance_variables["label_count"] = np.ones(n_rows, dtype=float)
    balance_variables["judge_level"] = scores - np.mean(scores)
    balance_variables["judge_spread"] = spread_raw - np.mean(spread_raw)
    # Centred deviations at float-noise size (a constant score, or a spread
    # that is identical on every row) are exactly zero. The scale includes the
    # score magnitude so a constant score (zero range) is covered too.
    noise = 1e-9 * max(
        float(np.max(scores) - np.min(scores)), float(np.max(np.abs(scores)))
    )
    balance_t: Dict[str, Optional[float]] = {}
    degenerate = False
    for name, values in balance_variables.items():
        x = values
        if name != "label_count":
            x = np.where(np.abs(values) <= noise, 0.0, values)
        deviation = (weights - 1.0) * x
        numerator = float(np.sum(deviation))
        # Prompt-cluster variance over all clusters, for both designs. Labels
        # are often drawn by prompt (every draw of a sampled prompt), and a
        # row-level variance then understates the variance by about the draws
        # per prompt. Under known propensities the declared row-level Poisson
        # variance is exact for row-level sampling and steadier than the
        # cluster estimate at small label counts, so the larger of the two is
        # used. An overflow is handled below as an unusable variance.
        with np.errstate(over="ignore", invalid="ignore"):
            variance = float(np.sum(np.bincount(cluster_index, weights=deviation) ** 2))
            if known:
                poisson = float(np.sum((1.0 - inclusion) / inclusion * x**2))
                variance = max(variance, poisson)
        tolerance = 1e-9 * max(1.0, float(np.sum(np.abs(deviation))))
        if variance > 0 and np.isfinite(variance) and np.isfinite(numerator):
            balance_t[name] = float(numerator / np.sqrt(variance))
        elif np.isfinite(numerator) and abs(numerator) <= tolerance:
            balance_t[name] = 0.0
        else:
            # A deviation with no usable design variance: unbalanced, with an
            # undefined t (None keeps the record JSON-safe).
            balance_t[name] = None
            degenerate = True

    failed: List[str] = []
    if effective < CORRECTION_EXEMPT_MIN_LABELLED_PROMPTS:
        failed.append("too_few_effective_labelled_prompts")
    if (
        design_effective_n is not None
        and design_effective_n < CORRECTION_EXEMPT_MIN_LABELLED_PROMPTS
    ):
        failed.append("low_design_effective_n")
    if certain_unlabelled:
        failed.append("unlabelled_rows_declared_certain")
    if labelled_outcomes_constant:
        failed.append("labelled_outcomes_constant")
    if (
        degenerate
        or max(abs(t) for t in balance_t.values() if t is not None)
        >= CORRECTION_DESIGN_BALANCE_T
    ):
        failed.append("labelled_rows_unbalanced")
    return {
        "design": str(label_design),
        "labelled_prompts": int(len(labelled_clusters)),
        "effective_labelled_prompts": float(effective),
        "design_effective_n": design_effective_n,
        "unlabelled_certain_rows": certain_unlabelled,
        "balance_t": balance_t,
        "passed": not failed,
        "failed": failed,
    }


def _design_words(check: Optional[Mapping[str, Any]]) -> str:
    design = check.get("design") if check is not None else None
    return "known-propensity" if design == "known_propensity" else "representative"


def _balance_text(check: Mapping[str, Any]) -> str:
    return ", ".join(
        (
            f"{name} t={float(value):+.2f}"
            if value is not None
            else f"{name} t undefined (no design variance)"
        )
        for name, value in (check.get("balance_t") or {}).items()
    )


def _certain_text(check: Mapping[str, Any]) -> str:
    count = int(check.get("unlabelled_certain_rows") or 0)
    return (
        f"{count} unlabelled row{'s have' if count != 1 else ' has'} a declared "
        "label propensity of 1 (certain to be labelled)"
    )


def corrected_gate_note(finding: str, check: Optional[Mapping[str, Any]]) -> str:
    """Why a map finding does not gate a residual-corrected policy."""
    effective = float((check or {}).get("effective_labelled_prompts", 0.0))
    return (
        f"{finding} describes the calibration map, not this estimate: the "
        f"estimate is residual-corrected with {effective:.0f} effective "
        f"labelled prompts ({_design_words(check)} design; labelled rows passed "
        "the design balance check), so it does not gate this policy"
    )


def correction_caution(
    policy: str, check: Optional[Mapping[str, Any]]
) -> Optional[str]:
    """WARNING text for a failed correction design check, or None.

    Warns only on failures the reported interval does not already show:
    labelled rows that do not match the declared design
    (``labelled_rows_unbalanced``), a small known-propensity design effective
    sample size (``low_design_effective_n``), and unlabelled rows declared at
    propensity 1 (``unlabelled_rows_declared_certain``). Too few labelled
    prompts shows in the interval, and constant labelled outcomes already warn.
    """
    if check is None:
        return None
    failed = list(check.get("failed") or [])
    sentences: List[str] = []
    if "unlabelled_rows_declared_certain" in failed:
        sentences.append(
            _certain_text(check)
            + ", which the declared design rules out: propensity 1 means a row "
            "is always labelled, not 'no reweighting'. The Horvitz-Thompson "
            "correction counts those rows as having zero residual, which pulls "
            "the level toward the calibration map's. Declare the probability "
            "with which each row was sent for labelling."
        )
    if "labelled_rows_unbalanced" in failed:
        design = _design_words(check)
        text = (
            f"its labelled rows do not look like a {design} sample of its "
            f"evaluation rows ({_balance_text(check)}). The corrected estimate "
            "assumes they are; labels chosen by score, availability or "
            "difficulty can bias it in either direction."
        )
        sentences.append(text if not sentences else "Also, " + text)
    if "low_design_effective_n" in failed:
        text = (
            "the declared label propensities give a design effective sample "
            f"size of {float(check.get('design_effective_n') or 0.0):.1f} "
            f"(< {CORRECTION_EXEMPT_MIN_LABELLED_PROMPTS}); a few rare "
            "high-weight rows dominate the Horvitz-Thompson correction and its "
            "interval can under-cover badly."
        )
        sentences.append(text if not sentences else "Also, " + text)
    if not sentences:
        return None
    return (
        f"Residual correction for policy {policy!r}: "
        + " ".join(sentences)
        + " Calibration-map gates still apply to this policy."
    )


def residual_corrected_suffix(record: Optional[Mapping[str, Any]]) -> str:
    """Display suffix for an audit or card whose finding was not applied.

    Non-empty only for a record with ``gate_exemption`` ``"residual_corrected"``
    (an augmented policy whose correction design check passed).
    """
    if record and record.get("gate_exemption") == "residual_corrected":
        return " (calibration map only; estimate residual-corrected)"
    return ""
