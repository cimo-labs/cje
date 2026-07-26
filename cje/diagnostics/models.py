"""
Diagnostic data models for CJE.

This module contains the diagnostics for the Direct-mode estimator:
- DirectDiagnostics: estimates, per-policy statuses, coverage badges
  (boundary cards), and calibration quality.

Computation logic is in the sibling modules (reward_boundary.py,
transport.py, robust_inference.py, etc.).
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Any
from enum import Enum


class Status(Enum):
    """Health status for diagnostics."""

    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DirectDiagnostics:
    """Diagnostics for the calibrated direct estimator.

    Direct mode has no importance weights, so there are no weight/overlap
    metrics here. The identification risk that matters is coverage: the
    per-policy boundary cards (the paper's coverage badge) record how much
    of each policy's fresh-draw judge-score mass falls outside the oracle
    calibration range. A "REFUSE-LEVEL" card means the scalar-support check
    does not support absolute claims for that policy. It does not certify
    rankings or residual transport.
    """

    # ========== Core Info (always present) ==========
    estimator_type: str  # "Direct"
    method: str  # "calibrated_direct" | "direct_oracle" | "naive_direct"
    n_samples_total: int
    n_samples_valid: int
    policies: List[str]

    # ========== Estimation Results (always present) ==========
    estimates: Dict[str, float]
    standard_errors: Dict[str, float]
    n_samples_used: Dict[str, int]

    # ========== Per-policy status ==========
    # CRITICAL when the policy's coverage badge refuses level claims.
    status_per_policy: Optional[Dict[str, Status]] = None

    # ========== Coverage badges (identification risk) ==========
    # Serialized BoundaryCard per policy (the paper's coverage badge:
    # status "OK" | "CAUTION" | "REFUSE-LEVEL"), plus "oracle_s_range".
    boundary_cards: Optional[Dict[str, Dict[str, Any]]] = None

    # ========== Calibration Diagnostics (None for naive mode) ==========
    calibration_rmse: Optional[float] = None
    calibration_coverage: Optional[float] = None  # P(|pred - oracle| < 0.1)
    calibration_tolerance: Optional[float] = None
    n_oracle_labels: Optional[int] = None

    # ========== Held-out residual transport (high-level API) ==========
    # Per-policy serialized TransportDiagnostics, or an explicit NOT_CHECKED
    # record when no independent probe was supplied.
    transport_audits: Optional[Dict[str, Dict[str, Any]]] = None
    transport_status_per_policy: Optional[Dict[str, str]] = None

    # ========== Computed Properties ==========

    @property
    def n_policies(self) -> int:
        """Number of target policies."""
        return len(self.policies)

    @property
    def filter_rate(self) -> float:
        """Fraction of samples filtered out."""
        if self.n_samples_total > 0:
            return 1.0 - (self.n_samples_valid / self.n_samples_total)
        return 0.0

    @property
    def best_policy(self) -> str:
        """Policy with highest estimate."""
        if not self.estimates:
            return "none"
        return max(self.estimates.items(), key=lambda x: x[1])[0]

    @property
    def is_calibrated(self) -> bool:
        """Check if this has calibration info."""
        return self.calibration_rmse is not None

    @property
    def refuse_level_policies(self) -> List[str]:
        """Policies whose coverage badge refuses level claims."""
        if not self.boundary_cards:
            return []
        return sorted(
            policy
            for policy, card in self.boundary_cards.items()
            if card.get("status") == "REFUSE-LEVEL"
        )

    @property
    def overall_status(self) -> Status:
        """Overall health status: the worst per-policy status."""
        from .gates import worst_status

        if not self.status_per_policy:
            return Status.GOOD
        return worst_status(*self.status_per_policy.values())

    def validate(self) -> List[str]:
        """Run self-consistency checks."""
        issues = []

        # Basic sanity checks
        if self.n_samples_valid > self.n_samples_total:
            issues.append(
                f"n_valid ({self.n_samples_valid}) > n_total ({self.n_samples_total})"
            )

        # Check for high filter rate
        if self.filter_rate > 0.5:
            issues.append(
                f"High filter rate: {self.filter_rate:.1%} of samples filtered"
            )

        # Check coverage badges (level-claim identification risk)
        if self.boundary_cards:
            for policy, card in self.boundary_cards.items():
                if card.get("status") == "REFUSE-LEVEL":
                    issues.append(
                        f"REFUSE-LEVEL for {policy}: "
                        f"{card.get('out_of_range', 0.0):.1%} of judge scores "
                        f"outside the oracle calibration range; do not ship "
                        f"level claims; this check does not certify rankings "
                        f"or residual transport"
                    )

        # Check estimates match policies
        for policy in self.estimates:
            if policy not in self.policies:
                issues.append(f"Estimate for unknown policy: {policy}")

        return issues

    def summary(self) -> str:
        """Generate concise summary."""
        lines = [
            f"Estimator: {self.estimator_type}",
            f"Method: {self.method}",
            f"Status: {self.overall_status.value}",
            f"Samples: {self.n_samples_valid}/{self.n_samples_total} valid ({100*(1-self.filter_rate):.1f}%)",
            f"Policies: {', '.join(self.policies)}",
            f"Best policy: {self.best_policy}",
        ]

        refuse_level = self.refuse_level_policies
        if refuse_level:
            lines.append(f"REFUSE-LEVEL: {', '.join(refuse_level)}")
        if self.transport_audits:
            statuses = ", ".join(
                f"{policy}={audit.get('status', 'NOT_CHECKED')}"
                for policy, audit in sorted(self.transport_audits.items())
            )
            lines.append(f"Residual transport: {statuses}")

        if self.is_calibrated:
            lines.append(f"Calibration RMSE: {self.calibration_rmse:.3f}")
            if self.calibration_coverage is not None:
                tolerance = (
                    f" (±{self.calibration_tolerance:g})"
                    if self.calibration_tolerance is not None
                    else ""
                )
                lines.append(
                    f"Calibration coverage{tolerance}: "
                    f"{self.calibration_coverage:.1%}"
                )
            if self.n_oracle_labels is not None:
                lines.append(f"Oracle labels: {self.n_oracle_labels}")

        # Add any validation issues
        issues = self.validate()
        if issues:
            lines.append("Issues: " + "; ".join(issues[:2]))

        return " | ".join(lines)

    def to_dict(self) -> Dict:
        """Export as dictionary for serialization."""
        from dataclasses import asdict

        d = asdict(self)
        d["overall_status"] = self.overall_status.value

        # Convert status_per_policy if present
        if d.get("status_per_policy"):
            d["status_per_policy"] = {
                policy: status.value if hasattr(status, "value") else status
                for policy, status in d["status_per_policy"].items()
            }

        return d

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=indent, default=str)
