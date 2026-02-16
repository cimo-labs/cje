"""Numerical experiments: when does audit-slice offset beat refit (and vice versa)?

This module simulates judge->oracle drift across time periods and compares:
1) Old calibrator plug-in estimates
2) Old calibrator + global audit-slice offset
3) Refit on audit slice (recent-only)
4) Refit on pooled old+new labels
5) Two-stage (covariate-aware) variants

The target metric is policy-level first moments: E[Y | policy].
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

import numpy as np
import pandas as pd

from cje.calibration.judge import JudgeCalibrator
from cje.diagnostics.transport import audit_transportability


Scenario = Literal[
    "intercept_shift",
    "slope_shift",
    "nonlinear_shift",
    "covariate_interaction_shift",
]

CalibratorKind = Literal["monotone", "two_stage"]


@dataclass(frozen=True)
class PolicySpec:
    name: str
    score_alpha: float
    score_beta: float
    covariate_prob: float


POLICIES: Tuple[PolicySpec, ...] = (
    PolicySpec("base", 2.5, 2.5, 0.50),
    PolicySpec("high_score", 5.0, 2.0, 0.30),
    PolicySpec("low_score", 2.0, 5.0, 0.70),
    PolicySpec("high_covariate", 2.5, 2.5, 0.90),
)


AUDIT_PROFILES: Dict[str, Dict[str, float]] = {
    "base_heavy": {
        "base": 0.75,
        "high_score": 0.10,
        "low_score": 0.10,
        "high_covariate": 0.05,
    },
    "balanced": {
        "base": 0.25,
        "high_score": 0.25,
        "low_score": 0.25,
        "high_covariate": 0.25,
    },
}


METHODS: Tuple[str, ...] = (
    "old_plugin",
    "old_plus_global_offset",
    "recent_refit_monotone",
    "pooled_refit_monotone",
    "recent_refit_two_stage",
    "pooled_refit_two_stage",
)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _oracle_function(
    score: np.ndarray,
    covariate: np.ndarray,
    scenario: Scenario,
    period: Literal["old", "new"],
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate oracle labels from latent score and covariate."""
    # Old-period baseline mapping
    if period == "old":
        mu = 0.08 + 0.75 * score + 0.08 * covariate
    elif scenario == "intercept_shift":
        # Pure mean shift; global offset should work very well.
        mu = 0.18 + 0.75 * score + 0.08 * covariate
    elif scenario == "slope_shift":
        # Residuals become score-dependent; global offset is insufficient.
        mu = 0.04 + 0.98 * score + 0.08 * covariate
    elif scenario == "nonlinear_shift":
        # Curvature drift: old linear calibrator misses shape.
        mu = 0.04 + 0.35 * score + 0.55 * (score**2) + 0.08 * covariate
    elif scenario == "covariate_interaction_shift":
        # Drift depends on observable covariate.
        mu = 0.08 + 0.70 * score + 0.22 * covariate + 0.20 * score * covariate
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    noise = rng.normal(0.0, 0.04, size=score.shape[0])
    return _clip01(mu + noise)


def _judge_score(
    latent_score: np.ndarray,
    covariate: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate observed judge score with noise and mild covariate distortion."""
    noise = rng.normal(0.0, 0.10, size=latent_score.shape[0])
    return _clip01(latent_score + 0.02 * covariate + noise)


def _sample_policy_data(
    policy: PolicySpec,
    n: int,
    scenario: Scenario,
    period: Literal["old", "new"],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    score = rng.beta(policy.score_alpha, policy.score_beta, size=n)
    covariate = rng.binomial(1, policy.covariate_prob, size=n).astype(float)
    judge = _judge_score(score, covariate, rng)
    oracle = _oracle_function(score, covariate, scenario, period, rng)
    return {
        "policy": np.full(n, policy.name, dtype=object),
        "score": score,
        "covariate": covariate,
        "judge": judge,
        "oracle": oracle,
    }


def _concat_data(parts: Iterable[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    part_list = list(parts)
    if not part_list:
        raise ValueError("No data parts to concatenate")
    out: Dict[str, np.ndarray] = {}
    for key in part_list[0].keys():
        out[key] = np.concatenate([p[key] for p in part_list], axis=0)
    return out


def _sample_audit_slice(
    n: int,
    scenario: Scenario,
    audit_profile: Dict[str, float],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    policy_names = list(audit_profile.keys())
    probs = np.array([audit_profile[name] for name in policy_names], dtype=float)
    probs = probs / probs.sum()
    choices = rng.choice(policy_names, size=n, p=probs)

    grouped: List[Dict[str, np.ndarray]] = []
    for policy in POLICIES:
        count = int(np.sum(choices == policy.name))
        if count == 0:
            continue
        grouped.append(
            _sample_policy_data(
                policy=policy,
                n=count,
                scenario=scenario,
                period="new",
                rng=rng,
            )
        )
    return _concat_data(grouped)


def _fit_calibrator(
    judge: np.ndarray,
    oracle: np.ndarray,
    covariate: np.ndarray | None,
    kind: CalibratorKind,
    seed: int,
) -> JudgeCalibrator:
    if kind == "two_stage":
        calibrator = JudgeCalibrator(
            random_seed=seed,
            calibration_mode="two_stage",
            covariate_names=["covariate"],
        )
        calibrator.fit_transform(
            judge_scores=judge,
            oracle_labels=oracle,
            covariates=covariate.reshape(-1, 1) if covariate is not None else None,
        )
        return calibrator

    calibrator = JudgeCalibrator(
        random_seed=seed,
        calibration_mode="monotone",
    )
    calibrator.fit_transform(
        judge_scores=judge,
        oracle_labels=oracle,
    )
    return calibrator


def _predict_mean(
    calibrator: JudgeCalibrator,
    judge: np.ndarray,
    covariate: np.ndarray,
    kind: CalibratorKind,
) -> float:
    if kind == "two_stage":
        pred = calibrator.predict(
            judge_scores=judge,
            covariates=covariate.reshape(-1, 1),
        )
    else:
        pred = calibrator.predict(judge_scores=judge)
    return float(np.mean(pred))


def _method_estimates(
    old_calibrator: JudgeCalibrator,
    old_labels: Dict[str, np.ndarray],
    audit_slice: Dict[str, np.ndarray],
    eval_by_policy: Dict[str, Dict[str, np.ndarray]],
    seed: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float | str]]:
    """Compute all method estimates and transport diagnostics."""
    method_to_estimates: Dict[str, Dict[str, float]] = {
        method: {} for method in METHODS
    }

    # Old plug-in estimates
    for policy, data in eval_by_policy.items():
        method_to_estimates["old_plugin"][policy] = _predict_mean(
            old_calibrator,
            judge=data["judge"],
            covariate=data["covariate"],
            kind="monotone",
        )

    # Global offset from audit slice
    old_audit_pred = old_calibrator.predict(judge_scores=audit_slice["judge"])
    delta_hat = float(np.mean(audit_slice["oracle"] - old_audit_pred))
    for policy, value in method_to_estimates["old_plugin"].items():
        method_to_estimates["old_plus_global_offset"][policy] = value + delta_hat

    # Recent-only refits
    recent_mono = _fit_calibrator(
        judge=audit_slice["judge"],
        oracle=audit_slice["oracle"],
        covariate=None,
        kind="monotone",
        seed=seed + 11,
    )
    recent_two_stage = _fit_calibrator(
        judge=audit_slice["judge"],
        oracle=audit_slice["oracle"],
        covariate=audit_slice["covariate"],
        kind="two_stage",
        seed=seed + 13,
    )

    # Pooled refits
    pooled_judge = np.concatenate([old_labels["judge"], audit_slice["judge"]], axis=0)
    pooled_oracle = np.concatenate(
        [old_labels["oracle"], audit_slice["oracle"]], axis=0
    )
    pooled_cov = np.concatenate(
        [old_labels["covariate"], audit_slice["covariate"]], axis=0
    )
    pooled_mono = _fit_calibrator(
        judge=pooled_judge,
        oracle=pooled_oracle,
        covariate=None,
        kind="monotone",
        seed=seed + 17,
    )
    pooled_two_stage = _fit_calibrator(
        judge=pooled_judge,
        oracle=pooled_oracle,
        covariate=pooled_cov,
        kind="two_stage",
        seed=seed + 19,
    )

    for policy, data in eval_by_policy.items():
        method_to_estimates["recent_refit_monotone"][policy] = _predict_mean(
            recent_mono,
            judge=data["judge"],
            covariate=data["covariate"],
            kind="monotone",
        )
        method_to_estimates["pooled_refit_monotone"][policy] = _predict_mean(
            pooled_mono,
            judge=data["judge"],
            covariate=data["covariate"],
            kind="monotone",
        )
        method_to_estimates["recent_refit_two_stage"][policy] = _predict_mean(
            recent_two_stage,
            judge=data["judge"],
            covariate=data["covariate"],
            kind="two_stage",
        )
        method_to_estimates["pooled_refit_two_stage"][policy] = _predict_mean(
            pooled_two_stage,
            judge=data["judge"],
            covariate=data["covariate"],
            kind="two_stage",
        )

    # Transport diagnostics for old calibrator on audit slice
    probe = [
        {
            "judge_score": float(j),
            "oracle_label": float(y),
        }
        for j, y in zip(audit_slice["judge"], audit_slice["oracle"])
    ]
    diag = audit_transportability(old_calibrator, probe, group_label="audit_slice")
    transport_info: Dict[str, float | str] = {
        "delta_hat": float(diag.delta_hat),
        "delta_ci_low": float(diag.delta_ci[0]),
        "delta_ci_high": float(diag.delta_ci[1]),
        "status": str(diag.status),
    }
    return method_to_estimates, transport_info


def _score_estimates(
    estimates: Dict[str, float],
    truth: Dict[str, float],
) -> Dict[str, float]:
    policies = sorted(truth.keys())
    errors = np.array([estimates[p] - truth[p] for p in policies], dtype=float)
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    ranking_correct = float(
        max(estimates.items(), key=lambda kv: kv[1])[0]
        == max(truth.items(), key=lambda kv: kv[1])[0]
    )
    return {
        "mae_policy_mean": mae,
        "rmse_policy_mean": rmse,
        "ranking_correct": ranking_correct,
    }


def run_experiment_suite(
    audit_sizes: List[int],
    n_reps: int,
    n_old_labels: int = 1500,
    n_eval_per_policy: int = 4000,
    seed: int = 42,
    scenarios: Iterable[Scenario] | None = None,
    audit_profiles: Iterable[str] | None = None,
) -> pd.DataFrame:
    rng_master = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    max_audit_size = max(audit_sizes)

    scenario_values: Tuple[Scenario, ...]
    if scenarios is None:
        scenario_values = (
            "intercept_shift",
            "slope_shift",
            "nonlinear_shift",
            "covariate_interaction_shift",
        )
    else:
        scenario_values = tuple(scenarios)

    profile_items: List[Tuple[str, Dict[str, float]]]
    if audit_profiles is None:
        profile_items = list(AUDIT_PROFILES.items())
    else:
        profile_items = [(name, AUDIT_PROFILES[name]) for name in audit_profiles]

    for scenario in scenario_values:
        for audit_profile_name, audit_profile in profile_items:
            for rep in range(n_reps):
                rep_seed = int(rng_master.integers(0, 2**31 - 1))
                rng = np.random.default_rng(rep_seed)

                # Old-period calibration labels from base policy.
                old_labels = _sample_policy_data(
                    policy=POLICIES[0],
                    n=n_old_labels,
                    scenario=scenario,
                    period="old",
                    rng=rng,
                )
                old_calibrator = _fit_calibrator(
                    judge=old_labels["judge"],
                    oracle=old_labels["oracle"],
                    covariate=None,
                    kind="monotone",
                    seed=rep_seed + 7,
                )

                # New-period evaluation data per policy (used for truth + estimates).
                eval_by_policy: Dict[str, Dict[str, np.ndarray]] = {}
                truth: Dict[str, float] = {}
                for policy in POLICIES:
                    data = _sample_policy_data(
                        policy=policy,
                        n=n_eval_per_policy,
                        scenario=scenario,
                        period="new",
                        rng=rng,
                    )
                    eval_by_policy[policy.name] = data
                    truth[policy.name] = float(np.mean(data["oracle"]))

                # Single audit pool per replicate; use prefixes for smaller sizes.
                audit_full = _sample_audit_slice(
                    n=max_audit_size,
                    scenario=scenario,
                    audit_profile=audit_profile,
                    rng=rng,
                )
                perm = rng.permutation(max_audit_size)
                for key in audit_full.keys():
                    audit_full[key] = audit_full[key][perm]

                for audit_size in sorted(audit_sizes):
                    audit_slice = {
                        key: value[:audit_size] for key, value in audit_full.items()
                    }
                    method_to_estimates, transport_info = _method_estimates(
                        old_calibrator=old_calibrator,
                        old_labels=old_labels,
                        audit_slice=audit_slice,
                        eval_by_policy=eval_by_policy,
                        seed=rep_seed + audit_size,
                    )

                    for method, estimates in method_to_estimates.items():
                        metrics = _score_estimates(estimates=estimates, truth=truth)
                        rows.append(
                            {
                                "scenario": scenario,
                                "audit_profile": audit_profile_name,
                                "replicate": rep,
                                "audit_size": audit_size,
                                "method": method,
                                "mae_policy_mean": metrics["mae_policy_mean"],
                                "rmse_policy_mean": metrics["rmse_policy_mean"],
                                "ranking_correct": metrics["ranking_correct"],
                                "transport_status": transport_info["status"],
                                "transport_delta_hat": transport_info["delta_hat"],
                                "transport_delta_ci_low": transport_info[
                                    "delta_ci_low"
                                ],
                                "transport_delta_ci_high": transport_info[
                                    "delta_ci_high"
                                ],
                            }
                        )

    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    status_counts = (
        df.groupby(
            ["scenario", "audit_profile", "audit_size", "method", "transport_status"],
            as_index=False,
        )["replicate"]
        .count()
        .rename(columns={"replicate": "n_status"})
    )

    summary = (
        df.groupby(
            ["scenario", "audit_profile", "audit_size", "method"], as_index=False
        )
        .agg(
            mae_policy_mean=("mae_policy_mean", "mean"),
            rmse_policy_mean=("rmse_policy_mean", "mean"),
            ranking_accuracy=("ranking_correct", "mean"),
            transport_delta_hat=("transport_delta_hat", "mean"),
            n=("replicate", "count"),
        )
        .sort_values(["scenario", "audit_profile", "audit_size", "mae_policy_mean"])
        .reset_index(drop=True)
    )

    status_pivot = (
        status_counts.pivot_table(
            index=["scenario", "audit_profile", "audit_size", "method"],
            columns="transport_status",
            values="n_status",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    for col in ("PASS", "WARN", "FAIL"):
        if col not in status_pivot.columns:
            status_pivot[col] = 0

    merged = summary.merge(
        status_pivot,
        on=["scenario", "audit_profile", "audit_size", "method"],
        how="left",
    )
    merged["pass_rate"] = merged["PASS"] / merged["n"]
    merged["warn_rate"] = merged["WARN"] / merged["n"]
    merged["fail_rate"] = merged["FAIL"] / merged["n"]
    return merged


def _plot_results(summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    # Plot 1: Method MAE at largest audit slice
    max_audit = int(summary["audit_size"].max())
    top = summary[summary["audit_size"] == max_audit].copy()
    scenarios = sorted(top["scenario"].unique().tolist())
    profiles = sorted(top["audit_profile"].unique().tolist())

    fig, axes = plt.subplots(
        nrows=len(scenarios),
        ncols=len(profiles),
        figsize=(5 * len(profiles), 3.2 * len(scenarios)),
        squeeze=False,
    )
    for i, scenario in enumerate(scenarios):
        for j, profile in enumerate(profiles):
            ax = axes[i][j]
            chunk = top[
                (top["scenario"] == scenario) & (top["audit_profile"] == profile)
            ].sort_values("mae_policy_mean")
            ax.bar(chunk["method"], chunk["mae_policy_mean"], color="#1f77b4")
            ax.set_title(f"{scenario}\n{profile}, audit={max_audit}")
            ax.tick_params(axis="x", rotation=60)
            ax.set_ylabel("Mean |policy bias|")
    fig.tight_layout()
    fig.savefig(output_dir / "method_mae_at_max_audit.png", dpi=200)
    plt.close(fig)

    # Plot 2: Offset vs recent refit as audit size grows
    compare = summary[
        summary["method"].isin(["old_plus_global_offset", "recent_refit_monotone"])
    ].copy()
    fig2, axes2 = plt.subplots(
        nrows=len(scenarios),
        ncols=len(profiles),
        figsize=(5 * len(profiles), 3.2 * len(scenarios)),
        squeeze=False,
    )
    for i, scenario in enumerate(scenarios):
        for j, profile in enumerate(profiles):
            ax = axes2[i][j]
            chunk = compare[
                (compare["scenario"] == scenario)
                & (compare["audit_profile"] == profile)
            ]
            for method in ["old_plus_global_offset", "recent_refit_monotone"]:
                sub = chunk[chunk["method"] == method].sort_values("audit_size")
                ax.plot(
                    sub["audit_size"],
                    sub["mae_policy_mean"],
                    marker="o",
                    label=method,
                )
            ax.set_title(f"{scenario}\n{profile}")
            ax.set_xlabel("Audit slice size")
            ax.set_ylabel("Mean |policy bias|")
            ax.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(output_dir / "offset_vs_refit_by_audit_size.png", dpi=200)
    plt.close(fig2)


def _parse_int_list(text: str) -> List[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    if min(values) < 10:
        raise ValueError("All audit sizes must be >= 10")
    return sorted(values)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run offset-vs-refit simulations for transport drift in judge calibration."
        )
    )
    parser.add_argument(
        "--audit-sizes",
        type=str,
        default="20,50,100,200",
        help="Comma-separated audit slice sizes.",
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=60,
        help="Replicates per scenario/profile.",
    )
    parser.add_argument(
        "--n-old-labels",
        type=int,
        default=1500,
        help="Old-period oracle labels used to fit the legacy calibrator.",
    )
    parser.add_argument(
        "--n-eval-per-policy",
        type=int,
        default=4000,
        help="Evaluation samples per policy per replicate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cje/experiments/offset_vs_refit_results"),
        help="Directory for CSV/JSON/plots.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib plot generation.",
    )
    args = parser.parse_args()

    audit_sizes = _parse_int_list(args.audit_sizes)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = run_experiment_suite(
        audit_sizes=audit_sizes,
        n_reps=args.n_reps,
        n_old_labels=args.n_old_labels,
        n_eval_per_policy=args.n_eval_per_policy,
        seed=args.seed,
    )
    summary = summarize_results(df)

    df.to_csv(output_dir / "offset_vs_refit_raw.csv", index=False)
    summary.to_csv(output_dir / "offset_vs_refit_summary.csv", index=False)
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(
            {
                "audit_sizes": audit_sizes,
                "n_reps": args.n_reps,
                "n_old_labels": args.n_old_labels,
                "n_eval_per_policy": args.n_eval_per_policy,
                "seed": args.seed,
                "audit_profiles": AUDIT_PROFILES,
                "methods": list(METHODS),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    if not args.no_plots:
        _plot_results(summary, output_dir)

    # Console digest: best method by scenario/profile at largest audit size.
    max_audit = max(audit_sizes)
    digest = (
        summary[summary["audit_size"] == max_audit]
        .sort_values(
            ["scenario", "audit_profile", "mae_policy_mean", "ranking_accuracy"],
            ascending=[True, True, True, False],
        )
        .groupby(["scenario", "audit_profile"], as_index=False)
        .first()
    )
    print("\nBest method at max audit size:")
    print(
        digest[
            [
                "scenario",
                "audit_profile",
                "method",
                "mae_policy_mean",
                "rmse_policy_mean",
                "ranking_accuracy",
            ]
        ].to_string(index=False)
    )
    print(f"\nOutputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
