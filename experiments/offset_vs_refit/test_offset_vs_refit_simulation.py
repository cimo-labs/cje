"""Tests for offset-vs-refit simulation experiments."""

from __future__ import annotations

import numpy as np
import pytest

import offset_vs_refit_simulation as simulation
from offset_vs_refit_simulation import (
    METHODS,
    run_experiment_suite,
    summarize_results,
)


def test_offset_vs_refit_smoke() -> None:
    """Simulation should return expected columns and methods."""
    df = run_experiment_suite(
        audit_sizes=[20],
        n_reps=1,
        n_old_labels=400,
        n_eval_per_policy=500,
        seed=7,
        scenarios=["intercept_shift"],
        audit_profiles=["balanced"],
    )
    assert not df.empty
    assert set(df["method"].unique()) == set(METHODS)
    required_cols = {
        "scenario",
        "audit_profile",
        "audit_size",
        "method",
        "mae_policy_mean",
        "rmse_policy_mean",
        "ranking_correct",
        "transport_status",
    }
    assert required_cols.issubset(set(df.columns))


def test_intercept_shift_offsets_improve_old_plugin() -> None:
    """Under intercept drift, offset corrections should beat legacy plug-in."""
    df = run_experiment_suite(
        audit_sizes=[50],
        n_reps=8,
        n_old_labels=800,
        n_eval_per_policy=1200,
        seed=17,
        scenarios=["intercept_shift"],
        audit_profiles=["balanced"],
    )
    summary = summarize_results(df)
    chunk = summary[
        (summary["scenario"] == "intercept_shift")
        & (summary["audit_profile"] == "balanced")
        & (summary["audit_size"] == 50)
    ]
    mae_old = float(chunk[chunk["method"] == "old_plugin"]["mae_policy_mean"].iloc[0])
    mae_global_offset = float(
        chunk[chunk["method"] == "old_plus_global_offset"]["mae_policy_mean"].iloc[0]
    )
    mae_policy_offset = float(
        chunk[chunk["method"] == "old_plus_policy_offset"]["mae_policy_mean"].iloc[0]
    )
    assert mae_global_offset < mae_old
    assert mae_policy_offset < mae_old


def test_slope_shift_policy_offset_beats_global_offset() -> None:
    """Under slope drift, policy offsets should beat one global offset."""
    df = run_experiment_suite(
        audit_sizes=[50],
        n_reps=8,
        n_old_labels=800,
        n_eval_per_policy=1200,
        seed=19,
        scenarios=["slope_shift"],
        audit_profiles=["balanced"],
    )
    summary = summarize_results(df)
    chunk = summary[
        (summary["scenario"] == "slope_shift")
        & (summary["audit_profile"] == "balanced")
        & (summary["audit_size"] == 50)
    ]
    mae_global_offset = float(
        chunk[chunk["method"] == "old_plus_global_offset"]["mae_policy_mean"].iloc[0]
    )
    mae_policy_offset = float(
        chunk[chunk["method"] == "old_plus_policy_offset"]["mae_policy_mean"].iloc[0]
    )
    assert mae_policy_offset < mae_global_offset


def test_slope_shift_refit_beats_global_offset() -> None:
    """Under slope drift, monotone refit should beat global offset."""
    df = run_experiment_suite(
        audit_sizes=[50],
        n_reps=6,
        n_old_labels=800,
        n_eval_per_policy=1200,
        seed=21,
        scenarios=["slope_shift"],
        audit_profiles=["base_heavy"],
    )
    summary = summarize_results(df)
    chunk = summary[
        (summary["scenario"] == "slope_shift")
        & (summary["audit_profile"] == "base_heavy")
        & (summary["audit_size"] == 50)
    ]
    mae_offset = float(
        chunk[chunk["method"] == "old_plus_global_offset"]["mae_policy_mean"].iloc[0]
    )
    mae_refit = float(
        chunk[chunk["method"] == "recent_refit_monotone"]["mae_policy_mean"].iloc[0]
    )
    assert mae_refit < mae_offset


@pytest.mark.parametrize("kind", ["monotone", "two_stage"])
def test_fit_cv_preserves_partial_labels_and_prompt_clusters(kind: str) -> None:
    """Missing labels stay missing, and repeated prompts never cross-fit each other."""
    judge = np.linspace(0.05, 0.95, 40)
    labels = 0.1 + 0.7 * judge
    labels[1::2] = np.nan
    prompt_ids = np.repeat([f"fit:{i}" for i in range(20)], 2)
    covariate = (np.arange(40) % 3).astype(float)
    calibrator = simulation._fit_calibrator(
        judge=judge,
        oracle=labels,
        covariate=covariate if kind == "two_stage" else None,
        kind=kind,
        seed=42,
        prompt_ids=prompt_ids,
    )
    observed = np.isfinite(labels)
    np.testing.assert_array_equal(calibrator._fit_oracle_labels, labels[observed])
    assert calibrator._fit_prompt_ids == prompt_ids[observed].tolist()
    for prompt_id in set(prompt_ids):
        assert len(set(calibrator._fold_ids[prompt_ids == prompt_id])) == 1
    if kind == "two_stage":
        np.testing.assert_array_equal(
            calibrator._fit_covariates[:, 0], covariate[observed]
        )


def test_transport_audits_are_graded_on_labels_held_out_from_old_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit the old fit once per cell with the declared family and no label leakage."""
    audit_calls = []
    original_audit = simulation.audit_transportability

    def inspect_audit(calibrator, probes, **kwargs):
        fit_ids = set(calibrator._fit_prompt_ids)
        probe_ids = {row["prompt_id"] for row in probes}
        assert fit_ids and probe_ids and fit_ids.isdisjoint(probe_ids)
        assert all(pid.startswith("old:") for pid in fit_ids)
        assert all(pid.startswith("audit:") for pid in probe_ids)
        assert kwargs["delta_max"] == 0.04
        assert kwargs["alpha"] == 0.10
        assert kwargs["family_size"] == 2
        diag = original_audit(calibrator, probes, **kwargs)
        assert diag.n_clusters == len(probes)
        assert diag.status in {"PASS", "FAIL", "INCONCLUSIVE"}
        audit_calls.append(diag)
        return diag

    monkeypatch.setattr(simulation, "audit_transportability", inspect_audit)
    df = run_experiment_suite(
        audit_sizes=[10, 30],
        n_reps=1,
        n_old_labels=100,
        n_eval_per_policy=150,
        scenarios=["intercept_shift"],
        audit_profiles=["balanced"],
        delta_max=0.04,
        audit_alpha=0.10,
        seed=9,
    )
    assert len(audit_calls) == 2  # Seven methods share each old-fit audit.
    assert set(df["transport_family_size"]) == {2}
    summary = summarize_results(df)
    rate_columns = [
        "pass_rate",
        "fail_rate",
        "inconclusive_rate",
        "not_graded_rate",
        "not_checked_rate",
    ]
    np.testing.assert_allclose(summary[rate_columns].sum(axis=1), 1)
    assert (summary["not_graded_rate"] == 0).all()
    assert (summary["not_checked_rate"] == 0).all()
    assert "warn_rate" not in summary


def test_reusing_old_fit_rows_as_probes_is_rejected() -> None:
    rng = np.random.default_rng(8)
    old = simulation._sample_policy_data(
        policy=simulation.POLICIES[0],
        n=20,
        scenario="intercept_shift",
        period="old",
        rng=rng,
        sample_prefix="old:8",
    )
    evaluation = simulation._sample_policy_data(
        policy=simulation.POLICIES[0],
        n=20,
        scenario="intercept_shift",
        period="new",
        rng=rng,
        sample_prefix="evaluation:8",
    )
    calibrator = simulation._fit_calibrator(
        old["judge"], old["oracle"], None, "monotone", 8, old["prompt_id"]
    )
    with pytest.raises(ValueError, match="prompt clusters must be disjoint"):
        simulation._method_estimates(
            calibrator,
            old,
            old,
            {"base": evaluation},
            seed=8,
            delta_max=0.05,
            audit_alpha=0.05,
            family_size=1,
        )
