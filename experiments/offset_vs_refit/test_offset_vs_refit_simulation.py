"""Tests for offset-vs-refit simulation experiments."""

from __future__ import annotations

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
