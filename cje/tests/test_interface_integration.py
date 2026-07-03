"""Integration tests for the high-level interface.

These tests validate that the public interface chooses sensible defaults
and runs end-to-end on real arena sample data.
"""

from pathlib import Path

import pytest

from cje.interface.analysis import analyze_dataset


pytestmark = [pytest.mark.integration, pytest.mark.uses_arena_sample]


def _arena_paths() -> tuple[Path, Path]:
    """Return (dataset_path, fresh_draws_dir) from examples directory."""
    here = Path(__file__).parent
    # Point to examples directory (shared with tutorials)
    dataset_path = (
        here.parent.parent / "examples" / "arena_sample" / "logged_data.jsonl"
    )
    fresh_draws_dir = here.parent.parent / "examples" / "arena_sample" / "fresh_draws"
    if not dataset_path.exists():
        pytest.skip(f"Arena sample not found: {dataset_path}")
    return dataset_path, fresh_draws_dir


def test_logged_data_path_raises_migration_error() -> None:
    """OPE modes were removed in 0.4.0; logged_data_path must fail loudly.

    NOTE(WP2): the exact migration copy is pinned by test_migration_errors;
    here we only assert that the temporary error fires and names the release.
    """
    dataset_path, _ = _arena_paths()

    with pytest.raises(ValueError, match="0.4.0"):
        analyze_dataset(logged_data_path=str(dataset_path))


def test_removed_estimator_names_raise_migration_error() -> None:
    """Removed OPE estimator names must raise, not fall through."""
    _, responses_dir = _arena_paths()

    for name in ("calibrated-ips", "raw-ips", "dr-cpo", "mrdr", "tmle", "stacked-dr"):
        with pytest.raises(ValueError, match="0.4.0"):
            analyze_dataset(fresh_draws_dir=str(responses_dir), estimator=name)


def test_direct_mode_with_calibration_data() -> None:
    """Direct mode with a dedicated calibration dataset (judge+oracle labels)."""
    dataset_path, responses_dir = _arena_paths()

    results = analyze_dataset(
        fresh_draws_dir=str(responses_dir),
        calibration_data_path=str(dataset_path),
        estimator="direct",
        verbose=False,
    )

    assert results is not None
    assert results.metadata.get("mode") == "direct"
    assert (
        results.metadata.get("estimand") == "on-policy evaluation on provided prompts"
    )
    assert len(results.estimates) > 0
    assert "target_policies" in results.metadata
    assert results.metadata.get("calibration") in (
        "from_calibration_data_combined",
        "from_calibration_data_only",
    )


def test_direct_mode_without_fresh_draws_raises_error() -> None:
    """Direct mode requires fresh draws."""
    with pytest.raises(ValueError, match="fresh_draws"):
        analyze_dataset(estimator="direct")


def test_direct_only_mode_works() -> None:
    """Test that Direct-only mode works with just fresh_draws_dir (no logged data)."""
    _, responses_dir = _arena_paths()

    # Direct-only mode: fresh draws without logged data
    results = analyze_dataset(
        fresh_draws_dir=str(responses_dir),
        estimator="auto",  # Should auto-select "direct"
        verbose=False,
    )

    assert results is not None
    assert results.metadata.get("mode") == "direct"
    # Fresh draws now include base policy with oracle labels (48% coverage)
    # So calibration should be "from_fresh_draws" using reward calibration
    assert results.metadata.get("calibration") == "from_fresh_draws"
    assert results.metadata.get("oracle_coverage", 0) > 0
    assert len(results.estimates) > 0
    assert "target_policies" in results.metadata


def test_mode_selection_metadata_populated() -> None:
    """mode_selection metadata is populated (0.3.0-output compat keys)."""
    _, responses_dir = _arena_paths()

    result = analyze_dataset(
        fresh_draws_dir=str(responses_dir),
        estimator="auto",
        verbose=False,
    )

    # Verify mode_selection metadata exists
    assert "mode_selection" in result.metadata
    mode_sel = result.metadata["mode_selection"]

    # Verify all required fields
    assert "mode" in mode_sel
    assert "estimator" in mode_sel
    assert "logprob_coverage" in mode_sel
    assert "has_fresh_draws" in mode_sel
    assert "has_logged_data" in mode_sel
    assert "reason" in mode_sel

    # Verify correct values for Direct-only mode
    assert mode_sel["mode"] == "direct"
    assert mode_sel["estimator"] == "direct"
    assert mode_sel["has_fresh_draws"] is True
    assert mode_sel["has_logged_data"] is False


def test_direct_estimates_clone_accurately() -> None:
    """Direct mode should accurately estimate clone policy value.

    Clone policy is nearly identical to the base policy (same model, same
    prompts). Ground truth is the mean calibrated reward in the logged data,
    which is used here purely as the calibration dataset.
    """
    from cje.data import load_dataset_from_jsonl
    import numpy as np

    dataset_path, fresh_draws_dir = _arena_paths()

    # Calculate ground truth: mean calibrated reward in the labeled data
    from cje.calibration import calibrate_dataset

    dataset = load_dataset_from_jsonl(str(dataset_path))
    calibrated_dataset, _ = calibrate_dataset(
        dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
        enable_cross_fit=True,
        n_folds=5,
    )
    ground_truth = float(
        np.mean([s.reward for s in calibrated_dataset.samples if s.reward is not None])
    )

    # Run Direct mode (fresh draws for estimation, labeled data for calibration)
    # Use cluster_robust inference since fresh draws don't have oracle labels for bootstrap
    results_direct = analyze_dataset(
        fresh_draws_dir=str(fresh_draws_dir),
        calibration_data_path=str(dataset_path),
        combine_oracle_sources=False,  # Calibrate on the labeled data alone
        estimator="direct",
        estimator_config={
            "inference_method": "cluster_robust"
        },  # Fresh draws lack oracle
        verbose=False,
    )

    clone_idx = results_direct.metadata["target_policies"].index("clone")
    clone_direct = float(results_direct.estimates[clone_idx])

    assert 0 <= clone_direct <= 1, f"Direct estimate {clone_direct} out of range"
    assert abs(clone_direct - ground_truth) < 0.05, (
        f"Direct ({clone_direct:.3f}) differs from truth ({ground_truth:.3f}) by "
        f"{abs(clone_direct - ground_truth):.3f}"
    )


def test_direct_legacy_multipolicy_flag_is_stripped_from_metadata() -> None:
    """Legacy use_multipolicy_eif configs should warn, be ignored, and not persist in metadata."""
    dataset_path, fresh_draws_dir = _arena_paths()

    with pytest.warns(FutureWarning, match="deprecated and ignored"):
        results = analyze_dataset(
            fresh_draws_dir=str(fresh_draws_dir),
            calibration_data_path=str(dataset_path),
            estimator="direct",
            estimator_config={
                "inference_method": "cluster_robust",
                "use_multipolicy_eif": False,
            },
            verbose=False,
        )

    estimator_config = results.metadata.get("estimator_config", {})
    assert "use_multipolicy_eif" not in estimator_config
    assert estimator_config.get("inference_method") == "cluster_robust"


def test_direct_ranks_unhelpful_as_worst() -> None:
    """Direct mode should rank unhelpful as the worst policy.

    The unhelpful policy is intentionally poor and should be ranked lowest among
    all three policies (clone, parallel_universe_prompt, unhelpful).
    """
    _, fresh_draws_dir = _arena_paths()

    # Run Direct mode (fresh draws only)
    results_direct = analyze_dataset(
        fresh_draws_dir=str(fresh_draws_dir),
        estimator="auto",
        verbose=False,
    )

    policies_direct = results_direct.metadata["target_policies"]
    unhelpful_idx_direct = policies_direct.index("unhelpful")
    unhelpful_direct = float(results_direct.estimates[unhelpful_idx_direct])

    # Direct: unhelpful should be the minimum across all policies
    assert unhelpful_direct == min(results_direct.estimates), (
        f"Direct mode: unhelpful ({unhelpful_direct:.3f}) should be lowest, "
        f"but estimates are {[f'{e:.3f}' for e in results_direct.estimates]}"
    )
