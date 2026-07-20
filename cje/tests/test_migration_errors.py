"""Pin the 0.4.0 migration errors VERBATIM.

0.4.0 removed the off-policy (IPS/DR) modes. Users landing on the old entry
points must get the exact migration copy from the design doc — these tests
compare full message strings, not substrings, so any drift in the copy fails
loudly. Covered surfaces:

- analyze_dataset no longer exposes logged_data_path
- estimator='<removed name>'              (_removed.py, shared by CLI and API)
- cje.advanced.<RemovedClass>             (module __getattr__ ImportError)
- the CLI's old logged-dataset positional (`cje analyze logged_data.jsonl`)
"""

import inspect
import json
import sys
from pathlib import Path

import pytest

from cje import analyze_dataset
from cje.interface._removed import (
    REMOVED_ESTIMATORS,
    validate_estimator_name,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# The exact copy (design doc §6). Written out literally — NOT imported from
# the modules under test — so the pin catches edits on either side.
# ---------------------------------------------------------------------------

EXPECTED_LOGGED_DATA_PATH_MESSAGE = """\
Off-policy evaluation was removed in cje-eval 0.4.0; 'logged_data_path' is no longer accepted.

CJE is now Direct-mode only: generate fresh draws from each policy, judge them,
and label a small oracle slice.

  * Have fresh draws?  Pass fresh_draws_dir=... or fresh_draws_data=... .
  * Your logged data has judge_score + oracle_label?  It still works as the
    calibration source: pass calibration_data_path="<your logged data>.jsonl".
  * Need IPS/DR from logged propensities?  Pin the frozen OPE line:
        pip install "cje-eval==0.3.*"
    (maintained on the 0.3.x branch; docs at the v0.3.0 tag; requires
    Python <=3.12 — on 3.13 use a 3.12 env for OPE)."""

EXPECTED_REMOVED_ESTIMATOR_TEMPLATE = """\
estimator='{name}' was removed in cje-eval 0.4.0 (Direct-mode only).
Off-policy estimators (calibrated-ips, raw-ips, dr-cpo, mrdr, tmle, stacked-dr)
live on the frozen 0.3.x line: pip install "cje-eval==0.3.*"
(requires Python <=3.12; on 3.13 use a 3.12 env for OPE).
Use estimator='calibrated-direct' (the default) with fresh draws instead."""

# Every estimator name that must raise: the six documented OPE estimators
# plus the ghost aliases 0.3.x still recognized in its mode-inference lists.
REMOVED_ESTIMATOR_NAMES = (
    "calibrated-ips",
    "raw-ips",
    "dr-cpo",
    "mrdr",
    "tmle",
    "stacked-dr",
    "oc-dr-cpo",
    "tr-cpo",
    "tr-cpo-e",
)

REMOVED_ADVANCED_NAMES = (
    "CalibratedIPS",
    "PrecomputedSampler",
    "DRCPOEstimator",
    "MRDREstimator",
    "TMLEEstimator",
    "StackedDREstimator",
)

_TINY_FRESH_DRAWS = {"policy_a": [{"prompt_id": "p0", "judge_score": 0.7}]}


# ---------------------------------------------------------------------------
# logged_data_path (API)
# ---------------------------------------------------------------------------


class TestLoggedDataPathRemoval:
    def test_kwarg_is_not_in_public_signature(self) -> None:
        params = inspect.signature(analyze_dataset).parameters
        assert "logged_data_path" not in params

    def test_removed_kwarg_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="logged_data_path"):
            analyze_dataset(logged_data_path="logged_data.jsonl")  # type: ignore[call-arg]

    def test_analysis_arguments_are_keyword_only(self) -> None:
        with pytest.raises(TypeError, match="positional"):
            analyze_dataset("responses/")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Removed estimator names (_removed.py — single source for CLI and API)
# ---------------------------------------------------------------------------


class TestRemovedEstimatorMigrationError:
    @pytest.mark.parametrize("name", REMOVED_ESTIMATOR_NAMES)
    def test_validator_raises_exact_message(self, name: str) -> None:
        with pytest.raises(ValueError) as excinfo:
            validate_estimator_name(name)
        assert str(excinfo.value) == EXPECTED_REMOVED_ESTIMATOR_TEMPLATE.format(
            name=name
        )

    def test_api_path_raises_exact_message(self) -> None:
        """analyze_dataset routes estimator names through the same validator."""
        with pytest.raises(ValueError) as excinfo:
            analyze_dataset(
                fresh_draws_data=_TINY_FRESH_DRAWS, estimator="calibrated-ips"
            )
        assert str(excinfo.value) == EXPECTED_REMOVED_ESTIMATOR_TEMPLATE.format(
            name="calibrated-ips"
        )

    def test_removed_list_matches_pinned_names(self) -> None:
        assert set(REMOVED_ESTIMATORS) == set(REMOVED_ESTIMATOR_NAMES)

    def test_surviving_names_still_validate(self) -> None:
        for name in ("calibrated-direct", "direct"):
            assert validate_estimator_name(name) == name

    def test_unknown_name_raises_plain_error(self) -> None:
        """Typos get the short unknown-estimator error, not migration copy."""
        with pytest.raises(ValueError, match="Unknown estimator type: bogus"):
            validate_estimator_name("bogus")


# ---------------------------------------------------------------------------
# cje.advanced removed classes
# ---------------------------------------------------------------------------


class TestAdvancedRemovedNames:
    @pytest.mark.parametrize("name", REMOVED_ADVANCED_NAMES)
    def test_attribute_access_raises_informative_import_error(self, name: str) -> None:
        from cje import advanced

        with pytest.raises(ImportError) as excinfo:
            getattr(advanced, name)
        assert str(excinfo.value) == (
            f"cje.advanced.{name} was removed in 0.4.0 — "
            f'pip install "cje-eval==0.3.*" for OPE'
        )

    def test_from_import_raises_import_error(self) -> None:
        """`from cje.advanced import CalibratedIPS` goes through __getattr__."""
        with pytest.raises(ImportError, match="removed in 0.4.0"):
            from cje.advanced import CalibratedIPS  # noqa: F401

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        from cje import advanced

        with pytest.raises(AttributeError):
            advanced.NoSuchThing

    def test_surviving_exports_still_importable(self) -> None:
        from cje.advanced import CalibratedDirectEstimator, calibrate_dataset

        assert CalibratedDirectEstimator is not None
        assert calibrate_dataset is not None


# ---------------------------------------------------------------------------
# CLI exit behavior for the removed 0.3.x invocations
# ---------------------------------------------------------------------------


def _run_cli(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    from cje.interface.cli import main

    monkeypatch.setattr(sys, "argv", ["cje", *argv])
    return main()


def _write_logged_dataset(path: Path, n: int = 5) -> None:
    """An 0.3.x logged dataset: logprob fields, no target_policy field."""
    records = [
        {
            "prompt_id": f"p{i}",
            "prompt": f"question {i}",
            "response": f"answer {i}",
            "base_policy_logprob": -10.0,
            "target_policy_logprobs": {"pi_target": -11.0},
            "judge_score": 0.1 + 0.15 * i,
        }
        for i in range(n)
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


class TestCLIMigrationErrors:
    def test_logged_positional_exits_1_with_exact_copy(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The 0.3.x `cje analyze logged_data.jsonl` invocation."""
        logged = tmp_path / "logged_data.jsonl"
        _write_logged_dataset(logged)

        exit_code = _run_cli(monkeypatch, "analyze", str(logged))

        assert exit_code == 1
        assert EXPECTED_LOGGED_DATA_PATH_MESSAGE in capsys.readouterr().err

    def test_logged_positional_with_fresh_draws_dir_exits_1(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The 0.3.x DR invocation (logged positional + --fresh-draws-dir):
        a positional alongside the alias always meant logged data."""
        logged = tmp_path / "logged_data.jsonl"
        _write_logged_dataset(logged)
        draws_dir = tmp_path / "responses"
        draws_dir.mkdir()

        exit_code = _run_cli(
            monkeypatch,
            "analyze",
            str(logged),
            "--fresh-draws-dir",
            str(draws_dir),
        )

        assert exit_code == 1
        assert EXPECTED_LOGGED_DATA_PATH_MESSAGE in capsys.readouterr().err

    def test_removed_estimator_exits_1_with_exact_copy(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Removed names are not argparse choices: the factory's migration
        error must reach stderr intact."""
        draws_dir = tmp_path / "responses"
        draws_dir.mkdir()

        exit_code = _run_cli(
            monkeypatch, "analyze", str(draws_dir), "--estimator", "tmle", "--quiet"
        )

        assert exit_code == 1
        assert EXPECTED_REMOVED_ESTIMATOR_TEMPLATE.format(name="tmle") in (
            capsys.readouterr().err
        )


# ---------------------------------------------------------------------------
# 0.5.0 tombstones. Unlike the 0.4.0 pins above these are deliberately NOT
# verbatim: they assert only that the removed name raises ImportError and
# that the message names the replacement, so the copy can be rewordsmithed
# without breaking tests.
# ---------------------------------------------------------------------------


class TestRemovedIn050Tombstones:
    """Highest-traffic names removed in 0.5.0 raise pointers, not attribute
    errors."""

    @pytest.mark.parametrize("module_name", ["cje.estimators", "cje.advanced"])
    def test_base_cje_estimator_points_at_direct_estimator(
        self, module_name: str
    ) -> None:
        import importlib

        module = importlib.import_module(module_name)
        with pytest.raises(ImportError, match="CalibratedDirectEstimator"):
            getattr(module, "BaseCJEEstimator")

    @pytest.mark.parametrize(
        "name", ["calibrate_from_raw_data", "calibrate_judge_scores"]
    )
    def test_removed_calibration_helpers_point_at_survivors(self, name: str) -> None:
        import cje.calibration

        with pytest.raises(ImportError, match="calibrate_dataset"):
            getattr(cje.calibration, name)
        with pytest.raises(ImportError, match="fit_cv"):
            getattr(cje.calibration, name)

    @pytest.mark.parametrize("module_name", ["cje", "cje.visualization"])
    def test_plot_calibration_comparison_removed(self, module_name: str) -> None:
        import importlib

        module = importlib.import_module(module_name)
        with pytest.raises(ImportError, match="removed in 0.5.0"):
            getattr(module, "plot_calibration_comparison")

    def test_compare_policies_bootstrap_points_at_compare_policies(self) -> None:
        import cje.diagnostics

        with pytest.raises(ImportError, match="compare_policies"):
            getattr(cje.diagnostics, "compare_policies_bootstrap")

    def test_040_tombstones_still_fire(self) -> None:
        """The 0.4.0 removals must keep raising after the 0.5.0 additions."""
        import cje.advanced

        with pytest.raises(ImportError, match="0.3"):
            getattr(cje.advanced, "CalibratedIPS")

    def test_unknown_names_still_raise_attribute_error(self) -> None:
        import cje.calibration
        import cje.estimators

        with pytest.raises(AttributeError):
            getattr(cje.estimators, "definitely_not_a_real_name")
        with pytest.raises(AttributeError):
            getattr(cje.calibration, "definitely_not_a_real_name")
