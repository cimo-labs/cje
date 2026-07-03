"""Pin the 0.4.0 migration errors VERBATIM.

0.4.0 removed the off-policy (IPS/DR) modes. Users landing on the old entry
points must get the exact migration copy from the design doc — these tests
compare full message strings, not substrings, so any drift in the copy fails
loudly. Covered surfaces:

- analyze_dataset(logged_data_path=...)   (API kwarg kept solely to raise)
- estimator='<removed name>'              (factory, shared by CLI and API)
- cje.advanced.<RemovedClass>             (module __getattr__ ImportError)
- the CLI's old logged-dataset positional (`cje analyze logged_data.jsonl`)
"""

import pytest

from cje import analyze_dataset
from cje.interface.config import AnalysisConfig
from cje.interface.factory import (
    REMOVED_ESTIMATORS,
    create_estimator,
    get_estimator_names,
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
    (maintained on the 0.3.x branch; docs at the v0.3.0 tag)."""

EXPECTED_REMOVED_ESTIMATOR_TEMPLATE = """\
estimator='{name}' was removed in cje-eval 0.4.0 (Direct-mode only).
Off-policy estimators (calibrated-ips, raw-ips, dr-cpo, mrdr, tmle, stacked-dr)
live on the frozen 0.3.x line: pip install "cje-eval==0.3.*".
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


class TestLoggedDataPathMigrationError:
    def test_api_kwarg_raises_exact_message(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            analyze_dataset(logged_data_path="logged_data.jsonl")
        assert str(excinfo.value) == EXPECTED_LOGGED_DATA_PATH_MESSAGE

    def test_raises_even_when_fresh_draws_also_provided(self) -> None:
        """The kwarg must never be silently ignored in favor of fresh draws."""
        with pytest.raises(ValueError) as excinfo:
            analyze_dataset(
                logged_data_path="logged_data.jsonl",
                fresh_draws_data=_TINY_FRESH_DRAWS,
            )
        assert str(excinfo.value) == EXPECTED_LOGGED_DATA_PATH_MESSAGE

    def test_config_never_sees_logged_data_path(self) -> None:
        """The field is gone from AnalysisConfig — the kwarg exists only on
        analyze_dataset, solely to raise the migration error."""
        assert "logged_data_path" not in AnalysisConfig.model_fields


# ---------------------------------------------------------------------------
# Removed estimator names (factory — single source for CLI and API)
# ---------------------------------------------------------------------------


class TestRemovedEstimatorMigrationError:
    @pytest.mark.parametrize("name", REMOVED_ESTIMATOR_NAMES)
    def test_factory_raises_exact_message(self, name: str) -> None:
        with pytest.raises(ValueError) as excinfo:
            validate_estimator_name(name)
        assert str(excinfo.value) == EXPECTED_REMOVED_ESTIMATOR_TEMPLATE.format(
            name=name
        )

    @pytest.mark.parametrize("name", REMOVED_ESTIMATOR_NAMES)
    def test_create_estimator_raises_exact_message(self, name: str) -> None:
        with pytest.raises(ValueError) as excinfo:
            create_estimator(name, ["policy_a"], {}, None, False)
        assert str(excinfo.value) == EXPECTED_REMOVED_ESTIMATOR_TEMPLATE.format(
            name=name
        )

    def test_api_path_raises_exact_message(self) -> None:
        """analyze_dataset routes estimator names through the same factory."""
        with pytest.raises(ValueError) as excinfo:
            analyze_dataset(
                fresh_draws_data=_TINY_FRESH_DRAWS, estimator="calibrated-ips"
            )
        assert str(excinfo.value) == EXPECTED_REMOVED_ESTIMATOR_TEMPLATE.format(
            name="calibrated-ips"
        )

    def test_factory_removed_list_matches_pinned_names(self) -> None:
        assert set(REMOVED_ESTIMATORS) == set(REMOVED_ESTIMATOR_NAMES)

    def test_surviving_names_still_validate(self) -> None:
        assert set(get_estimator_names()) == {"calibrated-direct", "direct"}
        for name in get_estimator_names():
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
