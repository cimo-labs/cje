"""Regression tests for the 0.5.0 ingestion unification.

Pins the two ingest defects fixed by the shared cje.data.ingest helpers:

- D1: scale normalization only existed on the in-memory path. A directory
  of 0-100-scale fresh draws crashed with a ValidationError, and a
  0-100-scale calibration file was silently filtered down to "No valid
  samples". Directory (and single-file) input now flows through
  fresh_draws_from_dict's joint scale detection, and out-of-[0,1]
  calibration values are a hard, actionable error.
- D2: discovery, loading, and the CLI each kept their own (disagreeing)
  filename pattern lists, so a directory of bare {policy}.jsonl files was
  discovered and then failed to load. All three now resolve through
  POLICY_FILE_PATTERNS.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from cje import analyze_dataset
from cje.data.fresh_draws import (
    discover_policies_from_fresh_draws,
    fresh_draws_data_from_dir,
    load_fresh_draws_auto,
)
from cje.data.ingest import POLICY_FILE_PATTERNS, resolve_policy_file
from cje.data.loaders import DatasetLoader

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_records(
    policy: str,
    n: int = 40,
    n_oracle: int = 20,
    scale: float = 1.0,
    seed: int = 13,
) -> List[Dict[str, Any]]:
    """Judge-scored records (first n_oracle carry oracle labels)."""
    rng = np.random.default_rng(seed + sum(ord(c) for c in policy))
    records = []
    for i in range(n):
        score = float(np.clip(rng.uniform(0.05, 0.95), 0, 1))
        record: Dict[str, Any] = {
            "prompt_id": f"p{i}",
            "judge_score": score * scale,
        }
        if i < n_oracle:
            oracle = float(np.clip(score + rng.normal(0, 0.05), 0, 1))
            record["oracle_label"] = oracle * scale
        records.append(record)
    return records


def _write_policy_file(directory: Path, relative: str, records: list) -> None:
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


# ---------------------------------------------------------------------------
# D2: discovery and loading agree on POLICY_FILE_PATTERNS
# ---------------------------------------------------------------------------


class TestDiscoveryLoadingAgreement:
    def test_bare_policy_jsonl_dir_end_to_end(self, tmp_path: Path) -> None:
        """A dir of bare {policy}.jsonl files (discovered but unloadable in
        0.4.x — the service died with FileNotFoundError) now analyzes."""
        for policy in ("policy_a", "policy_b"):
            _write_policy_file(tmp_path, f"{policy}.jsonl", _fresh_records(policy))

        results = analyze_dataset(fresh_draws_dir=str(tmp_path))

        assert results.metadata["target_policies"] == ["policy_a", "policy_b"]
        assert len(results.estimates) == 2
        assert not np.any(np.isnan(results.estimates))

    @pytest.mark.parametrize("pattern", POLICY_FILE_PATTERNS)
    def test_every_pattern_is_discovered_and_loaded(
        self, tmp_path: Path, pattern: str
    ) -> None:
        """For each canonical pattern, a dir using it is BOTH discovered and
        loaded — 0.4.x's three pattern lists disagreed pairwise."""
        for policy in ("policy_a", "policy_b"):
            _write_policy_file(
                tmp_path, pattern.format(policy=policy), _fresh_records(policy, n=30)
            )

        discovered = discover_policies_from_fresh_draws(tmp_path)
        assert discovered == ["policy_a", "policy_b"]

        for policy in discovered:
            resolved = resolve_policy_file(tmp_path, policy)
            assert resolved is not None and resolved.exists()

        data = fresh_draws_data_from_dir(tmp_path)
        assert set(data) == {"policy_a", "policy_b"}
        assert all(len(records) == 30 for records in data.values())

        results = analyze_dataset(fresh_draws_dir=str(tmp_path))
        assert results.metadata["target_policies"] == ["policy_a", "policy_b"]

    def test_policy_fresh_jsonl_pattern_was_dropped(self, tmp_path: Path) -> None:
        """{policy}_fresh.jsonl was never documented and is gone: the file no
        longer resolves for policy 'pi' (it now reads as policy 'pi_fresh'
        via the bare {policy}.jsonl pattern)."""
        _write_policy_file(tmp_path, "pi_fresh.jsonl", _fresh_records("pi", n=5))

        assert resolve_policy_file(tmp_path, "pi") is None
        with pytest.raises(FileNotFoundError, match="No fresh draw file found"):
            load_fresh_draws_auto(tmp_path, "pi")
        assert discover_policies_from_fresh_draws(tmp_path) == ["pi_fresh"]


# ---------------------------------------------------------------------------
# D1 (directory + single-file paths): joint auto-normalization
# ---------------------------------------------------------------------------


class TestDirectoryNormalization:
    def test_0_100_dir_matches_in_memory_bit_identical(self, tmp_path: Path) -> None:
        """A 0-100-scale directory (a hard ValidationError crash in 0.4.x)
        analyzes, and its estimates are bit-identical to passing the same
        records via fresh_draws_data."""
        data = {
            policy: _fresh_records(policy, scale=100.0)
            for policy in ("policy_a", "policy_b")
        }
        for policy, records in data.items():
            _write_policy_file(tmp_path, f"{policy}_responses.jsonl", records)

        results_dir = analyze_dataset(fresh_draws_dir=str(tmp_path))
        results_mem = analyze_dataset(fresh_draws_data=data)

        np.testing.assert_array_equal(results_dir.estimates, results_mem.estimates)
        np.testing.assert_array_equal(
            results_dir.standard_errors, results_mem.standard_errors
        )

        # Same NormalizationInfo shape as the in-memory path
        assert "normalization" in results_dir.metadata
        assert results_dir.metadata["normalization"] == (
            results_mem.metadata["normalization"]
        )
        norm_meta = results_dir.metadata["normalization"]
        assert norm_meta["results_scale"] == "oracle_original"
        assert not norm_meta["judge_score"]["is_identity"]
        # Results are reported on the original 0-100 oracle scale
        assert np.all(results_dir.estimates > 1.0)

    def test_single_file_fresh_draws_dir_via_api(self, tmp_path: Path) -> None:
        """analyze_dataset(fresh_draws_dir=<file>) accepts a single JSONL
        file with a target_policy field per record (previously CLI-only)."""
        path = tmp_path / "draws.jsonl"
        lines = []
        for policy in ("policy_a", "policy_b"):
            for record in _fresh_records(policy):
                lines.append(json.dumps({**record, "target_policy": policy}))
        path.write_text("\n".join(lines) + "\n")

        results = analyze_dataset(fresh_draws_dir=str(path))

        assert results.metadata["target_policies"] == ["policy_a", "policy_b"]
        assert len(results.estimates) == 2
        assert not np.any(np.isnan(results.estimates))

    def test_single_file_without_target_policy_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "draws.jsonl"
        path.write_text(
            "\n".join(json.dumps(r) for r in _fresh_records("policy_a", n=5)) + "\n"
        )

        with pytest.raises(ValueError, match="target_policy"):
            analyze_dataset(fresh_draws_dir=str(path))


# ---------------------------------------------------------------------------
# D1 (calibration path): hard error instead of silent filtering
# ---------------------------------------------------------------------------


class TestCalibrationDataRangeErrors:
    def test_0_100_calibration_file_raises_actionable_error(
        self, tmp_path: Path
    ) -> None:
        """A 0-100-scale calibration file used to be silently filtered down
        to 'No valid samples'. It must fail naming the observed range, the
        offending count, and the fresh_draws_data escape hatch."""
        for policy in ("policy_a", "policy_b"):
            _write_policy_file(
                tmp_path, f"{policy}_responses.jsonl", _fresh_records(policy)
            )
        calib_path = tmp_path / "calibration.jsonl"
        calib_records = [
            {"prompt_id": f"c{i}", "judge_score": 2.0 + i * 3.5, "oracle_label": 0.5}
            for i in range(30)
        ]
        calib_path.write_text("\n".join(json.dumps(r) for r in calib_records) + "\n")

        with pytest.raises(ValueError) as excinfo:
            analyze_dataset(
                fresh_draws_dir=str(tmp_path),
                calibration_data_path=str(calib_path),
            )

        message = str(excinfo.value)
        assert "judge_score values outside [0, 1]" in message
        assert "observed range 2.0-103.5" in message
        assert "30 rows" in message
        assert "fresh_draws_data" in message

    def test_out_of_range_oracle_label_raises(self) -> None:
        records = [
            {"prompt_id": f"p{i}", "judge_score": 0.5, "oracle_label": 50.0 + i}
            for i in range(3)
        ]

        with pytest.raises(ValueError) as excinfo:
            DatasetLoader()._convert_raw_data(records)

        message = str(excinfo.value)
        assert "oracle_label values outside [0, 1]" in message
        assert "3 rows" in message
        assert "fresh_draws_data" in message

    def test_in_range_calibration_data_still_loads(self) -> None:
        records = [
            {"prompt_id": f"p{i}", "judge_score": 0.5, "oracle_label": 0.5}
            for i in range(3)
        ]
        dataset = DatasetLoader()._convert_raw_data(records)
        assert dataset.n_samples == 3
