"""0.9.2 onboarding: package output and API.

Pins the behaviours a reader meets in printed output and docstrings:

- borrowed calibration made explicit (metadata, summary line, status,
  comparison and best_policy flags, one analyze_dataset WARNING);
- summary()'s paired-difference block (the README quickstart numbers);
- compare_policies by name;
- a float NaN oracle_label in records is unlabeled (one INFO log);
- metadata["cje_version"];
- the three data-quality warnings;
- REFUSE-LEVEL warning text in the judge's own units;
- the analyze_dataset docstring carries the agent rules.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

import cje
from cje import analyze_dataset
from cje.data.models import EstimationResult
from cje.diagnostics import Status
from cje.diagnostics.transport import TransportAuditConfig

pytestmark = pytest.mark.unit

# README quickstart data: production and candidate answer the same 20 prompts;
# only production carries labels (on its first 10 responses).
QUICKSTART_JUDGE = {
    "production": [
        0.62, 0.68, 0.72, 0.76, 0.79, 0.83, 0.85, 0.88, 0.91, 0.95,
        0.64, 0.69, 0.73, 0.77, 0.80, 0.84, 0.87, 0.89, 0.92, 0.94,
    ],
    "candidate": [
        0.70, 0.74, 0.75, 0.78, 0.81, 0.83, 0.86, 0.90, 0.93, 0.94,
        0.72, 0.76, 0.79, 0.80, 0.84, 0.85, 0.88, 0.89, 0.91, 0.95,
    ],
}  # fmt: skip
QUICKSTART_LABELS: List[Optional[float]] = [
    0.55, 0.60, 0.70, 0.74, 0.75, 0.80, 0.90, 0.92, 0.88, 0.97,
    None, None, None, None, None, None, None, None, None, None,
]  # fmt: skip

# The exact README quickstart summary() text the docs must reproduce.
QUICKSTART_SUMMARY = "\n".join(
    [
        "CJE Estimation Results (method: calibrated_direct)",
        ("  candidate   0.824  95% CI [0.766, 0.882]  [borrowed calibratio" "n]"),
        "  production  0.786  95% CI [0.696, 0.876]",
        (
            "candidate: no labels of its own; its estimate and every differen"
            "ce involving it assume production's calibration transfers, which"
            " the CI and p-value do not cover (the true difference can have e"
            "ither sign). Label >=20 random responses of candidate, or audit "
            "transport on a held-out random sample of their responses (size i"
            "t with plan_transport_audits)."
        ),
        "Best by point estimate: candidate (point estimate, not a test)",
        ("Limitations: borrowed calibration (residual transport NOT_CHECKE" "D)"),
        "Paired differences (p unadjusted):",
        (
            "  candidate - production: +0.038  95% CI [-0.027, +0.102]  p=0.2"
            "2  [borrowed calibration: candidate]"
        ),
        (
            "No reliable winner: every paired CI includes 0 (not evidence tha"
            "t they are equal)"
        ),
        "Status: warning",
    ]
)

NO_RELIABLE_WINNER = (
    "No reliable winner: every paired CI includes 0 "
    "(not evidence that they are equal)"
)

BORROWED_LINE = (
    "candidate: no labels of its own; its estimate and every differen"
    "ce involving it assume production's calibration transfers, which"
    " the CI and p-value do not cover (the true difference can have e"
    "ither sign). Label >=20 random responses of candidate, or audit "
    "transport on a held-out random sample of their responses (size i"
    "t with plan_transport_audits)."
)

COLLISION_TEXT = "differ only in case"
REPEATED_TEXT = "Repeated (policy, prompt_id) rows without a row_id"
ONE_POLICY_LABELS_TEXT = "Oracle labels are present on only 1 of"
BORROWED_TEXT = "Borrowed calibration:"


def _quickstart_draws() -> Dict[str, List[Dict[str, Any]]]:
    """The README quickstart records (s2 style: one policy labeled)."""
    return {
        "production": [
            {"prompt_id": f"q{i:02d}", "judge_score": s, "oracle_label": y}
            for i, (s, y) in enumerate(
                zip(QUICKSTART_JUDGE["production"], QUICKSTART_LABELS)
            )
        ],
        "candidate": [
            {"prompt_id": f"q{i:02d}", "judge_score": s}
            for i, s in enumerate(QUICKSTART_JUDGE["candidate"])
        ],
    }


def _both_labelled_draws() -> Dict[str, List[Dict[str, Any]]]:
    """Same prompts, with candidate's first 10 responses labeled too."""
    draws = _quickstart_draws()
    candidate_labels = [0.62, 0.66, 0.68, 0.75, 0.78, 0.80, 0.86, 0.90, 0.92, 0.93]
    for row, label in zip(draws["candidate"], candidate_labels):
        row["oracle_label"] = label
    return draws


def _records(caplog: pytest.LogCaptureFixture, text: str, level: int) -> List[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == level and text in record.getMessage()
    ]


@pytest.fixture(scope="module")
def s2_result() -> EstimationResult:
    return analyze_dataset(fresh_draws_data=_quickstart_draws())


@pytest.fixture(scope="module")
def both_result() -> EstimationResult:
    return analyze_dataset(fresh_draws_data=_both_labelled_draws())


# ---------------------------------------------------------------------------
# borrowed calibration made explicit
# ---------------------------------------------------------------------------


class TestBorrowedCalibration:
    def test_metadata_names_the_unlabeled_policy(
        self, s2_result: EstimationResult
    ) -> None:
        md = s2_result.metadata
        assert md["own_oracle_labels_by_policy"] == {"candidate": 0, "production": 10}
        assert md["calibration_label_sources"] == ["production"]
        assert md["transport_unverified"] == ["candidate"]

    def test_summary_prints_the_named_line_and_warning_status(
        self, s2_result: EstimationResult
    ) -> None:
        lines = s2_result.summary().splitlines()
        assert BORROWED_LINE in lines
        assert lines[-1] == "Status: warning"
        assert s2_result.diagnostics is not None
        statuses = s2_result.diagnostics.status_per_policy or {}
        assert statuses["candidate"] is Status.WARNING

    def test_comparisons_are_conditional_on_transport(
        self, s2_result: EstimationResult
    ) -> None:
        comparison = s2_result.compare_policies("candidate", "production")
        assert comparison["transport_unverified"] == ["candidate"]
        assert comparison["conditional_on_transport"] is True
        # `significant` keeps its meaning: the unadjusted test at alpha.
        assert comparison["significant"] == bool(comparison["p_value"] < 0.05)
        for entry in s2_result.compare_all_policies():
            assert entry["conditional_on_transport"] is True
            assert entry["transport_unverified"] == ["candidate"]

    def test_best_policy_is_not_decision_ready(
        self, s2_result: EstimationResult
    ) -> None:
        verdict = s2_result.best_policy()
        assert verdict.name == "candidate"
        assert verdict.decision_ready is False
        assert "candidate" in verdict.decision_note
        assert "production's calibration" in verdict.decision_note
        assert "plan_transport_audits" in verdict.decision_note

    def test_labeled_winner_ranked_against_unlabeled_runner_up(self) -> None:
        # Shift candidate's judge scores down so production wins: the ranking
        # still depends on candidate's borrowed calibration.
        draws = _quickstart_draws()
        for row in draws["candidate"]:
            row["judge_score"] = round(row["judge_score"] - 0.08, 2)
        result = analyze_dataset(fresh_draws_data=draws)
        verdict = result.best_policy()
        assert verdict.name == "production"
        assert verdict.decision_ready is False
        assert "candidate" in verdict.decision_note

    def test_analyze_dataset_logs_one_named_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="cje"):
            analyze_dataset(fresh_draws_data=_quickstart_draws())
        warnings = _records(caplog, BORROWED_TEXT, logging.WARNING)
        assert len(warnings) == 1
        assert "candidate has no oracle labels of its own" in warnings[0]
        assert "production's calibration" in warnings[0]

    def test_estimates_and_intervals_do_not_change(
        self, s2_result: EstimationResult
    ) -> None:
        # Pinned README numbers: the new output is additive.
        lower, upper = s2_result.confidence_interval()
        assert np.round(s2_result.estimates, 3).tolist() == [0.824, 0.786]
        assert np.round(lower, 3).tolist() == [0.766, 0.696]
        assert np.round(upper, 3).tolist() == [0.882, 0.876]
        assert s2_result.gates["candidate"].flagged is False

    def test_both_labelled_dataset_does_none_of_that(
        self, both_result: EstimationResult, caplog: pytest.LogCaptureFixture
    ) -> None:
        md = both_result.metadata
        assert md["own_oracle_labels_by_policy"] == {
            "candidate": 10,
            "production": 10,
        }
        assert md["calibration_label_sources"] == ["candidate", "production"]
        assert md["transport_unverified"] == []
        text = both_result.summary()
        assert "no labels of its own" not in text
        comparison = both_result.compare_policies("candidate", "production")
        assert comparison["transport_unverified"] == []
        assert comparison["conditional_on_transport"] is False
        verdict = both_result.best_policy()
        # Not borrowed, but the paired CI includes 0 here: a tie is not a
        # decision.
        assert comparison["ci_lower"] < 0 < comparison["ci_upper"]
        assert verdict.decision_ready is False
        assert "is not shown to beat production" in verdict.decision_note
        assert "not evidence the two are equal" in verdict.decision_note
        assert "no labels of its own" not in verdict.decision_note
        with caplog.at_level(logging.WARNING, logger="cje"):
            analyze_dataset(fresh_draws_data=_both_labelled_draws())
        assert not _records(caplog, BORROWED_TEXT, logging.WARNING)
        assert not _records(caplog, ONE_POLICY_LABELS_TEXT, logging.WARNING)

    def test_transport_pass_clears_the_flag_but_other_states_do_not(self) -> None:
        def labeled(n: int = 30) -> List[Dict[str, Any]]:
            return [
                {
                    "prompt_id": f"p{i}",
                    "judge_score": 0.2 + 0.4 * i / (n - 1),
                    "oracle_label": 0.2 + 0.4 * i / (n - 1),
                }
                for i in range(n)
            ]

        def probe(policy: str, n: int = 30) -> List[Dict[str, Any]]:
            return [
                {
                    "prompt_id": f"probe-{policy}-{i}",
                    "judge_score": 0.2 + 0.4 * (i % 20) / 19,
                    "oracle_label": 0.2 + 0.4 * (i % 20) / 19,
                }
                for i in range(n)
            ]

        draws = {"base": labeled(), "audited": labeled(), "ungraded": labeled()}
        for policy in ("audited", "ungraded"):
            for row in draws[policy]:
                row.pop("oracle_label")
        config = TransportAuditConfig(
            probes_by_policy={
                "audited": probe("audited"),
                "ungraded": probe("ungraded"),
            },
            delta_max_by_policy={"audited": 0.05},
        )
        result = analyze_dataset(
            fresh_draws_data=draws,
            estimator_config={"inference_method": "cluster_robust"},
            transport=config,
        )
        audits = result.metadata["transport_audits"]
        assert audits["audited"]["status"] == "PASS"
        assert audits["ungraded"]["status"] == "NOT_GRADED"
        assert result.metadata["own_oracle_labels_by_policy"]["audited"] == 0
        assert result.metadata["transport_unverified"] == ["ungraded"]
        text = result.summary()
        assert "audited: no labels of its own" not in text
        assert "ungraded: no labels of its own" in text
        assert "NOT_GRADED" in text

    def test_external_calibration_data_is_named_as_the_source(
        self, tmp_path: Path
    ) -> None:
        calibration = tmp_path / "labels.jsonl"
        rows = [
            {
                "prompt_id": f"c{i}",
                "judge_score": 0.5 + 0.45 * i / 29,
                "oracle_label": 0.45 + 0.5 * i / 29,
            }
            for i in range(30)
        ]
        calibration.write_text("".join(json.dumps(r) + "\n" for r in rows))
        draws = _quickstart_draws()
        for row in draws["production"]:
            row.pop("oracle_label")
        result = analyze_dataset(
            fresh_draws_data=draws, calibration_data_path=str(calibration)
        )
        assert result.metadata["calibration_label_sources"] == ["calibration_data"]
        assert result.metadata["transport_unverified"] == ["candidate", "production"]
        assert "calibration fit on calibration_data_path" in result.summary()

    def test_uncalibrated_results_have_nothing_to_transport(self) -> None:
        draws = _quickstart_draws()
        for row in draws["production"][2:]:
            row["oracle_label"] = None
        result = analyze_dataset(fresh_draws_data=draws)
        assert result.metadata["calibration_status"] == "UNCALIBRATED"
        assert result.metadata["transport_unverified"] == []
        assert result.metadata["calibration_label_sources"] == []
        # Labels are not used by the raw-judge fallback.
        assert result.metadata["own_oracle_labels_by_policy"]["production"] == 0
        assert result.best_policy().decision_ready is False


# ---------------------------------------------------------------------------
# summary()'s paired-comparison block
# ---------------------------------------------------------------------------


class TestSummaryPairedBlock:
    def test_readme_quickstart_summary_is_pinned(
        self, s2_result: EstimationResult
    ) -> None:
        assert s2_result.summary() == QUICKSTART_SUMMARY

    def test_paired_line_matches_compare_policies(
        self, s2_result: EstimationResult
    ) -> None:
        c = s2_result.compare_policies("candidate", "production")
        line = (
            f"  candidate - production: {c['difference']:+.3f}  95% CI "
            f"[{c['ci_lower']:+.3f}, {c['ci_upper']:+.3f}]  p={c['p_value']:.2f}"
            "  [borrowed calibration: candidate]"
        )
        assert line in s2_result.summary().splitlines()

    def test_no_reliable_winner_only_when_every_ci_includes_zero(self) -> None:
        def result(diff: float) -> EstimationResult:
            return EstimationResult(
                estimates=np.array([0.5 + diff, 0.5]),
                standard_errors=np.array([0.01, 0.01]),
                n_samples_used={"a": 100, "b": 100},
                method="calibrated_direct",
                influence_functions=None,
                diagnostics=None,
                metadata={"target_policies": ["a", "b"]},
            )

        tie = result(0.001).summary().splitlines()
        assert NO_RELIABLE_WINNER in tie
        assert "Best by point estimate: a (point estimate, not a test)" in tie
        clear = result(0.2).summary().splitlines()
        assert not any(line.startswith("No reliable winner") for line in clear)
        assert not any(line.startswith("No decision-ready") for line in clear)
        assert any(line.startswith("  a - b: +0.200  95% CI [") for line in clear)
        assert any(line.endswith("p<0.001") for line in clear)

    def test_three_policies_get_the_bh_note(self) -> None:
        result = EstimationResult(
            estimates=np.array([0.5, 0.52, 0.55]),
            standard_errors=np.array([0.02, 0.02, 0.02]),
            n_samples_used={"a": 50, "b": 50, "c": 50},
            method="calibrated_direct",
            influence_functions=None,
            diagnostics=None,
            metadata={"target_policies": ["a", "b", "c"]},
        )
        lines = result.summary().splitlines()
        pairs = [line for line in lines if " - " in line and "95% CI" in line]
        assert [line.split(":")[0].strip() for line in pairs] == [
            "a - b",
            "a - c",
            "b - c",
        ]
        assert any('compare_all_policies(adjust="bh")' in line for line in lines)

    def test_gate_flagged_pairs_are_marked(self) -> None:
        # A small p-value on a flagged input is not a reliable difference.
        result = analyze_dataset(fresh_draws_data=_refuse_draws(1.0))
        pairs = [
            line
            for line in result.summary().splitlines()
            if line.startswith("  base - candidate: ")
        ]
        assert len(pairs) == 1
        assert pairs[0].endswith(
            "  [gate-flagged: candidate]  [borrowed calibration: candidate]"
        )
        quickstart = analyze_dataset(fresh_draws_data=_quickstart_draws())
        assert "gate-flagged" not in quickstart.summary()

    def test_single_policy_has_no_paired_block(self) -> None:
        result = EstimationResult(
            estimates=np.array([0.5]),
            standard_errors=np.array([0.02]),
            n_samples_used={"a": 50},
            method="calibrated_direct",
            influence_functions=None,
            diagnostics=None,
            metadata={"target_policies": ["a"]},
        )
        assert "Paired differences" not in result.summary()

    def test_unavailable_pairs_are_reported_not_hidden(
        self, s2_result: EstimationResult
    ) -> None:
        restored = EstimationResult.from_dict(s2_result.to_dict(detail="summary"))
        text = restored.summary()
        assert "  candidate - production: unavailable" in text
        assert "No reliable winner" not in text


# ---------------------------------------------------------------------------
# compare_policies by name
# ---------------------------------------------------------------------------


class TestCompareByName:
    def test_names_equal_indices_with_names_attached(
        self, s2_result: EstimationResult
    ) -> None:
        policies = s2_result.target_policies
        i, j = policies.index("candidate"), policies.index("production")
        by_name = s2_result.compare_policies("candidate", "production")
        by_index = s2_result.compare_policies(i, j)
        assert by_name == by_index
        assert by_name["policy1"] == "candidate"
        assert by_name["policy2"] == "production"
        reverse = s2_result.compare_policies("production", "candidate")
        assert reverse["policy1"] == "production"
        assert reverse["difference"] == pytest.approx(-by_name["difference"])
        mixed = s2_result.compare_policies("candidate", j)
        assert mixed == by_index

    def test_bad_policy_arguments_fail_loudly(
        self, s2_result: EstimationResult
    ) -> None:
        with pytest.raises(ValueError, match="Did you mean 'candidate'"):
            s2_result.compare_policies("Candidate", "production")
        with pytest.raises(ValueError, match="not a policy"):
            s2_result.compare_policies("nope", "production")
        with pytest.raises(TypeError, match="integer index or a policy name"):
            s2_result.compare_policies(0.0, 1)  # type: ignore[arg-type]
        with pytest.raises(IndexError, match="out of range"):
            s2_result.compare_policies(0, 2)

    def test_serialized_comparisons_keep_their_names_when_reversed(
        self, s2_result: EstimationResult
    ) -> None:
        payload = s2_result.to_dict(detail="portable")
        stored = payload["pairwise_inference_state"]["comparisons"]["0-1"]
        for key in (
            "policy1",
            "policy2",
            "gate_flagged",
            "transport_unverified",
            "conditional_on_transport",
        ):
            assert key not in stored
        restored = EstimationResult.from_dict(payload)
        reverse = restored.compare_policies(1, 0)
        assert reverse["policy1"] == "production"
        assert reverse["policy2"] == "candidate"
        assert reverse["conditional_on_transport"] is True
        assert restored.metadata["transport_unverified"] == ["candidate"]

    def test_every_comparison_names_its_pair_without_target_policies(
        self,
    ) -> None:
        result = EstimationResult(
            estimates=np.array([0.6, 0.5]),
            standard_errors=np.array([0.02, 0.02]),
            n_samples_used={"a": 50, "b": 50},
            method="calibrated_direct",
            influence_functions=None,
            diagnostics=None,
        )
        comparison = result.compare_policies(0, 1)
        assert (comparison["policy1"], comparison["policy2"]) == ("0", "1")
        # No label provenance recorded: reliance on transport is unknown.
        assert comparison["conditional_on_transport"] is None


# ---------------------------------------------------------------------------
# NaN oracle labels in records are unlabeled
# ---------------------------------------------------------------------------


class TestNaNLabels:
    def _nan_draws(self) -> Dict[str, List[Dict[str, Any]]]:
        draws = _quickstart_draws()
        for row in draws["production"]:
            if row["oracle_label"] is None:
                row["oracle_label"] = float("nan")
        draws["candidate"][0]["oracle_label"] = np.float64("nan")
        return draws

    def test_nan_is_unlabeled_with_one_info_log(
        self, s2_result: EstimationResult, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="cje"):
            result = analyze_dataset(fresh_draws_data=self._nan_draws())
        infos = _records(caplog, "as unlabeled", logging.INFO)
        assert len(infos) == 1
        assert "Treated 11 NaN 'oracle_label' value(s) as unlabeled" in infos[0]
        np.testing.assert_allclose(result.estimates, s2_result.estimates)
        np.testing.assert_allclose(result.standard_errors, s2_result.standard_errors)
        assert result.metadata["own_oracle_labels_by_policy"] == {
            "candidate": 0,
            "production": 10,
        }

    def test_nan_in_jsonl_directory_is_unlabeled(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        for policy, rows in self._nan_draws().items():
            with open(tmp_path / f"{policy}.jsonl", "w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")  # writes bare NaN
        with caplog.at_level(logging.INFO, logger="cje"):
            result = analyze_dataset(fresh_draws_dir=str(tmp_path))
        assert len(_records(caplog, "as unlabeled", logging.INFO)) == 1
        assert result.metadata["own_oracle_labels_by_policy"]["production"] == 10

    @pytest.mark.parametrize("bad", ["nan", "", float("inf"), True, "high"])
    def test_other_invalid_labels_still_raise(self, bad: Any) -> None:
        draws = _quickstart_draws()
        draws["production"][0]["oracle_label"] = bad
        with pytest.raises(ValueError, match="oracle_label"):
            analyze_dataset(fresh_draws_data=draws)

    def test_validate_agrees_with_the_loaders(self) -> None:
        from cje.data.validation import validate_direct_data

        records = [
            {
                "prompt_id": f"p{i}",
                "judge_score": 0.2 + 0.05 * i,
                "oracle_label": 0.3 if i % 2 else float("nan"),
            }
            for i in range(10)
        ]
        _, issues = validate_direct_data(records)
        assert not any("non-numeric" in issue for issue in issues)
        records[1]["oracle_label"] = "nan"
        _, issues = validate_direct_data(records)
        assert any("non-numeric" in issue for issue in issues)

    def test_calibration_file_nan_labels_log_once(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        calibration = tmp_path / "labels.jsonl"
        rows = [
            {
                "prompt_id": f"c{i}",
                "judge_score": 0.5 + 0.45 * i / 39,
                "oracle_label": (0.45 + 0.5 * i / 39) if i % 4 else math.nan,
            }
            for i in range(40)
        ]
        calibration.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with caplog.at_level(logging.INFO, logger="cje"):
            analyze_dataset(
                fresh_draws_data=_quickstart_draws(),
                calibration_data_path=str(calibration),
            )
        infos = _records(caplog, "as unlabeled", logging.INFO)
        assert len(infos) == 1
        assert "Treated 10 NaN 'oracle_label' value(s) as unlabeled" in infos[0]

    def test_transport_probe_nan_label_is_missing_not_a_value(self) -> None:
        draws = _quickstart_draws()
        probe = [{"prompt_id": "x0", "judge_score": 0.8, "oracle_label": math.nan}]
        with pytest.raises(ValueError, match="has no label in oracle field"):
            analyze_dataset(
                fresh_draws_data=draws,
                transport=TransportAuditConfig(probes_by_policy={"candidate": probe}),
            )


# ---------------------------------------------------------------------------
# version in metadata
# ---------------------------------------------------------------------------


def test_metadata_records_the_cje_version(s2_result: EstimationResult) -> None:
    assert s2_result.metadata["cje_version"] == cje.__version__
    restored = EstimationResult.from_dict(s2_result.to_dict())
    assert restored.metadata["cje_version"] == cje.__version__


# ---------------------------------------------------------------------------
# data-quality warnings (no behaviour change)
# ---------------------------------------------------------------------------


class TestDataQualityWarnings:
    def _run(
        self, caplog: pytest.LogCaptureFixture, draws: Dict[str, Any]
    ) -> EstimationResult:
        with caplog.at_level(logging.WARNING, logger="cje"):
            return analyze_dataset(fresh_draws_data=draws)

    def test_clean_input_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        self._run(caplog, _both_labelled_draws())
        for text in (COLLISION_TEXT, REPEATED_TEXT, ONE_POLICY_LABELS_TEXT):
            assert not _records(caplog, text, logging.WARNING), text

    def test_policy_name_collision(self, caplog: pytest.LogCaptureFixture) -> None:
        draws = _both_labelled_draws()
        draws["Production "] = draws.pop("candidate")
        draws["gpt_4"] = [dict(r) for r in draws["production"]]
        draws["GPT-4"] = [dict(r) for r in draws["production"]]
        result = self._run(caplog, draws)
        warnings = _records(caplog, COLLISION_TEXT, logging.WARNING)
        assert len(warnings) == 1
        assert "'GPT-4' vs 'gpt_4'" in warnings[0]
        assert "'Production ' vs 'production'" in warnings[0]
        # Still analyzed as separate policies.
        assert len(result.target_policies) == 4

    def test_repeated_prompt_rows_without_row_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        draws = _both_labelled_draws()
        draws["candidate"].append(dict(draws["candidate"][12]))
        self._run(caplog, draws)
        warnings = _records(caplog, REPEATED_TEXT, logging.WARNING)
        assert len(warnings) == 1
        assert "candidate: 1 prompt_id(s), e.g. 'q12' x2" in warnings[0]

    def test_repeats_with_row_id_or_draw_idx_are_silent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        draws = _both_labelled_draws()
        first = dict(draws["candidate"][12], row_id="r-a")
        second = dict(draws["candidate"][12], row_id="r-b", judge_score=0.81)
        draws["candidate"][12] = first
        draws["candidate"].append(second)
        for index, row in enumerate(draws["production"]):
            row["draw_idx"] = 0
        draws["production"].append(dict(draws["production"][15], draw_idx=1))
        self._run(caplog, draws)
        assert not _records(caplog, REPEATED_TEXT, logging.WARNING)

    def test_labels_on_only_one_policy(self, caplog: pytest.LogCaptureFixture) -> None:
        # Too few labels to fit a calibrator: nothing is borrowed, so the
        # generic warning is the one that names the unlabeled policy.
        draws = _quickstart_draws()
        for row in draws["production"][2:]:
            row["oracle_label"] = None
        self._run(caplog, draws)
        warnings = _records(caplog, ONE_POLICY_LABELS_TEXT, logging.WARNING)
        assert len(warnings) == 1
        assert "1 of 2 policies (production has 2; candidate has none)" in (warnings[0])
        assert not _records(caplog, BORROWED_TEXT, logging.WARNING)

    def test_borrowed_calibration_warning_replaces_the_generic_one(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The quickstart borrows production's calibration: one WARNING names
        # candidate and production, not two saying the same thing.
        self._run(caplog, _quickstart_draws())
        assert len(_records(caplog, BORROWED_TEXT, logging.WARNING)) == 1
        assert not _records(caplog, ONE_POLICY_LABELS_TEXT, logging.WARNING)


# ---------------------------------------------------------------------------
# REFUSE-LEVEL in the judge's own units
# ---------------------------------------------------------------------------


def _refuse_draws(scale: float) -> Dict[str, List[Dict[str, Any]]]:
    rng = np.random.default_rng(0)
    n = 40
    base = rng.uniform(0.2, 0.6, n)
    candidate = rng.uniform(0.6, 1.0, n)
    labels = np.clip(base + rng.normal(0, 0.05, n), 0, 1)
    return {
        "base": [
            {
                "prompt_id": f"p{i}",
                "judge_score": float(base[i] * scale),
                "oracle_label": float(labels[i]),
            }
            for i in range(n)
        ],
        "candidate": [
            {"prompt_id": f"p{i}", "judge_score": float(candidate[i] * scale)}
            for i in range(n)
        ],
    }


@pytest.mark.parametrize("scale", [100.0, 1.0])
def test_refuse_level_prints_judge_units(
    scale: float, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="cje"):
        result = analyze_dataset(fresh_draws_data=_refuse_draws(scale))
    warnings = _records(caplog, "REFUSE-LEVEL for policy 'candidate'", logging.WARNING)
    assert len(warnings) == 1
    lo, hi = result.metadata["boundary_cards"]["candidate"]["oracle_s_range"]
    if scale == 1.0:
        expected = f"[{lo:.3f}, {hi:.3f}] (judge-score units)"
    else:
        assert lo > 1.0  # the card itself is in judge units
        expected = f"[{lo:.4g}, {hi:.4g}] (judge-score units)"
    assert expected in warnings[0]


def test_judge_unit_context_is_scoped() -> None:
    from cje.data.normalization import ScaleInfo
    from cje.diagnostics.reward_boundary import (
        _JUDGE_DISPLAY_SCALE,
        judge_units_for_warnings,
    )

    assert _JUDGE_DISPLAY_SCALE.get() is None
    with judge_units_for_warnings(ScaleInfo(0.0, 100.0)):
        assert _JUDGE_DISPLAY_SCALE.get() is not None
    assert _JUDGE_DISPLAY_SCALE.get() is None


# ---------------------------------------------------------------------------
# the analyze_dataset docstring carries the rules
# ---------------------------------------------------------------------------


def test_analyze_dataset_docstring_carries_the_rules() -> None:
    doc = analyze_dataset.__doc__ or ""
    for needle in (
        "import cje",
        "else None",
        "random",
        "row_id",
        "exactly as given",
        "Borrowed calibration",
        "transport_unverified",
        "plan_transport_audits",
    ):
        assert needle in doc, needle


# ---------------------------------------------------------------------------
# CLI: `cje analyze` prints the same paired block and borrowed lines
# ---------------------------------------------------------------------------


def test_cli_analyze_prints_paired_block_and_borrowed_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from cje.interface.cli import main

    for policy, rows in _quickstart_draws().items():
        with open(tmp_path / f"{policy}.jsonl", "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    assert main(["analyze", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "Best by point estimate: candidate (point estimate, not a test)" in lines
    assert "Paired differences (p unadjusted):" in lines
    assert (
        "  candidate - production: +0.038  95% CI [-0.027, +0.102]  p=0.22"
        "  [borrowed calibration: candidate]"
    ) in lines
    assert NO_RELIABLE_WINNER in lines
    assert BORROWED_LINE in lines


def test_readme_shows_the_real_quickstart_output() -> None:
    """README's printed-output block is the pinned summary() text, verbatim."""
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
    assert "```text\n" + QUICKSTART_SUMMARY + "\n```" in readme


def test_decision_ready_when_the_paired_test_separates_the_winner() -> None:
    rng = np.random.default_rng(7)
    draws = {}
    for policy, shift in (("strong", 0.25), ("weak", 0.0)):
        rows = []
        for i in range(200):
            score = float(np.clip(0.4 + shift + rng.normal(0, 0.08), 0, 1))
            label = float(np.clip(score + rng.normal(0, 0.05), 0, 1))
            rows.append(
                {
                    "prompt_id": f"q{i}",
                    "judge_score": score,
                    "oracle_label": label if i % 4 == 0 else None,
                }
            )
        draws[policy] = rows
    result = analyze_dataset(fresh_draws_data=draws)
    verdict = result.best_policy()
    assert verdict.name == "strong"
    assert verdict.decision_ready is True
    assert "beats weak" in verdict.decision_note


def test_paired_block_is_capped_for_many_policies() -> None:
    rng = np.random.default_rng(3)
    draws = {}
    for k in range(6):
        rows = []
        for i in range(60):
            score = float(np.clip(0.5 + 0.02 * k + rng.normal(0, 0.1), 0, 1))
            label = float(np.clip(score + rng.normal(0, 0.05), 0, 1))
            rows.append(
                {
                    "prompt_id": f"q{i}",
                    "judge_score": score,
                    "oracle_label": label if i % 3 == 0 else None,
                }
            )
        draws[f"p{k}"] = rows
    text = analyze_dataset(fresh_draws_data=draws).summary()
    assert "Paired differences with " in text
    assert "10 more pairs not shown" in text


# ---------------------------------------------------------------------------
# Round 3: borrowed-calibration output that agents cannot read as a lean
# ---------------------------------------------------------------------------


def _summary_result(
    estimates: List[float], unverified: List[str], names: Optional[List[str]] = None
) -> EstimationResult:
    names = names or ["candidate", "production"][: len(estimates)]
    return EstimationResult(
        estimates=np.array(estimates),
        standard_errors=np.full(len(estimates), 0.01),
        n_samples_used={name: 100 for name in names},
        method="calibrated_direct",
        influence_functions=None,
        diagnostics=None,
        metadata={
            "target_policies": names,
            "transport_unverified": unverified,
            "own_oracle_labels_by_policy": {
                name: 0 if name in unverified else 50 for name in names
            },
        },
    )


class TestBorrowedCalibrationOutput:
    def test_separated_borrowed_pair_is_not_a_decision(self) -> None:
        lines = _summary_result([0.70, 0.50], ["candidate"]).summary().splitlines()
        assert lines[1].endswith("  [borrowed calibration]")
        assert "borrowed" not in lines[2]
        pair = next(line for line in lines if line.startswith("  candidate - "))
        assert pair.endswith("p<0.001  [borrowed calibration: candidate]")
        assert (
            "No decision-ready winner: every paired CI that excludes 0 involves "
            "candidate, whose calibration transfer is unverified (see its line "
            "above). Name no winner and no lean until that changes."
        ) in lines
        assert not any(line.startswith("No reliable winner") for line in lines)

    def test_borrowed_line_comes_before_any_ranking(self) -> None:
        lines = _summary_result([0.70, 0.50], ["candidate"]).summary().splitlines()
        borrowed = next(
            i for i, line in enumerate(lines) if line.startswith("candidate: no labels")
        )
        best = next(i for i, line in enumerate(lines) if line.startswith("Best by"))
        assert borrowed < best
        assert lines[best + 1] == (
            "Limitations: borrowed calibration (residual transport NOT_CHECKED)"
        )

    def test_an_unconditional_separation_suppresses_the_line(self) -> None:
        # a and b both have labels and are clearly apart; c borrows.
        result = _summary_result([0.80, 0.50, 0.70], ["c"], names=["a", "b", "c"])
        text = result.summary()
        assert "No decision-ready winner" not in text
        assert "  b - c: " in text and "[borrowed calibration: c]" in text

    def test_decision_note_forbids_a_lean(self) -> None:
        verdict = _summary_result([0.70, 0.50], ["candidate"]).best_policy()
        assert verdict.decision_ready is False
        assert "Name no winner and no lean until that changes" in (
            verdict.decision_note or ""
        )

    def test_own_labelled_results_carry_no_borrowed_markers(self) -> None:
        text = _summary_result([0.70, 0.50], []).summary()
        assert "borrowed" not in text
        assert "No decision-ready winner" not in text


@pytest.mark.parametrize("alpha", [95, 1, 0, -0.1, "0.05", True, None])
def test_ci_rejects_a_confidence_level_or_a_name(alpha: Any) -> None:
    result = _summary_result([0.70, 0.50], [])
    with pytest.raises(ValueError, match="takes a significance level") as excinfo:
        result.ci(alpha)
    assert "dict(zip(result.target_policies, result.ci()))" in str(excinfo.value)


def test_ci_accepts_a_significance_level() -> None:
    result = _summary_result([0.70, 0.50], [])
    assert len(result.ci(0.1)) == 2
    assert len(result.ci(np.float64(0.05))) == 2


def test_analyze_dataset_docstring_points_agents_to_the_skill() -> None:
    doc = " ".join((analyze_dataset.__doc__ or "").split())
    assert doc.startswith(
        "Analyze policies using fresh draws (Direct mode). Coding agents: run "
        "``cje skill``"
    )
    assert "name no winner and no lean" in doc
    assert "is not corroboration" in doc


# ---------------------------------------------------------------------------
# Pre-release review: decision_ready is a tested claim, never a demotion
# ---------------------------------------------------------------------------


def _verdict_result(
    estimates: List[float],
    names: List[str],
    *,
    unverified: Optional[List[str]] = None,
    own: Optional[Dict[str, int]] = None,
    flagged: Optional[List[str]] = None,
    audits: Optional[Dict[str, Dict[str, Any]]] = None,
    se: float = 0.01,
    provenance: bool = True,
) -> EstimationResult:
    metadata: Dict[str, Any] = {
        "target_policies": names,
        "reliability_gates": {
            name: {
                "flagged": name in (flagged or []),
                "refused": False,
                "reasons": ["test flag"] if name in (flagged or []) else [],
            }
            for name in names
        },
    }
    if provenance:
        metadata["transport_unverified"] = list(unverified or [])
        metadata["own_oracle_labels_by_policy"] = own or {n: 50 for n in names}
    if audits:
        metadata["transport_audits"] = audits
    return EstimationResult(
        estimates=np.array(estimates, dtype=float),
        standard_errors=np.full(len(estimates), se),
        n_samples_used={name: 100 for name in names},
        method="calibrated_direct",
        influence_functions=None,
        diagnostics=None,
        metadata=metadata,
    )


class TestDecisionReadiness:
    def test_a_demotion_is_never_decision_ready(self) -> None:
        # The flagged leader is clearly better; the demoted verdict must not
        # be certified as beating it.
        result = _verdict_result([0.70, 0.50], ["good", "worse"], flagged=["good"])
        verdict = result.best_policy()
        assert verdict.name == "worse" and verdict.runner_up == "good"
        assert verdict.decision_ready is False
        assert "point-estimate leader good failed the reliability gates" in (
            verdict.decision_note
        )
        assert "beats" not in verdict.decision_note

    def test_every_other_policy_must_be_beaten(self) -> None:
        # a clearly beats b, but c is within noise of a.
        result = _verdict_result([0.70, 0.50, 0.68], ["a", "b", "c"], se=0.02)
        verdict = result.best_policy()
        assert verdict.name == "a"
        assert verdict.decision_ready is False
        assert "a - c" in verdict.decision_note

    def test_beating_every_other_policy_is_ready(self) -> None:
        result = _verdict_result([0.80, 0.50, 0.55], ["a", "b", "c"])
        verdict = result.best_policy()
        assert verdict.decision_ready is True
        assert verdict.decision_note.startswith(
            "a beats every other policy (closest: c, 95% CI ["
        )

    def test_a_borrowed_third_policy_blocks_the_ranking(self) -> None:
        result = _verdict_result(
            [0.80, 0.50, 0.55],
            ["a", "b", "c"],
            unverified=["c"],
            own={"a": 50, "b": 50, "c": 0},
        )
        verdict = result.best_policy()
        assert verdict.decision_ready is False
        assert "the ranking involves c" in verdict.decision_note

    def test_a_pass_audit_margin_must_be_cleared(self) -> None:
        audits = {"candidate": {"status": "PASS", "delta_max": 0.06}}
        own = {"candidate": 0, "production": 50}
        tight = _verdict_result(
            [0.55, 0.52], ["candidate", "production"], own=own, audits=audits
        )
        verdict = tight.best_policy()
        assert verdict.decision_ready is False
        assert "inside the transport margin (0.060)" in verdict.decision_note
        wide = _verdict_result(
            [0.70, 0.52], ["candidate", "production"], own=own, audits=audits
        )
        assert wide.best_policy().decision_ready is True

    def test_results_without_provenance_are_not_ready(self) -> None:
        result = _verdict_result([0.80, 0.50], ["a", "b"], provenance=False)
        verdict = result.best_policy()
        assert verdict.decision_ready is False
        assert "saved before cje-eval 0.9.2" in verdict.decision_note
        assert "Limitations: residual transport NOT_CHECKED" in result.summary()
        assert result.compare_policies("a", "b")["conditional_on_transport"] is None

    def test_failed_audit_remedy_does_not_invite_a_new_audit(self) -> None:
        result = _verdict_result(
            [0.70, 0.50],
            ["candidate", "production"],
            unverified=["candidate"],
            own={"candidate": 0, "production": 50},
            audits={"candidate": {"status": "FAIL", "delta_max": 0.03}},
        )
        note = result.best_policy().decision_note
        line = result._borrowed_calibration_line("candidate")
        for text in (note, line):
            assert "Transport audit FAILED for candidate" in text
            assert "do not re-audit to escape a FAIL" in text
            assert "audit transport on a held-out" not in text
        assert "unaudited" not in result.summary()


def test_plug_in_route_with_own_labels_is_uncorrected() -> None:
    draws = _quickstart_draws()
    for i, row in enumerate(draws["candidate"]):
        row["oracle_label"] = QUICKSTART_LABELS[i]
    result = analyze_dataset(
        fresh_draws_data=draws,
        estimator_config={"use_augmented_estimator": False},
    )
    assert result.metadata["point_estimator"]["routes"] == ["plug_in", "plug_in"]
    assert result.metadata["transport_unverified"] == ["candidate", "production"]
    text = result.summary()
    assert "[uncorrected calibration]" in text
    assert "its own labels do not correct its estimate (route plug_in)" in text
    assert result.best_policy().decision_ready is False


def test_more_than_ten_pairs_scope_the_closing_line() -> None:
    names = [f"p{i}" for i in range(6)]
    result = _verdict_result(
        [0.90, 0.50, 0.50, 0.50, 0.50, 0.50],
        names,
        unverified=["p0"],
        own={n: (0 if n == "p0" else 50) for n in names},
    )
    lines = result.summary().splitlines()
    assert "Paired differences with p0 (p unadjusted):" in lines
    assert any(
        line.startswith("No decision-ready winner: every paired CI with p0 that")
        for line in lines
    )


@pytest.mark.parametrize("container", ["ndarray", "series"])
def test_array_like_record_containers_still_work(container: str) -> None:
    import pandas as pd

    draws = _quickstart_draws()
    wrapped = {
        policy: (
            np.array(rows, dtype=object) if container == "ndarray" else pd.Series(rows)
        )
        for policy, rows in draws.items()
    }
    expected = analyze_dataset(fresh_draws_data=draws).estimates
    np.testing.assert_allclose(
        # The annotation says lists; array-like containers worked in 0.9.1.
        analyze_dataset(fresh_draws_data=wrapped).estimates,  # type: ignore[arg-type]
        expected,
    )


@pytest.mark.parametrize(
    "p_value,text",
    [
        (0.0496, "p=0.0496"),
        (0.0516, "p=0.052"),
        (0.04996, "p=0.04996"),
        (0.05004, "p=0.05004"),
        (0.05, "p=0.050"),
        (0.031, "p=0.031"),
        (0.0995, "p=0.10"),
        (0.22, "p=0.22"),
        (0.0500004, "p>0.05"),
        (0.0499996, "p<0.05"),
    ],
)
def test_p_values_near_the_threshold_stay_distinguishable(
    p_value: float, text: str
) -> None:
    from cje.data.models import _format_p_value

    assert _format_p_value(p_value) == text


class TestSecondReviewFixes:
    def test_a_lone_unverified_policy_is_not_decision_ready(self) -> None:
        result = _verdict_result(
            [0.70], ["candidate"], unverified=["candidate"], own={"candidate": 0}
        )
        verdict = result.best_policy()
        assert verdict.decision_ready is False
        assert "candidate's calibration transfer is unverified" in (
            verdict.decision_note
        )

    def test_a_difference_within_a_pass_margin_is_marked(self) -> None:
        result = _verdict_result(
            [0.55, 0.52],
            ["candidate", "production"],
            own={"candidate": 0, "production": 50},
            audits={"candidate": {"status": "PASS", "delta_max": 0.06}},
        )
        lines = result.summary().splitlines()
        pair = next(line for line in lines if line.startswith("  candidate - "))
        assert pair.endswith("[within transport margin 0.060]")
        assert any(
            line.startswith("No decision-ready winner: every paired CI that excludes 0")
            and "lies within the transport margin a PASS audit allows" in line
            for line in lines
        )

    def test_zero_standard_error_pairs_are_unavailable(self) -> None:
        result = _verdict_result([1.0, 0.0], ["a", "b"], se=0.0)
        lines = result.summary().splitlines()
        assert "  a - b: unavailable" in lines
        verdict = result.best_policy()
        assert verdict.decision_ready is False
        assert "p=1.00" not in verdict.decision_note

    def test_remedies_follow_the_route(self) -> None:
        def note(route: str) -> str:
            result = _verdict_result(
                [0.70, 0.50],
                ["candidate", "production"],
                unverified=["candidate"],
                own={"candidate": 10, "production": 50},
            )
            result.metadata["point_estimator"] = {"routes": [route, "augmented"]}
            return result._transport_actions(["candidate"])

        assert note("plug_in").startswith("Turn use_augmented_estimator back on")
        targeted = note("plug_in_targeted_unknown")
        assert targeted.startswith("Keep only probability-sampled labels")
        assert "combine_oracle_sources=False" in targeted

    def test_inconclusive_remedy_asks_for_a_new_audit(self) -> None:
        result = _verdict_result(
            [0.70, 0.50],
            ["candidate", "production"],
            unverified=["candidate"],
            own={"candidate": 0, "production": 50},
            audits={"candidate": {"status": "INCONCLUSIVE", "delta_max": 0.03}},
        )
        text = result._transport_actions(["candidate"])
        assert "run a new, pre-sized held-out audit on fresh probes" in text
        assert "extend" not in text

    def test_failed_audit_on_an_own_labelled_leader_points_to_correction(self) -> None:
        result = _verdict_result(
            [0.70, 0.50],
            ["candidate", "production"],
            flagged=["candidate"],
            own={"candidate": 40, "production": 50},
            audits={"candidate": {"status": "FAIL", "delta_max": 0.03}},
        )
        result.metadata["point_estimator"] = {"routes": ["augmented", "augmented"]}
        verdict = result.best_policy()
        assert verdict.decision_ready is False
        assert "re-run without the failed probes in transport=" in (
            verdict.decision_note
        )

    def test_plug_in_warning_is_not_headed_borrowed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        draws = _quickstart_draws()
        for i, row in enumerate(draws["candidate"]):
            row["oracle_label"] = QUICKSTART_LABELS[i]
        with caplog.at_level(logging.WARNING, logger="cje.interface.analysis"):
            analyze_dataset(
                fresh_draws_data=draws,
                estimator_config={"use_augmented_estimator": False},
            )
        messages = [r.getMessage() for r in caplog.records]
        assert any(m.startswith("Unverified calibration transfer:") for m in messages)
        assert not any(m.startswith("Borrowed calibration:") for m in messages)


class TestFinalCheckFixes:
    def _mixed(self, route: str, design: str = "representative") -> EstimationResult:
        result = _verdict_result(
            [0.70, 0.60, 0.50],
            ["A", "B", "C"],
            unverified=["A", "B", "C"],
            own={"A": 60, "B": 60, "C": 0},
        )
        result.metadata["point_estimator"] = {
            "routes": [route, route, "plug_in"],
            "label_design": design,
        }
        return result

    def test_each_policy_gets_its_own_remedy(self) -> None:
        text = self._mixed("plug_in")._transport_actions(["A", "B", "C"])
        assert text.startswith("Label >=20 random responses of C; turn ")
        assert "corrects the estimate of each of A and B" in text
        assert "of each of A, B and C" not in text

    def test_targeted_design_on_a_plug_in_route_gets_the_targeted_remedy(
        self,
    ) -> None:
        text = self._mixed("plug_in", "targeted_unknown")._transport_actions(["A"])
        assert text.startswith("Keep only probability-sampled labels")
        assert "label_design 'representative'" in text

    def test_own_labelled_augmented_pairs_never_carry_a_margin(self) -> None:
        # Saved before 0.9.2: no own-label counts, but the routes say augmented.
        result = _verdict_result(
            [0.55, 0.52],
            ["a", "b"],
            provenance=False,
            audits={"a": {"status": "PASS", "delta_max": 0.06}},
        )
        result.metadata["point_estimator"] = {"routes": ["augmented", "augmented"]}
        assert "within transport margin" not in result.summary()

    def test_a_single_policy_line_does_not_mention_differences(self) -> None:
        result = _verdict_result(
            [0.70], ["candidate"], unverified=["candidate"], own={"candidate": 0}
        )
        line = result._borrowed_calibration_line("candidate")
        assert "which its CI does not cover" in line
        assert "difference" not in line

    def test_a_policy_named_calibration_data_is_a_policy(self) -> None:
        result = _verdict_result([0.70, 0.50], ["calibration_data", "other"])
        result.metadata["calibration_label_sources"] = ["calibration_data"]
        assert result._calibration_source_phrase() == "calibration_data's calibration"
        result.metadata["calibration_data_path"] = "cal.jsonl"
        assert result._calibration_source_phrase() == (
            "the calibration fit on calibration_data_path"
        )
