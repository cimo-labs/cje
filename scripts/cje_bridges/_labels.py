"""Response identity and strict label joins shared by the standalone bridges."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping


def finite_number(value: Any, *, context: str) -> float:
    """Parse a label without accepting booleans or nonfinite values."""
    if isinstance(value, bool):
        raise ValueError(f"{context}: expected a finite number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context}: expected a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: expected a finite number, got {value!r}")
    return number


def _identifier(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        # DataFrame missing cells may be NaN, NaT, or pandas.NA. The last
        # cannot be coerced to bool; treat all of them as absent without
        # making pandas a dependency of the other converters.
        if value != value or (isinstance(value, Real) and not math.isfinite(value)):
            return None
    except (TypeError, ValueError):
        return None
    text = str(value).strip()
    return text or None


def first_identifier(*values: Any) -> str | None:
    """Choose the first nonmissing native ID across alternate SDK field names."""
    for value in values:
        identifier = _identifier(value)
        if identifier is not None:
            return identifier
    return None


@dataclass
class ResponseIds:
    """Prefer upstream IDs; disambiguate identical fallback response contents.

    Fallback IDs survive reordering distinct responses. Indistinguishable
    repeated responses need their occurrence number, so their order must be
    preserved when reimporting labels against an export without native IDs.
    """

    counts: Counter = field(default_factory=Counter)
    seen: set[tuple[str, str]] = field(default_factory=set)

    def create(
        self,
        policy: str,
        prompt_id: str,
        response: Any,
        *,
        native_id: Any = None,
    ) -> str:
        response_id = _identifier(native_id)
        if response_id is None:
            serialized = json.dumps(
                [policy, prompt_id, response],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]
            self.counts[(policy, digest)] += 1
            response_id = f"response::{digest}::{self.counts[(policy, digest)]}"
        key = (policy, response_id)
        if key in self.seen:
            raise ValueError(
                f"Duplicate response_id {response_id!r} for policy {policy!r}; "
                "each response needs its own upstream record/run ID."
            )
        self.seen.add(key)
        return response_id


@dataclass
class OracleLabels:
    # None in the response position means the legacy, prompt-only format.
    values: dict[tuple[str, str, str | None], float] = field(default_factory=dict)
    contexts: dict[tuple[str, str, str | None], str] = field(default_factory=dict)

    def apply(
        self,
        fresh_draws: Mapping[str, list[dict[str, Any]]],
        *,
        source_prompt_counts: Mapping[tuple[str, str], int] | None = None,
    ) -> None:
        """Join labels to exact draws; permit legacy keys only if unambiguous."""
        prompt_counts = (
            source_prompt_counts
            if source_prompt_counts is not None
            else Counter(
                (policy, sample["prompt_id"])
                for policy, samples in fresh_draws.items()
                for sample in samples
            )
        )
        matched: set[tuple[str, str, str | None]] = set()
        assignments = []
        for policy, samples in fresh_draws.items():
            for sample in samples:
                prompt_id = sample["prompt_id"]
                exact = (policy, prompt_id, sample["response_id"])
                legacy = (policy, prompt_id, None)
                if legacy in self.values:
                    if prompt_counts[(policy, prompt_id)] != 1:
                        raise ValueError(
                            f"{self.contexts[legacy]}: ambiguous legacy oracle label "
                            f"for {policy!r}, prompt {prompt_id!r}; multiple responses "
                            "exist. Regenerate the template and label response_id rows."
                        )
                    if exact in self.values:
                        raise ValueError(
                            f"{self.contexts[legacy]}: both legacy and response_id "
                            f"labels refer to response {sample['response_id']!r}."
                        )
                    key = legacy
                else:
                    key = exact
                if key in self.values:
                    matched.add(key)
                    assignments.append((sample, self.values[key]))
        unmatched = self.values.keys() - matched
        if unmatched:
            key = next(iter(unmatched))
            raise ValueError(
                f"{self.contexts[key]}: oracle label matches no exported response: "
                f"{key!r}. Reuse the original export and its generated identifiers."
            )
        for sample, label in assignments:
            sample["oracle_label"] = label


def load_oracle_labels(path: Path, policy_fields: Iterable[str]) -> OracleLabels:
    """Read sparse CSV/JSONL labels with location-aware validation errors."""
    labels = OracleLabels()
    policy_fields = tuple(policy_fields)

    def add(row: Any, line: int) -> None:
        context = f"{path}:line {line}"
        if not isinstance(row, dict):
            raise ValueError(f"{context}: expected an oracle-label object")
        raw_label = row.get("oracle_label")
        if raw_label is None or (isinstance(raw_label, str) and not raw_label.strip()):
            return
        label = finite_number(raw_label, context=f"{context}: oracle_label")
        policy = next(
            (
                identifier
                for field_name in policy_fields
                if (identifier := _identifier(row.get(field_name))) is not None
            ),
            None,
        )
        prompt_id = _identifier(row.get("prompt_id"))
        if policy is None or prompt_id is None:
            raise ValueError(f"{context}: a labeled row needs a policy and prompt_id")
        response_id = _identifier(row.get("response_id"))
        key = (policy, prompt_id, response_id)
        if key in labels.values:
            raise ValueError(
                f"{context}: duplicate oracle-label key {key!r}; first seen at "
                f"{labels.contexts[key]}. Keep one label per response_id."
            )
        labels.values[key] = label
        labels.contexts[key] = context

    with path.open(encoding="utf-8", newline="") as stream:
        if path.suffix.lower() == ".jsonl":
            for line_number, line in enumerate(stream, start=1):
                if line.strip():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"{path}:line {line_number}: invalid JSON: {exc.msg}"
                        ) from exc
                    add(row, line_number)
        else:
            reader = csv.DictReader(stream)
            for row in reader:
                add(row, reader.line_num)
    return labels
