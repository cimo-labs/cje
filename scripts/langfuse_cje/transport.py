"""GET-only Langfuse export; separate from CJE's pure converter."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Iterator

from cje.bridges.langfuse import identifier, validate_selection


def request_with_retry(client: Any, method: str, path: str, **kwargs: Any) -> Any:
    """Honor rate limits on read requests; never accept writes."""
    if method != "GET":
        raise ValueError("The exporter only permits GET requests")
    for attempt in range(5):
        response = client.request(method, path, **kwargs)
        if response.status_code != 429 or attempt == 4:
            return response
        try:
            delay = float(response.headers["Retry-After"])
            if not math.isfinite(delay) or delay < 0:
                raise ValueError("Invalid retry delay")
        except (KeyError, ValueError):
            delay = 5 * 2**attempt
        # A longer server delay must fail visibly, never be shortened and retried early.
        if delay > 300:
            return response
        remaining = max(1, math.ceil(delay))
        print(
            f"Langfuse rate limit: waiting {remaining}s before retrying {path}",
            flush=True,
        )
        while remaining:
            interval = min(30, remaining)
            time.sleep(interval)
            remaining -= interval
    raise RuntimeError("Unreachable retry state")


def pages(
    client: Any, path: str, params: dict, max_pages: int = 1000
) -> Iterator[dict]:
    """Exhaust cursor pages; never treat a limit or repeated cursor as complete."""
    cursor, seen = None, set()
    for _ in range(max_pages):
        query = {**params, "limit": 100}
        if cursor is not None:
            query["cursor"] = cursor
        response = request_with_retry(client, "GET", path, params=query)
        response.raise_for_status()
        page = response.json()
        if not isinstance(page.get("data"), list) or not isinstance(
            page.get("meta"), dict
        ):
            raise TypeError(f"{path}: invalid pagination envelope")
        yield from page["data"]
        cursor = page["meta"].get("cursor")
        if cursor is None:
            return
        identifier(cursor, "pagination cursor")
        if cursor in seen:
            raise ValueError(f"{path}: repeated pagination cursor")
        seen.add(cursor)
    raise ValueError(f"{path}: page limit reached; export incomplete")


def collect(client: Any, config: dict) -> dict:
    """Fetch items and all scores for their traces, retaining raw JSON types."""
    validate_selection(config)
    start, end = (
        datetime.fromisoformat(config[k].replace("Z", "+00:00"))
        for k in ("from_start_time", "to_start_time")
    )
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("Use an increasing, timezone-aware experiment time window")
    policies = config["policies"]
    if len(policies) != 2 or len(set(policies.values())) != 2:
        raise ValueError("This importer needs two distinct experiment IDs")
    items: dict[str, list[dict]] = {}
    scores: list[dict] = []
    for policy, experiment_id in policies.items():
        identifier(policy, "policy")
        identifier(experiment_id, "experiment ID")
        if "," in experiment_id:
            raise ValueError("Each policy must select one experiment ID")
        items[policy] = list(
            pages(
                client,
                "api/public/experiment-items",
                {
                    "fromStartTime": start.isoformat(),
                    "toStartTime": end.isoformat(),
                    "experimentId": experiment_id,
                    "fields": "core,dataset,io",
                },
            )
        )
        if not items[policy]:
            raise ValueError(f"{policy}: no experiment items in the selected window")
        if any(item.get("experimentId") != experiment_id for item in items[policy]):
            raise ValueError(f"{policy}: server returned a different experiment")
    traces = {
        identifier(i.get("traceId"), "traceId") for rows in items.values() for i in rows
    }
    for trace_id in traces:
        if "," in trace_id:
            raise ValueError("A trace ID cannot contain a filter separator")
    trace_ids = sorted(traces)
    for offset in range(0, len(trace_ids), 50):
        # Embedded experiment scores are capped at 50. Paginate Scores v3 instead.
        # Do not use the run's time window: human annotations may be added later.
        scores.extend(
            pages(
                client,
                "api/public/v3/scores",
                {
                    "traceId": ",".join(trace_ids[offset : offset + 50]),
                    "fields": "subject,details,annotation",
                },
            )
        )
    return {
        "schema": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "selection": config,
        "items": items,
        "scores": scores,
    }


def collect_dataset(client: Any, config: dict) -> dict:
    """Read all pages at a caller-declared version using the SDK's read endpoints."""
    from urllib.parse import quote
    from cje.bridges.langfuse import timestamp

    name = identifier(config.get("dataset_name"), "dataset name")
    version = timestamp(config.get("dataset_version"), "frozen dataset version")
    response = request_with_retry(
        client, "GET", f"api/public/v2/datasets/{quote(name, safe='')}"
    )
    response.raise_for_status()
    if response.json().get("id") != config["dataset_id"]:
        raise ValueError("Dataset name does not resolve to the configured dataset ID")
    items: list[dict] = []
    for page in range(1, 1001):
        response = request_with_retry(
            client,
            "GET",
            "api/public/dataset-items",
            params={
                "datasetName": name,
                "version": version.isoformat(),
                "limit": 100,
                "page": page,
            },
        )
        response.raise_for_status()
        envelope = response.json()
        meta = envelope.get("meta", {})
        total_pages = meta.get("totalPages")
        if (
            not isinstance(envelope.get("data"), list)
            or isinstance(total_pages, bool)
            or not isinstance(total_pages, int)
            or total_pages < 0
            or meta.get("page") != page
        ):
            raise ValueError("Invalid frozen dataset pagination envelope")
        items.extend(envelope["data"])
        if page >= total_pages:
            return {
                "schema": 1,
                "source": "Langfuse dataset-items API at the pinned version",
                "project_id": config["project_id"],
                "dataset_id": config["dataset_id"],
                "requested_version": config["dataset_version"],
                "items": items,
            }
    raise ValueError("Frozen dataset page limit reached; export incomplete")
