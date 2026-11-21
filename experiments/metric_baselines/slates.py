"""Build development candidate slates for the fixed Qilin temporal split."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


_T1 = 1_732_726_411
_T2 = 1_732_735_658
_TOP_K = 10
_DETAILS = "search_result_details_with_idx"
_ENGAGEMENT_FIELDS = ("like", "share", "comment")
_TRAIN_COLUMNS = [
    "query",
    "search_idx",
    "user_idx",
    "session_idx",
    *(
        f"{_DETAILS}.list.element.{field}"
        for field in ("search_timestamp", "position", "note_idx", "click", "collect", "page_time")
    ),
]
_DTYPES = {
    "query": "string",
    "request_id": "int64",
    "user_id": "int64",
    "note_id": "int64",
    "position": "int64",
    "click": "int64",
    "collect": "int64",
    "page_time": "float64",
    "request_key": "int64",
}
_ENGAGEMENT_DTYPES = {
    **dict.fromkeys(_ENGAGEMENT_FIELDS, "int64"),
    "session_id": "Int64",
    "request_start": "float64",
    "request_end": "float64",
}


def _session(request):
    return request["user_idx"], request["session_idx"]


def _request_timestamps(request):
    return [
        exposure["search_timestamp"]
        for exposure in request[_DETAILS] or []
        if exposure is not None and exposure["search_timestamp"] is not None
    ]


def _partition_requests(requests, test_sessions):
    fit, validation = [], []
    for request_key, request in enumerate(requests):
        if _session(request) in test_sessions:
            continue
        timestamps = _request_timestamps(request)
        if not timestamps:
            continue

        end = max(timestamps)
        if end < _T1:
            fit.append((request_key, request))
        elif end < _T2 and min(timestamps) >= _T1:
            validation.append((request_key, request))

    shared_sessions = {_session(request) for _, request in fit} & {
        _session(request) for _, request in validation
    }
    return fit, validation, shared_sessions


def _candidate_rows(request, depth, request_key, *, include_engagement=False):
    candidates = {}
    engagement = {}
    positions = range(1, depth + 1)
    for exposure in request[_DETAILS] or []:
        if exposure is None:
            continue
        position, note_id = exposure["position"], exposure["note_idx"]
        if position not in positions or note_id is None:
            continue

        old_position, old_click, old_collect, old_time = candidates.get(note_id, (depth, 0, 0, -1.0))
        click = int(exposure["click"] == 1)
        page_time = exposure["page_time"]
        if not click or page_time is None or not isfinite(page_time) or page_time < 0:
            page_time = -1.0
        candidates[note_id] = (
            min(int(position), old_position),
            old_click | click,
            old_collect | int(exposure["collect"] == 1),
            max(page_time, old_time),
        )
        if include_engagement:
            engagement[note_id] = tuple(
                old_value | int(exposure[field] == 1)
                for field, old_value in zip(_ENGAGEMENT_FIELDS, engagement.get(note_id, (0, 0, 0)))
            )

    if include_engagement:
        timestamps = _request_timestamps(request)
        request_metadata = (request["session_idx"], min(timestamps), max(timestamps))

    ordered = sorted(candidates.items(), key=lambda item: (item[1][0], item[0]))
    for note_id, (position, click, collect, page_time) in ordered[:depth]:
        row = (
            request["query"] or "",
            request["search_idx"],
            request["user_idx"],
            note_id,
            position,
            click,
            collect,
            page_time,
            request_key,
        )
        if include_engagement:
            row = (*row, *engagement[note_id], *request_metadata)
        yield row


def load_slates(
    train_path: Path,
    test_identifiers_path: Path,
    depth: int = _TOP_K,
    *,
    include_engagement: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return fit and validation rows with contiguous requests in original candidate order."""
    return load_slates_at_depths(
        train_path, test_identifiers_path, [depth], include_engagement=include_engagement
    )[depth]


def load_slates_at_depths(
    train_path: Path,
    test_identifiers_path: Path,
    depths: list[int],
    *,
    include_engagement: bool = False,
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """Reuse the temporal partition across depths, optionally including engagement and request bounds."""
    test_identifiers = json.loads(test_identifiers_path.read_text(encoding="utf-8"))
    test_sessions = {_session(request) for request in test_identifiers["rows"]}

    columns = _TRAIN_COLUMNS
    dtypes = _DTYPES
    if include_engagement:
        columns = [*_TRAIN_COLUMNS, *(f"{_DETAILS}.list.element.{field}" for field in _ENGAGEMENT_FIELDS)]
        dtypes = {**_DTYPES, **_ENGAGEMENT_DTYPES}

    requests = pq.ParquetFile(train_path).read(columns=columns).to_pylist()
    fit, validation, shared_sessions = _partition_requests(requests, test_sessions)

    def frame(partition, depth):
        rows = (
            row
            for request_key, request in partition
            if _session(request) not in shared_sessions
            for row in _candidate_rows(request, depth, request_key, include_engagement=include_engagement)
        )
        return pd.DataFrame(rows, columns=list(dtypes)).astype(dtypes)

    return {depth: (frame(fit, depth), frame(validation, depth)) for depth in depths}
