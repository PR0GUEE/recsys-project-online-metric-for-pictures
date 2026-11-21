"""Character-level query and note features for the metric baselines."""

from collections import Counter
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import NamedTuple
from unicodedata import category, normalize

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


CONTENT_PREFIX_LENGTH = 256
NOTE_COLUMNS = ("note_idx", "note_title", "note_content", "note_type")
TEXT_FEATURE_NAMES = (
    "title_unigram_jaccard",
    "title_query_unigram_coverage",
    "title_bigram_jaccard",
    "title_query_bigram_coverage",
    "query_in_title",
    "content_prefix_unigram_jaccard",
    "content_prefix_query_unigram_coverage",
    "content_prefix_bigram_jaccard",
    "content_prefix_query_bigram_coverage",
    "query_in_content_prefix",
    "normalized_title_length",
    "normalized_content_prefix_length",
    "note_text_missing",
)


class TextSignals(NamedTuple):
    text: str
    unigrams: set[str]
    bigrams: set[tuple[str, str]]


def _normalize_text(text: str | None) -> str:
    if not isinstance(text, str):
        return ""
    return "".join(
        char
        for char in normalize("NFKC", text).casefold()
        if not char.isspace() and not category(char).startswith("P")
    )


def _text_signals(normalized_text: str) -> TextSignals:
    return TextSignals(
        normalized_text,
        set(normalized_text),
        set(zip(normalized_text, normalized_text[1:])),
    )


def _overlap(query: TextSignals, document: TextSignals) -> tuple[float, ...]:
    shared_unigrams = len(query.unigrams & document.unigrams)
    shared_bigrams = len(query.bigrams & document.bigrams)
    unigram_union = len(query.unigrams) + len(document.unigrams) - shared_unigrams
    bigram_union = len(query.bigrams) + len(document.bigrams) - shared_bigrams
    return (
        shared_unigrams / unigram_union if unigram_union else 0.0,
        shared_unigrams / len(query.unigrams) if query.unigrams else 0.0,
        shared_bigrams / bigram_union if bigram_union else 0.0,
        shared_bigrams / len(query.bigrams) if query.bigrams else 0.0,
        bool(query.text and query.text in document.text),
    )


def _candidate_notes(
    paths: list[Path], note_ids: list[int]
) -> Iterator[tuple[int, str | None, str | None, object]]:
    if not note_ids:
        return
    candidate_ids = pa.array(note_ids)
    for path in paths:
        with pq.ParquetFile(path) as parquet:
            id_type = parquet.schema_arrow.field("note_idx").type
            shard_candidate_ids = candidate_ids.cast(id_type)
            for batch in parquet.iter_batches(columns=list(NOTE_COLUMNS)):
                matches = pc.is_in(batch.column("note_idx"), value_set=shard_candidate_ids)
                columns = batch.filter(matches).to_pydict()
                yield from zip(*(columns[column] for column in NOTE_COLUMNS))


def build_features(
    train: pd.DataFrame, val: pd.DataFrame, notes_paths: list[Path]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return float32 feature matrices in the original row order of both splits.

    Content features only use the first 256 raw characters. The title falls back
    to that prefix if it normalizes to empty. Missing notes get empty text.
    Jaccard divides shared distinct n-grams by their union. Query coverage
    divides by the distinct query n-grams. Empty denominators give zero.
    Integer note types keep the source code with -1 for missing. Other types use
    one-hot columns for up to two categories, selected by counts of distinct
    training notes of each type with lexical tie-breaking. Missing, rare and
    unseen categories get zeros in those columns.
    """
    columns = ["query", "note_id"]
    train_size = len(train)
    pairs = pd.concat([train[columns], val[columns]], ignore_index=True)
    note_rows = pairs.groupby("note_id", sort=False).indices
    queries = pairs["query"].to_numpy()
    text_features = np.zeros((len(pairs), len(TEXT_FEATURE_NAMES)), dtype=np.float32)
    text_features[:, -1] = 1.0
    feature_names = list(TEXT_FEATURE_NAMES)

    integer_note_type = True
    if notes_paths:
        with pq.ParquetFile(notes_paths[0]) as notes:
            note_type_schema = notes.schema_arrow.field("note_type").type
            integer_note_type = pa.types.is_integer(note_type_schema)
    if integer_note_type:
        note_types = np.full(len(pairs), -1, dtype=np.float32)
    else:
        note_types = np.full(len(pairs), None, dtype=object)
    train_type_counts: Counter[str] = Counter()

    @cache
    def query_signals(text: str | None) -> TextSignals:
        return _text_signals(_normalize_text(text))

    candidate_notes = _candidate_notes(notes_paths, list(note_rows))
    for note_id, title, content, note_type in candidate_notes:
        rows = note_rows.pop(note_id, None)
        if rows is None:
            continue
        normalized_content = _normalize_text((content or "")[:CONTENT_PREFIX_LENGTH])
        normalized_title = _normalize_text(title) or normalized_content
        if normalized_title or normalized_content:
            content_signals = _text_signals(normalized_content)
            title_signals = (
                content_signals
                if normalized_title == normalized_content
                else _text_signals(normalized_title)
            )
            for row in rows:
                query = query_signals(queries[row])
                text_features[row] = (
                    *_overlap(query, title_signals),
                    *_overlap(query, content_signals),
                    len(normalized_title),
                    len(normalized_content),
                    0.0,
                )
        if note_type is not None:
            if integer_note_type:
                note_types[rows] = note_type
            else:
                type_name = str(note_type)
                note_types[rows] = type_name
                if rows[0] < train_size:
                    train_type_counts[type_name] += 1

    if integer_note_type:
        feature_names.append("note_type_code")
        features = np.column_stack((text_features, note_types))
    else:
        categories = sorted(
            train_type_counts, key=lambda key: (-train_type_counts[key], key)
        )[:2]
        feature_names.extend(f"note_type={value}" for value in categories)
        encoded_types = [(note_types == value).astype(np.float32) for value in categories]
        features = np.column_stack((text_features, *encoded_types))
    return features[:train_size], features[train_size:], feature_names
