"""Training labels based on brief clicks followed by a similar search."""

import numpy as np
import pandas as pd

from features import _normalize_text, _text_signals
from slates import _T1


MAX_REQUERY_SECONDS = 60
MAX_BOUNCE_SECONDS = 3
MIN_BIGRAM_JACCARD = 0.50
MIN_SHORTER_QUERY_COVERAGE = 0.75
MIN_SHARED_BIGRAMS = 2

SESSION_KEYS = ["user_id", "session_id"]
POSITIVE_ACTIONS = ["like", "collect", "share", "comment"]


def _similar_query(query, successor):
    current, following = _normalize_text(query), _normalize_text(successor)
    if not current or not following:
        return False
    if current == following:
        return True

    current_bigrams = _text_signals(current).bigrams
    following_bigrams = _text_signals(following).bigrams
    shared = len(current_bigrams & following_bigrams)
    if shared < MIN_SHARED_BIGRAMS:
        return False
    return (
        shared / len(current_bigrams | following_bigrams) >= MIN_BIGRAM_JACCARD
        and shared / min(len(current_bigrams), len(following_bigrams)) >= MIN_SHORTER_QUERY_COVERAGE
    )


def fast_requery_after_bounce(frame: pd.DataFrame) -> np.ndarray:
    """Return a training-target mask in the frame's original candidate order.

    Supply complete training slates loaded with engagement metadata. The next
    request is the immediate successor represented in these slates for the same
    user and session. Bounds are first/last logged exposure timestamps. They
    serve as a proxy for search timing, without click, return, or query creation
    timestamps to attribute the later search to an individual candidate.
    """
    columns = ["request_key", *SESSION_KEYS, "request_start", "request_end", "query"]
    requests = frame.loc[frame["request_end"].lt(_T1), columns].drop_duplicates("request_key")
    requests = requests.sort_values([*SESSION_KEYS, "request_start", "request_key"])
    requests["tied_start"] = requests.duplicated([*SESSION_KEYS, "request_start"], keep=False)
    successor = requests.groupby(SESSION_KEYS, sort=False)[["request_start", "query", "tied_start"]].shift(-1)

    gap = successor["request_start"] - requests["request_end"]
    ordered = ~requests["tied_start"] & successor["tied_start"].eq(False)
    close = ordered & gap.gt(0) & gap.le(MAX_REQUERY_SECONDS)
    current_queries = requests.loc[close, "query"]
    next_queries = successor.loc[close, "query"]
    near_keys = [
        request_key
        for request_key, query, successor_query in zip(
            requests.loc[close, "request_key"], current_queries, next_queries
        )
        if _similar_query(query, successor_query)
    ]

    short_click = frame["click"].eq(1) & frame["page_time"].gt(0) & frame["page_time"].le(MAX_BOUNCE_SECONDS)
    strong_action = frame[POSITIVE_ACTIONS].eq(1).any(axis=1)
    return (short_click & ~strong_action & frame["request_key"].isin(near_keys)).to_numpy(dtype=bool)
