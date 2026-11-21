"""Measure position CTR and document overlap for frequent Qilin fit queries."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from features import _normalize_text
from position_audit import GROUP_LABELS, PAIR_KEY, POSITION_LIMIT, ctr_table, load_exposures, ratio
from slates import _DETAILS, _T1, _T2, _partition_requests, _session


N_BOOTSTRAP = 4000
SEED = 20261121
MIN_BOOTSTRAP_USERS = 10
GRAINS = {"position": tuple(range(1, POSITION_LIMIT + 1)), "group": GROUP_LABELS}


def fit_query_frequencies(train_path: Path, identifiers_path: Path) -> pd.Series:
    """Include every fit request, even when it has no eligible impression."""
    identifiers = json.loads(identifiers_path.read_text(encoding="utf-8"))
    test_sessions = {_session(request) for request in identifiers["rows"]}
    columns = ["query", "user_idx", "session_idx", f"{_DETAILS}.list.element.search_timestamp"]
    with pq.ParquetFile(train_path) as parquet:
        requests = parquet.read(columns=columns).to_pylist()
    fit, _, shared_sessions = _partition_requests(requests, test_sessions)
    queries = pd.Series(
        [_normalize_text(row["query"]) for _, row in fit if _session(row) not in shared_sessions],
        dtype="string",
    )
    return queries.value_counts(sort=False).sort_index().sort_values(ascending=False, kind="stable")


def percentile_intervals(samples: np.ndarray, estimable: np.ndarray) -> np.ndarray:
    bounds = np.full((samples.shape[1], 2), np.nan)
    for column in np.flatnonzero(estimable):
        finite = samples[:, column][np.isfinite(samples[:, column])]
        if len(finite):
            bounds[column] = np.quantile(finite, [0.025, 0.975])
    return bounds


def raw_ctr(
    exposures: pd.DataFrame, column: str, values: tuple, users: pd.Index, draws: np.ndarray
) -> pd.DataFrame:
    counts = ctr_table(exposures, column, values)
    group = exposures.groupby(column)
    counts["requests"] = group["request_key"].nunique().reindex(values, fill_value=0)
    counts["users"] = group["user_id"].nunique().reindex(values, fill_value=0)
    counts["impression_share"] = counts["impressions"] / counts["impressions"].sum()
    counts["click_share"] = counts["clicks"] / counts["clicks"].sum()
    counts["ctr_over_reference"] = counts["ctr"] / counts["ctr"].iloc[0]

    clustered = exposures.groupby(["user_id", column])["click"].agg(impressions="size", clicks="sum")
    n_impressions = clustered["impressions"].unstack(column, fill_value=0).reindex(
        index=users, columns=values, fill_value=0
    )
    n_clicks = clustered["clicks"].unstack(column, fill_value=0).reindex(
        index=users, columns=values, fill_value=0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        samples = (draws @ n_clicks.to_numpy(dtype=float)) / (draws @ n_impressions.to_numpy(dtype=float))
        relative = samples / samples[:, :1]
    estimable = counts["users"].ge(MIN_BOOTSTRAP_USERS) & counts["ctr"].between(0, 1, inclusive="neither")
    counts[["ctr_ci_lower", "ctr_ci_upper"]] = percentile_intervals(samples, estimable.to_numpy())
    counts[["ratio_ci_lower", "ratio_ci_upper"]] = percentile_intervals(
        relative, (estimable & estimable.iloc[0]).to_numpy()
    )
    return counts.reset_index()


def document_mix(
    matchable: pd.DataFrame, column: str, values: tuple
) -> tuple[pd.DataFrame, dict]:
    grouped = (
        matchable.groupby([column, *PAIR_KEY])["click"]
        .agg(impressions="size", clicks="sum", ctr="mean")
        .reset_index()
    )
    per_band = {value: group.set_index(PAIR_KEY) for value, group in grouped.groupby(column)}
    empty = grouped.iloc[:0].set_index(PAIR_KEY)
    reference = per_band.get(values[0], empty)
    rows = []
    for value in values:
        current = per_band.get(value, empty)
        common = reference.index.intersection(current.index)
        matched_reference, matched_current = reference.loc[common], current.loc[common]
        rows.append({
            column: value,
            "query_note_pairs": len(current),
            "reference_query_note_pairs": len(reference),
            "common_query_note_pairs": len(common),
            "query_note_jaccard": ratio(len(common), len(reference) + len(current) - len(common)),
            "matched_reference_impressions": matched_reference["impressions"].sum(),
            "matched_current_impressions": matched_current["impressions"].sum(),
            "matched_reference_clicks": matched_reference["clicks"].sum(),
            "matched_current_clicks": matched_current["clicks"].sum(),
            "matched_reference_equal_pair_ctr": matched_reference["ctr"].mean(),
            "matched_current_equal_pair_ctr": matched_current["ctr"].mean(),
        })
    span = matchable.groupby(PAIR_KEY)[column].nunique()
    return pd.DataFrame(rows), {
        "query_note_pairs_seen": len(span),
        "query_note_pairs_in_multiple_values": int(span.ge(2).sum()),
        "query_note_pairs_in_all_values": int(span.eq(len(values)).sum()),
    }


def count_concentration(exposures: pd.DataFrame) -> dict:
    document_counts = exposures.groupby(PAIR_KEY)["click"].agg(impressions="size", clicks="sum")
    most_shown = document_counts.sort_values("impressions", ascending=False, kind="stable").head(5)
    most_clicked = document_counts["clicks"].nlargest(5)
    total_clicks = int(exposures["click"].sum())
    return {
        "top_5_most_shown_query_note_impression_share": ratio(most_shown["impressions"].sum(), len(exposures)),
        "top_5_most_shown_query_note_click_share": ratio(most_shown["clicks"].sum(), total_clicks),
        "top_5_most_clicked_query_note_click_share": ratio(most_clicked.sum(), total_clicks),
    }


def analyze_cohort(
    name: str, frequencies: pd.Series, exposures: pd.DataFrame, rng: np.random.Generator
) -> tuple[dict, dict[str, pd.DataFrame]]:
    users = pd.Index(exposures["user_id"].unique()).sort_values()
    draws = rng.multinomial(len(users), np.full(len(users), 1 / len(users)), size=N_BOOTSTRAP)
    matchable = exposures.loc[~exposures.duplicated(["request_key", "note_id"], keep=False)]
    summary = {
        "queries": len(frequencies),
        "minimum_fit_requests_per_query": int(frequencies.min()),
        "fit_requests": int(frequencies.sum()),
        "fit_requests_with_eligible_impressions": int(exposures["request_key"].nunique()),
        "users": len(users),
        "impressions": len(exposures),
        "clicks": int(exposures["click"].sum()),
        "impressions_excluded_from_document_matching": len(exposures) - len(matchable),
        **count_concentration(exposures),
    }
    tables = {}
    for column, values in GRAINS.items():
        raw = raw_ctr(exposures, column, values, users, draws)
        overlap, summary[f"{column}_overlap"] = document_mix(matchable, column, values)
        table = raw.merge(overlap, on=column, validate="one_to_one")
        table["matched_current_impression_share"] = table["matched_current_impressions"] / table["impressions"]
        table["matched_reference_impression_share"] = table["matched_reference_impressions"] / table["impressions"].iloc[0]
        table.insert(0, "cohort", name)
        tables[column] = table
    return summary, tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test-identifiers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    exposures, split_summary = load_exposures(args.train, args.test_identifiers)
    frequency_all = fit_query_frequencies(args.train, args.test_identifiers)
    frequency = frequency_all.loc[frequency_all.index != ""]
    top = {"most_frequent_query": frequency.iloc[:1]}
    for limit in (20, 50):
        cutoff = frequency.iloc[min(limit, len(frequency)) - 1]
        top[f"top_{limit}_with_ties"] = frequency.loc[frequency.ge(cutoff)]
    rng = np.random.default_rng(SEED)
    cohort_summaries = {}
    tables = {column: [] for column in GRAINS}
    for name, cohort in top.items():
        subset = exposures.loc[exposures["query"].isin(cohort.index)]
        cohort_summaries[name], cohort_tables = analyze_cohort(name, cohort, subset, rng)
        for column, table in cohort_tables.items():
            tables[column].append(table)

    args.output.mkdir(parents=True, exist_ok=True)
    for column, frames in tables.items():
        pd.concat(frames, ignore_index=True).to_csv(args.output / f"ctr_by_{column}.csv", index=False)
    frequency_distribution = frequency.value_counts(sort=False).sort_index().rename("query_count").to_frame()
    frequency_distribution.index.name = "fit_requests_per_query"
    frequency_distribution["fit_requests"] = frequency_distribution.index * frequency_distribution["query_count"]
    frequency_distribution.to_csv(args.output / "query_frequency_distribution.csv")

    metadata = {
        "dataset": "THUIR/Qilin",
        **split_summary,
        "split_boundaries_utc_unix": [_T1, _T2],
        "inputs": [args.train.name, args.test_identifiers.name],
        "all_fit_impressions": len(exposures),
        "all_fit_clicks": int(exposures["click"].sum()),
        "nonempty_normalized_queries": len(frequency),
        "fit_requests_with_empty_normalized_query": int(frequency_all.get("", 0)),
        "normalized_queries_with_two_or_more_requests": int(frequency.ge(2).sum()),
        "fit_requests_with_a_query_seen_two_or_more_times": int(frequency.loc[frequency.ge(2)].sum()),
        "cohorts": cohort_summaries,
        "cohort_rule": (
            "Rank nonempty exact queries after NFKC, case folding, and removal of whitespace and "
            "Unicode punctuation. Count fit request rows before filtering impressions or using feedback. "
            "The top-20 and top-50 cohorts include ties at their frequency cutoff. "
            "The single most frequent query is unique in this input. Query strings and IDs are omitted."
        ),
        "ctr_rule": "Pool logged impressions at original ranks 1–30 with a note ID. Count click == 1.",
        "intervals": (
            f"95% pointwise percentile intervals over {N_BOOTSTRAP} user-cluster bootstrap draws "
            f"(seed {SEED}). Recompute CTR and its ratio to rank 1 or band 1–2 in the same draw. "
            f"Omit intervals for strata with fewer than {MIN_BOOTSTRAP_USERS} users, no clicks, or no non-clicks, and "
            "omit draws with a zero denominator. Intervals are descriptive, conditional on these cohorts."
        ),
        "document_mix_rule": (
            "For each rank or band, compare the set of (normalized query, note) pairs with rank 1 "
            "or band 1–2. Exclude repeated appearances of a note in the same request from matching. "
            "Reference rows compare the reference with itself. Average matched CTR equally across pairs. "
            "Compare matched impression counts with raw impression totals for coverage. "
            "Click concentration is descriptive and does not enter cohort selection."
        ),
        "interpretation": (
            "Query-only CTR mixes logging position and document relevance, changing document and user "
            "composition, and differences in which requests reach deeper ranks. Query-note overlap "
            "checks document composition; it cannot identify examination or establish equal relevance."
        ),
    }
    (args.output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
