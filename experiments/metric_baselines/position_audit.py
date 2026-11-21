"""Audit Qilin's logged position effects on the fit partition through rank 30."""

import argparse
from collections.abc import Sequence
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from features import _normalize_text
from slates import _DETAILS, _T1, _T2, _partition_requests, _session


POSITION_LIMIT = 30
GROUP_EDGES = (0, 2, 5, 10, 20, 30)
GROUP_LABELS = ("1-2", "3-5", "6-10", "11-20", "21-30")
POSITION_TO_GROUP = {
    position: label
    for lower, upper, label in zip(GROUP_EDGES[:-1], GROUP_EDGES[1:], GROUP_LABELS)
    for position in range(lower + 1, upper + 1)
}
PAIR_KEY = ["query", "note_id"]
EXPOSURE_COLUMNS = ["search_timestamp", "position", "note_idx", "click"]
WEIGHT_CAP = 3.0
ETAS = (0.0, 0.25, 0.5)


def load_exposures(train_path: Path, identifiers_path: Path) -> tuple[pd.DataFrame, dict]:
    identifiers = json.loads(identifiers_path.read_text(encoding="utf-8"))
    test_sessions = {_session(request) for request in identifiers["rows"]}
    columns = [
        "user_idx",
        "session_idx",
        "search_idx",
        "query",
        *(f"{_DETAILS}.list.element.{field}" for field in EXPOSURE_COLUMNS),
    ]
    with pq.ParquetFile(train_path) as parquet:
        requests = parquet.read(columns=columns).to_pylist()
    train, _, shared_sessions = _partition_requests(requests, test_sessions)
    fit = [
        (request_key, request)
        for request_key, request in train
        if _session(request) not in shared_sessions
    ]

    rows = []
    for request_key, request in fit:
        query = _normalize_text(request["query"])
        for exposure in request[_DETAILS] or []:
            if exposure is None:
                continue
            position, note = exposure["position"], exposure["note_idx"]
            if position in POSITION_TO_GROUP and note is not None:
                rows.append(
                    (
                        request_key,
                        request["user_idx"],
                        query,
                        note,
                        int(position),
                        int(exposure["click"] == 1),
                    )
                )
    exposures = pd.DataFrame(
        rows, columns=["request_key", "user_id", *PAIR_KEY, "position", "click"]
    )
    exposures["group"] = exposures["position"].map(POSITION_TO_GROUP)
    return exposures, {
        "dataset_revision": identifiers.get("revision"),
        "fit_request_rows": len(fit),
        "fit_requests_with_eligible_exposures": int(exposures["request_key"].nunique()),
        "fit_users_with_eligible_exposures": int(exposures["user_id"].nunique()),
        "cached_test_session_keys_excluded": len(test_sessions),
        "shared_fit_validation_session_keys_excluded": len(shared_sessions),
    }


def ctr_table(
    exposures: pd.DataFrame, column: str, ordered_values: Sequence[int | str]
) -> pd.DataFrame:
    counts = (
        exposures.groupby(column)["click"]
        .agg(impressions="size", clicks="sum")
        .reindex(pd.Index(ordered_values, name=column), fill_value=0)
    )
    counts["ctr"] = counts["clicks"] / counts["impressions"]
    return counts


def ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def match_groups(
    exposures: pd.DataFrame, raw_groups: pd.DataFrame, same_user: bool = False
) -> tuple[list[dict], dict]:
    """Give every eligible query-note pair equal weight in both bands."""
    keys = [*PAIR_KEY, "user_id"] if same_user else PAIR_KEY
    grouped = (
        exposures.groupby([*keys, "group"])["click"]
        .agg(ctr="mean", clicks="sum", impressions="size")
        .unstack("group")
        .reindex(
            columns=pd.MultiIndex.from_product(
                [("ctr", "clicks", "impressions"), GROUP_LABELS], names=[None, "group"]
            )
        )
    )
    support = grouped["impressions"].notna()
    rows = []
    for low, high in combinations(GROUP_LABELS, 2):
        # Each appearance of a note within a request is unique in this input.
        # Eligibility uses appearances in both bands before any clicks are read.
        matched = grouped.loc[support[low] & support[high]]
        ctr_low, ctr_high = matched["ctr"][[low, high]].mean()
        raw_low, raw_high = raw_groups["ctr"].loc[[low, high]]
        raw_ratio = ratio(raw_high, raw_low)
        matched_ratio = ratio(ctr_high, ctr_low)
        correction = None
        if matched_ratio is not None:
            correction = min(1 / matched_ratio, WEIGHT_CAP) if matched_ratio else WEIGHT_CAP
        rows.append({
            "match_unit": "query_note_user" if same_user else "query_note",
            "earlier_group": low,
            "later_group": high,
            "matched_units": len(matched),
            "matched_query_notes": (
                matched.index.droplevel("user_id").nunique() if same_user else len(matched)
            ),
            "matched_queries": matched.index.get_level_values("query").nunique(),
            "matched_units_with_zero_clicks_in_both_groups": int(
                (matched["clicks"][[low, high]].sum(axis=1) == 0).sum()
            ),
            "matched_earlier_impressions": int(matched["impressions"][low].sum()),
            "matched_later_impressions": int(matched["impressions"][high].sum()),
            "matched_earlier_clicks": int(matched["clicks"][low].sum()),
            "matched_later_clicks": int(matched["clicks"][high].sum()),
            "raw_earlier_ctr": raw_low,
            "raw_later_ctr": raw_high,
            "raw_later_minus_earlier_ctr": raw_high - raw_low,
            "raw_later_over_earlier_ctr": raw_ratio,
            "matched_earlier_ctr": ctr_low,
            "matched_later_ctr": ctr_high,
            "matched_later_minus_earlier_ctr": ctr_high - ctr_low,
            "matched_later_over_earlier_ctr": matched_ratio,
            "clipped_inverse_matched_ratio": correction,
            "raw_later_over_earlier_ctr_times_clipped_inverse": (
                raw_ratio * correction if raw_ratio is not None and correction is not None else None
            ),
        })
    overlap = {
        "units_seen": len(support),
        "units_in_two_or_more_groups": int(support.sum(axis=1).ge(2).sum()),
        "units_in_all_five_groups": int(support.all(axis=1).sum()),
    }
    return rows, overlap


def assumed_sensitivity(raw_positions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply theoretical position weights as a separate mechanical sensitivity."""
    scenarios = []
    for eta in ETAS:
        scenario = raw_positions.reset_index().copy()
        scenario["eta"] = eta
        scenario["group"] = scenario["position"].map(POSITION_TO_GROUP)
        scenario["assumed_e"] = np.power(scenario["position"].to_numpy(dtype=float), -eta)
        scenario["weight"] = (1 / scenario["assumed_e"]).clip(upper=WEIGHT_CAP)
        scenario["weighted_click_mass"] = scenario["clicks"] * scenario["weight"]
        scenarios.append(scenario)
    positions = pd.concat(scenarios, ignore_index=True)
    groups = positions.groupby(["eta", "group"], sort=False, as_index=False).agg(
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        weighted_click_mass=("weighted_click_mass", "sum"),
    )
    groups["weighted_click_mass_share"] = (
        groups["weighted_click_mass"] / groups.groupby("eta")["weighted_click_mass"].transform("sum")
    )
    groups["weighted_click_mass_per_impression"] = (
        groups["weighted_click_mass"] / groups["impressions"]
    )
    return positions, groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test-identifiers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    exposures, population = load_exposures(args.train, args.test_identifiers)
    raw_positions = ctr_table(exposures, "position", range(1, POSITION_LIMIT + 1))
    raw_groups = ctr_table(exposures, "group", GROUP_LABELS)
    empty_query = exposures["query"].eq("")
    ambiguous_request_note = exposures.duplicated(["request_key", "note_id"], keep=False)
    match_input = exposures.loc[~empty_query & ~ambiguous_request_note]
    contrasts, overlap = match_groups(match_input, raw_groups)
    within_user, within_user_overlap = match_groups(match_input, raw_groups, same_user=True)
    weights, sensitivity = assumed_sensitivity(raw_positions)

    args.output.mkdir(parents=True, exist_ok=True)
    raw_positions.to_csv(args.output / "raw_ctr_by_position.csv")
    raw_groups.to_csv(args.output / "raw_ctr_by_group.csv")
    pd.DataFrame(contrasts + within_user).to_csv(args.output / "matched_contrasts.csv", index=False)
    weights.to_csv(args.output / "assumed_weights_by_position.csv", index=False)
    sensitivity.to_csv(args.output / "assumed_sensitivity_by_group.csv", index=False)
    metadata = {
        "dataset": "THUIR/Qilin",
        **population,
        "split_boundaries_utc_unix": [_T1, _T2],
        "raw_exposures": len(exposures),
        "raw_clicks": int(exposures["click"].sum()),
        "raw_position_rule": "Use original positions 1–30, a non-null note ID, and click == 1.",
        "match_eligible_exposures": len(match_input),
        "exposures_with_empty_normalized_query": int(empty_query.sum()),
        "exposures_with_repeated_note_in_same_request": int(ambiguous_request_note.sum()),
        "query_normalization": "NFKC, casefold, remove whitespace and Unicode punctuation.",
        "matching": (
            "A normalized query and note must appear in different requests in both bands. "
            "Exclude all repeated note appearances within a request from matching to preserve "
            "unambiguous position. For each contrast average CTR first within each matching "
            "unit and band, then average units equally across both bands."
        ),
        "query_note_overlap": overlap,
        "query_note_user_overlap": within_user_overlap,
        "interpretation": (
            "The matched units can differ between contrasts. User mix, time, personalization, "
            "and request context can still confound the result. Matching does not identify "
            "a position propensity or control the latent relevance of each exposure."
        ),
        "empirical_weight_illustration": (
            "For each band pair separately, multiply the raw later/earlier CTR ratio by "
            "min(3, 1 / matched later/earlier CTR ratio). This is a mechanical illustration "
            "with a different population per contrast."
        ),
        "smooth_sensitivity": {
            "formula": "e(p) = p^(-eta); weight(p) = min(3, 1 / e(p))",
            "eta": ETAS,
            "inverse_cap": WEIGHT_CAP,
            "status": "Assumed scenarios; no curve fit to the matched contrasts.",
            "weighted_click_mass": "Sum of click times weight within each rank or band.",
        },
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
