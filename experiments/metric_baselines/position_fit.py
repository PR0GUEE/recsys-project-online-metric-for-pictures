"""Fit descriptive Qilin click and like curves from repeated training impressions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import brentq
from scipy.stats import norm

from position_audit import PAIR_KEY, POSITION_LIMIT, POSITION_TO_GROUP, load_exposures
from slates import _DETAILS, _T1, _T2


def attach_likes(exposures, train_path):
    """Read the raw like flag at the same request row, note, and original rank."""
    fields = [f"{_DETAILS}.list.element.{field}" for field in ("note_idx", "position", "like")]
    with pq.ParquetFile(train_path) as parquet:
        requests = parquet.read(columns=fields).column(_DETAILS)

    liked_keys = {
        (request_key, result["note_idx"], result["position"])
        for request_key in exposures["request_key"].unique()
        for result in requests[request_key].as_py() or []
        if result is not None and result["like"] == 1
    }
    keys = pd.MultiIndex.from_frame(exposures[["request_key", "note_id", "position"]])
    return exposures.assign(like=keys.isin(liked_keys).astype(np.int8))


def estimate_power_law(impressions, events, queries):
    """Profile out the Poisson intercept for each query–note match unit."""
    log_position = np.log(np.arange(1, impressions.shape[1] + 1))
    event_totals = events.sum(axis=1)
    informative = event_totals > 0
    impressions = impressions[informative]
    events = events[informative]
    event_totals = event_totals[informative]
    queries = queries[informative]
    observed_log_position = (events * log_position).sum(axis=1)
    first_log_position = log_position[np.argmax(impressions > 0, axis=1)]
    centered_log_position = np.maximum(log_position - first_log_position[:, None], 0)

    def position_probabilities(exponent):
        # Shift each row by its first supported rank to keep normalization stable.
        mass = impressions * np.exp(-exponent * centered_log_position)
        return mass / mass.sum(axis=1, keepdims=True)

    def score(exponent):
        expected = position_probabilities(exponent) @ log_position
        return float((observed_log_position - event_totals * expected).sum())

    exponent = 0.0
    if not len(event_totals):
        status = "no_events"
    elif score(0.0) >= 0:
        status = "lower_bound"
    elif not np.any(observed_log_position > event_totals * first_log_position + 1e-12):
        status = "no_finite_estimate"
    else:
        status = "interior"
        upper = 1.0
        while score(upper) < 0:
            upper *= 2
        exponent = float(brentq(score, 0.0, upper))

    probabilities = position_probabilities(exponent)
    expected_events = (event_totals[:, None] * probabilities).sum(axis=0)
    mean_log_position = probabilities @ log_position
    deviations = log_position - mean_log_position[:, None]
    information = float((event_totals * (probabilities * deviations**2).sum(axis=1)).sum())
    query_codes, query_names = pd.factorize(queries)
    query_scores = np.bincount(
        query_codes, weights=observed_log_position - event_totals * mean_log_position
    )
    query_count = len(query_names)
    standard_error = None
    interval = None
    if status == "interior" and query_count > 1 and information > 0:
        correction = query_count / (query_count - 1)
        standard_error = float(np.sqrt(correction * np.square(query_scores).sum()) / information)
        margin = norm.ppf(0.975) * standard_error
        interval = [float(max(0, exponent - margin)), float(exponent + margin)]

    finite = status in ("interior", "lower_bound")
    return {
        "status": status,
        "exponent": exponent if finite else None,
        "e_by_position": np.exp(-exponent * log_position).tolist() if finite else None,
        "cluster_robust_se_by_query": standard_error,
        "cluster_robust_95_ci": interval,
        "informative_query_clusters": query_count,
        "at_nonnegative_boundary": status == "lower_bound",
    }, expected_events if finite else np.full_like(expected_events, np.nan)


def fit_matched_curve(exposures, event, *, same_user=False, max_position=POSITION_LIMIT):
    keys = [*PAIR_KEY, "user_id"] if same_user else PAIR_KEY
    positions = range(1, max_position + 1)
    eligible = exposures.loc[exposures["position"].le(max_position)]
    counts = eligible.groupby([*keys, "position"])[event].agg(impressions="size", events="sum")
    positions_by_unit = counts.groupby(level=keys).size()
    matched = counts.loc[counts.index.droplevel("position").isin(positions_by_unit[positions_by_unit >= 2].index)]
    matrix = matched.unstack("position", fill_value=0).reindex(
        columns=pd.MultiIndex.from_product([["impressions", "events"], positions]), fill_value=0
    )
    impressions = matrix["impressions"].to_numpy()
    events = matrix["events"].to_numpy()
    queries = matrix.index.get_level_values("query").to_numpy()
    informative = events.sum(axis=1) > 0
    estimate, expected_events = estimate_power_law(impressions, events, queries)

    head_support = (impressions[:, :10] > 0).sum(axis=1)
    tail_support = (impressions[:, 10:] > 0).sum(axis=1)
    rank_totals = eligible.groupby("position")[event].agg(eligible_impressions="size", eligible_events="sum")
    profile = rank_totals.reindex(list(positions), fill_value=0).rename_axis("position").reset_index()
    profile["group"] = profile["position"].map(POSITION_TO_GROUP)
    profile["matched_impressions"] = impressions.sum(axis=0)
    profile["matched_events"] = events.sum(axis=0)
    profile["matched_units"] = (impressions > 0).sum(axis=0)
    profile["informative_matched_units"] = (impressions[informative] > 0).sum(axis=0)
    profile["expected_events"] = expected_events

    return {
        "fit_depth": max_position,
        "match_unit": "+".join(keys),
        "event": event,
        "eligible_units": len(positions_by_unit),
        "matched_units": len(matrix),
        "matched_query_notes": len(matrix.index.droplevel("user_id").unique()) if same_user else len(matrix),
        "matched_queries": matrix.index.get_level_values("query").nunique(),
        "matched_impressions": int(impressions.sum()),
        f"matched_{event}s": int(events.sum()),
        "informative_units": int(informative.sum()),
        f"zero_{event}_units": int((~informative).sum()),
        "units_touching_rank_one": int((impressions[:, 0] > 0).sum()),
        "units_with_at_least_two_ranks_in_top_10": int((head_support >= 2).sum()),
        "units_touching_both_top_10_and_later": int(((head_support > 0) & (tail_support > 0)).sum()),
        "units_with_at_least_two_ranks_after_top_10": int((tail_support >= 2).sum()),
        **estimate,
    }, profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test-identifiers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    exposures, population = load_exposures(args.train, args.test_identifiers)
    empty_query = exposures["query"].eq("")
    repeated_note = exposures.duplicated(["request_key", "note_id"], keep=False)
    eligible = attach_likes(exposures.loc[~empty_query & ~repeated_note], args.train)

    diagnostics = []
    curves_by_depth = {}
    for depth in (POSITION_LIMIT, 10):
        curves = {}
        for event in ("click", "like"):
            for label, same_user in (("matched_query_note", False), ("matched_same_user", True)):
                curve_name = label if event == "click" else f"{label}_like"
                curves[curve_name], profile = fit_matched_curve(
                    eligible, event, same_user=same_user, max_position=depth
                )
                diagnostics.append(profile.assign(curve=curve_name, event=event, fit_depth=depth))
        curves_by_depth[depth] = curves

    result = {
        "dataset": "THUIR/Qilin",
        **population,
        "split_boundaries_utc_unix": [_T1, _T2],
        "raw_exposures": len(exposures),
        "raw_clicks": int(exposures["click"].sum()),
        "match_eligible_exposures": len(eligible),
        "match_eligible_clicks": int(eligible["click"].sum()),
        "match_eligible_likes": int(eligible["like"].sum()),
        "like_input": "Read like == 1 from the same raw exposures used for click fitting, "
        "paired by parquet row, note, and original position.",
        "exposures_with_empty_normalized_query": int(empty_query.sum()),
        "exposures_with_repeated_note_in_same_request": int(repeated_note.sum()),
        "query_normalization": "NFKC, casefold, remove whitespace and Unicode punctuation.",
        "matching_rule": "At least two distinct original ranks in the fit range across requests.",
        "model": "Fit each event separately. Expected events at rank p within unit g = "
        "impressions(g,p) * intercept(g) * p^(-alpha). Profile the unit intercept by conditioning "
        "on its total events. Constrain alpha >= 0. Zero-event units enter support counts. "
        "Event-positive units enter the likelihood.",
        "interpretation": "Descriptive estimate from observational ranking variation. "
        "Time, personalization, request context, and exposure selection may confound the estimated shape. "
        "The relative curve serves as a feasibility proxy for reweighting. It cannot identify an examination "
        "propensity or the causal effect of rank. Query-clustered intervals reflect sampling uncertainty "
        "under the fitted model and do not measure these sources of bias.",
        "like_interpretation": "Exploratory fit for the position gradient of recorded likes. The rare "
        "event provides less support, and this outcome also reflects behavior after viewing the result.",
        "curves": curves_by_depth[POSITION_LIMIT],
        "head_only_sensitivity": curves_by_depth[10],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    pd.concat(diagnostics, ignore_index=True).to_csv(args.output / "fit_diagnostics.csv", index=False)
    (args.output / "curves.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
