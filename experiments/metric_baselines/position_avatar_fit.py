"""Refit Qilin positional curves after merging explicit avatar-query aliases."""

import argparse
import json
from pathlib import Path

from position_audit import PAIR_KEY, load_exposures
from position_fit import attach_likes, fit_matched_curve
from slates import _T1, _T2


AVATAR_GROUPS = {
    "women_avatars": ("女生头像", "头像女头"),
    "cat_avatars": ("头像猫猫", "猫咪头像", "猫猫头像"),
    "avatar_sets_for_three_friends": ("三个人闺蜜头像", "三个人闺蜜头"),
    "anime_avatars": ("动漫头像", "头像动漫"),
    "cute_avatars_for_girls": ("女生头像可爱", "可爱头像女头"),
}
EVENTS = ("click", "like")


def counts(frame):
    return {
        "requests": frame["request_key"].nunique(),
        "users": frame["user_id"].nunique(),
        "impressions": len(frame),
        "clicks": int(frame["click"].sum()),
        "likes": int(frame["like"].sum()),
    }


def varying_positions(exposures, keys):
    return exposures.groupby(keys)["position"].transform("nunique").gt(1)


def group_support(avatar_rows):
    exact_match = varying_positions(avatar_rows, PAIR_KEY)
    alias_match = varying_positions(avatar_rows, ["note_id"])
    newly_supported = avatar_rows.loc[alias_match & ~exact_match]
    exact_notes = set(avatar_rows.loc[exact_match, "note_id"])
    exact_pairs = avatar_rows.loc[exact_match, PAIR_KEY].drop_duplicates()

    # Count each cross-form pair of displays once, then recover unique supported displays.
    displays = avatar_rows.reset_index(drop=True).rename_axis("display").reset_index()
    joined = displays.merge(displays, on="note_id")
    cross_form = joined.loc[
        joined["query_x"].lt(joined["query_y"]) & joined["position_x"].ne(joined["position_y"])
    ]
    cross_form_displays = displays.loc[
        displays["display"].isin(cross_form["display_x"]) | displays["display"].isin(cross_form["display_y"])
    ]
    cross_form_totals = cross_form_displays.groupby("note_id")[list(EVENTS)].sum()
    return {
        "input": counts(avatar_rows),
        "by_alias": {alias: counts(rows) for alias, rows in avatar_rows.groupby("query")},
        "baseline_matched_query_note_units": len(exact_pairs),
        "updated_matched_intent_note_units": avatar_rows.loc[alias_match, "note_id"].nunique(),
        "new_intent_note_units_absent_from_baseline_matches": len(
            set(avatar_rows.loc[alias_match, "note_id"]) - exact_notes
        ),
        "newly_matched_displays": counts(newly_supported),
        "cross_form_note_count_at_different_ranks": cross_form["note_id"].nunique(),
        "cross_form_display_pairs_at_different_ranks": len(cross_form),
        "cross_form_note_count_at_different_ranks_same_user": cross_form.loc[
            cross_form["user_id_x"].eq(cross_form["user_id_y"]), "note_id"
        ].nunique(),
        "cross_form_displays_at_different_ranks": counts(cross_form_displays),
        "cross_form_notes_with_clicks": int(cross_form_totals["click"].gt(0).sum()),
        "cross_form_notes_with_likes": int(cross_form_totals["like"].gt(0).sum()),
    }


def fit_both_events(exposures):
    return {event: fit_matched_curve(exposures, event)[0] for event in EVENTS}


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

    aliases = {alias: variants[0] for variants in AVATAR_GROUPS.values() for alias in variants}
    is_avatar_alias = eligible["query"].isin(aliases)
    updated = eligible.assign(query=eligible["query"].replace(aliases))
    baseline_curves = fit_both_events(eligible)
    updated_curves = fit_both_events(updated)

    result = {
        "dataset": "THUIR/Qilin",
        **population,
        "split_boundaries_utc_unix": [_T1, _T2],
        "fit_input": counts(eligible),
        "updated_input": counts(updated),
        "exposures_with_empty_normalized_query": int(empty_query.sum()),
        "exposures_with_repeated_note_in_same_request": int(repeated_note.sum()),
        "query_rows_reassigned": int(eligible["query"].ne(updated["query"]).sum()),
        "avatar_alias_input": counts(eligible.loc[is_avatar_alias]),
        "cohort": {
            name: {
                "canonical_query": variants[0],
                "aliases": list(variants),
                **group_support(eligible.loc[eligible["query"].isin(variants)]),
            }
            for name, variants in AVATAR_GROUPS.items()
        },
        "matching_change": (
            "Keep every eligible exposure exactly once. Replace only exact normalized avatar "
            "aliases with a shared query key before matching the same note across ranks 1–30. "
            "Select aliases from fit query text. Keep format and explicit style requirements separate."
        ),
        "model": (
            "For click and like separately, profile the Poisson intercept for each query key and note. "
            "Fit the relative curve e(p) = p^(-alpha), constrained to alpha >= 0."
        ),
        "interpretation": (
            "Alias matching assumes comparable query meaning for the same note. The full fit remains "
            "observational. Alias membership does not establish relevance for every note in a slate. "
            "Avatar-only support covers very few intents and is exploratory, especially for likes."
        ),
        "baseline": baseline_curves,
        "updated": updated_curves,
        "avatar_only": fit_both_events(updated.loc[is_avatar_alias]),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "curves.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
