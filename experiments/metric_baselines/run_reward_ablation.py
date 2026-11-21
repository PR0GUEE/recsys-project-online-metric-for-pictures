"""Compare mild positional bonuses, local preferences and fast requery penalties."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from compare import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CATBOOST_PARAMS,
    CUTOFF,
    _rank_candidates,
    _request_starts,
    _validation_ndcg,
)
from engagement_eval import evaluate
from engagement_models import PAIRWISE_PARAMS, pointwise_predictions
import reformulation_signals
from run import DATA_REVISION
from run_avatar_rerank import DEPTH, bootstrap_metrics, event_weights, paired_delta, write_csv
from run_engagement import COMPOSITES, build_rank_features, composite, signals
from slates import _T1, _T2, load_slates_at_depths


ETA = 0.25
LOCAL_PAIR_STRENGTH = 0.10
REQUERY_PENALTY = 0.25
MATCHED_CONTROLS = {
    "point_full_correction": "point_raw",
    "point_mild_click": "point_raw",
    "point_mild_satisfied": "point_mild_click",
    "point_mild_satisfied_fast_requery": "point_mild_satisfied",
    "pair_mild_satisfied_local": "pair_mild_satisfied",
}
METRICS = (
    "ClickNDCG@10",
    "LikeNDCG@10",
    "LikePositiveNDCG@10",
    "ClickDCG@10_weighted",
    "LikeDCG@10_weighted",
)


def reward_targets(train, curves, requery_after_bounce):
    recipe = COMPOSITES["strong_actions"]
    scale = sum(weight for name, weight in recipe.items() if name != "skip")
    logged = signals(train)
    raw = composite(logged, recipe)
    position_index = train.position.to_numpy() - 1
    inverse_exposure = event_weights(curves)
    mild = 1 + ETA * (inverse_exposure["click"][position_index] - 1)
    clicked = logged["click"].astype(bool)
    satisfied = clicked & (logged["long_view"].astype(bool) | logged["collect"].astype(bool))
    click_bonus = recipe["click"] * (mild - 1) / scale
    full_adjustment = sum(
        recipe[event] * logged[event] * (inverse_exposure[event][position_index] - 1)
        for event in ("click", "like")
    ) / scale
    mild_satisfied = raw + click_bonus * satisfied
    targets = {
        "point_raw": raw,
        "point_full_correction": raw + full_adjustment,
        "point_mild_click": raw + click_bonus * clicked,
        "point_mild_satisfied": mild_satisfied,
        "point_mild_satisfied_fast_requery": mild_satisfied - REQUERY_PENALTY * requery_after_bounce / scale,
    }
    active = logged["active"].astype(bool)
    click_exposure = np.asarray(curves["click"]["e_by_position"])[position_index]
    return targets, satisfied, ~(clicked | active), mild, click_exposure


def preference_pairs(frame, target, satisfied, inactive, mild, exposure):
    """Use the same preferences in both arms and give local skips a small extra weight."""
    boundaries = np.append(_request_starts(frame), len(frame))
    position = frame.position.to_numpy()
    pair_blocks, uniform_blocks, bonus_blocks = [], [], []
    requests_with_local_pairs = 0
    local_pairs = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        reward = target[start:end]
        winner, loser = np.nonzero(reward[:, None] > reward[None, :])
        if not len(winner):
            continue
        winner, loser = start + winner, start + loser
        pair_blocks.append(np.column_stack((winner, loser)).astype(np.int32))
        uniform_blocks.append(np.full(len(winner), 1 / len(winner)))
        distance = position[winner] - position[loser]
        local = satisfied[winner] & inactive[loser] & (distance >= 1) & (distance <= 2)
        bonus = np.zeros(len(winner))
        count = int(local.sum())
        if count:
            bonus[local] = LOCAL_PAIR_STRENGTH * exposure[loser[local]] * mild[winner[local]] / count
            local_pairs += count
            requests_with_local_pairs += 1
        bonus_blocks.append(bonus)
    pairs = np.concatenate(pair_blocks)
    uniform = np.concatenate(uniform_blocks)
    bonus = np.concatenate(bonus_blocks)
    coverage = {
        "preference_pairs": len(pairs),
        "requests_with_preference_pairs": len(pair_blocks),
        "local_pairs": local_pairs,
        "requests_with_local_pairs": requests_with_local_pairs,
        "extra_pair_weight": float(bonus.sum()),
    }
    return pairs, uniform, uniform + bonus, coverage


def fit_pairwise(x_train, target, groups, pairs, weights, x_val):
    from catboost import CatBoostRanker, Pool

    pool = Pool(x_train, label=target, group_id=groups, pairs=pairs, pairs_weight=weights)
    return CatBoostRanker(**PAIRWISE_PARAMS).fit(pool).predict(x_val)


def comparisons(val, predictions, curves):
    labels = val[["click", "like", "like"]].to_numpy(dtype=bool)
    ndcg, positives, starts = _validation_ndcg(val, predictions.values(), labels=labels)
    sizes = np.diff(starts, append=len(val))
    groups = np.repeat(np.arange(len(starts)), sizes)
    ranks = np.arange(len(val)) - starts[groups] + 1
    discount = np.where(ranks <= CUTOFF, 1 / np.log2(ranks + 1), 0)
    weights = event_weights(curves)
    positions = val.position.to_numpy() - 1
    gains = val[["click", "like"]].to_numpy() * np.column_stack([
        weights[event][positions] for event in ("click", "like")
    ])
    orders = [np.arange(len(val)), *(_rank_candidates(s, groups) for s in predictions.values())]
    dcg = np.stack([
        np.add.reduceat(gains[order] * discount[:, None], starts, axis=0) for order in orders
    ], axis=1)
    eligible = np.ones((len(starts), len(METRICS)), dtype=bool)
    eligible[:, 2] = positives[:, 2] > 0
    means, replicates = bootstrap_metrics(
        np.concatenate([ndcg, dcg], axis=2), eligible, val.user_id.iloc[starts]
    )

    models = ["qilin", *predictions]
    rows = []
    for candidate_index, model in enumerate(predictions, start=1):
        baselines = dict.fromkeys(["qilin", "point_raw", MATCHED_CONTROLS.get(model, "point_raw")])
        for baseline in baselines:
            if baseline == model:
                continue
            baseline_index = models.index(baseline)
            for number, metric in enumerate(METRICS):
                absolute = paired_delta(means, replicates, candidate_index, baseline_index, number)
                relative = paired_delta(means, replicates, candidate_index, baseline_index, number, relative=True)
                rows.append({
                    "model": model, "baseline": baseline, "metric": metric,
                    "score": float(means[candidate_index, number]),
                    "baseline_score": float(means[baseline_index, number]),
                    "evaluated_requests": int(eligible[:, number].sum()),
                    **absolute,
                    **{f"relative_{key}": value for key, value in relative.items()},
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test-identifiers", required=True, type=Path)
    parser.add_argument("--notes", required=True, nargs="+", type=Path)
    parser.add_argument("--curves", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    curves = json.loads(args.curves.read_text(encoding="utf-8"))["updated"]
    slates = load_slates_at_depths(args.train, args.test_identifiers, [CUTOFF, DEPTH], include_engagement=True)
    train, val = slates[DEPTH]
    val = val[val.request_key.isin(slates[CUTOFF][1].request_key)].reset_index(drop=True)
    print(f"Loaded {len(train):,} train pairs and {len(val):,} validation pairs", flush=True)
    requery = reformulation_signals.fast_requery_after_bounce(train)
    targets, satisfied, inactive, mild, exposure = reward_targets(train, curves, requery)
    x_train, x_val, features = build_rank_features(train, val, args.notes)
    predictions, timings = {}, {}

    def record(name, predict, *arguments):
        print(f"Fitting {name}", flush=True)
        start = perf_counter()
        predictions[name] = predict(*arguments)
        timings[name] = perf_counter() - start

    for name, target in targets.items():
        record(name, pointwise_predictions, x_train, target, x_val)
    pair_target = targets["point_mild_satisfied"]
    pairs, uniform, augmented, pair_coverage = preference_pairs(train, pair_target, satisfied, inactive, mild, exposure)
    for name, weights in (("pair_mild_satisfied", uniform), ("pair_mild_satisfied_local", augmented)):
        record(name, fit_pairwise, x_train, pair_target, train.request_key.to_numpy(), pairs, weights, x_val)

    print("Scoring common validation candidates", flush=True)
    summary = evaluate(val, predictions, intervals=False)
    summary["comparisons"] = comparisons(val, predictions, curves)
    summary.update({
        "config": {
            "dataset_revision": DATA_REVISION, "split_boundaries_utc_unix": [_T1, _T2],
            "candidate_depth": DEPTH, "cutoff": CUTOFF, "features": features,
            "curve_file": str(args.curves), "composite_weights": COMPOSITES["strong_actions"],
            "eta": ETA, "satisfied_click": "Click with at least 10 seconds of page time or a save",
            "local_pair_rule": "Satisfied click ranks above inactive note at position minus one or two. "
                "Base preference weights sum to one per request. Extra pair weight is local_pair_strength "
                "times the upper note's relative exposure times the lower note's mild click weight, "
                "divided by the number of local pairs in the same request.",
            "local_pair_strength": LOCAL_PAIR_STRENGTH,
            "fast_requery_penalty_in_click_units": REQUERY_PENALTY,
            "fast_requery_signal": {
                "max_bounce_seconds": reformulation_signals.MAX_BOUNCE_SECONDS,
                "max_requery_seconds": reformulation_signals.MAX_REQUERY_SECONDS,
                "min_bigram_jaccard": reformulation_signals.MIN_BIGRAM_JACCARD,
                "min_shorter_query_coverage": reformulation_signals.MIN_SHORTER_QUERY_COVERAGE,
                "min_shared_bigrams": reformulation_signals.MIN_SHARED_BIGRAMS,
            },
            "weighted_evaluation": "Mean DCG@10 without IDCG. Inverse exposure uses the logged position. "
                "The same updated curves score all models, including Qilin.",
            "bootstrap": {"unit": "user_id", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
            "pointwise": CATBOOST_PARAMS, "pairwise": PAIRWISE_PARAMS,
        },
        "train": {"requests": int(train.request_key.nunique()), "rows": len(train),
                  "satisfied_click_rows": int(satisfied.sum()),
                  "satisfied_click_rows_with_bonus": int((satisfied & (mild > 1)).sum()),
                  "fast_requery_rows": int(requery.sum()),
                  "fast_requery_requests": int(train.loc[requery, "request_key"].nunique()),
                  **pair_coverage},
        "seconds_fit_and_predict": timings,
    })
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(args.output / "comparisons.csv", summary["comparisons"])


if __name__ == "__main__":
    main()
