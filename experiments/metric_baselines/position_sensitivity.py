"""Compare raw clicks with a prespecified position-weighting sensitivity check."""

import argparse
import json
from pathlib import Path

from catboost import CatBoostRegressor
import numpy as np

from compare import CATBOOST_PARAMS, _cluster_bootstrap, _rank_candidates, _request_starts
from features import build_features
from slates import load_slates_at_depths


def weights(positions, exponent):
    return np.minimum(np.power(positions.astype(float), exponent), 3.0)


def ndcg(gains, starts, group_index, order):
    """NDCG@10 with linear gains; positions here are ranks in the new list."""
    ranks = np.arange(len(gains)) - starts[group_index]
    discount = np.where(ranks < 10, 1 / np.log2(ranks + 2), 0.0)
    ideal_order = _rank_candidates(gains, group_index)
    ideal = np.add.reduceat(gains[ideal_order] * discount, starts)
    actual = np.add.reduceat(gains[order] * discount, starts)
    return np.divide(actual, ideal, out=np.zeros_like(actual), where=ideal > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test-identifiers", required=True, type=Path)
    parser.add_argument("--notes", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exponent", type=float, default=0.25)
    args = parser.parse_args()

    slates = load_slates_at_depths(args.train, args.test_identifiers, [10, 30])
    _, validation_10 = slates[10]
    train, val = slates[30]
    val = val[val.request_key.isin(validation_10.request_key)].reset_index(drop=True)
    print(f"Position sensitivity: {len(train):,} train and {len(val):,} validation pairs", flush=True)
    x_train, x_val, feature_names = build_features(train, val, args.notes)

    click_train = train.click.to_numpy(dtype=float)
    click_val = val.click.to_numpy(dtype=float)
    adjusted_train = click_train * weights(train.position.to_numpy(), args.exponent)
    adjusted_val = click_val * weights(val.position.to_numpy(), args.exponent)
    starts = _request_starts(val)
    group_index = np.repeat(np.arange(len(starts)), np.diff(starts, append=len(val)))
    orders = [np.arange(len(val))]
    models = ("qilin", "click", "weighted_click")
    for name, target in zip(models[1:], (click_train, adjusted_train)):
        print(f"Fitting {name}", flush=True)
        prediction = CatBoostRegressor(**CATBOOST_PARAMS).fit(x_train, target).predict(x_val)
        orders.append(_rank_candidates(prediction, group_index))

    labels = (click_val, adjusted_val)
    scores = np.stack([
        np.column_stack([ndcg(gain, starts, group_index, order) for gain in labels])
        for order in orders
    ], axis=1)
    bootstrap = _cluster_bootstrap(scores, np.ones((len(starts), len(labels)), dtype=bool), val.user_id.iloc[starts])
    result = {
        "candidate_depth": 30,
        "validation_requests": len(starts),
        "validation_pairs": len(val),
        "train_pairs": len(train),
        "features": feature_names,
        "assumed_examination_exponent": args.exponent,
        "weights_by_original_position": weights(np.arange(1, 31), args.exponent).tolist(),
        "results": {},
    }
    means = scores.mean(axis=0)
    for i, name in enumerate(models):
        result["results"][name] = {}
        for j, metric in enumerate(("click_ndcg_at_10", "weighted_click_ndcg_at_10")):
            summary = {"mean": float(means[i, j])}
            for baseline in (0, 1):
                delta = bootstrap[:, i, j] - bootstrap[:, baseline, j]
                summary[f"delta_vs_{models[baseline]}"] = float(means[i, j] - means[baseline, j])
                summary[f"ci95_vs_{models[baseline]}"] = np.quantile(delta, [0.025, 0.975]).tolist()
            result["results"][name][metric] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
