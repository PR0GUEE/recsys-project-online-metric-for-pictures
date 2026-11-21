"""Diagnose where the Qilin reranking gap appears on the existing validation split."""

import argparse
import json
from math import log2
from pathlib import Path

from catboost import CatBoostRegressor
import numpy as np

from compare import CATBOOST_PARAMS, _cluster_bootstrap, _rank_candidates, _request_starts, _validation_ndcg
from features import build_features
from slates import load_slates_at_depths


def reference_scores(clicks, starts, ranking):
    """Independent per-request ClickNDCG@10 using the full candidate pool."""
    discount = [1 / log2(rank + 2) for rank in range(10)]
    values = []
    for start, end in zip(starts, [*starts[1:], len(clicks)]):
        ideal = sum(discount[: min(10, int(clicks[start:end].sum()))])
        ranked_clicks = clicks[ranking[start : min(start + 10, end)]]
        dcg = sum(value * weight for value, weight in zip(ranked_clicks, discount))
        values.append(dcg / ideal if ideal else 0.0)
    return np.asarray(values)


def evaluate(train, val, x_train, x_val):
    starts = _request_starts(val)
    sizes = np.diff(starts, append=len(val))
    group_index = np.repeat(np.arange(len(starts)), sizes)
    original = np.arange(len(val))
    top10 = original - starts[group_index] < 10
    clicks = val.click.to_numpy(dtype=int)
    positions = val.position.to_numpy(dtype=int)
    train_pos = train.position.to_numpy(dtype=np.float32)[:, None]
    val_pos = positions.astype(np.float32)[:, None]
    y = train.click.to_numpy()
    predictions = {}
    inputs = {
        "text": (x_train, x_val),
        "position_only": (train_pos, val_pos),
        "text_and_position": (np.column_stack((x_train, train_pos)), np.column_stack((x_val, val_pos))),
    }
    for name, (fit_x, predict_x) in inputs.items():
        print(f"Fitting {name}", flush=True)
        predictions[name] = CatBoostRegressor(**CATBOOST_PARAMS).fit(fit_x, y).predict(predict_x)

    text = predictions["text"]
    def rank(scores):
        return _rank_candidates(scores, group_index)

    rng = np.random.default_rng(42)
    orders = {
        "qilin": original,
        "random": rank(rng.random(len(val))),
        "oracle": rank(clicks),
        "text_reversed": rank(-text),
        "text_within_qilin_top10": rank(np.where(top10, text, -np.inf)),
        **{name: rank(score) for name, score in predictions.items()},
    }
    scores = np.stack([reference_scores(clicks, starts, order) for order in orders.values()], axis=1)
    official, counts, _ = _validation_ndcg(val, list(predictions.values()))
    differences = [np.max(np.abs(scores[:, 0] - official[:, 0, 0]))]
    for index, name in enumerate(predictions, start=1):
        differences.append(np.max(np.abs(scores[:, list(orders).index(name)] - official[:, index, 0])))

    masks = np.ones((len(starts), 1), dtype=bool)
    sample = _cluster_bootstrap(scores[:, :, None], masks, val.user_id.iloc[starts])[:, :, 0]
    model_index = {name: i for i, name in enumerate(orders)}
    by_model = {}
    for i, name in enumerate(orders):
        values = scores[:, i]
        delta = values - scores[:, 0]
        interval = np.quantile(sample[:, i] - sample[:, 0], [0.025, 0.975])
        by_model[name] = {
            "click_ndcg_at_10": float(values.mean()),
            "delta_vs_qilin": float(delta.mean()),
            "ci95_vs_qilin": interval.tolist(),
            "queries_above_qilin": int((delta > 1e-12).sum()),
            "queries_below_qilin": int((delta < -1e-12).sum()),
            "historical_clicks_in_top10": int(clicks[orders[name][top10]].sum()),
        }

    decomposition = {}
    for a, b in [("qilin", "text_within_qilin_top10"), ("text_within_qilin_top10", "text"), ("text", "text_and_position")]:
        j, k = model_index[a], model_index[b]
        difference = scores[:, k] - scores[:, j]
        decomposition[f"{b}_minus_{a}"] = {
            "delta": float(difference.mean()),
            "ci95": np.quantile(sample[:, k] - sample[:, j], [0.025, 0.975]).tolist(),
        }
    text_top = orders["text"][top10]
    promoted = text_top[~top10[text_top]]
    displaced = original[top10 & ~np.isin(original, text_top)]
    return {
        "validation_requests": len(starts),
        "validation_pairs": len(val),
        "validation_requests_with_click": int((counts[:, 0] > 0).sum()),
        "config": CATBOOST_PARAMS,
        "reference_metric_max_abs_difference": float(max(differences)),
        "scores": by_model,
        "decomposition": decomposition,
        "click_support": {
            "original_top10": {"candidates": int(top10.sum()), "clicks": int(clicks[top10].sum())},
            "outside_original_top10": {"candidates": int((~top10).sum()), "clicks": int(clicks[~top10].sum())},
            "promoted_by_text": {"candidates": len(promoted), "clicks": int(clicks[promoted].sum())},
            "displaced_by_text": {"candidates": len(displaced), "clicks": int(clicks[displaced].sum())},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test-identifiers", required=True, type=Path)
    parser.add_argument("--notes", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    slates = load_slates_at_depths(args.train, args.test_identifiers, [10, 30])
    train, val = slates[30]
    val = val[val.request_key.isin(slates[10][1].request_key)].reset_index(drop=True)
    print(f"Loaded {len(train):,} train pairs and {len(val):,} validation pairs", flush=True)
    x_train, x_val, feature_names = build_features(train, val, args.notes)
    result = evaluate(train, val, x_train, x_val)
    result["train_pairs"] = len(train)
    result["text_model_features"] = feature_names
    result["validation_rows_without_text"] = int(x_val[:, feature_names.index("note_text_missing")].sum())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
