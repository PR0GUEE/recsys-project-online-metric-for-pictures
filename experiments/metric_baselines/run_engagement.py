"""Compare engagement targets and position-aware rerankers on Qilin validation."""

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from compare import CATBOOST_PARAMS, signal_targets
from engagement_eval import evaluate
from engagement_models import PAIRWISE_PARAMS, RIDGE_L2, pairwise_predictions, pointwise_predictions, ridge_predictions
from features import build_features
from run import DATA_REVISION
from slates import load_slates_at_depths


COMPOSITES = {
    # Units are weights relative to one click. The skip weight is subtracted.
    "balanced": dict(click=1, dwell=0.5, like=1, collect=0.5, other=0.5, skip=0),
    "strong_actions": dict(click=1, dwell=0.5, like=2, collect=1, other=0.5, skip=0),
    "strong_actions_with_skip": dict(click=1, dwell=0.5, like=2, collect=1, other=0.5, skip=0.15),
}


def split_calibration(train):
    requests = train.drop_duplicates("request_key")
    cutoff = float(requests.request_start.quantile(0.8, interpolation="higher"))
    earlier = requests[requests.request_end < cutoff]
    later = requests[requests.request_start >= cutoff]
    shared_sessions = set(zip(earlier.user_id, earlier.session_id)) & set(zip(later.user_id, later.session_id))

    def eligible(frame):
        request_keys = [key for key, user, session in zip(frame.request_key, frame.user_id, frame.session_id)
                        if (user, session) not in shared_sessions]
        return train[train.request_key.isin(request_keys)].reset_index(drop=True)

    return eligible(earlier), eligible(later), cutoff


def build_rank_features(train, val, notes):
    x_train, x_val, names = build_features(train, val, notes)
    return (
        np.column_stack((x_train, train.position.to_numpy(dtype=np.float32))),
        np.column_stack((x_val, val.position.to_numpy(dtype=np.float32))),
        [*names, "original_qilin_position"],
    )


def signals(frame):
    targets = signal_targets(frame)
    for name in ("like", "share", "comment"):
        targets[name] = frame[name].to_numpy(dtype=float)
    targets["other"] = ((targets["share"] > 0) | (targets["comment"] > 0)).astype(float)
    targets["active"] = ((targets["like"] > 0) | (targets["collect"] > 0) | (targets["other"] > 0)).astype(float)
    return targets


def composite(targets, weights):
    skip_cost = (targets["click"] - targets["skip_penalty"]) * (1 - targets["active"])
    reward = sum(weights[name] * targets[name] for name in ("click", "like", "collect", "other"))
    reward += weights["dwell"] * targets["dwell_log"]
    scale = sum(value for name, value in weights.items() if name != "skip")
    return (reward - weights["skip"] * skip_cost) / scale


def choose_composite(train, notes):
    inner_train, calibration, cutoff = split_calibration(train)
    print(f"Calibration: {inner_train.request_key.nunique():,} fit requests, "
          f"{calibration.request_key.nunique():,} later requests", flush=True)
    x_train, x_cal, _ = build_rank_features(inner_train, calibration, notes)
    targets = signals(inner_train)
    predictions = {}
    for name, weights in COMPOSITES.items():
        print(f"Calibrating {name}", flush=True)
        predictions[name] = pointwise_predictions(x_train, composite(targets, weights), x_cal)
    calibration_results = evaluate(calibration, predictions, intervals=False)
    candidates = [row for row in calibration_results["results"]
                  if row["model"] in COMPOSITES and row["metric"] == "LikeNDCG@10"]
    chosen = max(candidates, key=lambda row: row["ndcg"])["model"]
    calibration_results.update(cutoff_utc_unix=cutoff, train_requests=int(inner_train.request_key.nunique()),
                               train_pairs=len(inner_train), selected_composite=chosen)
    return chosen, calibration_results


def fit_models(train, x_train, x_val, selected_weights):
    y = signals(train)
    y["click_dwell"] = (y["click"] + 0.5 * y["dwell_log"]) / 1.5
    y["composite"] = composite(y, selected_weights)
    predictions, timings = {}, {}

    def record(model, predict, *args):
        print(f"Fitting {model}", flush=True)
        started = perf_counter()
        predictions[model] = predict(*args)
        timings[model] = perf_counter() - started

    for name in ("click", "skip_penalty", "long_view", "dwell_log", "like", "collect", "active", "click_dwell", "composite"):
        record(f"point_{name}", pointwise_predictions, x_train, y[name], x_val)
    groups = train.request_key.to_numpy()
    for name in ("click", "composite"):
        record(f"ridge_{name}", ridge_predictions, x_train, y[name], x_val)
        record(f"pair_{name}", pairwise_predictions, x_train, y[name], groups, x_val)
    return predictions, timings, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test-identifiers", required=True, type=Path)
    parser.add_argument("--notes", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    slates = load_slates_at_depths(args.train, args.test_identifiers, [10, 30], include_engagement=True)
    train, val = slates[30]
    val = val[val.request_key.isin(slates[10][1].request_key)].reset_index(drop=True)
    print(f"Loaded {len(train):,} train pairs and {len(val):,} validation pairs", flush=True)
    chosen, calibration = choose_composite(train, args.notes)
    print(f"Calibration chose {chosen}", flush=True)
    x_train, x_val, names = build_rank_features(train, val, args.notes)
    predictions, timings, targets = fit_models(train, x_train, x_val, COMPOSITES[chosen])
    result = evaluate(val, predictions)
    result.update(
        config={
            "dataset_revision": DATA_REVISION,
            "candidate_depth": 30,
            "metric_cutoff": 10,
            "features": names,
            "like_metric_for_calibration": "LikeNDCG@10",
            "composite_grid": COMPOSITES,
            "pointwise": CATBOOST_PARAMS,
            "pairwise": PAIRWISE_PARAMS,
            "ridge_l2": RIDGE_L2,
            "packages": {name: version(name) for name in ("catboost", "numpy", "pandas", "pyarrow")},
        },
        train={"requests": int(train.request_key.nunique()), "rows": len(train),
               "nonzero_target_rows": {name: int(np.count_nonzero(values)) for name, values in targets.items()}},
        calibration=calibration,
        selected_composite=chosen,
        seconds_fit_and_predict=timings,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = result["results"]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        csv_writer = csv.DictWriter(file, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(rows)


if __name__ == "__main__":
    main()
