"""Replay the existing Qilin rerankers with train-derived visibility weights."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from compare import BOOTSTRAP_REPLICATES, _cluster_bootstrap, _rank_candidates, _request_starts
from engagement_eval import evaluate
from engagement_models import pointwise_predictions
from run_engagement import COMPOSITES, build_rank_features, composite, signals
from slates import load_slates_at_depths


def score_scenarios(curves):
    yield "no_adjustment", "click", np.ones(30)
    yield "no_adjustment", "like", np.ones(30)
    for name, curve in curves.items():
        response = "like" if name.endswith("_like") else "click"
        inverse = 1 / np.asarray(curve["e_by_position"], dtype=float)
        yield f"{name}_cap3", response, np.minimum(inverse, 3.0)
        yield f"{name}_uncapped", response, inverse


def replay(val, predictions, curves):
    starts = _request_starts(val)
    group_sizes = np.diff(starts, append=len(val))
    groups = np.repeat(np.arange(len(starts)), group_sizes)
    logged_position = val.position.to_numpy(dtype=int)
    new_rank = np.arange(len(val)) - starts[groups] + 1
    discount = np.where(new_rank <= 10, 1 / np.log2(new_rank + 1), 0.0)
    models = ["qilin", *predictions]
    orders = [np.arange(len(val)), *(_rank_candidates(x, groups) for x in predictions.values())]
    scenarios = list(score_scenarios(curves))

    scores = np.empty((len(starts), len(models), len(scenarios)))
    for k, (_, action, weights) in enumerate(scenarios):
        # Each item's weight stays tied to its historical position after it moves.
        gain = val[action].to_numpy(dtype=float) * weights[logged_position - 1]
        for j, order in enumerate(orders):
            scores[:, j, k] = np.add.reduceat(gain[order] * discount, starts)

    replicates = _cluster_bootstrap(
        scores, np.ones((len(starts), len(scenarios)), dtype=bool), val.user_id.iloc[starts]
    )
    means = scores.mean(axis=0)
    rows = []
    for k, (scenario, action, weights) in enumerate(scenarios):
        base = means[0, k]
        for j, model in enumerate(models):
            delta = replicates[:, j, k] - replicates[:, 0, k]
            ratio_draws = delta / replicates[:, 0, k]
            lower, upper = np.quantile(delta, [0.025, 0.975])
            rel_lower, rel_upper = np.quantile(ratio_draws, [0.025, 0.975])
            rows.append({
                "scenario": scenario,
                "action": action,
                "model": model,
                "dcg_at_10_per_request": float(means[j, k]),
                "delta_vs_qilin": float(means[j, k] - base),
                "ci95_low": float(lower),
                "ci95_high": float(upper),
                "relative_delta_vs_qilin": float(means[j, k] / base - 1),
                "relative_ci95_low": float(rel_lower),
                "relative_ci95_high": float(rel_upper),
                "max_position_weight": float(weights.max()),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test-identifiers", required=True, type=Path)
    parser.add_argument("--notes", required=True, nargs="+", type=Path)
    parser.add_argument("--curves", required=True, type=Path)
    parser.add_argument("--engagement-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    calibration = json.loads(args.curves.read_text(encoding="utf-8"))
    earlier_run = json.loads(args.engagement_summary.read_text(encoding="utf-8"))
    slates = load_slates_at_depths(args.train, args.test_identifiers, [10, 30], include_engagement=True)
    train, val = slates[30]
    val = val[val.request_key.isin(slates[10][1].request_key)].reset_index(drop=True)
    print(f"Replaying existing models on {val.request_key.nunique():,} validation requests", flush=True)
    x_train, x_val, _ = build_rank_features(train, val, args.notes)
    y = signals(train)
    recipe_name = earlier_run["selected_composite"]
    predictions = {
        "point_click": pointwise_predictions(x_train, y["click"], x_val),
        "point_composite": pointwise_predictions(x_train, composite(y, COMPOSITES[recipe_name]), x_val),
    }
    rows = replay(val, predictions, calibration["curves"])
    result = {
        "validation_requests": int(val.request_key.nunique()),
        "validation_pairs": len(val),
        "candidate_depth": 30,
        "cutoff": 10,
        "replayed_composite": recipe_name,
        "curve_file": str(args.curves),
        "evaluation": "Average DCG over requests. Inverse visibility weight uses original position. "
                      "Standard log rank discount uses reranked position. No ideal DCG normalization.",
        "bootstrap": {"unit": "user_id", "replicates": BOOTSTRAP_REPLICATES, "curves_held_fixed": True},
        "raw_ndcg_replay": evaluate(val, predictions, intervals=False),
        "results": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
