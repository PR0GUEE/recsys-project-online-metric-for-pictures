"""Replay the composite reward with a click curve fitted only on avatar queries."""

import argparse
import json
from pathlib import Path

from compare import CUTOFF
from engagement_models import pointwise_predictions
from run_avatar_rerank import DEPTH, training_targets, write_csv
from run_engagement import COMPOSITES, build_rank_features
from run_reward_ablation import MATCHED_CONTROLS, comparisons
from slates import load_slates_at_depths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test-identifiers", required=True, type=Path)
    parser.add_argument("--notes", required=True, nargs="+", type=Path)
    parser.add_argument("--curves", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    calibration = json.loads(args.curves.read_text(encoding="utf-8"))
    pooled = calibration["updated"]
    avatar_click = calibration["avatar_only"]["click"]
    training_curves = {
        "point_global": calibration["baseline"],
        "point_pooled": pooled,
        "point_avatar_only_click": {"click": avatar_click, "like": pooled["like"]},
    }

    slates = load_slates_at_depths(args.train, args.test_identifiers, [CUTOFF, DEPTH], include_engagement=True)
    train, val = slates[DEPTH]
    val = val.loc[val.request_key.isin(slates[CUTOFF][1].request_key)].reset_index(drop=True)
    print(f"Loaded {len(train):,} train pairs and {len(val):,} validation pairs", flush=True)
    x_train, x_val, features = build_rank_features(train, val, args.notes)
    recipe = "strong_actions"
    targets = training_targets(train, training_curves, COMPOSITES[recipe])

    predictions = {}
    for name, target in targets.items():
        model = name or "point_raw"
        print(f"Fitting {model}", flush=True)
        predictions[model] = pointwise_predictions(x_train, target["composite"], x_val)

    MATCHED_CONTROLS.update(point_pooled="point_global", point_avatar_only_click="point_pooled")
    results = comparisons(val, predictions, pooled)
    summary = {
        "train_rows": len(train),
        "validation_rows": len(val),
        "validation_requests": int(val.request_key.nunique()),
        "features": features,
        "target": recipe,
        "evaluation_curves": "updated fixed common curves",
        "training_curves": {
            name: {event: curve["exponent"] for event, curve in curves.items()}
            for name, curves in training_curves.items()
        },
        "method": "Switch click curve from pooled to avatar-only, holding the like curve, candidate slates, "
        "training target components and model fixed. Avatar-only is an exploratory sensitivity.",
        "support": {
            key: avatar_click[key]
            for key in ("matched_units", "matched_impressions", "matched_clicks", "cluster_robust_95_ci")
        },
        "results": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(args.output / "comparisons.csv", results)


if __name__ == "__main__":
    main()
