"""Refit Qilin rerankers with and without the avatar-assisted position curves."""

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from compare import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CATBOOST_PARAMS,
    CUTOFF,
    _cluster_bootstrap,
    _rank_candidates,
    _request_starts,
    _validation_ndcg,
)
from engagement_eval import evaluate
from engagement_models import PAIRWISE_PARAMS, RIDGE_L2, pairwise_predictions, pointwise_predictions, ridge_predictions
from run import DATA_REVISION
from run_engagement import COMPOSITES, build_rank_features, composite, signals
from slates import _T1, _T2, load_slates_at_depths


MAX_POSITION_WEIGHT = 3.0
DEPTH = 30
EVENTS = ("click", "like")
LEGACY_CONTROLS = ("skip_penalty", "long_view", "dwell_log", "collect", "active", "click_dwell")


def event_weights(curves):
    """Return capped inverse relative exposure at the original rank."""
    return {event: np.minimum(MAX_POSITION_WEIGHT, 1 / np.asarray(curves[event]["e_by_position"]))
            for event in EVENTS}


def training_targets(train, curves, recipe):
    logged = signals(train)
    logged["composite"] = composite(logged, recipe)
    logged["click_dwell"] = (logged["click"] + 0.5 * logged["dwell_log"]) / 1.5
    targets = {"": {name: logged[name] for name in (*EVENTS, "composite", *LEGACY_CONTROLS)}}
    positions = train.position.to_numpy() - 1
    scale = sum(weight for name, weight in recipe.items() if name != "skip")
    for scheme, calibration in curves.items():
        weights = event_weights(calibration)
        adjusted = {event: logged[event] * weights[event][positions] for event in EVENTS}
        incremental_gain = sum(recipe[event] * (adjusted[event] - logged[event]) for event in EVENTS)
        adjusted["composite"] = logged["composite"] + incremental_gain / scale
        targets[scheme] = adjusted
    return targets


def fit_models(train, x_train, x_val, targets):
    models = {"point": pointwise_predictions, "ridge": ridge_predictions, "pair": pairwise_predictions}
    groups = train.request_key.to_numpy()
    predictions, timings, duplicates = {}, {}, {}
    observed_targets = {}
    for scheme, by_target in targets.items():
        suffix = f"_{scheme}" if scheme else ""
        for target, y in by_target.items():
            families = models if target in ("click", "composite") else ("point",)
            for family in families:
                name = f"{family}_{target}{suffix}"
                key = (family, target)
                previous = observed_targets.setdefault(key, [])
                duplicate = next((old for old, values in previous if np.array_equal(values, y)), None)
                if duplicate:
                    duplicates[name] = duplicate
                    predictions[name] = predictions[duplicate]
                    continue
                print(f"Fitting {name}", flush=True)
                started = perf_counter()
                extras = (groups,) if family == "pair" else ()
                predictions[name] = models[family](x_train, y, *extras, x_val)
                timings[name] = perf_counter() - started
                previous.append((name, y))
    return predictions, timings, duplicates


def bootstrap_metrics(scores, eligible, users):
    scores = np.where(eligible[:, None, :], scores, 0)
    mean = np.divide(scores.sum(axis=0), eligible.sum(axis=0),
                     out=np.full(scores.shape[1:], np.nan), where=eligible.sum(axis=0) > 0)
    return mean, _cluster_bootstrap(scores, eligible, users)


def paired_delta(means, replicates, candidate, baseline, metric, *, relative=False):
    point = means[candidate, metric] - means[baseline, metric]
    draws = replicates[:, candidate, metric] - replicates[:, baseline, metric]
    if relative:
        draws = np.divide(draws, replicates[:, baseline, metric], out=np.full_like(draws, np.nan),
                          where=replicates[:, baseline, metric] > 0)
        point = point / means[baseline, metric] if means[baseline, metric] > 0 else np.nan
    draws = draws[np.isfinite(draws)]
    low, high = np.quantile(draws, [0.025, 0.975]).tolist() if draws.size else (None, None)
    return {"delta": float(point) if np.isfinite(point) else None, "ci95_low": low, "ci95_high": high}


def paired_changes(models, metrics, means, replicates):
    for i, model in enumerate(models):
        for scheme in ("original", "avatar"):
            suffix = f"_{scheme}"
            if model.endswith(suffix):
                raw = model.removesuffix(suffix)
                comparisons = [raw, f"{raw}_original"] if scheme == "avatar" else [raw]
                for baseline in comparisons:
                    for k, metric in enumerate(metrics):
                        yield {"model": model, "baseline": baseline, "metric": metric,
                               **paired_delta(means, replicates, i, models.index(baseline), k)}


def raw_paired_changes(val, predictions):
    labels = val[["click", "like", "like"]].to_numpy(dtype=bool)
    scores, positives, starts = _validation_ndcg(val, predictions.values(), labels=labels)
    eligible = np.ones_like(positives, dtype=bool)
    eligible[:, -1] = positives[:, -1] > 0
    means, replicates = bootstrap_metrics(scores, eligible, val.user_id.iloc[starts])
    metrics = ("ClickNDCG@10", "LikeNDCG@10", "LikePositiveNDCG@10")
    return list(paired_changes(["qilin", *predictions], metrics, means, replicates))


def weighted_dcg(val, predictions, curves):
    starts = _request_starts(val)
    sizes = np.diff(starts, append=len(val))
    groups = np.repeat(np.arange(len(starts)), sizes)
    positions = val.position.to_numpy() - 1
    ranks = np.arange(len(val)) - starts[groups] + 1
    discounts = np.where(ranks <= CUTOFF, 1 / np.log2(ranks + 1), 0.0)
    models = ["qilin", *predictions]
    orders = [np.arange(len(val)), *(_rank_candidates(scores, groups) for scores in predictions.values())]
    scenarios = [(scheme, event) for scheme in curves for event in EVENTS]
    scores = np.empty((len(starts), len(orders), len(scenarios)))
    for k, (scheme, event) in enumerate(scenarios):
        gains = val[event].to_numpy() * event_weights(curves[scheme])[event][positions]
        for i, order in enumerate(orders):
            scores[:, i, k] = np.add.reduceat(gains[order] * discounts, starts)

    eligible = np.ones((len(starts), len(scenarios)), dtype=bool)
    means, draws = bootstrap_metrics(scores, eligible, val.user_id.iloc[starts])
    rows = []
    for k, (scheme, event) in enumerate(scenarios):
        for i, model in enumerate(models):
            difference = paired_delta(means, draws, i, 0, k)
            relative = paired_delta(means, draws, i, 0, k, relative=True)
            rows.append({"evaluation_curve": scheme, "action": event, "model": model,
                         "dcg_at_10_per_request": float(means[i, k]),
                         "delta_vs_qilin": difference["delta"], "ci95_low": difference["ci95_low"],
                         "ci95_high": difference["ci95_high"], "relative_delta_vs_qilin": relative["delta"],
                         "relative_ci95_low": relative["ci95_low"], "relative_ci95_high": relative["ci95_high"]})
    metrics = [f"{scheme}_{event}_weighted_DCG@10" for scheme, event in scenarios]
    return rows, list(paired_changes(models, metrics, means, draws))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test-identifiers", required=True, type=Path)
    parser.add_argument("--notes", required=True, nargs="+", type=Path)
    parser.add_argument("--curves", required=True, type=Path)
    parser.add_argument("--engagement-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    fit = json.loads(args.curves.read_text(encoding="utf-8"))
    curves = {"original": fit["baseline"], "avatar": fit["updated"]}
    previous = json.loads(args.engagement_summary.read_text(encoding="utf-8"))
    recipe_name = previous["selected_composite"]
    recipe = COMPOSITES[recipe_name]
    slates = load_slates_at_depths(args.train, args.test_identifiers, [CUTOFF, DEPTH], include_engagement=True)
    train, val = slates[DEPTH]
    val = val[val.request_key.isin(slates[CUTOFF][1].request_key)].reset_index(drop=True)
    print(f"Loaded {len(train):,} train pairs and {len(val):,} validation pairs", flush=True)
    x_train, x_val, features = build_rank_features(train, val, args.notes)
    targets = training_targets(train, curves, recipe)
    predictions, timings, duplicates = fit_models(train, x_train, x_val, targets)
    print("Scoring common validation slates and bootstrapping paired changes", flush=True)
    summary = evaluate(val, predictions)
    weighted_rows, weighted_changes = weighted_dcg(val, predictions, curves)
    changes = raw_paired_changes(val, predictions) + weighted_changes
    summary.update(
        config={
            "dataset_revision": DATA_REVISION,
            "split_boundaries_utc_unix": [_T1, _T2],
            "candidate_depth": DEPTH,
            "cutoff": CUTOFF,
            "features": features,
            "curve_file": str(args.curves),
            "composite_recipe": recipe_name,
            "composite_weights": recipe,
            "corrected_signals": list(EVENTS),
            "position_weight_cap": MAX_POSITION_WEIGHT,
            "weighted_evaluation": "Mean DCG at rank 10 without IDCG normalization. "
                                   "Logged position weights the historical action before applying new rank discount. "
                                   "Each evaluation curve is shared across all models.",
            "bootstrap": {"unit": "user_id", "replicates": BOOTSTRAP_REPLICATES,
                          "seed": BOOTSTRAP_SEED, "curves_held_fixed": True},
            "pointwise": CATBOOST_PARAMS,
            "pairwise": PAIRWISE_PARAMS,
            "ridge_l2": RIDGE_L2,
            "packages": {name: version(name) for name in ("catboost", "numpy", "pandas", "pyarrow")},
        },
        train={"requests": int(train.request_key.nunique()), "rows": len(train)},
        seconds_fit_and_predict=timings,
        duplicate_training_targets=duplicates,
        weighted_dcg=weighted_rows,
        paired_changes=changes,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(args.output / "summary.csv", summary["results"])
    write_csv(args.output / "weighted_dcg.csv", weighted_rows)
    write_csv(args.output / "paired_changes.csv", changes)


if __name__ == "__main__":
    main()
