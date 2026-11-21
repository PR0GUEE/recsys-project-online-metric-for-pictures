"""Compare Qilin candidate depths on a shared validation cohort at NDCG@10."""

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from compare import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CATBOOST_PARAMS,
    CUTOFF,
    METRIC_SIGNALS,
    METRICS,
    TARGET_DEFINITIONS,
    _binary_signals,
    _cluster_bootstrap,
    _coverage,
    _evaluation_mask,
    _rank_candidates,
    _request_starts,
    _summary_rows,
    _validation_ndcg,
    fit_predictions,
)
from features import build_features
from run import DATA_REVISION
from slates import _T1, _T2, load_slates_at_depths


def _note_index(frame):
    return pd.MultiIndex.from_frame(frame[["request_key", "note_id"]])


def _positive_counts(labels, starts):
    return np.add.reduceat(labels, starts, axis=0)


def _promoted_candidates(validation, predictions, added, labels, starts, models):
    sizes = np.diff(starts, append=len(validation))
    groups = np.repeat(np.arange(len(starts)), sizes)
    in_top10 = np.arange(len(validation)) - np.repeat(starts, sizes) < CUTOFF
    orders = [np.arange(len(validation))]
    orders.extend(_rank_candidates(scores, groups) for scores in predictions)
    results = {}
    for model, order in zip(models, orders):
        is_promoted = in_top10 & added[order]
        counts = _positive_counts(labels[order] & is_promoted[:, None], starts)
        results[model] = {
            "candidates": int(is_promoted.sum()),
            "requests": int(np.logical_or.reduceat(is_promoted, starts).sum()),
            "positive_candidates": dict(zip(METRIC_SIGNALS, counts.sum(axis=0).tolist())),
        }
    return results


def _depth_delta_rows(rows, scores, eligible, replicates, top10_scores, top10_replicates):
    denominators = eligible.sum(axis=0)
    differences = scores - top10_scores
    means = np.divide(
        differences.sum(axis=0),
        denominators,
        out=np.full(differences.shape[1:], np.nan),
        where=denominators > 0,
    )
    paired = replicates - top10_replicates
    for model_index in range(means.shape[0]):
        for metric_index in range(means.shape[1]):
            row = rows[model_index * len(METRICS) + metric_index]
            delta = means[model_index, metric_index]
            samples = paired[:, model_index, metric_index]
            samples = samples[np.isfinite(samples)]
            low, high = np.quantile(samples, [0.025, 0.975]).tolist() if len(samples) else (None, None)
            row.update(
                delta_vs_same_model_depth_10=float(delta) if np.isfinite(delta) else None,
                delta_vs_same_model_depth_10_ci95_low=low,
                delta_vs_same_model_depth_10_ci95_high=high,
            )


def _cross_depth_rows(depths, evaluation, scores_by_depth, counts, eligible, users, models):
    replicates = _cluster_bootstrap(np.concatenate(scores_by_depth, axis=1), eligible, users)
    replicates_by_depth = np.split(replicates, len(depths), axis=1)
    for depth, scores, replicates in zip(depths, scores_by_depth, replicates_by_depth):
        rows = _summary_rows(scores, counts, eligible, replicates, models)
        _depth_delta_rows(rows, scores, eligible, replicates, scores_by_depth[0], replicates_by_depth[0])
        yield from ({"depth": depth, "evaluation": evaluation, **row} for row in rows)


def run(args):
    slates = load_slates_at_depths(args.train, args.test_identifiers, args.depths)
    reference_depth = max(args.depths)
    fit_10, validation_10 = slates[CUTOFF]
    validation_requests = validation_10["request_key"].drop_duplicates()
    if args.smoke:
        fit_requests = fit_10["request_key"].drop_duplicates().head(500)
        validation_requests = validation_requests.head(300)
    slates = {
        depth: (
            train[train["request_key"].isin(fit_requests)] if args.smoke else train,
            validation[validation["request_key"].isin(validation_requests)],
        )
        for depth, (train, validation) in slates.items()
    }

    reference = slates[reference_depth][1]
    reference_index = _note_index(reference)
    reference_signals = _binary_signals(reference)
    reference_labels = np.column_stack([reference_signals[name] for name in METRIC_SIGNALS])
    reference_starts = _request_starts(reference)
    reference_counts = _positive_counts(reference_labels, reference_starts)
    reference_eligible = _evaluation_mask(reference_counts)
    reference_users = reference["user_id"].iloc[reference_starts]
    top10_candidates = _note_index(slates[CUTOFF][1])

    models = ["qilin", *TARGET_DEFINITIONS]
    results, coverage, common_scores = [], {}, []
    for depth, (train, validation) in slates.items():
        print(f"Depth {depth}: {len(train):,} train pairs, {len(validation):,} validation pairs", flush=True)
        x_train, x_val, feature_names = build_features(train, validation, args.notes)
        targets, predictions = fit_predictions(train, x_train, x_val)

        scores, counts, starts = _validation_ndcg(validation, predictions)
        eligible = _evaluation_mask(counts)
        users = validation["user_id"].iloc[starts]
        replicates = _cluster_bootstrap(scores, eligible, users)
        results.extend(
            {"depth": depth, "evaluation": "within_depth", **row}
            for row in _summary_rows(scores, counts, eligible, replicates, models)
        )

        notes = _note_index(validation)
        fixed_labels = reference_labels[reference_index.get_indexer(notes)]
        standardized, _, _ = _validation_ndcg(
            validation, predictions, labels=fixed_labels, ideal_positive_counts=reference_counts
        )
        common_scores.append(standardized)
        added = ~notes.isin(top10_candidates)
        added_counts = _positive_counts(fixed_labels & added[:, None], starts)
        missing_text_index = feature_names.index("note_text_missing")
        coverage[depth] = {
            "train": _coverage(train, _request_starts(train)),
            "validation": _coverage(validation, starts),
            "features": feature_names,
            "rows_without_note_text": {
                "train": int(x_train[:, missing_text_index].sum()),
                "validation": int(x_val[:, missing_text_index].sum()),
            },
            "negative_skip_penalty_train_rows": int(np.count_nonzero(targets["skip_penalty"] < 0)),
            "positive_dwell_log_train_rows": int(np.count_nonzero(targets["dwell_log"] > 0)),
            "added_validation_candidates": int(added.sum()),
            "validation_requests_with_added_candidates": int(np.logical_or.reduceat(added, starts).sum()),
            "added_positive_candidates": dict(zip(METRIC_SIGNALS, added_counts.sum(axis=0).tolist())),
            "validation_requests_with_added_positives": dict(
                zip(METRIC_SIGNALS, (added_counts > 0).sum(axis=0).tolist())
            ),
            "promoted_into_top10": _promoted_candidates(
                validation, predictions, added, fixed_labels, starts, models
            ),
        }

    # The deepest fit scored every candidate in the restricted slates as well.
    fixed_model_scores = []
    for _, validation in slates.values():
        rows_in_reference = reference_index.get_indexer(_note_index(validation))
        scores, _, _ = _validation_ndcg(
            validation,
            [scores[rows_in_reference] for scores in predictions],
            labels=reference_labels[rows_in_reference],
            ideal_positive_counts=reference_counts,
        )
        fixed_model_scores.append(scores)
    for evaluation, scores in (
        ("fixed_deepest_slate", common_scores),
        ("fixed_deepest_model", fixed_model_scores),
    ):
        results.extend(
            _cross_depth_rows(
                args.depths, evaluation, scores, reference_counts, reference_eligible, reference_users, models
            )
        )

    metadata = {
        "dataset": "THUIR/Qilin",
        "dataset_revision": DATA_REVISION,
        "split_boundaries_utc_unix": [_T1, _T2],
        "smoke": args.smoke,
        "depths": args.depths,
        "cutoff": CUTOFF,
        "validation_cohort": "Requests with an eligible depth-10 slate",
        "within_depth": "Each depth uses its own labels, ideal DCG and collect-positive requests",
        "fixed_deepest_slate": {
            "reference_depth": reference_depth,
            "definition": "Fit and score at each depth. Use shared note-level labels, ideal DCG "
            "and collect-positive request support from the deepest validation slate",
        },
        "fixed_deepest_model": "Reuse predictions from the deepest fit at every depth "
        "with the same evaluation as fixed_deepest_slate",
        "packages": {name: version(name) for name in ("catboost", "numpy", "pandas", "pyarrow")},
        "runtime": {"python": platform.python_version(), "machine": platform.machine()},
        "catboost_parameters": CATBOOST_PARAMS,
        "training_targets": TARGET_DEFINITIONS,
        "bootstrap": {
            "unit": "user_id",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "method": "Paired 95% percentile interval with request-weighted macro after user resampling",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    summary = {"config": metadata, "coverage": coverage, "results": results}
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as output:
        columns = list(next(row for row in results if row["evaluation"] == "fixed_deepest_slate"))
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test-identifiers", required=True, type=Path)
    parser.add_argument("--notes", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--depths", nargs="+", type=int, default=[10, 20, 30])
    parser.add_argument("--smoke", action="store_true", help="Run on 500 train and 300 validation requests")
    args = parser.parse_args()
    if min(args.depths) < CUTOFF:
        parser.error(f"Candidate depth must be at least {CUTOFF}")
    args.depths = sorted(set(args.depths) | {CUTOFF})
    run(args)


if __name__ == "__main__":
    main()
