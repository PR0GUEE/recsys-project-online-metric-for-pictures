"""Compare predeclared training targets on the same validation candidates."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


CATBOOST_PARAMS = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": 42,
    "loss_function": "RMSE",
    "verbose": False,
    "allow_writing_files": False,
}
TARGET_DEFINITIONS = {
    "click": "Binary click",
    "skip_penalty": (
        "click - (1 - click) * I(a click at a strictly greater original position "
        "in the same request) / log2(1 + original position)"
    ),
    "long_view": "click * I(page_time >= 10 seconds)",
    "dwell_log": "click * log1p(clip(page_time in seconds, 0, 60)) / log(61)",
    "collect": "Binary save",
}
METRICS = ("ClickNDCG@10", "LongViewNDCG@10", "CollectNDCG@10")
METRIC_SIGNALS = ("click", "long_view", "collect")
CUTOFF = 10
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261121


def _request_starts(frame):
    column = "request_key" if "request_key" in frame else "request_id"
    requests = frame[column].to_numpy()
    starts = np.ones(len(frame), dtype=bool)
    starts[1:] = requests[1:] != requests[:-1]
    return np.flatnonzero(starts)


def _binary_signals(frame):
    click = frame["click"].to_numpy(dtype=bool)
    page_time = frame["page_time"].to_numpy(dtype=float)
    return {
        "click": click,
        "long_view": click & (page_time >= 10),
        "collect": frame["collect"].to_numpy(dtype=bool),
    }


def signal_targets(frame):
    """Build the five training targets in the supplied candidate row order."""
    starts = _request_starts(frame)
    sizes = np.diff(starts, append=len(frame))
    signals = _binary_signals(frame)
    click = signals["click"]
    positions = frame["position"].to_numpy(dtype=int)
    page_time = frame["page_time"].to_numpy(dtype=float)
    last_click = np.maximum.reduceat(np.where(click, positions, 0), starts)
    skipped = (~click) & (positions < np.repeat(last_click, sizes))
    return {
        "click": click.astype(float),
        "skip_penalty": click.astype(float) - skipped / np.log2(1 + positions),
        "long_view": signals["long_view"].astype(float),
        "dwell_log": click * np.log1p(np.clip(page_time, 0, 60)) / np.log(61),
        "collect": signals["collect"].astype(float),
    }


def _rank_candidates(scores, group_index):
    # Equal predictions retain the candidate order within each request.
    return np.lexsort((np.arange(len(scores)), -scores, group_index))


def _request_ndcg(labels, starts, order, discounts, ideal):
    gains = labels[order] * discounts[:, None]
    dcg = np.add.reduceat(gains, starts, axis=0)
    return np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)


def _validation_ndcg(val, predictions, *, labels=None, ideal_positive_counts=None):
    starts = _request_starts(val)
    sizes = np.diff(starts, append=len(val))
    group_index = np.repeat(np.arange(len(starts)), sizes)
    if labels is None:
        labels_by_signal = _binary_signals(val)
        labels = np.column_stack([labels_by_signal[name] for name in METRIC_SIGNALS])
    positive_counts = np.add.reduceat(labels, starts, axis=0)
    discount_by_rank = np.concatenate((1 / np.log2(np.arange(2, CUTOFF + 2)), [0]))
    ranks = np.arange(len(val)) - np.repeat(starts, sizes)
    discounts = discount_by_rank[np.minimum(ranks, CUTOFF)]
    ideal_by_count = np.concatenate(([0], np.cumsum(discount_by_rank[:CUTOFF])))
    if ideal_positive_counts is None:
        ideal_positive_counts = positive_counts
    ideal = ideal_by_count[np.minimum(ideal_positive_counts, CUTOFF)]

    orders = [np.arange(len(val))]
    orders.extend(_rank_candidates(scores, group_index) for scores in predictions)
    scores_by_model = [_request_ndcg(labels, starts, order, discounts, ideal) for order in orders]
    return np.stack(scores_by_model, axis=1), positive_counts, starts


def _coverage(frame, starts):
    signals = _binary_signals(frame)
    click = signals["click"]
    measured_page_time = frame["page_time"].to_numpy(dtype=float) >= 0
    return {
        "rows": len(frame),
        "requests": len(starts),
        "users": int(frame["user_id"].nunique()),
        "positive_rows": {name: int(labels.sum()) for name, labels in signals.items()},
        "positive_requests": {
            name: int(np.logical_or.reduceat(labels, starts).sum())
            for name, labels in signals.items()
        },
        "clicked_rows_with_page_time": int((click & measured_page_time).sum()),
    }


def _evaluation_mask(positive_counts):
    eligible = np.ones_like(positive_counts, dtype=bool)
    collect_index = METRIC_SIGNALS.index("collect")
    eligible[:, collect_index] = positive_counts[:, collect_index] > 0
    return eligible


def _cluster_bootstrap(scores, eligible, users):
    """Generate paired macro means by sampling whole users with replacement."""
    request_count, model_count, metric_count = scores.shape
    user_index, distinct_users = pd.factorize(users, use_na_sentinel=False)
    user_count = len(distinct_users)
    flattened_scores = scores.reshape(request_count, model_count * metric_count)
    totals_by_user = np.zeros((user_count, model_count * metric_count + metric_count))
    np.add.at(totals_by_user, user_index, np.column_stack((flattened_scores, eligible)))

    probabilities = np.full(user_count, 1 / user_count)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    replicates = np.full((BOOTSTRAP_REPLICATES, model_count, metric_count), np.nan)
    batch_size = 64
    for start in range(0, BOOTSTRAP_REPLICATES, batch_size):
        end = min(start + batch_size, BOOTSTRAP_REPLICATES)
        draws = rng.multinomial(user_count, probabilities, size=end - start)
        weighted_totals = draws @ totals_by_user
        numerators = weighted_totals[:, :-metric_count].reshape(-1, model_count, metric_count)
        denominators = weighted_totals[:, None, -metric_count:]
        np.divide(numerators, denominators, out=replicates[start:end], where=denominators > 0)
    return replicates


def _summary_rows(scores, positive_counts, eligible, replicates, models):
    eligible_counts = eligible.sum(axis=0)
    means = np.divide(
        scores.sum(axis=0),
        eligible_counts,
        out=np.full(scores.shape[1:], np.nan),
        where=eligible_counts > 0,
    )
    rows = []
    for model_index, model in enumerate(models):
        for metric_index, metric in enumerate(METRICS):
            value = means[model_index, metric_index]
            row = {
                "model": model,
                "metric": metric,
                "ndcg": float(value) if np.isfinite(value) else None,
                "evaluated_requests": int(eligible_counts[metric_index]),
                "positive_requests": int(np.count_nonzero(positive_counts[:, metric_index])),
                "positive_rows": int(positive_counts[:, metric_index].sum()),
            }
            for baseline in ("qilin", "click"):
                baseline_index = models.index(baseline)
                delta = value - means[baseline_index, metric_index]
                paired = (
                    replicates[:, model_index, metric_index]
                    - replicates[:, baseline_index, metric_index]
                )
                paired = paired[np.isfinite(paired)]
                low, high = np.quantile(paired, [0.025, 0.975]).tolist() if len(paired) else (None, None)
                row[f"delta_vs_{baseline}"] = float(delta) if np.isfinite(delta) else None
                row[f"delta_vs_{baseline}_ci95_low"] = low
                row[f"delta_vs_{baseline}_ci95_high"] = high
                row[f"delta_vs_{baseline}_bootstrap_replicates"] = len(paired)
            rows.append(row)
    return rows


def fit_predictions(train, x_train, x_val):
    """Fit the predeclared targets and return them with their validation predictions."""
    from catboost import CatBoostRegressor

    targets = signal_targets(train)
    predictions = []
    for name, target in targets.items():
        print(f"Fitting CatBoost on {name}", flush=True)
        model = CatBoostRegressor(**CATBOOST_PARAMS)
        model.fit(x_train, target)
        predictions.append(model.predict(x_val))
    return targets, predictions


def compare(train, val, x_train, x_val, out_dir):
    """Fit one model per training target and evaluate shared validation metrics.

    Each contiguous request starts in original order. The feature matrices must
    follow the same positional row order as their corresponding frames.
    """
    from catboost import __version__ as catboost_version

    targets, predictions = fit_predictions(train, x_train, x_val)
    scores, positive_counts, val_starts = _validation_ndcg(val, predictions)
    eligible = _evaluation_mask(positive_counts)
    replicates = _cluster_bootstrap(scores, eligible, val["user_id"].iloc[val_starts])
    rows = _summary_rows(scores, positive_counts, eligible, replicates, ["qilin", *targets])
    coverage = {
        "train": _coverage(train, _request_starts(train)),
        "validation": _coverage(val, val_starts),
    }
    coverage["train"]["negative_skip_penalty_rows"] = int(np.count_nonzero(targets["skip_penalty"] < 0))
    coverage["train"]["positive_dwell_log_rows"] = int(np.count_nonzero(targets["dwell_log"] > 0))
    summary = {
        "config": {
            "catboost": {"version": catboost_version, "parameters": CATBOOST_PARAMS},
            "feature_count": x_train.shape[1],
            "training_targets": TARGET_DEFINITIONS,
            "qilin_baseline": "Original candidate order",
            "tie_break": "Original candidate order",
            "missing_page_time": "-1 seconds, treated as zero for long_view and dwell_log",
            "metrics": {
                "ClickNDCG@10": "Binary clicks, macro across all requests, zero without clicks",
                "LongViewNDCG@10": "Binary long views, macro across all requests, zero without long views",
                "CollectNDCG@10": "Binary saves, macro across requests with at least one save",
            },
            "bootstrap": {
                "unit": "user_id",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "confidence_level": 0.95,
                "method": "Paired percentile interval with request-weighted macro after user resampling",
            },
        },
        "coverage": coverage,
        "results": rows,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary
