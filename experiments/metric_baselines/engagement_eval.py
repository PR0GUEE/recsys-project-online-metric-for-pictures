"""Evaluate engagement rerankers on the same candidate slates."""

import numpy as np

from compare import _binary_signals, _cluster_bootstrap, _validation_ndcg


_METRICS = {
    "LikeNDCG@10": ("like", False),
    "LikePositiveNDCG@10": ("like", True),
    "ClickNDCG@10": ("click", False),
    "LongViewNDCG@10": ("long_view", False),
    "CollectNDCG@10": ("collect", True),
}


def evaluate(val, predictions: dict[str, np.ndarray], *, intervals=True):
    """Score original Qilin order and rerankers on contiguous validation requests.

    Prediction arrays follow validation row order. Higher scores rank first,
    with ties resolved by original candidate order. IDCG uses all candidates
    in each request, regardless of model. LikePositiveNDCG and CollectNDCG
    use requests with at least one corresponding positive. Other metrics
    use all requests, with zero gain for requests without positives.
    """
    signals = _binary_signals(val)
    signals["like"] = val["like"].to_numpy(dtype=bool)
    metrics = dict(_METRICS)
    if "share" in val and "comment" in val:
        signals["active_action"] = (
            signals["like"]
            | signals["collect"]
            | val["share"].to_numpy(dtype=bool)
            | val["comment"].to_numpy(dtype=bool)
        )
        metrics["ActiveActionNDCG@10"] = ("active_action", False)

    signal_names = list(signals)
    unique_labels = np.column_stack(list(signals.values()))
    metric_signal_indices = [signal_names.index(signal) for signal, _ in metrics.values()]
    unique_scores, unique_counts, starts = _validation_ndcg(
        val, predictions.values(), labels=unique_labels
    )
    scores = unique_scores[:, :, metric_signal_indices]
    positive_counts = unique_counts[:, metric_signal_indices]
    positive_only = np.array([conditional for _, conditional in metrics.values()])
    eligible = (~positive_only) | (positive_counts > 0)
    users = val["user_id"].iloc[starts]
    replicates = None
    if intervals:
        replicates = (
            _cluster_bootstrap(scores, eligible, users)
            if len(starts)
            else np.empty((0, *scores.shape[1:]))
        )

    coverage = {
        "rows": len(val),
        "requests": len(starts),
        "users": int(users.nunique()),
        "positive_rows": {name: int(labels.sum()) for name, labels in signals.items()},
        "positive_requests": dict(
            zip(signal_names, np.count_nonzero(unique_counts, axis=0).tolist())
        ),
        "clicked_rows_with_page_time": int(
            (signals["click"] & (val["page_time"].to_numpy(dtype=float) >= 0)).sum()
        ),
    }
    rows = _summary_rows(
        scores, positive_counts, eligible, replicates, ["qilin", *predictions], metrics
    )
    return {"coverage": coverage, "results": rows}


def _summary_rows(scores, positive_counts, eligible, replicates, models, metrics):
    eligible_counts = eligible.sum(axis=0)
    positive_requests = np.count_nonzero(positive_counts, axis=0)
    positive_rows = positive_counts.sum(axis=0)
    means = np.divide(
        scores.sum(axis=0),
        eligible_counts,
        out=np.full(scores.shape[1:], np.nan),
        where=eligible_counts > 0,
    )
    baselines = {name: models.index(name) for name in ("qilin", "point_click") if name in models}
    rows = []
    for model_index, model in enumerate(models):
        for metric_index, metric in enumerate(metrics):
            value = means[model_index, metric_index]
            row = {
                "model": model,
                "metric": metric,
                "ndcg": float(value) if np.isfinite(value) else None,
                "evaluated_requests": int(eligible_counts[metric_index]),
                "positive_requests": int(positive_requests[metric_index]),
                "positive_rows": int(positive_rows[metric_index]),
            }
            for baseline, baseline_index in baselines.items():
                key = f"delta_vs_{baseline}"
                delta = value - means[baseline_index, metric_index]
                row[key] = float(delta) if np.isfinite(delta) else None
                if replicates is not None:
                    paired = (
                        replicates[:, model_index, metric_index]
                        - replicates[:, baseline_index, metric_index]
                    )
                    paired = paired[np.isfinite(paired)]
                    low, high = (
                        np.quantile(paired, [0.025, 0.975]).tolist() if len(paired) else (None, None)
                    )
                    row[f"{key}_ci95_low"] = low
                    row[f"{key}_ci95_high"] = high
                    row[f"{key}_bootstrap_replicates"] = len(paired)
            rows.append(row)
    return rows
