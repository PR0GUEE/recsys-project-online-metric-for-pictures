"""Compare relevance models without a position feature and Qilin-anchored policies."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare import _rank_candidates, _request_starts
from engagement_eval import evaluate
from engagement_models import pointwise_predictions
from features import _normalize_text
from position_avatar_fit import AVATAR_GROUPS
from position_fit import fit_matched_curve
from position_free_click import predict_click_models
from position_free_preferences import predict_preference_models
from run_avatar_rerank import write_csv
from run_engagement import build_rank_features, split_calibration
from run_reward_ablation import MATCHED_CONTROLS, comparisons
from slates import load_slates_at_depths


FIXED_STRENGTH = 0.16
STRENGTHS = (0.0, 0.08, FIXED_STRENGTH, 0.30, 0.60)
CLICK_METRIC = "ClickNDCG@10"
LIKE_METRIC = "LikePositiveNDCG@10"
MATCHED_PAIRS = {
    "blind_ips_click": "blind_raw_click",
    "blind_exposure_click": "blind_raw_click",
    "blind_pair_skip": "blind_pair_control",
}


def position_curve_for_calibration(train):
    aliases = {form: group[0] for group in AVATAR_GROUPS.values() for form in group}
    query = train["query"].map(_normalize_text).replace(aliases)
    return fit_matched_curve(train.assign(query=query), "click")[0]


def relevance_models(x_train, train, x_predict, click_curve):
    predictions = predict_click_models(x_train, train, x_predict, click_curve)
    pair_predictions, support = predict_preference_models(x_train, train, x_predict, click_curve)
    return {**predictions, **pair_predictions}, support


def anchor_inputs(frame, score, reference_scores=None):
    """Translate relevance to midranks and choose the order to refine."""
    group = pd.Series(score).groupby(frame.request_key.to_numpy(), sort=False)
    rank = group.rank(method="average").to_numpy()
    size = group.transform("size").to_numpy()
    relative_score = (rank - 1) / np.maximum(1, size - 1)
    positions = frame.position.to_numpy()
    if reference_scores is not None:
        starts = _request_starts(frame)
        groups = np.repeat(np.arange(len(starts)), np.diff(starts, append=len(frame)))
        order = _rank_candidates(reference_scores, groups)
        positions = np.empty(len(frame), dtype=int)
        positions[order] = np.arange(len(frame)) - starts[groups] + 1
    return -np.log(positions), relative_score


def choose_strength(val, blind_predictions, reference_scores=None):
    lookup = {}
    for model, scores in blind_predictions.items():
        baseline, rank_score = anchor_inputs(val, scores, reference_scores)
        for strength in STRENGTHS:
            lookup[model, strength] = baseline + strength * rank_score

    names = {f"{model}_{strength:g}": (model, strength) for model, strength in lookup}
    results = evaluate(val, {f"{m}_{s:g}": scores for (m, s), scores in lookup.items()}, intervals=False)
    values = {(row["model"], row["metric"]): row["ndcg"] for row in results["results"]}
    control_name = "qilin" if reference_scores is None else f"{next(iter(blind_predictions))}_0"
    control_like = values[control_name, LIKE_METRIC]
    rows = []
    for label, (model, strength) in names.items():
        click = values[label, CLICK_METRIC]
        like = values[label, LIKE_METRIC]
        rows.append({"model": model, "strength": strength, "click_ndcg": click, "like_ndcg_positive": like,
                     "qualified": like >= control_like - 1e-12})

    def objective(row):
        return (row["click_ndcg"], row["like_ndcg_positive"], -row["strength"])

    selected = {}
    for model in blind_predictions:
        picked = max((row for row in rows if row["model"] == model and row["qualified"]), key=objective)
        selected[model] = picked
    overall = max(selected.values(), key=objective)
    return selected, overall, rows, {"click_ndcg": values[control_name, CLICK_METRIC], "like_ndcg_positive": control_like}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test-identifiers", type=Path, required=True)
    parser.add_argument("--notes", type=Path, nargs="+", required=True)
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    slates = load_slates_at_depths(args.train, args.test_identifiers, [10, 30], include_engagement=True)
    train, val = slates[30]
    val = val.loc[val.request_key.isin(slates[10][1].request_key)].reset_index(drop=True)
    fit, calibrate, cutoff = split_calibration(train)
    print(f"Full train {train.request_key.nunique():,} requests; validation {val.request_key.nunique():,}", flush=True)
    x_train, x_val, features = build_rank_features(train, val, args.notes)
    blind_indices = [index for index, name in enumerate(features) if name != "original_qilin_position"]
    features_without_position = [features[index] for index in blind_indices]
    blind_train, blind_val = x_train[:, blind_indices], x_val[:, blind_indices]

    fit_mask = train.request_key.isin(fit.request_key).to_numpy()
    calibration_mask = train.request_key.isin(calibrate.request_key).to_numpy()
    print("Fitting the position-free models on the earlier part of train", flush=True)
    inner_curve = position_curve_for_calibration(fit)
    inner_predictions, inner_pair_support = relevance_models(
        blind_train[fit_mask], fit, blind_train[calibration_mask], inner_curve
    )
    selected, winner, calibration, inner_qilin = choose_strength(calibrate, inner_predictions)
    inner_reference = pointwise_predictions(x_train[fit_mask], fit.click.to_numpy(dtype=float), x_train[calibration_mask])
    stacked, stacked_winner, stacked_calibration, inner_baseline = choose_strength(calibrate, inner_predictions, inner_reference)
    print(f"Frozen Qilin-anchored winner: {winner['model']} at strength {winner['strength']:g}", flush=True)
    print(f"Frozen CatBoost-anchored winner: {stacked_winner['model']} at strength {stacked_winner['strength']:g}", flush=True)

    evaluation_curves = json.loads(args.curves.read_text(encoding="utf-8"))["updated"]
    print("Refitting on full train", flush=True)
    blind_predictions, pair_support = relevance_models(blind_train, train, blind_val, evaluation_curves["click"])
    predictions = {"point_raw": pointwise_predictions(x_train, train.click.to_numpy(dtype=float), x_val), **blind_predictions}
    for model, score in blind_predictions.items():
        anchor, rank_score = anchor_inputs(val, score)
        predictions[f"{model}_fixed"] = anchor + FIXED_STRENGTH * rank_score
        predictions[f"{model}_tuned"] = anchor + selected[model]["strength"] * rank_score
        click_anchor, _ = anchor_inputs(val, score, predictions["point_raw"])
        predictions[f"{model}_on_catboost"] = click_anchor + stacked[model]["strength"] * rank_score

    print("Comparing held-out validation rankings", flush=True)
    for suffix in ("", "_fixed", "_tuned", "_on_catboost"):
        comparison_pairs = {f"{key}{suffix}": f"{control}{suffix}" for key, control in MATCHED_PAIRS.items()}
        MATCHED_CONTROLS.update(comparison_pairs)
    metrics = comparisons(val, predictions, evaluation_curves)
    output = {
        "features_without_position": features_without_position,
        "reference_features": features,
        "train": {"requests": int(train.request_key.nunique()), "rows": len(train), "preference_pairs": pair_support},
        "validation": evaluate(val, predictions, intervals=False)["coverage"],
        "details": {
            "anchor": "-log(Qilin original position or raw-click CatBoost's new rank) + strength * midrank of position-free relevance in [0, 1]",
            "fit_strengths": STRENGTHS,
            "fixed_strength": FIXED_STRENGTH,
            "selection_rule": "Maximize raw ClickNDCG@10 subject to LikeNDCG@10 on like-positive queries at least as high as the starting order; break ties using LikeNDCG and smaller mixing strength.",
            "reference": "CatBoost RMSE trained on raw click with original Qilin position, text similarity and note type.",
            "fixed_evaluation_curve_file": str(args.curves),
        },
        "calibration": {
            "fit_rows": len(fit), "requests": int(calibrate.request_key.nunique()), "rows": len(calibrate),
            "cutoff_utc_unix": cutoff, "click_exponent": inner_curve["exponent"], "qilin": inner_qilin,
            "pair_support": inner_pair_support, "selected": selected, "winner": winner, "results": calibration,
            "raw_click_catboost": inner_baseline, "selected_on_catboost": stacked,
            "winner_on_catboost": stacked_winner, "results_on_catboost": stacked_calibration,
        },
        "comparisons": metrics,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_csv(args.output / "comparisons.csv", metrics)
    write_csv(args.output / "calibration.csv", [
        {"starting_order": name, **row}
        for name, rows in (("qilin", calibration), ("catboost_raw_click", stacked_calibration)) for row in rows
    ])


if __name__ == "__main__":
    main()
