"""Measure the contribution of note media metadata on fixed Qilin slates."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from engagement_eval import evaluate
from engagement_models import pointwise_predictions
from run_avatar_rerank import DEPTH, write_csv
from run_engagement import COMPOSITES, build_rank_features, composite, signals
from run_reward_ablation import comparisons
from slates import _T1, _T2, load_slates_at_depths


CUTOFF = 10
MEDIA_COLUMNS = ("image_num", "video_duration", "video_width", "video_height")
MEDIA_FEATURES = (*MEDIA_COLUMNS, "listed_image_count")
VISUAL_PATTERN = r"头像|壁纸|背景图|风景|美甲|穿搭"


def media_features(train, val, notes_files):
    candidates = pd.concat([train.note_id, val.note_id], ignore_index=True)
    wanted = pa.array(candidates.unique())
    records = []
    for path in notes_files:
        with pq.ParquetFile(path) as parquet:
            shard_ids = wanted.cast(parquet.schema_arrow.field("note_idx").type)
            for batch in parquet.iter_batches(columns=["note_idx", "image_path", *MEDIA_COLUMNS]):
                batch = batch.filter(pc.is_in(batch.column("note_idx"), value_set=shard_ids))
                if not batch.num_rows:
                    continue
                rows = batch.drop_columns(["image_path"]).to_pandas()
                rows["listed_image_count"] = pc.list_value_length(batch.column("image_path")).to_numpy(
                    zero_copy_only=False
                )
                records.append(rows)

    metadata = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["note_idx", *MEDIA_FEATURES])
    metadata = metadata.drop_duplicates("note_idx").set_index("note_idx")
    aligned = metadata.reindex(candidates.to_numpy())
    values = aligned.loc[:, MEDIA_FEATURES].to_numpy(dtype=np.float32)
    values[values < 0] = np.nan
    coverage = {}
    for name, frame, data in (("train", train, aligned.iloc[:len(train)]), ("validation", val, aligned.iloc[len(train):])):
        path_present = data["listed_image_count"].gt(0).to_numpy()
        coverage[name] = {
            "rows": len(frame),
            "unique_notes": int(frame.note_id.nunique()),
            "rows_with_image_paths": int(path_present.sum()),
            "unique_notes_with_image_paths": int(frame.loc[path_present, "note_id"].nunique()),
            "rows_with_positive_video_duration": int(data["video_duration"].gt(0).sum()),
        }
    return values[:len(train)], values[len(train):], coverage


def split_scores(val, predictions, curves):
    raw = evaluate(val, predictions, intervals=False)
    return {**raw, "comparisons": comparisons(val, predictions, curves)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test-identifiers", type=Path, required=True)
    parser.add_argument("--notes", type=Path, nargs="+", required=True)
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    curves = json.loads(args.curves.read_text(encoding="utf-8"))["updated"]
    slates = load_slates_at_depths(args.train, args.test_identifiers, [CUTOFF, DEPTH], include_engagement=True)
    train, val = slates[DEPTH]
    val = val[val.request_key.isin(slates[CUTOFF][1].request_key)].reset_index(drop=True)
    print(f"Loaded {len(train):,} train pairs and {len(val):,} validation pairs", flush=True)
    x_train, x_val, features = build_rank_features(train, val, args.notes)
    media_train, media_val, coverage = media_features(train, val, args.notes)
    target = composite(signals(train), COMPOSITES["strong_actions"])
    predictions = {
        "point_raw": pointwise_predictions(x_train, target, x_val),
        "point_media_metadata": pointwise_predictions(
            np.column_stack((x_train, media_train)), target, np.column_stack((x_val, media_val))
        ),
    }

    is_visual = val["query"].str.contains(VISUAL_PATTERN, regex=True).to_numpy(dtype=bool)
    summary = {
        "config": {
            "split_boundaries_utc_unix": [_T1, _T2],
            "candidate_depth": DEPTH,
            "cutoff": CUTOFF,
            "baseline_features": features,
            "added_features": MEDIA_FEATURES,
            "uses_image_pixels": False,
            "visual_query_pattern": VISUAL_PATTERN,
            "training_target": "Frozen unweighted engagement composite strong_actions",
        },
        "media_coverage": coverage,
        "all_validation": split_scores(val, predictions, curves),
        "visual_queries": split_scores(
            val.loc[is_visual].reset_index(drop=True),
            {name: scores[is_visual] for name, scores in predictions.items()},
            curves,
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(args.output / "comparisons.csv", [
        {"slice": key, **row}
        for key in ("all_validation", "visual_queries")
        for row in summary[key]["comparisons"]
    ])


if __name__ == "__main__":
    main()
