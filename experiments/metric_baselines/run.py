"""Compare logged feedback targets for reranking Qilin search results."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq

from compare import compare
from features import build_features
from slates import load_slates


DATA_REVISION = "2d79000cef450b99f6423357ce86e5c132d32dd5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    test = parser.add_mutually_exclusive_group(required=True)
    test.add_argument("--test-identifiers", type=Path)
    test.add_argument("--test-parquet", type=Path, help="Only request and session IDs are read")
    parser.add_argument("--notes", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="Run on 500 train and 300 validation requests")
    args = parser.parse_args()

    with TemporaryDirectory(prefix="qilin-") as tmp:
        identifiers = args.test_identifiers
        if identifiers is None:
            rows = pq.ParquetFile(args.test_parquet).read(
                columns=["user_idx", "session_idx", "search_idx"]
            ).to_pylist()
            identifiers = Path(tmp) / "test_identifiers.json"
            identifiers.write_text(json.dumps({"revision": DATA_REVISION, "rows": rows}))
        train, validation = load_slates(args.train, identifiers)

    if args.smoke:
        train = train[train.request_id.isin(train.request_id.drop_duplicates().head(500))].reset_index(drop=True)
        validation = validation[
            validation.request_id.isin(validation.request_id.drop_duplicates().head(300))
        ].reset_index(drop=True)

    print(f"Slates loaded: {len(train):,} train pairs, {len(validation):,} validation pairs", flush=True)
    x_train, x_val, feature_names = build_features(train, validation, args.notes)
    print(f"Features ready: {', '.join(feature_names)}", flush=True)
    missing_text_index = feature_names.index("note_text_missing")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": "THUIR/Qilin",
        "dataset_revision": DATA_REVISION,
        "split_boundaries_utc_unix": [1732726411, 1732735658],
        "smoke": args.smoke,
        "features": feature_names,
        "packages": {name: version(name) for name in ("catboost", "numpy", "pandas", "pyarrow")},
        "rows": {"train": len(train), "validation": len(validation)},
        "rows_without_note_text": {
            "train": int(x_train[:, missing_text_index].sum()),
            "validation": int(x_val[:, missing_text_index].sum()),
        },
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    compare(train, validation, x_train, x_val, args.output)


if __name__ == "__main__":
    main()
