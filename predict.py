"""Predict is_housebuilding and number of homes from a csv of planning application proposals"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from config import add_n_homes, filter_in_scope

DEFAULT_KEEP_COLS = [
    "FID",
    "keyval",
    "reference",
    "local_auth",
    "url",
    "proposal",
    "dev_desc",
]
N_HOMES_COLS = ["n_homes", "n_homes_class", "n_homes_method"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="applications csv")
    parser.add_argument(
        "--model", required=True, type=Path, help="model written by train.py"
    )
    parser.add_argument("--output", required=True, type=Path, help="predictions csv")
    parser.add_argument(
        "--threshold",
        type=float,
        help="override the probability threshold saved with the model",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="predict on every row, skipping the application / development type filter",
    )
    parser.add_argument(
        "--text-col",
        default="proposal",
        help="proposal text col (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-cols",
        nargs="+",
        default=DEFAULT_KEEP_COLS,
        help="input cols copied to the output (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    missing = [c for c in [args.text_col, *args.keep_cols] if c not in df.columns]
    if missing:
        sys.exit(f"Columns not in {args.input}: {', '.join(missing)}")

    n_rows = len(df)
    if args.no_filter:
        print(f"Rows, no filter applied: {n_rows}")
    else:
        try:
            df = filter_in_scope(df)
        except ValueError as err:
            sys.exit(str(err))
        print(f"Rows after filter: {len(df)} of {n_rows}")

    model = joblib.load(args.model)
    threshold = model["threshold"] if args.threshold is None else args.threshold
    print(f"Threshold: {threshold}")

    # same lower case text as training, a couple of rows have no proposal
    text = df[args.text_col].fillna("").str.lower()
    pred_prob = model["pipeline"].predict_proba(text)[:, 1]

    # only the keep cols are copied, so a labelled input can't clash with the new cols
    out = df[args.keep_cols].copy()
    out["pred_prob"] = pred_prob
    out["is_housebuilding"] = (pred_prob >= threshold).astype(int)

    # number of homes only for predicted house building, blank for the rest
    homes = add_n_homes(
        df.loc[out["is_housebuilding"] == 1, [args.text_col]], args.text_col
    )
    out = out.join(homes[N_HOMES_COLS])

    print()
    print(out["is_housebuilding"].value_counts().sort_index().to_string())
    print("\nPredicted house building only:")
    print(homes["n_homes_class"].value_counts(dropna=False).sort_index().to_string())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nWrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
