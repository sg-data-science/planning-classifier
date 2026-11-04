"""Score a predictions csv from predict.py against another labelled copy of the same rows"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from sklearn.metrics import confusion_matrix

from config import add_n_homes, binary_scores, load_training
from n_homes import N_HOMES_CLASSES, n_homes_class

SWEEP_THRESHOLDS = [0.1, 0.25, 0.5, 0.75, 0.9]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", required=True, type=Path, help="csv written by predict.py"
    )
    parser.add_argument("--labels", required=True, type=Path, help="labelled csv")
    parser.add_argument(
        "--id-col",
        default="keyval",
        help="col joining the two files (default: %(default)s)",
    )
    parser.add_argument(
        "--label-col",
        default="is_housebuilding",
        help="0/1 house building col in the labels (default: %(default)s)",
    )
    parser.add_argument(
        "--n-homes-col",
        default="n_homes",
        help="number of homes col in the labels (default: %(default)s)",
    )
    parser.add_argument(
        "--text-col",
        default="proposal",
        help="proposal text col in the predictions (default: %(default)s)",
    )
    parser.add_argument(
        "--train-input",
        type=Path,
        help="training spreadsheet, also scores rows whose text was not seen in training",
    )
    parser.add_argument(
        "--errors-output",
        type=Path,
        help="optional csv of false positives, false negatives and n_homes class mismatches",
    )
    return parser.parse_args()


def load_joined(args):
    """Predictions joined one to one with their labels, label cols are label / n_homes_true"""
    # classes stay text, "1" must not become 1.0 when no row is "more than 10"
    preds = pd.read_csv(
        args.predictions, low_memory=False, dtype={"n_homes_class": "string"}
    )
    labels = pd.read_csv(args.labels, low_memory=False)

    required = {
        args.predictions: [
            args.id_col,
            "pred_prob",
            "is_housebuilding",
            "n_homes_class",
        ],
        args.labels: [args.id_col, args.label_col, args.n_homes_col],
    }
    for path, cols in required.items():
        have = preds.columns if path == args.predictions else labels.columns
        missing = [c for c in cols if c not in have]
        if missing:
            sys.exit(f"Columns not in {path}: {', '.join(missing)}")

    # negatives are needed to count false negatives, the join can't recover them
    if not (preds["is_housebuilding"] == 0).any():
        sys.exit(
            f"{args.predictions} has no predicted negatives, so missed house building "
            "can't be counted. Validate a file with all in-scope rows."
        )

    no_label = ~preds[args.id_col].isin(labels[args.id_col])
    if no_label.any():
        sys.exit(f"{no_label.sum()} predicted rows have no label in {args.labels}")

    for path, frame in ((args.predictions, preds), (args.labels, labels)):
        if frame[args.id_col].duplicated().any():
            sys.exit(f"Duplicate {args.id_col} values in {path}")

    labels = labels[[args.id_col, args.label_col, args.n_homes_col]].rename(
        columns={args.label_col: "label", args.n_homes_col: "n_homes_true"}
    )
    joined = preds.merge(labels, on=args.id_col, validate="one_to_one")
    if joined["label"].isna().any():
        sys.exit(f"{joined['label'].isna().sum()} predicted rows have a blank label")
    # text in the number of homes col counts as no number, as in training
    joined["n_homes_true"] = pd.to_numeric(joined["n_homes_true"], errors="coerce")

    print(f"Rows scored: {len(joined)}")
    print(f"Labelled rows not in the predictions: {len(labels) - len(joined)}")
    return joined


def print_stage1(val, args):
    print("\n## Stage 1 - house building classifier\n")
    columns = {"all rows": val}
    if args.train_input:
        if args.text_col not in val.columns:
            sys.exit(f"--train-input needs {args.text_col} in {args.predictions}")
        train_text = set(load_training(args.train_input)["proposal_text"])
        text = val[args.text_col].fillna("").str.lower()
        in_train = text.isin(train_text)
        print(
            f"Rows with a proposal also in the training data: {in_train.sum()} of {len(val)}\n"
        )
        columns["excluding text seen in training"] = val[~in_train]

    scores = {
        name: {
            "rows": len(frame),
            **binary_scores(frame["label"], frame["is_housebuilding"]),
        }
        for name, frame in columns.items()
    }
    print(pd.DataFrame(scores).round(4).to_string())

    cm = confusion_matrix(val["label"], val["is_housebuilding"])
    print()
    print(
        pd.DataFrame(
            cm,
            index=["actual: not housebuilding", "actual: housebuilding"],
            columns=["pred: not housebuilding", "pred: housebuilding"],
        ).to_string()
    )

    sweep = [
        {
            "threshold": threshold,
            **binary_scores(val["label"], (val["pred_prob"] >= threshold).astype(int)),
        }
        for threshold in SWEEP_THRESHOLDS
    ]
    print("\nOther thresholds")
    print(pd.DataFrame(sweep).round(4).to_string(index=False))


def print_stage2(val, args):
    # rows labelled house building where the text gives a number of homes
    rows = val[(val["label"] == 1) & (val["n_homes_true"] > 0)].copy()
    if rows.empty:
        print(
            "\nStage 2 skipped: no rows labelled house building with a number of homes"
        )
        return

    true_class = pd.Categorical(
        rows["n_homes_true"].map(n_homes_class),
        categories=N_HOMES_CLASSES,
        ordered=True,
    )

    if args.text_col in rows.columns:
        print("\n## Stage 2 - regex number of homes, on all labelled rows\n")
        # the predictions leave n_homes blank where stage 1 said no, so rerun the rules
        # to score the regex on its own
        regex = add_n_homes(rows.copy(), args.text_col)
        regex["class_true"] = true_class
        has_est = regex[regex["n_homes_class"].notna()]
        correct = has_est["n_homes_class"] == has_est["class_true"]
        print(f"Rows with a class: {len(has_est) / len(regex):.1%} of {len(regex)}")
        print(f"Class accuracy where classified: {correct.mean():.1%}")
        print(
            "Class accuracy over all rows (no class counted as wrong): "
            f"{correct.sum() / len(regex):.1%}"
        )
        print(
            "Exact number correct where estimated: "
            f"{(has_est['n_homes'] == has_est['n_homes_true']).mean():.1%}\n"
        )
        print(regex["n_homes_method"].value_counts().to_string())
        print()
        print(
            pd.crosstab(
                regex["class_true"],
                regex["n_homes_class"]
                .cat.add_categories("no class")
                .fillna("no class"),
                rownames=["actual"],
                colnames=["predicted"],
                dropna=False,
            ).to_string()
        )
    else:
        print(
            f"\nRegex only stage 2 skipped: no {args.text_col} col in the predictions"
        )

    print("\n## Stage 2 - end to end, rows the classifier also flagged\n")
    rows["class_true"] = true_class
    e2e = rows[rows["is_housebuilding"] == 1]
    e2e_est = e2e[e2e["n_homes_class"].notna()]
    print(
        f"Labelled house building rows with a number found by stage 1: "
        f"{len(e2e)} of {len(rows)}"
    )
    print(
        "Class accuracy of those, where classified: "
        f"{(e2e_est['n_homes_class'] == e2e_est['class_true'].astype(str)).mean():.1%}"
    )


def write_errors(val, path):
    """One row per false positive, false negative or n_homes class mismatch"""
    true_class = val["n_homes_true"].map(n_homes_class)
    both_flagged = (val["label"] == 1) & (val["is_housebuilding"] == 1)
    has_classes = true_class.notna() & val["n_homes_class"].notna()

    frames = [
        val[(val["label"] == 0) & (val["is_housebuilding"] == 1)].assign(
            error_type="false_positive"
        ),
        val[(val["label"] == 1) & (val["is_housebuilding"] == 0)].assign(
            error_type="false_negative"
        ),
        val[both_flagged & has_classes & (val["n_homes_class"] != true_class)].assign(
            error_type="n_homes_mismatch"
        ),
    ]
    errors = pd.concat(frames).sort_values("pred_prob", ascending=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(path, index=False)
    print(f"\nWrote {len(errors)} errors to {path}")
    print(errors["error_type"].value_counts().to_string())


def main():
    args = parse_args()
    val = load_joined(args)
    print_stage1(val, args)
    print_stage2(val, args)
    if args.errors_output:
        write_errors(val, args.errors_output)


if __name__ == "__main__":
    main()
