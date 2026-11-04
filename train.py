"""Cross-validate the house building classifier, then fit on all checked rows and save it"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from config import (
    CLASSIFICATION_THRESHOLD,
    binary_scores,
    build_pipeline,
    load_training,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="labelling spreadsheet (xlsx)"
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        help="where to write the model (joblib), metadata is written beside it as .json "
        "(default: models/housebuilding_<YYYYmmdd_HHMMSS>.joblib using the current "
        "date and time, so each run writes a new file and never overwrites a model)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=CLASSIFICATION_THRESHOLD,
        help="probability at or above which a row is housebuilding, saved with the model "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="cross-validation folds (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="fold shuffle seed (default: %(default)s)"
    )
    parser.add_argument(
        "--oof-output",
        type=Path,
        help="optional write a csv of out-of-fold predictions for manual inspection",
    )
    return parser.parse_args()


def cross_validate(df, skf, threshold):
    """Refit the pipeline on each fold's training split, print and return mean scores"""
    fold_scores = []
    for fold, (train_idx, test_idx) in enumerate(
        skf.split(df["proposal_text"], df["ml_class"]), start=1
    ):
        fold_model = build_pipeline()
        fold_model.fit(
            df["proposal_text"].iloc[train_idx], df["ml_class"].iloc[train_idx]
        )
        y_prob = fold_model.predict_proba(df["proposal_text"].iloc[test_idx])[:, 1]
        scores = binary_scores(
            df["ml_class"].iloc[test_idx], (y_prob >= threshold).astype(int)
        )
        fold_scores.append(scores)
        print(
            f"Fold {fold} - Precision: {scores['precision']:.4f}, "
            f"Recall: {scores['recall']:.4f}, F1: {scores['f1']:.4f}"
        )

    means = pd.DataFrame(fold_scores).mean().to_dict()
    print(f"\nMean Precision: {means['precision']:.4f}")
    print(f"Mean Recall: {means['recall']:.4f}")
    print(f"Mean F1: {means['f1']:.4f}")
    return means


def main():
    args = parse_args()
    if args.model_output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.model_output = Path("models") / f"housebuilding_{stamp}.joblib"

    df = load_training(args.input)
    class_counts = df["ml_class"].value_counts()
    print(f"Checked rows: {len(df)}")
    print(class_counts.to_string(), end="\n\n")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    cv_means = cross_validate(df, skf, args.threshold)

    # each row's prediction is from the fold where it was not in training
    y_prob = cross_val_predict(
        build_pipeline(),
        df["proposal_text"],
        df["ml_class"],
        cv=skf,
        method="predict_proba",
    )[:, 1]
    df["pred_prob"] = y_prob
    df["pred"] = (y_prob >= args.threshold).astype(int)

    cm = confusion_matrix(df["ml_class"], df["pred"])
    print("\nOut-of-fold confusion matrix")
    print(
        pd.DataFrame(
            cm,
            index=["actual: not housebuilding", "actual: housebuilding"],
            columns=["pred: not housebuilding", "pred: housebuilding"],
        ).to_string()
    )

    if args.oof_output:
        args.oof_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.oof_output)
        print(f"\nWrote out-of-fold predictions to {args.oof_output}")

    # final model trained on all checked rows, cross-validation was for info only
    pipeline = build_pipeline()
    pipeline.fit(df["proposal_text"], df["ml_class"])

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": args.threshold}, args.model_output)
    print(f"\nWrote model to {args.model_output}")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "rows": len(df),
        "negative_rows": int(class_counts.get(0, 0)),
        "positive_rows": int(class_counts.get(1, 0)),
        "threshold": args.threshold,
        "folds": args.folds,
        "seed": args.seed,
        "cv_mean_precision": cv_means["precision"],
        "cv_mean_recall": cv_means["recall"],
        "cv_mean_f1": cv_means["f1"],
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
    }
    metadata_path = args.model_output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
