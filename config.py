"""Shared settings and helpers for train.py, predict.py and validate.py"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline

from n_homes import N_HOMES_CLASSES, estimate_n_homes, n_homes_class

# currently 0.25, prioritising recall over precision.
# validate.py prints these results in its threshold table
CLASSIFICATION_THRESHOLD = 0.25

# prediction filter, other application types such as non-material variations,
# listed building consents and pre-application notices are out of scope
APPLICATION_TYPES = [
    "Planning Permission",
    "Approval of Matters Specified in Conditions",
    "Planning Permission in Principle",
    "Planning Application",
    "Section 42 Variation",
]
DEVELOPMENT_TYPES = [
    "housing",
    "other development",
    "amsc under two months",
    "amsc under four months",
    "amsc",
    "business and industry",
]

# input .xlsx column names -> clean names
TRAINING_COLS = {
    "Proposal text": "proposal_text",
    "Status": "status",
    "Is it a housebulding application?": "is_housebuilding",
    "Number of homes this application relates to the creation of": "n_homes",
}


def build_pipeline():
    """
    Setup the model pipeline
    word and char tf-idf features and logistic regression classifier
    """
    word_tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")
    char_tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)
    features = FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])
    classifier = LogisticRegression(C=10, max_iter=2000)

    return Pipeline([("features", features), ("classifier", classifier)])


def load_training(path):
    """Read the labelling spreadsheet, return manually checked rows with an ml_class col"""
    # skip first two rows and only the cols above, then rename them
    df = pd.read_excel(path, skiprows=2, usecols=list(TRAINING_COLS)).rename(
        columns=TRAINING_COLS
    )

    # make sure number of homes is integer, where text stored in this col just drop it
    df["n_homes"] = (
        pd.to_numeric(df["n_homes"], errors="coerce").astype("Int64").astype("string")
    )

    # convert all text across all cols to lower case
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    df[text_cols] = df[text_cols].apply(lambda s: s.str.lower())

    # just get the rows that have been checked manually
    checked = df[df["status"].str.contains("checked", regex=False, na=False)].copy()

    # binary ml class using the house building flag
    # any yes by reviewer is a positive, so:
    # yes but renewal
    # yes but change of use
    # yes but amendment
    # yes but app withdrawn ...etc
    checked["ml_class"] = (
        checked["is_housebuilding"]
        .str.contains("yes", regex=False, na=False)
        .astype(int)
    )
    return checked


def filter_in_scope(df):
    """Rows passing the prediction filter, blank development types are kept
    as not all authorities fill in this col, blank application types are removed"""
    missing = [c for c in ("appl_desc_is", "dev_desc_is") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Filter columns not in input: {', '.join(missing)}. "
            "Use --no-filter to predict on every row."
        )
    return df[
        df["appl_desc_is"].isin(APPLICATION_TYPES)
        & (df["dev_desc_is"].isin(DEVELOPMENT_TYPES) | df["dev_desc_is"].isna())
    ].copy()


def add_n_homes(frame, text_col):
    """add n_homes, n_homes_class and n_homes_method cols using the rules in n_homes.py"""
    est = frame[text_col].apply(estimate_n_homes)
    frame["n_homes"] = est.str[0].astype("Int64")
    frame["n_homes_class"] = pd.Categorical(
        frame["n_homes"].map(n_homes_class),
        categories=N_HOMES_CLASSES,
        ordered=True,
    )
    frame["n_homes_method"] = est.str[1]
    return frame


def binary_scores(y_true, y_pred):
    """precision / recall / f1 / accuracy of binary predictions"""
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }
