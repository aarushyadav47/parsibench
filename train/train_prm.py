"""Train PRM classifier."""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from parsibench.utils.rng import stable_hash_str


def _train_test_split(df: pd.DataFrame):
    test_mask = df["task_id"].apply(lambda x: stable_hash_str(str(x)) % 5 == 0)
    train_df = df[~test_mask]
    test_df = df[test_mask]
    return train_df, test_df


def _prepare_xy(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in {"label", "task_id"}]
    X = df[feature_cols].fillna(0).to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    return X, y, feature_cols


def train_prm(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    train_df, test_df = _train_test_split(df)

    X_train, y_train, feature_cols = _prepare_xy(train_df)
    X_test, y_test, _ = _prepare_xy(test_df)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Test classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, out_path)
    feature_spec = {"feature_names": feature_cols}
    with open(out_path.replace(".pkl", "_feature_spec.json"), "w", encoding="utf-8") as f:
        f.write(pd.Series(feature_spec).to_json())


def main():
    parser = argparse.ArgumentParser(description="Train PRM from CSV.")
    parser.add_argument("--csv", required=True, help="PRM dataset CSV path.")
    parser.add_argument("--out", required=True, help="Output pickle path.")
    args = parser.parse_args()
    train_prm(args.csv, args.out)


if __name__ == "__main__":
    main()

