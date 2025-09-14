"""Titanic models using scikit-learn: Logistic Regression and KNN.

Usage:
  - place train.csv and test.csv into the ./data directory.
  - Run: python titanic_models.py
  - Outputs accuracies and writes submission CSVs under ./submissions
"""

from __future__ import annotations

import argparse  # to parse CLI arguments if needed later
from pathlib import Path  # to work with filesystem paths in a portable way
from typing import Tuple

import numpy as np  # numerical helpers (means, arrays)
import pandas as pd  # dataframes for CSV I/O and processing


# scikit-learn building blocks for preprocessing and models
from sklearn.compose import ColumnTransformer  # apply different transforms to column groups
from sklearn.impute import SimpleImputer  # fill missing values
from sklearn.linear_model import LogisticRegression  # classification model
from sklearn.metrics import accuracy_score  # evaluation metric
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # splits and CV
from sklearn.neighbors import KNeighborsClassifier  # KNN classifier
from sklearn.pipeline import Pipeline  # chain preprocessing + model
from sklearn.preprocessing import OneHotEncoder  # turn categories into numeric columns



def build_preprocessor() -> ColumnTransformer:
    """impute missing values and one-hot encode.

    - Numeric: median imputation
    - Categorical: most-frequent imputation + one-hot encoding
    """
    # Which columns are numeric vs categorical (from Titanic data)
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # For numeric columns, fill missing values with the median
    numeric_transformer = SimpleImputer(strategy="median")

    # For categorical columns, fill missing values with the most frequent value,
    # then expand categories into separate 0/1 columns
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Apply numeric_transformer to numeric_features and categorical_transformer to categorical_features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def load_and_split_csv(
    csv_path: str | Path,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Read a single CSV and split into X_train, X_test, y_train, y_test.

    - Keeps class balance with stratification (recommended for classification)
    - Returns feature matrices (X_*) and label vectors (y_*)
    """
    # Read the whole dataset from disk
    df = pd.read_csv(csv_path)

    # Separate features (X) from target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Use y to keep class proportions in both splits (if requested)
    stratify_vec = y if stratify else None

    # Make a reproducible holdout split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_vec,
    )
    return X_train, X_test, y_train, y_test


def evaluate_classifier(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    """Return mean 5-fold CV accuracy for a classifier pipeline."""
    # Stratified K-fold preserves class balance in each fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # cross_val_score repeatedly fits the pipeline on train folds and scores on val folds
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    # Average the K scores to get a robust estimate
    return float(np.mean(scores))


def build_model_pipeline(model_name: str, preprocessor: ColumnTransformer) -> Pipeline:
    """Create a Pipeline for the requested model name."""
    name = model_name.lower()
    if name == "logistic":
        # Logistic Regression: linear classifier, good baseline for Titanic
        return Pipeline(steps=[("pre", preprocessor), ("model", LogisticRegression(max_iter=1000))])
    if name == "knn":
        # KNN: predicts based on nearest training examples
        return Pipeline(steps=[("pre", preprocessor), ("model", KNeighborsClassifier(n_neighbors=15, weights="distance"))])
    raise ValueError("model_name must be one of: 'logistic', 'knn'")



def train_single_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    X_holdout: pd.DataFrame | None = None,
    y_holdout: pd.Series | None = None,
) -> tuple[pd.DataFrame, float]:
    """Train one model and return (submission_df, accuracy).

    - If a holdout set is provided, accuracy is measured on it after fitting
    - Else, we report 5-fold cross-validation accuracy on the training data (why not maing use of test data here that we have defined? cus in kfold the shuffle and split happens on its own just the model is trained on k-1 splits and evaluated on kth split)
    """
    target_col = "Survived"

    # Split train_df into features and labels
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)

    # Create preprocessing + model pipeline
    preprocessor = build_preprocessor()
    pipeline = build_model_pipeline(model_name, preprocessor)

    # 1) Fit on training data
    pipeline.fit(X_train, y_train)

    # 2) Evaluate (either holdout or CV)
    if X_holdout is not None and y_holdout is not None:
        val_preds = pipeline.predict(X_holdout)
        acc = float(accuracy_score(y_holdout, val_preds))
    else:
        acc = evaluate_classifier(pipeline, X_train, y_train)

    print(f"Accuracy ({model_name}): {acc:.4f}")

    # 3) Predict for submission (on our test_df features)
    test_ids = test_df["PassengerId"].astype(int)
    preds = pipeline.predict(test_df).astype(int)

    # Format the predictions as a Kaggle-style submission
    submission = pd.DataFrame({"PassengerId": test_ids, "Survived": preds})
    return submission, float(acc)


def train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_holdout: pd.DataFrame | None = None,
    y_holdout: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train both models (Logistic, KNN) and return their submission DataFrames."""
    target_col = "Survived"

    # Split train_df into features and labels
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)

    # Shared preprocessing
    preprocessor = build_preprocessor()

    # Define pipelines
    logreg = Pipeline(steps=[("pre", preprocessor), ("model", LogisticRegression(max_iter=1000))])
    knn = Pipeline(steps=[("pre", preprocessor), ("model", KNeighborsClassifier(n_neighbors=15, weights="distance"))])

    # Fit first
    logreg.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    # Evaluate (holdout preferred, else CV)
    if X_holdout is not None and y_holdout is not None:
        logreg_acc = float(accuracy_score(y_holdout, logreg.predict(X_holdout)))
        knn_acc = float(accuracy_score(y_holdout, knn.predict(X_holdout)))
    else:
        logreg_acc = evaluate_classifier(logreg, X_train, y_train)
        knn_acc = evaluate_classifier(knn, X_train, y_train)

    print("Accuracy:")
    print(f"  Logistic Regression: {logreg_acc:.4f}")
    print(f"  KNN Classifier:      {knn_acc:.4f}")

    # Make predictions for final submission
    test_ids = test_df["PassengerId"].astype(int)
    logreg_pred = logreg.predict(test_df)
    knn_pred = knn.predict(test_df)

    # Wrap as DataFrames for saving
    logreg_sub = pd.DataFrame({"PassengerId": test_ids, "Survived": logreg_pred.astype(int)})
    knn_sub = pd.DataFrame({"PassengerId": test_ids, "Survived": knn_pred.astype(int)})

    return logreg_sub, knn_sub


def main() -> None:
    # Determine project directories (data for input, submissions for output)
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    submissions_dir = project_root / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load and split the combined CSV into train/holdout sets
    X_train, X_test, y_train, y_test = load_and_split_csv(
        data_dir / "Titanic-Dataset.csv", "Survived"
    )

    # Rebuild DataFrames to reuse downstream code that expects DataFrames
    train_df = pd.concat([X_train, y_train.rename("Survived")], axis=1)
    test_df = X_test.copy()

    # 2) Keep only the columns we intend to use (and the label on train)
    base_cols = ["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    needed_train_cols = base_cols + ["Survived"]

    train_df = train_df[needed_train_cols].copy()
    test_df = test_df[base_cols].copy()

    # 3) Train models, evaluate (on holdout), and get submission DataFrames
    logreg_sub, logreg_acc = train_single_model(train_df, test_df, "logistic", X_test, y_test)
    knn_sub, knn_acc = train_single_model(train_df, test_df, "knn", X_test, y_test)

    # 4) Save submissions to disk
    logreg_path = submissions_dir / "submission_logistic.csv"
    knn_path = submissions_dir / "submission_knn.csv"

    logreg_sub.to_csv(logreg_path, index=False)
    knn_sub.to_csv(knn_path, index=False)

    print("Saved submissions:")
    print(f"  {logreg_path}")
    print(f"  {knn_path}")


if __name__ == "__main__":
    main()
