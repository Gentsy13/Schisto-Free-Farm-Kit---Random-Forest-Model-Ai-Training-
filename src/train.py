"""
train.py – Train a Random Forest classifier for schistosomiasis risk prediction.

Usage
-----
    python src/train.py
    python src/train.py --data data/schisto_risk.csv --output models/rf_model.joblib

Options
-------
--data      Path to the CSV dataset          (default: data/schisto_risk.csv)
--output    Path to save the trained model   (default: models/rf_model.joblib)
--n_estimators  Number of trees              (default: 200)
--max_depth     Maximum tree depth           (default: None = unlimited)
--test_size     Fraction for test split      (default: 0.2)
--random_state  Random seed                  (default: 42)
--tune          Run GridSearchCV hyperparameter tuning (flag)
"""

import argparse
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.preprocess import (
    build_preprocessing_pipeline,
    load_data,
    split_features_target,
    train_test_split_data,
)
from src.evaluate import evaluate_model, print_metrics, plot_confusion_matrix, plot_feature_importance


def build_model_pipeline(
    n_estimators: int = 200,
    max_depth: int = None,
    random_state: int = 42,
) -> Pipeline:
    """Combine preprocessing and Random Forest into a single Pipeline.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Maximum depth of each tree.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Pipeline
        Full sklearn Pipeline (preprocessing + classifier).
    """
    preprocessor = build_preprocessing_pipeline()
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def tune_hyperparameters(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """Run a GridSearchCV over key Random Forest hyperparameters.

    Parameters
    ----------
    pipeline : Pipeline
        The model pipeline to tune.
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.

    Returns
    -------
    Pipeline
        Best estimator found by GridSearchCV.
    """
    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
    }
    print("Running GridSearchCV (this may take a few minutes)…")
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"Best params : {search.best_params_}")
    print(f"Best ROC-AUC: {search.best_score_:.4f}")
    return search.best_estimator_


def train(
    data_path: str = "data/schisto_risk.csv",
    output_path: str = "models/rf_model.joblib",
    n_estimators: int = 200,
    max_depth: int = None,
    test_size: float = 0.2,
    random_state: int = 42,
    tune: bool = False,
) -> Pipeline:
    """End-to-end training routine.

    Parameters
    ----------
    data_path : str
        Path to the CSV dataset.
    output_path : str
        Where to save the serialised model.
    n_estimators : int
        Number of trees (ignored when tune=True).
    max_depth : int or None
        Max tree depth (ignored when tune=True).
    test_size : float
        Proportion of data reserved for testing.
    random_state : int
        Random seed.
    tune : bool
        If True, run GridSearchCV instead of default fit.

    Returns
    -------
    Pipeline
        Trained model pipeline.
    """
    # 1. Load data
    print(f"Loading dataset from {data_path}…")
    df = load_data(data_path)
    print(f"  {len(df)} rows, {df.shape[1]} columns loaded.")

    # 2. Prepare features and target
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"  Class distribution (train): {dict(y_train.value_counts().sort_index())}")

    # 3. Build or tune the pipeline
    pipeline = build_model_pipeline(n_estimators, max_depth, random_state)
    if tune:
        pipeline = tune_hyperparameters(pipeline, X_train, y_train)
    else:
        print("Training Random Forest…")
        pipeline.fit(X_train, y_train)

    # 4. Evaluate
    metrics = evaluate_model(pipeline, X_test, y_test)
    print_metrics(metrics)

    # 5. Save plots
    plots_dir = os.path.join(os.path.dirname(output_path), "plots")
    plot_confusion_matrix(pipeline, X_test, y_test, os.path.join(plots_dir, "confusion_matrix.png"))

    rf = pipeline.named_steps["classifier"]
    feature_names = list(X.columns)
    plot_feature_importance(
        feature_names,
        rf.feature_importances_,
        os.path.join(plots_dir, "feature_importance.png"),
    )

    # 6. Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"\nModel saved to {output_path}")

    return pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Schisto Risk Random Forest model.")
    parser.add_argument("--data", default="data/schisto_risk.csv", help="Path to CSV dataset")
    parser.add_argument("--output", default="models/rf_model.joblib", help="Output model path")
    parser.add_argument("--n_estimators", type=int, default=200, help="Number of trees")
    parser.add_argument("--max_depth", type=lambda x: None if x in (None, "None", "") else int(x), default=None, help="Max tree depth (omit for unlimited)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--tune", action="store_true", help="Run GridSearchCV hyperparameter tuning")
    return parser.parse_args()


def _cli() -> None:
    """Entry-point for the ``schisto-train`` console script."""
    args = _parse_args()
    train(
        data_path=args.data,
        output_path=args.output,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        test_size=args.test_size,
        random_state=args.random_state,
        tune=args.tune,
    )


if __name__ == "__main__":
    _cli()
