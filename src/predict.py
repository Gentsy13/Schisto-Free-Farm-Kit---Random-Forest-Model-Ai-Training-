"""
predict.py – Run inference with a trained model.

Usage
-----
    python src/predict.py --model models/rf_model.joblib --input data/new_samples.csv

Or call predict() directly:

    from src.predict import predict
    results = predict("models/rf_model.joblib", new_df)
"""

import argparse
import os

import joblib
import pandas as pd

from src.preprocess import FEATURE_COLUMNS


def load_model(model_path: str):
    """Load a serialised sklearn Pipeline from disk.

    Parameters
    ----------
    model_path : str
        Path to the .joblib file.

    Returns
    -------
    sklearn Pipeline
        Loaded model pipeline.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict(model_path: str, df: pd.DataFrame) -> pd.DataFrame:
    """Run predictions on new data.

    Parameters
    ----------
    model_path : str
        Path to the saved model.
    df : pd.DataFrame
        DataFrame containing at least the feature columns.

    Returns
    -------
    pd.DataFrame
        Original dataframe with two new columns:
        - ``predicted_risk``  (0 = Low, 1 = High)
        - ``risk_probability`` (probability of High Risk)
    """
    model = load_model(model_path)
    X = df[FEATURE_COLUMNS]
    df = df.copy()
    df["predicted_risk"] = model.predict(X)
    df["risk_probability"] = model.predict_proba(X)[:, 1]
    df["risk_label"] = df["predicted_risk"].map({0: "Low Risk", 1: "High Risk"})
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Schisto Risk predictions.")
    parser.add_argument("--model", default="models/rf_model.joblib", help="Path to trained model")
    parser.add_argument("--input", required=True, help="Path to CSV with new samples")
    parser.add_argument("--output", default=None, help="Optional path to save predictions CSV")
    return parser.parse_args()


def _cli() -> None:
    """Entry-point for the ``schisto-predict`` console script."""
    args = _parse_args()
    data = pd.read_csv(args.input)
    results = predict(args.model, data)
    print(results[["predicted_risk", "risk_probability", "risk_label"]].to_string(index=False))
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nPredictions saved to {args.output}")


if __name__ == "__main__":
    _cli()
