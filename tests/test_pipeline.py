"""
tests/test_pipeline.py – Unit tests for the Schisto Risk ML pipeline.

Run with:
    python -m pytest tests/ -v
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    build_preprocessing_pipeline,
    load_data,
    split_features_target,
    train_test_split_data,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.train import build_model_pipeline, train
from src.predict import predict


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

TEST_SIZE_TOLERANCE = 0.01   # acceptable variance in train/test split ratio
MEAN_TOLERANCE = 0.1         # acceptable deviation from zero-mean after scaling
MIN_ACCURACY_THRESHOLD = 0.70  # minimum acceptable model accuracy on the test set


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "schisto_risk.csv"
)


@pytest.fixture(scope="module")
def sample_df():
    """Load the generated sample dataset once for the whole module."""
    return load_data(SAMPLE_CSV)


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory, sample_df):
    """Train a small model and return the path to the saved .joblib file."""
    tmpdir = tmp_path_factory.mktemp("models")
    model_path = str(tmpdir / "rf_model.joblib")
    train(
        data_path=SAMPLE_CSV,
        output_path=model_path,
        n_estimators=50,
        test_size=0.2,
        random_state=0,
    )
    return model_path


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_load_data_returns_dataframe(self, sample_df):
        assert isinstance(sample_df, pd.DataFrame)
        assert len(sample_df) > 0

    def test_feature_columns_present(self, sample_df):
        for col in FEATURE_COLUMNS:
            assert col in sample_df.columns, f"Missing column: {col}"

    def test_target_column_present(self, sample_df):
        assert TARGET_COLUMN in sample_df.columns

    def test_target_is_binary(self, sample_df):
        unique_values = set(sample_df[TARGET_COLUMN].unique())
        assert unique_values.issubset({0, 1})

    def test_split_features_target_shapes(self, sample_df):
        X, y = split_features_target(sample_df)
        assert X.shape[1] == len(FEATURE_COLUMNS)
        assert len(y) == len(sample_df)

    def test_train_test_split_sizes(self, sample_df):
        X, y = split_features_target(sample_df)
        X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2)
        total = len(X_train) + len(X_test)
        assert total == len(sample_df)
        assert abs(len(X_test) / total - 0.2) < TEST_SIZE_TOLERANCE

    def test_preprocessing_pipeline_transforms(self, sample_df):
        X, _ = split_features_target(sample_df)
        pipeline = build_preprocessing_pipeline()
        X_transformed = pipeline.fit_transform(X)
        assert X_transformed.shape == X.shape
        # StandardScaler should give near-zero mean
        assert abs(X_transformed.mean()) < MEAN_TOLERANCE


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------

class TestTraining:
    def test_model_pipeline_has_correct_steps(self):
        pipeline = build_model_pipeline(n_estimators=10)
        assert "preprocessor" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps

    def test_train_creates_model_file(self, trained_model_path):
        assert os.path.exists(trained_model_path)

    def test_trained_model_accuracy(self, sample_df, trained_model_path):
        """Model should achieve at least 70% accuracy on test data."""
        import joblib
        model = joblib.load(trained_model_path)
        X, y = split_features_target(sample_df)
        _, X_test, _, y_test = train_test_split_data(X, y, test_size=0.2, random_state=0)
        accuracy = (model.predict(X_test) == y_test).mean()
        assert accuracy >= MIN_ACCURACY_THRESHOLD, f"Accuracy too low: {accuracy:.4f}"


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------

class TestPrediction:
    def test_predict_returns_dataframe(self, sample_df, trained_model_path):
        subset = sample_df.head(10)
        result = predict(trained_model_path, subset)
        assert isinstance(result, pd.DataFrame)

    def test_predict_adds_expected_columns(self, sample_df, trained_model_path):
        subset = sample_df.head(5)
        result = predict(trained_model_path, subset)
        assert "predicted_risk" in result.columns
        assert "risk_probability" in result.columns
        assert "risk_label" in result.columns

    def test_predict_probabilities_in_range(self, sample_df, trained_model_path):
        result = predict(trained_model_path, sample_df)
        assert result["risk_probability"].between(0, 1).all()

    def test_predict_labels_valid(self, sample_df, trained_model_path):
        result = predict(trained_model_path, sample_df)
        assert set(result["risk_label"].unique()).issubset({"Low Risk", "High Risk"})

    def test_predict_missing_model_raises(self, sample_df):
        with pytest.raises(FileNotFoundError):
            predict("nonexistent_path/model.joblib", sample_df)
