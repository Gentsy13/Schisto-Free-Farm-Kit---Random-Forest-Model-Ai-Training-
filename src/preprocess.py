"""
preprocess.py – Data loading and preprocessing utilities.

Usage
-----
from src.preprocess import load_data, split_features_target, build_pipeline
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


FEATURE_COLUMNS = [
    "proximity_to_water_m",
    "water_contact_hrs_per_day",
    "irrigation_type",
    "soil_moisture_pct",
    "temperature_c",
    "rainfall_mm",
    "snail_habitat_index",
    "sanitation_score",
    "age_group",
    "ppe_usage",
]
TARGET_COLUMN = "schisto_risk"


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    df = pd.read_csv(csv_path)
    return df


def split_features_target(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLUMNS,
    target_col: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature_cols : list[str]
        Column names to use as features.
    target_col : str
        Name of the target column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y)
    """
    X = df[feature_cols]
    y = df[target_col]
    return X, y


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Stratified train/test split.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_preprocessing_pipeline() -> Pipeline:
    """Build a scikit-learn preprocessing pipeline.

    The pipeline:
    1. Imputes missing values with the median.
    2. Scales features using StandardScaler.

    Returns
    -------
    Pipeline
        A fitted-ready sklearn Pipeline.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
