"""
preprocess.py
TV1 — Data & Feature Lead

TODO: load raw data, xử lý missing value, encoding, scaling,
train/test split, xuất ra data/processed/train.csv và test.csv.
"""

from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COL = "credit_risk"


def load_processed_data():
    """
    Load the final train/test datasets created after feature selection.
    """
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    return train_df, test_df


def split_features_target(df):
    """
    Separate input features X and target y.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    return X, y


def get_feature_types(X):
    """
    Identify numerical and categorical features.
    """
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return numerical_features, categorical_features


def build_preprocessor(X, scale_numeric=True):
    """
    Create a preprocessing pipeline.

    - Categorical features: One-Hot Encoding
    - Numerical features:
        + StandardScaler for LR/SVM
        + passthrough for tree-based models such as Random Forest
    """

    numerical_features, categorical_features = get_feature_types(X)

    numerical_transformer = (
        StandardScaler()
        if scale_numeric
        else "passthrough"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_transformer,
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    return preprocessor