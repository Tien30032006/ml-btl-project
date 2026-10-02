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
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
 
TARGET_COL = "credit_risk"
GOOD_LABEL_IN_CSV = 1  # trong CSV: 1 = good, 0 = bad
RANDOM_STATE = 0
 
 
def load_processed_data():
    """Load train/test đã tạo sau bước feature selection."""
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    return train_df, test_df
 
 
def split_features_target(df):
    """
    Tách X và y, ĐỒNG THỜI đảo nhãn để positive class = default.
    y = 1 -> default (bad), y = 0 -> good.
    """
    X = df.drop(columns=[TARGET_COL])
    y = (df[TARGET_COL] != GOOD_LABEL_IN_CSV).astype(int)
    return X, y
 
 
def get_feature_types(X):
    """
    Numeric: mọi cột số. Categorical: mọi cột còn lại (object / string / category).
    Dùng exclude="number" để không bỏ sót cột string (pandas >= 3.0).
    """
    numerical_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()
 
    # Đảm bảo không cột nào bị mất âm thầm
    assert len(numerical_features) + len(categorical_features) == X.shape[1], \
        "Có cột không được phân loại num/cat!"
    return numerical_features, categorical_features
 
 
def build_preprocessor(X, scale_numeric=True):
    """
    - Categorical: One-Hot Encoding (handle_unknown="ignore")
    - Numerical: StandardScaler cho LR/SVM, passthrough cho tree-based (RF)
 
    Preprocessor này nằm trong Pipeline nên chỉ fit trên train
    (và trên train-fold khi cross-validation) -> không leakage.
    Mỗi pipeline nên gọi hàm này riêng để có object độc lập.
    """
    numerical_features, categorical_features = get_feature_types(X)
 
    numerical_transformer = StandardScaler() if scale_numeric else "passthrough"
 
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor
 