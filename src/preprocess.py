"""
preprocess.py
TV1 - Data & Feature Lead

Load dữ liệu gốc, làm sạch, encoding, scaling, train/test split.
Xuất ra data/processed/train.csv và test.csv.

Quy ước:
    positive class = default (bad)  -> y = 1
    good                            -> y = 0
Trong CSV gốc: credit_risk = 1 (good), 0 (bad).
"""

from pathlib import Path
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
PROCESSED_DIR= DATA_DIR / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"

TARGET_COL        = "credit_risk"
GOOD_LABEL_IN_CSV = 1          # 1 = good, 0 = bad
RANDOM_STATE      = 0
TEST_SIZE         = 0.20


# ----------------------------------------------------------------- helpers
def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Đọc file CSV gốc."""
    path = path or RAW_DIR / "modified_GermanCredit.csv"
    df = pd.read_csv(path)
    return df


def split_features_target(df: pd.DataFrame):
    """
    Tách X, y. Đảo nhãn để positive class = default.
    y = 1 -> default (bad)
    y = 0 -> good
    """
    X = df.drop(columns=[TARGET_COL])
    y = (df[TARGET_COL] != GOOD_LABEL_IN_CSV).astype(int)
    return X, y


def get_feature_types(X: pd.DataFrame):
    """
    Numeric  : mọi cột số.
    Categorical: mọi cột còn lại (object / string / category).
    """
    numerical_features   = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()
    assert len(numerical_features) + len(categorical_features) == X.shape[1], \
        "Có cột không được phân loại num/cat!"
    return numerical_features, categorical_features


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True):
    """
    - Categorical : One-Hot Encoding (handle_unknown="ignore")
    - Numerical   : StandardScaler cho LR/SVM, passthrough cho tree-based (RF)

    Preprocessor nằm trong Pipeline nên chỉ fit trên train (và trên
    train-fold khi cross-validation) -> không leakage.
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


def load_processed_data():
    """Load train/test đã tạo sau bước feature selection."""
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "test.csv")
    return train_df, test_df


# ----------------------------------------------------------------- main
def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Đọc dữ liệu gốc ...")
    df = load_raw_data()
    print(f"  shape gốc: {df.shape}")

    # loại bỏ cột không dùng nếu có (ví dụ job, telephone)
    drop_cols = [c for c in ["job", "telephone"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"  đã bỏ cột: {drop_cols}")

    # train/test split (stratified theo nhãn đảo)
    y_all = (df[TARGET_COL] != GOOD_LABEL_IN_CSV).astype(int)
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    # lưu danh sách features thực tế
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    (PROCESSED_DIR / "final_features.json").write_text(
        json.dumps(feature_cols, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    print(f"Đã lưu train.csv ({train_df.shape}) và test.csv ({test_df.shape})")
    print(f"Features: {feature_cols}")


if __name__ == "__main__":
    main()