"""Đánh giá mô hình đã huấn luyện trên tập test.

Ví dụ chạy:
    python src/evaluate.py --model models/random_forest.pkl --data data/processed/test.csv
"""
import argparse
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)


def load_test_data(data_path: str):
    """TODO: đọc dữ liệu test, trả về X_test, y_test."""
    raise NotImplementedError


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()

    model = joblib.load(args.model)
    X_test, y_test = load_test_data(args.data)
    metrics = evaluate(model, X_test, y_test)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
