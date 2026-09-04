"""Huấn luyện mô hình.

Ví dụ chạy:
    python src/train.py --config configs/config.yaml --model random_forest
"""
import argparse
import os
import joblib
import pandas as pd

from utils import set_seed, load_config

MODEL_REGISTRY = {
    # "random_forest": build_random_forest,
    # "svm": build_svm,
    # "logistic_regression": build_logreg,
}


def load_processed_data(processed_dir: str):
    """TODO: đọc dữ liệu đã tiền xử lý (train/val)."""
    raise NotImplementedError


def build_model(model_name: str, params: dict):
    """TODO: khởi tạo mô hình theo tên và tham số."""
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="models")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    X_train, y_train, X_val, y_val = load_processed_data(config["data"]["processed_dir"])
    model = build_model(args.model, config.get("models", {}).get(args.model, {}))
    model.fit(X_train, y_train)

    os.makedirs(args.output, exist_ok=True)
    joblib.dump(model, os.path.join(args.output, f"{args.model}.pkl"))
    print(f"Đã lưu model tại {args.output}/{args.model}.pkl")


if __name__ == "__main__":
    main()
