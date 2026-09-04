"""Tiền xử lý dữ liệu thô -> dữ liệu sẵn sàng cho huấn luyện.

Ví dụ chạy:
    python src/data_preprocessing.py --input data/raw --output data/processed
"""
import argparse
import os
import pandas as pd


def load_raw_data(input_dir: str) -> pd.DataFrame:
    """TODO: đọc dữ liệu thô từ input_dir."""
    raise NotImplementedError


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: xử lý missing value, outlier, encode categorical, v.v."""
    raise NotImplementedError


def split_and_save(df: pd.DataFrame, output_dir: str) -> None:
    """TODO: chia train/val/test và lưu vào output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/processed")
    args = parser.parse_args()

    df = load_raw_data(args.input)
    df = clean_data(df)
    split_and_save(df, args.output)


if __name__ == "__main__":
    main()
