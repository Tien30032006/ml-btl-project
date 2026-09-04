"""Các hàm tiện ích dùng chung cho project."""
import random
import numpy as np
import yaml


def set_seed(seed: int = 42) -> None:
    """Cố định seed để đảm bảo kết quả có thể tái hiện."""
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str) -> dict:
    """Đọc file cấu hình YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
