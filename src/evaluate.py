"""
evaluate.py
TV4 — Evaluation + Demo + Report Lead

Hiện có: calculate_metrics + expected_cost dùng chung cho mọi model.
TODO (Tiến): cost-sensitive threshold tuning, cost curve, SHAP explainability.

Quy ước: y_true = 1 là DEFAULT (positive class).
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

COST_FN = 5  # bỏ sót khách default (duyệt khoản vay xấu)
COST_FP = 1  # từ chối nhầm khách tốt


def expected_cost(y_true, y_pred, c_fn=COST_FN, c_fp=COST_FP):
    """Chi phí kỳ vọng trung bình trên mỗi mẫu theo cost matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return (c_fn * fn + c_fp * fp) / len(y_true)


def calculate_metrics(model, X_test, y_test, threshold=0.5):
    """
    Tính bộ metric chuẩn của nhóm. Model phải có predict_proba.
    Predict theo `threshold` (mặc định 0.5), positive = default.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": average_precision_score(y_test, y_prob),
        "Expected Cost": expected_cost(y_test, y_pred),
    }