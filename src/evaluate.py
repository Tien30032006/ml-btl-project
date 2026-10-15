"""
evaluate.py
TV4 - Evaluation + Demo + Report Lead

Chức năng:
    - Metrics: calculate_metrics, metrics_from_proba
    - Cost-sensitive: expected_cost, find_threshold_from_proba, find_best_threshold
    - Plots: ROC/PR, cost curve, confusion matrix
    - SHAP: compute_shap_values (dùng chung cho evaluate và app), explain_with_shap
    - main(): so sánh tất cả model đã train, chọn best theo OOF cost,
              lưu best_model.pkl + best_model_meta.json + final_comparison.csv

Quy ước:
    Positive class = DEFAULT (bad)  -> y = 1
    Cost matrix: FN = 5, FP = 1

Chạy:
    python src/evaluate.py               # đầy đủ, có SHAP
    python src/evaluate.py --skip-shap   # nhanh hơn

Lưu ý: import module này không có tác dụng phụ (không tạo thư mục, không đổi
backend matplotlib). Mọi việc đó nằm trong _setup_output() và chỉ chạy ở main().
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from preprocess import load_processed_data, split_features_target

# ----------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"
FIGURES_DIR  = REPORTS_DIR / "figures"

# ----------------------------------------------------------------- hằng số
COST_FN = 5   # bỏ sót khách default
COST_FP = 1   # từ chối nhầm khách tốt

RANDOM_STATE = 0
THRESHOLDS   = np.linspace(0.01, 0.99, 99)


def _setup_output() -> None:
    """Chuẩn bị môi trường xuất file (chỉ gọi từ main)."""
    matplotlib.use("Agg")
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 120
    for d in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _save(fig, fname: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname)
    plt.close(fig)
    print(f"  → {FIGURES_DIR / fname}")


# =====================================================================
# 1. METRICS
# =====================================================================
def expected_cost(y_true, y_pred, c_fn: float = COST_FN, c_fp: float = COST_FP) -> float:
    """Chi phí kỳ vọng trung bình trên mỗi mẫu theo cost matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    _, fp, fn, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return (c_fn * fn + c_fp * fp) / len(y_true)


def metrics_from_proba(y_true, y_proba, threshold: float = 0.5) -> dict:
    """
    Tính bộ metric từ xác suất đã có sẵn (không cần model).
    Dùng cho trang Batch Scoring khi upload CSV.
    """
    y_true  = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred  = (y_proba >= threshold).astype(int)

    metrics = {
        "Precision"    : float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall"       : float(recall_score(y_true, y_pred, zero_division=0)),
        "F1"           : float(f1_score(y_true, y_pred, zero_division=0)),
        "Expected Cost": float(expected_cost(y_true, y_pred)),
    }

    # AUC chỉ tính được khi có đủ 2 lớp
    if len(np.unique(y_true)) == 2:
        metrics["ROC-AUC"] = float(roc_auc_score(y_true, y_proba))
        metrics["PR-AUC"]  = float(average_precision_score(y_true, y_proba))
    else:
        metrics["ROC-AUC"] = float("nan")
        metrics["PR-AUC"]  = float("nan")
    return metrics


def calculate_metrics(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Tính metric cho một sklearn model (có predict_proba)."""
    y_proba = model.predict_proba(X_test)[:, 1]
    return metrics_from_proba(y_test, y_proba, threshold=threshold)


def find_threshold_from_proba(y_true, y_proba,
                              c_fn: float = COST_FN, c_fp: float = COST_FP):
    """
    Quét ngưỡng 0.01..0.99, chọn ngưỡng có expected cost nhỏ nhất.
    Trả về (best_threshold, best_cost, DataFrame cost theo threshold).
    """
    y_proba = np.asarray(y_proba)
    rows = [
        {"threshold": float(t),
         "expected_cost": float(expected_cost(y_true, (y_proba >= t).astype(int), c_fn, c_fp))}
        for t in THRESHOLDS
    ]
    df = pd.DataFrame(rows)
    best = df.loc[df["expected_cost"].idxmin()]
    return float(best["threshold"]), float(best["expected_cost"]), df


def find_best_threshold(model, X_val, y_val,
                        c_fn: float = COST_FN, c_fp: float = COST_FP):
    """Như find_threshold_from_proba nhưng tự gọi model.predict_proba."""
    y_proba = model.predict_proba(X_val)[:, 1]
    return find_threshold_from_proba(y_val, y_proba, c_fn, c_fp)


# =====================================================================
# 2. PLOTS
# =====================================================================
def plot_roc_pr_curves(model, X_test, y_test, model_name: str = "Model"):
    """Vẽ ROC & Precision-Recall curve."""
    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _  = roc_curve(y_test, y_proba)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(fpr, tpr, color="steelblue",
                 label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                title=f"ROC Curve - {model_name}")
    axes[0].legend(loc="lower right")

    axes[1].plot(rec, prec, color="darkorange",
                 label=f"AP = {average_precision_score(y_test, y_proba):.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title=f"Precision-Recall Curve - {model_name}")
    axes[1].legend(loc="upper right")

    _save(fig, "roc_pr_curves.png")


def plot_cost_curve(cost_df: pd.DataFrame, best_threshold: float):
    """Vẽ cost curve (trên test) và đánh dấu ngưỡng t* chọn từ OOF train."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(cost_df["threshold"], cost_df["expected_cost"], color="darkred")
    ax.axvline(best_threshold, color="green", linestyle="--",
               label=f"t* (OOF train) = {best_threshold:.2f}")
    ax.set(xlabel="Threshold", ylabel="Expected Cost (test)",
           title=f"Cost Curve (FN={COST_FN}, FP={COST_FP})")
    ax.legend()
    _save(fig, "cost_curves.png")


def plot_confusion(model, X_test, y_test, threshold: float, tag: str = "best_threshold"):
    """Vẽ confusion matrix."""
    y_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred Good", "Pred Bad"],
                yticklabels=["True Good", "True Bad"])
    ax.set_title(f"Confusion Matrix @ threshold = {threshold:.2f}")
    _save(fig, f"confusion_{tag}.png")


# =====================================================================
# 3. SHAP (dùng chung cho evaluate.py và streamlit_app.py)
# =====================================================================
def _to_dense(A) -> np.ndarray:
    return np.asarray(A.toarray() if hasattr(A, "toarray") else A)


def compute_shap_values(model, X, background=None):
    """
    Tính SHAP cho lớp default của pipeline (preprocessor + RF/LR).

    Trả về (shap_values có shape (n_samples, n_features), feature_names).
    `background` (DataFrame thô) chỉ dùng cho LinearExplainer; mặc định = X.
    """
    import shap

    preprocessor, clf = model[:-1], model[-1]
    A = _to_dense(preprocessor.transform(X))

    try:
        names = list(preprocessor.get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(A.shape[1])]

    kind = type(clf).__name__
    if kind == "RandomForestClassifier":
        sv = shap.TreeExplainer(clf).shap_values(A)
    elif kind == "LogisticRegression":
        bg = A if background is None else _to_dense(preprocessor.transform(background))
        sv = shap.LinearExplainer(clf, bg).shap_values(A)
    else:
        raise NotImplementedError(f"SHAP chưa hỗ trợ {kind}")

    # Tuỳ phiên bản shap/sklearn: list [class0, class1] hoặc ndarray (n, f, 2)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 1]
    return sv, names


def explain_with_shap(model, X_sample, max_display: int = 15) -> pd.Series:
    """Series: index = feature, value = mean(|SHAP|) trên các sample."""
    sv, names = compute_shap_values(model, X_sample)
    contrib = pd.Series(np.abs(sv).mean(axis=0), index=names)
    return contrib.sort_values(ascending=False).head(max_display)


def plot_shap_bar(contrib: pd.Series, fname: str = "shap_summary.png"):
    """Vẽ bar chart cho SHAP contribution (mean |SHAP|)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    contrib.iloc[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set(xlabel="mean(|SHAP value|)", title="SHAP - Top feature contributions")
    _save(fig, fname)


# =====================================================================
# 4. MODEL SELECTION - main
# =====================================================================
CANDIDATE_MODELS = {
    "Logistic Regression (tuned)": "lr_tuned.pkl",
    "SVM (tuned)"                : "svm_tuned.pkl",
    "Random Forest (tuned)"      : "rf_tuned.pkl",
    "Logistic Regression (base)" : "lr_baseline.pkl",
    "SVM (baseline)"             : "svm_baseline.pkl",
    "Random Forest (baseline)"   : "rf_baseline.pkl",
}


def _load_models() -> dict:
    """Load tất cả model đã train (chỉ những file tồn tại)."""
    models = {}
    for name, fname in CANDIDATE_MODELS.items():
        path = MODELS_DIR / fname
        if not path.exists():
            continue
        try:
            models[name] = joblib.load(path)
        except Exception as exc:
            print(f"  ! Không load được {fname}: {exc}")
    return models


def _oof_proba(model, X_train, y_train, cv):
    """Xác suất out-of-fold trên train, dùng để chọn threshold công bằng."""
    return cross_val_predict(
        model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]


def main(skip_shap: bool = False):
    _setup_output()

    print("=" * 65)
    print("EVALUATION - so sánh model & chọn best theo cost-sensitive")
    print("=" * 65)

    train_df, test_df = load_processed_data()
    X_train, y_train = split_features_target(train_df)
    X_test,  y_test  = split_features_target(test_df)

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Default rate train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")

    models = _load_models()
    if not models:
        raise FileNotFoundError(
            "Chưa có model nào trong models/. Chạy train_lr_svm.py và train_rf.py trước."
        )
    print(f"\nTìm thấy {len(models)} model: {list(models.keys())}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for name, model in models.items():
        print(f"\n── {name} ──")

        oof_p = _oof_proba(model, X_train, y_train, cv)
        best_t, best_cost_oof, _ = find_threshold_from_proba(y_train, oof_p)

        m_05   = calculate_metrics(model, X_test, y_test, threshold=0.5)
        m_best = calculate_metrics(model, X_test, y_test, threshold=best_t)

        rows.append({
            "Model"           : name,
            "Threshold*"      : round(best_t, 3),
            "PR-AUC (test)"   : round(m_05["PR-AUC"], 4),
            "ROC-AUC (test)"  : round(m_05["ROC-AUC"], 4),
            "Recall @0.5"     : round(m_05["Recall"], 4),
            "Precision @0.5"  : round(m_05["Precision"], 4),
            "F1 @0.5"         : round(m_05["F1"], 4),
            "Cost @0.5"       : round(m_05["Expected Cost"], 4),
            "Recall @t*"      : round(m_best["Recall"], 4),
            "Precision @t*"   : round(m_best["Precision"], 4),
            "F1 @t*"          : round(m_best["F1"], 4),
            "Cost @t* (test)" : round(m_best["Expected Cost"], 4),
            "Cost OOF (train)": round(best_cost_oof, 4),
        })

        print(f"  threshold*  : {best_t:.3f}")
        print(f"  Cost @0.5   : {m_05['Expected Cost']:.4f}")
        print(f"  Cost @t*    : {m_best['Expected Cost']:.4f}")
        print(f"  Recall @t*  : {m_best['Recall']:.4f}")
        print(f"  PR-AUC      : {m_05['PR-AUC']:.4f}")

    cmp_df = pd.DataFrame(rows).set_index("Model")
    cmp_df.to_csv(REPORTS_DIR / "final_comparison.csv")

    print("\n" + "=" * 65)
    print("BẢNG SO SÁNH (test set, positive = default)")
    print("=" * 65)
    print(cmp_df.to_string())

    best_name      = cmp_df["Cost OOF (train)"].idxmin()
    best_model     = models[best_name]
    best_threshold = float(cmp_df.loc[best_name, "Threshold*"])
    print(f"\nBest model (theo Cost OOF): {best_name} | Threshold* = {best_threshold:.3f}")

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

    test_metrics = calculate_metrics(best_model, X_test, y_test, threshold=best_threshold)
    cost_05      = calculate_metrics(best_model, X_test, y_test, threshold=0.5)["Expected Cost"]

    meta = {
        "name"        : best_name,
        "threshold"   : best_threshold,
        "cost_matrix" : {"FN": COST_FN, "FP": COST_FP},
        "test_metrics": {
            "Precision"      : round(test_metrics["Precision"], 4),
            "Recall"         : round(test_metrics["Recall"], 4),
            "F1"             : round(test_metrics["F1"], 4),
            "ROC-AUC"        : round(test_metrics["ROC-AUC"], 4),
            "PR-AUC"         : round(test_metrics["PR-AUC"], 4),
            "Cost @0.5"      : round(cost_05, 4),
            "Cost @t* (test)": round(test_metrics["Expected Cost"], 4),
        },
    }
    (MODELS_DIR / "best_model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  → {MODELS_DIR / 'best_model.pkl'}")
    print(f"  → {MODELS_DIR / 'best_model_meta.json'}")

    print("\nVẽ figures ...")
    plot_roc_pr_curves(best_model, X_test, y_test, model_name=best_name)
    _, _, cost_df = find_best_threshold(best_model, X_test, y_test)
    plot_cost_curve(cost_df, best_threshold)
    plot_confusion(best_model, X_test, y_test, 0.5, tag="default")
    plot_confusion(best_model, X_test, y_test, best_threshold, tag="best_threshold")

    if skip_shap:
        print("\nBỏ qua SHAP (--skip-shap).")
    else:
        try:
            contrib = explain_with_shap(best_model, X_test)
            print("\nTop features theo SHAP:")
            print(contrib.to_string())
            plot_shap_bar(contrib)
        except Exception as exc:
            print(f"SHAP không chạy được: {exc}")

    print("\nHOÀN TẤT EVALUATION.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="So sánh model và chọn best theo cost.")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Không tính/vẽ SHAP (nhanh hơn).")
    return parser.parse_args()


if __name__ == "__main__":
    main(skip_shap=_parse_args().skip_shap)