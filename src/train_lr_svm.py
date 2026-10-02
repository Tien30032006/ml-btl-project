"""
train_lr_svm.py
TV2 — Model Lead A: Logistic Regression + SVM

- Baseline vs tuned (GridSearchCV, StratifiedKFold 5)
- Positive class = default (xem preprocess.py)
- Lưu model vào models/, bảng kết quả vào reports/
"""

import time

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from evaluate import calculate_metrics
from preprocess import (
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    build_preprocessor,
    load_processed_data,
    split_features_target,
)

# ---------------------------------------------------------------- grids
# LR: liblinear hỗ trợ cả l1 và l2 -> đủ dùng, tránh lặp solver
LR_PARAM_GRID = [
    {
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l1", "l2"],
        "classifier__solver": ["liblinear"],
        "classifier__class_weight": [None, "balanced"],
    }
]

# SVM: gamma chỉ có tác dụng với rbf -> tách riêng linear / rbf
SVM_PARAM_GRID = [
    {
        "classifier__kernel": ["linear"],
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__class_weight": [None, "balanced"],
    },
    {
        "classifier__kernel": ["rbf"],
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__gamma": ["scale", 0.001, 0.01, 0.1],
        "classifier__class_weight": [None, "balanced"],
    },
]

# Metric không phụ thuộc ngưỡng -> khớp với bước threshold tuning sau này
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "f1": "f1"}
REFIT = "pr_auc"


def make_lr():
    return Pipeline([
        ("preprocessor", build_preprocessor(X_ref, scale_numeric=True)),
        ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ])


def make_svm(probability):
    # probability=False khi tuning (nhanh hơn nhiều); AUC dùng decision_function
    return Pipeline([
        ("preprocessor", build_preprocessor(X_ref, scale_numeric=True)),
        ("classifier", SVC(probability=probability, random_state=RANDOM_STATE)),
    ])


def tune(name, estimator, grid, cv, X_train, y_train):
    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring=SCORING,
        refit=REFIT,
        cv=cv,
        n_jobs=-1,
    )
    start = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start

    print(f"\n[{name}] tuning xong sau {elapsed:.1f}s")
    print(f"  best params : {search.best_params_}")
    print(f"  CV {REFIT}  : {search.best_score_:.4f}")
    return search, elapsed


def timed_fit(model, X_train, y_train):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    return time.perf_counter() - start


def main():
    global X_ref
    train_df, test_df = load_processed_data()
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)
    X_ref = X_train

    print("Phân bố nhãn train (1 = default):")
    print(y_train.value_counts().to_string())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---- Baseline (mỗi pipeline có preprocessor riêng)
    lr_baseline = make_lr()
    svm_baseline = make_svm(probability=True)
    t_lr_base = timed_fit(lr_baseline, X_train, y_train)
    t_svm_base = timed_fit(svm_baseline, X_train, y_train)

    # ---- Tuning
    lr_search, t_lr_tune = tune("LR", make_lr(), LR_PARAM_GRID, cv, X_train, y_train)
    svm_search, t_svm_tune = tune("SVM", make_svm(False), SVM_PARAM_GRID, cv, X_train, y_train)

    best_lr = lr_search.best_estimator_

    # SVM: fit lại bản cuối với probability=True để có predict_proba
    best_svm = clone(svm_search.best_estimator_).set_params(classifier__probability=True)
    best_svm.fit(X_train, y_train)

    # ---- Metrics trên test
    results = pd.DataFrame({
        "LR Baseline": calculate_metrics(lr_baseline, X_test, y_test),
        "LR Tuned": calculate_metrics(best_lr, X_test, y_test),
        "SVM Baseline": calculate_metrics(svm_baseline, X_test, y_test),
        "SVM Tuned": calculate_metrics(best_svm, X_test, y_test),
    })
    results.index.name = "Metric"
    results.loc["CV PR-AUC (tuned)"] = [
        None, lr_search.best_score_, None, svm_search.best_score_
    ]
    results.loc["Fit time baseline (s)"] = [t_lr_base, None, t_svm_base, None]
    results.loc["Tuning time (s)"] = [None, t_lr_tune, None, t_svm_tune]

    print("\nMETRICS COMPARISON (test set, positive = default, threshold = 0.5)")
    print(results.round(4).to_string())

    print("\nBest Logistic Regression Parameters:")
    print(lr_search.best_params_)
    print("\nBest SVM Parameters:")
    print(svm_search.best_params_)

    # ---- Lưu để evaluate.py / Streamlit dùng lại
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_baseline, MODELS_DIR / "lr_baseline.pkl")
    joblib.dump(best_lr, MODELS_DIR / "lr_tuned.pkl")
    joblib.dump(svm_baseline, MODELS_DIR / "svm_baseline.pkl")
    joblib.dump(best_svm, MODELS_DIR / "svm_tuned.pkl")
    results.round(4).to_csv(REPORTS_DIR / "lr_svm_results.csv")
    pd.DataFrame({
        "LR": [str(lr_search.best_params_)],
        "SVM": [str(svm_search.best_params_)],
    }).to_csv(REPORTS_DIR / "lr_svm_best_params.csv", index=False)
    print(f"\nĐã lưu model vào {MODELS_DIR} và kết quả vào {REPORTS_DIR}")


if __name__ == "__main__":
    main()