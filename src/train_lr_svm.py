"""
train_lr_svm.py
TV2 - Model Lead A: Logistic Regression + SVM

- Baseline vs tuned (GridSearchCV, StratifiedKFold 5)
- Positive class = default (y=1)
- Logistic Regression: API mới dùng l1_ratio (sklearn >= 1.5)
- SVM: dùng CalibratedClassifierCV để có predict_proba (sklearn >= 1.9)
- Lưu model vào models/, bảng kết quả vào reports/
"""

import time

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
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

# ----------------------------------------------------------------- grids
# Logistic Regression - API mới: dùng l1_ratio thay penalty
#   l1_ratio = 0.0  -> L2
#   l1_ratio = 1.0  -> L1
#   (0 < x < 1)     -> ElasticNet
# Solver "saga" hỗ trợ cả L1, L2, ElasticNet với l1_ratio.
LR_PARAM_GRID = [
    {
        "classifier__C"           : [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__l1_ratio"    : [0.0, 1.0],
        "classifier__solver"      : ["saga"],
        "classifier__class_weight": [None, "balanced"],
        "classifier__max_iter"    : [5000],
    }
]

# SVM - classifier là CalibratedClassifierCV bọc SVC,
# nên param của SVC phải truy cập qua "estimator__"
SVM_PARAM_GRID = [
    {
        "classifier__estimator__kernel"      : ["linear"],
        "classifier__estimator__C"           : [0.01, 0.1, 1, 10, 100],
        "classifier__estimator__class_weight": [None, "balanced"],
    },
    {
        "classifier__estimator__kernel"      : ["rbf"],
        "classifier__estimator__C"           : [0.1, 1, 10, 100],
        "classifier__estimator__gamma"       : ["scale", 0.001, 0.01, 0.1],
        "classifier__estimator__class_weight": [None, "balanced"],
    },
]

SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "f1": "f1"}
REFIT   = "pr_auc"


# ----------------------------------------------------------------- builders
def make_lr(X_ref):
    """Logistic Regression pipeline với preprocessor tích hợp."""
    return Pipeline([
        ("preprocessor", build_preprocessor(X_ref, scale_numeric=True)),
        ("classifier", LogisticRegression(
            solver="saga",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )),
    ])


def make_svm(X_ref):
    """
    SVM pipeline với xác suất được hiệu chỉnh bằng CalibratedClassifierCV.
    (sklearn >= 1.9 deprecate SVC(probability=True).)
    """
    return Pipeline([
        ("preprocessor", build_preprocessor(X_ref, scale_numeric=True)),
        ("classifier", CalibratedClassifierCV(
            estimator=SVC(random_state=RANDOM_STATE),
            method="sigmoid",
            cv=5,
            ensemble=False,
        )),
    ])


# ----------------------------------------------------------------- helpers
def tune(name, estimator, grid, cv, X_train, y_train):
    """GridSearchCV - chọn threshold theo PR-AUC."""
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
    print(f"  CV {REFIT} : {search.best_score_:.4f}")
    return search, elapsed


def timed_fit(model, X_train, y_train):
    """Fit model và trả về thời gian."""
    start = time.perf_counter()
    model.fit(X_train, y_train)
    return time.perf_counter() - start


# ----------------------------------------------------------------- main
def main():
    train_df, test_df = load_processed_data()
    X_train, y_train = split_features_target(train_df)
    X_test,  y_test  = split_features_target(test_df)

    print("Phân bố nhãn train (1 = default):")
    print(y_train.value_counts().to_string())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---- baseline
    lr_baseline  = make_lr(X_train)
    svm_baseline = make_svm(X_train)
    t_lr_base  = timed_fit(lr_baseline,  X_train, y_train)
    t_svm_base = timed_fit(svm_baseline, X_train, y_train)

    # ---- tuning
    lr_search,  t_lr_tune  = tune("LR",  make_lr(X_train),  LR_PARAM_GRID,  cv, X_train, y_train)
    svm_search, t_svm_tune = tune("SVM", make_svm(X_train), SVM_PARAM_GRID, cv, X_train, y_train)

    best_lr  = lr_search.best_estimator_
    best_svm = svm_search.best_estimator_    # đã có predict_proba

    # ---- metrics trên test
    results = pd.DataFrame({
        "LR Baseline" : calculate_metrics(lr_baseline,  X_test, y_test),
        "LR Tuned"    : calculate_metrics(best_lr,      X_test, y_test),
        "SVM Baseline": calculate_metrics(svm_baseline, X_test, y_test),
        "SVM Tuned"   : calculate_metrics(best_svm,     X_test, y_test),
    })
    results.index.name = "Metric"

    results.loc["CV PR-AUC (tuned)"]     = [None, lr_search.best_score_, None, svm_search.best_score_]
    results.loc["Fit time baseline (s)"] = [t_lr_base, None, t_svm_base, None]
    results.loc["Tuning time (s)"]       = [None, t_lr_tune, None, t_svm_tune]

    print("\nMETRICS COMPARISON (test set, positive = default, threshold = 0.5)")
    print(results.round(4).to_string())

    print("\nBest Logistic Regression Parameters:")
    print(lr_search.best_params_)
    print("\nBest SVM Parameters:")
    print(svm_search.best_params_)

    # ---- lưu
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(lr_baseline,  MODELS_DIR / "lr_baseline.pkl")
    joblib.dump(best_lr,      MODELS_DIR / "lr_tuned.pkl")
    joblib.dump(svm_baseline, MODELS_DIR / "svm_baseline.pkl")
    joblib.dump(best_svm,     MODELS_DIR / "svm_tuned.pkl")

    results.round(4).to_csv(REPORTS_DIR / "lr_svm_results.csv")
    pd.DataFrame({
        "LR" : [str(lr_search.best_params_)],
        "SVM": [str(svm_search.best_params_)],
    }).to_csv(REPORTS_DIR / "lr_svm_best_params.csv", index=False)

    print(f"\nĐã lưu model vào {MODELS_DIR} và kết quả vào {REPORTS_DIR}")


if __name__ == "__main__":
    main()