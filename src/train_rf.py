import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import json
import pandas as pd
from src.preprocess import load_processed_data, split_features_target, build_preprocessor


import time
import joblib
import numpy as np

from sklearn.model_selection import GridSearchCV, StratifiedKFold


#-------------------------------
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

#----------------------------------

from sklearn.metrics import (classification_report,
                            roc_auc_score,
                            average_precision_score,       
                            precision_score,
                            recall_score,
                            f1_score,
                            confusion_matrix,
                            make_scorer,

)
                            







#---------------------------------------
#---------------------------------------
# Caculate metrix - expect cost
#---------------------------------------
#---------------------------------------

C_FN = 5.0   # bỏ sót khách tệ -> mất tiền cho vay
C_FP = 1.0   # từ chối nhầm khách tốt ->  mất lợi nhuận

def expected_cost(y_true, y_pred, c_fn=C_FN, c_fp=C_FP):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return (c_fn * fn + c_fp * fp) / len(y_true)


def evaluate_model(model, x_test, y_test, threshold = 0.5,  fit_time=None, tuning_time=None,
                   cv_score=None, best_params=None, name="model"):
   
    t0 = time.perf_counter()
    y_proba = model.predict_proba(x_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    metrics = {
        "model":               name,
        "Precision":           precision_score(y_test, y_pred, zero_division=0),
        "Recall":              recall_score(y_test, y_pred, zero_division=0),
        "F1":                  f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC":             roc_auc_score(y_test, y_proba),
        "PR-AUC":              average_precision_score(y_test, y_proba),
        "Expected Cost":       expected_cost(y_test, y_pred),
        "CV Cost (tuned)": cv_score if cv_score is not None else np.nan,
        "Best params":         str(best_params) if best_params else "-",
        "Fit time baseline (s)": fit_time if fit_time is not None else np.nan,
        "Tuning time (s)":       tuning_time if tuning_time is not None else np.nan,
    }
    return metrics, y_proba


def print_metrics(metrics, y_test, y_pred):
    # in ra result model

    print(classification_report(y_test, y_pred, zero_division=0))

    print(f"Precision={metrics['Precision']:.4f} | Recall={metrics['Recall']:.4f} "
          f"| F1={metrics['F1']:.4f}")
    
    print(f"ROC-AUC={metrics['ROC-AUC']:.4f} | PR-AUC={metrics['PR-AUC']:.4f} "
          f"| Cost={metrics['Expected Cost']:.4f}")








# stepp 1: lay data
# load data from train test
train_df, test_df = load_processed_data()

# lay final fratures
with open("data/processed/final_features.json", "r") as fin:
    final_features = json.load(fin)

# split x, y
x_train, y_train = split_features_target(train_df)
x_test, y_test = split_features_target(test_df)





# Step 2: build pipeline for RF baseline


#------------------------------------
# khong scale bien so, chi OneHot encoder
preprocessor = build_preprocessor(x_train, scale_numeric = False)

#baseline model
rf_baseline = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)


# phép vo pipelone
pipeline_baseline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rf_baseline)
])






# Step 3: train va evalueat MODEL BASELINE


t0 = time.perf_counter()
pipeline_baseline.fit(x_train, y_train)
fit_time_baseline = time.perf_counter() - t0



# run and predict
base_metrics, y_proba_base = evaluate_model(
    pipeline_baseline, x_test, y_test,
    fit_time=fit_time_baseline,
    name="RF baseline",
)


y_pred_base = (y_proba_base >= 0.5).astype(int)
print_metrics(base_metrics, y_test, y_pred_base)
print(f"Fit time baseline: {fit_time_baseline:.2f}s")






# ===============
# GRID SEARCH CV 

print("\n=== TUNING (GridSearchCV) ===")


def expected_cost_scorer(estimator, X, y):
    """
    Custom scorer cho tune vì mặc định đang lấy pr auc, phải đổi qua chọn cost expect thấp nhất:
    - Lấy xác suất dự đoán
    - Tìm threshold tốt nhất (cost thấp nhất) trên chính fold này
    - Trả về -cost 
    """

    # 1. Lấy xác suất lớp 1 (khách tệ)
    y_proba = estimator.predict_proba(X)[:, 1]

    # 2. Quét threshold để tìm cost nhỏ nhất
    best_cost = np.inf
    for thr in np.linspace(0.05, 0.95, 91):
        y_pred = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        cost = (C_FN * fn + C_FP * fp) / len(y)
        if cost < best_cost:
            best_cost = cost

    return -best_cost





param_grid = {
    "model__n_estimators":      [200, 300, 500],
    "model__max_depth":         [None, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__max_features":      ["sqrt", "log2"],
}


# cross validation - chia thành 5 phần
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipeline_baseline, param_grid,
    scoring = expected_cost_scorer, 
    cv=cv,
    n_jobs = -1,  # dùng full cpu
    verbose = 1,   # log 
    refit = True,
)

t0 = time.perf_counter()
grid.fit(x_train, y_train)
tuning_time = time.perf_counter() - t0

print(f"Best params: {grid.best_params_}")
print(f"CV Expected Cost  (tuned): {-grid.best_score_:.4f}")
print(f"Tuning time: {tuning_time:.2f}s")






# ============ CHỌN THRESHOLD TRÊN OOF ============
from sklearn.model_selection import cross_val_predict

y_proba_oof = cross_val_predict(
    grid.best_estimator_, x_train, y_train,
    cv=cv, method="predict_proba", n_jobs=-1
)[:, 1]

best_thr, best_cost_oof = 0.5, np.inf
for thr in np.linspace(0.05, 0.95, 91):
    y_pred = (y_proba_oof >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred, labels=[0, 1]).ravel()
    cost = (C_FN * fn + C_FP * fp) / len(y_train)
    if cost < best_cost_oof:
        best_cost_oof, best_thr = cost, thr

print(f"Best threshold (OOF): {best_thr:.2f}")




# ============ ĐÁNH GIÁ TEST ============
best = grid.best_estimator_
y_proba_t = best.predict_proba(x_test)[:, 1]
y_pred_t  = (y_proba_t >= best_thr).astype(int)

tuned_metrics, _ = evaluate_model(
    best, x_test, y_test,
    threshold=best_thr,

    tuning_time=tuning_time,
    cv_score=-grid.best_score_,
    best_params=grid.best_params_,
    name="RF tuned (cost)",
)
print_metrics(tuned_metrics, y_test, y_pred_t)







# ========== results.csv ne ==========
# base_metrics và tuned_metrics là dict → chuyển thành DataFrame dạng cột
df = pd.DataFrame({
    "RF Baseline": base_metrics,
    "RF Tuned":    tuned_metrics,
})

df = df.drop(index="model", errors="ignore")

# Bỏ dòng "Best params" (đã lưu riêng ở rf_best_params.csv)
df = df.drop(index="Best params", errors="ignore")

df.loc["Time (s)"] = [
    df.loc["Fit time baseline (s)", "RF Baseline"] if "Fit time baseline (s)" in df.index else None,
    df.loc["Tuning time (s)", "RF Tuned"] if "Tuning time (s)" in df.index else None,
]
df = df.drop(index=["Fit time baseline (s)", "Tuning time (s)"], errors="ignore")

df_rounded = df.copy()
for col in df_rounded.columns:
    df_rounded[col] = df_rounded[col].apply(
        lambda x: round(x, 4) if isinstance(x, (int, float)) and not pd.isna(x) else x
    )







print("\n=== SUMMARY (RF) ===")
print(df_rounded.to_string())

df_rounded.to_csv("reports/rf_results.csv")



joblib.dump(best, "models/rf_tuned.pkl")
pd.DataFrame([grid.best_params_]).to_csv("reports/rf_best_params.csv", index=False)
df_rounded.to_csv("reports/rf_results.csv")
