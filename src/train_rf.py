"""
train_rf.py
Train Random Forest baseline + tuned (GridSearchCV, scoring=average_precision).
Lưu model, best params, kết quả vào models/ và reports/.
"""



# 1. set env and path

import sys
from pathlib import Path

# 1. set env and path


SRC_DIR = Path(__file__).resolve().parent         
PROJECT_ROOT = SRC_DIR.parent                     

DATA_DIR = PROJECT_ROOT/ "data"
PROCESS_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT/ "reports"


sys.path.insert(0, str(SRC_DIR))

print(PROJECT_ROOT, SRC_DIR)

# 2. Load data
import pandas as pd

from preprocess import load_processed_data, split_features_target

# 3. Build preprocessor & pipeline baseline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from preprocess import build_preprocessor
RANDOM_STATE = 0

# 4. Train baseline
import time
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

from evaluate import calculate_metrics


# 5. Tuning GridSearchCV (average_precision)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# 6. save model + result
import joblib





def main():

    # 2. Load data

    train_df, test_df = load_processed_data()

    # print(f"train df: {train_df.shape}")
    # print(f"test df: {test_df.shape}")
    # print(f"Target COl: {TARGET_COL}")


    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)

    print(f"\nX_train: {x_train.shape} | y_train: {y_train.shape}")
    print(f"X_test : {x_test.shape}  | y_test : {y_test.shape}")







    # 3. Build preprocessor & pipeline baseline


    preprocessor = build_preprocessor(x_train, scale_numeric=False)

    rf_baseline = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline_baseline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", rf_baseline),
    ])

    # print(pipeline_baseline)


    # 4. Train baseline
    t0 = time.perf_counter()
    pipeline_baseline.fit(x_train, y_train)

    fit_time = time.perf_counter() - t0



    ## Evaluate baseline
    threshold = 0.5
    base_metrics = calculate_metrics(pipeline_baseline, x_test, y_test, threshold=threshold)

    y_proba_base = pipeline_baseline.predict_proba(x_test)[:,1]
    y_pred_base = (y_proba_base >= threshold).astype(int)

    print("========Baseline=========")
    print(classification_report(y_test, y_pred_base, zero_division=0))

    print("========Metrics=========")
    for k, v in base_metrics.items():
        print(f"{k:15s}: {v:.4f}")

    print("========Confusion Matrix=========")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_base, labels=[0, 1]).ravel()
    print(f"TN = {tn} | FP = {fp}")
    print(f"FN = {fn} | TP = {tp}")

    print(f"Fit time baseline: {fit_time:.2f}s")






    # 5. Tuning GridSearchCV (average_precision)
    ## Set grid + cv

    param_grid = {
        "model__n_estimators":      [200, 300, 500],
        "model__max_depth":         [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__max_features":      ["sqrt", "log2"],
    }

    # CV 5-fold 
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # GridSearchCV với tune average_precision (PR-AUC)
    grid = GridSearchCV(
        estimator=pipeline_baseline,
        param_grid=param_grid,
        scoring="average_precision",  
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )


    ## Fit model
    print("Đang tuning ...")
    t0 = time.perf_counter()
    grid.fit(x_train, y_train)
    tuning_time = time.perf_counter() - t0
    print(f"Tuning xong. Tuning time: {tuning_time:.2f}s")


    print(f"\nBest params        : {grid.best_params_}")
    print(f"Best CV PR-AUC     : {grid.best_score_:.4f}")
    print(f"Best estimator     : {grid.best_estimator_.named_steps['model']}")
    print(f"Tuning time        : {tuning_time:.2f}s")



    ## Evaluate best model
    best_tuned = grid.best_estimator_


    threshold = 0.5
    tuned_metrics = calculate_metrics(best_tuned, x_test, y_test, threshold=threshold)

    y_proba_tuned = best_tuned.predict_proba(x_test)[:,1]
    y_pred_tuned = (y_proba_tuned >= threshold).astype(int)

    print("========Tuned =========")
    print(classification_report(y_test, y_pred_tuned, zero_division=0))

    print("========Metrics=========")
    for k, v in tuned_metrics.items():
        print(f"{k:15s}: {v:.4f}")

    print("========Confusion Matrix=========")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tuned, labels=[0, 1]).ravel()
    print(f"TN = {tn} | FP = {fp}") 
    print(f"FN = {fn} | TP = {tp}")

    print(f"Fit tuning time : {tuning_time:.2f}s")








    # 6. save model + result

    joblib.dump(pipeline_baseline, MODEL_DIR / "rf_baseline.pkl")
    joblib.dump(best_tuned,        MODEL_DIR / "rf_tuned.pkl")


    best_params_str = str(grid.best_params_)
    pd.DataFrame({"RF": [best_params_str]}).to_csv(
        REPORT_DIR / "rf_best_params.csv", index=False
    )

    results_df = pd.DataFrame({
        "RF Baseline @0.5": base_metrics,
        "RF Tuned @0.5":    tuned_metrics,
    })

    results_df.loc["Fit time baseline (s)"] = [fit_time, np.nan]
    results_df.loc["Tuning time (s)"]       = [np.nan, tuning_time]

    results_df = results_df.round(4)

    print("\n=== SUMMARY (RF) ===")
    print(results_df.to_string())

    results_df.to_csv(REPORT_DIR / "rf_results.csv")




if __name__ == "__main__":
    main()