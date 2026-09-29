from preprocess import (
    load_processed_data,
    split_features_target,
    build_preprocessor
)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Lấy data từ processed và chia ra features/nhãn
train_df, test_df = load_processed_data()
X_train, y_train = split_features_target(train_df)
X_test, y_test = split_features_target(test_df)

# Bộ tiền xử lý: categorical features thì onehotencoding, numerical features thì scale về phân phối chuẩn
preprocessor = build_preprocessor(X_train, scale_numeric=True)

# Pipeline Logistic Regression
lr_baseline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",LogisticRegression())
])
# Pipeline SVM
svm_baseline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(probability=True,random_state=0)) #probability=True để tính xác suất dùng trong tính metric
    #random_state để cố định kết quả random giống nhau cho nhiều lần chạy
])

# Train mô hình baseline
lr_baseline.fit(X_train, y_train)
svm_baseline.fit(X_train, y_train)

#Tuning các hyperparameter dùng cross validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

#Thử các tổ hợp tham số
lr_param_grid = [
    {
        # C là Regularization parameter  điều hòa giữa độ chính xác và nguy cơ overfitting
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        #penalty là hàm phạt để kiểm soát overfitting
        "classifier__penalty": ["l2"],
        #solver là thuật toán tối ưu tham số
        "classifier__solver": ["lbfgs", "saga", "liblinear"]#lbfgs chỉ dùng với l2
    },
    {
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l1"],
        "classifier__solver": ["saga", "liblinear"]
    }
]

svm_param_grid = {
    "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "classifier__kernel": ["linear", "rbf"], # tuyến tính hoặc radial basis function
    "classifier__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1] #Phạm vi ảnh hưởng của một điểm dữ liệu đến boundary
}

#KFold chia làm 5 phần với tỷ lệ nhãn của mỗi phần bằng nhau (Stratified)
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=0
)

#Tune các hyperparameter
lr_tuning = GridSearchCV(
    estimator=lr_baseline,
    param_grid=lr_param_grid,
    scoring="f1",   #Tiêu chí quyết định bộ hyperparameter nào là tốt nhất
    cv=cv,
    n_jobs=-1   #Để tận dụng hết các core cpu
)
svm_tuning = GridSearchCV(
    estimator=svm_baseline,
    param_grid=svm_param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)
lr_tuning.fit(X_train, y_train)
svm_tuning.fit(X_train, y_train)

#Bộ hyperparameter tốt nhất
best_lr=lr_tuning.best_estimator_
best_svm=svm_tuning.best_estimator_

#So sánh các metric trước và sau khi tune
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc
)
# Hàm tính metric
def calculate_metrics(model, X_test, y_test): 
    y_pred = model.predict(X_test)  #Giá trị dự đoán 
    y_prob = model.predict_proba(X_test)[:, 1]  #Xác suất lớp 1

    #Tính các metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_curve, precision_curve)

    #Trả về dictionary các metric
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    }
#Tính metric cho cả 4 model
lr_baseline_metrics = calculate_metrics(lr_baseline, X_test, y_test)
lr_tuned_metrics = calculate_metrics(best_lr, X_test, y_test)
svm_baseline_metrics = calculate_metrics(svm_baseline, X_test, y_test)
svm_tuned_metrics = calculate_metrics(best_svm, X_test, y_test)

import pandas as pd
#In ra bảng so sánh các metrics
comparison_all = pd.DataFrame({
    "Metric": ["Precision","Recall","F1","ROC-AUC","PR-AUC"],
    "LR Baseline": [*lr_baseline_metrics.values()],
    "LR Tuned": [*lr_tuned_metrics.values()],
    "SVM Baseline": [*svm_baseline_metrics.values()],
    "SVM Tuned": [*svm_tuned_metrics.values()]
})

print("\nMETRICS COMPARISON")
print(comparison_all.round(4)) # Làm tròn giá trị đến 4 chữ số thập phân

# In ra bộ hyperparameters tốt nhất
print("\nBest Logistic Regression Parameters:")
print(lr_tuning.best_params_)
print("\nBest SVM Parameters:")
print(svm_tuning.best_params_)