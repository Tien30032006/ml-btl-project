# Credit Risk Prediction - German Credit Data

Dự đoán khả năng một khoản vay bị **vỡ nợ (default)** dựa trên thông tin hồ sơ khách hàng, kèm ứng dụng demo Streamlit để chấm điểm từng hồ sơ hoặc cả file CSV.

Bài tập lớn môn Học máy (CO3117).

## Mục lục

1. [Thành viên nhóm](#thành-viên-nhóm)
2. [Bài toán và dữ liệu](#bài-toán-và-dữ-liệu)
3. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
4. [Cài đặt](#cài-đặt)
5. [Cách chạy](#cách-chạy)
6. [Quy trình xử lý](#quy-trình-xử-lý)
7. [Mô hình và kết quả](#mô-hình-và-kết-quả)
8. [Ứng dụng demo Streamlit](#ứng-dụng-demo-streamlit)
9. [Chạy bằng Docker](#chạy-bằng-docker)
10. [Lưu ý quan trọng](#lưu-ý-quan-trọng)
11. [Xử lý sự cố](#xử-lý-sự-cố)

## Thành viên nhóm

| MSSV | Họ tên | Vai trò |
|---|---|---|
| 2411964 | Trần Xuân Lộc | Data & Feature Lead (TV1) |
| 2412867 | Bùi Ngọc Minh Quân | Model Lead A - LR & SVM (TV2) |
| 2412443 | Trần Thiện Nhân | Model Lead B - Random Forest (TV3) |
| 2413468 | Huỳnh Tấn Tiến | Evaluation + Demo + Report Lead (TV4) |

## Bài toán và dữ liệu

**Bài toán:** phân loại nhị phân. Với mỗi hồ sơ vay, ước lượng xác suất `P(default)` rồi quyết định **duyệt** hoặc **từ chối** theo một ngưỡng.

**Dữ liệu:** German Credit Data (Statlog), 1.000 mẫu.

- UCI: <https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)>
- Bản CSV (Kaggle): <https://www.kaggle.com/datasets/jumpingdino/german-credit-dataset>
- File dùng trong project: `data/raw/modified_GermanCredit.csv` (20 đặc trưng gốc + nhãn `credit_risk`).

**Đặc trưng sau khi chọn lọc:** 18 cột (danh sách trong `data/processed/final_features.json`), bỏ `job` và `telephone`.

- Numeric (7): `duration`, `amount`, `installment_rate`, `present_residence`, `age`, `number_credits`, `people_liable`.
- Categorical (11): `status`, `credit_history`, `purpose`, `savings`, `employment_duration`, `personal_status_sex`, `other_debtors`, `property`, `other_installment_plans`, `housing`, `foreign_worker`.

**Chia dữ liệu:** stratified 80/20 → 800 mẫu train, 200 mẫu test. Tỷ lệ default ở cả hai tập đều khoảng **30%** (dữ liệu mất cân bằng).

**Quy ước nhãn (rất quan trọng):**

| Ở đâu | Giá trị | Ý nghĩa |
|---|---|---|
| File CSV (`credit_risk`) | `1` | Good (không default) |
| File CSV (`credit_risk`) | `0` | Bad (default) |
| Trong code (`y`) | `1` | **Default** (positive class) |
| Trong code (`y`) | `0` | Good |

Việc đảo nhãn được thực hiện ở một nơi duy nhất: `split_features_target()` trong `src/preprocess.py`. Mọi metric (Precision, Recall, PR-AUC, ...) đều tính cho lớp **default**.

**Cost matrix:** bỏ sót một khách default (FN) tốn **5**, từ chối nhầm một khách tốt (FP) tốn **1**.

|  | Dự đoán Good | Dự đoán Default |
|---|---|---|
| **Thực tế Good** | 0 | 1 (FP) |
| **Thực tế Default** | 5 (FN) | 0 |

Nếu xác suất được hiệu chỉnh tốt, ngưỡng tối ưu về lý thuyết là `1 / (1 + 5) ≈ 0.17`.

## Cấu trúc thư mục

```
.
├── data/
│   ├── raw/                  # dữ liệu gốc, không sửa
│   ├── processed/            # train.csv, test.csv, final_features.json
│   └── README.md             # nguồn dữ liệu, mô tả cột, mapping nhãn
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_selection.ipynb
│   └── 03_models.ipynb
├── src/
│   ├── preprocess.py         # hàm dùng chung: load, đảo nhãn, build preprocessor
│   ├── train_lr_svm.py       # Logistic Regression + SVM (baseline / tuned)
│   ├── train_rf.py           # Random Forest (baseline / tuned)
│   └── evaluate.py           # metric, chọn ngưỡng theo cost, biểu đồ, SHAP, chọn best model
├── models/
│   ├── lr_*.pkl, svm_*.pkl, rf_*.pkl     # 6 model (baseline + tuned)
│   ├── best_model.pkl                    # model tốt nhất, do evaluate.py tạo
│   └── best_model_meta.json              # tên model + ngưỡng tối ưu
├── app/
│   └── streamlit_app.py      # demo Streamlit (Overview / Single Scoring / Batch Scoring)
├── reports/
│   ├── lr_svm_results.csv, rf_results.csv      # kết quả từng nhóm model @0.5
│   ├── lr_svm_best_params.csv, rf_best_params.csv
│   ├── final_comparison.csv                    # bảng so sánh cuối cùng (evaluate.py)
│   ├── figures/                                # ROC/PR, cost curve, confusion matrix, SHAP
│   ├── report.pdf
│   └── slides.pdf
├── requirements.txt
├── Dockerfile
└── .gitignore
```

## Cài đặt

Yêu cầu: **Python 3.10+**.

```bash
git clone https://github.com/Tien30032006/ml-btl-project.git
cd ml-btl-project
git checkout feature/eval-demo

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Các file `.pkl` phụ thuộc phiên bản scikit-learn. Nếu thấy cảnh báo `InconsistentVersionWarning` khi load model, hãy train lại (xem bước 2 bên dưới) để model khớp với môi trường của bạn.

## Cách chạy

Tất cả lệnh chạy từ **thư mục gốc** của project.

**Bước 1 - Dữ liệu.** `data/processed/train.csv`, `test.csv` và `final_features.json` đã có sẵn trong repo (sinh ra từ notebook `02_feature_selection.ipynb`). `src/preprocess.py` chứa các hàm dùng chung; muốn tạo lại train/test từ `data/raw/modified_GermanCredit.csv` thì chạy `python src/preprocess.py`.

**Bước 2 - Huấn luyện mô hình.**

```bash
python src/train_lr_svm.py     # LR + SVM, vài chục giây
python src/train_rf.py         # RF, tuning mất vài phút (lâu hơn nếu ít CPU)
```

**Bước 3 - Đánh giá và chọn best model.**

```bash
python src/evaluate.py                 # đầy đủ, có SHAP
python src/evaluate.py --skip-shap     # nhanh hơn
```

Bước này tạo `models/best_model.pkl`, `models/best_model_meta.json`, `reports/final_comparison.csv` và các hình trong `reports/figures/`.

**Bước 4 - Chạy demo.**

```bash
streamlit run app/streamlit_app.py
```

Mở trình duyệt tại <http://localhost:8501>.

## Quy trình xử lý

1. **Tiền xử lý** (`build_preprocessor` trong `src/preprocess.py`):
   - Categorical: `OneHotEncoder(handle_unknown="ignore")`.
   - Numeric: `StandardScaler` cho LR/SVM, `passthrough` cho Random Forest.
   - Preprocessor nằm trong `Pipeline`, nên chỉ fit trên train (và trên train-fold khi cross-validation), không rò rỉ dữ liệu.
2. **Huấn luyện:** với mỗi họ mô hình có bản baseline và bản tuned bằng `GridSearchCV` (`StratifiedKFold`, 5 fold, `random_state=0`).
   - LR/SVM tối ưu theo PR-AUC (`average_precision`), theo dõi thêm ROC-AUC và F1.
   - RF tối ưu theo PR-AUC.
3. **Chọn ngưỡng theo cost** (`evaluate.py`): với mỗi model, tính xác suất out-of-fold trên **train** bằng 5-fold CV, quét ngưỡng từ 0.01 đến 0.99 và chọn ngưỡng có Expected Cost nhỏ nhất. Test set không tham gia bước này.
4. **Chọn best model:** model có Expected Cost out-of-fold trên train thấp nhất (tại ngưỡng riêng của nó). Test set chỉ dùng để báo cáo.
5. **Giải thích:** SHAP (`TreeExplainer` cho RF, `LinearExplainer` cho LR). SVM chưa hỗ trợ giải thích SHAP; nếu best model là SVM, app sẽ báo không tạo được giải thích.

## Mô hình và kết quả

**Tham số tốt nhất (GridSearchCV, tối ưu PR-AUC, 5-fold):**

| Model | Tham số | CV PR-AUC |
|---|---|---|
| Logistic Regression | `C=0.1`, `l1_ratio=0.0` (L2), `solver=saga`, `class_weight=None` | 0.632 |
| SVM | `kernel=rbf`, `C=1`, `gamma=scale`, `class_weight=None` (trùng với mặc định nên Baseline = Tuned) | 0.645 |
| Random Forest | `n_estimators=500`, `max_depth=10`, `max_features=sqrt`, `min_samples_split=2` (`class_weight=balanced` cố định) | 0.653 |

**Kết quả trên test set** (200 mẫu, positive = default). `t*` là ngưỡng chọn từ train out-of-fold.

| Model | ROC-AUC | PR-AUC | Cost @0.5 | t* | Precision @t* | Recall @t* | F1 @t* | Cost @t* |
|---|---|---|---|---|---|---|---|---|
| LR Baseline | 0.768 | 0.574 | 0.915 | 0.13 | 0.410 | 0.950 | 0.573 | 0.485 |
| LR Tuned | 0.779 | 0.596 | 0.875 | 0.19 | 0.423 | 0.917 | 0.579 | 0.500 |
| SVM Baseline | 0.785 | 0.594 | 0.880 | 0.14 | 0.377 | 0.967 | 0.542 | 0.530 |
| SVM Tuned | 0.785 | 0.594 | 0.880 | 0.14 | 0.377 | 0.967 | 0.542 | 0.530 |
| RF Baseline | 0.776 | 0.615 | 0.825 | 0.24 | 0.399 | 0.950 | 0.562 | 0.505 |
| **RF Tuned** | 0.782 | **0.637** | **0.680** | **0.31** | 0.407 | 0.917 | 0.564 | 0.525 |

Số liệu đầy đủ nằm trong `reports/final_comparison.csv`.

**Nhận xét:**

- Các model có ROC-AUC gần nhau (khoảng 0.77–0.79), nên khác biệt giữa chúng nhỏ.
- Hạ ngưỡng giảm Expected Cost rõ rệt. LR/SVM giảm mạnh nhất (khoảng 0.88 → 0.50) vì mặc định chúng không ưu tiên lớp default. RF dùng `class_weight=balanced` nên đã có cost thấp ngay ở ngưỡng 0.5 (0.68), và hạ ngưỡng còn giảm thêm được 0.68 → 0.525 (khoảng 23%). Đổi lại, Precision giảm xuống còn khoảng 0.4. Đây là đánh đổi chủ ý vì FN đắt gấp 5 lần FP.
- Vì `class_weight=balanced` làm lệch xác suất của RF, `t* = 0.31` không so sánh trực tiếp được với ngưỡng lý thuyết 1/6.
- **Best model được chọn: RF Tuned, ngưỡng 0.31**, theo cost out-of-fold trên train (RF Tuned 0.529, LR Tuned 0.545, SVM 0.554). Khoảng cách này rất nhỏ. Trên test, LR Baseline có cost thấp nhất (0.485 so với 0.525 của RF Tuned), tức chênh khoảng 8 đơn vị chi phí trên 200 mẫu, chưa tới 2 lần bỏ sót khách default. Vì vậy chênh lệch này nằm trong dao động ngẫu nhiên, và cách chọn không nhìn test được giữ để kết quả báo cáo trung thực.
- Cost @t* trên test (0.525) gần với cost OOF trên train (0.529), cho thấy ngưỡng không bị chọn quá khớp với dữ liệu train.
- SHAP cho thấy các đặc trưng ảnh hưởng nhiều nhất: `status` (tài khoản checking: "no checking account", "< 100 DM"), `duration`, `credit_history`, `savings`, `age`, `amount` và `purpose`.

**Các hình trong `reports/figures/`:** `roc_pr_curves.png`, `cost_curves.png`, `confusion_default.png`, `confusion_best_threshold.png`, `shap_summary.png`.

## Ứng dụng demo Streamlit

Ứng dụng đọc `models/best_model.pkl` và ngưỡng mặc định từ `models/best_model_meta.json`. Thanh bên có slider để đổi ngưỡng từ chối.

| Trang | Chức năng |
|---|---|
| **Overview** | Giới thiệu bài toán, vì sao không dùng Accuracy, bảng so sánh 6 model, các hình ROC/PR, cost curve, confusion matrix, SHAP |
| **Single Scoring** | Nhập 18 đặc trưng của một hồ sơ, xem `P(default)`, mức rủi ro, quyết định duyệt/từ chối và giải thích SHAP |
| **Batch Scoring** | Tải lên CSV, nhận file kết quả có thêm cột `p_default`, `decision`, `risk_band`. Nếu file có cột `credit_risk` (1 = good, 0 = bad) sẽ tính thêm metric |

**Định dạng file cho Batch Scoring:**

- Phải có đủ 18 cột đặc trưng trong `data/processed/final_features.json`. Cột thừa (như `job`, `telephone`) được bỏ qua.
- Giá trị categorical phải đúng chuỗi như trong dữ liệu gốc (ví dụ `no checking account`, `own`, `rent`).
- Trang có nút tải file mẫu để dùng làm khuôn.

## Chạy bằng Docker

```bash
docker build -t credit-risk-demo .
docker run -p 8501:8501 credit-risk-demo
```

Sau đó mở <http://localhost:8501>.

Image copy cả thư mục `models/` và `reports/`, vì vậy hãy chạy `python src/evaluate.py` để có `best_model.pkl` **trước khi** build.

## Lưu ý quan trọng

- **Không dùng Accuracy làm metric chính.** Dữ liệu mất cân bằng (30% default) và chi phí bất đối xứng. Dùng PR-AUC, Recall và Expected Cost.
- **Không chọn ngưỡng hay chọn model bằng test set.** Cả hai đều dựa trên xác suất out-of-fold của train. Test chỉ để báo cáo.
- **Xác suất của SVM/RF có thể chưa hiệu chỉnh tốt**, nên `P(default)` là điểm xếp hạng rủi ro hơn là xác suất chính xác.
- Tập dữ liệu chỉ có 1.000 mẫu, các con số có độ bất định lớn. Đừng diễn giải chênh lệch nhỏ giữa các model như khác biệt thật sự.
- Các file `.pkl` là pickle: chỉ load file từ nguồn tin cậy.

## Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| Streamlit báo "Chưa có `models/best_model.pkl`" | Chạy `python src/evaluate.py` |
| `ModuleNotFoundError: preprocess` | Chạy lệnh từ thư mục gốc, ví dụ `python src/train_rf.py` |
| `InconsistentVersionWarning` khi load model | Train lại bằng `train_lr_svm.py` và `train_rf.py`, rồi chạy lại `evaluate.py` |
| Batch Scoring báo "Thiếu cột" | Đối chiếu tên cột với `data/processed/final_features.json` |
| `train_rf.py` chạy rất lâu | Giảm `param_grid` trong file hoặc chạy trên máy nhiều CPU hơn (grid hiện tại: 72 tổ hợp × 5 fold) |
| SHAP lỗi hoặc quá chậm | Chạy `python src/evaluate.py --skip-shap` |