# Dữ liệu và phân tích preprocessing

Tài liệu này tập hợp nguồn dữ liệu, ý nghĩa cột, quy ước nhãn, kết quả EDA và
quyết định preprocessing của dự án. Phân tích có thể xem/chạy lại trong
[`01_eda.ipynb`](../notebooks/01_eda.ipynb) và
[`02_feature_selection.ipynb`](../notebooks/02_feature_selection.ipynb).

## 1. Nguồn và phiên bản dữ liệu

- Bộ dữ liệu gốc: **Statlog (German Credit Data)**, Hans Hofmann, 1994,
  [UCI ML Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data),
  [DOI: 10.24432/C5NC77](https://doi.org/10.24432/C5NC77), giấy phép CC BY 4.0.
- Mô tả thuộc tính gốc: [UCI `german.doc`](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc).
- File thực tế được phân tích: [`raw/modified_GermanCredit.csv`](raw/modified_GermanCredit.csv).
  Đây là bản CSV đã có tên cột và nhãn dạng chuỗi, không phải file `german.data`
  nguyên bản của UCI. README chính của repo dẫn
  [bản CSV trên Kaggle](https://www.kaggle.com/datasets/jumpingdino/german-credit-dataset).
  Repo chưa lưu ngày tải, phiên bản Kaggle hay script chuyển đổi từ UCI;
  không suy đoán các thông tin này. Khi tái lập kết quả, dùng chính CSV trong repo.
- Bản modified có quy ước riêng: `credit_risk` dùng **1/0**, khác nhãn **1/2**
  của UCI. Cột `status` cũng có chuỗi `... < 100 DM`, trong khi mô tả UCI ghi
  `... < 0 DM`; giữ nguyên giá trị CSV và không tự sửa mapping khi chưa có
  thông tin về bước chuyển đổi dữ liệu.

## 2. Quy mô và chất lượng dữ liệu

Kết quả kiểm tra trực tiếp trên CSV và notebook EDA:

| Nội dung | Kết quả |
|---|---|
| Số dòng | 1.000 |
| Số cột | 21: 20 đặc trưng + `credit_risk` |
| Đặc trưng trước chọn lọc | 7 numeric, 13 categorical |
| Giá trị thiếu (`isna`) | 0 |
| Dòng trùng hoàn toàn (`duplicated`) | 0 |
| Good (`credit_risk = 1`) | 700, chiếm 70% |
| Bad (`credit_risk = 0`) | 300, chiếm 30% |

Không cần imputation hoặc xóa dòng trùng cho bản CSV hiện tại. Những mức như
`unknown/no savings account`, `no checking account`, `unknown/no property` là
các giá trị categorical hiện hữu; không tự chuyển chúng thành NaN. Dữ liệu mất
cân bằng ở mức 70/30, nên chia stratified để bảo toàn tỷ lệ nhãn; không resample
trong EDA hoặc các file processed.

## 3. Ý nghĩa các cột

Tên và giá trị bên dưới là của CSV đang dùng; ý nghĩa nghiệp vụ đối chiếu với
tài liệu thuộc tính UCI ở mục nguồn. Numeric/categorical là cách pipeline hiện tại
xử lý cột, không khẳng định mọi cột có dtype số đều là đại lượng liên tục.

| Cột | Ý nghĩa và giá trị trong CSV | Xử lý |
|---|---|---|
| `status` | Tình trạng tài khoản thanh toán, 4 mức; có mức không có tài khoản | Categorical |
| `duration` | Kỳ hạn vay, 4–72 tháng | Numeric |
| `credit_history` | Lịch sử trả nợ, 5 mức | Categorical |
| `purpose` | Mục đích vay, 10 mức, ví dụ mua xe, học tập | Categorical |
| `amount` | Số tiền vay, 250–18.424 DM | Numeric |
| `savings` | Tiền tiết kiệm/trái phiếu, 5 mức | Categorical |
| `employment_duration` | Thâm niên việc làm, 5 mức, bao gồm thất nghiệp | Categorical |
| `installment_rate` | Mức tỷ lệ trả góp trên thu nhập khả dụng; mã 1–4 | Numeric |
| `personal_status_sex` | Tình trạng hôn nhân kết hợp giới tính, 4 mức quan sát | Categorical |
| `other_debtors` | Người đồng vay/bảo lãnh, hoặc không có | Categorical |
| `present_residence` | Mức thời gian cư trú tại nơi ở hiện tại; mã 1–4 | Numeric |
| `property` | Nhóm tài sản, 4 mức | Categorical |
| `age` | Tuổi, 19–75 năm | Numeric |
| `other_installment_plans` | Khoản trả góp khác: ngân hàng, cửa hàng, không có | Categorical |
| `housing` | Nhà thuê, sở hữu hoặc ở miễn phí | Categorical |
| `number_credits` | Số khoản tín dụng tại ngân hàng, 1–4 | Numeric |
| `job` | Nhóm nghề nghiệp, 4 mức | Categorical; loại sau feature selection |
| `people_liable` | Số người cần cấp dưỡng, 1–2 | Numeric |
| `telephone` | Có điện thoại đăng ký tên khách hàng hay không, `yes`/`no` | Categorical; loại sau feature selection |
| `foreign_worker` | Người lao động nước ngoài, `yes`/`no` | Categorical |
| `credit_risk` | Nhãn rủi ro tín dụng: `1 = good`, `0 = bad` | Target; không đưa vào X |

Các mã như `installment_rate`, `present_residence` giữ dạng số theo thiết kế
hiện tại. Không diễn giải mã 4 thành tỷ lệ chính xác 4% hay đúng 4 năm khi chưa
có bảng quy đổi của bản modified. `amount` có đơn vị DM của bộ dữ liệu lịch sử.

## 4. Mapping nhãn và chi phí phân loại sai

| Nguồn | Good | Bad/default |
|---|---|---|
| UCI nguyên bản | 1 | 2 |
| CSV trong `raw/` và `processed/`: `credit_risk` | 1 | 0 |
| Nhãn `y` cho model trong `src/` | 0 | 1 (positive class) |

[`split_features_target()`](../src/preprocess.py) thực hiện
`y = (df["credit_risk"] != 1).astype(int)` khi đưa dữ liệu vào model.
**Không đảo nhãn trước khi lưu train/test CSV**, tránh đảo hai lần.
Notebook 02 giữ nhãn CSV cho chi-square, RF selector và CV so sánh tập đặc trưng;
ROC-AUC/F1-macro trong phần đó không phải Precision/Recall cho default của model cuối.

Cost matrix UCI gán chi phí 5 khi khách bad bị dự đoán good và 1 cho chiều ngược
lại. Với positive = default của dự án, tương ứng **FN = 5**, **FP = 1**;
phân loại đúng có chi phí 0. Đây là lý do cần xem Recall, PR-AUC và expected cost,
không chỉ Accuracy trên dữ liệu 70/30.

## 5. Phân tích EDA và quyết định về outlier

`amount`, `duration`, `age` có đuôi phải và một số điểm ngoài boxplot.
Theo quy tắc 1,5 × IQR trên toàn bộ dữ liệu, các thống kê mô tả là:

| Cột | Min–max | Cận dưới–trên IQR | Số giá trị ngoài cận |
|---|---|---|---|
| `amount` | 250–18.424 | -2.544,625–7.882,375 | 72 |
| `duration` | 4–72 | -6–42 | 70 |
| `age` | 19–75 | 4,5–64,5 | 23 |

**Giữ nguyên toàn bộ mẫu: không xóa, cắt ngọn/winsorize hoặc log-transform.**
Điểm ngoài IQR chưa đủ để kết luận lỗi nhập liệu; giữ các khoản vay lớn hoặc
khách hàng cao tuổi giúp tránh mất thông tin. Các biến số rời rạc như
`people_liable` cần diễn giải theo miền giá trị thay vì tự động xóa điểm ngoài boxplot.
Random Forest thường ít nhạy với độ lớn cực trị do dùng ngưỡng chia nhánh.
LR/SVM dùng StandardScaler trong pipeline, nhưng scaling không xử lý triệt để
outlier; đây vẫn là một giới hạn cần theo dõi qua kết quả CV.

Các ngưỡng IQR trên chỉ phục vụ mô tả EDA, không dùng để lọc dữ liệu.
Nếu thử log-transform, clipping hay RobustScaler về sau, cần chọn phương án
bằng CV trên train, học tham số trên từng train-fold và giữ test cho đánh giá cuối.

Pearson correlation lớn nhất theo trị tuyệt đối giữa các cột số là
`duration`–`amount`, khoảng **0,625**. Không có cặp nào vượt **0,8**, nên không
loại đặc trưng theo tiêu chí tương quan này. Một số nhóm categorical có ít mẫu
(ví dụ `foreign_worker = no` có 37 dòng); hiện giữ nguyên nhóm và dùng
`OneHotEncoder(handle_unknown="ignore")` để xử lý mức mới khi suy luận.

## 6. Feature selection và lý do loại `telephone`, `job`

Notebook 02 chia train/test trước; chi-square và RF selector chỉ dùng 800 dòng
train. RF dùng 300 cây, `class_weight="balanced"`, `random_state=42`.
Importance của cột gốc được tính bằng tổng importance các cột one-hot tương ứng.
Kết quả **đã lưu trong notebook**:

| Cột | Chi-square p-value | RF importance | Hạng RF / 20 |
|---|---|---|---|
| `telephone` | 0,219541 | 0,026888 | 16 |
| `job` | 0,724695 | 0,037620 | 12 |

Cả hai có p-value > 0,05 nên chưa có đủ bằng chứng về liên hệ đơn biến với nhãn;
RF importance cũng tương đối thấp so với các cột đầu bảng, nhưng **không phải
hai cột thấp nhất**. Không dùng riêng p-value hoặc importance để tự động loại cột:
ví dụ `personal_status_sex` có p ≈ 0,0709 nhưng vẫn được giữ. RF importance có
thể dao động nhẹ khi đổi phiên bản scikit-learn, không phải bằng chứng nhân quả.

Để kiểm tra quyết định, notebook so sánh Logistic Regression cân bằng lớp bằng
StratifiedKFold 5-fold (`shuffle=True`, `random_state=42`) trên train.
One-hot và StandardScaler nằm trong pipeline, fit lại trên từng train-fold.

| Tập đặc trưng | Số cột | ROC-AUC mean ± std | F1-macro mean ± std |
|---|---|---|---|
| Đầy đủ | 20 | 0,787165 ± 0,067699 | 0,708415 ± 0,068879 |
| Bỏ `telephone` | 19 | 0,786942 ± 0,072135 | 0,709452 ± 0,060533 |
| Bỏ `telephone` và `job` | 18 | 0,790513 ± 0,073139 | 0,704052 ± 0,058888 |

**Chọn 18 cột** để giảm số đầu vào với ưu tiên ROC-AUC và hiệu năng tương đương
tập đầy đủ. AUC tăng rất nhỏ, F1-macro giảm nhẹ; chênh lệch nằm trong dao động
giữa các fold nên không kết luận cải thiện có ý nghĩa thống kê hoặc mọi metric
đều tốt hơn. Test không được dùng để chọn tập đặc trưng. Vì các ứng viên đã được
gợi ý từ toàn bộ train, CV này là phép so sánh nội bộ; kết quả test cuối mới là
đánh giá trên tập giữ riêng.

Danh sách cuối và thứ tự cột lưu ở [`processed/final_features.json`](processed/final_features.json).
Phân theo nhóm:

- Numeric (7): `duration`, `amount`, `installment_rate`, `present_residence`,
  `age`, `number_credits`, `people_liable`.
- Categorical (11): `status`, `credit_history`, `purpose`, `savings`,
  `employment_duration`, `personal_status_sex`, `other_debtors`, `property`,
  `other_installment_plans`, `housing`, `foreign_worker`.

## 7. Pipeline, kết quả xuất và tái lập

1. Đọc CSV gốc; tách target khỏi 20 cột đầu vào.
2. Chia stratified 80/20: train 800 dòng (560 good, 240 bad), test 200 dòng
   (140 good, 60 bad).
3. Phân tích train, so sánh tập đặc trưng và bỏ `telephone`, `job`.
4. Xuất 18 cột đặc trưng + nhãn CSV, chưa encoding/scaling:
   [`processed/train.csv`](processed/train.csv) (800 × 19),
   [`processed/test.csv`](processed/test.csv) (200 × 19),
   cùng `final_features.json` (18 tên cột).
5. Khi huấn luyện, đảo nhãn bằng `split_features_target()`; one-hot categorical,
   StandardScaler cho LR/SVM và passthrough numeric cho RF. Đặt preprocessor
   trong model pipeline để chỉ fit trên train/train-fold.

`get_feature_types()` và hai notebook chọn số bằng `include="number"`, chọn các
cột còn lại bằng `exclude="number"`. Cách này nhận cả `object`, `str`, `string`,
`category`, tránh phụ thuộc dtype chuỗi mặc định thay đổi từ pandas 3.0;
xem [hướng dẫn chuyển đổi dtype chuỗi của pandas](https://pandas.pydata.org/docs/user_guide/migration-3-strings.html).
[`requirements.txt`](../requirements.txt) ghim `pandas==3.1.0`, đúng phiên bản
đã kiểm tra trong môi trường hiện tại. Dockerfile cài từ requirements này.
Các dependency khác chưa được khóa toàn bộ, nên số liệu có thể khác nhẹ khi đổi môi trường.

Chạy notebook 01 rồi 02 với working directory là thư mục gốc dự án để tái lập
phân tích và các CSV hiện có. **Notebook 02 dùng seed 42 cho split**, đúng các
CSV đang lưu. Lệnh `python src/preprocess.py` là đường xuất dữ liệu bằng script,
đang dùng `RANDOM_STATE = 0`: vẫn có kích thước/tỷ lệ nhãn như trên nhưng thành
phần dòng khác. Không trộn kết quả model từ hai cách chia; nếu sinh lại split thì
cần huấn luyện và đánh giá lại các model liên quan.

`raw/` giữ dữ liệu đầu vào; `processed/` chứa dữ liệu sau chọn đặc trưng và
chia tập. Kết luận preprocessing hiện tại: không cần làm đầy missing/xóa trùng,
giữ outlier, giữ tỷ lệ nhãn, dùng 18 đặc trưng và fit mọi phép biến đổi mô hình
trên train/train-fold.
