# Credit Risk Prediction — German Credit Data

Dự đoán khả năng một khoản vay bị vỡ nợ (default) dựa trên thông tin hồ sơ khách hàng.
Bài tập lớn môn Học máy (CO3117).

## Thành viên nhóm

| MSSV | Họ tên | Vai trò |
|---|---|---|
| ... | ... | Data & Feature Lead (TV1) |
| ... | ... | Model Lead A — LR & SVM (TV2) |
| ... | ... | Model Lead B — Random Forest (TV3) |
| ... | ... | Evaluation + Demo + Report Lead (TV4) |

## Cấu trúc thư mục

```
.
├── data/
│   ├── raw/              # dữ liệu gốc, không sửa
│   ├── processed/        # train.csv, test.csv sau khi clean + split
│   └── README.md         # nguồn dữ liệu, mô tả cột, mapping nhãn
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_selection.ipynb
│   └── 03_models.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── models/                # model đã huấn luyện (.pkl)
├── app/
│   └── streamlit_app.py   # demo Streamlit (Overview / Single Scoring / Batch Scoring)
├── reports/
│   ├── report.pdf
│   └── slides.pdf
├── requirements.txt
├── Dockerfile
└── .gitignore
```

## Môi trường

- Python 3.10+
- Cài thư viện: `pip install -r requirements.txt`

## Cách chạy

1. Tải dataset German Credit Data, đặt vào `data/raw/`.
2. Chạy pipeline tiền xử lý: `python src/preprocess.py`
3. Huấn luyện mô hình: `python src/train.py`
4. Đánh giá mô hình: `python src/evaluate.py`
5. Chạy demo Streamlit: `streamlit run app/streamlit_app.py`

## Chạy bằng Docker

```
docker build -t credit-risk-demo .
docker run -p 8501:8501 credit-risk-demo
```

Sau đó mở trình duyệt tại `http://localhost:8501`.

## Ghi chú

- Không dùng Accuracy làm metric chính do dữ liệu mất cân bằng lớp.
- Cost matrix: phân loại sai một khách "bad" thành "good" tốn kém gấp 5 lần so với ngược lại.
- Xem chi tiết kế hoạch và phân công trong `reports/`.
