# [Tên bài toán] — Bài tập lớn Học máy (CO3117)

> Nhóm: MSSV1 - MSSV2 - MSSV3 - ...
> Giảng viên hướng dẫn: ...
> Học kỳ: ...

## 1. Mô tả bài toán

Mô tả ngắn gọn bài toán, tại sao chọn dataset này, và thách thức chính của dữ liệu.

## 2. Cấu trúc thư mục

```
.
├── data/
│   ├── raw/            # Dữ liệu gốc (hoặc link tải nếu file lớn, xem data/README.md)
│   └── processed/      # Dữ liệu sau tiền xử lý
├── notebooks/           # Notebook EDA, thử nghiệm nhanh
├── src/                  # Source code chính
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── models/               # Model đã huấn luyện (hoặc link tải nếu lớn)
├── reports/              # Báo cáo tổng kết (report.pdf/docx) + hình ảnh
├── slides/               # Slide thuyết trình
├── configs/              # File cấu hình (config.yaml)
├── tests/                # Unit test (nếu có)
├── Dockerfile
├── requirements.txt
└── README.md
```

## 3. Môi trường & cài đặt

### Cách 1: dùng virtualenv + pip
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Cách 2: dùng Docker
```bash
docker build -t ml-btl .
docker run --rm -it -v $(pwd):/app ml-btl
```

## 4. Cách chạy code

```bash
# Tiền xử lý dữ liệu
python src/data_preprocessing.py --input data/raw --output data/processed

# Huấn luyện mô hình
python src/train.py --config configs/config.yaml --model random_forest

# Đánh giá mô hình
python src/evaluate.py --model models/random_forest.pkl --data data/processed/test.csv
```

> Cập nhật lại phần này cho đúng với script thực tế của nhóm.

## 5. Dữ liệu

Xem chi tiết trong [`data/README.md`](data/README.md). Nếu dataset lớn, để link Google Drive/Kaggle tại đây thay vì commit trực tiếp.

## 6. Kết quả tóm tắt

| Mô hình | Metric 1 | Metric 2 | Ghi chú |
|---|---|---|---|
| Model A | | | |
| Model B | | | |

Chi tiết đầy đủ xem trong `reports/`.

## 7. Thành viên nhóm

| MSSV | Họ tên | Công việc phụ trách |
|---|---|---|
| | | |
| | | |
| | | |

## 8. Ghi chú

- Được phép dùng thư viện open-source, nhưng không dùng API thương mại thay thế toàn bộ pipeline.
- AI chỉ được dùng để tham khảo/giải thích/debug, không copy nguyên code/báo cáo.
