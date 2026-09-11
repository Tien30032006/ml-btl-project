# Data

## Nguồn dữ liệu

- Dataset: German Credit Data (Statlog)
- UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
- Bản CSV tiện dụng (Kaggle): https://www.kaggle.com/datasets/jumpingdino/german-credit-dataset
- Ngày tải: ...
- Version: ...

## Quy mô

- 1.000 mẫu
- 20 đặc trưng gốc (7 numeric, 13 categorical)
- 1 nhãn nhị phân (default / không default)

## Cost matrix

Phân loại sai một khách "bad" thành "good" tốn kém gấp 5 lần so với ngược lại
(FN đắt gấp 5 lần FP). Dùng để thiết kế cost-sensitive threshold ở bước đánh giá.

## Mô tả cột

| Cột | Ý nghĩa | Kiểu |
|---|---|---|
| ... | ... | ... |

## Mapping nhãn

| Giá trị gốc | Ý nghĩa | Nhãn dùng trong project |
|---|---|---|
| 1 | Good (không default) | 0 |
| 2 | Bad (default) | 1 |

## Quy ước thư mục

- `raw/` — dữ liệu gốc, **không chỉnh sửa**.
- `processed/` — `train.csv`, `test.csv` sau khi làm sạch và chia tách (stratified 80/20).
