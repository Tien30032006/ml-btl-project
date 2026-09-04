# Hướng dẫn làm việc nhóm

## Quy trình Git

1. Clone repo:
   ```bash
   git clone <repo-url>
   cd ml-btl-project
   ```
2. Tạo nhánh riêng cho công việc của mình, không push trực tiếp lên `main`:
   ```bash
   git checkout -b feature/ten-cong-viec
   ```
3. Commit rõ ràng, dễ hiểu:
   ```bash
   git add .
   git commit -m "Thêm hàm tiền xử lý dữ liệu missing value"
   ```
4. Push nhánh và tạo Pull Request để các thành viên khác review trước khi merge vào `main`.
   ```bash
   git push origin feature/ten-cong-viec
   ```

## Quy ước đặt tên nhánh

- `feature/...` — thêm chức năng mới
- `fix/...` — sửa lỗi
- `docs/...` — cập nhật tài liệu/báo cáo

## Quy ước code

- Đặt tên biến/hàm rõ nghĩa, tiếng Anh.
- Mỗi hàm/module nên có docstring ngắn mô tả input/output.
- Không commit dữ liệu lớn hoặc model lớn trực tiếp — dùng link chia sẻ (xem `data/README.md`).

## Phân công

Cập nhật bảng phân công công việc trong `README.md` (mục "Thành viên nhóm") và cập nhật thường xuyên tiến độ.
