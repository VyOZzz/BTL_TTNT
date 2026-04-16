# Hướng dẫn cấu trúc dự án BTL_TTNT

Tài liệu này giải thích nhanh các file code chính, cấu trúc thư mục, và cách chạy dự án.

## 1) Tổng quan cấu trúc thư mục

Tại thư mục gốc `BTL_TTNT/` hiện có:

- `test_CF.py`: File code chính cho bài toán Collaborative Filtering (gợi ý theo người dùng).
- `test.py`: File test nhỏ (chạy phép tính `pow`) để kiểm tra Python chạy được.
- `CF.ipynb`: Notebook để thử nghiệm tương tác.
- `ex.dat`: Dữ liệu rating mẫu đang được `test_CF.py` sử dụng trực tiếp.
- `ex2.dat`: Dữ liệu mẫu bổ sung.
- `ml-100k/`: Bộ dữ liệu MovieLens 100k (tham khảo/mở rộng).

## 2) Giải thích file code chính

## `test_CF.py`

File này định nghĩa lớp `CF` với luồng xử lý chính:

1. **Khởi tạo** (`__init__`):
   - Nhận ma trận dữ liệu `Y_data` dạng `[user_id, item_id, rating]`.
   - Thiết lập `k` (số hàng xóm gần nhất).
   - Mặc định dùng `cosine_similarity` để tính độ tương đồng user-user.

2. **Chuẩn hóa dữ liệu** (`normalize_Y`):
   - Tính trung bình rating của từng user (`mu`).
   - Trừ đi trung bình để tạo rating đã chuẩn hóa.
   - Chuyển dữ liệu sang ma trận thưa (`scipy.sparse`) để tối ưu bộ nhớ/tính toán.

3. **Tính tương đồng** (`similarity`):
   - Tạo ma trận tương đồng giữa các user dựa trên dữ liệu đã chuẩn hóa.

4. **Huấn luyện** (`fit`):
   - Gọi `normalize_Y()` + `similarity()`.

5. **Dự đoán rating** (`pred`):
   - Với user `u` và item `i`, lấy những user đã đánh giá `i`.
   - Chọn `k` user tương đồng nhất.
   - Tính rating dự đoán theo weighted average.

6. **Gợi ý item** (`recommend` + `print_recommendation`):
   - Duyệt các item user chưa đánh giá.
   - Nếu điểm dự đoán > 0 thì đưa vào danh sách gợi ý.

### Luồng chạy ở cuối file

Phần cuối `test_CF.py` đang chạy trực tiếp theo thứ tự:

- Đọc dữ liệu từ `ex.dat`
- Chuyển sang numpy array
- Tạo model `CF(..., k=2)`
- `fit()`
- In khuyến nghị

Lưu ý: file **chưa có** `if __name__ == "__main__":`, nên nếu import file này từ nơi khác thì đoạn chạy demo cuối file vẫn sẽ chạy.

## `test.py`

- Chỉ có 1 dòng: `print(pow(80,17,3149))`.
- Dùng để kiểm tra nhanh môi trường Python, không phải logic chính của project.

## `CF.ipynb`

- Notebook hỗ trợ chạy thử và trình bày kết quả theo dạng tương tác (phù hợp demo/phân tích).

## 3) Cách chạy dự án

Vì chưa có `requirements.txt`, cần cài thư viện theo import trong `test_CF.py`.

## Bước 1: Tạo môi trường ảo

### Windows PowerShell

```powershell
cd "C:\Project\ProjectPython\BTL_TTNT"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Bước 2: Cài dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install pandas numpy scipy scikit-learn jupyter
```

## Bước 3: Chạy chương trình chính

```powershell
python test_CF.py
```

Kết quả mong đợi: in ra danh sách recommendation cho từng user.

## Bước 4 (tuỳ chọn): Chạy file test nhỏ

```powershell
python test.py
```

## Bước 5 (tuỳ chọn): Mở notebook

```powershell
jupyter notebook CF.ipynb
```

## 4) Gợi ý cải thiện cấu trúc (tuỳ chọn)

- Tạo `requirements.txt` để cài đặt nhanh hơn.
- Bọc đoạn chạy demo cuối `test_CF.py` bằng `if __name__ == "__main__":` để tránh side effect khi import.
- Tách class `CF` ra module riêng nếu muốn mở rộng thành project lớn hơn.
