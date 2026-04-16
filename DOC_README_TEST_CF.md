# Tài liệu đọc hiểu thuật toán trong `test_CF.py`

Tài liệu này giúp bạn đọc nhanh và hiểu đúng luồng thuật toán Collaborative Filtering đang có trong project.

---

## 1) File này đang làm gì?

`test_CF.py` cài đặt **User-based Collaborative Filtering** (gợi ý theo người dùng tương tự), không phải item-based.

Ý tưởng chính:
- Mỗi user có một vector rating các item.
- Chuẩn hóa rating theo từng user (trừ trung bình của user đó) để giảm bias “người dễ/chặt tay”.
- Tính độ tương đồng giữa các user bằng cosine similarity.
- Dự đoán điểm user `u` cho item `i` bằng trung bình có trọng số từ `k` user giống `u` nhất (trong nhóm đã rating item `i`).

---

## 2) Cấu trúc dữ liệu đầu vào

Biến đầu vào chính: `Y_data` (numpy array), mỗi dòng có dạng:

```text
[user_id, item_id, rating]
```

Ví dụ:

```text
[0, 2, 4]
```

nghĩa là user `0` đã chấm item `2` điểm `4`.

---

## 3) Giải thích từng hàm trong class `CF`

## `__init__(self, Y_data, k, dist_func=cosine_similarity)`

Mục tiêu: khởi tạo mô hình.

- `self.Y_data`: dữ liệu rating thô.
- `self.k`: số hàng xóm gần nhất dùng khi dự đoán.
- `self.dist_func`: hàm đo similarity (mặc định cosine).
- `self.n_users`: số user, tính bằng `max(user_id) + 1`.
- `self.n_items`: số item, tính bằng `max(item_id) + 1`.

Lưu ý: cách đếm này giả sử ID bắt đầu từ 0.

---

## `add(self, new_data)`

Mục tiêu: thêm rating mới vào dữ liệu cũ.

- Dùng `np.concatenate` để nối `new_data` vào `self.Y_data`.
- Comment trong code nói rõ giả sử không có user/item hoàn toàn mới (chỉ thêm rating cho tập user/item sẵn có).

---

## `normalize_Y(self)`

Đây là bước quan trọng nhất trước khi tính similarity.

### Việc hàm làm:
1. Tạo bản copy `self.Ybar_data` từ `self.Y_data`.
2. Với mỗi user `n`:
   - Lấy tất cả rating của user đó.
   - Tính trung bình `mu[n]`.
   - Trừ trung bình: `rating_normalized = rating - mu[n]`.
3. Tạo ma trận sparse `self.Ybar` kích thước `(n_items, n_users)`:
   - Hàng là item
   - Cột là user
   - Giá trị là rating đã chuẩn hóa

### Vì sao dùng sparse matrix?
Vì ma trận rating thường rất thưa (đa số ô không có rating), sparse giúp tiết kiệm bộ nhớ và tăng hiệu năng.

---

## `similarity(self)`

Mục tiêu: tạo ma trận tương đồng user-user.

Code:

```python
self.S = self.dist_func(self.Ybar.T, self.Ybar.T)
```

Giải thích:
- `self.Ybar` là `(items x users)` nên `self.Ybar.T` là `(users x items)`.
- Tính cosine giữa các hàng của `self.Ybar.T` → ra ma trận `S` kích thước `(users x users)`.
- `S[u, v]` = mức độ giống nhau giữa user `u` và user `v`.

---

## `fit(self)`

Mục tiêu: huấn luyện mô hình theo pipeline chuẩn.

Gồm 2 bước:
1. `normalize_Y()`
2. `similarity()`

---

## `pred(self, u, i, normalized=1)`

Mục tiêu: dự đoán điểm của user `u` cho item `i`.

### Luồng trong code:
1. Tìm các user đã rating item `i`.
2. Lấy similarity giữa `u` và các user đó.
3. Chọn `k` user có similarity cao nhất.
4. Lấy rating chuẩn hóa của nhóm hàng xóm cho item `i`.
5. Tính weighted average:

\[
\hat{r}_{u,i}^{(norm)} = \frac{\sum\limits_{v \in N_k(u,i)} s(u,v)\,r_{v,i}^{(norm)}}{\sum\limits_{v \in N_k(u,i)} |s(u,v)|}
\]

6. Nếu `normalized=0`, cộng lại `mu[u]` để về thang điểm gốc.

### Ý nghĩa tham số `normalized`
- `1`: trả về điểm đã chuẩn hóa.
- `0`: trả về điểm dự đoán ở thang điểm gốc.

---

## `recommend(self, u)`

Mục tiêu: đưa ra danh sách item đề xuất cho user `u`.

Luồng:
1. Lấy danh sách item mà `u` đã rating.
2. Duyệt toàn bộ item trong hệ thống.
3. Với item chưa rating, gọi `pred(u, i)`.
4. Nếu điểm dự đoán `> 0` thì thêm vào danh sách gợi ý.

Lưu ý: ngưỡng `> 0` đang dùng trên **điểm chuẩn hóa** (vì gọi `pred` mặc định `normalized=1`).

---

## `print_recommendation(self)`

Mục tiêu: in danh sách item gợi ý cho từng user.

---

## 4) Luồng chạy ở cuối file (script section)

Phần cuối `test_CF.py` chạy trực tiếp theo thứ tự:

1. Đọc file `ex.dat` bằng pandas.
2. Chuyển sang numpy array `Y_data`.
3. Khởi tạo `CF(Y_data, k=2)`.
4. `fit()`.
5. `print_recommendation()`.

Điểm cần nhớ: hiện chưa có `if __name__ == "__main__":`, nên import file này từ chỗ khác sẽ vẫn chạy block này.

---

## 5) Tóm tắt tư duy thuật toán (ngắn gọn)

Mô hình trả lời câu hỏi:
> “User A sẽ thích item X không?”

Bằng cách:
1. Tìm các user có hành vi giống A.
2. Nhìn xem nhóm user giống A đánh giá X ra sao.
3. Lấy trung bình có trọng số theo độ giống nhau.

---

## 6) Một số giới hạn hiện tại (để bạn đọc code không bị vướng)

- Chưa xử lý rõ trường hợp mẫu số bằng 0 trong `pred` (khi tổng `|similarity|` rất nhỏ/0).
- Chưa tách phần demo chạy file ra khỏi class (thiếu `if __name__ == "__main__":`).
- Hàm `recommend` đang dùng ngưỡng khá đơn giản (`rating > 0`).

Các điểm này không làm bạn khó hiểu thuật toán cốt lõi, nhưng quan trọng khi muốn nâng cấp chất lượng hệ thống.

---

## 7) Cách đọc file nhanh nhất (đề xuất)

Nếu bạn muốn nắm nhanh trong 10-15 phút:
1. Đọc `__init__` để hiểu input/output chính.
2. Đọc kỹ `normalize_Y` (đây là linh hồn của CF theo user).
3. Đọc `similarity` để xác nhận đang là user-user.
4. Đọc `pred` để hiểu công thức dự đoán.
5. Đọc `recommend` để thấy rule ra quyết định cuối cùng.

Sau đó chạy `python test_CF.py` và đối chiếu output với luồng trên là bạn sẽ hiểu rất chắc.
