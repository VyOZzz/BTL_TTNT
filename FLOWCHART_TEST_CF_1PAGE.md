# Flowchart 1 trang - Thuật toán trong `test_CF.py`

Mục tiêu: đọc trong 2-3 phút để nắm luồng trước khi báo cáo.

---

## 1) Luồng tổng thể (từ dữ liệu đến gợi ý)

```text
ex.dat
  |
  v
Đọc dữ liệu -> Y_data [user_id, item_id, rating]
  |
  v
Khởi tạo model CF(Y_data, k=2)
  |
  v
fit()
  |
  +--> normalize_Y()
  |      - Tính mu[user] = rating trung bình/user
  |      - Chuẩn hóa: rating' = rating - mu[user]
  |      - Tạo ma trận sparse Ybar (items x users)
  |
  +--> similarity()
         - Tính S = cosine_similarity(Ybar.T, Ybar.T)
         - S là ma trận tương đồng user-user
  |
  v
print_recommendation()
  |
  +--> với từng user u: recommend(u)
            |
            +--> duyệt từng item i chưa rating
                    |
                    +--> pred(u, i)
                          - lấy user đã rating i
                          - chọn k user giống u nhất
                          - weighted average theo similarity
                    |
                    +--> nếu điểm dự đoán > 0 -> thêm vào gợi ý
```

---

## 2) Flowchart chi tiết hàm `pred(u, i)`

```text
Bắt đầu pred(u, i)
  |
  v
Tìm ids các dòng có item_id == i
  |
  v
Lấy users_rated_i
  |
  v
Tính sim = S[u, users_rated_i]
  |
  v
Chọn top-k similarity lớn nhất
  |
  v
Lấy rating chuẩn hóa r của top-k user cho item i
  |
  v
Tính:
score_norm = sum(sim * r) / sum(abs(sim))
  |
  +--> normalized=1 ? ---- Có ----> trả score_norm
  |                         |
  |                         Không
  v
trả score_norm + mu[u]
```

---

## 3) Công thức cần nói khi báo cáo

\[
\hat{r}_{u,i}^{(norm)} = \frac{\sum\limits_{v \in N_k(u,i)} s(u,v)\,r_{v,i}^{(norm)}}{\sum\limits_{v \in N_k(u,i)} |s(u,v)|}
\]

- `N_k(u,i)`: tập `k` user giống `u` nhất trong nhóm đã đánh giá item `i`.
- `s(u,v)`: độ tương đồng giữa user `u` và `v`.
- `r_{v,i}^{(norm)}`: rating đã chuẩn hóa của `v` cho `i`.

Nếu muốn điểm thang gốc:
\[
\hat{r}_{u,i} = \hat{r}_{u,i}^{(norm)} + \mu_u
\]

---

## 4) 3 câu chốt để trình bày ngắn gọn

1. Đây là **User-based CF**: tìm người dùng giống nhau trước, rồi suy ra item nên gợi ý.
2. Trước khi đo giống nhau, hệ thống **chuẩn hóa rating theo từng user** để giảm thiên lệch cá nhân.
3. Dự đoán điểm bằng **trung bình có trọng số theo cosine similarity** của top-k hàng xóm.

---

## 5) Ghi chú nhanh khi bị hỏi phản biện

- Vì sao cần sparse matrix? -> Ma trận rating rất thưa, sparse tiết kiệm RAM và tăng tốc.
- Vì sao ngưỡng gợi ý là `> 0`? -> Vì đang dùng điểm chuẩn hóa; dương nghĩa là cao hơn mức trung bình của user.
- Khác item-based ở đâu? -> User-based tính giống giữa user-user; item-based sẽ tính item-item.
