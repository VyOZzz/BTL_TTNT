# BTL_TTNT - Collaborative Filtering Demo

## Cai dat

```bash
pip install -r requirements.txt
```

## Chay ban console

```bash
python test_CF.py
```

## Chay giao dien Streamlit

```bash
streamlit run streamlit_app.py
```

## Tinh nang giao dien

- Chon nguon du lieu: `ex.dat`, `ex2.dat` hoac upload file moi.
- Chon `k` (so lang gieng gan nhat) va huan luyen mo hinh ngay tren giao dien.
- Goi y theo 2 che do:
  - `Theo nguong diem`: giu item co diem du doan > nguong.
  - `Top-N`: lay N item co diem du doan cao nhat.
- Du doan diem cho cap `user-item` voi tuy chon diem chuan hoa/diem goc.
- Co bieu do truc quan:
  - Phan bo diem rating trong du lieu.
  - So luot danh gia theo user.
  - So luong item duoc goi y theo user.

## Dinh dang du lieu

Moi dong gom 3 cot cach nhau boi khoang trang:

```text
user_id item_id rating
```

Vi du:

```text
0 0 5.
0 1 4.
1 0 5.
```


