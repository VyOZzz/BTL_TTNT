import numpy as np
import pandas as pd
import streamlit as st

from test_CF import CF, load_ratings


def parse_uploaded_file(uploaded_file):
    """Parse uploaded whitespace-separated ratings file to numpy array."""
    r_cols = ['user_id', 'item_id', 'rating']
    ratings = pd.read_csv(uploaded_file, sep=r'\s+', names=r_cols, engine='python')
    ratings['user_id'] = ratings['user_id'].astype(np.int32)
    ratings['item_id'] = ratings['item_id'].astype(np.int32)
    ratings['rating'] = ratings['rating'].astype(np.float64)
    return ratings.to_numpy(), ratings


def build_model(y_data, k):
    model = CF(y_data, k=k)
    model.fit()
    return model


def recommend_for_user(model, user_id, mode='threshold', top_n=5, threshold=0.0, normalized=True):
    """Recommend items for one user using score-threshold or Top-N ranking."""
    ids = np.where(model.Y_data[:, 0] == user_id)[0]
    rated_items = set(model.Y_data[ids, 1].astype(np.int32).tolist())
    candidates = [i for i in range(model.n_items) if i not in rated_items]

    scored_items = []
    for item_id in candidates:
        score = float(model.pred(user_id, item_id, normalized=1 if normalized else 0))
        scored_items.append((item_id, score))

    scored_items.sort(key=lambda x: x[1], reverse=True)
    if mode == 'top_n':
        return [item for item, _ in scored_items[:top_n]], scored_items
    return [item for item, score in scored_items if score > threshold], scored_items


st.set_page_config(page_title='CF Demo', layout='wide')
st.title('Collaborative Filtering Demo')
st.caption('Giao dien demo thuat toan goi y trong `test_CF.py` (de nhin, de hieu).')

with st.sidebar:
    st.header('Cau hinh mo hinh')
    source = st.radio('Nguon du lieu', ['ex.dat', 'ex2.dat', 'Upload file'])
    k = st.slider('So lang gieng gan nhat (k)', min_value=1, max_value=10, value=2)

    uploaded_file = None
    if source == 'Upload file':
        uploaded_file = st.file_uploader('Chon file du lieu (.dat/.txt/.csv)', type=['dat', 'txt', 'csv'])

    st.divider()
    st.header('Cau hinh goi y')
    recommendation_mode = st.radio(
        'Che do goi y',
        ['Theo nguong diem', 'Top-N'],
        help='Theo nguong: giu item co diem > nguong. Top-N: lay N item diem cao nhat.',
    )
    normalized = st.checkbox(
        'Dung diem chuan hoa',
        value=True,
        help='Bat: diem da tru trung binh user (so sanh so thich tuong doi). Tat: cong lai trung binh user.',
    )
    threshold = st.number_input('Nguong diem', value=0.0, step=0.1, format='%.2f')
    top_n = st.slider('So luong goi y Top-N', min_value=1, max_value=20, value=5)

if source == 'Upload file':
    if uploaded_file is None:
        st.info('Vui long upload file de tiep tuc.')
        st.stop()
    y_data, ratings_df = parse_uploaded_file(uploaded_file)
else:
    y_data = load_ratings(source)
    ratings_df = pd.DataFrame(y_data, columns=['user_id', 'item_id', 'rating'])

if y_data.size == 0:
    st.error('Du lieu rong, khong the huan luyen mo hinh.')
    st.stop()

st.subheader('Du lieu dau vao')
st.dataframe(ratings_df, use_container_width=True)

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.markdown('**Phan bo diem rating**')
    rating_dist = ratings_df['rating'].value_counts().sort_index()
    st.bar_chart(rating_dist)

with chart_col2:
    st.markdown('**So luot danh gia theo user**')
    ratings_per_user = ratings_df.groupby('user_id').size().sort_index()
    st.bar_chart(ratings_per_user)

if st.button('Huan luyen mo hinh'):
    try:
        st.session_state['model'] = build_model(y_data, k)
        st.session_state['k'] = k
        st.success(f'Huan luyen thanh cong voi k = {k}.')
    except Exception as exc:
        st.error(f'Loi khi huan luyen: {exc}')

model = st.session_state.get('model')
if model is None:
    st.warning('Nhan "Huan luyen mo hinh" de bat dau.')
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader('Goi y item cho user')
    user_for_recommend = st.selectbox('Chon user', list(range(model.n_users)), key='recommend_user')
    if st.button('Xem goi y'):
        mode = 'top_n' if recommendation_mode == 'Top-N' else 'threshold'
        items, scored_items = recommend_for_user(
            model,
            user_for_recommend,
            mode=mode,
            top_n=top_n,
            threshold=threshold,
            normalized=normalized,
        )
        st.write(f'User `{user_for_recommend}` nen xem: `{items}`')
        st.caption('Top diem du doan tren cac item chua danh gia:')
        preview = pd.DataFrame(scored_items, columns=['item_id', 'pred_score']).head(10)
        st.dataframe(preview, use_container_width=True)

with col2:
    st.subheader('Du doan diem user-item')
    user_for_pred = st.selectbox('User du doan', list(range(model.n_users)), key='pred_user')
    item_for_pred = st.selectbox('Item du doan', list(range(model.n_items)), key='pred_item')

    if st.button('Du doan diem'):
        score = model.pred(user_for_pred, item_for_pred, normalized=1 if normalized else 0)
        label = 'diem chuan hoa' if normalized else 'diem goc (co cong mean user)'
        st.metric('Ket qua', f'{float(score):.4f}')
        st.caption(f'Loai diem: {label}')

st.subheader('Tong hop goi y cho tat ca user')
all_recs = []
for u in range(model.n_users):
    mode = 'top_n' if recommendation_mode == 'Top-N' else 'threshold'
    rec_items, _ = recommend_for_user(
        model,
        u,
        mode=mode,
        top_n=top_n,
        threshold=threshold,
        normalized=normalized,
    )
    all_recs.append({'user_id': u, 'recommended_items': rec_items, 'so_luong_goi_y': len(rec_items)})

all_recs_df = pd.DataFrame(all_recs)
st.dataframe(all_recs_df, use_container_width=True)

st.markdown('**So luong item duoc goi y theo user**')
rec_count = all_recs_df.set_index('user_id')['so_luong_goi_y']
st.bar_chart(rec_count)



