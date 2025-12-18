import streamlit as st
import joblib
import re
import nltk
import pandas as pd
import os
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  # Thêm dòng này phòng hờ logic chia tách

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="Movie Sentiment AI", page_icon="🎬")

# --- 2. TẢI DỮ LIỆU NLTK & CẤU HÌNH STOPWORDS MỚI ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

# --- CẬP NHẬT QUAN TRỌNG: Logic Stopwords khớp với file Train ---
custom_stop_words = set(stopwords.words("english"))
negation_words = {"not", "no", "never", "nor", "n't"}
stop_words = {
                 lemmatizer.lemmatize(w)
                 for w in custom_stop_words
             } - negation_words  # Giữ lại các từ phủ định


# --- 3. CÁC HÀM XỬ LÝ (CORE FUNCTIONS) ---

# --- CẬP NHẬT QUAN TRỌNG: Hàm Clean Text khớp với file Train ---
def clean_text(text):
    text = text.lower()  # Lower trước
    text = BeautifulSoup(text, "html.parser").get_text()

    # Giữ lại dấu nháy đơn (') cho các từ như don't, can't
    text = re.sub(r"[^a-z']", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if lemmatizer.lemmatize(w) not in stop_words
    ]
    return " ".join(words)


def translate_to_english(text):
    try:
        translator = GoogleTranslator(source='auto', target='en')
        return translator.translate(text)
    except Exception:
        return text


# Hàm lưu Feedback
def save_feedback(text, correct_label):
    file_path = "feedback.csv"
    new_data = pd.DataFrame({'review': [text], 'sentiment': [correct_label]})

    if not os.path.exists(file_path):
        new_data.to_csv(file_path, index=False, mode='w')
    else:
        new_data.to_csv(file_path, index=False, mode='a', header=False)


# --- CẬP NHẬT QUAN TRỌNG: Hàm Retrain dùng thuật toán tối ưu ---
def retrain_model():
    status_text = st.empty()
    status_text.info("⏳ Đang tải dữ liệu gốc và feedback mới...")

    # 1. Đọc dữ liệu gốc
    try:
        df_orig = pd.read_csv("imdb.csv")
    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy file imdb.csv gốc!")
        return

    # 2. Đọc dữ liệu feedback (nếu có)
    if os.path.exists("feedback.csv"):
        try:
            df_feed = pd.read_csv("feedback.csv")
            df_final = pd.concat([df_orig, df_feed], ignore_index=True)
            status_text.info(f"Đã tìm thấy {len(df_feed)} mẫu feedback mới. Đang gộp dữ liệu...")
        except pd.errors.EmptyDataError:
            df_final = df_orig
            status_text.warning("File feedback.csv bị lỗi hoặc rỗng. Chỉ dùng dữ liệu gốc.")
    else:
        df_final = df_orig
        status_text.info("Chưa có feedback mới. Chỉ train lại trên dữ liệu gốc.")

    # 3. Xử lý dữ liệu
    status_text.info("⏳ Đang xử lý văn bản (Clean Text - Logic Mới)...")
    df_final["review_clean"] = df_final["review"].apply(clean_text)

    le_new = LabelEncoder()
    y = le_new.fit_transform(df_final['sentiment'])

    status_text.info("⏳ Đang Vector hóa (TF-IDF N-gram)...")
    # Cấu hình TF-IDF chuẩn optimized
    tfidf_new = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )
    X_tfidf = tfidf_new.fit_transform(df_final["review_clean"])

    status_text.info("⏳ Đang Train Model (Logistic Regression Tuned)...")
    # Cấu hình Model chuẩn optimized
    model_new = LogisticRegression(
        C=2.0,
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced"
    )
    model_new.fit(X_tfidf, y)

    # 4. Lưu đè file cũ
    joblib.dump(model_new, 'sentiment_model.pkl')
    joblib.dump(tfidf_new, 'tfidf_vectorizer.pkl')
    joblib.dump(le_new, 'label_encoder.pkl')

    status_text.success("✅ Đã cập nhật Model thành công! Hãy tải lại trang (F5) để áp dụng.")
    st.cache_resource.clear()


# --- 4. LOAD MODEL ---
@st.cache_resource
def load_models():
    try:
        m = joblib.load('sentiment_model.pkl')
        v = joblib.load('tfidf_vectorizer.pkl')
        l = joblib.load('label_encoder.pkl')
        return m, v, l
    except FileNotFoundError:
        return None, None, None


model, tfidf, le = load_models()

# --- 5. GIAO DIỆN (UI) ---
st.title("🎬 Mô Hình AI Phân Tích Cảm Xúc Dựa Trên Đánh Giá")

# Sidebar
with st.sidebar:
    st.header("⚙️ Khu vực Admin")
    st.write("Cập nhật kiến thức mới cho AI từ phản hồi.")
    if st.button("🚀 Train lại Model ngay"):
        retrain_model()

    if os.path.exists("feedback.csv"):
        try:
            count = len(pd.read_csv("feedback.csv"))
            st.write(f"Đang có **{count}** mẫu feedback chờ học.")
        except:
            st.write("File feedback trống.")
    else:
        st.write("Chưa có feedback nào.")

# Main UI
user_input = st.text_area("Nhập bình luận phim (Việt/Anh):", height=100)
analyze_btn = st.button("🔍 Phân Tích")

# Session State
if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None
if 'translated_text' not in st.session_state:
    st.session_state['translated_text'] = None
if 'show_fix_form' not in st.session_state:
    st.session_state['show_fix_form'] = False

if analyze_btn and user_input:
    if model is None:
        st.error("Chưa tìm thấy model! Hãy chạy file train_final.py trước hoặc bấm nút Train bên trái.")
    else:
        with st.spinner('Đang suy nghĩ...'):
            eng_text = translate_to_english(user_input)

            # QUAN TRỌNG: Dùng hàm clean_text mới
            clean = clean_text(eng_text)

            vec = tfidf.transform([clean])
            pred_idx = model.predict(vec)[0]
            pred_label = le.inverse_transform([pred_idx])[0]
            proba = model.predict_proba(vec).max() * 100

            st.session_state['prediction_result'] = {'label': pred_label, 'proba': proba}
            st.session_state['translated_text'] = eng_text
            st.session_state['show_fix_form'] = False

# Hiển thị kết quả
if st.session_state['prediction_result']:
    res = st.session_state['prediction_result']
    eng_txt = st.session_state['translated_text']

    st.divider()
    if res['label'] == 'positive':  # Giả sử label encoded là 'positive'
        # Kiểm tra lại label gốc trong dataset của bạn (0/1 hay pos/neg)
        # Nếu dùng code cũ của bạn thì output là 'positive'/'negative' chuỗi
        st.success(f"Kết quả: **TÍCH CỰC (KHEN)** (Độ tin cậy: {res['proba']:.1f}%)")
    else:
        st.error(f"Kết quả: **TIÊU CỰC (CHÊ)** (Độ tin cậy: {res['proba']:.1f}%)")

    if user_input != eng_txt:
        st.caption(f"Dịch sang Anh: {eng_txt}")
        st.caption(
            f"Cleaned Text (Debug): {clean_text(eng_txt)}")  # Dòng này để bạn debug xem nó có giữ lại chữ 'not' không

    st.write("---")
    st.write("**AI dự đoán có đúng không?**")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("👍 Đúng rồi"):
            st.toast("Cảm ơn bạn đã xác nhận!")
    with c2:
        if st.button("👎 Sai rồi (Sửa lại)"):
            st.session_state['show_fix_form'] = True

    if st.session_state['show_fix_form']:
        with st.form("fix_form"):
            st.write("Hãy dạy lại AI: Theo bạn, câu này thực ra là gì?")
            correct_val = st.radio("Nhãn đúng là:", ["positive", "negative"])
            submit_fix = st.form_submit_button("Gửi Feedback")

            if submit_fix:
                save_feedback(eng_txt, correct_val)
                st.success("✅ Đã lưu phản hồi! Hãy bấm 'Train lại Model' bên menu trái để AI học ngay.")
                st.session_state['show_fix_form'] = False
