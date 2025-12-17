import pandas as pd
import re
import nltk
import joblib
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. CẤU HÌNH & TẢI NLTK
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()

# Tùy chỉnh danh sách stopwords: Giữ lại từ phủ định (not, no, never...)
custom_stop_words = set(stopwords.words("english"))
negation_words = {"not", "no", "never", "nor", "n't"}
stop_words = {
    lemmatizer.lemmatize(w)
    for w in custom_stop_words
} - negation_words

# 2. HÀM LÀM SẠCH DỮ LIỆU
def clean_text(text):
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()

    # Giữ lại dấu nháy đơn cho các từ như don't, can't
    text = re.sub(r"[^a-z']", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if lemmatizer.lemmatize(w) not in stop_words
    ]

    return " ".join(words)

# 3. TẢI DỮ LIỆU VÀ XỬ LÝ
print("Đang tải và xử lý dữ liệu...")
df = pd.read_csv("imdb.csv")
df["review_clean"] = df["review"].apply(clean_text)

# Mã hóa nhãn (Sentiment)
le = LabelEncoder()
df["sentiment"] = le.fit_transform(df["sentiment"])

X = df["review_clean"]
y = df["sentiment"]

# Chia tập Train/Test (Stratify để cân bằng nhãn)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. TRÍCH XUẤT ĐẶC TRƯNG (TF-IDF OPTIMIZED)
print("Đang vector hóa dữ liệu...")
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),  # Dùng cả cụm từ (bigram)
    max_features=50000,
    min_df=3,
    max_df=0.9,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 5. HUẤN LUYỆN MODEL
print("Đang huấn luyện mô hình...")
model = LogisticRegression(
    C=2.0,
    solver="liblinear",
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

# 6. ĐÁNH GIÁ MODEL
print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. LƯU MODEL (QUAN TRỌNG)
print("\nĐang lưu model và vectorizer...")
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("✅ Đã lưu thành công: sentiment_model.pkl, tfidf_vectorizer.pkl, label_encoder.pkl")

# 8. TEST NHANH (Tùy chọn)
while True:
    text = input("\nNhập review để test (hoặc gõ 'exit' để thoát): ")
    if text.lower() == "exit":
        break

    cleaned = clean_text(text)
    pred = model.predict(tfidf.transform([cleaned]))
    label = le.inverse_transform(pred)

    print(f"Dự đoán: {label[0]}")