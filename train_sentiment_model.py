import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
import nltk
from nltk.corpus import stopwords
import re

FILE_NAME = 'IMDB Dataset.csv'
try:
    df = pd.read_csv(FILE_NAME)
    df = df.head(10000) 
    print(f"Đã tải {len(df)} mẫu từ file '{FILE_NAME}'.")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file '{FILE_NAME}'. Vui lòng đảm bảo file nằm cùng thư mục.")
    exit()

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text): 
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text.lower()) 
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

df['cleaned_review'] = df['review'].apply(preprocess_text)

df['sentiment_encoded'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)


X = df['cleaned_review']
y = df['sentiment_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000, solver='liblinear')), 
])

print("\nBắt đầu đào tạo mô hình...")
pipeline.fit(X_train, y_train)
print("Đào tạo hoàn tất.")

# Đánh giá (chỉ để tham khảo)
accuracy = pipeline.score(X_test, y_test)
print(f"\nĐộ chính xác mô hình trên tập Test: {accuracy:.2f}")

# --- Lưu mô hình ---
MODEL_PATH = 'sentiment_model.pkl'
with open(MODEL_PATH, 'wb') as file:
    pickle.dump(pipeline, file)

print(f"\nMô hình đã được lưu vào file '{MODEL_PATH}'.")