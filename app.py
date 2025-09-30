from flask import Flask, request, jsonify, render_template
import pickle
import nltk
from nltk.corpus import stopwords
import re
import os

app = Flask(__name__)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """Tiền xử lý văn bản, loại bỏ HTML, chuyển về chữ thường và loại bỏ stop words."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text.lower()) 
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

MODEL_PATH = 'sentiment_model.pkl'
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    print(f"Đã tải thành công mô hình AI từ '{MODEL_PATH}'.")
except FileNotFoundError:
    print(f"LỖI: KHÔNG TÌM THẤY file mô hình '{MODEL_PATH}'. Vui lòng chạy code đào tạo mô hình trước.")
    model = None

@app.route('/')
def home():
    """Route mặc định: Trả về giao diện người dùng (index.html)."""
    return render_template('index.html')

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    """API endpoint nhận dữ liệu POST và trả về kết quả dự đoán."""
    
    if model is None:
        return jsonify({'error': 'Mô hình AI chưa sẵn sàng. Vui lòng kiểm tra file sentiment_model.pkl.'}), 500

    data = request.get_json()
    input_text = data.get('text', '')

    if not input_text:
        return jsonify({'result': 'Vui lòng nhập văn bản để phân tích.'})

    processed_text = preprocess_text(input_text)
    
    prediction = model.predict([processed_text])[0]
    
    sentiment_label = 'Tích cực (Positive) 😄' if prediction == 1 else 'Tiêu cực (Negative) 😞'

    return jsonify({
        'status': 'success',
        'input_text': input_text,
        'predicted_sentiment': sentiment_label
    })

if __name__ == '__main__':
    app.run(debug=True)