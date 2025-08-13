import streamlit as st
import joblib
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import re

# إعداد الصفحة
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎭",
    layout="centered"
)

# ===== تحسين المظهر باستخدام CSS =====
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    body {
        background-color: #0e1117; /* دارك مود */
    }

    .custom-title {
        font-size: 36px;
        font-weight: bold;
        color: #4da3ff;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .custom-subtitle {
        font-size: 14px;
        color: #aaaaaa;
    }

    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #555;
        padding: 10px;
        font-size: 16px;
        background-color: #1a1d23;
        color: white;
    }

    div.stButton > button:first-child {
        background-color: #4da3ff !important;
        color: white !important;
        border-radius: 8px !important;
        height: 3em !important;
        font-size: 16px !important;
        font-weight: bold !important;
        border: none !important;
        box-shadow: none !important;
    }

    div.stButton > button:first-child:hover {
        background-color: #1f6fb2 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ===== تحميل الموديل والأدوات =====
@st.cache_resource
def load_model_and_tools():
    try:
        model = load_model('sentiment_lstm_model.keras')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        with open('label_mappings.pkl', 'rb') as f:
            label_mappings = pickle.load(f)
        return model, tfidf_vectorizer, label_mappings
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# ===== تنظيف النص =====
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ===== التنبؤ =====
def predict_sentiment(text, model, tfidf_vectorizer, label_mappings):
    if not text.strip():
        return None, 0.0
    cleaned_text = clean_text(text)
    text_vectorized = tfidf_vectorizer.transform([cleaned_text])
    text_dense = text_vectorized.toarray()
    text_reshaped = text_dense.reshape(text_dense.shape[0], 1, text_dense.shape[1])
    prediction_probs = model.predict(text_reshaped, verbose=0)
    prediction_class = np.argmax(prediction_probs, axis=1)[0]
    confidence = float(np.max(prediction_probs))
    sentiment = label_mappings['label_to_sentiment'][prediction_class]
    return sentiment, confidence

# ===== الواجهة الرئيسية =====
def main():
    st.markdown("""
        <div class="custom-title">
            🎭 Sentiment Analysis App
            <span class="custom-subtitle">Analyze the sentiment of your text using AI</span>
        </div>
    """, unsafe_allow_html=True)
    
    model, tfidf_vectorizer, label_mappings = load_model_and_tools()
    if model is None:
        st.error("Failed to load model. Make sure all required files are present.")
        st.stop()
    
    text_input = st.text_area(
        "Enter your text:",
        height=120,
        placeholder="Type your text here... e.g., 'This movie is amazing!'"
    )
    
    analyze = st.button("🔍 Analyze Sentiment")
    
    if analyze and text_input.strip():
        with st.spinner('Analyzing...'):
            sentiment, confidence = predict_sentiment(
                text_input, model, tfidf_vectorizer, label_mappings
            )
            if sentiment:
                # ألوان باهتة تناسب الدارك مود
                if sentiment == "Positive":
                    bg_color = "#1B3D2F"  # أخضر باهت داكن
                elif sentiment == "Neutral":
                    bg_color = "#1B2F3D"  # أزرق باهت داكن
                else:
                    bg_color = "#3D1B1B"  # أحمر باهت داكن
                
                icon = "😊" if sentiment == 'Positive' else "😞" if sentiment == 'Negative' else "😐"
                
                st.markdown(f"""
                    <div style="
                        background-color:{bg_color};
                        padding: 20px;
                        border-radius: 12px;
                        box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
                        text-align: center;
                        margin-top: 20px;
                        color: white;
                    ">
                        <h2>{icon} {sentiment}</h2>
                        <p style='color:#ccc;'>Confidence: {confidence:.1%}</p>
                    </div>
                """, unsafe_allow_html=True)
    elif analyze and not text_input.strip():
        st.warning("Please enter some text to analyze!")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#888;">Made by <b>Salahaldin</b> using TensorFlow & Streamlit</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()