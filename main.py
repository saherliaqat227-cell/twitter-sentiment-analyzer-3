"""
Sentiment Analysis - Frontend (Streamlit) - Simple version
============================================================
Dropdown se model select karein (SVM / SimpleRNN / LSTM / GRU),
usi model ka prediction dikhega. Koi graphs nahi, seedha result.

HOW TO USE:
1. Put these files in the SAME folder as this app.py:
       - sentiment_svm_pipeline.pkl
       - label_encoder.pkl
       - sentiment_rnn_model.keras
       - LSTM_model.keras
       - GRU_model.keras
2. In PyCharm terminal run:
       streamlit run app.py
"""

import os
import re
import pickle

import numpy as np
import streamlit as st
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def model_path(filename):
    return os.path.join(BASE_DIR, filename)


# -----------------------------------------------------------------
# Page config
# -----------------------------------------------------------------
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")

st.markdown(
    """
    <style>
        .main-title { font-size: 2.2rem; font-weight: 700; text-align: center; margin-bottom: 0.1rem; }
        .sub-title { text-align: center; color: #888; margin-bottom: 1.5rem; }
        .sentiment-badge {
            display: inline-block; padding: 6px 16px; border-radius: 20px;
            font-weight: 600; font-size: 1rem; color: white;
        }
        .card {
            border: 1px solid rgba(128,128,128,0.25); border-radius: 12px;
            padding: 20px; text-align: center; margin-top: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

SENTIMENT_STYLE = {
    "positive":   {"emoji": "😊", "color": "#2ecc71"},
    "negative":   {"emoji": "😞", "color": "#e74c3c"},
    "neutral":    {"emoji": "😐", "color": "#95a5a6"},
    "irrelevant": {"emoji": "🤷", "color": "#9b59b6"},
}


def style_for(label):
    return SENTIMENT_STYLE.get(str(label).lower(), {"emoji": "🔹", "color": "#3498db"})


def preprocess_text(raw_text):
    cleaned = str(raw_text).lower()
    cleaned = re.sub(r'http\S+|www\.\S+', ' ', cleaned)
    cleaned = re.sub(r'@\w+', ' ', cleaned)
    cleaned = re.sub(r'#', '', cleaned)
    cleaned = re.sub(r'[^a-z\s!?]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# -----------------------------------------------------------------
# Load models
# -----------------------------------------------------------------
@st.cache_resource(show_spinner="Models load ho rahe hain...")
def load_everything():
    with open(model_path('sentiment_svm_pipeline.pkl'), 'rb') as f:
        svm_pipeline = pickle.load(f)
    with open(model_path('label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)

    rnn_model = tf.keras.models.load_model(model_path('sentiment_rnn_model.keras'))
    lstm_model = tf.keras.models.load_model(model_path('LSTM_model.keras'))
    gru_model = tf.keras.models.load_model(model_path('GRU_model.keras'))

    return svm_pipeline, label_encoder, rnn_model, lstm_model, gru_model


st.markdown('<div class="main-title">💬 Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Model select karein aur text ka sentiment check karein</div>', unsafe_allow_html=True)

try:
    svm_pipeline, label_encoder, rnn_model, lstm_model, gru_model = load_everything()
except FileNotFoundError as e:
    st.error(
        f"❌ Model file nahi mili: **{e.filename if hasattr(e, 'filename') else e}**\n\n"
        "Pehle training script run karein taake sab .pkl / .keras files is folder mein ban jayein."
    )
    st.stop()

classes = label_encoder.classes_

MODELS = {
    "SVM": "svm",
    "SimpleRNN": "rnn",
    "LSTM": "lstm",
    "GRU": "gru",
}

# -----------------------------------------------------------------
# UI
# -----------------------------------------------------------------
selected_model = st.selectbox("Model chunein:", list(MODELS.keys()))

user_text = st.text_area(
    "Apna text likhein 👇",
    placeholder="e.g. This game update is absolutely amazing, I love it!",
    height=100,
)

if st.button("🔍 Predict", use_container_width=True):
    if not user_text.strip():
        st.warning("Pehle kuch text likhein.")
        st.stop()

    cleaned = preprocess_text(user_text)
    if not cleaned:
        st.warning("Text mein valid words nahi mile, dusra text try karein.")
        st.stop()

    model_key = MODELS[selected_model]

    if model_key == "svm":
        pred = svm_pipeline.predict([cleaned])[0]
        label = label_encoder.inverse_transform([pred])[0]
        confidence = None
    else:
        input_tensor = tf.constant([cleaned])
        model_map = {"rnn": rnn_model, "lstm": lstm_model, "gru": gru_model}
        probs = model_map[model_key].predict(input_tensor, verbose=0)[0]
        label = classes[int(np.argmax(probs))]
        confidence = float(np.max(probs))

    style = style_for(label)
    conf_text = f"<div style='color:#888; margin-top:6px;'>Confidence: {confidence:.1%}</div>" if confidence is not None else ""

    st.markdown(
        f"""
        <div class="card">
            <div style="font-size:2.5rem;">{style['emoji']}</div>
            <span class="sentiment-badge" style="background-color:{style['color']};">{label}</span>
            <div style="color:#888; margin-top:8px; font-size:0.85rem;">Model: {selected_model}</div>
            {conf_text}
        </div>
        """,
        unsafe_allow_html=True,
    )