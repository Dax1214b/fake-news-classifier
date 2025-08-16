import joblib
from pathlib import Path
import streamlit as st
import pandas as pd

from src.utils import load_data

MODEL_PATH = Path("models/model.joblib")
DEFAULT_DATA = Path("data/sample.csv")

st.set_page_config(page_title="Fake News Classifier", page_icon="📰", layout="centered")

st.title("📰 Fake News Classifier")
st.write("Paste a headline or short news text and the model will predict whether it's **REAL** or **FAKE**.")

# Ensure model exists; if not, train quickly on sample
if not MODEL_PATH.exists():
    st.info("Model not found. Training a quick demo model on sample data...")
    import subprocess, sys
    subprocess.run([sys.executable, "src/train.py", "--data", str(DEFAULT_DATA), "--out", str(MODEL_PATH)], check=True)

model = joblib.load(MODEL_PATH)

with st.form("predict"):
    text = st.text_area("News text", height=150, placeholder="Type or paste news text here...")
    submitted = st.form_submit_button("Predict")

if submitted:
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        pred = model.predict([text])[0]
        proba = model.predict_proba([text])[0]
        labels = list(model.classes_)
        conf = dict(zip(labels, proba))
        st.subheader("Prediction")
        st.write(f"**{pred}**")
        st.subheader("Confidence")
        st.json({k: float(v) for k, v in conf.items()})

st.caption("Model: TF‑IDF + Logistic Regression. Replace data/sample.csv with a larger dataset for better accuracy.")
