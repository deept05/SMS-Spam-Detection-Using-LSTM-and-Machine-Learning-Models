import streamlit as st
import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# App configuration
st.set_page_config(page_title="SMS Spam Detection", layout="centered")
st.title("📩 SMS Spam Detection App")

# Load models and artifacts
@st.cache_resource
def load_artifacts():
    nb_model = pickle.load(open("nb_model.pkl", "rb"))
    svm_model = pickle.load(open("svm_model.pkl", "rb"))
    rf_model = pickle.load(open("rf_model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    lstm_model = load_model("lstm_model.h5")
    return nb_model, svm_model, rf_model, tfidf, tokenizer, lstm_model

# Load everything
nb_model, svm_model, rf_model, tfidf, tokenizer, lstm_model = load_artifacts()

# Model selection dropdown
model_choice = st.selectbox("Select a Model", ["Naive Bayes", "SVM", "Random Forest", "LSTM"])

# Text input for user
user_input = st.text_area("Enter SMS message to classify", height=150)

# Predict button
if st.button("Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a message first.")
    else:
        if model_choice == "LSTM":
            sequence = tokenizer.texts_to_sequences([user_input])
            padded_seq = pad_sequences(sequence, maxlen=50)
            prediction = lstm_model.predict(padded_seq)[0][0]
            label = "Spam" if prediction > 0.5 else "Not Spam"
        else:
            vectorized = tfidf.transform([user_input])
            if model_choice == "Naive Bayes":
                label = "Spam" if nb_model.predict(vectorized)[0] == 1 else "Not Spam"
            elif model_choice == "SVM":
                label = "Spam" if svm_model.predict(vectorized)[0] == 1 else "Not Spam"
            else:  # Random Forest
                label = "Spam" if rf_model.predict(vectorized)[0] == 1 else "Not Spam"

        st.success(f"📢 Prediction: **{label}**")
