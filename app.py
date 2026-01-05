import streamlit as st
import joblib

# Load models
clf = joblib.load("models/classifier.pkl")
tfidf_clf = joblib.load("models/tfidf_vectorizer.pkl")

reg = joblib.load("models/regressor.pkl")
tfidf_reg = joblib.load("models/tfidf_vectorizer_reg.pkl")

st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("🧠 AutoJudge")
st.subheader("Predict Programming Problem Difficulty")

st.write("Paste the problem details below:")

# Input boxes
title = st.text_area("Problem Title")
description = st.text_area("Problem Description")
input_desc = st.text_area("Input Description")
output_desc = st.text_area("Output Description")

if st.button("Predict Difficulty"):
    # Combine text
    text = f"{title} {description} {input_desc} {output_desc}"

    # -------- Classification --------
    X_clf = tfidf_clf.transform([text])
    pred_class = clf.predict(X_clf)[0]

    # -------- Regression --------
    X_reg = tfidf_reg.transform([text])
    pred_score = reg.predict(X_reg)[0]

    st.success("Prediction Complete 🎯")

    st.markdown(f"### 🏷 Difficulty Class: **{pred_class}**")
    st.markdown(f"### 🔢 Difficulty Score: **{pred_score:.2f}**")
