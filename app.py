import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model and preprocessing tools
model = pickle.load(open("student_model.pkl", "rb"))
scaler = pickle.load(open("student_scaler.pkl", "rb"))
feature_columns = pickle.load(open("student_features.pkl", "rb"))
label_encoder = pickle.load(open("student_labelencoder.pkl", "rb"))

st.title("🎓 Student Performance Prediction App")

st.write("Enter the student’s performance metrics:")

# Inputs dynamically from feature columns
user_input = {}
for col in feature_columns:
    user_input[col] = st.number_input(f"{col}", value=0.0)

if st.button("Predict"):
    df_input = pd.DataFrame([user_input])
    scaled = scaler.transform(df_input)
    prediction = model.predict(scaled)
    result = label_encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted Student Category: **{result}**")
