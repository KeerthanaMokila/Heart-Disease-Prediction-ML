import streamlit as st
import numpy as np
import pickle

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")
st.markdown("Enter patient details to predict risk.")

st.warning("This is a machine learning demo, not a medical tool.")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = pickle.load(open("models/disease_model.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# =========================
# INPUTS (13 FEATURES)
# =========================
age = st.number_input("Age", 1, 120, 45)
sex = st.number_input("Sex (0=female, 1=male)", 0, 1, 1)
cp = st.number_input("Chest Pain Type", 0, 3, 1)
trestbps = st.number_input("Resting BP", 80, 200, 130)
chol = st.number_input("Cholesterol", 100, 600, 240)
fbs = st.number_input("Fasting Blood Sugar", 0, 1, 0)
restecg = st.number_input("ECG Result", 0, 2, 1)
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.number_input("Exercise Angina", 0, 1, 0)
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.number_input("Slope", 0, 2, 1)
ca = st.number_input("Major Vessels", 0, 4, 0)
thal = st.number_input("Thal", 0, 3, 2)

# =========================
# PREDICTION
# =========================
if st.button("Predict"):

    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]])

    try:
        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

    except Exception as e:
        st.error(f"Error: {e}")
