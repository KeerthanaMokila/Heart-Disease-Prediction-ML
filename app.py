import streamlit as st
import numpy as np
import pickle

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Load Model and Scaler
# -----------------------------
@st.cache_resource
def load_model():
    try:
        with open("models/disease_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        return model, scaler

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None


model, scaler = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("❤️ Heart Disease Prediction System")

st.markdown("""
This application predicts the likelihood of **Heart Disease**
using a trained Machine Learning model.

> **Note:** This tool is for educational purposes only and should not replace professional medical advice.
""")

st.divider()

# -----------------------------
# Input Form
# -----------------------------
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=45
        )

        sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male"
        )

        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3]
        )

        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80,
            max_value=250,
            value=130
        )

        chol = st.number_input(
            "Cholesterol (mg/dl)",
            min_value=100,
            max_value=600,
            value=240
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2]
        )

    with col2:

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=60,
            max_value=220,
            value=150
        )

        exang = st.selectbox(
            "Exercise Induced Angina",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        oldpeak = st.number_input(
            "Oldpeak",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        slope = st.selectbox(
            "Slope",
            [0, 1, 2]
        )

        ca = st.selectbox(
            "Number of Major Vessels",
            [0, 1, 2, 3, 4]
        )

        thal = st.selectbox(
            "Thal",
            [0, 1, 2, 3]
        )

    predict = st.form_submit_button("🔍 Predict")

# -----------------------------
# Prediction
# -----------------------------
if predict:

    if model is None or scaler is None:
        st.stop()

    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    try:

        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(scaled_data)[0][1]
        else:
            probability = None

        st.divider()

        if prediction[0] == 1:

            st.error("⚠️ High Risk of Heart Disease")

            if probability is not None:
                st.metric(
                    "Risk Probability",
                    f"{probability*100:.2f}%"
                )

        else:

            st.success("✅ Low Risk of Heart Disease")

            if probability is not None:
                st.metric(
                    "Risk Probability",
                    f"{probability*100:.2f}%"
                )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.markdown(
"""
### About

This Heart Disease Prediction System uses a Machine Learning model trained on clinical parameters to estimate the likelihood of heart disease.

**Disclaimer:** This application is intended for educational purposes only and should not be used as a substitute for professional medical diagnosis or treatment.
"""
)
