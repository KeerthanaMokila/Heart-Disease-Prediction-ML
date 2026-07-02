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
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("models/disease_model.pkl", "rb"))
        scaler = pickle.load(open("models/scaler.pkl", "rb"))
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


model, scaler = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("❤️ Heart Disease Prediction System")
st.markdown(
    "Predict the likelihood of heart disease using a Machine Learning model."
)

st.divider()

# -----------------------------
# Input Section
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    cp = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=250,
        value=130
    )

    chol = st.number_input(
        "Serum Cholesterol (mg/dl)",
        min_value=100,
        max_value=600,
        value=240
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    restecg = st.selectbox(
        "Resting ECG",
        options=[0, 1, 2]
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
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=[0, 1, 2]
    )

    ca = st.selectbox(
        "Major Vessels Colored by Fluoroscopy",
        options=[0, 1, 2, 3, 4]
    )

    thal = st.selectbox(
        "Thalassemia",
        options=[0, 1, 2, 3]
    )

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Heart Disease", use_container_width=True):

    if model is None or scaler is None:
        st.error("Model could not be loaded.")
    else:

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

            st.divider()

            if prediction[0] == 1:
                st.error("⚠️ High Risk of Heart Disease")
                st.write(
                    "The model predicts that the patient is at a higher risk of heart disease. "
                    "Please consult a qualified healthcare professional for further evaluation."
                )

            else:
                st.success("✅ Low Risk of Heart Disease")
                st.write(
                    "The model predicts a lower risk of heart disease. "
                    "Maintain a healthy lifestyle and continue regular medical check-ups."
                )

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "⚠️ This application is for educational purposes only and should not be used as a substitute for professional medical advice."
)
