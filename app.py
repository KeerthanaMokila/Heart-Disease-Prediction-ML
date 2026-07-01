import streamlit as st
import numpy as np
import pickle

# 1. ADD THIS HERE (RIGHT AFTER IMPORTS)
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("models/disease_model.pkl", "rb"))
        scaler = pickle.load(open("models/scaler.pkl", "rb"))
        return model, scaler
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None


# 2. LOAD MODEL ONCE
model, scaler = load_model()

# 3. UI STARTS HERE
st.title("Heart Disease Prediction App")
st.write("Enter patient details below")

# inputs...
age = st.number_input("Age", 1, 120, 45)
cp = st.number_input("Chest Pain Type", 0, 3, 1)

# prediction button
if st.button("Predict"):
    if model is not None:
        input_data = np.array([[age, cp]])
        input_data = scaler.transform(input_data)
        result = model.predict(input_data)

        st.success("High Risk" if result[0] == 1 else "Low Risk")
    else:
        st.error("Model not loaded")
