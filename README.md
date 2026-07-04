# ❤️ Heart Disease Prediction System

A Machine Learning web application that predicts the likelihood of heart disease based on patient health parameters. The application is built using **Python**, **Scikit-learn**, and **Streamlit**, providing an interactive interface for real-time predictions.

---

## 🚀 Live Demo

**Streamlit App:** https://heart-disease-prediction-ml-ijihf3uacoualx8w8mdxnw.streamlit.app/

---

## 📖 Project Overview

Heart disease is one of the leading causes of death worldwide. Early prediction can help identify individuals at higher risk and encourage timely medical consultation.

This project uses a trained Machine Learning model to predict whether a patient is likely to have heart disease based on clinical features.

---

## ✨ Features

* Interactive Streamlit web application
* Predicts heart disease risk using Machine Learning
* Accepts 13 clinical input features
* Real-time prediction
* Clean and user-friendly interface
* Deployable on Streamlit Community Cloud

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Scikit-learn
* NumPy
* Pickle

---

## 📊 Input Features

The model uses the following patient information:

| Feature                 | Description                                    |
| ----------------------- | ---------------------------------------------- |
| Age                     | Patient age                                    |
| Sex                     | Female (0), Male (1)                           |
| Chest Pain Type         | Type of chest pain                             |
| Resting Blood Pressure  | Resting blood pressure (mm Hg)                 |
| Cholesterol             | Serum cholesterol (mg/dl)                      |
| Fasting Blood Sugar     | >120 mg/dl (0/1)                               |
| Resting ECG             | Resting electrocardiographic results           |
| Maximum Heart Rate      | Maximum heart rate achieved                    |
| Exercise Induced Angina | Yes/No                                         |
| Oldpeak                 | ST depression induced by exercise              |
| Slope                   | Slope of peak exercise ST segment              |
| Major Vessels           | Number of major vessels colored by fluoroscopy |
| Thal                    | Thalassemia status                             |

---

## 📁 Project Structure

```text
Heart-Disease-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── disease_model.pkl
│   └── scaler.pkl
│
└── notebook.ipynb
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Heart-Disease-Prediction.git
```

Move into the project folder:

```bash
cd Heart-Disease-Prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🎯 How to Use

1. Open the Streamlit application.
2. Enter the patient's clinical details.
3. Click **Predict**.
4. View whether the model predicts a **High Risk** or **Low Risk** of heart disease.

---

## 📸 Application Preview

Add a screenshot of your Streamlit application here.

Example:

```
images/app_screenshot.png
```

---

## 📌 Future Improvements

* Display prediction probability
* Improve UI with custom styling
* Add visualizations and charts
* Support multiple machine learning models
* Improve model performance through hyperparameter tuning

---

## ⚠️ Disclaimer

This application is intended for educational and demonstration purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult a qualified healthcare professional for medical advice.

---

## 👩‍💻 Author

**Keerthana Mokila**

GitHub: https://github.com/keerthanamokila23

LinkedIn:https://www.linkedin.com/in/keerthana-mokila-2a1480367
