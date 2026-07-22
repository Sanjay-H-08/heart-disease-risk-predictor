import pickle
import numpy as np
import streamlit as st

# 1. Page Configuration 
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🩺", layout="wide")

# 2. Load Model Parameters 
with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)

weights = model_data["weights"]
bias = model_data["bias"]
mean = model_data["mean"]
std = model_data["std"]

st.title("Heart Disease Risk Predictor 🩺")
st.write("Enter the patient's clinical metrics below to assess heart disease risk.")
st.divider()

# 3. Collect Patient Features using Tabs 
tab1, tab2, tab3 = st.tabs(["👤 Patient Info", "🩺 Vitals", "🏃 Test Results"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age 👤", min_value=1, max_value=120, value=50)
    with col2:
        sex_option = st.selectbox("Sex 🚻", ["Male", "Female"])
        sex = 1 if sex_option == "Male" else 0

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg) 🩺",
            min_value=80,
            max_value=200,
            value=120,
        )
        chol = st.number_input(
            "Serum Cholesterol (mg/dl) 🧪",
            min_value=100,
            max_value=600,
            value=200,
        )
        fbs_option = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl 🍬", ["False", "True"]
        )
        fbs = 1 if fbs_option == "True" else 0
    with col2:
        restecg = st.selectbox("Resting ECG Results (0-2) 📈", [0, 1, 2])
        thalach = st.number_input(
            "Max Heart Rate Achieved 💓",
            min_value=60,
            max_value=220,
            value=150,
        )

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        cp = st.selectbox("Chest Pain Type (0-3) 🫀", [0, 1, 2, 3])
        exang_option = st.selectbox("Exercise-Induced Angina 🏃", ["No", "Yes"])
        exang = 1 if exang_option == "Yes" else 0
        oldpeak = st.number_input(
            "ST Depression (oldpeak) 📉",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
        )
    with col2:
        slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2) 📐", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (0-3) 🩺", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (0-3) 🧬", [0, 1, 2, 3])

st.divider()

# 4. Combine Inputs into 2D Array
user_inputs = [
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
    thal,]
input_array = np.array([user_inputs])

# 5. Prediction & Rich Visual Output 
if st.button("Predict Heart Disease Risk 🩺", use_container_width=True):
    # Input Scaling
    input_scaled = (input_array - mean.values) / std.values

    # Model Math
    z = input_scaled @ weights + bias
    prob = 1 / (1 + np.exp(-z))
    prob_val = float(prob[0])
    prob_percent = prob_val * 100

    prediction = 1 if prob_val >= 0.5 else 0

    st.subheader("Results Analysis 📈")

    col_metric, col_bar = st.columns([1, 2])

    with col_metric:
        st.metric(label="Calculated Risk Probability", value=f"{prob_percent:.1f}%")

    with col_bar:
        st.write("**Risk Spectrum:**")
        st.progress(prob_val)

    if prediction == 1:
        st.error("⚠️ **High Risk Detected:** The model indicates a high probability of heart disease. Further clinical evaluation is recommended.")
    else:
        st.success( "✅ **Low Risk Detected:** The model indicates a low probability of heart disease based on the provided parameters.")
    