import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import streamlit as st
import pandas as pd

from src.predict import load_model, predict_single


# ============================
# LOAD MODEL
# ============================
MODEL_PATH = "models/best_model.pkl"
model = load_model(MODEL_PATH)


# ============================
# UI
# ============================
st.set_page_config(page_title="Telco Churn Prediction", page_icon="📊")
st.title("📊 Telco Customer Churn Prediction")
st.write("Ứng dụng dự đoán khả năng khách hàng rời bỏ dịch vụ.")
st.markdown("---")


# ============================
# FORM NHẬP DỮ LIỆU
# ============================
with st.form("customer_form"):

    st.subheader("Thông tin khách hàng")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Giới tính", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Thời gian sử dụng (tháng)", min_value=0, value=12)

    with col2:
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])

    st.subheader("Dịch vụ internet")

    col3, col4 = st.columns(2)
    with col3:
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
    with col4:
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])

    st.subheader("Thanh toán")

    col5, col6 = st.columns(2)
    with col5:
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col6:
        PaymentMethod = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=20.0)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, value=100.0)

    submitted = st.form_submit_button("Predict")


# ============================
# DỰ ĐOÁN
# ============================
if submitted:

    df_raw = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": SeniorCitizen,
                "Partner": Partner,
                "Dependents": Dependents,
                "tenure": tenure,
                "PhoneService": PhoneService,
                "MultipleLines": MultipleLines,
                "InternetService": InternetService,
                "OnlineSecurity": OnlineSecurity,
                "OnlineBackup": OnlineBackup,
                "DeviceProtection": DeviceProtection,
                "TechSupport": TechSupport,
                "StreamingTV": StreamingTV,
                "StreamingMovies": StreamingMovies,
                "Contract": Contract,
                "PaperlessBilling": PaperlessBilling,
                "PaymentMethod": PaymentMethod,
                "MonthlyCharges": MonthlyCharges,
                "TotalCharges": TotalCharges,
            }
        ]
    )

    # Clean + Extract Features + Predict
    prob, label = predict_single(model, df_raw)

    st.markdown("---")
    st.subheader("Kết quả dự đoán")

    st.metric("Trạng thái", label)
    st.metric("Xác suất churn", f"{prob:.3f}")

    if label == "Churn":
        st.error("Khách hàng có nguy cơ rời bỏ dịch vụ.")
    else:
        st.success("Khách hàng ổn định.")
