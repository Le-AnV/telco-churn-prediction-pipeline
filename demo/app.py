import streamlit as st
import pandas as pd
import joblib

# =========================================================
# TẢI MODEL
# =========================================================
MODEL_PATH = "models/best_model.pkl"
model = joblib.load(MODEL_PATH)


# =========================================================
# XỬ LÝ DỮ LIỆU TRƯỚC KHI DỰ ĐOÁN
# =========================================================
def preprocess_for_prediction(df):

    # Ép kiểu số và xử lý thiếu
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(
        0
    )
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)

    # Chuẩn hóa dữ liệu đặc biệt
    for c in [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]:
        df[c] = df[c].replace("No internet service", "No")

    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    # Chuyển Yes/No thành 1/0
    yes_no_cols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
    ]
    for c in yes_no_cols:
        df[c] = df[c].map({"Yes": 1, "No": 0})

    # Encode giới tính
    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})

    # Tạo các thuộc tính mới đúng như model dùng khi training
    df["AvgMonthlySpent"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["tenure_charge_interaction"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    df["spending_intensity"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    # Tổng số dịch vụ add-on
    addon_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["num_services"] = df[addon_cols].sum(axis=1)

    # Phân nhóm tenure
    df["tenure_bin"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 48, 1000],
        labels=["0-6m", "6-12m", "1-2y", "2-4y", "4+y"],
        include_lowest=True,
    )

    return df


# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Telco Churn Prediction", page_icon="📊")
st.title("📊 Telco Customer Churn Prediction")
st.write(
    "Ứng dụng dự đoán khách hàng có rời bỏ dịch vụ hay không dựa trên thông tin đầu vào."
)
st.markdown("---")

# =========================================================
# GIAO DIỆN NHẬP DỮ LIỆU
# =========================================================
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

    st.subheader("Các dịch vụ mở rộng")

    col3, col4 = st.columns(2)
    with col3:
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No"])

    with col4:
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])

    st.subheader("Thông tin thanh toán")

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

# =========================================================
# DỰ ĐOÁN
# =========================================================
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

    df_clean = preprocess_for_prediction(df_raw.copy())

    prob = model.predict_proba(df_clean)[0][1]
    label = "🚨 Churn" if prob >= 0.5 else "✅ Not Churn"

    st.markdown("---")
    st.subheader("Kết quả dự đoán")
    st.metric("Trạng thái dự đoán", label)
    st.metric("Xác suất churn", f"{prob:.3f}")

    if prob >= 0.5:
        st.error("Khách hàng có nguy cơ rời bỏ dịch vụ.")
    else:
        st.success("Khách hàng ổn định và ít có khả năng churn.")
