import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


# ===================== BASIC CLEANING =====================


# Encode các cột dạng object có mapping cố định (Yes/No, Male/Female, …)
def encode_binary_map(df, col, mapping):
    df[col] = df[col].map(mapping)
    return df


# Ép kiểu số + fillna bằng median để tránh nhiễu
def to_float_fill_median(df, col):
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())
    return df


# Làm sạch 2 cột phí chính
def clean_monthly_total_charges(df):
    df = to_float_fill_median(df, "MonthlyCharges")
    df = to_float_fill_median(df, "TotalCharges")
    return df


# Chuẩn hoá MultipleLines → gom về No / Yes
def fix_multiple_lines(df):
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    return df


# Gom các cột internet có “No internet service” về “No”
def fix_no_internet(df, cols):
    for c in cols:
        df[c] = df[c].replace("No internet service", "No")
    return df


# Tự động tìm tất cả các cột dạng Yes/No và encode thành 0/1
def encode_yes_no(df, exclude=None):
    if exclude is None:
        exclude = []

    yes_no_cols = [
        c for c in df.columns if c not in exclude and df[c].isin(["Yes", "No"]).all()
    ]

    for c in yes_no_cols:
        df[c] = df[c].map({"No": 0, "Yes": 1})

    return df


# Xoá trùng, xoá dòng lỗi
def final_clean(df, drop_na=True):
    df = df.drop_duplicates()
    if drop_na:
        df = df.dropna(how="any", axis=0)
    return df


# ===================== FEATURE ENGINEERING =====================


# Mức chi tiêu trung bình theo số tháng sử dụng
def fe_avg_monthly_spent(df):
    df["AvgMonthlySpent"] = df["TotalCharges"] / (df["tenure"] + 1)
    return df


# Chia tenure thành các nhóm – mạnh trong churn
def fe_tenure_bins(df):
    # luôn ép về float trước khi fill
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["tenure"] = df["tenure"].fillna(0).astype(float)

    # ép lại để không có giá trị âm
    df["tenure"] = df["tenure"].clip(lower=0)

    bins = [0, 6, 12, 24, 48, 1000]

    df["tenure_bin"] = pd.cut(
        df["tenure"],
        bins=bins,
        labels=["0-6m", "6-12m", "1-2y", "2-4y", "4+y"],
        include_lowest=True,
    )
    return df


# Số lượng dịch vụ internet khách đang dùng
def fe_num_services(df, service_cols):
    df["num_services"] = df[service_cols].sum(axis=1)
    return df


# Các biến tương tác → giúp model học mạnh hơn
def fe_interactions(df):
    df["spending_intensity"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
    return df


# ===================== MASTER PIPELINE =====================


# Toàn bộ quy trình clean + encode + FE cho Telco dataset
def clean_telco_data(df, clean_label=False, drop_id=True):

    # Danh sách các dịch vụ
    internet_service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    # --- BASIC CLEAN ---
    df = clean_monthly_total_charges(df)
    df = fix_multiple_lines(df)
    df = fix_no_internet(df, internet_service_cols)

    # --- ENCODE ---
    df = encode_binary_map(df, "gender", {"Male": 0, "Female": 1})
    df = encode_yes_no(df, exclude=["Churn"])

    # --- FEATURE ENGINEERING ---
    df = fe_avg_monthly_spent(df)
    df = fe_tenure_bins(df)
    df = fe_num_services(df, internet_service_cols)
    df = fe_interactions(df)

    # --- LABEL (chỉ dùng cho training) ---
    if clean_label:
        df = encode_binary_map(df, "Churn", {"Yes": 1, "No": 0})

    # --- DROP ID (optional) ---
    if drop_id and "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # --- FINAL CLEAN ---
    df = final_clean(df)

    return df


# ===================== PREPROCESSOR =====================
def create_preprocessor():

    # Continuous Numerical
    numeric_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "AvgMonthlySpent",
        "spending_intensity",
    ]

    # Categorical (OneHotEncoder)
    categorical_features_onehot = ["InternetService", "PaymentMethod"]

    # Categorical (OrdinalEncoder)
    categorical_features_ordinal = ["Contract", "tenure_bin"]
    Contract_order = ["Month-to-month", "One year", "Two year"]
    tenure_bin_order = ["0-6m", "6-12m", "1-2y", "2-4y", "4+y"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat_onehot",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features_onehot,
            ),
            (
                "cat_ordinal",
                OrdinalEncoder(categories=[Contract_order, tenure_bin_order]),
                categorical_features_ordinal,
            ),
        ],
        remainder="passthrough",
    )

    return preprocessor
