import os, sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import pandas as pd
import joblib
from src.preprocessing import clean_telco_data


# Load mô hình đã train
def load_model(path="models/best_model.pkl"):
    return joblib.load(path)


# Dự đoán cho 1 dòng dữ liệu
def predict_single(model, df_raw):
    """
    df_raw: DataFrame chỉ có 1 record (chưa clean)
    """

    # Clean đúng pipeline cho inference
    df_clean = clean_telco_data(df_raw.copy(), clean_label=False, drop_id=False)

    prob = model.predict_proba(df_clean)[0][1]
    label = "Churn" if prob >= 0.5 else "Not Churn"

    return prob, label


# ------------------------------
# Dự đoán cho file CSV (chưa phát triển)
# ------------------------------
def predict_csv(csv_path, model):
    df = pd.read_csv(csv_path)

    df_clean = clean_telco_data(df.copy(), clean_label=False)

    probs = model.predict_proba(df_clean)[:, 1]
    preds = (probs >= 0.5).astype(int)

    df["prediction"] = preds
    df["churn_probability"] = probs

    return df
