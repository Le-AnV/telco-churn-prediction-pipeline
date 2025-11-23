import pandas as pd
import joblib


# ==================================================
# LOAD PIPELINE (preprocess + model)
# ==================================================


def load_model(path="models/best_model.pkl"):
    """
    Load pipeline .pkl đã train (gồm preprocess + model).
    """
    model = joblib.load(path)
    return model


# ==================================================
# PREDICT SINGLE CUSTOMER
# ==================================================


def predict_single(input_dict, pipeline):
    """
    Dự đoán churn cho 1 khách hàng mới.

    input_dict: dict chứa thông tin khách hàng (giống cột trong X)
    pipeline: model đã load từ .pkl
    """
    df = pd.DataFrame([input_dict])
    prob = pipeline.predict_proba(df)[0][1]
    label = int(prob >= 0.5)

    return label, prob


# ==================================================
# PREDICT BATCH CSV
# ==================================================


def predict_csv(csv_path, pipeline):
    """
    Dự đoán churn cho 1 file CSV mới.
    """
    df = pd.read_csv(csv_path)
    probs = pipeline.predict_proba(df)[:, 1]
    preds = (probs >= 0.5).astype(int)

    df["prediction"] = preds
    df["churn_probability"] = probs

    return df
