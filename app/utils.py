"""
utils.py
Small helper functions shared by the Flask app.
"""

import os
import pickle
import pandas as pd

# Folder this file lives in, so paths work no matter where the app is run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
TRANSFORMER_PATH = os.path.join(BASE_DIR, "transformer.pkl")


def load_artifacts():
    """Load the trained model and preprocessing pipeline from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(TRANSFORMER_PATH, "rb") as f:
        transformer = pickle.load(f)
    return model, transformer


def customer_json_to_dataframe(customer_dict):
    """Turn a single customer dict into a one-row DataFrame."""
    return pd.DataFrame([customer_dict])


def preprocess_and_predict(customer_dict, model, transformer):
    """Run the full pipeline: dict -> DataFrame -> transform -> predict."""
    df = customer_json_to_dataframe(customer_dict)

    # Make sure TotalCharges is numeric, same as training
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    X_transformed = transformer.transform(df)

    churn_probability = float(model.predict_proba(X_transformed)[0][1])
    churn_prediction = "Yes" if churn_probability >= 0.5 else "No"

    return churn_probability, churn_prediction
