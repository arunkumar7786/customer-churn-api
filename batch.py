"""
batch.py
Reads a CSV of customers, sends each row to the running /predict API,
and writes the scored results to scored_customers.csv.

Usage:
    python batch.py --input test_data/all_customers.csv
"""

import argparse
import logging
import os
import requests
import pandas as pd

API_URL = "http://localhost:8000/predict"


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/batch_log.txt",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",
    )


def score_customers(input_path):
    df = pd.read_csv(input_path)

    # Drop columns that shouldn't go to the model but keep them for the output file
    id_col = "customerID" if "customerID" in df.columns else None
    drop_cols = [c for c in ["Unnamed: 0", "Churn"] if c in df.columns]
    features_df = df.drop(columns=drop_cols) if drop_cols else df.copy()

    total_requests = 0
    failures = 0
    probabilities = []
    predictions = []

    for _, row in features_df.iterrows():
        customer_dict = row.where(pd.notnull(row), None).to_dict()
        total_requests += 1

        try:
            response = requests.post(API_URL, json={"customer": customer_dict}, timeout=10)
            if response.status_code == 200:
                result = response.json()
                probabilities.append(result.get("churn_probability"))
                predictions.append(result.get("churn_prediction"))
            else:
                failures += 1
                logging.error(
                    f"Row failed with status {response.status_code}: {response.text}"
                )
                probabilities.append(None)
                predictions.append(None)
        except Exception as e:
            failures += 1
            logging.error(f"Request exception: {str(e)}")
            probabilities.append(None)
            predictions.append(None)

    df["churn_probability"] = probabilities
    df["churn_prediction"] = predictions

    valid_probs = [p for p in probabilities if p is not None]
    avg_probability = sum(valid_probs) / len(valid_probs) if valid_probs else 0.0

    logging.info(f"Total requests: {total_requests}")
    logging.info(f"Failed predictions: {failures}")
    logging.info(f"Average churn probability: {avg_probability:.4f}")

    print(f"Total requests: {total_requests}")
    print(f"Failed predictions: {failures}")
    print(f"Average churn probability: {avg_probability:.4f}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch score customers for churn.")
    parser.add_argument("--input", required=True, help="Path to input CSV of customers")
    args = parser.parse_args()

    setup_logging()
    scored_df = score_customers(args.input)
    scored_df.to_csv("scored_customers.csv", index=False)
    print("Saved scored_customers.csv")
