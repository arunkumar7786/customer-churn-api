"""
main.py
Flask app that serves the churn model on a /predict endpoint.

Run from the project root as a module:
    python -m app.main
"""

from flask import Flask, request, jsonify

from app.utils import load_artifacts, preprocess_and_predict

app = Flask(__name__)

# Load the model + transformer once at startup, not on every request
model, transformer = load_artifacts()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)

    if not data or "customer" not in data:
        return jsonify({"error": "Request body must contain a 'customer' object"}), 400

    try:
        churn_probability, churn_prediction = preprocess_and_predict(
            data["customer"], model, transformer
        )
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400

    return jsonify(
        {
            "churn_probability": round(churn_probability, 4),
            "churn_prediction": churn_prediction,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="localhost", port=8000)
