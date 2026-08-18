# Customer Churn Prediction API

A simple deployment of a churn prediction model: a real-time Flask API and a
batch scoring script.

## Project Structure

```
customer-churn-api/
├── app/
│   ├── __init__.py
│   ├── main.py            # Flask app, /predict endpoint
│   ├── model.pkl          # Trained RandomForestClassifier
│   ├── transformer.pkl    # Preprocessing pipeline (imputer + one-hot encoder)
│   └── utils.py           # Helper functions
├── test_data/
│   ├── sample_input.json
│   └── all_customers.csv
├── train_model.py         # Script used to train model.pkl / transformer.pkl
├── batch.py                # Batch scoring script
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```



## Running the API

From the project root:

```bash
python -m app.main
```

The app listens on `http://localhost:8000`.

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_data/sample_input.json
```

Response:

```json
{
  "churn_probability": 0.83,
  "churn_prediction": "Yes"
}
```

## Batch Scoring

With the API running in one terminal, run in another:

```bash
python batch.py --input test_data/all_customers.csv
```

This sends each row to `/predict`, writes `scored_customers.csv`, and logs
total requests, failures, and average churn probability to
`logs/batch_log.txt`.

---

## Maintenance Plan

### 🧠 Retraining
The model should be retrained monthly, or immediately if batch-monitoring
shows a meaningful drop in average predicted churn accuracy versus actual
churn outcomes (once labels are available, e.g. a customer's next billing
cycle). Retraining uses the same `train_model.py` script against the latest
`gold_churn_data.csv` snapshot, keeping the same train/test split logic so
results stay comparable over time. New model and transformer files are only
promoted to production after test-set accuracy, precision, and recall meet
or beat the currently deployed model.

### 📉 Drift Detection
Two things are tracked over time: (1) **data drift** — comparing the
distribution of incoming features (e.g. average `tenure`, `MonthlyCharges`,
contract type mix) in `batch_log.txt` against the training distribution, and
flagging large shifts; (2) **prediction drift** — watching the average
churn probability logged by `batch.py` each run. A sudden jump or drop
compared to historical averages signals the model may no longer match
current customer behavior and should trigger a review.

### 🏷️ Versioning
Each trained `model.pkl` / `transformer.pkl` pair is versioned together
(they must always match, since the model expects the transformer's exact
output shape). Versions are tagged by training date (e.g.
`model_2026-08-16.pkl`), with the current production version symlinked or
copied into `app/`. Older versions are kept for a few cycles so a bad
deploy can be rolled back quickly.
