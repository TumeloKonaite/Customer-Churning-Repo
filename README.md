# Customer Churn Prediction Web App

[![CI](https://github.com/TumeloKonaite/Customer-Churning-Repo/actions/workflows/ci.yml/badge.svg)](https://github.com/TumeloKonaite/Customer-Churning-Repo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Development Status](https://img.shields.io/badge/Status-Active-success.svg)](#development-status)
[![Git Workflow](https://img.shields.io/badge/GitHub-Flow-blue.svg)](https://docs.github.com/en/get-started/quickstart/github-flow)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue.svg)](#docker-quick-start)

## Overview
This Flask web application predicts customer churn using a scikit-learn model. Users enter customer attributes via a simple UI, and the app returns whether the customer is likely to churn along with guidance for retention actions.

Training is notebook-independent via a pipeline and a CLI-style entrypoint. It saves model artifacts and metadata (including evaluation metrics and feature schema) under `artifacts/`.

## Features
- User-friendly input form for customer data
- Real-time churn prediction
- Actionable output (high risk vs. low risk)
- Preprocessing + model bundled into a single pipeline artifact
- Docker support for easy deployment (exposes port 5001)

## Retention Decisioning (ROI Layer)
The app includes a lightweight, deterministic decision engine that turns churn probability into a recommended retention action and a simple ROI proxy. It estimates CLV using a balance and tenure-based heuristic, assigns actions using fixed probability thresholds, and computes expected net gain as:

```
net_gain = (p_churn * clv) - action_cost
```

All assumptions and thresholds are documented in `src/decisioning.py`.

### ROI Example Table
The table below uses the same formulas and action costs defined in `src/decisioning.py`.

| Scenario | p_churn | Balance | Tenure | EstimatedSalary | CLV (proxy) | Action | Action Cost | Expected Net Gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Low risk | 0.20 | 10,000 | 5 | 50,000 | 13,000 | No action | 0 | 2,600 |
| Medium risk | 0.45 | 2,500 | 2 | 60,000 | 5,300 | Retention email | 5 | 2,380 |
| High risk | 0.75 | 15,000 | 8 | 80,000 | 19,000 | Discount or retention call | 50 | 14,200 |

## Live Demo
![Live demo placeholder](docs/demo-placeholder.svg)

This demo shows the full flow:
- user enters customer attributes
- submits the form
- receives a churn prediction with confidence score and guidance

## Quickstart

### 1. Set up a virtual environment
Choose **one** option:

**Option A: using uv (recommended)**
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Option B: using Python venv**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run common workflows
```bash
make train   # train model and save to artifacts/
make run     # start the web app locally
make test    # run tests
```

### 3. Open the app
Visit: http://localhost:5001

## Model Training
- Entrypoint: `src/train.py` (runs the pipeline and writes `artifacts/metadata.json`)
- Pipeline: `src/pipeline/training_pipeline.py`
- Output artifacts:
  - `artifacts/model.pkl`
  - `artifacts/metadata.json`
  - `artifacts/schema.json`
  - `artifacts/feature_columns.json`

## Docker Quick Start

### Using Docker Compose (Recommended)
```bash
docker-compose up --build
```

### Using Docker Directly
```bash
# Build the image
docker build -t churn-predictor .

# Optional: pre-train during build to avoid startup delays
# docker build --build-arg RUN_TRAINING=1 -t churn-predictor .

# Run the container
docker run -p 5001:5001 churn-predictor
```

Visit http://localhost:5001 to access the application.

## API Contract

### Health Check
`GET /health`

**200 OK**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-01T12:34:56.789012",
  "model_loaded": true,
  "metadata": {
    "training_date": "2026-02-01T09:15:00",
    "model_name": "churn_predictor",
    "version": "1.0.0"
  }
}
```

### Predict
`POST /api/predict` (Content-Type: application/json)

**Request body**
```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Male",
  "Age": 40,
  "Tenure": 3,
  "Balance": 60000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 50000
}
```

**200 OK**
```json
{
  "status": "success",
  "p_churn": 0.42,
  "predicted_label": 1,
  "clv": 67500.0,
  "recommended_action": "Retention email",
  "net_gain": 28345.0,
  "model_name": "churn_predictor",
  "model_version": "1.0.0",
  "timestamp": "2026-02-01T12:34:56.789012"
}
```

**Errors**
- `400 Bad Request`: missing or invalid fields (see `errors` array)
- `415 Unsupported Media Type`: Content-Type is not `application/json`
- `503 Service Unavailable`: model artifacts not ready

### Curl Example
```bash
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Male",
    "Age": 40,
    "Tenure": 3,
    "Balance": 60000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000
  }'
```

### Artifacts + Auto-Training
The app requires model artifacts under `artifacts/` (`schema.json`, `preprocessor.pkl`, `encoder.pkl`, etc.).
If these files are missing, the container will auto-train on startup by default.

- Control this behavior with `AUTO_TRAIN`:
  - `AUTO_TRAIN=1` (default): train if artifacts are missing
  - `AUTO_TRAIN=0`: skip training (prediction will fail if artifacts are absent)

By default, training runs in the background so the server can start quickly. You can
control this with `AUTO_TRAIN_ASYNC`:
- `AUTO_TRAIN_ASYNC=1` (default): train in background
- `AUTO_TRAIN_ASYNC=0`: train synchronously before app starts

You can also pre-train at build time with `RUN_TRAINING=1` (default in compose).
If a volume is mounted, the container will restore artifacts from an internal
image cache on first start to avoid re-training.

For compatibility with some external smoke tests, the container can also
forward port 5000 to the app port via `ENABLE_PORT_5000=1` (default).
Docker Compose uses a named volume `artifacts` so trained files persist across restarts.

## Project Structure
```
Customer-Churning-Repo/
+- .github/
�  +- workflows/
�     +- ci.yml
+- application.py
+- artifacts/
+- dataset/
�  +- Churn_Modelling.csv
+- docs/
�  +- demo-placeholder.svg
+- logs/
+- notebooks/
�  +- Churning problem using multiple Classification Models.ipynb
+- src/
�  +- components/
�  +- metrics.py
�  +- pipeline/
�  +- decisioning.py
�  +- train.py
�  +- utils.py
+- templates/
+- tests/
�  +- test_decisioning.py
�  +- test_metrics.py
�  +- test_training_metadata.py
+- pyproject.toml
+- requirements.txt
+- Makefile
+- README.md
```

## Development Status
Active

## Model Card
### Model Details
- Model type: scikit-learn binary classifier in a preprocessing pipeline
- Task: predict customer churn (1 = churn, 0 = stay)
- Output: class label plus churn probability

### Intended Use
- Support retention workflows by flagging high-risk customers
- Provide a lightweight ROI proxy via the decisioning layer

### Data
- Training data: `dataset/Churn_Modelling.csv`
- Features: numeric and categorical customer attributes used in the UI and API

### Metrics
- Metrics are stored in `artifacts/metadata.json` after training
- Evaluation utilities in `src/metrics.py`

### Limitations
- The ROI layer is a heuristic, not a causal estimate
- Model performance can drift as customer behavior changes
- Not intended for automated adverse decisions without review

## Contributing
1. Create a feature branch
2. Commit changes
3. Open a pull request

## License
MIT � see `LICENSE`.
