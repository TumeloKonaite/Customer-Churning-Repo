# Customer Churn Prediction Web App

[![CI](https://github.com/TumeloKonaite/Customer-Churning-Repo/actions/workflows/ci.yml/badge.svg)](https://github.com/TumeloKonaite/Customer-Churning-Repo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Development Status](https://img.shields.io/badge/Status-Active-success.svg)](#development-status)
[![Git Workflow](https://img.shields.io/badge/GitHub-Flow-blue.svg)](https://docs.github.com/en/get-started/quickstart/github-flow)

## Overview
This Flask web application predicts customer churn using a scikit-learn model. Users enter customer attributes via a simple UI, and the app returns whether the customer is likely to churn along with guidance for retention actions.

Training is notebook-independent via a pipeline and a CLI-style entrypoint. It saves model artifacts and metadata (including evaluation metrics and feature schema) under `artifacts/`.

## Features
- User-friendly input form for customer data
- Real-time churn prediction
- Actionable output (high risk vs. low risk)
- Preprocessing + model bundled into a single pipeline artifact

## Retention Decisioning (ROI Layer)
The app includes a lightweight, deterministic decision engine that turns churn probability
into a recommended retention action and a simple ROI proxy. It estimates CLV using a balance
and tenure-based heuristic, assigns actions using fixed probability thresholds, and computes
expected net gain as:

```
net_gain = (p_churn * clv) - action_cost
```

All assumptions and thresholds are documented in `src/decisioning.py`.

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
Visit: [http://127.0.0.1:5001](http://127.0.0.1:5001)

## Model Training
- Entrypoint: `src/train.py` (runs the pipeline and writes `artifacts/metadata.json`)
- Pipeline: `src/pipeline/training_pipeline.py`
- Output artifacts:
  - `artifacts/model.pkl`
  - `artifacts/metadata.json`
  - `artifacts/schema.json`
  - `artifacts/feature_columns.json`

## Project Structure
```
Customer-Churning-Repo/
├─ .github/
│  └─ workflows/
│     └─ ci.yml
├─ application.py
├─ artifacts/
├─ dataset/
│  └─ Churn_Modelling.csv
├─ docs/
│  └─ demo-placeholder.svg
├─ logs/
├─ notebooks/
│  └─ Churning problem using multiple Classification Models.ipynb
├─ src/
│  ├─ components/
│  ├─ metrics.py
│  ├─ pipeline/
│  ├─ decisioning.py
│  ├─ train.py
│  └─ utils.py
├─ templates/
├─ tests/
│  ├─ test_decisioning.py
│  ├─ test_metrics.py
│  └─ test_training_metadata.py
├─ pyproject.toml
├─ requirements.txt
├─ Makefile
└─ README.md
```

## Development Status
Active

## Contributing
1. Create a feature branch
2. Commit changes
3. Open a pull request

## License
MIT — see `LICENSE`.
