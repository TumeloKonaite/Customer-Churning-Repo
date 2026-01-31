# Customer Churn Prediction Web App

[![CI](https://github.com/TumeloKonaite/Customer-Churning-Repo/actions/workflows/ci.yml/badge.svg)](https://github.com/TumeloKonaite/Customer-Churning-Repo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Development Status](https://img.shields.io/badge/Status-Active-success.svg)](#development-status)
[![Git Workflow](https://img.shields.io/badge/GitHub-Flow-blue.svg)](https://docs.github.com/en/get-started/quickstart/github-flow)

## Overview
This Flask web application predicts customer churn using a pre-trained scikit-learn model. Users enter customer attributes via a simple UI, and the app returns whether the customer is likely to churn along with guidance for retention actions.

The training pipeline is built in the notebook and saves a serialized artifact to `artifacts/model.pkl`.

## Features
- User-friendly input form for customer data
- Real-time churn prediction
- Actionable output (high risk vs. low risk)
- Preprocessing + model bundled into a single pipeline artifact

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
Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Model Training
- Notebook: `notebooks/Churning problem using multiple Classification Models.ipynb`
- Output artifact: `artifacts/model.pkl`

## Project Structure
```
Customer-Churning-Repo/
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
│  ├─ pipeline/
│  └─ utils.py
├─ templates/
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
