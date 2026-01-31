# Customer Churn Prediction Web App

[![CI](https://github.com/YOUR_GITHUB_USERNAME/Customer-Churning-Repo/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/Customer-Churning-Repo/actions/workflows/ci.yml)
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

## Quickstart
1. Create and activate a virtual environment
   - `uv venv`
   - `source .venv/bin/activate`
2. Install dependencies
   - `uv pip install -r requirements.txt`
3. Run the app
   - `python application.py`
4. Open the app in your browser (default: `http://127.0.0.1:5000`)

## Model Training
- Notebook: `notebooks/Churning problem using multiple Classification Models.ipynb`
- Output artifact: `artifacts/model.pkl`

## Project Structure
```
Customer-Churning-Repo/
├─ application.py
├─ dataset/
│  └─ Churn_Modelling.csv
├─ notebooks/
│  └─ Churning problem using multiple Classification Models.ipynb
├─ templates/
├─ models/
├─ artifacts/
├─ requirements.txt
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
