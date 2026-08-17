# Customer Churn Prediction

This repository has one responsibility: train and serve a customer churn model. It supports local Python execution and Modal production deployment.

## Features

- model training and artifact generation
- single-customer churn prediction through JSON or the web form
- batch churn prediction through JSON or CSV
- per-record batch validation with `fail_fast` and `partial` modes
- model health and metadata reporting
- Modal deployment with build-time model training

Prediction responses contain model outputs only. The application does not make retention decisions, calculate ROI, generate outreach, or send email.

## Setup

Python 3.12 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Train and run locally

`application.py` is the sole local Flask entrypoint.

```bash
python -m src.train
python application.py
```

The application listens on `http://localhost:5001` by default. Set `PORT` or `FLASK_DEBUG=1` when needed.

Equivalent Make targets are available:

```bash
make train
make run
make test
```

Training reads `dataset/Churn_Modelling.csv` and writes the model, preprocessing objects, schema, metrics, and metadata under `artifacts/`.

## Web UI

- `/` — navigation
- `/predictdata` — single-customer prediction form
- `/predictbatch` — CSV batch upload and prediction results

The UI displays only the predicted churn/stay label, churn probability, and validation or model errors.

## HTTP API

### Health

```bash
curl http://localhost:5001/health
```

The response reports service health, artifact readiness in `model_loaded`, and model metadata.

### Single prediction

```bash
curl -X POST http://localhost:5001/api/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88
  }'
```

Successful response shape:

```json
{
  "status": "success",
  "predicted_label": 1,
  "p_churn": 0.82,
  "model_name": "churn_predictor",
  "model_version": "1.0.0",
  "timestamp": "2026-08-17T12:00:00+00:00"
}
```

`p_churn` is `null` when the trained estimator does not provide probabilities.

### JSON batch prediction

POST to `/api/predict/batch` (or `/api/batch_predict`) with a `records` list and an optional mode:

```json
{
  "records": [
    {
      "customer_id": "CUST_001",
      "CreditScore": 619,
      "Geography": "France",
      "Gender": "Female",
      "Age": 42,
      "Tenure": 2,
      "Balance": 0,
      "NumOfProducts": 1,
      "HasCrCard": 1,
      "IsActiveMember": 1,
      "EstimatedSalary": 101348.88
    }
  ],
  "options": {"mode": "partial"}
}
```

Each successful result contains exactly `index`, `id`, `predicted_label`, and `p_churn`. The envelope also contains validation errors, a summary, model metadata, and a timestamp. `fail_fast` stops validation at the first invalid record; `partial` scores every valid record and returns row-level errors for invalid records.

### CSV batch prediction

```bash
curl -X POST http://localhost:5001/api/batch_predict_csv \
  -F 'file=@customers.csv' \
  -F 'options={"mode":"partial"}'
```

CSV files require the ten model fields shown above. `customer_id`, `row_id`, or `id` may be included for response passthrough. The maximum batch size is 100.

## Smoke tests

The cross-platform smoke scripts validate `/health` and `/api/predict` against a local or Modal URL.

```bash
BASE_URL=http://localhost:5001 ./scripts/smoke.sh
```

```powershell
./scripts/smoke.ps1 -BaseUrl "https://your-modal-url"
```

## Modal deployment

Modal is the only production deployment target. Authenticate the Modal CLI, then serve a development deployment or deploy to production:

```bash
modal serve modal_app.py
modal deploy modal_app.py
```

The Modal image installs `requirements.txt`, copies the application, trains the model during image construction, and serves the Flask WSGI app. Runtime scaling and timeout settings are defined in `modal_app.py`.

GitHub deployment runs on pushes to `main` and manual dispatch. Configure these repository secrets:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

## Dependencies

`pyproject.toml` is the dependency source of truth. `requirements.txt` and `uv.lock` are generated from it. Visualization libraries are isolated in the optional `notebook` extra and are not installed in the production Modal image.

```bash
uv lock
uv export --no-dev --no-emit-project --no-annotate --no-header -o requirements.txt
```

For notebook exploration:

```bash
python -m pip install -e '.[notebook]'
```

## Project layout

```text
application.py             Local Flask entrypoint and HTTP/UI routes
modal_app.py               Modal image build and WSGI deployment
src/train.py               Training command and metadata generation
src/components/            Data ingestion, transformation, and model training
src/pipeline/               Training and prediction pipelines
src/services/              Batch validation and prediction service
templates/                 Prediction-only web UI
scripts/smoke.sh           POSIX smoke test
scripts/smoke.ps1          PowerShell smoke test
tests/                      API, batch, metrics, and training tests
notebooks/                  Optional exploratory analysis
```

## Model limitations

The model is trained on the included bank churn dataset. Its quality depends on the representativeness and freshness of that data. Predictions should be monitored for drift and evaluated for fairness before use in consequential workflows.

## License

See [LICENSE](LICENSE).
